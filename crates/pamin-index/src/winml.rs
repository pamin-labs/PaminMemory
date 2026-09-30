//! Windows ML's certified vendor catalog, kept separate from model loading.
//! ABI: Microsoft.Windows.AI.MachineLearning 2.4.89, WinMLEpCatalog.h.
use std::ffi::{CStr, c_char, c_void};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ort::environment::Environment;
use ort::ep::ExecutionProviderLibrary;
use sha2::{Digest, Sha256};
use windows_sys::Win32::Foundation::FreeLibrary;
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
    LoadLibraryExW,
};

type Handle = *mut c_void;
type Status = i32;
type Callback = unsafe extern "system" fn(Handle, *const Info, *mut c_void) -> i32;

const CATALOG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/winml-catalog.dll"));
const LICENSE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/winml-license.txt"));

#[repr(C)]
struct Info {
    name: *const c_char,
    version: *const c_char,
    package_family: *const c_char,
    library_path: *const c_char,
    package_root: *const c_char,
    ready: i32,
    certification: i32,
}

struct Module(usize);
impl Drop for Module {
    fn drop(&mut self) {
        // SAFETY: this owns the handle returned by LoadLibraryExW.
        unsafe {
            FreeLibrary(self.0 as _);
        }
    }
}

struct Registered {
    _libraries: Vec<ExecutionProviderLibrary>,
    _module: Module,
}

struct Api {
    create: unsafe extern "system" fn(*mut Handle) -> Status,
    release: unsafe extern "system" fn(Handle),
    enumerate: unsafe extern "system" fn(Handle, Callback, *mut c_void) -> Status,
    ready: unsafe extern "system" fn(Handle) -> Status,
    path_size: unsafe extern "system" fn(Handle, *mut usize) -> Status,
    path: unsafe extern "system" fn(Handle, usize, *mut c_char, *mut usize) -> Status,
}

/// Hold the runtime and vendor libraries for every model's lifetime. A missing
/// optional runtime leaves the existing GPU/SIMD CPU path available.
pub(crate) fn register(environment: &Arc<Environment>) {
    static REGISTERED: OnceLock<Option<Registered>> = OnceLock::new();
    REGISTERED.get_or_init(|| match load(environment) {
        Ok(registered) => Some(registered),
        Err(error) => {
            tracing::debug!(%error, "Windows ML catalog unavailable; using existing providers");
            None
        }
    });
}

fn load(environment: &Arc<Environment>) -> Result<Registered, String> {
    use std::os::windows::ffi::OsStrExt;
    let path = catalog_path().map_err(|error| error.to_string())?;
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: use an absolute installation path and restricted dependency
    // search flags, never the working directory; `wide` is NUL terminated.
    let handle = unsafe {
        LoadLibraryExW(
            wide.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
        )
    };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error().to_string());
    }
    let module = Module(handle as usize);
    unsafe fn symbol<T: Copy>(module: &Module, name: &CStr) -> Result<T, String> {
        // SAFETY: the module is live and names are NUL terminated. Callers
        // supply exact signatures from the pinned SDK header.
        let address = unsafe { GetProcAddress(module.0 as _, name.as_ptr().cast()) }
            .ok_or_else(|| format!("missing {}", name.to_string_lossy()))?;
        assert_eq!(std::mem::size_of::<T>(), std::mem::size_of_val(&address));
        // SAFETY: the corresponding symbol has the caller's SDK signature.
        Ok(unsafe { std::mem::transmute_copy(&address) })
    }
    // SAFETY: each signature above matches WinMLEpCatalog.h in SDK 2.4.89.
    let api = unsafe {
        Api {
            create: symbol(&module, c"WinMLEpCatalogCreate")?,
            release: symbol(&module, c"WinMLEpCatalogRelease")?,
            enumerate: symbol(&module, c"WinMLEpCatalogEnumProviders")?,
            ready: symbol(&module, c"WinMLEpEnsureReady")?,
            path_size: symbol(&module, c"WinMLEpGetLibraryPathSize")?,
            path: symbol(&module, c"WinMLEpGetLibraryPath")?,
        }
    };
    let mut catalog = std::ptr::null_mut();
    // SAFETY: catalog is a writable out parameter, and the runtime is live.
    if unsafe { (api.create)(&mut catalog) } < 0 || catalog.is_null() {
        return Err("creating the Windows ML catalog failed".into());
    }
    let mut context = Context {
        api: &api,
        environment,
        libraries: Vec::new(),
    };
    // SAFETY: the catalog owns each callback's handles/strings; context lives
    // until synchronous enumeration finishes. The callback catches panics.
    let status =
        unsafe { (api.enumerate)(catalog, collect, (&mut context as *mut Context<'_>).cast()) };
    // SAFETY: release the live catalog exactly once after enumeration.
    unsafe {
        (api.release)(catalog);
    }
    if status < 0 {
        return Err("enumerating Windows ML providers failed".into());
    }
    Ok(Registered {
        _libraries: context.libraries,
        _module: module,
    })
}

/// The embedded, checksum-verified catalog survives `cargo install` and a
/// read-only application directory. Vendor EPs remain managed by Windows ML.
fn catalog_path() -> std::io::Result<PathBuf> {
    let local = std::env::var_os("LOCALAPPDATA")
        .ok_or_else(|| std::io::Error::other("LOCALAPPDATA is unavailable"))?;
    let local = PathBuf::from(local);
    if !local.is_absolute() {
        return Err(std::io::Error::other("LOCALAPPDATA is not absolute"));
    }
    let identity = format!("{:x}", Sha256::digest(CATALOG));
    let directory = local.join("pamin/runtimes/windows-ml").join(identity);
    std::fs::create_dir_all(&directory)?;
    install(&directory.join("license.txt"), LICENSE)?;
    let path = directory.join("Microsoft.Windows.AI.MachineLearning.dll");
    install(&path, CATALOG)?;
    Ok(path)
}

/// Never overwrite a mapped DLL. Independent processes publish complete files
/// atomically; an existing file must match the embedded bytes before loading.
fn install(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Ok(existing) = std::fs::read(path) {
        if existing == bytes {
            return Ok(());
        }
        return Err(std::io::Error::other("Windows ML cache checksum mismatch"));
    }
    let pending = path.with_extension(format!("{}.partial", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&pending)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        match std::fs::rename(&pending, path) {
            Ok(()) => Ok(()),
            Err(_) if std::fs::read(path).is_ok_and(|existing| existing == bytes) => Ok(()),
            Err(error) => Err(error),
        }
    })();
    let _ = std::fs::remove_file(pending);
    result
}

struct Context<'a> {
    api: &'a Api,
    environment: &'a Arc<Environment>,
    libraries: Vec<ExecutionProviderLibrary>,
}

unsafe extern "system" fn collect(ep: Handle, info: *const Info, context: *mut c_void) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if ep.is_null() || info.is_null() || context.is_null() {
            return;
        }
        // SAFETY: these pointers belong to the catalog's synchronous callback.
        let (info, context) = unsafe { (&*info, &mut *context.cast::<Context<'_>>()) };
        if info.name.is_null() || info.certification != 1 {
            return;
        }
        // SAFETY: the catalog supplies a NUL-terminated provider name.
        let Ok(name) = (unsafe { CStr::from_ptr(info.name) }).to_str() else {
            return;
        };
        if !matches!(
            name,
            "OpenVINOExecutionProvider" | "QNNExecutionProvider" | "VitisAIExecutionProvider"
        ) {
            return;
        }
        // The catalog lists compatible providers. EnsureReady installs a
        // missing certified EP or adds its existing package to this process.
        // SAFETY: `ep` is live until catalog enumeration returns.
        let status = unsafe { (context.api.ready)(ep) };
        if status < 0 {
            tracing::warn!(
                provider = name,
                hresult = format_args!("{status:#010x}"),
                "certified provider could not become ready"
            );
            return;
        }
        let mut size = 0;
        // SAFETY: size is a writable out parameter for this live EP.
        let status = unsafe { (context.api.path_size)(ep, &mut size) };
        if status < 0 || !(1..=32768).contains(&size) {
            tracing::warn!(
                provider = name,
                hresult = format_args!("{status:#010x}"),
                size,
                "certified provider returned no usable library path"
            );
            return;
        }
        let mut path = vec![0u8; size];
        // SAFETY: path contains exactly the capacity passed to the API.
        let status =
            unsafe { (context.api.path)(ep, size, path.as_mut_ptr().cast(), std::ptr::null_mut()) };
        if status < 0 {
            tracing::warn!(
                provider = name,
                hresult = format_args!("{status:#010x}"),
                "certified provider library path could not be read"
            );
            return;
        }
        let Ok(path) = CStr::from_bytes_until_nul(&path) else {
            return;
        };
        let Ok(path) = path.to_str() else {
            return;
        };
        match context.environment.register_ep_library(name, path) {
            Ok(library) => context.libraries.push(library),
            Err(error) => {
                tracing::warn!(provider = name, %error, "certified provider could not register")
            }
        }
    }))
    .map_or(0, |()| 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_installed_catalog_is_reused_but_corruption_is_not_overwritten() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("catalog.dll");
        install(&path, b"original").unwrap();
        install(&path, b"original").unwrap();
        std::fs::write(&path, b"corrupt").unwrap();
        assert!(install(&path, b"original").is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"corrupt");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
