//! Carry the Windows ML catalog with an installed executable, including
//! `cargo install`, which does not install DLLs beside the binary.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    build_identity();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let (arch, dll_hash) = match std::env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
        "x86_64" => (
            "x64",
            "b401a47552a745755d65100828faefcf3df3895c759f96ac625ddf990711c8dd",
        ),
        "aarch64" => (
            "arm64",
            "457295ca2aec34db4aec7064cb401520bc85f29fd18f7e7bb9e25ddb849bedb1",
        ),
        other => {
            unavailable(
                &out,
                &format!("Windows ML has no catalog binary for {other}"),
            );
            return;
        }
    };
    let host = std::env::var("HOST").unwrap();
    let status = extractor(host.contains("windows"))
        .env("WINML_BUILD_OUT", &out)
        .env("WINML_BUILD_ARCH", arch)
        .env("WINML_BUILD_DLL_HASH", dll_hash)
        .status();
    if !status.is_ok_and(|status| status.success()) {
        unavailable(
            &out,
            "optional Windows ML catalog could not be fetched; GPU/CPU remain available",
        );
    }
}

fn build_identity() {
    use sha2::{Digest, Sha256};
    fn sources(directory: &std::path::Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(directory).expect("read inference sources") {
            let path = entry.expect("source entry").path();
            if path.is_dir() {
                sources(&path, files);
            } else {
                files.push(path);
            }
        }
    }
    let mut files = vec![PathBuf::from("build.rs"), PathBuf::from("Cargo.toml")];
    sources(std::path::Path::new("src"), &mut files);
    // Registry installs need not carry the workspace files.
    for name in ["../../Cargo.toml", "../../Cargo.lock"] {
        if std::path::Path::new(name).is_file() {
            files.push(PathBuf::from(name));
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(path.to_string_lossy().as_bytes());
        hash.update(std::fs::read(path).expect("read build identity input"));
    }
    for key in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_CFG_TARGET_FEATURE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
        hash.update(key.as_bytes());
        hash.update(
            std::env::var_os(key)
                .unwrap_or_default()
                .to_string_lossy()
                .as_bytes(),
        );
    }
    if let Ok(output) = Command::new(std::env::var_os("RUSTC").expect("rustc path"))
        .arg("--version")
        .output()
    {
        hash.update(output.stdout);
    }
    println!(
        "cargo:rustc-env=PAMIN_INFERENCE_BUILD={:x}",
        hash.finalize()
    );
}

fn unavailable(out: &std::path::Path, reason: &str) {
    println!("cargo:warning={reason}");
    // A missing dependency is always changed in Cargo's fingerprint. Retry
    // optional extraction on the next build without requiring cargo clean.
    println!(
        "cargo:rerun-if-changed={}",
        out.join("winml-catalog-unavailable.retry").display()
    );
    // Never embed partial/unverified bytes after a failed extraction.
    std::fs::write(out.join("winml-catalog.dll"), []).expect("write optional catalog placeholder");
    std::fs::write(out.join("winml-license.txt"), []).expect("write optional license placeholder");
}

fn extractor(windows: bool) -> Command {
    let mut command = if windows {
        let powershell = PathBuf::from(std::env::var_os("SystemRoot").unwrap())
            .join("System32/WindowsPowerShell/v1.0/powershell.exe");
        let mut command = Command::new(powershell);
        command.args(["-NoProfile", "-NonInteractive", "-Command", POWERSHELL]);
        command
    } else {
        // Only cross-builds need Python; native Windows uses its system tools.
        let mut command = Command::new("python3");
        command.args(["-c", PYTHON]);
        command
    };
    command
        .env("WINML_BUILD_URL", URL)
        .env("WINML_BUILD_PACKAGE_HASH", PACKAGE_HASH);
    command
}

const URL: &str = "https://api.nuget.org/v3-flatcontainer/microsoft.windows.ai.machinelearning/2.4.89/microsoft.windows.ai.machinelearning.2.4.89.nupkg";
const PACKAGE_HASH: &str = "5c68ecfb947223267abf159a023f5192ad42725e4e9cc995e7c1470dd54dff63";

// Microsoft.Windows.AI.MachineLearning 2.4.89. Extract only the catalog and its
// license; keep the project's existing ONNX Runtime and DirectML versions.
const PYTHON: &str = r#"
import hashlib, os, subprocess, zipfile
from pathlib import Path
out=Path(os.environ['WINML_BUILD_OUT'])
arch=os.environ['WINML_BUILD_ARCH']
dll=out/'winml-catalog.dll'
license=out/'winml-license.txt'
expected=os.environ['WINML_BUILD_DLL_HASH']
if dll.exists() and license.exists() and hashlib.sha256(dll.read_bytes()).hexdigest()==expected:
    raise SystemExit(0)
package=out/'winml.nupkg'
try:
    if not package.exists():
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--proto','=https','--proto-redir','=https','--max-time','30','--output',str(package),os.environ['WINML_BUILD_URL']],check=True)
    if hashlib.sha256(package.read_bytes()).hexdigest()!=os.environ['WINML_BUILD_PACKAGE_HASH']:
        raise RuntimeError('Windows ML package checksum mismatch')
    with zipfile.ZipFile(package) as archive:
        image=archive.read(f'runtimes/win-{arch}/native/Microsoft.Windows.AI.MachineLearning.dll')
        if hashlib.sha256(image).hexdigest()!=expected:
            raise RuntimeError('Windows ML DLL checksum mismatch')
        license.write_bytes(archive.read('license.txt'))
        dll.write_bytes(image)
finally:
    package.unlink(missing_ok=True)
"#;

const POWERSHELL: &str = r#"
$ErrorActionPreference='Stop'
function Digest([string]$path) {
    $sha=[System.Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($sha.ComputeHash([System.IO.File]::ReadAllBytes($path))).Replace('-','').ToLowerInvariant() }
    finally { $sha.Dispose() }
}
$dll=Join-Path $env:WINML_BUILD_OUT 'winml-catalog.dll'
$license=Join-Path $env:WINML_BUILD_OUT 'winml-license.txt'
if ((Test-Path $dll) -and (Test-Path $license) -and ((Digest $dll) -eq $env:WINML_BUILD_DLL_HASH)) { exit 0 }
$package=Join-Path $env:WINML_BUILD_OUT 'winml.nupkg'
try {
    if (!(Test-Path $package)) {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 30 -Uri $env:WINML_BUILD_URL -OutFile $package
    }
    if ((Digest $package) -ne $env:WINML_BUILD_PACKAGE_HASH) { throw 'Windows ML package checksum mismatch' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive=[System.IO.Compression.ZipFile]::OpenRead($package)
    try {
        $entry=$archive.GetEntry("runtimes/win-$($env:WINML_BUILD_ARCH)/native/Microsoft.Windows.AI.MachineLearning.dll")
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$dll,$true)
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($archive.GetEntry('license.txt'),$license,$true)
    } finally { $archive.Dispose() }
    if ((Digest $dll) -ne $env:WINML_BUILD_DLL_HASH) { throw 'Windows ML DLL checksum mismatch' }
} finally { if (Test-Path $package) { Remove-Item $package } }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_catalog_is_retried_by_the_next_cargo_build() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("project");
        std::fs::create_dir_all(project.join("src")).unwrap();
        std::fs::write(
            project.join("Cargo.toml"),
            r#"[package]
name="catalog-retry-fixture"
version="0.0.0"
edition="2024"
"#,
        )
        .unwrap();
        std::fs::write(project.join("src/lib.rs"), "").unwrap();
        let source = include_str!("build.rs");
        let start = source.find("fn unavailable(").unwrap();
        let end = source[start..].find("\nfn extractor(").unwrap() + start;
        // Exercise the actual recovery function without compiling unrelated
        // unused extractor branches under the caller's -D warnings flags.
        let mut script = source[start..end].to_string();
        script.push_str(
            r#"
fn main() {
    let out=std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    unavailable(&out,"fixture optional catalog unavailable");
    let mut counter=std::fs::OpenOptions::new().create(true).append(true)
        .open(std::env::var_os("WINML_TEST_BUILD_COUNT").unwrap()).unwrap();
    std::io::Write::write_all(&mut counter,b"run\n").unwrap();
}
"#,
        );
        std::fs::write(project.join("build.rs"), script).unwrap();
        let counter = root.path().join("runs");
        for _ in 0..2 {
            let output = Command::new("cargo")
                .args(["build", "--offline", "--quiet"])
                .current_dir(&project)
                .env("CARGO_TARGET_DIR", root.path().join("target"))
                .env("WINML_TEST_BUILD_COUNT", &counter)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        assert_eq!(
            std::fs::read_to_string(counter).unwrap().lines().count(),
            2,
            "Cargo reused an unavailable catalog without retrying the extractor"
        );
    }

    #[test]
    fn unavailable_catalog_discards_partial_bytes() {
        let out = tempfile::tempdir().unwrap();
        std::fs::write(out.path().join("winml-catalog.dll"), b"unverified").unwrap();
        unavailable(out.path(), "fixture unavailable");
        assert!(
            std::fs::read(out.path().join("winml-catalog.dll"))
                .unwrap()
                .is_empty()
        );
        assert!(
            std::fs::read(out.path().join("winml-license.txt"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn corrupted_catalog_package_is_rejected_before_extraction() {
        let out = tempfile::tempdir().unwrap();
        std::fs::write(out.path().join("winml.nupkg"), b"corrupt").unwrap();
        let output = extractor(cfg!(windows))
            .env("WINML_BUILD_OUT", out.path())
            .env("WINML_BUILD_ARCH", "x64")
            .env("WINML_BUILD_DLL_HASH", "unused")
            .env("PYTHONOPTIMIZE", "1")
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("package checksum mismatch"));
        assert!(!out.path().join("winml-catalog.dll").exists());
        assert!(!out.path().join("winml.nupkg").exists());
    }
}
