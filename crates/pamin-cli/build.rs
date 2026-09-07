//! Makes the built binary able to find the retrieval engine's native library.
//!
//! The engine ships as a dynamic library that the linker records by name only.
//! Cargo's test harness happens to set a library search path, so tests pass
//! either way, but a plainly invoked binary fails at startup with a missing
//! library. Since the whole point of the CLI is that someone can run it, that
//! has to be solved here rather than in the harness.
//!
//! Two things are needed. The library is copied next to the binary so the
//! shipped pair is self-contained, and an `@executable_path` rpath is recorded
//! so the binary looks beside itself. An absolute rpath to the build directory
//! is added as well, which is what keeps `cargo run` working from a tree where
//! nothing has been copied yet.
//!
//! The dependency's own build script emits an rpath, but `rustc-link-arg`
//! applies only to the crate that declares it, so it never reaches this binary.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let Ok(out_dir) = std::env::var("OUT_DIR") else {
        return;
    };
    let out_dir = PathBuf::from(out_dir);

    // OUT_DIR is <target>/<profile>/build/<crate>-<hash>/out, so two levels up
    // is the directory holding every build script's output, and three is the
    // profile directory where the binary is written.
    let Some(build_root) = out_dir.ancestors().nth(2) else {
        return;
    };
    let Some(profile_dir) = out_dir.ancestors().nth(3) else {
        return;
    };

    let Some(library) = find_library(build_root) else {
        // Not fatal: the library may be installed system-wide, or supplied
        // through ZVEC_LIB_DIR. Warn rather than fail so those paths still work.
        println!("cargo:warning=zvec native library not found; the binary may not start");
        return;
    };

    if let Some(directory) = library.parent() {
        println!(
            "cargo:rustc-link-arg-bins=-Wl,-rpath,{}",
            directory.display()
        );
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path"),
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }

    if let Some(name) = library.file_name() {
        let destination = profile_dir.join(name);
        let _ = std::fs::copy(&library, &destination);
    }
}

/// Finds the engine's native library among the build script outputs.
fn find_library(build_root: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(build_root).ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("zvec-rust-sys-") {
            continue;
        }
        if let Some(found) = search(&entry.path().join("out")) {
            return Some(found);
        }
    }

    None
}

/// Looks for the library within one build script's output, one level deep.
fn search(dir: &Path) -> Option<PathBuf> {
    let matches = |path: &Path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("libzvec_c_api."))
    };

    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let path = entry.path();
        if path.is_file() && matches(&path) {
            return Some(path);
        }
        if path.is_dir() {
            for nested in std::fs::read_dir(&path).ok()?.flatten() {
                let nested = nested.path();
                if nested.is_file() && matches(&nested) {
                    return Some(nested);
                }
            }
        }
    }

    None
}
