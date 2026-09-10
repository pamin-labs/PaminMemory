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
//! so the binary looks beside itself. Both happen here, in this order, so a
//! plain `cargo run` finds the copy this script just made.
//!
//! That covers running from the tree and shipping the pair, but not
//! `cargo install`, which the README opens with. Cargo installs the executable
//! and nothing else, so the installed binary has no library beside it and falls
//! back on the absolute rpath into `target/` -- a directory the next
//! `cargo clean` removes. The binary then stops starting, on a machine where it
//! worked yesterday, with a message about a missing library nobody asked for.
//!
//! So the library is also staged under `CARGO_HOME`, which is the one location
//! known at build time that outlives `target/`, and that copy is recorded as an
//! rpath too. Writing outside `OUT_DIR` is against Cargo's advice; the
//! alternative is an install path the project documents and that does not work.
//!
//! Deliberately absent: an rpath into the dependency's own build directory. It
//! would resolve on the machine that built the binary and nowhere else, which
//! is precisely the failure being fixed, and having it there would hide the
//! fix -- the loader would use it in preference and the installed binary would
//! look fine right up until someone cleaned the tree.
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

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path"),
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }

    let Some(name) = library.file_name() else {
        return;
    };

    let _ = std::fs::copy(&library, profile_dir.join(name));

    if let Some(staged) = stage_under_cargo_home(&library, name) {
        println!("cargo:rustc-link-arg-bins=-Wl,-rpath,{}", staged.display());
    }
}

/// Copies the library somewhere an installed binary can still find it.
///
/// Returns the directory to record as an rpath, or `None` when there is
/// nowhere to put it -- in which case the binary is exactly as installable as
/// it was before, which is to say only alongside the tree that built it.
fn stage_under_cargo_home(library: &Path, name: &std::ffi::OsStr) -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var("CARGO_HOME").ok()?);
    // Named after this project rather than dropped into a shared `lib`: the
    // file belongs to one version of one binary, and a shared directory is how
    // two tools end up disagreeing about which copy is the right one.
    let staged = home.join("lib").join("pamin");

    std::fs::create_dir_all(&staged).ok()?;

    // Copying onto a library a running process has mapped would corrupt it, so
    // the copy lands beside the target and is moved into place. A rename within
    // one directory is atomic, and a process already running keeps the copy it
    // opened.
    let destination = staged.join(name);
    let pending = staged.join(format!(".{}.pending", name.to_string_lossy()));
    std::fs::copy(library, &pending).ok()?;
    std::fs::rename(&pending, &destination).ok()?;

    Some(staged)
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
