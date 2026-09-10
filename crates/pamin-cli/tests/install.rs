//! Checks that an installed `pamin` runs without the tree that built it.
//!
//! The retrieval engine is a dynamic library. Cargo installs the executable
//! and nothing beside it, so an installed binary can only work by finding the
//! library somewhere else -- and if that somewhere else is `target/`, the next
//! `cargo clean` breaks a binary that worked yesterday.
//!
//! Nothing else covers this. Every other test runs the binary through
//! `CARGO_BIN_EXE_pamin`, which is the copy inside `target/` with the build
//! tree still around it, so the one arrangement a user actually gets is the one
//! arrangement never exercised. That is the same shape as the two defects the
//! end-to-end suite exists for, and it is why this is a separate file: it
//! builds, installs, and inspects rather than driving a workspace.
//!
//! Ignored by default because it runs a release build.

use std::path::Path;
use std::process::Command;

/// Strips the library search path Cargo sets for test binaries.
///
/// Cargo points `LD_LIBRARY_PATH` at the build directories so a test binary can
/// find native libraries, and a child process inherits it. Leaving it in place
/// would let the installed binary find the engine through the build tree and
/// report that everything is fine -- which is how the defect this file is about
/// survived in the first place.
fn as_a_user_would(command: &mut Command) -> &mut Command {
    command
        .env_remove("LD_LIBRARY_PATH")
        .env_remove("DYLD_LIBRARY_PATH")
        .env_remove("DYLD_FALLBACK_LIBRARY_PATH")
}

/// Installs into a temporary root and runs what came out.
#[test]
#[ignore = "runs a release build and installs it"]
fn an_installed_binary_runs_without_the_tree_that_built_it() {
    let root = tempfile::tempdir().expect("temp install root");
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    let install = Command::new(env!("CARGO"))
        .args(["install", "--path"])
        .arg(manifest)
        .arg("--root")
        .arg(root.path())
        .args(["--locked", "--force"])
        .output()
        .expect("running cargo install");

    assert!(
        install.status.success(),
        "cargo install failed: {}",
        String::from_utf8_lossy(&install.stderr)
    );

    let installed = root.path().join("bin").join("pamin");
    let version = as_a_user_would(Command::new(&installed).arg("--version"))
        .output()
        .unwrap_or_else(|error| panic!("running {}: {error}", installed.display()));

    assert!(
        version.status.success(),
        "the installed binary did not start: {}",
        String::from_utf8_lossy(&version.stderr)
    );

    assert_no_library_comes_from_the_build_tree(&installed);
}

/// Fails when a library the installed binary needs resolves inside `target/`.
///
/// Starting is not enough on the machine that just built it: the build tree is
/// still there, so a binary that depends on it starts here and fails for
/// everyone else. What has to hold is where the libraries come from.
///
/// Linux only. `ldd` is what reports resolved paths, and the equivalent on
/// macOS reports what was recorded rather than what was found, which is a
/// weaker claim than this test is making.
#[cfg(target_os = "linux")]
fn assert_no_library_comes_from_the_build_tree(installed: &Path) {
    let target = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .join("target");

    let resolved = as_a_user_would(Command::new("ldd").arg(installed))
        .output()
        .expect("running ldd");
    let resolved = String::from_utf8_lossy(&resolved.stdout);

    let from_the_tree: Vec<&str> = resolved
        .lines()
        .filter(|line| line.contains(&*target.to_string_lossy()))
        .collect();

    assert!(
        from_the_tree.is_empty(),
        "the installed binary loads libraries out of the build tree, so it \
         stops working at the next `cargo clean`:\n{}",
        from_the_tree.join("\n")
    );
}

#[cfg(not(target_os = "linux"))]
fn assert_no_library_comes_from_the_build_tree(_installed: &Path) {}
