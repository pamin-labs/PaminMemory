//! Names the commit this crate is compiled from, as `PAMIN_ENGINE_COMMIT`,
//! and whether its checkout had tracked changes then, as
//! `PAMIN_ENGINE_DIRTY`.
//!
//! Only the evaluation harnesses read it (`tests/harness/shape.rs`): a shape
//! record names the evaluated commit, and a test binary built at one commit
//! and run after a checkout moved would otherwise name the later one. Only
//! a run compiled from a clean checkout, and run from a clean checkout at
//! the same commit, writes the canonical record: edits compiled in and
//! restored before the run leave no trace a run could see.
//!
//! The script reruns when `HEAD` or the ref it points at moves, when the
//! index changes (a stage, or a restore), and when anything the workspace
//! compiles changes: its crates, its manifests, and the files outside the
//! crates that `include_str!` or `include_bytes!` reads. So the flag follows
//! an edit made after a clean build. `SOURCES` names those roots, and a
//! harness test fails when an include reads from outside them. The dirty
//! check itself covers every tracked file, a superset of `SOURCES`, and the
//! untracked and ignored files under `SOURCES` that a build compiles in
//! (`tests/harness/compiled.rs`, shared with the run-time check).
//! `HEAD`'s reflog is watched as well: a commit on a branch whose ref was
//! packed creates a loose ref that did not exist to be watched, and leaves
//! `HEAD` and `packed-refs` alone, but it appends to the reflog. A worktree
//! keeps its own `HEAD`, reflog and index in its own git dir, and shares the
//! refs in the common one.
//!
//! It also names the build variables that choose the native libraries
//! linked in, as `PAMIN_ENGINE_BUILD_ENVIRONMENT`, which pamin-index's build
//! fingerprint does not hold; a record holds both.
//!
//! The flag errs one way only. A flag left at "dirty" by a rerun that did not
//! happen marks a clean run noncanonical, which costs a rerun of the
//! evaluation; it never makes a dirty build canonical, and `git` failing to
//! say counts as dirty for the same reason. Outside a git checkout, as from
//! a source archive, the value is `unknown` and nothing changes.

use std::path::PathBuf;
use std::process::Command;

#[path = "tests/harness/compiled.rs"]
mod compiled;

/// What the workspace compiles, relative to its root.
const SOURCES: [&str; 4] = [
    "crates",
    "Cargo.toml",
    "Cargo.lock",
    // Benchmark manifests the evaluation harnesses compile in.
    "benchmarks/results/retrieval",
];

/// Build variables that change what is compiled or linked in: those the
/// `ort-sys` and `zvec` build scripts read to choose the native ONNX Runtime
/// and zvec libraries, and a rustc wrapper. They are what
/// `benchmarks/harnesses/restart-floor-2026-09-30/build-guarded.py.in`
/// records as a build's provenance (`ORT_*`, `ZVEC_*`), but for the
/// toolchain and its flags, which are recorded as rustc's version and the
/// flags it was given. Named one by one, because Cargo reruns this script on
/// a change to a variable only when it was named.
const BUILD_ENVIRONMENT: [&str; 20] = [
    "ORT_CUDA_VERSION",
    "ORT_CXX_STDLIB",
    "ORT_EXT_IOS_XCFWK_LOCATION",
    "ORT_EXT_IOS_XCFWK_PATH",
    "ORT_IOS_XCFWK_LOCATION",
    "ORT_IOS_XCFWK_PATH",
    "ORT_LIB_LOCATION",
    "ORT_LIB_PATH",
    "ORT_LIB_PROFILE",
    "ORT_OFFLINE",
    "ORT_PREFER_DYNAMIC_LINK",
    "ORT_SKIP_DOWNLOAD",
    "ORT_VCPKG_TARGET",
    "ZVEC_AUTO_BUILD",
    "ZVEC_INCLUDE_DIR",
    "ZVEC_LIB_DIR",
    "ZVEC_PREBUILT_URL",
    "ZVEC_REPO_URL",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
];

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8(output.stdout).ok()?.trim().to_string())
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let commit = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    println!("cargo:rustc-env=PAMIN_ENGINE_COMMIT={commit}");
    // Without the optional lock, `status` does not refresh the index it
    // would otherwise rewrite, which is watched below.
    let tracked = git(&[
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=no",
    ])
    .is_none_or(|changes| !changes.is_empty());
    // Untracked and ignored files count only where a build compiles them in,
    // such as a gitignored `tests/scratch_*.rs`.
    let args = compiled::status_args(&SOURCES.join(":"));
    let untracked = git(&args.iter().map(String::as_str).collect::<Vec<_>>())
        .is_none_or(|status| compiled::compiled_in(&status));
    let dirty = tracked || untracked;
    println!("cargo:rustc-env=PAMIN_ENGINE_DIRTY={dirty}");
    println!("cargo:rustc-env=PAMIN_ENGINE_SOURCES={}", SOURCES.join(":"));
    // The build environment that chooses the native libraries linked in and
    // how rustc is wrapped, as the restart-floor harness records it:
    // `NAME=value` for each one set, separated by 0x1f.
    let build_environment: Vec<String> = BUILD_ENVIRONMENT
        .iter()
        .filter_map(|name| {
            println!("cargo:rerun-if-env-changed={name}");
            std::env::var(name)
                .ok()
                .map(|value| format!("{name}={value}"))
        })
        .collect();
    println!(
        "cargo:rustc-env=PAMIN_ENGINE_BUILD_ENVIRONMENT={}",
        build_environment.join("\x1f")
    );

    // Watched only if they exist: Cargo reruns on every build for a path
    // that does not.
    let mut watched: Vec<PathBuf> = Vec::new();
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let git_dir = PathBuf::from(git_dir);
        watched.push(git_dir.join("HEAD"));
        watched.push(git_dir.join("logs").join("HEAD"));
        watched.push(git_dir.join("index"));
    }
    if let Some(common) = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]) {
        let common = PathBuf::from(common);
        watched.push(common.join("packed-refs"));
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
            watched.push(common.join(reference));
        }
    }
    if let Some(root) = git(&["rev-parse", "--show-toplevel"]) {
        let root = PathBuf::from(root);
        // Every crate in the workspace, not only this one: the harnesses
        // compile its dependencies too. The build output is outside them.
        watched.extend(SOURCES.map(|path| root.join(path)));
    }
    for path in watched.into_iter().filter(|path| path.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
