//! Names the commit this crate is compiled from, as `PAMIN_ENGINE_COMMIT`.
//!
//! Only the evaluation harnesses read it (`tests/harness/shape.rs`): a shape
//! record names the evaluated commit, and a test binary built at one commit
//! and run after a checkout moved would otherwise name the later one. Only
//! a run whose clean checkout is at this commit writes the canonical record.
//!
//! The script reruns only when `HEAD` or the ref it points at moves, so an
//! ordinary edit does not rebuild the crate. `HEAD`'s reflog is watched as
//! well: a commit on a branch whose ref was packed creates a loose ref that
//! did not exist to be watched, and leaves `HEAD` and `packed-refs` alone,
//! but it appends to the reflog. A worktree keeps its own `HEAD` and reflog
//! in its own git dir, and shares the refs in the common one. Outside a git checkout, as from
//! a source archive, the value is `unknown` and nothing changes.

use std::path::PathBuf;
use std::process::Command;

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

    // Watched only if they exist: Cargo reruns on every build for a path
    // that does not.
    let mut watched: Vec<PathBuf> = Vec::new();
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let git_dir = PathBuf::from(git_dir);
        watched.push(git_dir.join("HEAD"));
        watched.push(git_dir.join("logs").join("HEAD"));
    }
    if let Some(common) = git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]) {
        let common = PathBuf::from(common);
        watched.push(common.join("packed-refs"));
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
            watched.push(common.join(reference));
        }
    }
    for path in watched.into_iter().filter(|path| path.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
