//! Which untracked or ignored files a build of the workspace could compile in.
//!
//! Shared by the engine's build script, which records whether a binary was
//! built from a dirty checkout, and by `shape`, which asks the same of the
//! checkout a run starts in. A tracked change anywhere counts as dirty in
//! both; this decides the untracked and ignored files, which `git` only lists
//! when asked for them -- a gitignored `tests/scratch_*.rs` is compiled into
//! its test binary all the same.

/// The `git status` arguments that list the untracked and ignored files
/// under `sources`, the workspace-relative roots the build script watches.
/// `:(top)` makes each root relative to the repository, wherever `git` runs.
pub fn status_args(sources: &str) -> Vec<String> {
    let mut args: Vec<String> = [
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--ignored",
        "--",
    ]
    .map(String::from)
    .to_vec();
    args.extend(sources.split(':').map(|source| format!(":(top){source}")));
    args
}

/// Whether `status`, the output of [`status_args`], names a file a build
/// could compile in. Only untracked (`??`) and ignored (`!!`) lines are read;
/// tracked changes are counted over the whole checkout elsewhere.
pub fn compiled_in(status: &str) -> bool {
    status.lines().any(|line| {
        let (code, path) = line.split_at(line.len().min(3));
        matches!(code.trim_end(), "??" | "!!") && compiled(path.trim_matches('"'))
    })
}

/// Whether the file at `path`, relative to the repository, could be
/// compiled: Rust source in a crate's `src`, `tests`, `benches` or `examples`
/// tree, a build script, or a manifest. Anything under a `target` directory
/// is build output. Outside `crates/`, every file under a watched root is a
/// file a harness compiles in with `include_str!`.
pub fn compiled(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.contains(&"target") {
        return false;
    }
    if parts.first() != Some(&"crates") {
        return true;
    }
    let name = parts.last().copied().unwrap_or_default();
    if matches!(name, "Cargo.toml" | "Cargo.lock" | "build.rs") {
        return true;
    }
    name.ends_with(".rs")
        && parts
            .iter()
            .any(|part| matches!(*part, "src" | "tests" | "benches" | "examples"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ignored_scratch_test_is_compiled_in() {
        assert!(compiled_in(
            "!! crates/pamin-engine/tests/scratch_rules.rs\n"
        ));
        assert!(compiled_in("?? crates/pamin-index/src/new.rs\n"));
        assert!(compiled_in("?? crates/pamin-cli/Cargo.toml\n"));
        assert!(compiled_in("?? benchmarks/results/retrieval/extra.json\n"));
        assert!(compiled_in("?? \"crates/pamin-core/src/with space.rs\"\n"));
    }

    #[test]
    fn build_output_and_stray_files_are_not() {
        assert!(!compiled_in(
            "!! crates/pamin-engine/target/debug/build.rs\n"
        ));
        assert!(!compiled_in(
            "!! crates/pamin-engine/tests/__pycache__/x.pyc\n"
        ));
        assert!(!compiled_in("?? crates/pamin-engine/src/notes.md\n"));
        assert!(!compiled_in("!! crates/pamin-engine/src/lib.rs.bk\n"));
        assert!(!compiled_in(""));
    }

    #[test]
    fn tracked_changes_are_left_to_the_whole_checkout_count() {
        assert!(!compiled_in(" M crates/pamin-engine/src/lib.rs\n"));
    }

    #[test]
    fn every_watched_root_is_asked_for_from_the_top() {
        let args = status_args("crates:Cargo.toml");
        assert_eq!(
            &args[args.len() - 2..],
            [":(top)crates", ":(top)Cargo.toml"]
        );
    }
}
