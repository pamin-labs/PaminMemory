//! The evaluation knobs, `PAMIN_EVAL_*`, read in one place.
//!
//! libtest owns a test binary's command line, so a harness cannot take flags
//! the way `pamin` does; its knobs are environment variables. They are named
//! `PAMIN_EVAL_*` so that a bare `PAMIN_*` always means a product setting, and
//! they are read here, once per run, into the values the libraries take --
//! never by the libraries, and never written back to the environment.
//!
//! Std only, so every crate's tests can include it with `#[path]`.

// Each including test uses some of it.
#![allow(dead_code)]

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Every knob a harness or test reads, with what it sets. A `PAMIN_EVAL_*`
/// variable not listed here is refused by [`check`]: a misspelled knob that
/// silently measured the default is the failure this list exists to stop.
pub const KNOBS: &[(&str, &str)] = &[
    // What a run measures: the product's settings, as a harness sets them.
    ("PAMIN_EVAL_DEVICE", "auto or cpu (`--device`)"),
    (
        "PAMIN_EVAL_INFERENCE_THREADS",
        "threads per pass (`--inference-threads`)",
    ),
    (
        "PAMIN_EVAL_EP_LIBRARIES",
        "plugin execution providers, a path list (`--ep-library`)",
    ),
    (
        "PAMIN_EVAL_PREPARED",
        "load the mapped copy, true or false (`--prepared`)",
    ),
    (
        "PAMIN_EVAL_FUSED_ATTENTION",
        "load the fused graph, true or false (`--fused-attention`)",
    ),
    (
        "PAMIN_EVAL_RERANK_BATCH",
        "pairs per reranker pass (no flag)",
    ),
    (
        "PAMIN_EVAL_RERANK_BATCH_TOKENS",
        "padded tokens per reranker pass (no flag)",
    ),
    (
        "PAMIN_EVAL_RERANK_MAX_TOKENS",
        "longest candidate the reranker reads (no flag)",
    ),
    (
        "PAMIN_EVAL_RERANK_DEPTH",
        "candidates reranked (`search --rerank-depth`)",
    ),
    (
        "PAMIN_EVAL_SEARCH_EFFORT",
        "vector graph search width (`search --search-effort`)",
    ),
    (
        "PAMIN_EVAL_UNINDEXED_BUDGET",
        "documents outside the vector graph before upkeep builds it (no flag)",
    ),
    ("PAMIN_EVAL_JIT", "PostgreSQL JIT, true or false (`--jit`)"),
    (
        "PAMIN_EVAL_POSTGRES_DIR",
        "a PostgreSQL installation (`--postgres-dir`)",
    ),
    // Where things are, and the protocol with re-executed children. These
    // change no result.
    ("PAMIN_EVAL_HOME", "datasets, models and workspaces"),
    (
        "PAMIN_EVAL_MODEL_CACHE",
        "the model cache a model test loads from",
    ),
    (
        "PAMIN_EVAL_MODELS",
        "an existing model cache, to copy instead of downloading",
    ),
    (
        "PAMIN_EVAL_JOINT_GRAPH",
        "a joint embedding graph to compare",
    ),
    (
        "PAMIN_EVAL_PARITY_CORPORA",
        "tokenizer parity corpora, a path list",
    ),
    ("PAMIN_EVAL_NATIVE_MODEL_SOURCE", "a cached FP16 BGE export"),
    (
        "PAMIN_EVAL_NATIVE_CACHE",
        "the native preparation test's cache",
    ),
    ("PAMIN_EVAL_EXPECT_DEVICE", "the device a child must report"),
    (
        "PAMIN_EVAL_EMBEDDING_OUTPUT",
        "where a child writes its embeddings",
    ),
    (
        "PAMIN_EVAL_RERANK_OUTPUT",
        "where a child writes its scores",
    ),
    ("PAMIN_EVAL_PREPARED_ARM", "a measuring child's arm"),
    ("PAMIN_EVAL_PREPARED_CACHE", "a measuring child's cache"),
    ("PAMIN_EVAL_SHARED_ARM", "a measuring child's arm"),
    ("PAMIN_EVAL_SHARED_CACHE", "a measuring child's cache"),
    (
        "PAMIN_EVAL_SHARED_EMBEDDER_COPY",
        "a prepared copy a child loads",
    ),
    (
        "PAMIN_EVAL_SHARED_ACCURATE_COPY",
        "a prepared copy a child loads",
    ),
];

/// Product variables a harness may find set and has no use for: they say
/// where a workspace is or what to log, which `pamin` reads and a harness
/// does not.
///
/// Not `PAMIN_VECTOR_INDEX`. It changes what `pamin` measures, a harness opens
/// the vector index it names in code, and a run with it set would be recorded
/// under that index while measuring the other one.
const PRODUCT: &[&str] = &["PAMIN_HOME", "PAMIN_PROJECT", "PAMIN_LOG"];

/// Variables a build script sets with `cargo:rustc-env` -- `pamin-index`'s
/// two and `pamin-engine`'s four -- which Cargo also sets on every test of
/// that package it runs. They describe the build, are no setting, and are
/// what a record names its build by, so no check refuses them and no arm is
/// named by them.
pub const BUILD: [&str; 6] = [
    "PAMIN_ENGINE_BUILD_ENVIRONMENT",
    "PAMIN_ENGINE_COMMIT",
    "PAMIN_ENGINE_DIRTY",
    "PAMIN_ENGINE_SOURCES",
    "PAMIN_INFERENCE_BUILD",
    "PAMIN_INFERENCE_CARGO",
];

/// Refuses any `PAMIN_*` variable a run would silently ignore: a misspelled
/// knob, or a product setting the libraries no longer read (`PAMIN_DEVICE`,
/// `PAMIN_RERANK_DEPTH`), which a harness takes as `PAMIN_EVAL_*` instead.
pub fn check() {
    for (name, _) in std::env::vars_os() {
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("PAMIN_")
            || PRODUCT.contains(&name)
            || BUILD.contains(&name)
            || KNOBS.iter().any(|(knob, _)| *knob == name)
        {
            continue;
        }
        let renamed = name.replacen("PAMIN_", "PAMIN_EVAL_", 1);
        if KNOBS.iter().any(|(knob, _)| *knob == renamed) {
            panic!("{name} is not read by a test: set {renamed} instead");
        }
        panic!("{name} is not read by a test; the knobs are listed in tests/support/eval.rs");
    }
}

/// The knob's value parsed as a `T`, or `None` when it is not set. Refuses a
/// value that does not parse, rather than measuring the default under a name
/// that says otherwise.
pub fn parsed<T: FromStr>(name: &str) -> Option<T>
where
    T::Err: Display,
{
    let value = text(name)?;
    match value.parse() {
        Ok(parsed) => Some(parsed),
        Err(error) => panic!("{name}={value:?}: {error}"),
    }
}

/// A true-or-false knob: `true`, `on` or `1`; `false`, `off` or `0`.
pub fn switch(name: &str) -> Option<bool> {
    let value = text(name)?;
    match value.to_ascii_lowercase().as_str() {
        "true" | "on" | "1" => Some(true),
        "false" | "off" | "0" => Some(false),
        _ => panic!("{name}={value:?}: expected true or false"),
    }
}

/// A path knob.
pub fn path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

/// A path knob that must be absolute and UTF-8, for a path a record is named
/// by: a relative one names different files from different working
/// directories under one spelling.
pub fn absolute_path(name: &str) -> Option<PathBuf> {
    let path = path(name)?;
    absolute(name, &path);
    Some(path)
}

/// A knob holding a list of absolute paths, split as `PATH` is.
///
/// Refuses a relative one: the product registers only absolute libraries and
/// warns past the rest, so an arm given one would run without it.
pub fn paths(name: &str) -> Vec<PathBuf> {
    let paths: Vec<PathBuf> = std::env::var_os(name)
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    for path in &paths {
        absolute(name, path);
    }
    paths
}

/// Refuses a relative path, and one that is not UTF-8: a record spells its
/// settings as text, and two paths that differ only in bytes text cannot hold
/// would be spelt, and named, alike. Refuses one that does not exist too: the
/// product warns past a library it cannot load, so the arm would run without
/// it under a record named by it.
fn absolute(name: &str, path: &Path) {
    assert!(
        path.is_absolute(),
        "{name}: {} is not an absolute path",
        path.display()
    );
    assert!(
        path.to_str().is_some(),
        "{name}: {} is not UTF-8",
        path.display()
    );
    assert!(path.exists(), "{name}: {} does not exist", path.display());
}

fn text(name: &str) -> Option<String> {
    debug_assert!(
        KNOBS.iter().any(|(knob, _)| *knob == name),
        "{name} is read but not listed in KNOBS"
    );
    match std::env::var(name) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("{name}: {error}"),
    }
}

/// `cargo test` sets this package's build-script variables on the test
/// itself, and the check lets them through. In a package whose build script
/// sets them, they are asserted present first, so the test fails rather than
/// passes if Cargo stops setting them; a package with none of them (the
/// store's tests include this file too) only runs the check.
#[test]
fn the_check_lets_build_variables_through() {
    let emits = matches!(env!("CARGO_PKG_NAME"), "pamin-index" | "pamin-engine");
    let set: Vec<&str> = BUILD
        .into_iter()
        .filter(|name| std::env::var_os(name).is_some())
        .collect();
    assert!(
        !emits || !set.is_empty(),
        "cargo test set none of {BUILD:?}, so this checks nothing"
    );
    check();
}
