//! The plumbing every evaluation harness needs and none of them measures:
//! where its data lives, which profile it runs, and how a corpus gets into an
//! engine.
//!
//! Each harness had its own copy of each of these. They agreed, which is the
//! reason to keep one: a harness whose corpus went in differently from
//! another's would be comparing two write paths rather than two corpora.

// Included by harnesses that use only some of it, like the other shared
// modules here.
#![allow(dead_code)]

pub mod compiled;
pub mod shape;
pub mod vectors;

use std::path::PathBuf;

use pamin_engine::{Drained, Engine, Owed, Write};
use pamin_index::Profile;

/// This process's turn at loading models: held by an evaluation from
/// before it reads its settings ([`shape::Shape::publish`]) until its record
/// is written, and by every other test here that loads a model
/// ([`load_models`]), so no two of them load at once.
///
/// It is also the turn at the process environment: an experimental arm
/// that sets a variable the product reads (`PAMIN_RERANK_DEPTH`,
/// `PAMIN_SEARCH_EFFORT`) does so only while holding it ([`turn_taken`]),
/// so no evaluation's settings snapshot reads another test's transient arm.
///
/// What the loads fetched and prepared is counted process wide
/// (`pamin_index::LoadActivity`), and libtest runs the tests of one binary
/// side by side: without turns, another test's download would land in an
/// evaluation's interval and name a warm run cold. A flag and a condition
/// variable rather than a held `MutexGuard`, so the turn is `Send` and is
/// let go of however the test ends, a panic included.
pub struct ModelsTurn(());

static TURN: std::sync::Mutex<bool> = std::sync::Mutex::new(false);
static TURN_FREED: std::sync::Condvar = std::sync::Condvar::new();

/// Waits for this process's turn at loading models (see [`ModelsTurn`]).
pub fn load_models() -> ModelsTurn {
    let mut taken = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while *taken {
        taken = TURN_FREED
            .wait(taken)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    *taken = true;
    ModelsTurn(())
}

/// Whether some test holds the turn (see [`ModelsTurn`]): asserted by code
/// that sets the process environment for an experimental arm, which runs
/// only inside a test that holds it.
pub fn turn_taken() -> bool {
    *TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Drop for ModelsTurn {
    fn drop(&mut self) {
        *TURN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = false;
        TURN_FREED.notify_one();
    }
}

/// The profile the floors were measured against, and the product default.
pub const DEFAULT_PROFILE: &str = "accuracy";

/// Where the harnesses keep datasets, models and workspaces between runs:
/// `PAMIN_EVAL_HOME`, or `pamin-eval` under the system's temporary directory.
pub fn eval_home() -> PathBuf {
    std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"))
}

/// Where one dataset lives: the directory `var` names, or `name` under
/// [`eval_home`].
pub fn dataset_dir(var: &str, name: &str) -> PathBuf {
    match std::env::var(var) {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => eval_home().join(name),
    }
}

/// The profile to measure: `PAMIN_PROFILE`, or [`DEFAULT_PROFILE`].
pub fn profile() -> (String, Profile) {
    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).unwrap_or_else(|| panic!("unknown profile {named}"));
    (named, profile)
}

/// Writes one memory, promoted, unless its topic is already in the project.
///
/// Whether it wrote, so a harness resuming over a workspace it filled on an
/// earlier run can say how much of the corpus was new.
pub async fn write_absent(
    engine: &Engine,
    topic: &str,
    content: &str,
    language: &str,
    reason: &str,
) -> bool {
    let existing =
        pamin_store::repository::find_topic(engine.database.pool(), engine.project, topic)
            .await
            .expect("look for the topic");
    if existing.is_some() {
        return false;
    }
    engine
        .write(&Write {
            topic,
            content,
            content_hash: &content.len().to_string(),
            verdict: pamin_core::FilterDecision::Promoted,
            reason,
            promoted: true,
            language: Some(language),
            language_confidence: None,
            observed_at: time::OffsetDateTime::now_utc(),
            validity: pamin_core::Validity::ALWAYS,
        })
        .await
        .unwrap_or_else(|error| panic!("writing {topic}: {error}"));
    true
}

/// Hands `engine`'s index to `shape` under `label` before the run writes
/// any of its `total` documents (see `Shape::indexed`): loads the embedder,
/// so the producer that will embed them is known, and counts what the index
/// already holds.
pub async fn index(
    shape: &mut shape::Shape,
    label: &str,
    engine: &Engine,
    workspace: &std::path::Path,
    total: usize,
) {
    let loading = engine.clone();
    tokio::task::spawn_blocking(move || loading.load_embedder())
        .await
        .expect("the embedder load panicked")
        .expect("load the embedding model");
    let found = engine.indexed_documents().expect("count documents") as usize;
    shape.indexed(label, engine, workspace, found, total);
}

/// Runs the cascade until nothing is owed, and fails if anything still is.
pub async fn drain(engine: &Engine) -> Drained {
    let drained = engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain the cascade");
    assert_eq!(
        drained.pending, 0,
        "the corpus is not fully indexed: {} jobs still owed",
        drained.pending
    );
    drained
}
