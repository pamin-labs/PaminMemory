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

use std::path::PathBuf;

use pamin_engine::{Drained, Engine, Owed, Write};
use pamin_index::Profile;

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
