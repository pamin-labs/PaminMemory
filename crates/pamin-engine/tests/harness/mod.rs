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

use pamin_core::Fusion;
use pamin_engine::{Depths, Drained, Engine, Owed, SearchHit, Write};

/// What records and projects call the one embedding model.
///
/// The name of the profile that shipped it, kept from when there were
/// profiles so that records and cached corpora written then are the ones
/// read and compared with now.
pub const PROFILE: &str = "accuracy";

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

/// The settings a run measures: the product's defaults, and whatever the
/// environment running the harness changes about them.
///
/// Read once per call into a value the harness hands to [`Engine::open`],
/// rather than read by the libraries on every load: the libraries read no
/// environment, so a sweep varies a setting by building another value, never
/// by writing a variable every other test thread in the binary shares.
pub fn settings() -> pamin_engine::Settings {
    let default = pamin_engine::Settings::default();
    pamin_engine::Settings {
        inference: pamin_index::InferenceSettings::from_environment(),
        store: pamin_store::StoreSettings::from_environment(),
        model_idle: pamin_core::env::positive("PAMIN_MODEL_IDLE")
            .map(std::time::Duration::from_secs)
            .unwrap_or(default.model_idle),
        unindexed_budget: pamin_core::env::positive("PAMIN_UNINDEXED_BUDGET")
            .unwrap_or(default.unindexed_budget),
    }
}

/// The depths a run searches with: the product's, and the reranker's depth
/// and the vector graph's width as the environment running the harness says.
/// A search's own value, so an arm that varies one passes another [`Depths`].
pub fn depths() -> Depths {
    Depths {
        rerank: pamin_core::env::positive("PAMIN_RERANK_DEPTH").unwrap_or(Depths::DEFAULT.rerank),
        effort: pamin_core::env::positive("PAMIN_SEARCH_EFFORT").unwrap_or(Depths::DEFAULT.effort),
        ..Depths::DEFAULT
    }
}

/// [`PROFILE`], once nothing asks for another.
///
/// `PAMIN_PROFILE` chose the embedding model a harness measured. It chooses
/// nothing now, and a run that set it would be recorded under the name of a
/// model it did not use, so it is refused rather than ignored.
pub fn profile() -> String {
    if let Some(named) = std::env::var_os("PAMIN_PROFILE") {
        panic!(
            "PAMIN_PROFILE={} was removed: there is one embedding model. Unset it to continue.",
            named.to_string_lossy()
        );
    }
    PROFILE.to_string()
}

/// Whether a measured search reranks: the shipped search, or fusion alone.
///
/// The product has one search, and it reranks. What an arm compares it with
/// is fusion alone, [`Engine::search_fused`] -- the list the reranker is
/// handed, cut at the caller's limit. Named as the reranking tiers were, `off`
/// and `accurate`, so records written while there were tiers compare with
/// the ones written since.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rerank {
    Off,
    #[default]
    Accurate,
}

impl Rerank {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Accurate => "accurate",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "accurate" => Some(Self::Accurate),
            _ => None,
        }
    }

    /// The search this arm runs.
    pub async fn search(
        self,
        engine: &Engine,
        query: &str,
        limit: u32,
        depths: Depths,
    ) -> anyhow::Result<Vec<SearchHit>> {
        match self {
            Self::Off => {
                engine
                    .search_fused(query, limit, depths, Fusion::default())
                    .await
            }
            Self::Accurate => engine.search_reranked(query, limit, depths).await,
        }
    }
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
