//! Everything about a process that its caller decides, in one value.
//!
//! The engine, the index and the store read no environment for a setting:
//! the binary builds a [`Settings`] from its flags and its configuration file,
//! and a harness builds the one it measures. Each default is the constant the
//! measurement that settled it is documented beside, so this file holds none.
//!
//! What varies per search rather than per process -- how deep each channel,
//! the graph and the reranker reach, how wide the vector graph is searched --
//! is [`crate::Depths`], passed with the search.

use std::time::Duration;

use pamin_index::InferenceSettings;
use pamin_store::StoreSettings;

/// The settings of one process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// How models load and run: device, threads, mapped copies, batching.
    pub inference: InferenceSettings,
    /// How the workspace's PostgreSQL cluster is installed and configured.
    pub store: StoreSettings,
    /// How long a model, and the indexes pinning it, may sit unused before a
    /// resident process gives them back. See `MODEL_IDLE` in `engine.rs`.
    pub model_idle: Duration,
    /// How many documents may sit outside the vector graph before upkeep
    /// builds it; [`pamin_index::projection::UNINDEXED_BUDGET`] in the
    /// product, lowered by a test that has to reach the case. Never zero,
    /// which would optimize the index on every drain.
    pub unindexed_budget: std::num::NonZeroU64,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            inference: InferenceSettings::default(),
            store: StoreSettings::default(),
            model_idle: crate::engine::MODEL_IDLE,
            unindexed_budget: pamin_index::projection::UNINDEXED_BUDGET,
        }
    }
}
