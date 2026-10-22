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
    /// resident process gives them back, in seconds. See
    /// [`pamin_core::settings::MODEL_IDLE`]. Never zero, which would give a
    /// model back the tick after it loaded and make every search a model load.
    pub model_idle: std::num::NonZeroU64,
    /// How many documents may sit outside the vector graph before upkeep
    /// builds it; [`pamin_index::projection::UNINDEXED_BUDGET`] in the
    /// product, lowered by a test that has to reach the case. Never zero,
    /// which would optimize the index on every drain.
    pub unindexed_budget: std::num::NonZeroU64,
}

impl Settings {
    /// [`Self::model_idle`] as a duration.
    pub fn idle(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.model_idle.get())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            inference: InferenceSettings::default(),
            store: StoreSettings::default(),
            model_idle: std::num::NonZeroU64::new(pamin_core::settings::MODEL_IDLE.as_secs())
                .expect("a model may sit idle for at least a second"),
            unindexed_budget: pamin_index::projection::UNINDEXED_BUDGET,
        }
    }
}
