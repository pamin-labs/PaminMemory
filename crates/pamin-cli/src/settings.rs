//! The settings the server runs with, from the variables that configured it.
//!
//! Transitional: read from the same `PAMIN_*` variables the libraries read
//! before settings were passed in, until the binary takes them as flags.

use std::time::Duration;

use pamin_engine::{Depths, Settings};

/// The settings of this process.
pub fn from_environment() -> Settings {
    let default = Settings::default();
    Settings {
        inference: pamin_index::InferenceSettings::from_environment(),
        store: pamin_store::StoreSettings::from_environment(),
        model_idle: pamin_core::env::positive("PAMIN_MODEL_IDLE")
            .map(Duration::from_secs)
            .unwrap_or(default.model_idle),
        unindexed_budget: pamin_core::env::positive("PAMIN_UNINDEXED_BUDGET")
            .unwrap_or(default.unindexed_budget),
    }
}

/// A search's depths: the two a request carries, and the reranker's depth and
/// the vector graph's width as the server's environment says.
pub fn depths(channel: u32, graph: u8) -> Depths {
    Depths {
        channel,
        graph,
        rerank: pamin_core::env::positive("PAMIN_RERANK_DEPTH").unwrap_or(Depths::DEFAULT.rerank),
        effort: pamin_core::env::positive("PAMIN_SEARCH_EFFORT").unwrap_or(Depths::DEFAULT.effort),
    }
}
