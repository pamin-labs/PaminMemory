//! What more than one of the index's test files needs.

// Each including test uses some of it.
#![allow(dead_code)]

pub mod eval;

use pamin_index::{DeviceChoice, InferenceSettings, RerankSettings};

/// The inference settings a test or harness measures: the defaults, and what
/// the `PAMIN_EVAL_*` knobs change about them. Refuses an unknown or retired
/// `PAMIN_*` variable first; see [`eval::check`].
pub fn inference() -> InferenceSettings {
    eval::check();
    let default = InferenceSettings::default();
    InferenceSettings {
        device: eval::parsed::<String>("PAMIN_EVAL_DEVICE")
            .map(|name| {
                DeviceChoice::parse(&name)
                    .unwrap_or_else(|| panic!("PAMIN_EVAL_DEVICE={name:?}: expected auto or cpu"))
            })
            .unwrap_or(default.device),
        threads: eval::parsed("PAMIN_EVAL_INFERENCE_THREADS"),
        ep_libraries: eval::paths("PAMIN_EVAL_EP_LIBRARIES"),
        prepared: eval::switch("PAMIN_EVAL_PREPARED").unwrap_or(default.prepared),
        fused_attention: eval::switch("PAMIN_EVAL_FUSED_ATTENTION")
            .unwrap_or(default.fused_attention),
        rerank: RerankSettings {
            batch: eval::parsed("PAMIN_EVAL_RERANK_BATCH"),
            batch_tokens: eval::parsed::<std::num::NonZeroUsize>("PAMIN_EVAL_RERANK_BATCH_TOKENS")
                .map_or(default.rerank.batch_tokens, |n| n.get()),
            max_tokens: eval::parsed::<std::num::NonZeroUsize>("PAMIN_EVAL_RERANK_MAX_TOKENS")
                .map_or(default.rerank.max_tokens, |n| n.get()),
        },
    }
}
