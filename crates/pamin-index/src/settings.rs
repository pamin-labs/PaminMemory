//! What a caller decides about inference, passed in rather than read.
//!
//! Nothing in this crate reads the process environment for a setting. A
//! value read from the environment deep inside a model load is a value every
//! caller in the process shares, which a test can only vary by writing the
//! environment -- process-wide, and racing every other test thread in the same
//! binary. So the binary decides (flags, its configuration file) and hands the
//! result down, and a harness builds the value it wants.
//!
//! Each default is the constant documented beside the measurement that settled
//! it, so this file holds none of its own. The device choice, the defaults a
//! client shows for these settings, and how a path is spelled are in
//! `pamin_core::settings`, which the command-line client shares without
//! linking this crate.

use std::num::NonZeroUsize;
use std::path::PathBuf;

pub use pamin_core::settings::{DeviceChoice, shown_path, spelled_path, spelled_paths};

/// How the reranker batches and truncates. See the constants in
/// [`crate::reranking`] for what settled each default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RerankSettings {
    /// The most pairs a pass takes, or `None` for the measured rule: one pair
    /// a pass where CPU passes overlap, four otherwise.
    pub batch: Option<NonZeroUsize>,
    /// How many padded tokens one pass may hold. Never zero: a pass that
    /// may hold nothing would take no pair at all.
    pub batch_tokens: NonZeroUsize,
    /// The longest candidate the model reads, in tokens. Never zero: a
    /// reranker that reads nothing of a candidate scores none of them.
    pub max_tokens: NonZeroUsize,
}

impl Default for RerankSettings {
    fn default() -> Self {
        Self {
            batch: None,
            batch_tokens: NonZeroUsize::new(crate::reranking::BATCH_TOKENS)
                .expect("a pass holds at least one token"),
            max_tokens: NonZeroUsize::new(crate::reranking::MAX_TOKENS)
                .expect("the reranker reads at least one token"),
        }
    }
}

/// Everything about running a model that is the caller's to decide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceSettings {
    /// Which devices may be planned on.
    pub device: DeviceChoice,
    /// Threads one forward pass may use, or on the CPU how many single-thread
    /// passes run at once; `None` is one per core.
    pub threads: Option<NonZeroUsize>,
    /// Absolute paths of NPU plugin execution-provider libraries to register.
    pub ep_libraries: Vec<PathBuf>,
    /// Whether a CPU load reads a mapped copy of the model (see
    /// `crate::prepared`) rather than its download.
    pub prepared: bool,
    /// Whether a mapped copy's attention-fused graph is loaded where one was
    /// kept.
    pub fused_attention: bool,
    /// Whether a mapped copy's graph is loaded with its word embeddings read
    /// by this process rather than mapped by the runtime, where that graph was
    /// kept (see `crate::word_embeddings`).
    pub split_embeddings: bool,
    /// The reranker's batching and truncation.
    pub rerank: RerankSettings,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            device: DeviceChoice::default(),
            threads: None,
            ep_libraries: Vec::new(),
            prepared: pamin_core::settings::PREPARED,
            fused_attention: pamin_core::settings::FUSED_ATTENTION,
            split_embeddings: pamin_core::settings::SPLIT_EMBEDDINGS,
            rerank: RerankSettings::default(),
        }
    }
}

impl InferenceSettings {
    /// How many threads inference may keep busy: [`threads`](Self::threads),
    /// or one per core.
    pub fn budget(&self) -> std::io::Result<usize> {
        Ok(match self.threads {
            Some(threads) => threads.get(),
            None => std::thread::available_parallelism()?.get(),
        })
    }

    /// Every setting that differs from the default, as `(name, value)` pairs
    /// in a fixed order.
    ///
    /// What execution plans are keyed by, and what names an evaluation arm.
    /// Empty at the defaults, so a default configuration keys its plans
    /// exactly as a process with no settings in its environment did before
    /// settings were passed in, and plans measured then are reused.
    pub fn overrides(&self) -> Vec<(String, String)> {
        let default = Self::default();
        let mut overrides = Vec::new();
        let mut differs = |name: &str, differs: bool, value: String| {
            if differs {
                overrides.push((name.to_string(), value));
            }
        };
        differs(
            "device",
            self.device != default.device,
            self.device.label().to_string(),
        );
        differs(
            "inference_threads",
            self.threads != default.threads,
            self.threads.map_or(String::new(), |n| n.to_string()),
        );
        differs(
            "ep_libraries",
            self.ep_libraries != default.ep_libraries,
            spelled_paths(&self.ep_libraries),
        );
        differs(
            "prepared",
            self.prepared != default.prepared,
            self.prepared.to_string(),
        );
        differs(
            "fused_attention",
            self.fused_attention != default.fused_attention,
            self.fused_attention.to_string(),
        );
        differs(
            "split_embeddings",
            self.split_embeddings != default.split_embeddings,
            self.split_embeddings.to_string(),
        );
        differs(
            "rerank_batch",
            self.rerank.batch != default.rerank.batch,
            self.rerank.batch.map_or(String::new(), |n| n.to_string()),
        );
        differs(
            "rerank_batch_tokens",
            self.rerank.batch_tokens != default.rerank.batch_tokens,
            self.rerank.batch_tokens.to_string(),
        );
        differs(
            "rerank_max_tokens",
            self.rerank.max_tokens != default.rerank.max_tokens,
            self.rerank.max_tokens.to_string(),
        );
        overrides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_override_nothing() {
        assert!(InferenceSettings::default().overrides().is_empty());
    }

    #[test]
    fn each_setting_that_differs_is_named_once() {
        let settings = InferenceSettings {
            device: DeviceChoice::Cpu,
            threads: NonZeroUsize::new(2),
            prepared: false,
            rerank: RerankSettings {
                max_tokens: NonZeroUsize::new(192).unwrap(),
                ..RerankSettings::default()
            },
            ..InferenceSettings::default()
        };
        assert_eq!(
            settings.overrides(),
            [
                ("device", "cpu"),
                ("inference_threads", "2"),
                ("prepared", "false"),
                ("rerank_max_tokens", "192"),
            ]
            .map(|(name, value)| (name.to_string(), value.to_string()))
        );
    }
}
