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
//! it, so this file holds none of its own.

use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Which devices a model may be planned on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeviceChoice {
    /// Every viable plan, measured: the mixed accelerator policy in ADR 0001.
    #[default]
    Auto,
    /// Optimized CPU only, for a measurement that must be comparable with a
    /// CPU one or a machine whose accelerator belongs to something else.
    Cpu,
}

impl DeviceChoice {
    /// Both, in the order the setting documents them.
    pub const ALL: [Self; 2] = [Self::Auto, Self::Cpu];

    /// The name the setting takes.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
        }
    }

    /// The choice a name stands for, ignoring ASCII case.
    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|choice| choice.label().eq_ignore_ascii_case(label.trim()))
    }
}

/// How the reranker batches and truncates. See the constants in
/// [`crate::reranking`] for what settled each default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RerankSettings {
    /// The most pairs a pass takes, or `None` for the measured rule: one pair
    /// a pass where CPU passes overlap, four otherwise.
    pub batch: Option<NonZeroUsize>,
    /// How many padded tokens one pass may hold.
    pub batch_tokens: usize,
    /// The longest candidate the model reads, in tokens.
    pub max_tokens: usize,
}

impl Default for RerankSettings {
    fn default() -> Self {
        Self {
            batch: None,
            batch_tokens: crate::reranking::BATCH_TOKENS,
            max_tokens: crate::reranking::MAX_TOKENS,
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
    /// The reranker's batching and truncation.
    pub rerank: RerankSettings,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            device: DeviceChoice::Auto,
            threads: None,
            ep_libraries: Vec::new(),
            prepared: true,
            fused_attention: true,
            rerank: RerankSettings::default(),
        }
    }
}

impl InferenceSettings {
    /// The settings the environment variables that configured inference
    /// before settings were passed in still name.
    ///
    /// Transitional: what callers use until the binary passes its own.
    pub fn from_environment() -> Self {
        let positive = |name: &str| pamin_core::env::positive::<usize>(name);
        let default = Self::default();
        Self {
            device: if pamin_core::env::is("PAMIN_DEVICE", "cpu") {
                DeviceChoice::Cpu
            } else {
                DeviceChoice::Auto
            },
            threads: positive("PAMIN_INFERENCE_THREADS").and_then(NonZeroUsize::new),
            ep_libraries: std::env::var_os("PAMIN_EP_LIBRARIES")
                .map(|paths| std::env::split_paths(&paths).collect())
                .unwrap_or_default(),
            prepared: !pamin_core::env::is("PAMIN_PREPARED", "off"),
            fused_attention: !pamin_core::env::is("PAMIN_FUSED_ATTENTION", "off"),
            rerank: RerankSettings {
                batch: positive("PAMIN_RERANK_BATCH").and_then(NonZeroUsize::new),
                batch_tokens: positive("PAMIN_RERANK_BATCH_TOKENS")
                    .unwrap_or(default.rerank.batch_tokens),
                max_tokens: positive("PAMIN_RERANK_MAX_TOKENS")
                    .unwrap_or(default.rerank.max_tokens),
            },
        }
    }

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
            std::env::join_paths(&self.ep_libraries)
                .map(|joined| joined.to_string_lossy().into_owned())
                .unwrap_or_else(|_| format!("{:?}", self.ep_libraries)),
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
                max_tokens: 192,
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

    #[test]
    fn a_device_is_read_by_name_ignoring_case() {
        assert_eq!(DeviceChoice::parse("CPU"), Some(DeviceChoice::Cpu));
        assert_eq!(DeviceChoice::parse("auto"), Some(DeviceChoice::Auto));
        assert_eq!(DeviceChoice::parse("gpu"), None);
    }
}
