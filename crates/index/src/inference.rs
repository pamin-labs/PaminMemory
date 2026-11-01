//! What this crate's settings mean to `pamin_runtime`'s device choice.
//!
//! The devices a machine offers, loading a session and choosing a model's
//! device by measuring it are `pamin_runtime`'s (`device`, `session`,
//! `plan`). What stays here is how [`crate::InferenceSettings`] keys that
//! choice: which accelerators it allows, and which of its settings name a
//! measured plan.

use pamin_runtime::device::Target;
pub use pamin_runtime::device::{CUDA_SELECTION, Device, cuda_devices};
pub use pamin_runtime::plan::{PlanOrigin, PlanUse};
use pamin_runtime::plan::{ReferenceShape, References, RuntimeModel};

use crate::error::Result;

/// The accelerators a model may try before the CPU, best first; see
/// [`pamin_runtime::device::accelerators`]. [`crate::DeviceChoice::Cpu`]
/// empties this, for a measurement that must be comparable with a CPU one or
/// a machine whose GPU belongs to something else.
pub(crate) fn accelerators(settings: &crate::InferenceSettings) -> Result<Vec<(Device, Target)>> {
    if settings.device == crate::DeviceChoice::Cpu {
        return Ok(Vec::new());
    }
    Ok(pamin_runtime::device::accelerators(&settings.ep_libraries)?)
}

/// Legacy discovery-order helper retained only for policy regression tests.
#[cfg(test)]
pub(crate) fn preferred<T>(
    mut load: impl FnMut(Device, Target) -> Result<T>,
) -> Result<(T, Device)> {
    for (device, provider) in accelerators(&crate::InferenceSettings::default())? {
        match load(device, provider) {
            Ok(model) => return Ok((model, device)),
            Err(error) => tracing::warn!(
                device = device.name(),
                %error,
                "model could not use this accelerator; trying the next provider"
            ),
        }
    }
    load(Device::Cpu, vec![pamin_runtime::device::cpu()].into()).map(|model| (model, Device::Cpu))
}

/// The settings an execution plan is keyed by, as its namespace and key spell
/// them: every [`EXECUTION_SETTINGS`] setting that differs from the default,
/// then the thread count.
///
/// The plans were keyed by every `PAMIN_MEMORY_` variable in the environment before
/// settings were passed in, and by `threads:None` where none was set, so a
/// default configuration spells this exactly as a clean environment did and
/// the plans it measured are reused.
fn dispatch_identity(settings: &crate::InferenceSettings) -> String {
    format!(
        "{:?}|threads:{:?}",
        execution_overrides(settings),
        settings.threads.map(std::num::NonZeroUsize::get)
    )
}

/// The settings that change how a model executes, as
/// [`InferenceSettings::overrides`](crate::InferenceSettings::overrides)
/// names them: the device it may run on, the provider libraries and the
/// threads it runs with, and which copy of its weights -- the download, or a
/// prepared copy with or without fused attention, and with or without its
/// word embeddings split out -- it loads. A plan's key is named by these.
///
/// Only these: a plan's key named every `PAMIN_MEMORY_` variable set, so a knob
/// that never reaches a model -- the reranker's depth, the search's effort,
/// where the workspace is -- recalibrated every model and renamed every
/// plan. What a model's own settings change, such as the reranker's batch
/// and token limits, its identity names (see `reranking`).
pub const EXECUTION_SETTINGS: [&str; 6] = [
    "device",
    "inference_threads",
    "ep_libraries",
    "prepared",
    "fused_attention",
    "split_embeddings",
];

/// The [`EXECUTION_SETTINGS`] among `settings`' overrides, in the order
/// [`overrides`](crate::InferenceSettings::overrides) gives them.
fn execution_overrides(settings: &crate::InferenceSettings) -> Vec<(String, String)> {
    settings
        .overrides()
        .into_iter()
        .filter(|(name, _)| EXECUTION_SETTINGS.contains(&name.as_str()))
        .collect()
}

/// `identity`'s model on the device `settings` allow that measures fastest;
/// see [`pamin_runtime::plan::measured`], keyed by this build and by the
/// settings that change how a model executes.
pub(crate) fn measured<T: RuntimeModel>(
    settings: &crate::InferenceSettings,
    identity: &str,
    cache_dir: &std::path::Path,
    references: &std::cell::RefCell<References>,
    shape: ReferenceShape,
    load: impl FnMut(Device, Target, bool) -> Result<T>,
    evaluate: impl FnMut(&mut T, Device) -> Result<std::time::Duration>,
) -> Result<(T, Device)> {
    pamin_runtime::plan::measured(
        pamin_runtime::plan::Measure {
            accelerators: accelerators(settings)?,
            dispatch: &dispatch_identity(settings),
            build: env!("PAMIN_MEMORY_INFERENCE_BUILD"),
            ep_libraries: &settings.ep_libraries,
        },
        identity,
        cache_dir,
        references,
        shape,
        load,
        evaluate,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IndexError;

    /// Plans measured before settings were passed in are reused: a default
    /// configuration spells the settings part of a plan's namespace and key
    /// exactly as a process with no `PAMIN_MEMORY_` variable set did, an empty list
    /// of variables and no thread count.
    #[test]
    fn a_default_configuration_keys_plans_as_a_clean_environment_did() {
        let clean: Vec<(std::ffi::OsString, std::ffi::OsString)> = Vec::new();
        assert_eq!(
            dispatch_identity(&crate::InferenceSettings::default()),
            format!("{clean:?}|threads:{:?}", None::<usize>)
        );
        assert_eq!(
            dispatch_identity(&crate::InferenceSettings::default()),
            "[]|threads:None"
        );
        let cpu = crate::InferenceSettings {
            device: crate::DeviceChoice::Cpu,
            ..crate::InferenceSettings::default()
        };
        assert_ne!(
            dispatch_identity(&cpu),
            dispatch_identity(&crate::InferenceSettings::default())
        );
    }

    /// A plan's key names the settings that change how a model executes,
    /// and no other: the reranker's batching and truncation calibrate
    /// nothing again, and every execution setting does.
    #[test]
    fn a_plan_key_names_only_execution_settings() {
        use std::num::NonZeroUsize;
        let shipped = crate::InferenceSettings::default();
        let reranking = crate::InferenceSettings {
            rerank: crate::RerankSettings {
                batch: NonZeroUsize::new(8),
                batch_tokens: NonZeroUsize::new(4096).unwrap(),
                max_tokens: NonZeroUsize::new(192).unwrap(),
            },
            ..crate::InferenceSettings::default()
        };
        assert_eq!(reranking.overrides().len(), 3);
        assert_eq!(dispatch_identity(&reranking), dispatch_identity(&shipped));
        let executing = [
            crate::InferenceSettings {
                device: crate::DeviceChoice::Cpu,
                ..crate::InferenceSettings::default()
            },
            crate::InferenceSettings {
                threads: NonZeroUsize::new(2),
                ..crate::InferenceSettings::default()
            },
            crate::InferenceSettings {
                ep_libraries: vec!["/npu/provider.so".into()],
                ..crate::InferenceSettings::default()
            },
            crate::InferenceSettings {
                prepared: false,
                ..crate::InferenceSettings::default()
            },
            crate::InferenceSettings {
                fused_attention: false,
                ..crate::InferenceSettings::default()
            },
            crate::InferenceSettings {
                split_embeddings: false,
                ..crate::InferenceSettings::default()
            },
        ];
        for (name, settings) in EXECUTION_SETTINGS.iter().zip(&executing) {
            assert_eq!(
                execution_overrides(settings)
                    .iter()
                    .map(|(named, _)| named.as_str())
                    .collect::<Vec<_>>(),
                [*name]
            );
            assert_ne!(
                dispatch_identity(settings),
                dispatch_identity(&shipped),
                "{name}"
            );
        }
    }

    /// Two plugin libraries whose paths differ only in bytes that are not
    /// UTF-8 key different plans and quarantines: a failure in one does not
    /// send a caller of the other to the CPU.
    #[cfg(unix)]
    #[test]
    fn byte_distinct_library_paths_key_distinct_dispatch() {
        use std::os::unix::ffi::OsStrExt;
        let library = |bytes: &[u8]| crate::InferenceSettings {
            ep_libraries: vec![std::ffi::OsStr::from_bytes(bytes).into()],
            ..crate::InferenceSettings::default()
        };
        assert_ne!(
            dispatch_identity(&library(b"/opt/ep-\x80.so")),
            dispatch_identity(&library(b"/opt/ep-\x81.so"))
        );
    }

    #[test]
    fn unavailable_accelerators_end_with_one_cpu_attempt() {
        let mut attempted = Vec::new();
        let (model, device) = preferred(|device, _| {
            attempted.push(device);
            if device == Device::Cpu {
                Ok(42)
            } else {
                Err(IndexError::Engine("unavailable accelerator".into()))
            }
        })
        .expect("CPU fallback");
        assert_eq!((model, device), (42, Device::Cpu));
        let expected: Vec<_> = accelerators(&crate::InferenceSettings::default())
            .expect("no plugin library is named")
            .into_iter()
            .map(|(device, _)| device)
            .chain([Device::Cpu])
            .collect();
        assert_eq!(attempted, expected);
    }
}
