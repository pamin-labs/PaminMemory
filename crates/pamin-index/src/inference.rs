//! How much of the machine one forward pass is allowed to use.
//!
//! Every model in this crate runs on ONNX Runtime, some through `fastembed`
//! and the rest through [`session`], and neither set this at first, so all
//! inherited `fastembed`'s default: intra-op threads equal to
//! `available_parallelism()`. One pass therefore takes every core.
//!
//! That is the right default for a single query and the wrong one for a server.
//! Measured on four cores with nothing shared -- one model per worker, no lock
//! of ours anywhere -- throughput *falls* as callers are added, 182 embeddings
//! a second at one worker to 54 at eight, because the second caller finds the
//! first caller's threads rather than an idle core. Splitting the cores between
//! callers instead of between the layers of one pass is the other way to divide
//! them, and which one wins is a property of the machine rather than of this
//! code.
//!
//! So it is a setting, and its default is what the library already did.

use std::path::PathBuf;

use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;

use crate::error::{IndexError, Result};

/// An ONNX Runtime session over `model`, built the way `fastembed` 6.1 builds
/// one.
///
/// The rerankers and BGE-M3 were loaded by `fastembed` until they needed a
/// tokenizer it would not let them share (see `crate::tokenizer`), and what
/// its builder chose decides the scores: the execution providers in order,
/// ONNX Runtime's layout optimizations, and [`threads`] or one per core. So
/// those are what this chooses, in the order it chose them. It asked for
/// nothing else on these platforms -- its DirectML adjustments are behind a
/// feature of its own that this build does not enable.
///
/// `model` is asked for only once the providers have registered, because
/// asking for it can mean downloading it, and each device has an export of
/// its own. Found the other way round, a CPU-only Linux machine fetched the
/// `accurate` reranker's 1,136 MB half-precision export before learning that
/// CUDA would not register -- which it does in about a millisecond -- and
/// then kept the file, never loading it, beside the int8 export it runs.
pub(crate) fn session(
    providers: Vec<ExecutionProviderDispatch>,
    model: impl FnOnce() -> Result<PathBuf>,
) -> Result<Session> {
    session_on(providers, intra_threads()?, model)
}

/// [`session`] on `threads` intra-op threads rather than [`intra_threads`].
pub(crate) fn session_on(
    providers: Vec<ExecutionProviderDispatch>,
    threads: usize,
    model: impl FnOnce() -> Result<PathBuf>,
) -> Result<Session> {
    let unready =
        |error: &dyn std::fmt::Display| IndexError::Engine(format!("preparing a session: {error}"));
    let mut builder = Session::builder()
        .map_err(|error| unready(&error))?
        .with_execution_providers(providers)
        .map_err(|error| unready(&error))?
        .with_optimization_level(crate::prepared::LEVEL)
        .map_err(|error| unready(&error))?
        .with_intra_threads(threads)
        .map_err(|error| unready(&error))?;
    let model = model()?;
    builder
        .commit_from_file(&model)
        .map_err(|error| IndexError::Engine(format!("loading {}: {error}", model.display())))
}

/// How many cores a forward pass may use: [`threads`], or one per core.
pub(crate) fn intra_threads() -> Result<usize> {
    Ok(match threads() {
        Some(threads) => threads,
        None => std::thread::available_parallelism()?.get(),
    })
}

/// Intra-op threads per inference session, or `None` for one per core.
///
/// Read from `PAMIN_INFERENCE_THREADS`. Unset -- the default -- means
/// `available_parallelism()`, which is what this crate did before the setting
/// existed, so an unconfigured workspace behaves exactly as it used to.
///
/// A value that is not a positive number is ignored rather than refused: this
/// is a performance knob, and a typo in it should not stop a search from
/// working.
pub(crate) fn threads() -> Option<usize> {
    std::env::var("PAMIN_INFERENCE_THREADS")
        .ok()?
        .parse::<usize>()
        .ok()
        .filter(|threads| *threads > 0)
}

/// Where a model's forward passes run.
///
/// Recorded on every loaded model rather than inferred, because the answer is
/// not what the platform says: every x86-64 Linux build can use CUDA, and one
/// on a machine with no GPU, or with the wrong CUDA, runs on the CPU -- and a
/// score cannot be read without knowing which. The CPU runs the int8 export and a GPU the fp16 one, so the two do
/// not produce bit-identical scores and are not interchangeable in a
/// measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda,
    CoreMl,
    DirectMl,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
        }
    }
}

/// The CPU, with ONNX Runtime's memory arena off.
///
/// The arena keeps every buffer a pass ever needed and hands it back to the
/// next pass instead of to the system, so a model's resident size settles at
/// its largest batch and stays there. Measured on the `accurate` reranker, six
/// passes of sixteen passages at 256 tokens: 1,682 MB resident with the arena
/// against 1,077 MB without -- 605 MB of activations held for a batch shape
/// that recurs only when the next search runs -- with scores bit-identical and
/// no slower pass. Through the whole search path on MIRACL (the `MEMORY` arm),
/// a hundred `accurate` searches hold 5,914 MB anonymous against 6,208 MB --
/// less than the six fixed passes, because a real search scores fewer and
/// shorter pairs. The embedder saves nothing measurable there, since a query
/// is one short text; it takes the same setting because its largest batch is
/// a bulk write's, which that arm does not exercise.
///
/// The arena holds activations. The weights are the other half, and loaded
/// from the file the hub serves they are copied onto the heap: +664 MB
/// anonymous for the `accurate` reranker's 570 MB export in a bare session.
/// So a CPU session loads a prepared copy whose weights ONNX Runtime maps
/// from disk instead, +11 MB in the same session -- see `crate::prepared`.
pub(crate) fn cpu() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().with_arena_allocator(false).build()
}

/// The accelerators to try before the CPU, best first.
///
/// Whatever this platform's runtime carries, with no build flag: CUDA on
/// x86-64 Linux, Core ML on Apple silicon, DirectML on Windows, nothing
/// elsewhere. A machine without the device -- or, for CUDA, without the
/// driver, CUDA 13 and cuDNN 9 -- fails to register it and runs on the CPU,
/// so a GPU is used when there is one and costs nothing when there is not.
/// `PAMIN_DEVICE=cpu` empties this, for a measurement that must be comparable
/// with a CPU one or a machine whose GPU belongs to something else.
///
/// Each is set to fail loudly on registration rather than fall through to the
/// CPU inside ONNX Runtime, which is its default. A silent fallback would load
/// the GPU's fp16 export onto the CPU -- slower than the int8 one it was
/// chosen over -- and report nothing; failing here lets the caller load the
/// CPU's own export instead and record that it did.
pub(crate) fn accelerators() -> Vec<(Device, ExecutionProviderDispatch)> {
    if std::env::var("PAMIN_DEVICE").is_ok_and(|device| device.eq_ignore_ascii_case("cpu")) {
        return Vec::new();
    }
    vec![
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        (
            Device::Cuda,
            ort::ep::CUDA::default().build().error_on_failure(),
        ),
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        (
            Device::CoreMl,
            ort::ep::CoreML::default().build().error_on_failure(),
        ),
        #[cfg(target_os = "windows")]
        (
            Device::DirectMl,
            ort::ep::DirectML::default().build().error_on_failure(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device that will not register is found before its model is asked for,
    /// since asking can mean downloading an export this machine cannot run.
    ///
    /// Whether an accelerator registers is a property of the machine, so each
    /// is first registered on a builder of its own; only one that fails there
    /// says anything about the order. On a machine with no GPU -- every one
    /// this suite has run on -- that is each of them.
    #[test]
    fn a_device_that_will_not_register_is_never_asked_for_its_model() {
        for (device, provider) in accelerators() {
            let registers = Session::builder()
                .expect("a session builder")
                .with_execution_providers([provider.clone()])
                .is_ok();
            if registers {
                continue;
            }
            let asked = std::cell::Cell::new(false);
            let loaded = session(vec![provider], || {
                asked.set(true);
                Err(IndexError::Engine("no model here".into()))
            });
            assert!(
                loaded.is_err(),
                "{} registered the second time",
                device.name()
            );
            assert!(
                !asked.get(),
                "{}'s model was asked for before the device refused to register",
                device.name()
            );
        }
    }
}
