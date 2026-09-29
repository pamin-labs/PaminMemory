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
//!
//! Whether an idle thread in that pool spins is not a setting: it blocks. See
//! [`session`].

use std::path::{Path, PathBuf};

use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, builder::SessionBuilder};

use crate::error::{IndexError, Result};

/// An ONNX Runtime session over `model`, built the way `fastembed` 6.1 builds
/// one.
///
/// The reranker and BGE-M3 were loaded by `fastembed` until they needed a
/// tokenizer it would not let them share (see `crate::tokenizer`), and what
/// its builder chose decides the scores: the execution providers in order,
/// ONNX Runtime's layout optimizations, and [`threads`] or one per core. So
/// those are what this chooses, in the order it chose them. This also applies
/// DirectML's required execution flags and excludes MatMulAddFusion on CoreML
/// to keep transposed constant weights out of the serialized graph.
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
    let mut builder = options(providers)?;
    let model = model()?;
    let session = commit(&mut builder, &model)?;
    report_model(&session, &model);
    Ok(session)
}

/// Preparation probes use the product options, but are not scoring sessions.
pub(crate) fn probe(model: &Path) -> Result<Session> {
    commit(&mut options(vec![cpu()])?, model)
}

fn commit(builder: &mut SessionBuilder, model: &Path) -> Result<Session> {
    let session = builder
        .commit_from_file(model)
        .map_err(|error| IndexError::Engine(format!("loading {}: {error}", model.display())))?;
    Ok(session)
}

/// Preparation probes do not report themselves as scoring sessions.
fn report_model(session: &Session, model: &Path) {
    tracing::info!(model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"), "loaded ONNX graph");
    match assigned_providers(session) {
        Ok(nodes) => tracing::info!(
            assigned_nodes = %serde_json::to_string(&nodes).expect("serialize provider counts"),
            "ONNX graph execution-provider assignment"
        ),
        Err(error) => tracing::warn!(%error, "could not inspect execution-provider assignment"),
    }
}

/// Counts assigned nodes after basic optimizations, not executed kernel time.
/// CoreML dispatch does not reveal its internal CPU/GPU/ANE placement.
fn assigned_providers(session: &Session) -> Result<std::collections::BTreeMap<String, usize>> {
    use ort::AsPointer;
    let checked = |status| {
        // SAFETY: each status comes directly from ORT; this consumes it once
        // and the error wrapper releases it, including on an early return.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|error| IndexError::Engine(format!("reading provider assignment: {error}")))
    };
    let mut subgraphs = std::ptr::null();
    let mut count = 0;
    // SAFETY: the session and both output variables remain live. Returned
    // arrays and entries are borrowed from this session, never freed here.
    checked(unsafe {
        (ort::api().Session_GetEpGraphAssignmentInfo)(session.ptr(), &mut subgraphs, &mut count)
    })?;
    let mut providers = std::collections::BTreeMap::new();
    if count == 0 {
        return Ok(providers);
    }
    if subgraphs.is_null() {
        return Err(IndexError::Engine(
            "provider assignment returned a null array".into(),
        ));
    }
    // SAFETY: ORT returned `count` entries, valid for the session borrow.
    for &subgraph in unsafe { std::slice::from_raw_parts(subgraphs, count) } {
        if subgraph.is_null() {
            return Err(IndexError::Engine(
                "provider assignment returned a null subgraph".into(),
            ));
        }
        let mut name = std::ptr::null();
        let mut nodes = std::ptr::null();
        let mut node_count = 0;
        // SAFETY: this is a live session-owned subgraph and valid outputs.
        checked(unsafe { (ort::api().EpAssignedSubgraph_GetEpName)(subgraph, &mut name) })?;
        checked(unsafe {
            (ort::api().EpAssignedSubgraph_GetNodes)(subgraph, &mut nodes, &mut node_count)
        })?;
        if name.is_null() {
            return Err(IndexError::Engine(
                "provider assignment returned a null name".into(),
            ));
        }
        // SAFETY: ORT returns a session-owned NUL-terminated provider name.
        let name = unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        *providers.entry(name).or_default() += node_count;
    }
    Ok(providers)
}

/// What [`session`] asks of ONNX Runtime before it has a model to load.
///
/// One thing here is not `fastembed`'s choice, and does not change a score:
/// intra-op threads block when they run out of work instead of spinning.
/// ONNX Runtime lets them spin by default, which saves a wake-up when the next
/// piece of work comes at once and holds a core for as long as it does not --
/// and the embedder and the reranker each own a pool of one thread per core,
/// so on a machine with anything else to do the spinning threads take cores
/// from the threads that have work, the other model's included.
///
/// Measured through `pamin serve` and `pamin search` at the defaults, one
/// hundred queries (sixty from the own corpus, forty from an XQuAD-R subset),
/// a fresh server per arm so no query is a cache hit, three rounds in rotated
/// order, on four cores shared with other work: with two busy loops beside
/// the server a search took 0.754 of what it did spinning (geometric mean of
/// per-query ratios; faster on 86 of 100, sign-flip `p = 0.0001`), and at the
/// machine's own load -- 2.5 to 13 -- 1.033 (`p = 0.068`), eight concurrent
/// callers getting 1.17 times the throughput. Rankings, query and passage
/// embeddings and `accurate` scores were bit-identical. The rule, written
/// before the run, is in `docs/adr/0001-tech-selection.md`.
fn options(providers: Vec<ExecutionProviderDispatch>) -> Result<SessionBuilder> {
    let unready =
        |error: &dyn std::fmt::Display| IndexError::Engine(format!("preparing a session: {error}"));
    let threads = match threads() {
        Some(threads) => threads,
        None => std::thread::available_parallelism()?.get(),
    };
    let builder = Session::builder().map_err(|error| unready(&error))?;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let builder = if providers
        .iter()
        .any(|provider| provider.downcast_ref::<ort::ep::CoreML>().is_some())
    {
        // Gemm fusion transposes large constant weights into inline MIL
        // tensors. Keep the original initializer blobs for CoreML instead.
        builder
            .with_disabled_optimizers("MatMulAddFusion")
            .map_err(|error| unready(&error))?
    } else {
        builder
    };
    #[cfg(target_os = "windows")]
    let builder = if providers
        .iter()
        .any(|provider| provider.downcast_ref::<ort::ep::DirectML>().is_some())
    {
        // DirectML requires these before registration. Registration alone
        // does not validate them; model loading otherwise falls back to CPU.
        builder
            .with_memory_pattern(false)
            .map_err(|error| unready(&error))?
            .with_parallel_execution(false)
            .map_err(|error| unready(&error))?
    } else {
        builder
    };
    builder
        .with_execution_providers(providers)
        .map_err(|error| unready(&error))?
        .with_config_entry("session.record_ep_graph_assignment_info", "1")
        .map_err(|error| unready(&error))?
        .with_optimization_level(crate::prepared::LEVEL)
        .map_err(|error| unready(&error))?
        .with_intra_threads(threads)
        .map_err(|error| unready(&error))?
        .with_intra_op_spinning(false)
        .map_err(|error| unready(&error))
}

/// Static shapes prevent CoreML from silently rejecting unbounded ANE regions.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn fixed_coreml(
    model: impl FnOnce() -> Result<PathBuf>,
    rows: usize,
    tokens: usize,
) -> Result<(Session, PathBuf)> {
    let path = model()?;
    let cache = coreml_cache(&path, rows, tokens)?;
    std::fs::create_dir_all(cache.parent().expect("cache has a parent"))?;
    // Keep the lock outside the directory being rebuilt. Unlinking a held
    // lock would let another process lock a different inode concurrently.
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cache.with_extension("lock"))?;
    lock.lock()?;
    let reused = cache.join("ready").is_file();
    if !reused {
        reset_coreml_cache(&cache)?;
    }
    let load = || -> Result<Session> {
        let provider = ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
            .with_model_cache_dir(cache.to_string_lossy())
            .build()
            .error_on_failure();
        let error =
            |e: &dyn std::fmt::Display| IndexError::Engine(format!("static CoreML session: {e}"));
        let mut builder = options(vec![provider])?
            .with_dimension_override("batch_size", rows as i64)
            .map_err(|e| error(&e))?
            .with_dimension_override("sequence_length", tokens as i64)
            .map_err(|e| error(&e))?;
        builder.commit_from_file(&path).map_err(|e| error(&e))
    };
    let session = match load() {
        Ok(session) => session,
        Err(error) if reused => {
            tracing::warn!(%error, "cached CoreML package failed to load; rebuilding once");
            reset_coreml_cache(&cache)?;
            load()?
        }
        Err(error) => return Err(error),
    };
    // A crash before every partition loads leaves no marker. Its files must
    // be rebuilt under the same lock rather than mistaken for a valid cache.
    std::fs::File::create(cache.join("ready"))?.sync_all()?;
    report_model(&session, &path);
    Ok((session, path))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn reset_coreml_cache(cache: &Path) -> Result<()> {
    if cache.exists() {
        std::fs::remove_dir_all(cache)?;
    }
    std::fs::create_dir_all(cache)?;
    Ok(())
}

/// The prepared parent already identifies the external weights by content.
/// Also key the graph, runtime and overrides: ORT's URL key omits all three.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn coreml_cache(model: &std::path::Path, rows: usize, tokens: usize) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let parent = model.parent().ok_or_else(|| {
        IndexError::Engine("prepared CoreML model has no parent directory".into())
    })?;
    let graph = crate::prepared::source_digest(model)?;
    let runtime = format!("{:x}", Sha256::digest(ort::info().as_bytes()));
    // Bump this policy version when the compile options above change.
    Ok(parent
        .join("coreml-all-v2")
        .join(runtime)
        .join(format!("{graph}-{rows}-{tokens}")))
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
    pamin_core::env::positive("PAMIN_INFERENCE_THREADS")
}

/// The execution provider selected for a model's forward passes.
///
/// Recorded on every loaded model rather than inferred, because the answer is
/// not what the platform says: every x86-64 Linux build can use CUDA, and one
/// on a machine with no GPU, or with the wrong CUDA, runs on the CPU -- and a
/// score cannot be read without knowing which. The CPU runs the int8 export and a GPU the fp16 one, so the two do
/// not produce bit-identical scores and are not interchangeable in a
/// measurement. A registered accelerator may still delegate unsupported
/// nodes to CPU; CoreML itself may use CPU, GPU or the Neural Engine. Inspect
/// the runtime's node profile before calling this a GPU measurement.
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

/// One provider policy for every model. The loader owns model/export choice;
/// this owns accelerator order, reporting and the final CPU attempt.
pub(crate) fn preferred<T>(
    mut load: impl FnMut(Device, Vec<ExecutionProviderDispatch>) -> Result<T>,
) -> Result<(T, Device)> {
    for (device, provider) in accelerators() {
        match load(device, vec![provider]) {
            Ok(model) => return Ok((model, device)),
            Err(error) => tracing::warn!(
                device = device.name(),
                %error,
                "model could not use this accelerator; trying the next provider"
            ),
        }
    }
    load(Device::Cpu, vec![cpu()]).map(|model| (model, Device::Cpu))
}

/// The accelerators to try before the CPU, best first.
///
/// Whatever this platform's runtime carries, with no build flag: CUDA on
/// x86-64 Linux, Core ML on Apple silicon, DirectML on Windows, nothing
/// elsewhere. A machine without the device -- or, for CUDA, without the
/// driver, CUDA 13 and cuDNN 9 -- fails to register it and runs on the CPU,
/// so the available provider is tried before the CPU. Successful registration
/// does not establish how much of the graph that provider accelerates.
/// `PAMIN_DEVICE=cpu` empties this, for a measurement that must be comparable
/// with a CPU one or a machine whose GPU belongs to something else.
///
/// Each is set to fail loudly on registration rather than fall through to the
/// CPU inside ONNX Runtime, which is its default. A silent fallback would load
/// the GPU's fp16 export onto the CPU -- slower than the int8 one it was
/// chosen over -- and report nothing; failing here lets the caller load the
/// CPU's own export instead and record that it did.
pub(crate) fn accelerators() -> Vec<(Device, ExecutionProviderDispatch)> {
    if pamin_core::env::is("PAMIN_DEVICE", "cpu") {
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

    #[test]
    fn preparation_probes_do_not_claim_to_be_product_model_loads() {
        // Tracing callsite interest is process-global. Other parallel session
        // tests use these same callsites without this thread-local collector.
        // Isolate the capture, while retaining real model loads and assertions.
        const CHILD: &str = "MODEL_LOAD_CAPTURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "inference::tests::preparation_probes_do_not_claim_to_be_product_model_loads",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(
                status.success(),
                "isolated model provenance regression failed"
            );
            return;
        }
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let capture = Capture(Arc::default());
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("attention.onnx.partial");
        std::fs::write(
            &model,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )
        .unwrap();
        tracing::subscriber::with_default(subscriber, || {
            let _probe = probe(&model).unwrap();
            assert!(
                capture.0.lock().unwrap().is_empty(),
                "probe logged a product load"
            );
            let _product = session(vec![cpu()], || Ok(model.clone())).unwrap();
        });
        let bytes = capture.0.lock().unwrap();
        let log = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(log.matches("loaded ONNX graph").count(), 1);
        assert!(log.contains("attention.onnx.partial"));
    }

    #[test]
    fn sessions_record_actual_provider_assignment() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        std::fs::write(
            &path,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )
        .unwrap();
        let session = session(vec![cpu()], || Ok(path.clone())).unwrap();
        let assigned = assigned_providers(&session).unwrap();
        assert!(
            assigned
                .get("CPUExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        assert_eq!(
            assigned.len(),
            1,
            "a CPU-only session reported another provider"
        );
        let unrecorded = Session::builder()
            .unwrap()
            .with_execution_providers(vec![cpu()])
            .unwrap()
            .commit_from_file(path)
            .unwrap();
        assert!(
            assigned_providers(&unrecorded).is_err(),
            "assignment inspection worked without enabling its collection premise"
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
        let expected: Vec<_> = accelerators()
            .into_iter()
            .map(|(device, _)| device)
            .chain([Device::Cpu])
            .collect();
        assert_eq!(attempted, expected);
    }

    /// Every session's intra-op threads block rather than spin; see
    /// [`options`]. Read back from ONNX Runtime rather than from our own
    /// call, because an entry that never reached the options is exactly the
    /// failure this guards: the runtime's default is to spin.
    #[test]
    fn idle_inference_threads_block_rather_than_spin() {
        use ort::AsPointer;

        let builder = options(vec![cpu()]).expect("session options");
        let mut value = [0 as std::ffi::c_char; 8];
        let mut size = value.len();
        // SAFETY: the options pointer is live for the borrow of `builder`, the
        // key is NUL-terminated, and `size` is the length of `value`, which
        // the runtime writes no further than.
        let status = unsafe {
            (ort::api().GetSessionConfigEntry)(
                builder.ptr(),
                c"session.intra_op.allow_spinning".as_ptr(),
                value.as_mut_ptr(),
                &mut size,
            )
        };
        assert!(
            status.0.is_null(),
            "no spinning entry reached the options, so ONNX Runtime spins"
        );
        // SAFETY: on success the runtime wrote a NUL-terminated string of
        // `size` bytes, NUL included, into `value`.
        let set = unsafe { std::ffi::CStr::from_ptr(value.as_ptr()) };
        assert_eq!(set, c"0", "intra-op spinning is not off");
    }

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
            let registers = options(vec![provider.clone()]).is_ok();
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn coreml_cache_isolates_graph_content_and_static_shapes() {
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("model.onnx");
        std::fs::write(&model, b"first graph").unwrap();
        let first = coreml_cache(&model, 4, 64).unwrap();
        assert_eq!(first, coreml_cache(&model, 4, 64).unwrap());
        assert_ne!(first, coreml_cache(&model, 4, 128).unwrap());
        assert_ne!(first, coreml_cache(&model, 2, 64).unwrap());
        std::fs::write(&model, b"updated graph").unwrap();
        assert_ne!(first, coreml_cache(&model, 4, 64).unwrap());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles two tiny native CoreML models"]
    fn coreml_compiled_cache_preserves_shapes_and_updated_weights()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let model = temporary.path().join("model.onnx");
        // Same dynamic MatMul/Add/Relu graph; diagonal weights are 2 or 3.
        // Loading 64, then 128, then 64 also checks compiled cache reuse.
        let models: [(&[u8], f32); 2] = [
            (include_bytes!("../tests/fixtures/coreml-cache-2.onnx"), 2.0),
            (include_bytes!("../tests/fixtures/coreml-cache-3.onnx"), 3.0),
        ];
        for (bytes, scale) in models {
            std::fs::write(&model, bytes)?;
            for tokens in [64, 128, 64] {
                let (mut session, _) = fixed_coreml(|| Ok(model.clone()), 4, tokens)?;
                let input = ort::value::Tensor::from_array((
                    [4, tokens, 8],
                    vec![0.25f32; 4 * tokens * 8],
                ))?;
                let output = session.run(ort::inputs!["x" => input])?;
                let (shape, values) = output["y"].try_extract_tensor::<f32>()?;
                assert_eq!(shape.as_ref(), [4, tokens as i64, 8]);
                assert!(values.iter().all(|value| *value == scale * 0.25 + 0.125));
                let cache = coreml_cache(&model, 4, tokens)?;
                assert!(
                    std::fs::read_dir(cache)?.count() > 1,
                    "CoreML wrote no cache"
                );
            }
        }
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles a tiny native CoreML model and corrupts only its temporary cache"]
    fn coreml_rebuilds_interrupted_and_damaged_published_caches()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let model = temporary.path().join("model.onnx");
        std::fs::write(
            &model,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )?;
        for published in [false, true] {
            let (session, _) = fixed_coreml(|| Ok(model.clone()), 4, 64)?;
            drop(session);
            let cache = coreml_cache(&model, 4, 64)?;
            assert!(cache.join("ready").is_file());
            let mut damaged = 0;
            for directory in std::fs::read_dir(&cache)? {
                let directory = directory?.path();
                if !directory.is_dir() {
                    continue;
                }
                for partition in std::fs::read_dir(directory)? {
                    let package = partition?.path().join("model");
                    if package.join("Manifest.json").is_file() {
                        std::fs::remove_file(package.join("Manifest.json"))?;
                        std::fs::remove_dir_all(package.join("compiled_model.mlmodelc"))?;
                        damaged += 1;
                    }
                }
            }
            assert!(damaged > 0, "did not reproduce the missing manifest");
            if !published {
                std::fs::remove_file(cache.join("ready"))?;
            }
            let (mut session, _) = fixed_coreml(|| Ok(model.clone()), 4, 64)?;
            let output = session.run(ort::inputs!["x" => ort::value::Tensor::from_array((
                [4, 64, 8], vec![0.25f32; 4 * 64 * 8],
            ))?])?;
            let (_, values) = output["y"].try_extract_tensor::<f32>()?;
            assert!(values.iter().all(|value| *value == 0.625));
            assert!(cache.join("ready").is_file());
        }
        Ok(())
    }

    /// MatMul/Add fusion transposes a large FFN weight into an inline MIL
    /// constant. Check the serialized native graph, not our option value.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles a small native CoreML model"]
    fn coreml_keeps_matrix_weights_out_of_the_graph()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::onnx::*;
        use ort::ep::coreml::{ComputeUnits, ModelFormat};

        fn tensor(name: &str, dimensions: &[u64], values: &[f32]) -> Vec<u8> {
            let mut tensor = Vec::new();
            for &dimension in dimensions {
                put_varint_field(&mut tensor, TENSOR_DIMS, dimension);
            }
            put_varint_field(&mut tensor, TENSOR_DATA_TYPE, 1);
            put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            put_bytes(&mut tensor, TENSOR_RAW_DATA, &bytes);
            tensor
        }
        fn info(name: &str, dimensions: &[u64]) -> Vec<u8> {
            let mut shape = Vec::new();
            for &dimension in dimensions {
                let mut value = Vec::new();
                put_varint_field(&mut value, 1, dimension);
                put_bytes(&mut shape, 1, &value);
            }
            let mut tensor = Vec::new();
            put_varint_field(&mut tensor, TENSOR_TYPE_ELEM_TYPE, 1);
            put_bytes(&mut tensor, 2, &shape);
            let mut ty = Vec::new();
            put_bytes(&mut ty, TYPE_TENSOR, &tensor);
            let mut info = Vec::new();
            put_bytes(&mut info, VALUE_INFO_NAME, name.as_bytes());
            put_bytes(&mut info, VALUE_INFO_TYPE, &ty);
            info
        }
        fn node(op: &str, inputs: &[&str], output: &str) -> Vec<u8> {
            let mut node = Vec::new();
            for input in inputs {
                put_bytes(&mut node, NODE_INPUT, input.as_bytes());
            }
            put_bytes(&mut node, NODE_OUTPUT, output.as_bytes());
            put_bytes(&mut node, NODE_NAME, op.as_bytes());
            put_bytes(&mut node, NODE_OP_TYPE, op.as_bytes());
            node
        }
        let mut graph = Vec::new();
        put_bytes(&mut graph, 2, b"FFN matrix and bias");
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("MatMul", &["x", "weight"], "product"),
        );
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("Add", &["product", "bias"], "y"),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("weight", &[1024, 4096], &vec![1.0 / 4096.0; 1024 * 4096]),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("bias", &[4096], &vec![0.125; 4096]),
        );
        put_bytes(&mut graph, GRAPH_INPUT, &info("x", &[1, 32, 1024]));
        put_bytes(
            &mut graph,
            GRAPH_VALUE_INFO,
            &info("product", &[1, 32, 4096]),
        );
        put_bytes(&mut graph, GRAPH_OUTPUT, &info("y", &[1, 32, 4096]));
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 8);
        let mut opset = Vec::new();
        put_varint_field(&mut opset, 2, 17);
        put_bytes(&mut model, MODEL_OPSET_IMPORT, &opset);
        put_bytes(&mut model, MODEL_GRAPH, &graph);

        let cache = tempfile::tempdir()?;
        let provider = ort::ep::CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_compute_units(ComputeUnits::CPUAndGPU)
            .with_model_cache_dir(cache.path().display())
            .build()
            .error_on_failure();
        let mut session = options(vec![provider])?.commit_from_memory(&model)?;
        let output = session.run(ort::inputs!["x" => ort::value::Tensor::from_array(([1,32,1024],vec![0.25f32;32*1024]))?])?;
        let (_, values) = output["y"].try_extract_tensor::<f32>()?;
        assert_eq!(values.len(), 32 * 4096);
        assert!(
            values.iter().all(|v| (*v - 0.1875).abs() < 1e-5),
            "matrix or bias result changed"
        );

        let mut directories = vec![cache.path().to_path_buf()];
        let mut graphs = Vec::new();
        let mut native_matmul = false;
        while let Some(directory) = directories.pop() {
            for entry in std::fs::read_dir(directory)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    directories.push(entry.path());
                } else if entry.file_name() == "model.mlmodel" {
                    let bytes = std::fs::read(entry.path())?;
                    native_matmul |= bytes.windows(6).any(|word| word == b"matmul");
                    graphs.push(bytes.len() as u64);
                }
            }
        }
        println!("native CoreML graph sizes: {graphs:?}");
        assert!(
            graphs.iter().any(|&bytes| bytes > 0),
            "no native CoreML graph was produced; the premise fell back to CPU"
        );
        assert!(
            native_matmul,
            "the matrix operation was not lowered into the native CoreML graph"
        );
        assert!(
            graphs.iter().all(|&bytes| bytes < 1024 * 1024),
            "matrix weights were serialized inline instead of into the weights file: {graphs:?}"
        );
        Ok(())
    }
}
