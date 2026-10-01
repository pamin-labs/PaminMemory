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

/// Legacy providers or an explicitly discovered plugin device. Model code
/// owns exports; this module owns physical device selection and fallback.
#[derive(Clone)]
pub(crate) enum Target {
    Providers(Vec<ExecutionProviderDispatch>),
    Npu { provider: String, id: u32 },
}

impl From<Vec<ExecutionProviderDispatch>> for Target {
    fn from(providers: Vec<ExecutionProviderDispatch>) -> Self {
        Self::Providers(providers)
    }
}

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
    target: impl Into<Target>,
    model: impl FnOnce() -> Result<PathBuf>,
) -> Result<Session> {
    let target = target.into();
    let expected = match &target {
        Target::Npu { provider, .. } => Some(provider.clone()),
        _ => None,
    };
    let mut builder = options(target)?;
    let model = model()?;
    let session = commit(&mut builder, &model)?;
    if let Some(provider) = expected {
        let assigned = assigned_providers(&session)?;
        if assigned.get(&provider).copied().unwrap_or(0) == 0 {
            return Err(IndexError::Engine(format!(
                "NPU {provider} was assigned no model nodes"
            )));
        }
    }
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
            model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"),
            assigned_nodes = %serde_json::to_string(&nodes).expect("serialize provider counts"),
            "ONNX graph execution-provider assignment"
        ),
        Err(error) => {
            tracing::warn!(model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"), %error, "could not inspect execution-provider assignment")
        }
    }
}

/// Counts assigned nodes after basic optimizations, not executed kernel time.
/// CoreML dispatch does not reveal its internal CPU/GPU/ANE placement.
pub(crate) fn assigned_providers(
    session: &Session,
) -> Result<std::collections::BTreeMap<String, usize>> {
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
pub(crate) fn options(target: impl Into<Target>) -> Result<SessionBuilder> {
    let target = target.into();
    let providers = match &target {
        Target::Providers(providers) => providers.clone(),
        Target::Npu { .. } => gpu_providers()
            .into_iter()
            .map(|(_, ep)| ep.fail_silently())
            .chain([cpu()])
            .collect(),
    };
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
    let builder = match target {
        Target::Providers(_) => builder,
        Target::Npu { provider, id } => {
            let environment = ort::environment::Environment::current().map_err(|e| unready(&e))?;
            let devices: Vec<_> = environment
                .devices()
                .filter(|device| {
                    device.hardware_device().ty() == ort::memory::DeviceType::NPU
                        && device.hardware_device().id() == id
                        && device.ep().is_ok_and(|name| name == provider)
                })
                .collect();
            if devices.is_empty() {
                return Err(IndexError::Engine(format!(
                    "NPU {provider}/{id} unavailable"
                )));
            }
            builder
                .with_devices(devices, None)
                .map_err(|e| unready(&e))?
        }
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

/// Refuse an unavailable CoreML provider before fetching its export. The
/// cache-aware session registers again once the content-derived path is known.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn coreml_source(model: impl FnOnce() -> Result<PathBuf>) -> Result<PathBuf> {
    drop(options(vec![coreml().build().error_on_failure()])?);
    model()
}

/// Static shapes prevent CoreML from silently rejecting unbounded ANE regions.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn fixed_coreml(
    model: impl FnOnce() -> Result<PathBuf>,
    rows: usize,
    tokens: usize,
) -> Result<(Session, PathBuf)> {
    let path = coreml_source(model)?;
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
    recover_coreml_cache(&cache)?;
    let reused = cache.join("ready").is_file();
    if !reused {
        reset_coreml_cache(&cache)?;
    }
    let load = || -> Result<Session> {
        let provider = coreml()
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
            rebuild_coreml_cache(&cache, load)?
        }
        Err(error) => return Err(error),
    };
    // A crash before every partition loads leaves no marker. Its files must
    // be rebuilt under the same lock rather than mistaken for a valid cache.
    publish_coreml_cache(&cache)?;
    report_model(&session, &path);
    Ok((session, path))
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn recover_coreml_cache(cache: &Path) -> Result<()> {
    let previous = cache.with_extension("previous");
    if previous.exists() {
        if cache.join("ready").is_file() {
            std::fs::remove_dir_all(previous)?;
        } else {
            if cache.exists() {
                std::fs::remove_dir_all(cache)?;
            }
            std::fs::rename(previous, cache)?;
        }
    }
    Ok(())
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn publish_coreml_cache(cache: &Path) -> Result<()> {
    let partial = cache.join("ready.partial");
    std::fs::File::create(&partial)?.sync_all()?;
    std::fs::rename(partial, cache.join("ready"))?;
    Ok(())
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn rebuild_coreml_cache<T>(cache: &Path, load: impl FnOnce() -> Result<T>) -> Result<T> {
    let previous = cache.with_extension("previous");
    std::fs::rename(cache, &previous)?;
    std::fs::create_dir_all(cache)?;
    match load().and_then(|model| {
        publish_coreml_cache(cache)?;
        Ok(model)
    }) {
        Ok(model) => {
            // Publication succeeded. Failed retirement can be retried on the
            // next load; it must not discard the valid replacement.
            if let Err(error) = std::fs::remove_dir_all(previous) {
                tracing::warn!(%error, "could not retire previous CoreML cache");
            }
            Ok(model)
        }
        Err(error) => {
            recover_coreml_cache(cache)?;
            Err(error)
        }
    }
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
    Npu,
}

impl Device {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
            Self::Npu => "npu",
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

/// Legacy discovery-order helper retained only for policy regression tests.
#[cfg(test)]
pub(crate) fn preferred<T>(
    mut load: impl FnMut(Device, Target) -> Result<T>,
) -> Result<(T, Device)> {
    for (device, provider) in accelerators() {
        match load(device, provider) {
            Ok(model) => return Ok((model, device)),
            Err(error) => tracing::warn!(
                device = device.name(),
                %error,
                "model could not use this accelerator; trying the next provider"
            ),
        }
    }
    load(Device::Cpu, vec![cpu()].into()).map(|model| (model, Device::Cpu))
}

/// Small CPU output references reused for per-session validation. These are
/// proof fixtures, not a resident second model or a query-result cache.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct References {
    pub(crate) vectors: Option<Vec<Vec<f32>>>,
    pub(crate) queries: Option<Vec<Vec<f32>>>,
    pub(crate) scores: Option<Vec<f32>>,
}

#[derive(Clone)]
struct CachedPlan {
    device: Device,
    target: Target,
    revalidate: Option<std::time::Instant>,
    references: References,
}

impl CachedPlan {
    fn fresh(device: Device, target: Target, numerical: bool) -> Self {
        Self {
            device,
            target,
            references: References::default(),
            revalidate: Some(
                std::time::Instant::now()
                    + std::time::Duration::from_secs(if numerical { 300 } else { 86400 }),
            ),
        }
    }
    fn with_references(mut self, references: References) -> Self {
        self.references = references;
        self
    }
}

/// Calibrate complete model-call fixtures, then reuse the validated target
/// through idle reloads. This is a bounded workload choice, not a claim about
/// every query shape or an accelerator's internal hardware placement.
#[derive(Clone, Default)]
pub(crate) struct RuntimePlan {
    namespace: String,
    target: String,
    retry_at: Option<std::time::Instant>,
    restore_target: Option<String>,
}

pub(crate) trait RuntimeModel {
    fn runtime_plan(&self) -> &RuntimePlan;
    fn runtime_plan_mut(&mut self) -> &mut RuntimePlan;
}

impl<T: RuntimeModel> RuntimeModel for Box<T> {
    fn runtime_plan(&self) -> &RuntimePlan {
        (**self).runtime_plan()
    }
    fn runtime_plan_mut(&mut self) -> &mut RuntimePlan {
        (**self).runtime_plan_mut()
    }
}

type RuntimeFailures = std::collections::HashMap<(PathBuf, String, String), std::time::Instant>;
fn runtime_failures() -> &'static std::sync::Mutex<RuntimeFailures> {
    static FAILED: std::sync::OnceLock<std::sync::Mutex<RuntimeFailures>> =
        std::sync::OnceLock::new();
    FAILED.get_or_init(Default::default)
}

const RUNTIME_RETRY: std::time::Duration = std::time::Duration::from_secs(300);

fn quarantine_runtime(cache_dir: &Path, plan: &RuntimePlan, now: std::time::Instant) {
    let root = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| cache_dir.to_path_buf());
    let mut failures = runtime_failures()
        .lock()
        .expect("runtime quarantine poisoned");
    // Bound metadata even if many distinct failed exports are loaded.
    if failures.len() >= MAX_DISK_PLANS {
        let oldest = failures
            .iter()
            .min_by_key(|(_, until)| **until)
            .map(|(key, _)| key.clone());
        if let Some(oldest) = oldest {
            failures.remove(&oldest);
        }
    }
    failures.insert(
        (root, plan.namespace.clone(), plan.target.clone()),
        now + RUNTIME_RETRY,
    );
}

#[cfg(test)]
fn runtime_available(cache_dir: &Path, namespace: &str, target: &str) -> bool {
    let root = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| cache_dir.to_path_buf());
    let failed = runtime_failures()
        .lock()
        .expect("runtime quarantine poisoned");
    failed
        .get(&(root, namespace.into(), target.into()))
        .is_none_or(|until| *until <= std::time::Instant::now())
}

struct RuntimeSnapshot {
    blocked: Option<(std::time::Instant, String)>,
    unavailable: std::collections::HashSet<String>,
}

fn runtime_snapshot(
    cache_dir: &Path,
    namespace: &str,
    discovered: &[String],
    now: std::time::Instant,
) -> RuntimeSnapshot {
    let root = cache_dir
        .canonicalize()
        .unwrap_or_else(|_| cache_dir.to_path_buf());
    let failures = runtime_failures()
        .lock()
        .expect("runtime quarantine poisoned");
    let mut unavailable = std::collections::HashSet::new();
    let mut blocked = None;
    for ((path, scope, target), until) in failures.iter() {
        if path != &root || scope != namespace {
            continue;
        }
        if *until > now {
            unavailable.insert(target.clone());
        }
        let deadline = if *until > now {
            *until
        } else if !discovered.contains(target) {
            now + RUNTIME_RETRY
        } else {
            continue;
        };
        if blocked
            .as_ref()
            .is_none_or(|(earliest, _)| deadline < *earliest)
        {
            blocked = Some((deadline, target.clone()));
        }
    }
    RuntimeSnapshot {
        blocked,
        unavailable,
    }
}

#[cfg(test)]
fn runtime_retry(
    cache_dir: &Path,
    namespace: &str,
    discovered: &[String],
    now: std::time::Instant,
) -> Option<(std::time::Instant, String)> {
    runtime_snapshot(cache_dir, namespace, discovered, now).blocked
}

/// Expired fallbacks must reach a real model call even when the request is
/// cached. Loading/replacement is deferred to retry_model, which publishes
/// a replacement only after the caller's complete operation succeeds.
pub(crate) fn needs_revalidation<T: RuntimeModel>(model: &T) -> bool {
    needs_revalidation_at(model, std::time::Instant::now())
}

fn needs_revalidation_at<T: RuntimeModel>(model: &T, now: std::time::Instant) -> bool {
    model.runtime_plan().retry_at.is_some_and(|at| at <= now)
}

/// Retry complete owned results transactionally. An expired healthy fallback
/// stays resident until the replacement completes the caller's real input.
pub(crate) fn retry_model<T: RuntimeModel, R>(
    model: &mut T,
    device: &mut Device,
    cache_dir: &Path,
    operation: impl FnMut(&mut T, Device) -> Result<R>,
    reload: impl FnMut() -> Result<(T, Device)>,
) -> Result<(R, bool)> {
    retry_model_with_clock(
        model,
        device,
        cache_dir,
        std::time::Instant::now,
        operation,
        reload,
    )
}

#[cfg(test)]
fn retry_model_at<T: RuntimeModel, R>(
    model: &mut T,
    device: &mut Device,
    cache_dir: &Path,
    now: std::time::Instant,
    operation: impl FnMut(&mut T, Device) -> Result<R>,
    reload: impl FnMut() -> Result<(T, Device)>,
) -> Result<(R, bool)> {
    retry_model_with_clock(model, device, cache_dir, || now, operation, reload)
}

fn retry_model_with_clock<T: RuntimeModel, R>(
    model: &mut T,
    device: &mut Device,
    cache_dir: &Path,
    mut clock: impl FnMut() -> std::time::Instant,
    mut operation: impl FnMut(&mut T, Device) -> Result<R>,
    mut reload: impl FnMut() -> Result<(T, Device)>,
) -> Result<(R, bool)> {
    let now = clock();
    let revalidating = needs_revalidation_at(model, now);
    let mut failed_devices = Vec::new();
    let mut replacement = None;
    if revalidating {
        model.runtime_plan_mut().retry_at = Some(now + RUNTIME_RETRY);
        match reload() {
            Ok(selected) => replacement = Some(selected),
            Err(error) => {
                model.runtime_plan_mut().retry_at = Some(clock() + RUNTIME_RETRY);
                tracing::warn!(%error, "revalidation load failed; retaining healthy fallback")
            }
        }
    }
    loop {
        let (active, selected) = match &mut replacement {
            Some((active, selected)) => (active, *selected),
            None => (&mut *model, *device),
        };
        match operation(active, selected) {
            Ok(result) => {
                let replaced = replacement.is_some();
                if let Some((active, selected)) = replacement {
                    *model = active;
                    *device = selected;
                }
                return Ok((result, replaced));
            }
            Err(error) => {
                if selected == Device::Cpu {
                    if revalidating && replacement.is_some() {
                        return operation(model, *device).map(|result| (result, false));
                    }
                    return Err(error);
                }
                let target = active.runtime_plan().target.clone();
                if failed_devices.contains(&target) {
                    if revalidating {
                        return operation(model, *device).map(|result| (result, false));
                    }
                    return Err(error);
                }
                tracing::warn!(device=selected.name(), %error, "accelerator execution failed; qualifying remaining plans");
                failed_devices.push(target);
                let now = clock();
                quarantine_runtime(cache_dir, active.runtime_plan(), now);
                if model.runtime_plan().restore_target.is_none() {
                    model.runtime_plan_mut().restore_target = Some(failed_devices[0].clone());
                }
                model.runtime_plan_mut().retry_at = Some(now + RUNTIME_RETRY);
                drop(replacement.take());
                let (mut active, selected) = match reload() {
                    Ok(plan) => plan,
                    Err(_) if revalidating => {
                        model.runtime_plan_mut().retry_at = Some(clock() + RUNTIME_RETRY);
                        return operation(model, *device).map(|result| (result, false));
                    }
                    Err(error) => return Err(error),
                };
                if failed_devices.contains(&active.runtime_plan().target) {
                    if revalidating {
                        return operation(model, *device).map(|result| (result, false));
                    }
                    return Err(error.context("recovery selected the failing provider"));
                }
                active.runtime_plan_mut().retry_at = earlier_deadline(
                    active.runtime_plan().retry_at,
                    model.runtime_plan().retry_at,
                );
                active.runtime_plan_mut().restore_target =
                    model.runtime_plan().restore_target.clone();
                replacement = Some((active, selected));
            }
        }
    }
}

// Cargo profile options can arrive through ancestor/global config or --config
// without corresponding build-script environment variables. Bind timing plans
// to the actual executable once per process, covering those effective builds.
fn executable_identity() -> Option<&'static str> {
    static IDENTITY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    IDENTITY.get_or_init(mapped_image_identity).as_deref()
}

fn mapped_image_identity() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        executable_hash(Path::new("/proc/self/exe")).map(|hash| format!("elf:{hash}"))
    }
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn _dyld_get_image_header(index: u32) -> *const u8;
        }
        // SAFETY: dyld's index-zero header belongs to the running executable,
        // remains mapped for this process, and carries trusted load commands.
        let commands = unsafe {
            let header = _dyld_get_image_header(0);
            if header.is_null() {
                return None;
            }
            let bytes = std::slice::from_raw_parts(header, 32);
            if u32::from_ne_bytes(bytes[..4].try_into().ok()?) != 0xfeedfacf {
                return None;
            }
            let size = u32::from_ne_bytes(bytes[20..24].try_into().ok()?) as usize;
            if size > 1_048_576 {
                return None;
            }
            std::slice::from_raw_parts(header.add(32), size)
        };
        let uuid = mach_uuid(commands)?;
        Some(format!(
            "macho:{}",
            uuid.iter().map(|b| format!("{b:02x}")).collect::<String>()
        ))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    } // Existing opaque Windows providers remain process-local.
}

#[cfg(any(target_os = "macos", test))]
fn mach_uuid(commands: &[u8]) -> Option<[u8; 16]> {
    let mut offset = 0;
    while offset < commands.len() {
        let header = commands.get(offset..offset.checked_add(8)?)?;
        let command = u32::from_ne_bytes(header[..4].try_into().ok()?);
        let size = u32::from_ne_bytes(header[4..8].try_into().ok()?) as usize;
        if size < 8 {
            return None;
        }
        let data = commands.get(offset..offset.checked_add(size)?)?;
        if command == 0x1b && size == 24 {
            return data[8..24].try_into().ok();
        }
        offset += size;
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn executable_hash(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 32768];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Some(format!("{:x}", hash.finalize()))
}

pub(crate) fn measured<T: RuntimeModel>(
    identity: &str,
    cache_dir: &Path,
    references: &std::cell::RefCell<References>,
    mut load: impl FnMut(Device, Target, bool) -> Result<T>,
    evaluate: impl FnMut(&mut T, Device) -> Result<std::time::Duration>,
) -> Result<(T, Device)> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static PLANS: OnceLock<Mutex<HashMap<String, CachedPlan>>> = OnceLock::new();
    // Model revision/export and supported shape policy live in identity.
    // Runtime/build and dispatch settings distinguish execution plans.
    let mut settings: Vec<_> = std::env::vars_os()
        .filter(|(key, _)| key.to_string_lossy().starts_with("PAMIN_"))
        .collect();
    settings.sort();
    let executable = executable_identity();
    let namespace = format!(
        "{identity}|executable:{executable:?}|{}|{}|{settings:?}|threads:{:?}",
        env!("PAMIN_INFERENCE_BUILD"),
        ort::info(),
        threads()
    );
    let discovered = accelerators();
    let target_ids: Vec<_> = discovered
        .iter()
        .map(|(device, target)| target_identity(*device, target))
        .collect();
    let snapshot = runtime_snapshot(
        cache_dir,
        &namespace,
        &target_ids,
        std::time::Instant::now(),
    );
    let blocked = snapshot.blocked;
    let load = |device, target: Target, validated| -> Result<T> {
        let target_id = target_identity(device, &target);
        let mut model = load(device, target, validated)?;
        *model.runtime_plan_mut() = RuntimePlan {
            namespace: namespace.clone(),
            target: target_id,
            retry_at: blocked.as_ref().map(|(at, _)| *at),
            restore_target: blocked.as_ref().map(|(_, target)| target.clone()),
        };
        Ok(model)
    };
    let plans: Vec<_> = discovered
        .into_iter()
        .filter(|(device, target)| {
            !snapshot
                .unavailable
                .contains(&target_identity(*device, target))
        })
        .collect();
    if plans.is_empty() {
        let directory = cache_dir.join("compute-plans-v1");
        if directory.is_dir() {
            let _lock = file_lock(&directory.join("plans.lock"));
            let _ = prune_plans(&directory, None);
        }
        let mut load = load;
        return load(Device::Cpu, vec![cpu()].into(), false).map(|model| (model, Device::Cpu));
    }
    let signature: Vec<_> = plans
        .iter()
        .map(|(device, target)| match target {
            Target::Npu { provider, id } => format!("{provider}:{id}"),
            _ => device.name().to_string(),
        })
        .collect();
    let settings: Vec<_> = std::env::vars_os()
        .filter(|(key, _)| key.to_string_lossy().starts_with("PAMIN_"))
        .collect();
    let mut settings = settings;
    settings.sort();
    let libraries: Vec<_> = std::env::var_os("PAMIN_EP_LIBRARIES")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .map(|path| {
            let metadata = std::fs::metadata(&path).ok();
            (
                path,
                metadata.as_ref().map(std::fs::Metadata::len),
                metadata.and_then(|m| m.modified().ok()),
            )
        })
        .collect();
    let cuda = plans.iter().any(|(device, _)| *device == Device::Cuda);
    let cuda_inventory = cuda.then(cuda_identity).flatten();
    let cuda_settings = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "NVIDIA_VISIBLE_DEVICES",
    ]
    .map(|name| (name, std::env::var_os(name)));
    let key = format!(
        "persistent-plan-v6|executable:{executable:?}|build:{}|{identity}|{signature:?}|{settings:?}|threads:{:?}|cores:{:?}|runtime:{}|host:{}|features:{:?}|libraries:{libraries:?}|cuda:{cuda_inventory:?}|cuda-settings:{cuda_settings:?}",
        env!("PAMIN_INFERENCE_BUILD"),
        threads(),
        std::thread::available_parallelism(),
        ort::info(),
        host_identity(),
        crate::prepared::features()
    );
    let cache = PLANS.get_or_init(Default::default);
    // Only hashes and fixed-fixture outputs reach disk; the full key can
    // contain paths/environment values and is never persisted or logged.
    // Opaque DirectML adapter ordinals and NPU ids do not establish stable
    // physical hardware/driver identity across processes. Keep those plans
    // process-local; candidate measurement/accelerator execution is unchanged.
    let location = plan_file(cache_dir, &key);
    let coordination = host_calibration_directory();
    let disk = (executable.is_some()
        && coordination_supported(coordination.as_deref())
        && persistence_supported(&plans, cuda_inventory.is_some()))
    .then(|| location.clone())
    .flatten();
    if !cache
        .lock()
        .expect("compute-plan cache poisoned")
        .contains_key(&key)
        && let Some(plan) = disk.as_ref().and_then(|path| {
            let _lock = plan_metadata_lock(path);
            read_plan(path, &key, &plans)
        })
    {
        cache
            .lock()
            .expect("compute-plan cache poisoned")
            .insert(key.clone(), plan);
    }
    let (mut model, device, deadline) = calibrated_with_references(
        &key,
        plans,
        cache,
        references,
        PlanFiles {
            record: disk.as_deref(),
            directory: coordination.as_deref(),
        },
        load,
        evaluate,
    )?;
    finish_runtime_selection(model.runtime_plan_mut(), blocked, deadline);
    Ok((model, device))
}

fn earlier_deadline(
    a: Option<std::time::Instant>,
    b: Option<std::time::Instant>,
) -> Option<std::time::Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn finish_runtime_selection(
    plan: &mut RuntimePlan,
    blocked: Option<(std::time::Instant, String)>,
    deadline: Option<std::time::Instant>,
) {
    plan.retry_at = earlier_deadline(deadline, blocked.as_ref().map(|(at, _)| *at));
    plan.restore_target = blocked.map(|(_, target)| target);
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DiskPlan {
    fingerprint: String,
    target: String,
    expires: u64,
    references: References,
}

fn target_identity(device: Device, target: &Target) -> String {
    match target {
        Target::Npu { provider, id } => format!("npu:{provider}:{id}"),
        _ => device.name().to_string(),
    }
}

fn fingerprint(key: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(key.as_bytes()))
}

fn plan_file(root: &Path, key: &str) -> Option<PathBuf> {
    let directory = root.join("compute-plans-v1");
    std::fs::create_dir_all(&directory).ok()?;
    let _lock = file_lock(&directory.join("plans.lock"))?;
    let path = directory.join(format!("{}.json", fingerprint(key)));
    prune_plans(&directory, Some(&path)).ok()?;
    Some(path)
}

const MAX_DISK_PLANS: usize = 256;

fn plan_metadata_lock(path: &Path) -> Option<std::fs::File> {
    file_lock(&path.parent()?.join("plans.lock"))
}

/// Call while holding plans.lock. One fixed metadata lock replaces per-key
/// lock files; never unlink legacy locks, whose old inode may have waiters.
fn prune_plans(directory: &Path, preserve: Option<&Path>) -> std::io::Result<()> {
    let now = unix_seconds();
    let mut valid = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "partial")
            && let Some((hash, id)) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.split_once('.'))
            && hash.len() == 64
            && hash.bytes().all(|b| b.is_ascii_hexdigit())
            && uuid::Uuid::parse_str(id).is_ok()
        {
            // Current writers hold plans.lock. Older writers may still hold
            // their per-key lock; never remove a live legacy publication.
            let legacy = directory.join(format!("{hash}.lock"));
            let _legacy = if legacy.exists() {
                let lock = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(legacy)?;
                if lock.try_lock().is_err() {
                    continue;
                }
                Some(lock)
            } else {
                None
            };
            std::fs::remove_file(path)?;
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if path.extension().is_none_or(|ext| ext != "json")
            || stem.len() != 64
            || !stem.bytes().all(|b| b.is_ascii_hexdigit())
            || !entry.file_type()?.is_file()
        {
            continue;
        }
        let record = if entry.metadata()?.len() <= 1_048_576 {
            serde_json::from_slice::<DiskPlan>(&std::fs::read(&path)?).ok()
        } else {
            None
        };
        if let Some(record) =
            record.filter(|r| r.fingerprint == stem && r.expires > now && r.expires <= now + 86400)
        {
            valid.push((record.expires, path));
        } else {
            std::fs::remove_file(path)?;
        }
    }
    valid.sort();
    let excess = valid.len().saturating_sub(MAX_DISK_PLANS);
    for (_, path) in valid
        .into_iter()
        .filter(|(_, path)| Some(path.as_path()) != preserve)
        .take(excess)
    {
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn valid_references(references: &References) -> bool {
    fn matrix(values: &Option<Vec<Vec<f32>>>) -> bool {
        values.as_ref().is_none_or(|rows| {
            !rows.is_empty()
                && rows.len() <= 64
                && !rows[0].is_empty()
                && rows[0].len() <= 65536
                && rows
                    .iter()
                    .all(|row| row.len() == rows[0].len() && row.iter().all(|v| v.is_finite()))
        })
    }
    matrix(&references.vectors)
        && matrix(&references.queries)
        && references.scores.as_ref().is_none_or(|scores| {
            (4..=64).contains(&scores.len()) && scores.iter().all(|v| v.is_finite())
        })
}

fn host_calibration_directory() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    let cache = PathBuf::from(std::env::var_os("LOCALAPPDATA")?);
    #[cfg(target_os = "macos")]
    let cache = PathBuf::from(std::env::var_os("HOME")?).join("Library/Caches");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    let directory = cache.join("pamin-inference");
    std::fs::create_dir_all(&directory).ok()?;
    Some(directory)
}

fn coordination_supported(directory: Option<&Path>) -> bool {
    let Some(directory) = directory else {
        return false;
    };
    let Ok(file) = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(directory.join("calibration.lock"))
    else {
        return false;
    };
    match file.try_lock() {
        Ok(()) | Err(std::fs::TryLockError::WouldBlock) => true,
        Err(std::fs::TryLockError::Error(_)) => false,
    }
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn read_plan(path: &Path, key: &str, plans: &[(Device, Target)]) -> Option<CachedPlan> {
    if std::fs::metadata(path).ok()?.len() > 1_048_576 {
        return None;
    }
    let saved: DiskPlan = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    let now = unix_seconds();
    if saved.fingerprint != fingerprint(key)
        || saved.expires <= now
        || saved.expires > now + 86400
        || !valid_references(&saved.references)
    {
        return None;
    }
    let (device, target) = if saved.target == Device::Cpu.name() {
        (Device::Cpu, vec![cpu()].into())
    } else {
        plans
            .iter()
            .find(|(device, target)| target_identity(*device, target) == saved.target)?
            .clone()
    };
    Some(CachedPlan {
        device,
        target,
        revalidate: Some(
            std::time::Instant::now() + std::time::Duration::from_secs(saved.expires - now),
        ),
        references: saved.references,
    })
}

fn write_plan(path: &Path, key: &str, plan: &CachedPlan) -> std::io::Result<()> {
    let remaining = plan.revalidate.map_or(86400, |deadline| {
        deadline
            .saturating_duration_since(std::time::Instant::now())
            .as_secs()
    });
    if remaining == 0 {
        let _ = std::fs::remove_file(path);
        return Ok(());
    }
    let record = DiskPlan {
        fingerprint: fingerprint(key),
        target: target_identity(plan.device, &plan.target),
        expires: unix_seconds() + remaining.min(86400),
        references: plan.references.clone(),
    };
    let pending = path.with_extension(format!("{}.partial", uuid::Uuid::now_v7()));
    let result = (|| {
        std::fs::write(&pending, serde_json::to_vec(&record)?)?;
        match std::fs::rename(&pending, path) {
            #[cfg(target_os = "windows")]
            Err(_) if path.exists() => {
                // The per-key file lock excludes readers/writers. A crash
                // between removal and publication is just a safe cache miss.
                std::fs::remove_file(path)?;
                std::fs::rename(&pending, path)
            }
            result => result,
        }
    })();
    let _ = std::fs::remove_file(pending);
    result
}

fn remember_plan(
    key: &str,
    disk: Option<&Path>,
    cache: &std::sync::Mutex<std::collections::HashMap<String, CachedPlan>>,
    plan: CachedPlan,
) {
    cache
        .lock()
        .expect("compute-plan cache poisoned")
        .insert(key.into(), plan.clone());
    if let Some(path) = disk {
        let _lock = plan_metadata_lock(path);
        if let Err(error) = write_plan(path, key, &plan)
            .and_then(|()| prune_plans(path.parent().expect("plan directory"), Some(path)))
        {
            tracing::debug!(%error, "compute-plan persistence unavailable");
        }
    }
}

fn fresh_plan(plan: &CachedPlan) -> bool {
    plan.revalidate
        .is_none_or(|deadline| std::time::Instant::now() < deadline)
}

fn file_lock(path: &Path) -> Option<std::fs::File> {
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .ok()?;
    lock.lock().ok()?;
    Some(lock)
}

fn persistence_supported(plans: &[(Device, Target)], cuda_identified: bool) -> bool {
    plans.iter().all(|(device, _)| match device {
        Device::Cpu | Device::CoreMl => true,
        Device::Cuda => cuda_identified,
        Device::DirectMl | Device::Npu => false,
    })
}

fn cuda_identity() -> Option<String> {
    #[cfg(unix)]
    {
        let mut command = std::process::Command::new("nvidia-smi");
        command.args([
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader",
        ]);
        cuda_inventory(&bounded_inventory(
            command,
            std::time::Duration::from_secs(2),
        )?)
    }
    #[cfg(not(unix))]
    None
}

/// At most one inventory child and cleanup worker may exist in this process.
/// A wedged driver keeps the permit until the killed child is actually reaped;
/// further probes return no identity without adding children or threads.
#[cfg(unix)]
struct InventoryPermit(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(unix)]
impl InventoryPermit {
    fn acquire(busy: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Option<Self> {
        busy.compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .ok()?;
        Some(Self(busy))
    }
}

#[cfg(unix)]
impl Drop for InventoryPermit {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(unix)]
fn bounded_inventory(
    command: std::process::Command,
    timeout: std::time::Duration,
) -> Option<Vec<u8>> {
    static BUSY: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicBool>> =
        std::sync::OnceLock::new();
    let permit = InventoryPermit::acquire(BUSY.get_or_init(Default::default).clone())?;
    inventory_with_permit(command, timeout, permit)
}

#[cfg(unix)]
fn inventory_with_permit(
    mut command: std::process::Command,
    timeout: std::time::Duration,
    permit: InventoryPermit,
) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    use std::process::Stdio;
    // The standard library owns nonblocking configuration and descriptor
    // transfer. A socket pair avoids handwritten fcntl FFI.
    let (mut output, writer) = UnixStream::pair().ok()?;
    output.set_nonblocking(true).ok()?;
    command
        .stdout(Stdio::from(OwnedFd::from(writer)))
        .stderr(Stdio::null());
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("cuda-inventory".into())
        .spawn(move || {
            let Some(mut child) = command.spawn().ok() else {
                // No child exists: release admission before waking the caller.
                drop(permit);
                let _ = send.send(None);
                return;
            };
            let deadline = std::time::Instant::now() + timeout;
            let result = (|| {
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                loop {
                    let exited = child.try_wait().ok()?;
                    loop {
                        match output.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(read) => {
                                if bytes.len() + read > 65536 {
                                    return None;
                                }
                                bytes.extend_from_slice(&buffer[..read]);
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => return None,
                        }
                        if std::time::Instant::now() >= deadline {
                            return None;
                        }
                    }
                    if let Some(status) = exited {
                        return status.success().then_some(bytes);
                    }
                    if std::time::Instant::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            })();
            if result.is_none() {
                let _ = child.kill();
                // This wait can outlast the caller's deadline but retains the sole
                // permit. No subsequent inventory can create another reaper.
                let _ = child.wait();
            }
            drop(permit);
            let _ = send.send(result);
        })
        .ok()?;
    receive
        .recv_timeout(timeout + std::time::Duration::from_millis(100))
        .ok()?
}

#[cfg(any(unix, test))]
fn cuda_inventory(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?.trim();
    if text.is_empty()
        || text.lines().any(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            fields.len() != 5
                || fields.iter().any(|field| field.is_empty())
                || !fields[1].starts_with("GPU-")
        })
    {
        return None;
    }
    Some(text.to_string())
}

fn host_identity() -> &'static str {
    static HOST: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HOST.get_or_init(|| {
        let mut pieces = vec![
            std::env::consts::OS.to_string(),
            std::env::consts::ARCH.to_string(),
        ];
        for key in ["COMPUTERNAME", "PROCESSOR_IDENTIFIER"] {
            if let Ok(value) = std::env::var(key) {
                pieces.push(value);
            }
        }
        #[cfg(unix)]
        for (command, args) in [("hostname", vec![]), ("uname", vec!["-rs"])] {
            if let Ok(output) = std::process::Command::new(command).args(args).output() {
                pieces.push(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }
        }
        #[cfg(target_os = "macos")]
        if let Ok(output) = std::process::Command::new("sysctl")
            .args(["-n", "hw.model", "machdep.cpu.brand_string"])
            .output()
        {
            pieces.push(String::from_utf8_lossy(&output.stdout).trim().to_string());
        }
        pieces.join("|")
    })
}

#[derive(Clone, Copy, Default)]
struct PlanFiles<'a> {
    record: Option<&'a Path>,
    directory: Option<&'a Path>,
}

impl<'a> PlanFiles<'a> {
    #[cfg(test)]
    fn for_record(path: &'a Path) -> Self {
        Self {
            record: Some(path),
            directory: path.parent(),
        }
    }
}

fn calibrated_with_references<T>(
    key: &str,
    plans: Vec<(Device, Target)>,
    cache: &std::sync::Mutex<std::collections::HashMap<String, CachedPlan>>,
    references: &std::cell::RefCell<References>,
    files: PlanFiles<'_>,
    mut load: impl FnMut(Device, Target, bool) -> Result<T>,
    mut evaluate: impl FnMut(&mut T, Device) -> Result<std::time::Duration>,
) -> Result<(T, Device, Option<std::time::Instant>)> {
    if plans.is_empty() {
        return load(Device::Cpu, vec![cpu()].into(), false)
            .map(|model| (model, Device::Cpu, None));
    }
    let mut disk = if coordination_supported(files.directory) {
        files.record
    } else {
        None
    };
    // Serialize only actual calibration misses. Cached loading and output
    // validation may fetch/compile a model and must not hold either global lock.
    static CALIBRATION: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let (_calibration, _host_lock) = loop {
        let remembered = cache
            .lock()
            .expect("compute-plan cache poisoned")
            .get(key)
            .cloned()
            .filter(fresh_plan);
        if let Some(mut plan) = remembered {
            let device = plan.device;
            let deadline = plan.revalidate;
            references.replace(std::mem::take(&mut plan.references));
            match load(device, plan.target.clone(), true).and_then(|mut model| {
                if device != Device::Cpu {
                    evaluate(&mut model, device)?;
                }
                Ok(model)
            }) {
                Ok(model) => {
                    if let Some(path) = disk
                        && !path.exists()
                    {
                        plan.references = references.borrow().clone();
                        remember_plan(key, disk, cache, plan);
                    }
                    return Ok((model, device, deadline));
                }
                Err(error) => {
                    tracing::warn!(%error, "cached compute plan failed; recalibrating");
                    cache
                        .lock()
                        .expect("compute-plan cache poisoned")
                        .remove(key);
                    if let Some(path) = disk {
                        let _lock = plan_metadata_lock(path);
                        let _ = std::fs::remove_file(path);
                    }
                }
            }
        }
        let calibration = CALIBRATION.lock().expect("compute calibration poisoned");
        let host_lock = files
            .directory
            .and_then(|directory| file_lock(&directory.join("calibration.lock")));
        if host_lock.is_none() {
            disk = None; // Never publish/reuse cross-process timing without coordination.
        }
        // A different caller/process may have populated the plan while we waited.
        if cache
            .lock()
            .expect("compute-plan cache poisoned")
            .get(key)
            .is_some_and(fresh_plan)
        {
            continue;
        }
        if let Some(plan) = disk.and_then(|path| {
            let _lock = plan_metadata_lock(path);
            read_plan(path, key, &plans)
        }) {
            cache
                .lock()
                .expect("compute-plan cache poisoned")
                .insert(key.into(), plan);
            continue;
        }
        break (calibration, host_lock);
    };
    references.replace(References::default());
    let mut reference = load(Device::Cpu, vec![cpu()].into(), false)?;
    let mut transient_failure = false;
    let mut numerical_failure = false;
    let rounds = 3;
    let mut observations: Vec<Vec<(std::time::Duration, bool)>> = vec![Vec::new(); plans.len()];
    let mut quarantined = vec![false; plans.len()];
    for round in 0..rounds {
        for offset in 0..plans.len() {
            let index = (round + offset) % plans.len();
            if quarantined[index] {
                continue;
            }
            let (device, target) = plans[index].clone();
            let mut candidate = match load(device, target, false) {
                Ok(model) => model,
                Err(error) => {
                    numerical_failure |= matches!(error, IndexError::Numerical(_));
                    quarantined[index] |= matches!(
                        error,
                        IndexError::Incompatible(_) | IndexError::Numerical(_)
                    );
                    transient_failure |= !quarantined[index];
                    tracing::warn!(device=device.name(),%error,"compute candidate unavailable");
                    continue;
                }
            };
            let before = evaluate(&mut reference, Device::Cpu)?;
            let elapsed = match evaluate(&mut candidate, device) {
                Ok(elapsed) => elapsed,
                Err(error) => {
                    numerical_failure |= matches!(error, IndexError::Numerical(_));
                    quarantined[index] |= matches!(
                        error,
                        IndexError::Incompatible(_) | IndexError::Numerical(_)
                    );
                    transient_failure |= !quarantined[index];
                    tracing::warn!(device=device.name(),%error,"compute candidate failed validation");
                    continue;
                }
            };
            let after = evaluate(&mut reference, Device::Cpu)?;
            let control = before.min(after);
            if control.is_zero() {
                return Err(IndexError::Engine(
                    "compute calibration produced no duration".into(),
                ));
            }
            observations[index].push((elapsed, elapsed < control));
            tracing::info!(
                round,
                device = device.name(),
                candidate_us = elapsed.as_micros(),
                cpu_before_us = before.as_micros(),
                cpu_after_us = after.as_micros(),
                ratio_to_cpu = elapsed.as_secs_f64() / control.as_secs_f64(),
                "complete model-call calibration"
            );
        }
    }
    let mut fastest = (Device::Cpu, vec![cpu()].into());
    let mut fastest_elapsed = std::time::Duration::MAX;
    for (index, values) in observations.iter_mut().enumerate() {
        // A complete rotated process-local comparison with a majority of
        // wins over both adjacent CPU controls; rank by absolute median time.
        if values.len() != rounds || values.iter().filter(|(_, wins)| *wins).count() <= rounds / 2 {
            continue;
        }
        values.sort_by_key(|(elapsed, _)| *elapsed);
        let elapsed = values[values.len() / 2].0;
        if elapsed < fastest_elapsed {
            fastest_elapsed = elapsed;
            fastest = plans[index].clone();
        }
    }
    let (device, target) = fastest;
    if device == Device::Cpu {
        let deadline = if !transient_failure {
            let plan = CachedPlan::fresh(device, target, numerical_failure)
                .with_references(references.borrow().clone());
            let deadline = plan.revalidate;
            remember_plan(key, disk, cache, plan);
            deadline
        } else {
            Some(std::time::Instant::now() + RUNTIME_RETRY)
        };
        return Ok((reference, device, deadline));
    }
    match load(device, target.clone(), true).and_then(|mut model| {
        evaluate(&mut model, device)?;
        Ok(model)
    }) {
        Ok(model) => {
            // A validated winner remains useful when another target is down.
            // Reuse it, but revisit failed alternatives at the short deadline.
            let plan = CachedPlan::fresh(device, target, numerical_failure || transient_failure)
                .with_references(references.borrow().clone());
            let deadline = plan.revalidate;
            remember_plan(key, disk, cache, plan);
            Ok((model, device, deadline))
        }
        Err(error) => {
            tracing::warn!(%error, "calibrated winner failed to reload; using optimized CPU");
            Ok((
                reference,
                Device::Cpu,
                Some(std::time::Instant::now() + RUNTIME_RETRY),
            ))
        }
    }
}

#[cfg(test)]
fn calibrated<T>(
    key: &str,
    plans: Vec<(Device, Target)>,
    cache: &std::sync::Mutex<std::collections::HashMap<String, CachedPlan>>,
    load: impl FnMut(Device, Target, bool) -> Result<T>,
    evaluate: impl FnMut(&mut T, Device) -> Result<std::time::Duration>,
) -> Result<(T, Device)> {
    calibrated_with_references(
        key,
        plans,
        cache,
        &std::cell::RefCell::default(),
        PlanFiles::default(),
        load,
        evaluate,
    )
    .map(|(model, device, _)| (model, device))
}

/// Three uncached calls after a warm call. The callback includes tokenizer,
/// inference and output processing; cached query/score lookups are excluded.
pub(crate) fn time_calls(mut call: impl FnMut() -> Result<()>) -> Result<std::time::Duration> {
    call()?;
    let mut times = [std::time::Duration::ZERO; 3];
    for elapsed in &mut times {
        let start = std::time::Instant::now();
        call()?;
        *elapsed = start.elapsed();
    }
    times.sort();
    Ok(times[1])
}

/// Use the same modern format for every CoreML model. The legacy NeuralNetwork
/// format cannot accept the standard normalization operators used by modern
/// transformer graphs. ALL permits the framework's CPU/GPU/ANE combination.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn coreml() -> ort::ep::CoreML {
    ort::ep::CoreML::default()
        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ort::ep::coreml::ComputeUnits::All)
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
pub(crate) fn accelerators() -> Vec<(Device, Target)> {
    if pamin_core::env::is("PAMIN_DEVICE", "cpu") {
        return Vec::new();
    }
    let mut result = Vec::new();
    if let Ok(environment) = ort::environment::Environment::current() {
        register_plugins(&environment);
        #[cfg(target_os = "windows")]
        crate::winml::register(&environment);
        for device in environment.devices() {
            let hardware = device.hardware_device();
            if hardware.ty() == ort::memory::DeviceType::NPU
                && let Ok(provider) = device.ep()
                && provider != "CoreMLExecutionProvider"
            {
                let target = Target::Npu {
                    provider: provider.into(),
                    id: hardware.id(),
                };
                result.push((Device::Npu, target));
            }
        }
    }
    result.extend(
        gpu_providers()
            .into_iter()
            .map(|(device, ep)| (device, vec![ep.error_on_failure()].into())),
    );
    result
}

/// Retain registered libraries for the environment/session lifetime. Paths
/// are explicit because vendor plugin installation differs by platform.
fn register_plugins(environment: &std::sync::Arc<ort::environment::Environment>) {
    static LIBRARIES: std::sync::OnceLock<Vec<ort::ep::ExecutionProviderLibrary>> =
        std::sync::OnceLock::new();
    LIBRARIES.get_or_init(|| {
        std::env::var_os("PAMIN_EP_LIBRARIES").into_iter()
            .flat_map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
            .enumerate().filter_map(|(index, path)| {
                if !path.is_absolute() {
                    tracing::warn!(path = %path.display(), "execution-provider library must be absolute");
                    return None;
                }
                match environment.register_ep_library(format!("pamin-plugin-{index}"), &path) {
                    Ok(library) => Some(library),
                    Err(error) => { tracing::warn!(path = %path.display(), %error, "execution-provider library unavailable"); None }
                }
            }).collect()
    });
}

fn gpu_providers() -> Vec<(Device, ExecutionProviderDispatch)> {
    vec![
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        (Device::Cuda, ort::ep::CUDA::default().build()),
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        (Device::CoreMl, coreml().build()),
        #[cfg(target_os = "windows")]
        (Device::DirectMl, ort::ep::DirectML::default().build()),
    ]
}

#[cfg(test)]
mod tests {

    struct RuntimeFixture<T> {
        value: T,
        plan: RuntimePlan,
    }
    impl<T> RuntimeFixture<T> {
        fn new(value: T, target: &str) -> Self {
            Self {
                value,
                plan: RuntimePlan {
                    namespace: "fixture-revision-runtime-build-shape".into(),
                    target: target.into(),
                    retry_at: None,
                    restore_target: None,
                },
            }
        }
    }
    impl<T> RuntimeModel for RuntimeFixture<T> {
        fn runtime_plan(&self) -> &RuntimePlan {
            &self.plan
        }
        fn runtime_plan_mut(&mut self) -> &mut RuntimePlan {
            &mut self.plan
        }
    }

    #[test]
    fn runtime_recovery_keeps_gpu_before_cpu_and_retries_whole_results() {
        let root = tempfile::tempdir().unwrap();
        let mut model = RuntimeFixture::new(0, "npu:first:1");
        let mut device = Device::Npu;
        let mut alternatives = [Device::Cuda, Device::Cpu].into_iter();
        let mut attempts = Vec::new();
        let (result, replaced) = retry_model(
            &mut model,
            &mut device,
            root.path(),
            |_, device| {
                attempts.push(device);
                if device == Device::Cpu {
                    Ok(vec![1.0, 2.0])
                } else {
                    Err(IndexError::Engine("failure after a partial batch".into()))
                }
            },
            || {
                let selected = alternatives.next().unwrap();
                Ok((RuntimeFixture::new(1, selected.name()), selected))
            },
        )
        .unwrap();
        assert_eq!(attempts, [Device::Npu, Device::Cuda, Device::Cpu]);
        assert_eq!(result, [1.0, 2.0]);
        assert!(replaced);
        assert_eq!(device, Device::Cpu);
        assert!(model.plan.retry_at.is_some());
    }

    #[test]
    fn failed_recovery_does_not_publish_an_intermediate_backend() {
        let root = tempfile::tempdir().unwrap();
        let mut model = RuntimeFixture::new(0, "npu:first:1");
        let mut device = Device::Npu;
        let mut loads = 0;
        let result: Result<((), bool)> = retry_model(
            &mut model,
            &mut device,
            root.path(),
            |_, _| Err(IndexError::Engine("execution fault".into())),
            || {
                loads += 1;
                if loads == 1 {
                    Ok((RuntimeFixture::new(1, "cuda"), Device::Cuda))
                } else {
                    Err(IndexError::Engine("CPU load failed".into()))
                }
            },
        );
        assert!(result.is_err());
        assert_eq!(model.value, 0);
        assert_eq!(device, Device::Npu);
        assert_eq!(model.plan.target, "npu:first:1");
        let (next, replaced) = retry_model(
            &mut model,
            &mut device,
            root.path(),
            |_, selected| {
                if selected == Device::Cpu {
                    Ok(7)
                } else {
                    Err(IndexError::Engine("execution fault".into()))
                }
            },
            || Ok((RuntimeFixture::new(2, "cpu"), Device::Cpu)),
        )
        .unwrap();
        assert_eq!(next, 7);
        assert!(replaced);
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn execution_failure_reloads_a_qualified_remaining_device_and_retries() {
        let root = tempfile::tempdir().unwrap();
        let mut model = RuntimeFixture::new(0, "npu:first:1");
        let mut device = Device::Npu;
        let (result, replaced) = retry_model(
            &mut model,
            &mut device,
            root.path(),
            |model, device| {
                if device == Device::Npu {
                    Err(IndexError::Engine("device reset".into()))
                } else {
                    Ok(model.value + 1)
                }
            },
            || {
                assert!(!runtime_available(
                    root.path(),
                    "fixture-revision-runtime-build-shape",
                    "npu:first:1"
                ));
                Ok((RuntimeFixture::new(41, "cpu"), Device::Cpu))
            },
        )
        .unwrap();
        assert_eq!(result, 42);
        assert!(replaced);
        assert_eq!(device, Device::Cpu);
        assert_eq!(model.value, 41);
    }

    #[test]
    fn runtime_quarantine_is_scoped_to_model_and_selected_target() {
        let root = tempfile::tempdir().unwrap();
        let fixture = RuntimeFixture::new(0, "npu:first:1");
        quarantine_runtime(root.path(), &fixture.plan, std::time::Instant::now());
        assert!(!runtime_available(
            root.path(),
            &fixture.plan.namespace,
            "npu:first:1"
        ));
        assert!(runtime_available(
            root.path(),
            "different-model-revision",
            "npu:first:1"
        ));
        assert!(runtime_available(
            root.path(),
            &fixture.plan.namespace,
            "npu:first:2"
        ));
        let mut model = fixture;
        let mut device = Device::Npu;
        let (_, replaced) = retry_model(
            &mut model,
            &mut device,
            root.path(),
            |model, _| {
                if model.plan.target.ends_with(":2") {
                    Ok(())
                } else {
                    Err(IndexError::Engine("first target reset".into()))
                }
            },
            || Ok((RuntimeFixture::new(1, "npu:first:2"), Device::Npu)),
        )
        .unwrap();
        assert!(
            replaced,
            "same coarse Device may select a different NPU target"
        );
    }

    #[test]
    fn resident_deadline_forces_real_input_before_publishing_a_replacement() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let mut model = RuntimeFixture::new(0, "cpu");
        model.plan.retry_at = Some(now + RUNTIME_RETRY);
        let mut device = Device::Cpu;
        assert!(!needs_revalidation_at(&model, now));
        assert!(needs_revalidation_at(&model, now + RUNTIME_RETRY));
        let mut attempts = 0;
        let (answer, replaced) = retry_model_at(
            &mut model,
            &mut device,
            root.path(),
            now + RUNTIME_RETRY,
            |candidate, _| {
                attempts += 1;
                Ok(candidate.value)
            },
            || Ok((RuntimeFixture::new(42, "npu:first:1"), Device::Npu)),
        )
        .unwrap();
        assert_eq!(answer, 42);
        assert_eq!(attempts, 1);
        assert!(replaced);
        assert_eq!(device, Device::Npu);
        assert!(!needs_revalidation_at(&model, now + RUNTIME_RETRY * 2));
    }

    #[test]
    fn revalidation_real_input_failure_retains_the_healthy_fallback() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let mut model = RuntimeFixture::new(42, "cpu");
        model.plan.retry_at = Some(now);
        let mut device = Device::Cpu;
        let mut loads = 0;
        let (answer, replaced) = retry_model_at(
            &mut model,
            &mut device,
            root.path(),
            now,
            |candidate, selected| {
                if selected == Device::Cpu {
                    Ok(candidate.value)
                } else {
                    Err(IndexError::Engine("actual shape fails".into()))
                }
            },
            || {
                loads += 1;
                if loads == 1 {
                    Ok((RuntimeFixture::new(0, "cuda"), Device::Cuda))
                } else {
                    Err(IndexError::Engine("CPU recreation unavailable".into()))
                }
            },
        )
        .unwrap();
        assert_eq!(answer, 42);
        assert!(!replaced);
        assert_eq!(model.value, 42);
        assert_eq!(device, Device::Cpu);
        assert_eq!(model.plan.retry_at, Some(now + RUNTIME_RETRY));
    }

    #[test]
    fn cold_load_inherits_quarantine_and_completed_measurement_clears_it() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let failed = RuntimeFixture::new(0, "npu:first:1");
        quarantine_runtime(root.path(), &failed.plan, now);
        let blocked = runtime_retry(root.path(), &failed.plan.namespace, &[], now).unwrap();
        assert_eq!(blocked, (now + RUNTIME_RETRY, "npu:first:1".into()));
        let mut cpu = RuntimeFixture::new(0, "cpu");
        finish_runtime_selection(&mut cpu.plan, Some(blocked), None);
        assert_eq!(cpu.plan.retry_at, Some(now + RUNTIME_RETRY));
        assert_eq!(cpu.plan.restore_target.as_deref(), Some("npu:first:1"));
        // Once every available plan has been remeasured, CPU (or another
        // target) can legitimately be the new winner, without endless retry.
        for target in ["cpu", "cuda", "npu:second:2"] {
            cpu.plan.target = target.into();
            finish_runtime_selection(&mut cpu.plan, None, None);
            assert!(cpu.plan.retry_at.is_none());
            assert!(cpu.plan.restore_target.is_none());
        }
        finish_runtime_selection(&mut cpu.plan, None, Some(now + RUNTIME_RETRY));
        assert_eq!(cpu.plan.retry_at, Some(now + RUNTIME_RETRY));
    }

    #[test]
    fn missing_target_keeps_rediscovery_after_its_quarantine_expires() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let failed = RuntimeFixture::new(0, "npu:reset:1");
        quarantine_runtime(root.path(), &failed.plan, now);
        let later = now + RUNTIME_RETRY * 2;
        let blocked = runtime_retry(root.path(), &failed.plan.namespace, &[], later).unwrap();
        assert_eq!(blocked, (later + RUNTIME_RETRY, "npu:reset:1".into()));
        let mut model = RuntimeFixture::new(0, "cpu");
        finish_runtime_selection(&mut model.plan, Some(blocked), None);
        assert_eq!(model.plan.retry_at, Some(later + RUNTIME_RETRY));
        assert!(
            runtime_retry(
                root.path(),
                &failed.plan.namespace,
                &["npu:reset:1".into()],
                later
            )
            .is_none()
        );
    }

    #[test]
    fn transient_revalidation_retries_but_a_measured_cpu_winner_stops() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let mut model = RuntimeFixture::new(0, "cpu");
        model.plan.retry_at = Some(now);
        let mut device = Device::Cpu;
        let mut loads = 0;
        for step in 0..3 {
            let at = now + RUNTIME_RETRY * step;
            let (_, replaced) = retry_model_at(
                &mut model,
                &mut device,
                root.path(),
                at,
                |_, _| Ok(()),
                || {
                    loads += 1;
                    let mut candidate = RuntimeFixture::new(1, "cpu");
                    finish_runtime_selection(
                        &mut candidate.plan,
                        None,
                        (step != 1).then_some(at + RUNTIME_RETRY),
                    );
                    Ok((candidate, Device::Cpu))
                },
            )
            .unwrap();
            assert_eq!(replaced, step < 2);
        }
        assert_eq!(loads, 2);
        assert!(model.plan.retry_at.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn inventory_spawn_failure_returns_without_the_inventory_timeout() {
        let root = tempfile::tempdir().unwrap();
        let command = std::process::Command::new(root.path().join("missing-nvidia-smi"));
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        let start = std::time::Instant::now();
        assert!(
            inventory_with_permit(command, std::time::Duration::from_secs(3), permit).is_none()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert!(
            InventoryPermit::acquire(busy).is_some(),
            "spawn failure retained inventory admission"
        );
    }

    #[test]
    fn effective_executable_changes_invalidate_the_measured_plan_identity() {
        let root = tempfile::tempdir().unwrap();
        let binary = root.path().join("binary");
        std::fs::write(&binary, b"effective build A").unwrap();
        let before = executable_hash(&binary).unwrap();
        std::fs::write(&binary, b"effective build B").unwrap();
        let after = executable_hash(&binary).unwrap();
        assert_ne!(
            fingerprint(&format!("plan|{before}")),
            fingerprint(&format!("plan|{after}"))
        );
        assert!(executable_hash(&root.path().join("missing")).is_none());
    }

    #[test]
    fn valid_winner_is_reused_when_another_candidate_is_transiently_down() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::default();
        let failed_loads = Cell::new(0);
        let plans = vec![
            (Device::Cuda, vec![cpu()].into()),
            (Device::CoreMl, vec![cpu()].into()),
        ];
        for _ in 0..2 {
            let (_, chosen) = calibrated(
                "mixed-provider-winner",
                plans.clone(),
                &cache,
                |device, _, _| {
                    if device == Device::Cuda {
                        failed_loads.set(failed_loads.get() + 1);
                        Err(IndexError::Engine("driver unavailable".into()))
                    } else {
                        Ok(device)
                    }
                },
                |_, device| {
                    Ok(Duration::from_millis(if device == Device::Cpu {
                        10
                    } else {
                        1
                    }))
                },
            )
            .unwrap();
            assert_eq!(chosen, Device::CoreMl);
        }
        assert_eq!(
            failed_loads.get(),
            3,
            "idle reload repeated full calibration"
        );
        let saved = cache.lock().unwrap();
        assert!(
            saved["mixed-provider-winner"]
                .revalidate
                .unwrap()
                .saturating_duration_since(std::time::Instant::now())
                <= RUNTIME_RETRY
        );
    }

    #[test]
    fn malformed_reference_shapes_are_cache_misses() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("plan.json");
        let key = "reranker-shape-control";
        let mut plan = CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false);
        for count in [0, 1, 2, 3, 65] {
            plan.references.scores = Some(vec![0.0; count]);
            write_plan(&path, key, &plan).unwrap();
            assert!(read_plan(&path, key, &[]).is_none());
        }
        plan.references.scores = Some(vec![0.0; 4]);
        write_plan(&path, key, &plan).unwrap();
        assert!(read_plan(&path, key, &[]).is_some());
        plan.references.vectors = Some(vec![vec![1.0], vec![1.0, 2.0]]);
        write_plan(&path, key, &plan).unwrap();
        assert!(read_plan(&path, key, &[]).is_none());
    }

    #[test]
    fn unavailable_coordination_disables_disk_choices() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let not_directory = root.path().join("file");
        std::fs::write(&not_directory, b"fixture").unwrap();
        let record = root.path().join("plan.json");
        assert!(!coordination_supported(Some(&not_directory)));
        assert!(!coordination_supported(None));
        let cache = std::sync::Mutex::default();
        let references = std::cell::RefCell::default();
        calibrated_with_references(
            "uncoordinated",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            &references,
            PlanFiles {
                record: Some(&record),
                directory: Some(&not_directory),
            },
            |device, _, _| Ok(device),
            |_, device| {
                Ok(Duration::from_millis(if device == Device::Cpu {
                    10
                } else {
                    1
                }))
            },
        )
        .unwrap();
        assert!(!record.exists());
    }

    #[test]
    fn workspaces_share_one_user_calibration_directory() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        assert_ne!(first.path(), second.path());
        let lock_a = host_calibration_directory()
            .unwrap()
            .join("calibration.lock");
        let lock_b = host_calibration_directory()
            .unwrap()
            .join("calibration.lock");
        assert_eq!(
            lock_a, lock_b,
            "workspace model cache must not partition host timing coordination"
        );
    }

    #[test]
    fn selected_validity_does_not_follow_another_callers_cache_update() {
        use std::time::Duration;
        let now = std::time::Instant::now();
        let deadline = now + Duration::from_secs(60);
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let key = "owned-validity";
        let mut original = CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false);
        original.revalidate = Some(deadline);
        cache.lock().unwrap().insert(key.into(), original);
        let references = std::cell::RefCell::default();
        let (_, device, selected_deadline) = calibrated_with_references(
            key,
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            &references,
            PlanFiles::default(),
            |device, _, _| {
                cache.lock().unwrap().insert(
                    key.into(),
                    CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false),
                );
                Ok(device)
            },
            |_, _| Ok(Duration::from_millis(1)),
        )
        .unwrap();
        assert_eq!(device, Device::Cuda);
        assert_eq!(selected_deadline, Some(deadline));
        let mut model = RuntimeFixture::new(0, "cuda");
        finish_runtime_selection(
            &mut model.plan,
            Some((now + RUNTIME_RETRY, "npu".into())),
            selected_deadline,
        );
        assert_eq!(model.plan.retry_at, Some(deadline));
    }

    #[test]
    fn quarantine_snapshot_selects_effective_deadlines_and_is_immutable() {
        let root = tempfile::tempdir().unwrap();
        let now = std::time::Instant::now();
        let missing = RuntimeFixture::new(0, "npu:missing");
        let live = RuntimeFixture::new(0, "cuda");
        quarantine_runtime(root.path(), &missing.plan, now - RUNTIME_RETRY * 2);
        quarantine_runtime(
            root.path(),
            &live.plan,
            now - std::time::Duration::from_secs(240),
        );
        let snapshot = runtime_snapshot(root.path(), &live.plan.namespace, &["cuda".into()], now);
        assert_eq!(
            snapshot.blocked.unwrap(),
            (now + std::time::Duration::from_secs(60), "cuda".into())
        );
        assert!(snapshot.unavailable.contains("cuda"));
        let empty = runtime_snapshot(root.path(), "other-scope", &[], now);
        let mut concurrent = RuntimeFixture::new(0, "new-target");
        concurrent.plan.namespace = "other-scope".into();
        quarantine_runtime(root.path(), &concurrent.plan, now);
        assert!(empty.blocked.is_none() && empty.unavailable.is_empty());
        let after = runtime_snapshot(root.path(), "other-scope", &[], now);
        assert!(after.blocked.is_some() && after.unavailable.contains("new-target"));
    }

    #[test]
    fn slow_failed_operation_starts_quarantine_at_observation_time() {
        let root = tempfile::tempdir().unwrap();
        let entry = std::time::Instant::now();
        let observed = entry + RUNTIME_RETRY * 2;
        let clock = std::cell::Cell::new(entry);
        let mut model = RuntimeFixture::new(0, "cuda");
        let namespace = model.plan.namespace.clone();
        let mut device = Device::Cuda;
        let (answer, _) = retry_model_with_clock(
            &mut model,
            &mut device,
            root.path(),
            || clock.get(),
            |_, device| {
                if device == Device::Cuda {
                    clock.set(observed);
                    Err(IndexError::Engine("slow driver failure".into()))
                } else {
                    Ok(42)
                }
            },
            || Ok((RuntimeFixture::new(0, "cpu"), Device::Cpu)),
        )
        .unwrap();
        assert_eq!(answer, 42);
        assert_eq!(model.plan.retry_at, Some(observed + RUNTIME_RETRY));
        assert_eq!(
            runtime_snapshot(root.path(), &namespace, &["cuda".into()], observed)
                .blocked
                .unwrap()
                .0,
            observed + RUNTIME_RETRY
        );
    }

    #[test]
    fn mapped_macho_uuid_is_not_a_mutable_executable_path() {
        let mut commands = Vec::new();
        commands.extend_from_slice(&0x1bu32.to_ne_bytes());
        commands.extend_from_slice(&24u32.to_ne_bytes());
        commands.extend_from_slice(&[17; 16]);
        assert_eq!(mach_uuid(&commands), Some([17; 16]));
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("installed-binary");
        std::fs::write(&path, b"replacement executable").unwrap();
        assert_eq!(mach_uuid(&commands), Some([17; 16]));
        assert!(mach_uuid(&commands[..20]).is_none());
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(mapped_image_identity().is_some());
    }

    #[test]
    fn one_accelerator_must_win_multiple_interleaved_rounds() {
        use std::cell::Cell;
        use std::time::Duration;
        let attempts = Cell::new(0);
        let cache = std::sync::Mutex::default();
        let (_, selected) = calibrated(
            "single-accelerator",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            |device, _, _| Ok(device),
            |_, device| {
                let millis = if device == Device::Cpu {
                    10
                } else {
                    let round = attempts.get();
                    attempts.set(round + 1);
                    if round == 0 { 1 } else { 20 }
                };
                Ok(Duration::from_millis(millis))
            },
        )
        .unwrap();
        assert_eq!(attempts.get(), 3);
        assert_eq!(
            selected,
            Device::Cpu,
            "one lucky accelerator interval selected a cached plan"
        );
    }

    #[test]
    fn calibrated_plan_chooses_time_not_discovery_and_reuses_validation() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let checks = Cell::new(0);
        let run = |accelerator: u64| {
            calibrated(
                "fixture",
                vec![(Device::Cuda, vec![cpu()].into())],
                &cache,
                |device, _, _validated| Ok(device),
                |_model, device| {
                    checks.set(checks.get() + 1);
                    Ok(Duration::from_millis(if device == Device::Cpu {
                        10
                    } else {
                        accelerator
                    }))
                },
            )
        };
        assert_eq!(run(20).unwrap().1, Device::Cpu);
        let validated = checks.get();
        assert_eq!(
            run(1).unwrap().1,
            Device::Cpu,
            "reuse previously measured plan"
        );
        assert_eq!(checks.get(), validated, "idle reload repeated calibration");
        cache.lock().unwrap().clear();
        assert_eq!(run(1).unwrap().1, Device::Cuda);
    }

    #[test]
    fn a_failed_cached_plan_recalibrates_and_invalid_output_is_quarantined() {
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::from([(
            "fixture".into(),
            CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false),
        )]));
        let (_, device) = calibrated(
            "fixture",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            |device, _, cached| {
                if cached && device == Device::Cuda {
                    Err(IndexError::Engine("stale fixture target".into()))
                } else {
                    Ok(device)
                }
            },
            |_model, device| {
                if device == Device::Cuda {
                    Err(IndexError::Incompatible("invalid fixture output".into()))
                } else {
                    Ok(Duration::from_millis(10))
                }
            },
        )
        .unwrap();
        assert_eq!(device, Device::Cpu);
        assert_eq!(cache.lock().unwrap()["fixture"].device, Device::Cpu);
    }

    #[test]
    fn qualified_candidates_are_ordered_by_absolute_call_time() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let controls = Cell::new(0);
        let (_, device) = calibrated(
            "fixture",
            vec![
                (Device::Cuda, vec![cpu()].into()),
                (Device::DirectMl, vec![cpu()].into()),
            ],
            &cache,
            |device, _, _| Ok(device),
            |_, device| {
                let ms = match device {
                    Device::Cpu => {
                        let n = controls.get();
                        controls.set(n + 1);
                        if n < 2 { 10 } else { 20 }
                    }
                    Device::Cuda => 5,
                    _ => 6,
                };
                Ok(Duration::from_millis(ms))
            },
        )
        .unwrap();
        assert_eq!(
            device,
            Device::Cuda,
            "ratio must not choose slower DirectML"
        );
    }

    #[test]
    fn transient_fallback_is_retried_on_idle_reload() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let recovered = Cell::new(false);
        let run = || {
            calibrated(
                "fixture",
                vec![(Device::Cuda, vec![cpu()].into())],
                &cache,
                |device, _, _| {
                    if device == Device::Cuda && !recovered.get() {
                        Err(IndexError::Engine("temporary pressure".into()))
                    } else {
                        Ok(device)
                    }
                },
                |_, device| {
                    Ok(Duration::from_millis(if device == Device::Cpu {
                        10
                    } else {
                        1
                    }))
                },
            )
        };
        assert_eq!(run().unwrap().1, Device::Cpu);
        assert!(
            cache.lock().unwrap().is_empty(),
            "temporary CPU fallback was cached"
        );
        recovered.set(true);
        assert_eq!(run().unwrap().1, Device::Cuda);
    }

    #[test]
    fn numerical_quarantine_expires_before_an_idle_reload() {
        use std::time::{Duration, Instant};
        let cache = std::sync::Mutex::new(std::collections::HashMap::from([(
            "fixture".into(),
            CachedPlan {
                device: Device::Cpu,
                target: vec![cpu()].into(),
                revalidate: Some(Instant::now()),
                references: References::default(),
            },
        )]));
        let (_, device) = calibrated(
            "fixture",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            |device, _, _| Ok(device),
            |_, device| {
                Ok(Duration::from_millis(if device == Device::Cpu {
                    10
                } else {
                    1
                }))
            },
        )
        .unwrap();
        assert_eq!(
            device,
            Device::Cuda,
            "expired CPU quarantine blocked a repaired accelerator"
        );
    }

    #[test]
    fn cold_calibrations_do_not_load_competing_reference_sessions() {
        use std::cell::Cell;
        use std::sync::mpsc;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (loaded_tx, loaded_rx) = mpsc::channel();
        let cache = &cache;
        std::thread::scope(|scope| {
            let first = scope.spawn(move || {
                let announced = Cell::new(false);
                calibrated(
                    "first",
                    vec![(Device::Cuda, vec![cpu()].into())],
                    cache,
                    |device, _, _| Ok(device),
                    |_, _| {
                        if !announced.replace(true) {
                            started_tx.send(()).unwrap();
                            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                        }
                        Ok(Duration::from_millis(10))
                    },
                )
                .unwrap();
            });
            started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let second = scope.spawn(|| {
                ready_tx.send(()).unwrap();
                calibrated(
                    "second",
                    vec![(Device::Cuda, vec![cpu()].into())],
                    cache,
                    |device, _, _| {
                        loaded_tx.send(()).unwrap();
                        Ok(device)
                    },
                    |_, _| Ok(Duration::from_millis(10)),
                )
                .unwrap();
            });
            ready_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let competed = loaded_rx.recv_timeout(Duration::from_millis(100)).is_ok();
            release_tx.send(()).unwrap();
            first.join().unwrap();
            second.join().unwrap();
            assert!(
                !competed,
                "a second calibration loaded a reference while the first was timed"
            );
        });
    }

    #[test]
    fn the_returned_winner_instance_must_execute_successfully() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let loads = Cell::new(0);
        let (_, device) = calibrated(
            "fixture",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            |device, _, _| {
                if device == Device::Cuda {
                    loads.set(loads.get() + 1);
                }
                Ok((device, loads.get()))
            },
            |model, device| {
                if device == Device::Cuda && model.1 > 3 {
                    Err(IndexError::Numerical("replacement lazy failure".into()))
                } else {
                    Ok(Duration::from_millis(if device == Device::Cpu {
                        10
                    } else {
                        1
                    }))
                }
            },
        )
        .unwrap();
        assert_eq!(device, Device::Cpu, "unvalidated replacement was returned");
        assert!(cache.lock().unwrap().is_empty());
    }

    #[test]
    fn candidate_order_is_rotated_before_a_winner_is_cached() {
        use std::cell::Cell;
        use std::time::Duration;
        let cache = std::sync::Mutex::new(std::collections::HashMap::new());
        let cuda = Cell::new(0);
        let dml = Cell::new(0);
        let (_, device) = calibrated(
            "fixture",
            vec![
                (Device::Cuda, vec![cpu()].into()),
                (Device::DirectMl, vec![cpu()].into()),
            ],
            &cache,
            |device, _, _| Ok(device),
            |_, device| {
                let ms = match device {
                    Device::Cpu => 100,
                    Device::Cuda => {
                        cuda.set(cuda.get() + 1);
                        6
                    }
                    _ => {
                        let n = dml.get();
                        dml.set(n + 1);
                        if n == 0 { 1 } else { 9 }
                    }
                };
                Ok(Duration::from_millis(ms))
            },
        )
        .unwrap();
        assert_eq!(
            device,
            Device::Cuda,
            "one discovery-order fast interval decided the winner"
        );
        assert!(cuda.get() >= 3 && dml.get() >= 3);
    }

    #[test]
    fn a_cached_accelerator_session_reuses_cpu_reference_outputs() {
        use std::time::Duration;
        let saved = References {
            scores: Some(vec![1.0; 4]),
            ..Default::default()
        };
        let cache = std::sync::Mutex::new(std::collections::HashMap::from([(
            "fixture".into(),
            CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false).with_references(saved),
        )]));
        let references = std::cell::RefCell::default();
        let (_, device, _) = calibrated_with_references(
            "fixture",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            &references,
            PlanFiles::default(),
            |device, _, _| Ok(if device == Device::Cpu { 1.0 } else { 9.0 }),
            |actual, device| {
                let mut proof = references.borrow_mut();
                match &proof.scores {
                    None => proof.scores = Some(vec![*actual]),
                    Some(expected) if *actual == expected[0] => {}
                    _ => return Err(IndexError::Numerical("wrong replacement output".into())),
                };
                Ok(Duration::from_millis(if device == Device::Cpu {
                    10
                } else {
                    1
                }))
            },
        )
        .unwrap();
        assert_eq!(
            device,
            Device::Cpu,
            "cached target bypassed its stored CPU output proof"
        );
    }

    #[test]
    fn cached_model_load_does_not_wait_for_another_process_calibration() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "cached").unwrap();
        let held = file_lock(&path.parent().unwrap().join("calibration.lock")).unwrap();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let cache = std::sync::Mutex::new(std::collections::HashMap::from([(
                "cached".into(),
                CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false),
            )]));
            let result = calibrated_with_references(
                "cached",
                vec![(Device::Cuda, vec![cpu()].into())],
                &cache,
                &std::cell::RefCell::default(),
                PlanFiles::for_record(&path),
                |device, _, cached| {
                    assert!(cached);
                    Ok(device)
                },
                |_, _| Ok(Duration::from_millis(1)),
            );
            sent.send(result.map(|(_, device, _)| device)).unwrap();
        });
        let result = received.recv_timeout(Duration::from_secs(2));
        drop(held);
        worker.join().unwrap();
        assert_eq!(result.unwrap().unwrap(), Device::Cuda);
    }

    #[test]
    fn a_validated_ram_hit_republishes_a_retired_disk_record() {
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "retired").unwrap();
        let cache = std::sync::Mutex::new(std::collections::HashMap::from([(
            "retired".into(),
            CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false).with_references(References {
                scores: Some(vec![1.0; 4]),
                ..Default::default()
            }),
        )]));
        calibrated_with_references(
            "retired",
            vec![(Device::Cuda, vec![cpu()].into())],
            &cache,
            &std::cell::RefCell::default(),
            PlanFiles::for_record(&path),
            |_, _, cached| {
                assert!(cached);
                Ok(())
            },
            |_, _| unreachable!("cached CPU should not recalibrate"),
        )
        .unwrap();
        assert_eq!(
            read_plan(&path, "retired", &[]).unwrap().references.scores,
            Some(vec![1.0; 4])
        );
    }

    #[test]
    fn nonpersistent_plans_still_hold_the_host_calibration_transaction() {
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "opaque").unwrap();
        let dir = path.parent().unwrap();
        let mut checked = false;
        calibrated_with_references(
            "opaque",
            vec![(Device::Npu, vec![cpu()].into())],
            &std::sync::Mutex::default(),
            &std::cell::RefCell::default(),
            PlanFiles {
                record: None,
                directory: Some(dir),
            },
            |device, _, _| {
                if !checked {
                    let other = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open(dir.join("calibration.lock"))
                        .unwrap();
                    assert!(
                        other.try_lock().is_err(),
                        "disabled persistence also disabled calibration coordination"
                    );
                    checked = true;
                }
                Ok(device)
            },
            |_, device| {
                Ok(std::time::Duration::from_millis(if device == Device::Cpu {
                    10
                } else {
                    1
                }))
            },
        )
        .unwrap();
        assert!(!path.exists(), "opaque plan was persisted");
    }

    #[test]
    fn cold_calibration_publishes_before_releasing_the_host_lock() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "publication").unwrap();
        let cache = std::sync::Mutex::default();
        let plans = vec![(Device::Cuda, vec![cpu()].into())];
        let (_, device, _) = calibrated_with_references(
            "publication",
            plans.clone(),
            &cache,
            &std::cell::RefCell::default(),
            PlanFiles::for_record(&path),
            |device, _, _| Ok(device),
            |_, device| {
                Ok(Duration::from_millis(if device == Device::Cpu {
                    10
                } else {
                    1
                }))
            },
        )
        .unwrap();
        assert_eq!(device, Device::Cuda);
        // This is the real publication path, not manual publication in the test.
        let _host = file_lock(&path.parent().unwrap().join("calibration.lock")).unwrap();
        assert!(read_plan(&path, "publication", &plans).is_some());
        drop(_host);
        let (_, second, _) = calibrated_with_references(
            "publication",
            plans,
            &std::sync::Mutex::default(),
            &std::cell::RefCell::default(),
            PlanFiles::for_record(&path),
            |device, _, cached| {
                assert!(cached, "waiting process repeated calibration");
                Ok(device)
            },
            |_, _| Ok(Duration::from_millis(1)),
        )
        .unwrap();
        assert_eq!(second, Device::Cuda);
    }

    #[test]
    fn a_calibration_miss_rechecks_the_disk_plan_after_waiting() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "published").unwrap();
        let held = file_lock(&path.parent().unwrap().join("calibration.lock")).unwrap();
        let (started, ready) = std::sync::mpsc::channel();
        let worker_path = path.clone();
        let worker = std::thread::spawn(move || {
            started.send(()).unwrap();
            calibrated_with_references(
                "published",
                vec![(Device::Cuda, vec![cpu()].into())],
                &std::sync::Mutex::default(),
                &std::cell::RefCell::default(),
                PlanFiles::for_record(&worker_path),
                |device, _, cached| {
                    assert!(cached, "miss ignored the newly published plan");
                    Ok(device)
                },
                |_, _| Ok(Duration::from_millis(1)),
            )
            .unwrap()
            .1
        });
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        {
            let _lock = plan_metadata_lock(&path);
            write_plan(
                &path,
                "published",
                &CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false),
            )
            .unwrap();
        }
        drop(held);
        assert_eq!(worker.join().unwrap(), Device::Cuda);
    }

    #[test]
    fn plan_maintenance_retires_expired_records_and_bounds_storage() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("compute-plans-v1");
        std::fs::create_dir(&dir).unwrap();
        for i in 0..MAX_DISK_PLANS + 4 {
            let key = format!("fixture-{i}");
            let path = dir.join(format!("{}.json", fingerprint(&key)));
            write_plan(
                &path,
                &key,
                &CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false),
            )
            .unwrap();
        }
        let stale = dir.join(format!("{}.json", fingerprint("expired")));
        std::fs::write(
            &stale,
            serde_json::to_vec(&DiskPlan {
                fingerprint: fingerprint("expired"),
                target: "cpu".into(),
                expires: unix_seconds(),
                references: References::default(),
            })
            .unwrap(),
        )
        .unwrap();
        let unrelated = dir.join("leave-me.json");
        std::fs::write(&unrelated, b"not our record").unwrap();
        plan_file(root.path(), "requested").unwrap();
        assert!(!stale.exists());
        assert!(unrelated.exists());
        let remaining = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                let p = e.as_ref().unwrap().path();
                p.extension().is_some_and(|ext| ext == "json") && p != unrelated
            })
            .count();
        assert!(remaining <= MAX_DISK_PLANS);
        let locks: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let p = e.unwrap().path();
                p.extension().is_some_and(|ext| ext == "lock").then_some(p)
            })
            .collect();
        assert_eq!(locks, vec![dir.join("plans.lock")]);
        let cache = std::sync::Mutex::default();
        for i in 0..4 {
            let key = format!("queued-{i}");
            let path = dir.join(format!("{}.json", fingerprint(&key)));
            remember_plan(
                &key,
                Some(&path),
                &cache,
                CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false),
            );
        }
        let plans = std::fs::read_dir(&dir)
            .unwrap()
            .filter(|e| {
                let p = e.as_ref().unwrap().path();
                p.extension().is_some_and(|ext| ext == "json") && p != unrelated
            })
            .count();
        assert!(
            plans <= MAX_DISK_PLANS,
            "queued publications exceeded the bound"
        );
    }

    #[test]
    fn retention_preserves_a_new_short_quarantine_at_capacity() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("compute-plans-v1");
        std::fs::create_dir(&dir).unwrap();
        for i in 0..MAX_DISK_PLANS {
            let key = format!("positive-{i}");
            write_plan(
                &dir.join(format!("{}.json", fingerprint(&key))),
                &key,
                &CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), false),
            )
            .unwrap();
        }
        let path = dir.join(format!("{}.json", fingerprint("quarantine")));
        remember_plan(
            "quarantine",
            Some(&path),
            &std::sync::Mutex::default(),
            CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), true),
        );
        assert!(read_plan(&path, "quarantine", &[]).is_some());
        assert_eq!(
            std::fs::read_dir(dir)
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "json"))
                .count(),
            MAX_DISK_PLANS
        );
    }

    #[test]
    fn abandoned_partials_are_retired_but_live_legacy_writes_are_preserved() {
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "partial-fixture").unwrap();
        let dir = path.parent().unwrap();
        let abandoned = path.with_extension(format!("{}.partial", uuid::Uuid::now_v7()));
        let other = dir.join("unrelated.partial");
        std::fs::write(&abandoned, b"interrupted").unwrap();
        std::fs::write(&other, b"unrelated").unwrap();
        let legacy = file_lock(&path.with_extension("lock")).unwrap();
        {
            let _lock = plan_metadata_lock(&path);
            prune_plans(dir, None).unwrap();
        }
        assert!(
            abandoned.exists(),
            "deleted an in-flight legacy publication"
        );
        drop(legacy);
        {
            let _lock = plan_metadata_lock(&path);
            prune_plans(dir, None).unwrap();
        }
        assert!(!abandoned.exists());
        assert!(other.exists());
    }

    #[test]
    fn opaque_adapter_ids_do_not_enable_cross_process_plan_reuse() {
        for device in [Device::DirectMl, Device::Npu] {
            assert!(!persistence_supported(
                &[(device, vec![cpu()].into())],
                true
            ));
        }
        assert!(!persistence_supported(
            &[(Device::Cuda, vec![cpu()].into())],
            false
        ));
        assert!(persistence_supported(
            &[(Device::Cuda, vec![cpu()].into())],
            true
        ));
        assert!(persistence_supported(
            &[(Device::CoreMl, vec![cpu()].into())],
            false
        ));
    }

    #[cfg(unix)]
    #[test]
    fn inventory_probe_bounds_stalls_and_output_and_reaps_children() {
        use std::time::Duration;
        let mut success = std::process::Command::new("sh");
        success.args(["-c", "printf '0, GPU-test, pci, name, driver\\n'"]);
        let bytes = bounded_inventory(success, Duration::from_secs(1)).unwrap();
        assert!(cuda_inventory(&bytes).is_some());
        let mut failure = std::process::Command::new("sh");
        failure.args(["-c", "exit 1"]);
        assert!(bounded_inventory(failure, Duration::from_secs(1)).is_none());
        let mut stalled = std::process::Command::new("sh");
        stalled.args(["-c", "exec sleep 30"]);
        let start = std::time::Instant::now();
        assert!(bounded_inventory(stalled, Duration::from_millis(50)).is_none());
        assert!(start.elapsed() < Duration::from_secs(2));
        let mut noisy = std::process::Command::new("sh");
        noisy.args(["-c", "while :; do printf 'too much inventory'; done"]);
        assert!(bounded_inventory(noisy, Duration::from_secs(1)).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn inventory_admission_remains_held_until_cleanup_releases_it() {
        use std::sync::{Arc, atomic::AtomicBool};
        let busy = Arc::new(AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        let (ready, started) = std::sync::mpsc::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _permit = permit;
            ready.send(()).unwrap();
            wait.recv().unwrap();
        });
        started.recv().unwrap();
        for _ in 0..100 {
            assert!(InventoryPermit::acquire(busy.clone()).is_none());
        }
        release.send(()).unwrap();
        worker.join().unwrap();
        assert!(InventoryPermit::acquire(busy).is_some());
    }

    #[cfg(unix)]
    #[test]
    fn stalled_inventory_child_has_exited_before_admission_is_released() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let pid_file = root.path().join("child.pid");
        let mut command = std::process::Command::new("sh");
        command.args(["-c", "echo $$ > \"$1\"; exec sleep 30", "inventory"]);
        command.arg(&pid_file);
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        assert!(inventory_with_permit(command, Duration::from_millis(100), permit).is_none());
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let _released = loop {
            if let Some(permit) = InventoryPermit::acquire(busy.clone()) {
                break permit;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cleanup did not release admission"
            );
            std::thread::yield_now();
        };
        let pid = std::fs::read_to_string(pid_file).unwrap();
        // kill -0 distinguishes an exited/reaped process from a still-live or
        // zombie child; it neither signals nor changes the child.
        assert!(
            !std::process::Command::new("kill")
                .args(["-0", pid.trim()])
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap()
                .success()
        );
        assert!(InventoryPermit::acquire(busy).is_none());
    }

    #[test]
    fn cuda_inventory_requires_device_ids_and_distinguishes_devices() {
        let first = cuda_inventory(b"0, GPU-a, 0000:01:00.0, A100, 580.0\n").unwrap();
        let second = cuda_inventory(b"0, GPU-b, 0000:02:00.0, A100, 580.0\n").unwrap();
        assert_ne!(fingerprint(&first), fingerprint(&second));
        assert!(cuda_inventory(b"").is_none());
        assert!(cuda_inventory(b"0, N/A, 0000:01:00.0, A100, 580.0").is_none());
        assert!(cuda_inventory(b"NVIDIA-SMI has failed").is_none());
    }

    #[test]
    fn persisted_plans_reuse_proofs_and_reject_wrong_identity_or_expiry() {
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "fixture host runtime model").unwrap();
        let proof = References {
            scores: Some(vec![1.0, 2.0, 3.0, 4.0]),
            ..Default::default()
        };
        let plan =
            CachedPlan::fresh(Device::Cuda, vec![cpu()].into(), false).with_references(proof);
        write_plan(&path, "fixture host runtime model", &plan).unwrap();
        let candidates = vec![(Device::Cuda, vec![cpu()].into())];
        let loaded = read_plan(&path, "fixture host runtime model", &candidates).unwrap();
        assert_eq!(loaded.device, Device::Cuda);
        assert_eq!(loaded.references.scores, Some(vec![1.0, 2.0, 3.0, 4.0]));
        assert!(read_plan(&path, "different runtime", &candidates).is_none());
        assert!(read_plan(&path, "fixture host runtime model", &[]).is_none());
        let mut record: DiskPlan = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        record.expires = unix_seconds();
        std::fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(read_plan(&path, "fixture host runtime model", &candidates).is_none());
        std::fs::write(&path, b"damaged").unwrap();
        assert!(read_plan(&path, "fixture host runtime model", &candidates).is_none());
    }

    #[test]
    fn persisted_numerical_quarantine_has_a_bounded_lifetime() {
        let root = tempfile::tempdir().unwrap();
        let path = plan_file(root.path(), "fixture").unwrap();
        write_plan(
            &path,
            "fixture",
            &CachedPlan::fresh(Device::Cpu, vec![cpu()].into(), true),
        )
        .unwrap();
        let record: DiskPlan = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert!(record.expires > unix_seconds());
        assert!(record.expires <= unix_seconds() + 300);
    }

    #[test]
    fn absent_plugin_npu_fails_before_model_fetch() {
        let fetched = std::cell::Cell::new(false);
        let result = session(
            Target::Npu {
                provider: "absent-test-provider".into(),
                id: u32::MAX,
            },
            || {
                fetched.set(true);
                Ok("unused.onnx".into())
            },
        );
        assert!(result.is_err());
        assert!(!fetched.get());
    }

    #[test]
    fn failed_cache_retry_preserves_the_published_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().join("cache");
        std::fs::create_dir(&cache).unwrap();
        std::fs::write(cache.join("ready"), b"published").unwrap();
        let result: Result<()> = rebuild_coreml_cache(&cache, || {
            std::fs::write(cache.join("partial"), b"failed replacement")?;
            Err(IndexError::Engine("transient provider failure".into()))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(cache.join("ready")).unwrap(), b"published");
        assert!(!cache.join("partial").exists());
        assert!(!cache.with_extension("previous").exists());
        std::fs::rename(&cache, cache.with_extension("previous")).unwrap();
        std::fs::create_dir(&cache).unwrap();
        std::fs::write(cache.join("partial"), b"interrupted retry").unwrap();
        recover_coreml_cache(&cache).unwrap();
        assert_eq!(std::fs::read(cache.join("ready")).unwrap(), b"published");
        rebuild_coreml_cache(&cache, || {
            std::fs::write(cache.join("replacement"), b"loaded")?;
            Ok(())
        })
        .unwrap();
        assert!(!cache.with_extension("previous").exists());
        assert_eq!(std::fs::read(cache.join("replacement")).unwrap(), b"loaded");
    }
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires the native CoreML execution provider"]
    fn modern_coreml_assigns_normalization_that_legacy_format_cannot() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("normalization.onnx");
        std::fs::write(
            &path,
            include_bytes!("../tests/fixtures/coreml-normalization.onnx"),
        )?;
        let mut modern = session(vec![coreml().build().error_on_failure()], || {
            Ok(path.clone())
        })?;
        let assigned = assigned_providers(&modern)?;
        assert!(
            assigned
                .get("CoreMLExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        let legacy = session(
            vec![
                ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::NeuralNetwork)
                    .build()
                    .error_on_failure(),
            ],
            || Ok(path.clone()),
        )?;
        let old = assigned_providers(&legacy)?;
        assert_eq!(old.get("CoreMLExecutionProvider").copied().unwrap_or(0), 0);
        assert!(
            old.get("CPUExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        let input = ort::value::Tensor::from_array(([1, 4], vec![1.0f32, 2.0, 3.0, 4.0]))
            .map_err(|error| IndexError::Engine(error.to_string()))?;
        let output = modern
            .run(ort::inputs!["x" => input])
            .map_err(|error| IndexError::Engine(error.to_string()))?;
        let (_, values) = output["y"]
            .try_extract_tensor::<f32>()
            .map_err(|error| IndexError::Engine(error.to_string()))?;
        let expected = [-1.5f32, -0.5, 0.5, 1.5].map(|x| x / 1.25001f32.sqrt());
        assert_eq!(values.len(), expected.len());
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(value, expected)| (*value - expected).abs() < 1e-3)
        );
        Ok(())
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
            let registers = options(provider.clone()).is_ok();
            if registers {
                continue;
            }
            let asked = std::cell::Cell::new(false);
            let loaded = session(provider, || {
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
