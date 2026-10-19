//! A tokenizer and an ONNX Runtime session, and one forward pass through both.
//!
//! What the reranker and BGE-M3 used `fastembed` for, taken in here so that
//! their tokenizers can share a vocabulary (see `crate::tokenizer`) -- its
//! types own theirs and have no constructor that takes one. What each model
//! does with the outputs stays with the model: the reranker reads a logit per
//! pair, the embedder a vector per text.
//!
//! The inputs are built the way `fastembed` 6.1 built them, because they are
//! the scores: a batch encoded with special tokens and padded to its longest
//! member, its ids and attention mask fed as `i64` in the batch's shape, and
//! token type ids only to a graph that declares an input of that name. A
//! prepared copy's split graph (see `crate::word_embeddings`) also takes the
//! batch's word embeddings, which are read here for its ids.

use std::borrow::Cow;
use std::ffi::CString;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use ort::session::{Session, SessionInputValue, SessionOutputs};
use ort::value::{DynValue, Tensor};
use tokenizers::{EncodeInput, Encoding};

use crate::error::{IndexError, Result};
use crate::hub::Repository;
use crate::tokenizer::Tokenizer;
use crate::word_embeddings::WordTable;

pub(crate) struct Encoder {
    runtime_plan: crate::inference::RuntimePlan,
    tokenizer: Tokenizer,
    session: Session,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fixed: Option<(Session, Session)>,
    /// Whether the graph takes token type ids. XLM-R's family ignores them and
    /// most of its exports do not declare the input.
    token_type_ids: bool,
    /// Where a split graph's word embeddings are read from, for the input that
    /// takes them; `None` for a graph that gathers its own. Shared by both
    /// sessions, which load the same graph.
    word_embeddings: Option<Arc<WordTable>>,
    /// The graph [`run_each`](Self::run_each) loads its single-threaded
    /// session from, and that session once it has been loaded; `None` runs
    /// every text on [`session`](Self::session); and how many passes it runs
    /// at once.
    side_by_side: Option<(PathBuf, Option<SharedCpuSession>, usize)>,
    /// The most passes [`run_each`](Self::run_each) has had running at once.
    #[cfg(test)]
    most_at_once: usize,
    /// The repository and commit the model was read from, as
    /// [`Repository::revision`] resolves it once its files are fetched.
    revision: String,
}

/// A forward pass's outputs, by name, in the graph's order.
pub(crate) type Outputs = Vec<(String, DynValue)>;

impl crate::inference::RuntimeModel for Encoder {
    fn runtime_plan(&self) -> &crate::inference::RuntimePlan {
        &self.runtime_plan
    }
    fn runtime_plan_mut(&mut self) -> &mut crate::inference::RuntimePlan {
        &mut self.runtime_plan
    }
    fn revision(&self) -> &str {
        &self.revision
    }
}

impl Encoder {
    /// The model `model` finds, on `providers`, with the tokenizer of the
    /// repository it came from, truncating at `max_length` tokens.
    ///
    /// The session first and the tokenizer second, as `fastembed` loaded them,
    /// so a device that will not take the model is found before a tokenizer is
    /// read for nothing -- and, since `model` is asked for only once the
    /// device has registered, before its export is fetched for nothing.
    pub(crate) fn load(
        threads: usize,
        model: impl FnOnce() -> Result<PathBuf>,
        repository: &Repository,
        max_length: usize,
        providers: impl Into<crate::inference::Target>,
    ) -> Result<Self> {
        let session = crate::inference::session(threads, providers, model)?;
        Self::from_session(session, repository, max_length)
    }

    pub(crate) fn from_session(
        session: Session,
        repository: &Repository,
        max_length: usize,
    ) -> Result<Self> {
        let tokenizer = crate::tokenizer::load(repository, max_length)?;
        Ok(Self::from_parts(session, tokenizer, repository.revision()))
    }

    fn from_parts(session: Session, tokenizer: Tokenizer, revision: String) -> Self {
        let token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Self {
            runtime_plan: crate::inference::RuntimePlan::default(),
            tokenizer,
            session,
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            fixed: None,
            token_type_ids,
            word_embeddings: None,
            side_by_side: None,
            #[cfg(test)]
            most_at_once: 0,
            revision,
        }
    }

    /// The model on the CPU over a session of `threads` intra-op threads, with
    /// [`run_each`](Self::run_each) running several texts at once on one
    /// single-threaded session over the same `model` -- a mapped copy (see
    /// `crate::prepared`). `revision` is the [`Repository::revision`] `model`
    /// and `tokenizer` were read from.
    ///
    /// A split graph's word embeddings are read from the table described
    /// beside it, which is opened here once for both sessions. A description
    /// naming an input the graph does not take is refused here, rather than
    /// by the runtime on every pass that fed it.
    pub(crate) fn side_by_side(
        model: PathBuf,
        threads: usize,
        tokenizer: Tokenizer,
        revision: String,
    ) -> Result<Self> {
        let session = crate::inference::cpu_session(&model, threads)?;
        let word_embeddings = WordTable::beside(&model)?.map(Arc::new);
        if let Some(table) = &word_embeddings
            && !session
                .inputs()
                .iter()
                .any(|input| input.name() == table.input())
        {
            return Err(failed(&format!(
                "{} takes no input named {}, which its word-embedding table is described for",
                model.display(),
                table.input()
            )));
        }
        Ok(Self {
            word_embeddings,
            side_by_side: Some((model, None, threads)),
            ..Self::from_parts(session, tokenizer, revision)
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn load_fixed_coreml(
        threads: usize,
        model: impl FnOnce() -> Result<PathBuf>,
        repository: &Repository,
        max_length: usize,
    ) -> Result<Self> {
        if max_length > 256 {
            return Err(failed(
                &"static CoreML reranker currently requires at most 256 tokens",
            ));
        }
        let path = crate::inference::coreml_source(threads, model)?;
        // Validate every independently compiled shape inside `preferred`'s
        // fallback window. A lazy shape failure otherwise breaks later searches
        // after the model has already been accepted as CoreML.
        let (session, short, long) = coreml_buckets(|rows, tokens| {
            crate::inference::fixed_coreml(threads, || Ok(path.clone()), rows, tokens)
                .map(|loaded| loaded.0)
        })?;
        let tokenizer = crate::tokenizer::load(repository, max_length)?;
        Ok(Self {
            runtime_plan: crate::inference::RuntimePlan::default(),
            tokenizer,
            session,
            fixed: Some((short, long)),
            token_type_ids: false,
            word_embeddings: None,
            side_by_side: None,
            #[cfg(test)]
            most_at_once: 0,
            revision: repository.revision(),
        })
    }

    /// Registration without assigned nodes is a CPU fallback, not acceleration.
    pub(crate) fn require_accelerator(&self, device: crate::inference::Device) -> Result<()> {
        use crate::inference::Device;
        let provider = match device {
            // The explicit NPU target is checked by inference::session.
            Device::Cpu | Device::Npu => return Ok(()),
            Device::Cuda => "CUDAExecutionProvider",
            Device::CoreMl => "CoreMLExecutionProvider",
            Device::DirectMl => "DmlExecutionProvider",
        };
        let assigned = crate::inference::assigned_providers(&self.session)?;
        if assigned.get(provider).copied().unwrap_or(0) == 0 {
            return Err(crate::error::IndexError::Engine(format!(
                "{provider} registered but was assigned no model nodes"
            )));
        }
        Ok(())
    }

    /// Whether [`run_each`](Self::run_each) runs passes side by side: the CPU.
    pub(crate) fn runs_side_by_side(&self) -> bool {
        self.side_by_side.is_some()
    }

    pub(crate) fn batch_limits(&self, budget: usize, most: usize) -> (usize, usize) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.fixed.is_some() {
            // Native buckets execute at most 512 tokens; larger sweep settings
            // cannot enlarge a compiled session's physical shape.
            return (budget.min(512), most.min(4));
        }
        (budget, most)
    }

    pub(crate) fn batching_length(&self, length: usize) -> usize {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.fixed.is_some() {
            return native_shape(length).1;
        }
        length
    }

    pub(crate) fn execution_shape(&self, rows: usize, length: usize) -> (usize, usize) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if self.fixed.is_some() {
            return native_shape(length);
        }
        (rows, length)
    }

    /// The actual pair truncation limit, after the model's own cap is applied.
    pub(crate) fn maximum_tokens(&self) -> usize {
        self.tokenizer
            .get_truncation()
            .expect("loaded encoders configure truncation")
            .max_length
    }

    /// How many passes [`run_each`](Self::run_each) runs at once for
    /// `batches` batches, with the shared session it runs them on loaded: 1
    /// when it runs them one after another on [`session`](Self::session).
    ///
    /// `run_each` calls this itself; a caller that times its passes calls it
    /// first, so that loading the session is not counted as running them.
    pub(crate) fn ready(&mut self, batches: usize) -> Result<usize> {
        let Some((model, shared, budget)) = &mut self.side_by_side else {
            return Ok(1);
        };
        let wanted = (*budget).min(batches);
        // One pass at a time gains nothing on a session of its own.
        if wanted < 2 {
            return Ok(1);
        }
        if shared.is_none() {
            // The shared session only saves time, and `session` still runs
            // every batch: one that will not load leaves the passes to it,
            // and stops passes being planned for running side by side.
            match SharedCpuSession::load(model) {
                Ok(session) => *shared = Some(session),
                Err(error) => {
                    tracing::warn!(%error, "could not load the side-by-side session");
                    self.side_by_side = None;
                    return Ok(1);
                }
            }
        }
        Ok(wanted)
    }

    /// Each of `batches`, from [`encode`](Self::encode), in a forward pass of
    /// its own as [`run_encoded`](Self::run_encoded) runs it, its outputs
    /// taken by `read`, in `batches`' order.
    ///
    /// On [`session`](Self::session) one after another, unless the model was
    /// loaded [`side_by_side`](Self::side_by_side) and there is more than one
    /// batch and more than one thread in the inference budget: then as many
    /// passes at once as the budget allows, all on one session of one
    /// intra-op thread, each thread taking the next batch when it finishes
    /// one. A pass hands each of the graph's nodes to its threads and waits
    /// for them, and on a short input that hand-off is most of the pass;
    /// passes side by side pay none of it. Measured through `Embedder` on four
    /// cores (ONNX Runtime 1.28, Xeon with AMX), 500 texts stored 64 at a
    /// time, two runs each, when each pass still had a session of its own:
    /// BGE-M3 from 38.3 and 40.3 s to 18.7 and 17.3 s, every vector
    /// bit-identical to one thread's.
    ///
    /// One session for every pass rather than one a pass, so the graph, its
    /// kernels and its weights are loaded once whatever the budget; see
    /// [`run_shared`] for how a session is run from several threads at once.
    /// The session is loaded the first time it is wanted, so a model that only
    /// ever answers queries loads none.
    pub(crate) fn run_each<T: Send>(
        &mut self,
        batches: Vec<Vec<Encoding>>,
        read: impl Fn(Outputs) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        let workers = self.ready(batches.len())?;
        let session = match &self.side_by_side {
            Some((_, Some(session), _)) if workers > 1 => session,
            _ => {
                return batches
                    .into_iter()
                    .map(|batch| read(owned(self.run_encoded(batch)?)))
                    .collect();
            }
        };
        let (tokenizer, token_type_ids, word_embeddings) = (
            &self.tokenizer,
            self.token_type_ids,
            self.word_embeddings.as_deref(),
        );
        // Longest pass first. The passes share one queue, so the last to start
        // decides when the call returns; started last, the longest pass leaves
        // the other workers idle while it runs. Results are put back in the
        // caller's order below, and each pass is computed alone either way.
        let mut order: Vec<(usize, Vec<Encoding>)> = batches.into_iter().enumerate().collect();
        order.sort_by_key(|(at, batch)| (std::cmp::Reverse(padded_tokens(batch)), *at));
        let pending = &std::sync::Mutex::new(order.into_iter());
        let read = &read;
        #[cfg(test)]
        let (running, most) = (&AtomicUsize::new(0), &AtomicUsize::new(0));
        let mut done: Vec<(usize, Result<T>)> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(move || {
                        let mut done = Vec::new();
                        loop {
                            let next = pending.lock().expect("a pass panicked").next();
                            let Some((at, mut batch)) = next else {
                                return done;
                            };
                            let outputs = pad(tokenizer, &mut batch).and_then(|()| {
                                let shape = [batch.len(), batch[0].len()];
                                let feed = feed(&batch, shape, token_type_ids, word_embeddings)?;
                                #[cfg(test)]
                                most.fetch_max(running.fetch_add(1, SeqCst) + 1, SeqCst);
                                let outputs = run_shared(session, &feed);
                                #[cfg(test)]
                                running.fetch_sub(1, SeqCst);
                                outputs
                            });
                            done.push((at, outputs.and_then(read)));
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| {
                    worker
                        .join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        });
        #[cfg(test)]
        {
            self.most_at_once = self.most_at_once.max(most.load(SeqCst));
        }
        done.sort_unstable_by_key(|(at, _)| *at);
        done.into_iter().map(|(_, result)| result).collect()
    }

    /// How many sessions [`run_each`](Self::run_each) has loaded beside
    /// [`session`](Self::session), and the most passes it has had running on
    /// them at one moment.
    #[cfg(test)]
    pub(crate) fn side_by_side_passes(&self) -> (usize, usize) {
        let loaded = self
            .side_by_side
            .as_ref()
            .map_or(0, |(_, shared, _)| usize::from(shared.is_some()));
        (loaded, self.most_at_once)
    }

    /// Each of `inputs` tokenized on its own: truncated and given its special
    /// tokens, but not padded, so its length is its own.
    ///
    /// For a caller that groups inputs by length before running them, and so
    /// has to know the lengths first. [`run_encoded`](Self::run_encoded) pads
    /// a group of these exactly as `encode_batch` pads what it tokenizes:
    /// `encode_batch` is `encode` on each input and then this padding, so the
    /// model is handed the same ids either way.
    pub(crate) fn encode<'s, E>(&self, inputs: Vec<E>) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>>,
    {
        inputs
            .into_iter()
            .map(|input| {
                self.tokenizer
                    .encode(input, true)
                    .map_err(|error| failed(&error))
            })
            .collect()
    }

    /// One forward pass over `batch`, from [`encode`](Self::encode), padded to
    /// its longest member.
    pub(crate) fn run_encoded(&mut self, batch: Vec<Encoding>) -> Result<SessionOutputs<'_>> {
        self.run_padded(batch).map(|(_, outputs)| outputs)
    }

    /// The exact padded encodings sent to the session, for a model that pools
    /// token vectors using its attention mask.
    pub(crate) fn run_padded(
        &mut self,
        mut batch: Vec<Encoding>,
    ) -> Result<(Vec<Encoding>, SessionOutputs<'_>)> {
        pad(&self.tokenizer, &mut batch)?;
        let outputs = self.forward(&batch)?;
        Ok((batch, outputs))
    }

    fn forward(&mut self, encodings: &[Encoding]) -> Result<SessionOutputs<'_>> {
        let length = encodings
            .first()
            .ok_or_else(|| failed(&"nothing to encode"))?
            .len();
        let logical_rows = encodings.len();
        let (rows, tokens) = self.execution_shape(logical_rows, length);
        if logical_rows > rows || length > tokens {
            return Err(failed(&"batch exceeds the static execution shape"));
        }
        let feed = feed(
            encodings,
            [rows, tokens],
            self.token_type_ids,
            self.word_embeddings.as_deref(),
        )?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let session = if let Some((short, long)) = &mut self.fixed {
            match tokens {
                64 => short,
                128 => &mut self.session,
                _ => long,
            }
        } else {
            &mut self.session
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let session = &mut self.session;
        let outputs = session.run(feed).map_err(|error| failed(&error))?;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let mut outputs = outputs;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if rows != logical_rows {
            let output = outputs
                .get_mut("logits")
                .ok_or_else(|| failed(&"static reranker returned no logits"))?;
            let (shape, values) = output
                .try_extract_tensor::<f32>()
                .map_err(|error| failed(&error))?;
            if shape.as_ref() != [rows as i64, 1] {
                return Err(failed(&"static reranker returned an unexpected shape"));
            }
            *output = Tensor::from_array(([logical_rows, 1], values[..logical_rows].to_vec()))
                .map_err(|error| failed(&error))?
                .into_dyn();
        }
        Ok(outputs)
    }
}

/// What a session is fed for `encodings` in the execution shape `shape`.
///
/// With `word_embeddings`, a split graph's: the row of every id, padding
/// included, in a `[batch, sequence, width]` tensor that owns the rows as
/// they were read, with no copy between.
fn feed(
    encodings: &[Encoding],
    shape: [usize; 2],
    token_type_ids: bool,
    word_embeddings: Option<&WordTable>,
) -> Result<Vec<(Cow<'static, str>, SessionInputValue<'static>)>> {
    let column = |field: fn(&Encoding) -> &[u32], padding| {
        let values = input_values(encodings, shape, field, padding);
        Tensor::from_array((shape, values)).map_err(|error| failed(&error))
    };

    let ids = input_values(encodings, shape, Encoding::get_ids, 1);
    let embedded = word_embeddings
        .map(|table| table.embed(&ids).map(|rows| (table, rows)))
        .transpose()?;
    let mut feed = ort::inputs![
        "input_ids" => Tensor::from_array((shape, ids)).map_err(|error| failed(&error))?,
        "attention_mask" => column(Encoding::get_attention_mask, 0)?,
    ];
    if token_type_ids {
        feed.push((
            "token_type_ids".into(),
            column(Encoding::get_type_ids, 0)?.into(),
        ));
    }
    if let Some((table, rows)) = embedded {
        let [batch, sequence] = shape;
        feed.push((
            table.input().to_string().into(),
            Tensor::from_array(([batch, sequence, table.width()], rows))
                .map_err(|error| failed(&error))?
                .into(),
        ));
    }
    Ok(feed)
}

/// Fill only the model inputs. Static CoreML rows duplicate the last logical
/// row, as before, without cloning token text, offsets, or word mappings.
fn input_values(
    encodings: &[Encoding],
    [rows, tokens]: [usize; 2],
    field: fn(&Encoding) -> &[u32],
    padding: i64,
) -> Vec<i64> {
    let mut values = Vec::with_capacity(rows * tokens);
    for row in 0..rows {
        let encoding = &encodings[row.min(encodings.len() - 1)];
        values.extend(field(encoding).iter().copied().map(i64::from));
        values.resize((row + 1) * tokens, padding);
    }
    values
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn coreml_buckets<T>(mut load: impl FnMut(usize, usize) -> Result<T>) -> Result<(T, T, T)> {
    Ok((load(4, 128)?, load(4, 64)?, load(2, 256)?))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn native_shape(length: usize) -> (usize, usize) {
    if length <= 64 {
        (4, 64)
    } else if length <= 128 {
        (4, 128)
    } else {
        (2, 256)
    }
}

/// The one session [`run_shared`] runs from several threads at once: loaded
/// on ONNX Runtime's CPU execution provider alone, with every node of the
/// graph assigned to it.
///
/// Concurrent `Run` is safe on some execution providers and not on others,
/// so this type is the only way onto the concurrent path, and it admits the
/// CPU alone. ONNX Runtime documents `Run` as callable from several threads
/// on one session (`InferenceSession::Run` in
/// `onnxruntime/core/session/inference_session.h`), and an execution provider
/// that cannot take that says so through `ConcurrentRunSupported`, for which
/// ONNX Runtime then serializes `Run` itself (XNNPACK, WebGPU, WebNN, JS in
/// 1.28). DirectML does not declare it and documents that only one thread may
/// call `Run` at a time; CUDA and TensorRT have concurrency reports against
/// them. None of those may reach this type: [`load`](Self::load) builds the
/// session from the CPU provider alone and refuses it unless ONNX Runtime
/// assigned every node to that provider.
struct SharedCpuSession(Session);

impl SharedCpuSession {
    fn load(model: &std::path::Path) -> Result<Self> {
        let session = crate::inference::cpu_session(model, 1)?;
        cpu_alone(&crate::inference::assigned_providers(&session)?)?;
        Ok(Self(session))
    }
}

/// Whether `assigned`, a session's node count by execution provider, puts
/// every node on the CPU provider. An empty assignment is refused too: it
/// means ONNX Runtime recorded none, not that there was nothing to assign.
fn cpu_alone(assigned: &std::collections::BTreeMap<String, usize>) -> Result<()> {
    if assigned.is_empty() {
        return Err(failed(&"a shared session recorded no provider assignment"));
    }
    match assigned
        .iter()
        .find(|(provider, nodes)| *provider != "CPUExecutionProvider" && **nodes > 0)
    {
        Some((provider, _)) => Err(failed(&format!(
            "a shared session assigned nodes to {provider}, not the CPU alone"
        ))),
        None => Ok(()),
    }
}

/// One forward pass over `feed` on `session`, which other threads may be
/// running at the same moment, as [`Session::run`] would make it.
///
/// `ort` runs a session only through `&mut`: its author does not trust ONNX
/// Runtime's `Run` from several threads, naming "certain EP allocators or
/// statistics trackers" in `ort`'s source, and recommends a session per
/// thread. Here the session is a [`SharedCpuSession`] -- the CPU provider
/// alone, its arena off, no profiling -- of one intra-op thread, so that every
/// pass runs on the thread that asked for it. So this is `Run` itself, through
/// ONNX Runtime's C API, called on a shared session -- the call the `ort`
/// crate makes, without the exclusive borrow it puts in front of it.
fn run_shared(
    session: &SharedCpuSession,
    feed: &[(Cow<'static, str>, SessionInputValue<'static>)],
) -> Result<Outputs> {
    use ort::AsPointer;
    let session = &session.0;
    let names = |names: &mut dyn Iterator<Item = &str>| -> Result<Vec<CString>> {
        names
            .map(|name| CString::new(name).map_err(|error| failed(&error)))
            .collect()
    };
    let inputs = names(&mut feed.iter().map(|(name, _)| name.as_ref()))?;
    let outputs = names(&mut session.outputs().iter().map(|output| output.name()))?;
    let input_names: Vec<_> = inputs.iter().map(|name| name.as_ptr()).collect();
    let output_names: Vec<_> = outputs.iter().map(|name| name.as_ptr()).collect();
    let values: Vec<_> = feed.iter().map(|(_, value)| value.ptr()).collect();
    let mut fetched = vec![std::ptr::null_mut(); outputs.len()];
    // SAFETY: the session is live for the call, and ONNX Runtime allows `Run`
    // on it from several threads at once on the CPU provider, the only one a
    // `SharedCpuSession` admits (see above). Every name is a
    // NUL-terminated string and every input a live value, all outliving the
    // call; `fetched` has a slot for each output name, which `Run` fills.
    let status = unsafe {
        (ort::api().Run)(
            session.ptr().cast_mut(),
            std::ptr::null(),
            input_names.as_ptr(),
            values.as_ptr(),
            values.len(),
            output_names.as_ptr(),
            output_names.len(),
            fetched.as_mut_ptr(),
        )
    };
    // SAFETY: the status comes straight from ONNX Runtime and is read once.
    unsafe { ort::Error::result_from_status(status) }.map_err(|error| failed(&error))?;
    let owner = session.inner();
    outputs
        .into_iter()
        .zip(fetched)
        .map(|(name, value)| {
            let value = NonNull::new(value).ok_or_else(|| failed(&"an output came back empty"))?;
            // SAFETY: `Run` succeeded, so `value` is an output it allocated
            // and this pass now owns; `owner` keeps the session alive for as
            // long as the value is.
            let value = unsafe { DynValue::from_ptr(value, Some(std::sync::Arc::clone(&owner))) };
            Ok((name.to_string_lossy().into_owned(), value))
        })
        .collect()
}

/// `outputs` as [`run_shared`] returns them.
fn owned(outputs: SessionOutputs<'_>) -> Outputs {
    outputs
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
}

fn failed(error: &dyn std::fmt::Display) -> IndexError {
    IndexError::Engine(format!("running the model: {error}"))
}

/// Pads `batch` the way the tokenizer's own `encode_batch` pads what it
/// tokenizes: to the longest member, with the tokenizer's padding settings.
fn pad(tokenizer: &Tokenizer, batch: &mut [Encoding]) -> Result<()> {
    if let Some(params) = tokenizer.get_padding() {
        tokenizers::pad_encodings(batch, params).map_err(|error| failed(&error))?;
    }
    Ok(())
}

/// The tokens `batch` runs once [`pad`]ded: its members times the longest.
fn padded_tokens(batch: &[Encoding]) -> usize {
    batch.len() * batch.iter().map(Encoding::len).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {

    #[test]
    fn tensor_padding_matches_full_encoding_padding() {
        let tokenizer = crate::tokenizer::tests::fixture();
        for (count, rows, tokens) in [(1, 4, 64), (3, 4, 128), (2, 2, 256), (4, 4, 8)] {
            let mut logical = tokenizer
                .encode_batch(vec![("memory", "memory memory"); count], true)
                .unwrap();
            pad(&tokenizer, &mut logical).unwrap();
            let mut old = logical.clone();
            for encoding in &mut old {
                encoding.pad(tokens, 1, 0, "<pad>", tokenizers::PaddingDirection::Right);
            }
            while old.len() < rows {
                old.push(old.last().unwrap().clone());
            }
            for (field, padding) in [
                (Encoding::get_ids as fn(&Encoding) -> &[u32], 1),
                (Encoding::get_attention_mask, 0),
                (Encoding::get_type_ids, 0),
            ] {
                let expected: Vec<i64> = old
                    .iter()
                    .flat_map(|encoding| field(encoding).iter().copied().map(i64::from))
                    .collect();
                assert_eq!(
                    input_values(&logical, [rows, tokens], field, padding),
                    expected
                );
            }
        }
    }

    /// Only a session whose every node is on the CPU provider may be run from
    /// several threads at once.
    #[test]
    fn only_the_cpu_alone_reaches_the_shared_session() {
        let assigned = |entries: &[(&str, usize)]| {
            entries
                .iter()
                .map(|(provider, nodes)| (provider.to_string(), *nodes))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert!(cpu_alone(&assigned(&[("CPUExecutionProvider", 900)])).is_ok());
        assert!(cpu_alone(&assigned(&[])).is_err(), "nothing recorded");
        for other in [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "TensorrtExecutionProvider",
            "XnnpackExecutionProvider",
        ] {
            assert!(
                cpu_alone(&assigned(&[("CPUExecutionProvider", 10), (other, 1)])).is_err(),
                "{other} reached the shared session"
            );
        }
    }

    /// A real graph loaded for sharing records its nodes on the CPU provider,
    /// so the guard above admits it rather than refusing every model.
    #[test]
    fn a_cpu_graph_loads_as_a_shared_session() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/coreml-normalization.onnx");
        SharedCpuSession::load(&model).expect("a CPU graph is admitted");
    }

    /// A shared session that will not load leaves every pass to `session`,
    /// and stops the reranker planning passes to run side by side.
    #[test]
    fn a_shared_session_that_will_not_load_stops_side_by_side_passes() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/coreml-normalization.onnx");
        let mut encoder = Encoder::side_by_side(
            model.clone(),
            4,
            crate::tokenizer::tests::fixture(),
            String::from("fixture"),
        )
        .expect("a CPU graph loads");
        encoder.side_by_side = Some((model.with_file_name("missing.onnx"), None, 4));
        assert_eq!(encoder.ready(4).expect("falls back"), 1);
        assert!(!encoder.runs_side_by_side());
    }

    #[test]
    fn every_native_shape_must_load_before_accepting_the_accelerator() {
        for rejected in [64, 128, 256] {
            let result = super::coreml_buckets(|_, tokens| {
                if tokens == rejected {
                    Err(super::failed(&"rejected shape"))
                } else {
                    Ok(tokens)
                }
            });
            assert!(
                result.is_err(),
                "accepted a model whose {rejected} bucket fails"
            );
        }
        assert_eq!(
            super::coreml_buckets(|_, tokens| Ok(tokens)).unwrap(),
            (128, 64, 256)
        );
    }
    use super::*;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn native_buckets_keep_short_inputs_short_without_truncating_longer_inputs() {
        for (length, shape) in [
            (1, (4, 64)),
            (64, (4, 64)),
            (65, (4, 128)),
            (128, (4, 128)),
            (129, (2, 256)),
            (256, (2, 256)),
        ] {
            assert_eq!(native_shape(length), shape);
            assert!(length <= shape.1);
            assert!(shape.0 * shape.1 <= 512);
        }
    }

    /// Tokenizing each pair alone and padding a group of them hands the model
    /// what tokenizing that group together does.
    ///
    /// The reranker tokenizes every candidate once, to learn its length, and
    /// then groups them; this is what makes that the same model input as
    /// tokenizing each group when it runs. Lengths chosen to straddle the
    /// fixture's eight-token limit, so truncation is in the comparison too.
    #[test]
    fn a_group_encoded_alone_and_padded_is_the_group_encoded_together() {
        let tokenizer = crate::tokenizer::tests::fixture();
        let pairs: Vec<(&str, &str)> = [
            "memory",
            "mem ory mymory memory memory memory memory",
            "",
            "memory memory  ",
        ]
        .iter()
        .map(|text| ("memory ", *text))
        .collect();

        let mut alone: Vec<Encoding> = pairs
            .iter()
            .map(|pair| tokenizer.encode(*pair, true).expect("encode"))
            .collect();
        let lengths: Vec<usize> = alone.iter().map(Encoding::len).collect();
        assert!(
            lengths.iter().any(|length| *length != lengths[0]),
            "the fixture's pairs all have one length, so nothing here pads"
        );
        pad(&tokenizer, &mut alone).expect("pad");
        let together = tokenizer.encode_batch(pairs, true).expect("encode");

        let ids = |encodings: &[Encoding]| -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
            encodings
                .iter()
                .map(|encoding| {
                    (
                        encoding.get_ids().to_vec(),
                        encoding.get_attention_mask().to_vec(),
                        encoding.get_type_ids().to_vec(),
                    )
                })
                .collect()
        };
        assert_eq!(ids(&alone), ids(&together));
    }
}
