//! A tokenizer and an ONNX Runtime session, and one forward pass through both.
//!
//! What the rerankers and the embedder BGE-M3 used `fastembed` for, taken in
//! here so that their tokenizers could share a vocabulary (see
//! `crate::tokenizer`) -- its types own theirs and have no constructor that
//! takes one. pplx-embed, which replaced BGE-M3, runs here too. What each model
//! does with the outputs stays with the model: the reranker reads a logit per
//! pair, the embedder a vector per text.
//!
//! The inputs are built the way `fastembed` 6.1 built them, because they are
//! the scores: a batch encoded with special tokens and padded to its longest
//! member, its ids and attention mask fed as `i64` in the batch's shape, and
//! token type ids only to a graph that declares an input of that name.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ort::ep::ExecutionProviderDispatch;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use tokenizers::{EncodeInput, Encoding};

use crate::error::{IndexError, Result};
use crate::hub::Repository;
use crate::tokenizer::Tokenizer;

pub(crate) struct Encoder {
    tokenizer: Tokenizer,
    session: Session,
    /// Whether the graph takes token type ids. XLM-R's family ignores them and
    /// most of its exports do not declare the input.
    token_type_ids: bool,
    /// The graph `session` was loaded from, and on what, so that
    /// [`run_each`](Self::run_each) can open more sessions over it.
    model: PathBuf,
    providers: Vec<ExecutionProviderDispatch>,
    /// Sessions of one intra-op thread each, over the same graph, that
    /// [`run_each`](Self::run_each) runs texts on side by side. Opened the
    /// first time it is handed more than one text, so a process that only
    /// ever embeds a query never pays for them, and kept: a cascade round
    /// hands it up to sixty-four at a time, round after round.
    workers: Vec<Session>,
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
        model: impl FnOnce() -> Result<PathBuf>,
        repository: &Repository,
        max_length: usize,
        providers: Vec<ExecutionProviderDispatch>,
    ) -> Result<Self> {
        let mut path = None;
        let session = crate::inference::session(providers.clone(), || {
            let model = model()?;
            path = Some(model.clone());
            Ok(model)
        })?;
        let model = path.ok_or_else(|| failed(&"the session was loaded from nowhere"))?;
        let tokenizer = crate::tokenizer::load(repository, max_length)?;
        let token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Ok(Self {
            tokenizer,
            session,
            token_type_ids,
            model,
            providers,
            workers: Vec::new(),
        })
    }

    /// One forward pass over `inputs` -- texts, or pairs of them -- as one
    /// batch.
    pub(crate) fn run<'s, E>(&mut self, inputs: Vec<E>) -> Result<SessionOutputs<'_>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|error| failed(&error))?;
        self.forward(&encodings)
    }

    /// Each of `inputs` tokenized on its own: truncated and given its special
    /// tokens, but not padded, so its length is its own.
    ///
    /// For a caller that groups inputs by length before running them, and so
    /// has to know the lengths first. [`run_encoded`](Self::run_encoded) pads
    /// a group of these exactly as [`run`](Self::run) pads what it tokenizes:
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
    pub(crate) fn run_encoded(&mut self, mut batch: Vec<Encoding>) -> Result<SessionOutputs<'_>> {
        pad(&self.tokenizer, &mut batch)?;
        self.forward(&batch)
    }

    /// Each of `texts` in a forward pass of its own, and what `read` makes of
    /// each pass's outputs, in the order of `texts`.
    ///
    /// Side by side, one text to a session of one thread, on as many sessions
    /// as a single pass would have had threads ([`crate::inference::threads`])
    /// -- rather than one after another with every thread on each. The work is
    /// the same and so are the outputs: a pass does not depend on how many
    /// threads ran it, and pplx's vectors for 592 texts (six sets of 32 MuSiQue
    /// paragraphs, 330 XQuAD-R sentences in eleven languages and 70 more
    /// paragraphs) came back bit for bit the same this way as each
    /// embedded alone on the one session. What changes is that a pass split
    /// across threads waits for its slowest thread at every operator it
    /// splits, and passes on threads of their own never wait.
    ///
    /// Measured through `Embedder::embed_passages` on those six sets, six
    /// pairs of processes alternating which ran first, on four cores shared
    /// with another run (load average 6 to 7): 459 ms a paragraph against 545
    /// one after another, 0.84 of the time in every pair (0.839 to 0.847), or
    /// 2.18 paragraphs a second against 1.83. On a machine with nothing else
    /// running the waiting costs less, so the gain may be smaller there; that
    /// has not been measured.
    ///
    /// The price is memory. Each session maps the weights again, so resident
    /// size counts them once per session, but the pages are the page cache's
    /// and shared: proportional set size after 64 paragraphs was 981 MB
    /// against 884, all of the difference anonymous, 282 MB against 186.
    ///
    /// Longest first, each session taking the next text as it finishes one,
    /// so a batch does not end on one long text that started last.
    pub(crate) fn run_each<T: Send>(
        &mut self,
        texts: &[&str],
        read: impl Fn(&SessionOutputs<'_>) -> Result<T> + Sync,
    ) -> Result<Vec<T>> {
        let encodings = self.encode(texts.to_vec())?;
        let parallel = crate::inference::intra_threads()?.min(encodings.len());
        if parallel <= 1 {
            return encodings
                .into_iter()
                .map(|encoding| read(&self.run_encoded(vec![encoding])?))
                .collect();
        }
        while self.workers.len() < parallel {
            let model = self.model.clone();
            self.workers.push(crate::inference::session_on(
                self.providers.clone(),
                1,
                || Ok(model),
            )?);
        }

        let mut order: Vec<usize> = (0..encodings.len()).collect();
        order.sort_by_key(|at| std::cmp::Reverse(encodings[*at].len()));
        let next = AtomicUsize::new(0);
        let (order, next, encodings, read) = (&order, &next, &encodings, &read);
        let token_type_ids = self.token_type_ids;
        let ran: Vec<Vec<(usize, Result<T>)>> = std::thread::scope(|scope| {
            let workers: Vec<_> = self.workers[..parallel]
                .iter_mut()
                .map(|session| {
                    scope.spawn(move || {
                        let mut ran = Vec::new();
                        while let Some(at) = order.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let text = std::slice::from_ref(&encodings[*at]);
                            ran.push((
                                *at,
                                forward(session, token_type_ids, text)
                                    .and_then(|outputs| read(&outputs)),
                            ));
                        }
                        ran
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().expect("an encoder worker panicked"))
                .collect()
        });

        let mut outputs: Vec<Option<T>> =
            std::iter::repeat_with(|| None).take(texts.len()).collect();
        for (at, result) in ran.into_iter().flatten() {
            outputs[at] = Some(result?);
        }
        outputs
            .into_iter()
            .map(|result| result.ok_or_else(|| failed(&"a text went unread")))
            .collect()
    }

    fn forward(&mut self, encodings: &[Encoding]) -> Result<SessionOutputs<'_>> {
        forward(&mut self.session, self.token_type_ids, encodings)
    }
}

/// One forward pass of `session` over `encodings`, which are padded to one
/// length.
fn forward<'s>(
    session: &'s mut Session,
    token_type_ids: bool,
    encodings: &[Encoding],
) -> Result<SessionOutputs<'s>> {
    let length = encodings
        .first()
        .ok_or_else(|| failed(&"nothing to encode"))?
        .len();
    let shape = [encodings.len(), length];
    let column = |field: fn(&Encoding) -> &[u32]| {
        let values: Vec<i64> = encodings
            .iter()
            .flat_map(|encoding| field(encoding).iter().map(|value| i64::from(*value)))
            .collect();
        Tensor::from_array((shape, values)).map_err(|error| failed(&error))
    };

    let mut feed = ort::inputs![
        "input_ids" => column(Encoding::get_ids)?,
        "attention_mask" => column(Encoding::get_attention_mask)?,
    ];
    if token_type_ids {
        feed.push((
            "token_type_ids".into(),
            column(Encoding::get_type_ids)?.into(),
        ));
    }
    session.run(feed).map_err(|error| failed(&error))
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

#[cfg(test)]
mod tests {
    use super::*;

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
