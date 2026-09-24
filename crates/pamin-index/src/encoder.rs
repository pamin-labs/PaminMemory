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
        let session = crate::inference::session(providers, model)?;
        let tokenizer = crate::tokenizer::load(repository, max_length)?;
        let token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");
        Ok(Self {
            tokenizer,
            session,
            token_type_ids,
        })
    }

    /// One forward pass over `inputs` -- texts, or pairs of them -- as one
    /// batch.
    pub(crate) fn run<'s, E>(&mut self, inputs: Vec<E>) -> Result<SessionOutputs<'_>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let failed = |error: &dyn std::fmt::Display| {
            IndexError::Engine(format!("running the model: {error}"))
        };
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|error| failed(&error))?;
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
        if self.token_type_ids {
            feed.push((
                "token_type_ids".into(),
                column(Encoding::get_type_ids)?.into(),
            ));
        }
        self.session.run(feed).map_err(|error| failed(&error))
    }
}
