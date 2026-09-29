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
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fixed: Option<(PathBuf, Option<Session>, Option<Session>)>,
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
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            fixed: None,
            token_type_ids,
        })
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn load_fixed_coreml(
        model: impl FnOnce() -> Result<PathBuf>,
        repository: &Repository,
        max_length: usize,
    ) -> Result<Self> {
        if max_length > 256 {
            return Err(failed(
                &"static CoreML reranker currently requires at most 256 tokens",
            ));
        }
        let (session, path) = crate::inference::fixed_coreml(model, 4, 128)?;
        let tokenizer = crate::tokenizer::load(repository, max_length)?;
        Ok(Self {
            tokenizer,
            session,
            fixed: Some((path, None, None)),
            token_type_ids: false,
        })
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
        let mut fixed_encodings;
        let encodings = if (rows, tokens) != (logical_rows, length) {
            fixed_encodings = encodings.to_vec();
            for encoding in &mut fixed_encodings {
                encoding.pad(tokens, 1, 0, "<pad>", tokenizers::PaddingDirection::Right);
            }
            while fixed_encodings.len() < rows {
                fixed_encodings.push(fixed_encodings.last().expect("nonempty").clone());
            }
            &fixed_encodings
        } else {
            encodings
        };
        let shape = [rows, tokens];
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
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let session = if let Some((path, short, long)) = &mut self.fixed {
            if tokens != 128 {
                let bucket = if tokens == 64 { short } else { long };
                if bucket.is_none() {
                    *bucket =
                        Some(crate::inference::fixed_coreml(|| Ok(path.clone()), rows, tokens)?.0);
                }
                bucket.as_mut().expect("loaded native bucket")
            } else {
                &mut self.session
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
