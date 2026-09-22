//! A typed-decision model driven as a pointwise relevance judge.
//!
//! Every reranker in this crate before this one is a cross-encoder: a pair goes
//! in, one relevance logit comes out, and `fastembed`'s `TextRerank` drives it.
//! This drives something else — an encoder with a decision head that plants a
//! `[MASK]` marker per answer option and scores the markers — and it exists for
//! one property the cross-encoders do not have.
//!
//! **A cross-encoder's logit is calibrated against nothing.** It separates the
//! candidates of one shortlist and means nothing between two queries, which is
//! the constraint [`pamin_core::Combine::Banded`] was derived from and the
//! reason a fixed relevance threshold is not expressible here. A model trained
//! against strictly proper scoring rules reports a probability instead, and a
//! probability is comparable across queries. Three things this project wants
//! need exactly that: a weighted-sum fusion, an abstention gate, and the
//! published threshold result it has so far only been able to refute in its
//! uncalibrated form.
//!
//! ## Why this is a port and not a guess
//!
//! The sequence format is published, so none of it is inferred:
//!
//! ```text
//! [CLS] "<qtype> question: <instructions>" [SEP]
//!   [MASK]" false: no, the statement does not hold"
//!   [MASK]" true: yes, the statement holds"          [SEP]
//!   <state>                                          [SEP]
//! ```
//!
//! with the markers at the two `[MASK]` indices, the head truncated to whatever
//! the options leave of `head_max_len`, each option's text capped, and the state
//! filling the rest of `max_len`. The upstream `render_options` fixes those two
//! labels for a `noul` question and its own comment records the property this
//! module turns on: *"Noul is always `[false, true]` so `p[1] == noul`"*. So the
//! document goes in the **state** and only two short labels go in the head
//! budget — which is why `head_max_len` never constrained the document, though
//! an earlier reading of this design assumed it did. What that budget rules out
//! is the *listwise* shape, one pass and one softmax over a whole shortlist, and
//! only that.
//!
//! ## What is unmeasured, and must not be assumed
//!
//! The model was trained on typed decisions over business workflows and **never
//! on relevance**. Its card carries no retrieval benchmark of any kind. So this
//! module is the instrument for finding out whether it can judge relevance at
//! all, and a result from it is a measurement rather than a confirmation.
//!
//! And the published multilingual checkpoint ships **uncalibrated**: its
//! configuration carries identity temperatures, which is the over-confident
//! state its own card describes. [`Judge::with_temperature`] exists so a
//! temperature fitted on held-out judgements can be supplied, and until one is
//! the probabilities are ordered correctly and scaled wrongly — fine for
//! ranking, useless for a threshold.

use std::path::{Path, PathBuf};

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::error::{IndexError, Result};

/// The export this drives.
///
/// An independent port rather than an official release, and taken for the
/// reasons `NOTICE` records: Apache-2.0 as upstream, float16 weights with a
/// float32 decision tail, opset 18 with standard operators only, and its own
/// card reports 63 of 63 answers agreeing with the reference runtime at a
/// maximum probability error of 5.1e-4 **on the ONNX Runtime CPU provider** —
/// which is the provider this project runs.
pub(crate) const REPOSITORY: &str = "mizchi/laya-multilingual-onnx";

/// The graph, the tokenizer, and the prompt limits.
const MODEL: &str = "model.onnx";
const TOKENIZER: &str = "tokenizer/tokenizer.json";
const LIMITS: &str = "rl_agent_config.json";

/// `noul`, the two-option question type, whose second option is `true`.
const QTYPE_NOUL: i64 = 2;

/// What the model is asked about the state.
///
/// **A judgement call, and the one input here that nothing checks.** A
/// zero-shot model's answer moves with the wording of its question, and this
/// phrasing was chosen to match the shape the model was trained on — a
/// statement that either holds or does not — rather than to read like a
/// retrieval instruction. A different phrasing is a different measurement, so
/// it is a constant with a name instead of a string buried in a function.
const STATEMENT: &str = "the document answers the query";

/// The two option labels, exactly as upstream renders them for a `noul`
/// question. `false` first, so the softmax's second entry is the probability
/// the statement holds.
const OPTIONS: [&str; 2] = [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
];

/// Tokens of an option's text past this are dropped, as upstream does.
const OPTION_TOKENS: usize = 48;

/// Prompt limits, read from the checkpoint rather than hardcoded, because they
/// are properties of how it was trained.
#[derive(serde::Deserialize)]
struct Limits {
    max_len: usize,
    head_max_len: usize,
}

/// A loaded typed-decision model.
pub struct Judge {
    session: Session,
    tokenizer: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    max_len: usize,
    head_max_len: usize,
    /// Divides the marker logits before the softmax. `1.0` is the identity the
    /// published checkpoint ships, which is the uncalibrated state.
    temperature: f32,
}

impl Judge {
    /// Fetches the export and opens a session on it.
    pub fn load(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;

        let repository = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .model(REPOSITORY.to_string());

        let fetch = |name: &str| -> Result<PathBuf> {
            repository.get(name).map_err(|error| {
                IndexError::Engine(format!("fetching {name} for the typed judge: {error}"))
            })
        };

        let limits: Limits = serde_json::from_slice(&std::fs::read(fetch(LIMITS)?)?)
            .map_err(|error| IndexError::Engine(format!("reading the prompt limits: {error}")))?;

        let tokenizer = Tokenizer::from_file(fetch(TOKENIZER)?)
            .map_err(|error| IndexError::Engine(format!("loading the tokenizer: {error}")))?;

        // Resolved through the tokenizer rather than the encoder's config,
        // because that is where the reference implementation reads them from
        // and the two disagree: the config gives one id for `cls`, `sep` and
        // `eos` alike, where the tokenizer distinguishes `<bos>` from `<eos>`.
        let id = |token: &str| -> Result<u32> {
            tokenizer
                .token_to_id(token)
                .ok_or_else(|| IndexError::Engine(format!("the tokenizer has no {token} token")))
        };

        let mut builder = Session::builder()
            .map_err(|error| IndexError::Engine(format!("an onnx session: {error}")))?;
        // The same setting the cross-encoders take, for the same reason: see
        // `crate::inference`.
        if let Some(threads) = crate::inference::threads() {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| IndexError::Engine(format!("session threads: {error}")))?;
        }
        // By path, so the runtime maps the file rather than being handed six
        // hundred megabytes of copy.
        let session = builder
            .commit_from_file(fetch(MODEL)?)
            .map_err(|error| IndexError::Engine(format!("loading the typed judge: {error}")))?;

        Ok(Self {
            session,
            cls: id("<bos>")?,
            sep: id("<eos>")?,
            mask: id("<mask>")?,
            pad: id("<pad>")?,
            tokenizer,
            max_len: limits.max_len,
            head_max_len: limits.head_max_len,
            temperature: 1.0,
        })
    }

    /// Supplies a temperature fitted on held-out judgements.
    ///
    /// Without one the probabilities below are ordered correctly and scaled
    /// wrongly, which is enough to rank and not enough to threshold.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        if temperature > 0.0 {
            self.temperature = temperature;
        }
        self
    }

    /// Encodes text without special tokens, which the sequence adds itself.
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|error| IndexError::Engine(format!("tokenizing: {error}")))?
            .get_ids()
            .to_vec())
    }

    /// Builds one sequence and the positions of its two markers.
    ///
    /// Follows the published `build_sequence` step for step, including the
    /// order the budgets are spent in: the options are measured first, the head
    /// gets what they leave, and the state gets what is left of `max_len`. Doing
    /// it in another order changes which text survives truncation.
    fn sequence(&self, query: &str, document: &str) -> Result<(Vec<u32>, [i64; 2])> {
        let head = self.encode(&format!("noul question: {STATEMENT}"))?;

        let mut options: Vec<Vec<u32>> = Vec::with_capacity(OPTIONS.len());
        for option in OPTIONS {
            let mut ids = vec![self.mask];
            let mut text = self.encode(&format!(" {option}"))?;
            text.truncate(OPTION_TOKENS);
            ids.extend(text);
            options.push(ids);
        }

        let spent: usize = options.iter().map(Vec::len).sum();
        let mut budget = self.head_max_len.saturating_sub(spent);
        if budget < 16 {
            // Too many or too long options: shrink every option evenly, as
            // upstream does, rather than dropping one.
            let each = ((self.head_max_len.saturating_sub(16)) / options.len().max(1)).max(4);
            for option in &mut options {
                option.truncate(each);
            }
            budget = self
                .head_max_len
                .saturating_sub(options.iter().map(Vec::len).sum::<usize>());
        }

        let mut ids = Vec::with_capacity(self.max_len);
        ids.push(self.cls);
        ids.extend(head.iter().take(budget.max(8)));
        ids.push(self.sep);

        let mut markers = [0i64; 2];
        for (at, option) in options.iter().enumerate() {
            markers[at] = ids.len() as i64;
            ids.extend(option);
        }
        ids.push(self.sep);

        // The state is a JSON object, which is what upstream's `serialize_state`
        // produces for anything that is not already a string. Both fields go in
        // so the judgement is about the pair rather than about the document.
        let state = serde_json::json!({ "query": query, "document": document }).to_string();
        let room = self.max_len.saturating_sub(ids.len() + 1);
        let mut encoded = self.encode(&state)?;
        encoded.truncate(room);
        ids.extend(encoded);
        ids.push(self.sep);
        ids.truncate(self.max_len);

        Ok((ids, markers))
    }

    /// `P(relevant)` for each document, in the order given.
    ///
    /// One batched pass. The state is re-encoded per document, which is what the
    /// reference implementation does too — the batch is a launch, not shared
    /// work, and this project measured the published latencies agreeing with
    /// that reading.
    pub fn judge(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut built = Vec::with_capacity(documents.len());
        for document in documents {
            built.push(self.sequence(query, document)?);
        }
        let width = built.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        let rows = built.len();

        let mut input_ids = Vec::with_capacity(rows * width);
        let mut attention = Vec::with_capacity(rows * width);
        let mut marker_pos = Vec::with_capacity(rows * OPTIONS.len());
        for (ids, markers) in &built {
            input_ids.extend(ids.iter().map(|id| i64::from(*id)));
            input_ids.extend(std::iter::repeat_n(i64::from(self.pad), width - ids.len()));
            attention.extend(std::iter::repeat_n(1i64, ids.len()));
            attention.extend(std::iter::repeat_n(0i64, width - ids.len()));
            marker_pos.extend_from_slice(markers);
        }
        // Both markers are always present: a `noul` question has exactly two
        // options and the head budget guarantees room for them, so there is no
        // masked-out marker to describe.
        let marker_mask = vec![true; rows * OPTIONS.len()];

        let shape = [rows as i64, width as i64];
        let markers = [rows as i64, OPTIONS.len() as i64];
        let tensor = |name: &'static str,
                      value: Tensor<i64>|
         -> (&'static str, SessionInputValue<'_>) { (name, value.into()) };
        let inputs = vec![
            tensor(
                "input_ids",
                Tensor::from_array((shape, input_ids)).map_err(onnx)?,
            ),
            tensor(
                "attention_mask",
                Tensor::from_array((shape, attention)).map_err(onnx)?,
            ),
            tensor(
                "marker_pos",
                Tensor::from_array((markers, marker_pos)).map_err(onnx)?,
            ),
            (
                "marker_mask",
                SessionInputValue::from(Tensor::from_array((markers, marker_mask)).map_err(onnx)?),
            ),
            tensor(
                "qtype",
                Tensor::from_array(([rows as i64], vec![QTYPE_NOUL; rows])).map_err(onnx)?,
            ),
        ];

        let outputs = self.session.run(inputs).map_err(onnx)?;
        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;

        // The graph is documented as returning `[batch, markers]`, and a shape
        // that disagrees means the export changed under us. Checked rather than
        // indexed blindly, because reading the wrong axis is the failure mode
        // that produces plausible numbers.
        let columns = shape.last().copied().unwrap_or(0) as usize;
        if shape.len() != 2 || shape[0] as usize != rows || columns < OPTIONS.len() {
            return Err(IndexError::Engine(format!(
                "the typed judge returned logits shaped {shape:?} for {rows} rows of \
                 {} options",
                OPTIONS.len()
            )));
        }

        Ok((0..rows)
            .map(|row| {
                let pair = &logits[row * columns..row * columns + OPTIONS.len()];
                // Softmax over the two options, shifted for stability, with the
                // temperature applied first. `OPTIONS[1]` is `true`, so the
                // second entry is the probability the statement holds.
                let scaled: Vec<f32> = pair.iter().map(|z| z / self.temperature).collect();
                let highest = scaled.iter().copied().fold(f32::MIN, f32::max);
                let exponentiated: Vec<f32> = scaled.iter().map(|z| (z - highest).exp()).collect();
                let total: f32 = exponentiated.iter().sum();
                if total > 0.0 {
                    exponentiated[1] / total
                } else {
                    0.0
                }
            })
            .collect())
    }
}

fn onnx(error: impl std::fmt::Display) -> IndexError {
    IndexError::Engine(format!("the typed judge: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two labels have to stay in this order, and a reader should not have
    /// to take that on trust.
    ///
    /// Upstream's own comment is *"Noul is always `[false, true]` so
    /// `p[1] == noul`"*, and every probability this module returns is the second
    /// softmax entry. Swap the constants and the module returns
    /// `1 - P(relevant)` with nothing failing, which would look like a model
    /// that ranks backwards rather than like a typo.
    #[test]
    fn the_true_option_is_second() {
        assert!(OPTIONS[0].starts_with("false:"));
        assert!(OPTIONS[1].starts_with("true:"));
    }

    /// `noul` is question type two, and the graph takes the number.
    #[test]
    fn the_question_type_is_noul() {
        assert_eq!(QTYPE_NOUL, 2, "0 is choice and 1 is score");
    }
}
