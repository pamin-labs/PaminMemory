//! Late interaction: one vector per token, scored by `MaxSim`.
//!
//! Every reranker in [`crate::reranking`] is a cross-encoder: it reads the
//! query and one candidate together, so its work cannot be done before the
//! query arrives and its cost is a forward pass per candidate. Measured, that
//! is the largest single number in a search -- 1,588 ms for the `accurate`
//! tier over fifteen candidates, against 84 ms for the whole search without
//! it.
//!
//! Late interaction is the other shape. The query and the document are encoded
//! **separately**, into one small vector per token rather than one per text,
//! and the score is
//!
//! ```text
//! MaxSim(q, d) = sum over query tokens of max over document tokens of q . d
//! ```
//!
//! Which is a dot product. So a document's vectors can be computed when it is
//! written and the query-time cost is one encoder pass over the query plus
//! arithmetic -- the same restructuring the vector channel already gets from
//! embedding once and searching many times, applied to the stage that is
//! currently paying a pass per candidate.
//!
//! **What it costs is storage, and that is the whole trade.** One 128-float
//! vector per token against one per text: about 12.8 KB per hundred tokens
//! where a single embedding is 3 KB for the whole memory. Nothing here stores
//! them yet, and nothing should until the accuracy is measured -- which is why
//! this module encodes documents on demand and says so.
//!
//! ## The model
//!
//! `lightonai/mLateOn`, Apache-2.0, a 22-layer multilingual ModernBERT with a
//! 256,002-token vocabulary, exported to ONNX with an int8 variant of 312 MB
//! -- smaller than the `accurate` tier's 571 MB cross-encoder.
//!
//! **The projection head is not in the export.** The ONNX graph ends at the
//! encoder's 768-dimensional hidden states; the three layers that take those
//! to 128 dimensions ship as separate `safetensors` files, and applying them
//! is this module's job:
//!
//! | | shape | residual |
//! | --- | --- | --- |
//! | `1_Dense` | 768 -> 1536 | yes |
//! | `2_Dense` | 1536 -> 768 | yes |
//! | `3_Dense` | 768 -> 128 | no |
//!
//! A residual across a change of shape cannot be an addition, so the two
//! residual layers carry **two** matrices each -- `linear.weight` and
//! `residual.weight`, both `[out, in]` -- and the layer is `W x + R x`. Reading
//! only `linear.weight` would produce vectors that are the right shape, look
//! plausible, and rank wrongly. That is checked here by shape rather than
//! assumed.
//!
//! ## The three configurations disagree, and the tokenizer wins
//!
//! The same trap [`crate::typed`] documents, in a second checkpoint.
//! `config.json` gives `cls_token_id: 1`, `sep_token_id: 1` and
//! `pad_token_id: 0`; `tokenizer_config.json` names `<bos>`, `<eos>` and
//! `<mask>`; `onnx_config.json` gives `mask_token_id: 4` and
//! `pad_token_id: 4`. They cannot all be right. The tokenizer is what the
//! reference implementation encodes with, so the sequence here is whatever
//! `tokenizer.encode` produces with its own special tokens, with the prefix
//! id inserted at position one -- which is `insert_prefix_token` in the
//! published code, step for step.
//!
//! The sequence length limits disagree too: 299 in `tokenizer_config.json`,
//! 8,191 in `sentence_bert_config.json`, 8,192 in `onnx_config.json`. The
//! corpora this is measured on average about fifty tokens a memory, so none of
//! the three binds, and the largest is taken with this note rather than a
//! choice being hidden.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::error::{IndexError, Result};

/// Where the weights come from.
const REPOSITORY: &str = "lightonai/mLateOn";

/// The int8 export. 312 MB against the float export's 1.2 GB, and the same
/// quantization the cross-encoder tiers already download.
const MODEL: &str = "model_int8.onnx";

/// The tokenizer, which is the authority on the special tokens.
const TOKENIZER: &str = "tokenizer.json";

/// The three projection layers, in the order `modules.json` applies them.
const DENSE: [&str; 3] = [
    "1_Dense/model.safetensors",
    "2_Dense/model.safetensors",
    "3_Dense/model.safetensors",
];

/// The added tokens that tell the encoder which side it is reading.
///
/// From `onnx_config.json`, and they are the last two ids of a 256,002-token
/// vocabulary, which is what an added pair looks like.
const QUERY_PREFIX: u32 = 256_000;
const DOCUMENT_PREFIX: u32 = 256_001;

/// What the head projects to.
const DIMS: usize = 128;

/// One `[out, in]` weight matrix, row-major, as `safetensors` stores it.
struct Matrix {
    out: usize,
    inp: usize,
    weight: Vec<f32>,
}

impl Matrix {
    /// `y = W x`, into a caller-owned buffer so a token's three layers do not
    /// allocate four times.
    fn apply(&self, x: &[f32], into: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        debug_assert_eq!(into.len(), self.out);
        for (row, value) in into.iter_mut().enumerate() {
            let weights = &self.weight[row * self.inp..(row + 1) * self.inp];
            *value = weights.iter().zip(x).map(|(w, x)| w * x).sum();
        }
    }

    /// `y += W x`, for the residual half.
    fn add(&self, x: &[f32], into: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        debug_assert_eq!(into.len(), self.out);
        for (row, value) in into.iter_mut().enumerate() {
            let weights = &self.weight[row * self.inp..(row + 1) * self.inp];
            *value += weights.iter().zip(x).map(|(w, x)| w * x).sum::<f32>();
        }
    }
}

/// One of the three layers: a projection, and a residual when the shape
/// changes under one.
struct Layer {
    linear: Matrix,
    residual: Option<Matrix>,
}

/// The head the ONNX export leaves out.
struct Head {
    layers: Vec<Layer>,
}

impl Head {
    /// 768 hidden dimensions to 128, L2-normalised.
    ///
    /// Normalised because the reference implementation's `encode` normalises by
    /// default, and because `MaxSim` is then a dot product rather than a
    /// division per pair.
    fn project(&self, hidden: &[f32]) -> Vec<f32> {
        let mut current = hidden.to_vec();
        for layer in &self.layers {
            let mut next = vec![0.0; layer.linear.out];
            layer.linear.apply(&current, &mut next);
            if let Some(residual) = &layer.residual {
                residual.add(&current, &mut next);
            }
            current = next;
        }

        let norm = current
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if norm > 0.0 {
            for value in &mut current {
                *value /= norm;
            }
        }
        current
    }
}

/// Reads one tensor out of a `safetensors` file.
///
/// Parsed here rather than through a crate: the format is an eight-byte
/// little-endian header length, a JSON header, then the raw data at the
/// offsets the header gives. Nineteen megabytes of `f32` across three files
/// does not need a dependency, and `serde_json` is already in the tree.
fn tensor(path: &Path, name: &str) -> Result<Matrix> {
    #[derive(serde::Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [usize; 2],
    }

    let bytes = std::fs::read(path)?;
    let length = bytes
        .get(..8)
        .map(|head| u64::from_le_bytes(head.try_into().expect("eight bytes")) as usize)
        .ok_or_else(|| IndexError::Engine(format!("{} is not safetensors", path.display())))?;
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(
        bytes
            .get(8..8 + length)
            .ok_or_else(|| IndexError::Engine(format!("{} is truncated", path.display())))?,
    )
    .map_err(|error| IndexError::Engine(format!("{}: {error}", path.display())))?;

    let entry: Entry = header
        .get(name)
        .ok_or_else(|| {
            IndexError::Engine(format!(
                "{} has no tensor {name}, only {:?}",
                path.display(),
                header.keys().collect::<Vec<_>>()
            ))
        })
        .and_then(|value| {
            serde_json::from_value(value.clone())
                .map_err(|error| IndexError::Engine(format!("{name}: {error}")))
        })?;

    // Checked rather than trusted, because a head read at the wrong precision
    // or the wrong shape gives vectors that are the right size and rank
    // wrongly, which nothing downstream can notice.
    if entry.dtype != "F32" {
        return Err(IndexError::Engine(format!(
            "{name} is {} where this reads F32",
            entry.dtype
        )));
    }
    let [out, inp] = entry.shape[..] else {
        return Err(IndexError::Engine(format!(
            "{name} has shape {:?} where a matrix has two axes",
            entry.shape
        )));
    };

    let data = bytes
        .get(8 + length + entry.data_offsets[0]..8 + length + entry.data_offsets[1])
        .ok_or_else(|| IndexError::Engine(format!("{name} runs past the end of the file")))?;
    if data.len() != out * inp * 4 {
        return Err(IndexError::Engine(format!(
            "{name} is {} bytes for a {out}x{inp} matrix of f32",
            data.len()
        )));
    }

    Ok(Matrix {
        out,
        inp,
        weight: data
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect(),
    })
}

/// A loaded late-interaction encoder.
pub struct LateInteraction {
    session: Session,
    tokenizer: Tokenizer,
    head: Head,
    output: String,
}

impl LateInteraction {
    /// Fetches the export and the head, and opens a session on the export.
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
                IndexError::Engine(format!("fetching {name} for late interaction: {error}"))
            })
        };

        let tokenizer = Tokenizer::from_file(fetch(TOKENIZER)?)
            .map_err(|error| IndexError::Engine(format!("loading the tokenizer: {error}")))?;

        // The residual is present exactly when the layer changes shape, which
        // is what `pylate`'s `Dense` does and what the three configurations
        // say. Read as "whichever tensors the file has" rather than hardcoded,
        // so a checkpoint that disagrees fails loudly here.
        let mut layers = Vec::with_capacity(DENSE.len());
        for name in DENSE {
            let path = fetch(name)?;
            let linear = tensor(&path, "linear.weight")?;
            let residual = (linear.out != linear.inp)
                .then(|| tensor(&path, "residual.weight"))
                .transpose()?;
            layers.push(Layer { linear, residual });
        }

        let head = Head { layers };
        let last = head
            .layers
            .last()
            .ok_or_else(|| IndexError::Engine("the head has no layers".into()))?;
        if last.linear.out != DIMS {
            return Err(IndexError::Engine(format!(
                "the head projects to {} dimensions where this model is documented at {DIMS}",
                last.linear.out
            )));
        }

        let mut builder = Session::builder()
            .map_err(|error| IndexError::Engine(format!("an onnx session: {error}")))?;
        if let Some(threads) = crate::inference::threads() {
            builder = builder
                .with_intra_threads(threads)
                .map_err(|error| IndexError::Engine(format!("session threads: {error}")))?;
        }
        let session = builder
            .commit_from_file(fetch(MODEL)?)
            .map_err(|error| IndexError::Engine(format!("loading the encoder: {error}")))?;

        // Read off the graph rather than named from the card, because the
        // export's own name for its hidden states is not documented anywhere
        // this could check.
        let output = session
            .outputs()
            .first()
            .ok_or_else(|| IndexError::Engine("the export has no outputs".into()))?
            .name()
            .to_string();

        Ok(Self {
            session,
            tokenizer,
            head,
            output,
        })
    }

    /// One sequence: the tokenizer's own special tokens, with the side's
    /// prefix inserted at position one.
    ///
    /// `insert_prefix_token` in the published implementation, which puts the
    /// prefix *after* the opening special token rather than before it.
    fn sequence(&self, text: &str, prefix: u32) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| IndexError::Engine(format!("tokenizing: {error}")))?;
        let ids = encoded.get_ids();
        let (first, rest) = ids
            .split_first()
            .ok_or_else(|| IndexError::Engine("the tokenizer produced no tokens".into()))?;

        let mut sequence = Vec::with_capacity(ids.len() + 1);
        sequence.push(*first);
        sequence.push(prefix);
        sequence.extend_from_slice(rest);
        Ok(sequence)
    }

    /// Encodes one side of the comparison: a vector per token, per text.
    ///
    /// Padded to the longest sequence in the batch and masked, and the padded
    /// positions are dropped from the result rather than returned as zeros --
    /// a padded position is not a token, and `MaxSim` maximises, so one left in
    /// could only ever raise a score.
    fn encode(&mut self, texts: &[&str], prefix: u32) -> Result<Vec<Vec<Vec<f32>>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let sequences: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| self.sequence(text, prefix))
            .collect::<Result<_>>()?;
        let longest = sequences.iter().map(Vec::len).max().unwrap_or(0);

        // The pad id is the one the tokenizer resolves, not the one
        // `config.json` gives: see the note at the top of this file. Padded
        // positions are masked out, so which id fills them cannot change a
        // score -- it is asserted rather than argued because a model that
        // attends to padding would make it matter.
        let pad = self
            .tokenizer
            .token_to_id("<mask>")
            .ok_or_else(|| IndexError::Engine("the tokenizer has no <mask> token".into()))?;

        let mut ids = Vec::with_capacity(sequences.len() * longest);
        let mut mask = Vec::with_capacity(sequences.len() * longest);
        for sequence in &sequences {
            for position in 0..longest {
                let present = position < sequence.len();
                ids.push(i64::from(if present { sequence[position] } else { pad }));
                mask.push(i64::from(present));
            }
        }

        let shape = [sequences.len() as i64, longest as i64];
        let named = |name: &'static str,
                     value: Tensor<i64>|
         -> (&'static str, SessionInputValue<'_>) { (name, value.into()) };
        let outputs = self
            .session
            .run(vec![
                named("input_ids", Tensor::from_array((shape, ids)).map_err(onnx)?),
                named(
                    "attention_mask",
                    Tensor::from_array((shape, mask)).map_err(onnx)?,
                ),
            ])
            .map_err(onnx)?;

        let (dimensions, hidden) = outputs[self.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(onnx)?;

        // Checked, not indexed blindly. Reading the wrong axis is the failure
        // mode that produces plausible numbers.
        let [batch, tokens, width] = dimensions[..] else {
            return Err(IndexError::Engine(format!(
                "the encoder returned {dimensions:?} where this expects three axes"
            )));
        };
        if batch as usize != sequences.len() || tokens as usize != longest {
            return Err(IndexError::Engine(format!(
                "the encoder returned {batch}x{tokens} for {}x{longest}",
                sequences.len()
            )));
        }
        let width = width as usize;

        Ok(sequences
            .iter()
            .enumerate()
            .map(|(text, sequence)| {
                (0..sequence.len())
                    .map(|position| {
                        let at = (text * longest + position) * width;
                        self.head.project(&hidden[at..at + width])
                    })
                    .collect()
            })
            .collect())
    }

    /// The query's token vectors.
    pub fn query(&mut self, text: &str) -> Result<Vec<Vec<f32>>> {
        Ok(self
            .encode(&[text], QUERY_PREFIX)?
            .pop()
            .unwrap_or_default())
    }

    /// One document's token vectors per text.
    ///
    /// Computed on demand here. In a system that shipped this they would be
    /// computed when the memory was written and read back at search time,
    /// which is the entire reason to prefer late interaction over a
    /// cross-encoder -- so a latency figure taken from this function is the
    /// figure for the arrangement nobody would ship.
    pub fn documents(&mut self, texts: &[&str]) -> Result<Vec<Vec<Vec<f32>>>> {
        self.encode(texts, DOCUMENT_PREFIX)
    }
}

/// `MaxSim`: for each query token, its best match anywhere in the document.
///
/// Sum rather than mean, which is what the published scoring does, so the
/// value grows with the query's length and is comparable across documents for
/// one query and not across queries. A document with no tokens scores zero
/// rather than negative infinity, because it is not a better match than
/// anything.
pub fn max_sim(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
    query
        .iter()
        .map(|q| {
            document
                .iter()
                .map(|d| q.iter().zip(d).map(|(q, d)| q * d).sum::<f32>())
                .fold(0.0_f32, f32::max)
        })
        .sum()
}

fn onnx(error: ort::Error) -> IndexError {
    IndexError::Engine(format!("the late-interaction encoder: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(out: usize, inp: usize, weight: &[f32]) -> Matrix {
        assert_eq!(weight.len(), out * inp);
        Matrix {
            out,
            inp,
            weight: weight.to_vec(),
        }
    }

    /// A residual layer is `W x + R x`, and reading only `W` is the mistake
    /// this pins.
    ///
    /// The shapes in the checkpoint change under both residual layers, so the
    /// residual cannot be an addition of the input and is a second matrix. An
    /// implementation that skipped it would produce vectors of the right
    /// width, normalised, plausible, and ranked wrongly -- which nothing
    /// downstream can detect. Hand-computed: `W = [[1,0],[0,1]]`,
    /// `R = [[0,1],[1,0]]`, `x = [3,4]` gives `[3,4] + [4,3] = [7,7]`, and
    /// normalising that is `[1/sqrt2, 1/sqrt2]`.
    #[test]
    fn a_residual_layer_adds_its_own_matrix_not_its_input() {
        let with = Head {
            layers: vec![Layer {
                linear: matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]),
                residual: Some(matrix(2, 2, &[0.0, 1.0, 1.0, 0.0])),
            }],
        };
        let projected = with.project(&[3.0, 4.0]);
        let root = 1.0 / 2.0_f32.sqrt();
        assert!(
            (projected[0] - root).abs() < 1e-6 && (projected[1] - root).abs() < 1e-6,
            "{projected:?}"
        );

        // And without the residual the same input gives a different direction,
        // so the assertion above is not true of both.
        let without = Head {
            layers: vec![Layer {
                linear: matrix(2, 2, &[1.0, 0.0, 0.0, 1.0]),
                residual: None,
            }],
        };
        let plain = without.project(&[3.0, 4.0]);
        assert!(
            (plain[0] - 0.6).abs() < 1e-6 && (plain[1] - 0.8).abs() < 1e-6,
            "{plain:?}"
        );
    }

    /// The three layers compose in order, and the widths chain.
    #[test]
    fn the_head_chains_its_layers() {
        let head = Head {
            layers: vec![
                Layer {
                    linear: matrix(4, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
                    residual: None,
                },
                Layer {
                    linear: matrix(2, 4, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    residual: None,
                },
            ],
        };
        // [1,2] -> [1,2,3,0] -> [1,2] -> normalised.
        let projected = head.project(&[1.0, 2.0]);
        let norm = 5.0_f32.sqrt();
        assert!(
            (projected[0] - 1.0 / norm).abs() < 1e-6 && (projected[1] - 2.0 / norm).abs() < 1e-6,
            "{projected:?}"
        );
    }

    /// `MaxSim` maximises over the document and sums over the query.
    ///
    /// Both halves matter and swapping them is a plausible bug: summing over
    /// the document would make a long document score higher for nothing, and
    /// maximising over the query would reduce a query to its single best
    /// token.
    #[test]
    fn max_sim_maximises_over_the_document_and_sums_over_the_query() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Two tokens, each perfectly matching one query token, plus a third
        // that matches neither.
        let document = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        assert!((max_sim(&query, &document) - 2.0).abs() < 1e-6);

        // A document holding only the first gets one of the two.
        assert!((max_sim(&query, &document[..1]) - 1.0).abs() < 1e-6);

        // And an empty document is zero rather than a negative infinity that
        // would sort below a document that matched nothing.
        assert_eq!(max_sim(&query, &[]), 0.0);
    }

    /// The `safetensors` header is read by its own offsets.
    ///
    /// Parsed by hand in this file, so the format is pinned by a round trip
    /// rather than by trust: eight little-endian bytes of header length, a
    /// JSON header, then the data at the offsets it gives.
    #[test]
    fn a_safetensors_matrix_round_trips() {
        let values: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let header = br#"{"linear.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let directory = tempfile::tempdir().expect("a temporary directory");
        let path = directory.path().join("model.safetensors");
        std::fs::write(&path, &bytes).expect("writing the file");

        let read = tensor(&path, "linear.weight").expect("reading the tensor");
        assert_eq!((read.out, read.inp), (2, 3));
        assert_eq!(read.weight, values.to_vec());

        // And a name that is not there names what is, rather than panicking.
        let missing = tensor(&path, "residual.weight")
            .err()
            .expect("no such tensor");
        assert!(
            format!("{missing}").contains("linear.weight"),
            "the error should say what the file does have: {missing}"
        );
    }
}
