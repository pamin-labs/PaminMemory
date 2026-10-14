//! EmbeddingGemma 2, int8, read from a local directory: the experimental
//! `gemma` profile.
//!
//! The export is not published anywhere yet, so there is no download path:
//! `PAMIN_EG2_DIR` names a directory holding `model.onnx`, the
//! `model.onnx.data` it names, and the tokenizer's `tokenizer.json`,
//! `config.json` and `tokenizer_config.json`. Without it the profile does not
//! load, and says why.
//!
//! The graph takes `input_ids` and `attention_mask` and returns
//! `sentence_embedding`, the attention-masked mean of the last hidden state
//! projected to 768 dimensions and L2-normalized -- the model's own pooling
//! and normalization, inside the graph, so a vector is used as it comes, at
//! its full width. Its Matryoshka training makes a prefix a vector too, but
//! 512 dimensions measured below 768 through the search entry point (XQuAD-R
//! cross-lingual nDCG@10 -0.0028, same-language -0.0017), so no width is cut.
//!
//! What this export is: every matrix int8 with per-token u8 activations, the
//! rotary embedding and grouped-query attention as ONNX Runtime's contrib
//! operators, opset 21. Activations have to be unsigned on x86: the signed
//! kernels run at a hundredth of the speed. Some of those activations are
//! quantized per tensor, so -- like BGE-M3's int8 export -- a text's vector
//! depends on what shares its batch, and it is run one text at a time for the
//! reason `crate::embedding` gives for that model.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};

/// The directory the model is read from.
const DIRECTORY: &str = "PAMIN_EG2_DIR";

const MODEL: &str = "model.onnx";
const DATA: &str = "model.onnx.data";
const TOKENIZER: &str = "tokenizer.json";

/// The prompts the model was trained with, for queries and for stored text.
///
/// The document prompt names a title, and memories have one -- the topic. It
/// is `none` here because that is what the retrieval evaluation that chose
/// this export used, and changing it changes every stored vector.
pub(crate) const PREFIXES: (&str, &str) = ("task: search result | query: ", "title: none | text: ");

/// The width the graph returns, and the width this profile stores.
pub(crate) const WIDTH: u32 = 768;

/// The configured directory, or why there is none.
pub(crate) fn directory() -> Result<PathBuf> {
    std::env::var_os(DIRECTORY)
        .map(PathBuf::from)
        .ok_or_else(|| {
            IndexError::Engine(format!(
                "the gemma profile reads its model from a local directory; set {DIRECTORY} \
                 to one holding {MODEL}, {DATA} and {TOKENIZER}"
            ))
        })
}

/// The identity recorded with every vector this profile stores.
///
/// A digest of the graph, its weights and its tokenizer, and the width (the
/// `:d768` suffix stays so indexes built before the width was fixed open): two
/// exports of this model differ in nothing but their quantization -- the two
/// final candidates' weight files are the same size to the byte -- and each
/// produces its own space. Read once per process; hashing the 275 MB of
/// weights takes about a second. With no directory configured, or one that
/// cannot be read, the identity says so, and no load succeeds under it.
pub(crate) fn identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let digest = directory().and_then(|directory| {
            let mut combined = String::new();
            for file in [MODEL, DATA, TOKENIZER] {
                combined.push_str(&crate::prepared::source_digest(&directory.join(file))?);
            }
            Ok(format!("{:x}", Sha256::digest(combined.as_bytes())))
        });
        match digest {
            Ok(digest) => format!("embeddinggemma-2-int8@{}:d{WIDTH}", &digest[..16]),
            Err(_) => "embeddinggemma-2-int8:unconfigured".to_string(),
        }
    })
}

/// The model on `target`, with its tokenizer, truncating at `max_length`.
pub(crate) fn load(
    max_length: usize,
    target: impl Into<crate::inference::Target>,
) -> Result<Encoder> {
    let directory = directory()?;
    let tokenizer = crate::tokenizer::load_dir(&directory, max_length)?;
    Encoder::load_with(|| Ok(directory.join(MODEL)), tokenizer, target)
        .map_err(|error| error.context("loading embedding model"))
}

/// A graph vector, refused unless it has the width the graph declares; the
/// graph's own normalization stands, untouched.
pub(crate) fn checked(vector: Vec<f32>) -> Result<Vec<f32>> {
    if vector.len() != WIDTH as usize {
        return Err(IndexError::Engine(format!(
            "the gemma export returned {} dimensions, not {WIDTH}",
            vector.len()
        )));
    }
    Ok(vector)
}

/// Where this profile's model lives, for the plan cache's identity.
pub(crate) fn source() -> Result<String> {
    let directory = directory()?;
    Ok(canonical(&directory).display().to_string())
}

fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The product's embedder runs the graph as well as the Python wheel does.
    ///
    /// The reference is `parity.json` in `PAMIN_EG2_DIR`: for each text, the
    /// vector this graph returned under ONNX Runtime 1.28's Python wheel and
    /// the vector the unquantized export of the same model returned, one text
    /// per pass, through the Hugging Face tokenizer, with both prompts,
    /// truncated at 512 tokens. It names the graph and weights it came from,
    /// and the test refuses a reference made from other files.
    ///
    /// Not bit-identical to the wheel, and not within the 0.9999 an
    /// accelerator is held to: the binary this crate links is another build
    /// of the same runtime release, and every activation here is rounded to
    /// eight bits, so a last-bit difference in a float kernel moves a rounding
    /// boundary. Measured on these texts, this build and the wheel agree at
    /// 0.9980 to 0.9994, and each is 0.9980 to 0.9994 from the unquantized
    /// model -- two draws of the same quantization error, neither nearer the
    /// model. The unquantized export of the same model matches the wheel to
    /// 1e-7 here. So the contract is that: no further from the unquantized
    /// vector than the wheel is, give or take 0.0005, and 0.997 from the
    /// wheel's -- which a wrong prompt, pooling, width or truncation is not.
    #[test]
    #[ignore = "needs the local EmbeddingGemma 2 export in PAMIN_EG2_DIR"]
    fn gemma_matches_the_python_reference() {
        let directory = directory().expect("set PAMIN_EG2_DIR");
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.join("parity.json")).expect("read parity.json"),
        )
        .expect("parse parity.json");
        for (file, key) in [(MODEL, "model_sha256"), (DATA, "data_sha256")] {
            assert_eq!(
                crate::prepared::source_digest(&directory.join(file)).unwrap(),
                reference[key].as_str().unwrap(),
                "parity.json was made from a different {file}"
            );
        }
        let cache = tempfile::tempdir().unwrap();
        let mut embedder =
            crate::Embedder::load(crate::Profile::Gemma, cache.path()).expect("load gemma");
        let cases = reference["cases"].as_array().unwrap();
        assert!(cases.iter().any(|case| case["kind"] == "query"));
        assert!(cases.iter().any(|case| case["kind"] == "document"));
        assert!(cases.iter().any(|case| case["tokens"] == 512));
        let vector = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect()
        };
        let cosine = |left: &[f32], right: &[f32]| {
            f64::from(crate::projection::dot(left, right))
                / f64::from(
                    crate::projection::dot(left, left).sqrt()
                        * crate::projection::dot(right, right).sqrt(),
                )
        };
        let mut failed = Vec::new();
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let (wheel, unquantized) = (vector(&case["vector"]), vector(&case["fp32"]));
            let actual = match case["kind"].as_str().unwrap() {
                "query" => embedder.embed_query(text).unwrap(),
                _ => embedder.embed_passage(text).unwrap(),
            };
            let (agreement, ours, theirs) = (
                cosine(&actual, &wheel),
                cosine(&actual, &unquantized),
                cosine(&wheel, &unquantized),
            );
            println!(
                "{} {} tokens: against the wheel {agreement:.5}; against fp32 {ours:.5}, \
                 the wheel {theirs:.5}",
                case["kind"], case["tokens"]
            );
            if agreement < 0.997 || ours < theirs - 0.0005 {
                failed.push(text.chars().take(40).collect::<String>());
            }
        }
        assert!(failed.is_empty(), "outside the contract: {failed:?}");
    }

    #[test]
    fn the_full_width_is_untouched_and_any_other_refused() {
        let full: Vec<f32> = (0..768).map(|i| ((i % 7) as f32 - 3.0) / 50.0).collect();
        assert_eq!(checked(full.clone()).unwrap(), full);
        assert!(checked(vec![1.0; 512]).is_err());
    }
}
