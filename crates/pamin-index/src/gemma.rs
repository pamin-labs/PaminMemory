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
//! and normalization, inside the graph, so a vector is used as it comes. Its
//! Matryoshka training makes a prefix of it a vector too: `PAMIN_EG2_DIMS`
//! keeps the first 512, 256 or 128 dimensions and normalizes them again.
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

mod language;

/// The directory the model is read from.
const DIRECTORY: &str = "PAMIN_EG2_DIR";

/// How many of the 768 dimensions to keep.
const DIMENSIONS: &str = "PAMIN_EG2_DIMS";

const MODEL: &str = "model.onnx";
const DATA: &str = "model.onnx.data";
const TOKENIZER: &str = "tokenizer.json";

/// The prompts the model was trained with, for queries and for stored text.
///
/// The document prompt names a title, and memories have one -- the topic. It
/// is `none` here because that is what the retrieval evaluation that chose
/// this export used, and changing it changes every stored vector.
pub(crate) const PREFIXES: (&str, &str) = ("task: search result | query: ", "title: none | text: ");

/// The width the graph returns.
const WIDTH: u32 = 768;

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

/// The vector width this profile stores: 768 unless `PAMIN_EG2_DIMS` names
/// one of the model's Matryoshka widths.
///
/// Any other value falls back to 768 with a warning. The width is part of
/// [`identity`], so a mistyped setting can build an index of the wrong width
/// but never put two widths in one.
pub(crate) fn dimensions() -> u32 {
    static CHOSEN: OnceLock<u32> = OnceLock::new();
    *CHOSEN.get_or_init(|| match std::env::var(DIMENSIONS) {
        Err(_) => WIDTH,
        Ok(value) => match value.trim().parse() {
            Ok(width @ (768 | 512 | 256 | 128)) => width,
            _ => {
                tracing::warn!(%value, "{DIMENSIONS} is not 768, 512, 256 or 128; using 768");
                WIDTH
            }
        },
    })
}

/// The identity recorded with every vector this profile stores.
///
/// A digest of the graph, its weights and its tokenizer, and the width: two
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
            Ok(digest) => named(&digest[..16], dimensions(), language::chosen()),
            Err(_) => "embeddinggemma-2-int8:unconfigured".to_string(),
        }
    })
}

/// The identity of vectors stored at `width` under `transform`.
///
/// A transform applied to stored text makes another space, so it is named;
/// one applied to queries alone leaves stored vectors the graph's, and the
/// identity is what it was before there were transforms.
fn named(digest: &str, width: u32, transform: language::Transform) -> String {
    match transform.stored_tag() {
        None => format!("embeddinggemma-2-int8@{digest}:d{width}"),
        Some(tag) => format!("embeddinggemma-2-int8@{digest}:d{width}:{tag}"),
    }
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

/// A graph vector cut to [`dimensions`] and normalized again.
///
/// At the full width the graph's own normalization stands, untouched.
pub(crate) fn truncated(mut vector: Vec<f32>) -> Result<Vec<f32>> {
    if vector.len() != WIDTH as usize {
        return Err(IndexError::Engine(format!(
            "the gemma export returned {} dimensions, not {WIDTH}",
            vector.len()
        )));
    }
    let width = dimensions() as usize;
    if width < vector.len() {
        vector.truncate(width);
        let norm = crate::projection::dot(&vector, &vector).sqrt();
        if norm.is_nan() || norm <= 0.0 {
            return Err(IndexError::Numerical("a zero gemma vector".into()));
        }
        for value in &mut vector {
            *value /= norm;
        }
    }
    Ok(vector)
}

/// A query vector, already cut by [`truncated`], under the language
/// transform `PAMIN_EG2_QUERY_TRANSFORM` chooses -- see [`language`].
pub(crate) fn query(vector: Vec<f32>) -> Result<Vec<f32>> {
    language::query(language::chosen(), vector)
}

/// A stored text's vector, already cut by [`truncated`]: the graph's own,
/// unless `PAMIN_EG2_QUERY_TRANSFORM` names a `both-*` variant -- see
/// [`language`].
pub(crate) fn stored(vector: Vec<f32>) -> Result<Vec<f32>> {
    language::stored(language::chosen(), vector)
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
        assert_eq!(dimensions(), WIDTH, "compare at the graph's own width");
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
            // The graph's vector: the language transform comes after it, and
            // is checked against its own reference in `language`.
            let actual = match case["kind"].as_str().unwrap() {
                "query" => {
                    let raw = embedder
                        .embed_one(&format!("{}{text}", PREFIXES.0))
                        .unwrap();
                    assert_eq!(
                        embedder.embed_query(text).unwrap(),
                        query(raw.clone()).unwrap()
                    );
                    raw
                }
                _ => {
                    let raw = embedder
                        .embed_one(&format!("{}{text}", PREFIXES.1))
                        .unwrap();
                    assert_eq!(
                        embedder.embed_passage(text).unwrap(),
                        stored(raw.clone()).unwrap()
                    );
                    raw
                }
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
    fn a_shorter_width_is_renormalized_and_the_full_one_is_untouched() {
        let full: Vec<f32> = (0..768).map(|i| ((i % 7) as f32 - 3.0) / 50.0).collect();
        // The process-wide width is whatever the environment says; only the
        // full width's behaviour is fixed, so test that and the arithmetic.
        if dimensions() == WIDTH {
            assert_eq!(truncated(full.clone()).unwrap(), full);
        } else {
            let cut = truncated(full).unwrap();
            assert_eq!(cut.len(), dimensions() as usize);
            assert!((crate::projection::dot(&cut, &cut) - 1.0).abs() < 1e-5);
        }
        assert!(truncated(vec![1.0; 512]).is_err());
    }

    /// A variant that changes stored vectors changes the identity, so an
    /// index built under it never opens under another; one that changes only
    /// queries, and `off`, leave it as it was before there were variants.
    #[test]
    fn the_identity_names_a_stored_transform_and_only_that() {
        use language::Transform::*;
        let before = "embeddinggemma-2-int8@0123456789abcdef:d768";
        for transform in [Off, Center, CenterLsar1, CenterLsar2, InterpLsar2] {
            assert_eq!(named("0123456789abcdef", 768, transform), before);
        }
        let one = named("0123456789abcdef", 768, BothCenterLsar1);
        let two = named("0123456789abcdef", 768, BothCenterLsar2);
        assert_ne!(one, before);
        assert_ne!(two, before);
        assert_ne!(one, two);
        assert_ne!(
            named("0123456789abcdef", 512, BothCenterLsar2),
            two,
            "the width stays part of it"
        );
    }
}
