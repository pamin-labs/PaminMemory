//! EmbeddingGemma 2, int8: the experimental `gemma` profile.
//!
//! Fetched from the hub on first use like every other model here (see
//! `crate::hub`), from `pamin-labs/embeddinggemma-2-text-int8-onnx` pinned to
//! one commit, each file checked against the size and SHA-256 that commit
//! publishes: `model.onnx`, the `model.onnx.data` it names, and the
//! tokenizer's `tokenizer.json`, `config.json` and `tokenizer_config.json`.
//!
//! The graph takes `input_ids` and `attention_mask` and returns
//! `sentence_embedding`, the attention-masked mean of the last hidden state
//! projected to 768 dimensions and L2-normalized -- the model's own pooling
//! and normalization, inside the graph, so a vector is used as it comes, at
//! its full width. Its Matryoshka training makes a prefix a vector too, but
//! 512 dimensions measured below 768 through the search entry point (XQuAD-R
//! cross-lingual nDCG@10 -0.0028, same-language -0.0017), so no width is cut.
//! Those figures, and every other measured on this profile before the pinned
//! revision, were taken on the export's first revision (`2d2e08e`), which
//! differs from the pinned one in `model.onnx` only.
//!
//! What this export is: every matrix int8 with per-token u8 activations, the
//! rotary embedding and grouped-query attention as ONNX Runtime's contrib
//! operators, opset 21. Activations have to be unsigned on x86: the signed
//! kernels run at a hundredth of the speed. The pinned revision quantizes
//! every activation per token, so its card reports a text's vector bitwise
//! identical alone and in a batch; the first revision shared one scale across
//! a batch. The profile still runs one text per pass, as `crate::embedding`
//! does for BGE-M3: batching it has not been measured here.

use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};
use crate::hub::{Pinned, Repository};

mod language;

/// Where the export is published, and the commit it is read at: a commit,
/// not a tag, so the files cannot move under an index built from them.
const REPOSITORY: &str = "pamin-labs/embeddinggemma-2-text-int8-onnx";
const REVISION: &str = "39c3eaaf07991c73bfe0e489b55b1b3e6aaa9ac5";

const MODEL: Pinned = Pinned {
    name: "model.onnx",
    size: 1_266_037,
    sha256: "1b77c1d1548181dec3b9a38997794512665e06f624cbf1ddaa3334c6941481a2",
};
const DATA: Pinned = Pinned {
    name: "model.onnx.data",
    size: 273_266_688,
    sha256: "ae126793db6cb0107fe9711c9d5a820126a0cc1a336172c4871a44d4f1daa763",
};
const TOKENIZER: Pinned = Pinned {
    name: "tokenizer.json",
    size: 32_170_510,
    sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
};
const TOKENIZER_CONFIG: Pinned = Pinned {
    name: "tokenizer_config.json",
    size: 1_599,
    sha256: "17bd5d6e9364ca49a534e1502076593317c298d4a663623091ed45388f004874",
};
const CONFIG: Pinned = Pinned {
    name: "config.json",
    size: 4_455,
    sha256: "b8f1e9931b57fbc054acdb445c41765d55b0074c58d145fa82839941ad1b5bb3",
};

/// The prompts the model was trained with, for queries and for stored text.
///
/// The document prompt names a title, and memories have one -- the topic. It
/// is `none` here because that is what the retrieval evaluation that chose
/// this export used, and changing it changes every stored vector.
pub(crate) const PREFIXES: (&str, &str) = ("task: search result | query: ", "title: none | text: ");

/// The width the graph returns, and the width this profile stores.
pub(crate) const WIDTH: u32 = 768;

/// The pinned export, cached under `cache_dir`.
pub(crate) fn repository(cache_dir: &std::path::Path) -> Result<Repository> {
    Repository::open_at(cache_dir, REPOSITORY, REVISION)
}

/// The identity recorded with every vector this profile stores.
///
/// A digest of the pinned SHA-256 of the graph, its weights and its
/// tokenizer, and the width: the same digest the profile took of those files
/// when it read them from a local directory, so an index built from these
/// files keeps its identity, and one built from the first revision's graph
/// (`embeddinggemma-2-int8@13db5bcca6ab6934:d768`) is refused. Two revisions
/// of this export can differ in the graph alone, as those two do, and each
/// produces its own space.
pub(crate) fn identity() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let combined: String = [MODEL, DATA, TOKENIZER]
            .iter()
            .map(|file| file.sha256)
            .collect();
        let digest = format!("{:x}", Sha256::digest(combined.as_bytes()));
        format!("embeddinggemma-2-int8@{}:d{WIDTH}", &digest[..16])
    })
}

/// The model on `target`, with its tokenizer, truncating at `max_length`,
/// downloading the pinned files first if they are not cached.
pub(crate) fn load(
    cache_dir: &std::path::Path,
    max_length: usize,
    target: impl Into<crate::inference::Target>,
) -> Result<Encoder> {
    let repository = repository(cache_dir)?;
    let model = repository.get_pinned(&MODEL)?;
    for file in [&DATA, &TOKENIZER, &TOKENIZER_CONFIG, &CONFIG] {
        repository.get_pinned(file)?;
    }
    // One snapshot directory holds every file of the revision; the weights
    // are found beside the graph, as the graph names them.
    let directory = model
        .parent()
        .ok_or_else(|| IndexError::Engine("a model file with no directory".into()))?;
    let tokenizer = crate::tokenizer::load_dir(directory, max_length)?;
    Encoder::load_with(|| Ok(model), tokenizer, target)
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

/// A query vector, already checked by [`checked`], under the language
/// transform -- see [`language`]. Stored vectors are the graph's own, so the
/// transform is not part of [`identity`].
pub(crate) fn query(vector: Vec<f32>) -> Result<Vec<f32>> {
    language::query(vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The product's embedder runs the graph as well as the Python wheel does.
    ///
    /// The reference is the `parity.json` file `PAMIN_TEST_EG2_PARITY` names:
    /// for each text, the vector the pinned graph returned under ONNX Runtime
    /// 1.28's Python wheel and the vector the unquantized export of the same
    /// model returned, one text per pass, through the Hugging Face tokenizer,
    /// with both prompts, truncated at 512 tokens. It names the graph and
    /// weights it came from, and the test refuses a reference made from other
    /// files than the pinned ones. The model is read from, or downloaded into,
    /// `PAMIN_TEST_MODEL_CACHE`.
    ///
    /// Not bit-identical to the wheel, and not within the 0.9999 an
    /// accelerator is held to: the binary this crate links is another build
    /// of the same runtime release, and every activation here is rounded to
    /// eight bits, so a last-bit difference in a float kernel moves a rounding
    /// boundary. Measured on these texts with the pinned graph, this build and
    /// the wheel agree at 0.9980 to 0.9997, and they are 0.9979 to 0.9994 and
    /// 0.9981 to 0.9995 from the unquantized model -- two draws of the same
    /// quantization error, neither nearer the model. The unquantized export of the same model matches the wheel to
    /// 1e-7 here. So the contract is that: no further from the unquantized
    /// vector than the wheel is, give or take 0.0005, and 0.997 from the
    /// wheel's -- which a wrong prompt, pooling, width or truncation is not.
    #[test]
    #[ignore = "downloads the pinned EmbeddingGemma 2 export; set PAMIN_TEST_MODEL_CACHE and PAMIN_TEST_EG2_PARITY"]
    fn gemma_matches_the_python_reference() {
        let reference: serde_json::Value = serde_json::from_slice(
            &std::fs::read(
                std::env::var_os("PAMIN_TEST_EG2_PARITY").expect("set PAMIN_TEST_EG2_PARITY"),
            )
            .expect("read parity.json"),
        )
        .expect("parse parity.json");
        for (file, key) in [(MODEL, "model_sha256"), (DATA, "data_sha256")] {
            assert_eq!(
                file.sha256,
                reference[key].as_str().unwrap(),
                "parity.json was made from a different {}",
                file.name
            );
        }
        let cache = std::path::PathBuf::from(
            std::env::var_os("PAMIN_TEST_MODEL_CACHE").expect("set PAMIN_TEST_MODEL_CACHE"),
        );
        let mut embedder =
            crate::Embedder::load(crate::Profile::Gemma, &cache).expect("load gemma");
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

    /// The identity is the digest the profile took of the files themselves:
    /// the first revision's hashes give the identity indexes were built
    /// under then, and the pinned revision's another.
    #[test]
    fn the_identity_names_the_pinned_graph() {
        let first = format!(
            "{:x}",
            Sha256::digest(
                [
                    "42c290bdd40f310737899cb6d5ea45d454bd30cadf34a39f6a22ec70ab266248",
                    DATA.sha256,
                    TOKENIZER.sha256
                ]
                .concat()
                .as_bytes()
            )
        );
        assert_eq!(&first[..16], "13db5bcca6ab6934");
        assert_eq!(identity(), "embeddinggemma-2-int8@8a326248ba277cb7:d768");
    }

    #[test]
    fn the_full_width_is_untouched_and_any_other_refused() {
        let full: Vec<f32> = (0..768).map(|i| ((i % 7) as f32 - 3.0) / 50.0).collect();
        assert_eq!(checked(full.clone()).unwrap(), full);
        assert!(checked(vec![1.0; 512]).is_err());
    }
}
