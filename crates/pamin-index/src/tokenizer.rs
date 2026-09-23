//! One loaded copy of a vocabulary, however many models tokenize with it.
//!
//! With its weights mapped from a prepared copy (see `crate::prepared`), what
//! an XLM-R-family model still holds is almost all tokenizer. Loading the
//! 250,002-piece Unigram vocabulary that BGE-M3 and the `accurate` reranker
//! both use adds 280 MiB of anonymous memory, and loading it a second time adds
//! another 280: the `tokenizers` crate builds a trie over every piece, with a
//! hash map in each of its nodes, and a model holds its own. Against that, a
//! bare ONNX Runtime session over a prepared copy holds about 12 MiB.
//!
//! Every model that runs through `crate::encoder` uses that vocabulary. The
//! `accurate`, `balanced` and `noncommercial` rerankers' `tokenizer.json`
//! describe the same model as BGE-M3's -- every piece and every score bit for
//! bit, the same unknown-token id, no byte fallback -- and the `fast`
//! reranker's differs only in leaving the byte-fallback flag unstated. What
//! they differ in is around the model: the `accurate` reranker strips trailing
//! whitespace and replaces a run of spaces with `▁` where the embedder
//! replaces it with one space. So the model is what is shared, and the
//! normalizer, pre-tokenizer, post-processor, added tokens, padding and
//! truncation stay each tokenizer's own.
//!
//! Sharing it changes no output. A Unigram model segments the text it is
//! handed as a function of its pieces and their scores; its one piece of
//! mutable state is a cache of those segmentations, keyed by the text, which
//! returns what segmenting would have.
//!
//! A model is found by what it contains rather than by which repository it
//! came from: a SHA-256 over its section of `tokenizer.json`, parsed and
//! written back out (see [`canonical`]), so two files' differing layout cannot
//! hide that they hold the same vocabulary, and a vocabulary that differs in
//! one score is a different key. The registry holds it weakly. It is freed
//! when the last model using it is -- an idle model released by the engine
//! takes its share of the vocabulary with it -- and a later load builds it
//! again.

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use tokenizers::{
    AddedToken, DecoderWrapper, Model, ModelWrapper, NormalizerWrapper, PaddingParams,
    PaddingStrategy, PostProcessorWrapper, PreTokenizedString, PreTokenizerWrapper, Token,
    TokenizerImpl, TruncationDirection, TruncationParams,
};

use crate::error::{IndexError, Result};
use crate::hub::Repository;

/// A tokenizer whose model may be shared with other tokenizers.
///
/// `tokenizers::Tokenizer` with the model behind an `Arc`: the crate's own
/// type holds its model by value, so two of them cannot share one.
pub(crate) type Tokenizer = TokenizerImpl<
    Shared,
    NormalizerWrapper,
    PreTokenizerWrapper,
    PostProcessorWrapper,
    DecoderWrapper,
>;

/// A tokenizer model, shared with every other loaded tokenizer whose model
/// section is the same.
///
/// Deserializing one is what finds or builds it -- see the module
/// documentation -- so a tokenizer read from its file through
/// `TokenizerImpl`'s own parsing gets its added tokens, normalizer and the
/// rest exactly as `tokenizers::Tokenizer` would, and only the model differs.
pub(crate) struct Shared(Arc<ModelWrapper>);

/// The models loaded in this process, by the digest of what they contain.
///
/// A list rather than a map: it holds one entry per distinct vocabulary a
/// process has loaded, which is one or two.
static LOADED: Mutex<Vec<([u8; 32], Weak<ModelWrapper>)>> = Mutex::new(Vec::new());

impl<'de> Deserialize<'de> for Shared {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let described = canonical(serde_json::Value::deserialize(deserializer)?);
        let mut digest = Sha256::new();
        serde_json::to_writer(&mut digest, &described).map_err(serde::de::Error::custom)?;
        let key: [u8; 32] = digest.finalize().into();

        // Held while a model is built, so two tokenizers loading the same
        // vocabulary at once build it once: the second waits and finds it.
        let mut loaded = LOADED.lock().unwrap_or_else(PoisonError::into_inner);
        loaded.retain(|(_, model)| model.strong_count() > 0);
        if let Some(model) = loaded
            .iter()
            .find(|(held, _)| *held == key)
            .and_then(|(_, model)| model.upgrade())
        {
            return Ok(Self(model));
        }
        let model =
            Arc::new(ModelWrapper::deserialize(described).map_err(serde::de::Error::custom)?);
        loaded.push((key, Arc::downgrade(&model)));
        Ok(Self(model))
    }
}

/// A model section with what its parser would assume written out, so that
/// two files describing the same model hash alike.
///
/// Only Unigram's, the one this crate's models use. Its parser reads four
/// fields, ignores any other, takes a missing unknown-token id as none and a
/// missing `byte_fallback` as false -- and the `fast` reranker's file and the
/// E5 models' leave `byte_fallback` out where BGE-M3's states it, over the
/// same 250,002 pieces and scores. The model is then built from this, not
/// from the file, so the key is a digest of exactly what was built. Any other
/// type is hashed as written, which can only miss a match, never make one.
fn canonical(mut model: serde_json::Value) -> serde_json::Value {
    if model["type"] == "Unigram"
        && let serde_json::Value::Object(fields) = &mut model
    {
        fields.retain(|field, _| {
            matches!(
                field.as_str(),
                "type" | "vocab" | "unk_id" | "byte_fallback"
            )
        });
        fields.entry("unk_id").or_insert(serde_json::Value::Null);
        fields
            .entry("byte_fallback")
            .or_insert(serde_json::Value::Bool(false));
    }
    model
}

/// Every method is the shared model's own, including the one with a default
/// the trait provides, so a tokenizer over this runs exactly the code it would
/// over the model held by value.
impl Model for Shared {
    type Trainer = <ModelWrapper as Model>::Trainer;

    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        self.0.tokenize(sequence)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0.id_to_token(id)
    }

    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        self.0.get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.0.get_vocab_size()
    }

    fn save(
        &self,
        folder: &Path,
        prefix: Option<&str>,
    ) -> tokenizers::Result<Vec<std::path::PathBuf>> {
        self.0.save(folder, prefix)
    }

    fn get_trainer(&self) -> Self::Trainer {
        self.0.get_trainer()
    }

    fn tokenize_in_pretokenized(
        &self,
        pretokenized: &mut PreTokenizedString,
        truncation: Option<(usize, TruncationDirection)>,
    ) -> tokenizers::Result<()> {
        self.0.tokenize_in_pretokenized(pretokenized, truncation)
    }
}

/// A model repository's tokenizer, truncating at `max_length` tokens.
pub(crate) fn load(repository: &Repository, max_length: usize) -> Result<Tokenizer> {
    let read = |file: &str| -> Result<Vec<u8>> { Ok(std::fs::read(repository.get(file)?)?) };
    build(
        &read("tokenizer.json")?,
        &read("config.json")?,
        &read("special_tokens_map.json")?,
        &read("tokenizer_config.json")?,
        max_length,
    )
}

/// A tokenizer from its four files, configured the way `fastembed` 6.1's
/// `load_tokenizer` configures one.
///
/// Every model that runs through `crate::encoder` used to be loaded by
/// `fastembed`, and what that function decides -- where a text is truncated,
/// what it is padded with, which special tokens are registered -- decides the
/// ids a model sees. So it is followed step for step rather than improved on:
/// the length limit is the smaller of `max_length` and the tokenizer's own
/// `model_max_length`, a batch is padded to its longest member with the pad
/// token and the id `config.json` names, and every entry of
/// `special_tokens_map.json` is added as a special token -- one given as an
/// object only when it states all five of its flags.
fn build(
    tokenizer: &[u8],
    config: &[u8],
    special_tokens_map: &[u8],
    tokenizer_config: &[u8],
    max_length: usize,
) -> Result<Tokenizer> {
    fn invalid(file: &str, error: impl std::fmt::Display) -> IndexError {
        IndexError::Engine(format!("reading the tokenizer's {file}: {error}"))
    }
    let json = |file: &str, bytes: &[u8]| -> Result<serde_json::Value> {
        serde_json::from_slice(bytes).map_err(|error| invalid(file, error))
    };
    let config = json("config.json", config)?;
    let special_tokens_map = json("special_tokens_map.json", special_tokens_map)?;
    let tokenizer_config = json("tokenizer_config.json", tokenizer_config)?;
    let mut tokenizer =
        Tokenizer::from_bytes(tokenizer).map_err(|error| invalid("tokenizer.json", error))?;

    // Through `f32` because that is the conversion `fastembed` makes: a model
    // that declares no real limit writes 1e30 here, which fits an `f64`.
    let model_max_length = tokenizer_config["model_max_length"]
        .as_f64()
        .ok_or_else(|| invalid("tokenizer_config.json", "no numeric `model_max_length`"))?
        as f32;
    let max_length = max_length.min(model_max_length as usize);
    let pad_id = config["pad_token_id"].as_u64().unwrap_or(0) as u32;
    let pad_token = tokenizer_config["pad_token"]
        .as_str()
        .ok_or_else(|| invalid("tokenizer_config.json", "no string `pad_token`"))?
        .to_string();

    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_token,
            pad_id,
            ..Default::default()
        }))
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|error| invalid("tokenizer_config.json", error))?;

    if let serde_json::Value::Object(tokens) = special_tokens_map {
        for value in tokens.values() {
            let token = match value {
                serde_json::Value::String(content) => Some(AddedToken {
                    content: content.clone(),
                    special: true,
                    ..Default::default()
                }),
                serde_json::Value::Object(_) => match (
                    value["content"].as_str(),
                    value["single_word"].as_bool(),
                    value["lstrip"].as_bool(),
                    value["rstrip"].as_bool(),
                    value["normalized"].as_bool(),
                ) {
                    (
                        Some(content),
                        Some(single_word),
                        Some(lstrip),
                        Some(rstrip),
                        Some(normalized),
                    ) => Some(AddedToken {
                        content: content.into(),
                        special: true,
                        single_word,
                        lstrip,
                        rstrip,
                        normalized,
                    }),
                    _ => None,
                },
                _ => None,
            };
            if let Some(token) = token {
                tokenizer
                    .add_special_tokens([token])
                    .map_err(|error| invalid("special_tokens_map.json", error))?;
            }
        }
    }
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Unigram tokenizer over a handful of pieces, with a normalizer of the
    /// caller's choosing -- the one part two tokenizers sharing a model differ
    /// in.
    fn tokenizer_json(normalizer: &str, scores: [f64; 3]) -> Vec<u8> {
        format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {{"id": 0, "content": "<s>", "single_word": false, "lstrip": false,
                      "rstrip": false, "normalized": false, "special": true}},
                    {{"id": 1, "content": "<pad>", "single_word": false, "lstrip": false,
                      "rstrip": false, "normalized": false, "special": true}},
                    {{"id": 2, "content": "</s>", "single_word": false, "lstrip": false,
                      "rstrip": false, "normalized": false, "special": true}}
                ],
                "normalizer": {normalizer},
                "pre_tokenizer": {{"type": "Metaspace", "replacement": "▁",
                                   "prepend_scheme": "always", "split": true}},
                "post_processor": {{"type": "RobertaProcessing", "sep": ["</s>", 2],
                                    "cls": ["<s>", 0], "trim_offsets": true,
                                    "add_prefix_space": true}},
                "decoder": null,
                "model": {{
                    "type": "Unigram",
                    "unk_id": 3,
                    "vocab": [["<s>", 0.0], ["<pad>", 0.0], ["</s>", 0.0], ["<unk>", 0.0],
                              ["▁memory", {}], ["▁mem", {}], ["ory", {}],
                              ["▁", -3.0], ["m", -4.0], ["e", -4.0], ["o", -4.0],
                              ["r", -4.0], ["y", -4.0]],
                    "byte_fallback": false
                }}
            }}"#,
            scores[0], scores[1], scores[2]
        )
        .into_bytes()
    }

    const SCORES: [f64; 3] = [-1.0, -2.0, -2.0];
    const STRIP: &str = r#"{"type": "Strip", "strip_left": false, "strip_right": true}"#;
    const CONFIG: &[u8] = br#"{"pad_token_id": 1}"#;
    const SPECIAL: &[u8] = br#"{"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"}"#;
    const TOKENIZER_CONFIG: &[u8] = br#"{"model_max_length": 512, "pad_token": "<pad>"}"#;

    fn load(json: &[u8]) -> Tokenizer {
        build(json, CONFIG, SPECIAL, TOKENIZER_CONFIG, 8).expect("build the tokenizer")
    }

    /// Two tokenizers whose model sections match share one model, whatever
    /// else about them differs; one whose vocabulary differs in a single
    /// score does not.
    #[test]
    fn a_vocabulary_is_loaded_once_and_only_for_its_own_content() {
        // A score no other test uses, so a model another test loaded cannot
        // stand in for this one.
        let scores = [-1.25, -2.0, -2.0];
        let plain = load(&tokenizer_json("null", scores));
        let stripping = load(&tokenizer_json(STRIP, scores));
        assert!(
            Arc::ptr_eq(&plain.get_model().0, &stripping.get_model().0),
            "two tokenizers with the same vocabulary hold two copies of it"
        );

        // The `fast` reranker's file leaves this flag out where BGE-M3's
        // states its default.
        let unstated = String::from_utf8(tokenizer_json("null", scores))
            .expect("utf-8")
            .replace(",\n                    \"byte_fallback\": false", "");
        assert!(
            !unstated.contains("byte_fallback"),
            "the flag is still there"
        );
        let unstated = load(unstated.as_bytes());
        assert!(
            Arc::ptr_eq(&plain.get_model().0, &unstated.get_model().0),
            "a file leaving a flag at its default was not recognised as the same vocabulary"
        );

        let other = load(&tokenizer_json("null", [-1.25, -2.0, -2.5]));
        assert!(
            !Arc::ptr_eq(&plain.get_model().0, &other.get_model().0),
            "a different vocabulary was handed a shared one"
        );
    }

    /// Once no tokenizer holds a model, it is freed rather than kept by the
    /// registry.
    #[test]
    fn a_vocabulary_nobody_holds_is_freed() {
        let tokenizer = load(&tokenizer_json("null", [-1.5, -2.0, -2.0]));
        let model = Arc::downgrade(&tokenizer.get_model().0);
        drop(tokenizer);
        assert!(
            model.upgrade().is_none(),
            "the registry kept the model alive"
        );
    }

    /// A tokenizer over a shared model encodes exactly as the crate's own
    /// `Tokenizer` does over the same file, with the configuration applied the
    /// same way.
    #[test]
    fn a_shared_model_encodes_as_the_model_held_by_value() {
        let json = tokenizer_json(STRIP, SCORES);
        let shared = load(&json);

        let mut owned = tokenizers::Tokenizer::from_bytes(&json).expect("the crate's own");
        owned
            .with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_token: "<pad>".into(),
                pad_id: 1,
                ..Default::default()
            }))
            .with_truncation(Some(TruncationParams {
                max_length: 8,
                ..Default::default()
            }))
            .expect("truncation");
        for content in ["<s>", "</s>", "<pad>"] {
            owned
                .add_special_tokens([AddedToken {
                    content: content.into(),
                    special: true,
                    ..Default::default()
                }])
                .expect("a special token");
        }

        let texts = vec![
            "memory",
            "memory memory  ",
            "",
            "mem ory mymory memory memory memory memory",
        ];
        let pairs: Vec<(&str, &str)> = texts.iter().map(|text| ("memory ", *text)).collect();
        let ids = |encodings: Vec<tokenizers::Encoding>| -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
            encodings
                .into_iter()
                .map(|encoding| {
                    (
                        encoding.get_ids().to_vec(),
                        encoding.get_attention_mask().to_vec(),
                        encoding.get_type_ids().to_vec(),
                    )
                })
                .collect()
        };
        assert_eq!(
            ids(shared.encode_batch(texts.clone(), true).expect("encode")),
            ids(owned.encode_batch(texts, true).expect("encode"))
        );
        assert_eq!(
            ids(shared.encode_batch(pairs.clone(), true).expect("encode")),
            ids(owned.encode_batch(pairs, true).expect("encode"))
        );
    }
}
