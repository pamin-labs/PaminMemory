//! One loaded copy of a vocabulary, however many models tokenize with it, in
//! a more compact layout than the `tokenizers` crate's own. With BGE-M3 and
//! the `accurate` reranker loaded, the process holds about 250 MiB less
//! anonymous memory with it (see `docs/measured.md`).
//!
//! With its weights mapped from a prepared copy (see `crate::prepared`), what
//! an XLM-R-family model still holds is almost all tokenizer. The `tokenizers`
//! crate's Unigram model over the 250,002-piece vocabulary that BGE-M3 and the
//! `accurate` reranker both use adds 280 MiB of anonymous memory: it builds a
//! trie over every piece with a hash map in each of its nodes, and keeps every
//! piece twice more, in a list and in a map from piece to id. A bare ONNX
//! Runtime session over a prepared copy holds about 12 MiB.
//!
//! So a Unigram vocabulary is held here as a [`Unigram`]: the pieces end to
//! end with their scores, and a double-array trie (`yada`) from each piece to
//! its id, which is a few bytes a piece where a hash-map node is a few
//! hundred. It segments a text step for step as `tokenizers`' own model does,
//! so the ids are the same; see [`Unigram::tokenize`].
//!
//! Every model that runs through `crate::encoder` uses that vocabulary but
//! one, whose model is not Unigram and is held as `tokenizers` builds it. The
//! `accurate` reranker's `tokenizer.json` describes the same model as
//! BGE-M3's -- every piece and every score bit for bit, the same
//! unknown-token id, no byte fallback -- and the `fast`
//! reranker's differs only in leaving the byte-fallback flag unstated. What
//! they differ in is around the model: the `accurate` reranker strips trailing
//! whitespace and replaces a run of spaces with `▁` where the embedder
//! replaces it with one space. So the model is what is shared, and the
//! normalizer, pre-tokenizer, post-processor, added tokens, padding and
//! truncation stay each tokenizer's own, and are `tokenizers`' own.
//!
//! Sharing it changes no output. A Unigram model segments the text it is
//! handed as a function of its pieces and their scores, and this one holds no
//! mutable state at all.
//!
//! A model is found by what it contains rather than by which repository it
//! came from: a SHA-256 over its section of `tokenizer.json`, parsed and
//! written back out (see [`canonical`]), so two files' differing layout cannot
//! hide that they hold the same vocabulary, and a vocabulary that differs in
//! one score is a different key. The registry holds it weakly. It is freed
//! when the last model using it is -- an idle model released by the engine
//! takes its share of the vocabulary with it -- and a later load builds it
//! again.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};
use tokenizers::models::TrainerWrapper;
use tokenizers::models::unigram::{UnigramError, UnigramTrainer};
use tokenizers::{
    AddedToken, DecoderWrapper, Model, ModelWrapper, NormalizerWrapper, PaddingParams,
    PaddingStrategy, PostProcessorWrapper, PreTokenizerWrapper, Token, TokenizerImpl,
    TruncationParams,
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
pub(crate) struct Shared(Arc<Vocabulary>);

/// A loaded model: a Unigram vocabulary in this module's layout, or any other
/// model as `tokenizers` builds it.
#[expect(
    clippy::large_enum_variant,
    reason = "one per loaded vocabulary, behind an `Arc`"
)]
enum Vocabulary {
    Unigram(Unigram),
    Other(ModelWrapper),
}

/// The models loaded in this process, by the digest of what they contain.
///
/// A list rather than a map: it holds one entry per distinct vocabulary a
/// process has loaded, which is one or two.
static LOADED: Mutex<Vec<([u8; 32], Weak<Vocabulary>)>> = Mutex::new(Vec::new());

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
        let model = Arc::new(Vocabulary::new(described).map_err(serde::de::Error::custom)?);
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

impl Vocabulary {
    /// The model `described` -- a [`canonical`] model section -- describes.
    ///
    /// A Unigram vocabulary [`Unigram::new`] cannot hold, and every other
    /// type, is built by `tokenizers`.
    fn new(described: serde_json::Value) -> serde_json::Result<Self> {
        #[derive(Deserialize)]
        struct Fields {
            vocab: Vec<(String, f64)>,
            unk_id: Option<usize>,
            byte_fallback: bool,
        }
        if described["type"] != "Unigram" {
            return ModelWrapper::deserialize(described).map(Self::Other);
        }
        // By value, so the pieces are moved out of the parsed file rather
        // than copied beside it.
        let Fields {
            vocab,
            unk_id,
            byte_fallback,
        } = Fields::deserialize(described)?;
        Ok(match Unigram::new(&vocab, unk_id, byte_fallback) {
            Some(model) => Self::Unigram(model),
            None => Self::Other(
                tokenizers::models::unigram::Unigram::from(vocab, unk_id, byte_fallback)
                    .map_err(serde::de::Error::custom)?
                    .into(),
            ),
        })
    }
}

/// A Unigram model that segments exactly as `tokenizers`' own, in a few
/// bytes a piece.
///
/// What `tokenizers::models::unigram::Unigram` holds a piece in three times
/// -- a list of pieces and scores, a map from piece to id, and a trie with a
/// hash map in every node -- this holds once: the pieces end to end, their
/// scores, and a double-array trie from each piece's bytes to its id, which
/// answers both the prefix search a segmentation is made of and the lookup of
/// one piece's id.
struct Unigram {
    /// Each piece's id, by its bytes. A piece listed twice is found under its
    /// later id, as `tokenizers`' map from piece to id finds it.
    trie: yada::DoubleArray<Vec<u8>>,
    /// The pieces end to end, in id order.
    pieces: String,
    /// Where each piece ends in `pieces`, by id.
    ends: Vec<u32>,
    scores: Vec<f64>,
    unk_id: Option<u32>,
    /// What a character no piece starts with scores: ten below the lowest
    /// score, as `tokenizers` and SentencePiece score it.
    unk_score: f64,
    byte_fallback: bool,
}

impl Unigram {
    /// The model over `vocab`, or `None` for a vocabulary `tokenizers` would
    /// reject or that the trie cannot hold exactly.
    ///
    /// The trie holds no empty key and no key with a NUL byte; `tokenizers`'
    /// trie never offers the first as a match and does hold the second, so a
    /// vocabulary with either is left to it. No model this crate loads has
    /// one.
    fn new(vocab: &[(String, f64)], unk_id: Option<usize>, byte_fallback: bool) -> Option<Self> {
        if vocab.is_empty()
            || u32::try_from(vocab.len()).is_err()
            || unk_id.is_some_and(|id| id >= vocab.len())
            || vocab
                .iter()
                .any(|(piece, _)| piece.is_empty() || piece.contains('\0'))
        {
            return None;
        }
        let mut keys: Vec<(&[u8], u32)> = (0..)
            .zip(vocab)
            .map(|(id, (piece, _))| (piece.as_bytes(), id))
            .collect();
        // Sorted by key, the later id of a key first, so deduplicating keeps
        // the later.
        keys.sort_unstable_by(|a, b| a.0.cmp(b.0).then(b.1.cmp(&a.1)));
        keys.dedup_by(|next, kept| next.0 == kept.0);
        let trie =
            yada::DoubleArray::new(yada::builder::DoubleArrayBuilder::build(&keys).ok()?).ok()?;
        drop(keys);

        let mut pieces = String::with_capacity(vocab.iter().map(|(piece, _)| piece.len()).sum());
        let mut ends = Vec::with_capacity(vocab.len());
        let mut scores = Vec::with_capacity(vocab.len());
        let mut min_score = f64::INFINITY;
        for (piece, score) in vocab {
            pieces.push_str(piece);
            ends.push(u32::try_from(pieces.len()).ok()?);
            scores.push(*score);
            if *score < min_score {
                min_score = *score;
            }
        }
        Some(Self {
            trie,
            pieces,
            ends,
            scores,
            unk_id: unk_id.map(|id| id as u32),
            unk_score: min_score - 10.0,
            byte_fallback,
        })
    }

    fn piece(&self, id: u32) -> Option<&str> {
        let id = id as usize;
        let end = *self.ends.get(id)? as usize;
        let start = id
            .checked_sub(1)
            .map_or(0, |before| self.ends[before] as usize);
        Some(&self.pieces[start..end])
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        // The trie takes neither as a key, and holds no piece that is either.
        if token.is_empty() || token.contains('\0') {
            return None;
        }
        self.trie.exact_match_search(token)
    }

    /// `sentence` segmented as `tokenizers`' `Unigram::tokenize` segments it.
    ///
    /// Step for step its `encode_optimized` -- SentencePiece's Viterbi
    /// without a lattice -- and then its `tokenize`, so that every tie between
    /// two equally scored segmentations falls the same way: start positions
    /// visited in order, a piece's candidates shortest first, an end position
    /// taken over only by a strictly higher score. Where it would allocate a
    /// string for each piece the trie matches and look its id up in a map,
    /// this has the id from the trie.
    fn tokenize(&self, sentence: &str) -> tokenizers::Result<Vec<Token>> {
        #[derive(Clone)]
        struct Best {
            id: u32,
            score: f64,
            start: Option<usize>,
        }
        fn offer(node: &mut Best, id: u32, score: f64, start: usize) {
            if node.start.is_none() || score > node.score {
                *node = Best {
                    id,
                    score,
                    start: Some(start),
                };
            }
        }
        let bytes = sentence.as_bytes();
        let mut best = vec![
            Best {
                id: 0,
                score: 0.0,
                start: None,
            };
            bytes.len() + 1
        ];
        // The trie is searched no further than the next NUL byte: no piece
        // holds one, and the trie cannot be asked about one.
        let nul_from = |start: usize| {
            bytes[start..]
                .iter()
                .position(|&byte| byte == 0)
                .map_or(bytes.len(), |at| start + at)
        };
        let mut nul = nul_from(0);
        let mut start = 0;
        while start < bytes.len() {
            if start > nul {
                nul = nul_from(start);
            }
            let here = best[start].score;
            let char_len = sentence[start..]
                .chars()
                .next()
                .expect("a character")
                .len_utf8();
            let mut single = false;
            for (id, len) in self.trie.common_prefix_search(&bytes[start..nul]) {
                offer(
                    &mut best[start + len],
                    id,
                    self.scores[id as usize] + here,
                    start,
                );
                single |= len == char_len;
            }
            if !single {
                let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                offer(
                    &mut best[start + char_len],
                    unk_id,
                    self.unk_score + here,
                    start,
                );
            }
            start += char_len;
        }

        let mut path = Vec::new();
        let mut end = bytes.len();
        while end > 0 {
            let node = &best[end];
            let start = node.start.expect("every character boundary is reached");
            path.push((start, end, node.id));
            end = start;
        }
        path.reverse();

        // A run of unknown pieces is one token, as `tokenizers` fuses them,
        // with the id its text has if it has one.
        let mut tokens = Vec::with_capacity(path.len());
        let mut pieces = path.into_iter().peekable();
        while let Some((start, mut end, id)) = pieces.next() {
            if Some(id) != self.unk_id {
                tokens.push(Token::new(id, sentence[start..end].into(), (start, end)));
                continue;
            }
            while let Some((_, next_end, _)) =
                pieces.next_if(|&(_, _, next)| Some(next) == self.unk_id)
            {
                end = next_end;
            }
            let text = &sentence[start..end];
            if let Some(id) = self.token_to_id(text) {
                tokens.push(Token::new(id, text.into(), (start, end)));
                continue;
            }
            if self.byte_fallback {
                let bytes: Option<Vec<Token>> = text
                    .bytes()
                    .map(|byte| {
                        let name = format!("<0x{byte:02X}>");
                        self.token_to_id(&name)
                            .map(|id| Token::new(id, name, (start, end)))
                    })
                    .collect();
                if let Some(bytes) = bytes {
                    tokens.extend(bytes);
                    continue;
                }
            }
            tokens.push(Token::new(id, text.into(), (start, end)));
        }
        Ok(tokens)
    }

    /// The model as `tokenizers` would build it, for what only it does.
    fn expanded(&self) -> tokenizers::Result<tokenizers::models::unigram::Unigram> {
        let vocab = (0..self.ends.len() as u32)
            .map(|id| {
                (
                    self.piece(id).expect("an id").to_owned(),
                    self.scores[id as usize],
                )
            })
            .collect();
        tokenizers::models::unigram::Unigram::from(
            vocab,
            self.unk_id.map(|id| id as usize),
            self.byte_fallback,
        )
    }
}

/// Every method answers as the model `tokenizers` would build from the same
/// section answers.
impl Model for Shared {
    type Trainer = TrainerWrapper;

    fn tokenize(&self, sequence: &str) -> tokenizers::Result<Vec<Token>> {
        match &*self.0 {
            Vocabulary::Unigram(model) => model.tokenize(sequence),
            Vocabulary::Other(model) => model.tokenize(sequence),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        match &*self.0 {
            Vocabulary::Unigram(model) => model.token_to_id(token),
            Vocabulary::Other(model) => model.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match &*self.0 {
            Vocabulary::Unigram(model) => model.piece(id).map(str::to_owned),
            Vocabulary::Other(model) => model.id_to_token(id),
        }
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        match &*self.0 {
            // In id order, so a piece listed twice keeps its later id.
            Vocabulary::Unigram(model) => (0..model.ends.len() as u32)
                .filter_map(|id| Some((model.piece(id)?.to_owned(), id)))
                .collect(),
            Vocabulary::Other(model) => model.get_vocab(),
        }
    }

    fn get_vocab_size(&self) -> usize {
        match &*self.0 {
            Vocabulary::Unigram(model) => model.ends.len(),
            Vocabulary::Other(model) => model.get_vocab_size(),
        }
    }

    fn save(
        &self,
        folder: &Path,
        prefix: Option<&str>,
    ) -> tokenizers::Result<Vec<std::path::PathBuf>> {
        match &*self.0 {
            Vocabulary::Unigram(model) => model.expanded()?.save(folder, prefix),
            Vocabulary::Other(model) => model.save(folder, prefix),
        }
    }

    fn get_trainer(&self) -> Self::Trainer {
        match &*self.0 {
            Vocabulary::Unigram(_) => UnigramTrainer::default().into(),
            Vocabulary::Other(model) => model.get_trainer(),
        }
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
///
/// Over any model, so that a test can build `tokenizers`' own model into a
/// tokenizer configured exactly as this one.
fn build<M: Model + DeserializeOwned>(
    tokenizer: &[u8],
    config: &[u8],
    special_tokens_map: &[u8],
    tokenizer_config: &[u8],
    max_length: usize,
) -> Result<
    TokenizerImpl<M, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>,
> {
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
        TokenizerImpl::from_bytes(tokenizer).map_err(|error| invalid("tokenizer.json", error))?;

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
pub(crate) mod tests {
    use super::*;

    /// A tokenizer built the way [`load`](super::load) builds one, over the
    /// vocabulary below, for other modules' tests.
    pub(crate) fn fixture() -> Tokenizer {
        load(&tokenizer_json(STRIP, SCORES))
    }

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

    /// A generator of the same pseudo-random numbers on every platform.
    struct Random(u64);

    impl Random {
        fn below(&mut self, bound: usize) -> usize {
            // xorshift64
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound as u64) as usize
        }
    }

    /// What a model does with a text, in a form two models can be compared
    /// by.
    type Segmentation = tokenizers::Result<Vec<(u32, String, (usize, usize))>>;

    fn segment(model: &impl Model, texts: &[String]) -> Vec<Segmentation> {
        texts
            .iter()
            .map(|text| {
                model.tokenize(text).map(|tokens| {
                    tokens
                        .into_iter()
                        .map(|token| (token.id, token.value, token.offsets))
                        .collect()
                })
            })
            .collect()
    }

    /// The same tokens, or the same error.
    fn same(left: &Segmentation, right: &Segmentation) -> bool {
        match (left, right) {
            (Ok(left), Ok(right)) => left == right,
            (Err(left), Err(right)) => left.to_string() == right.to_string(),
            _ => false,
        }
    }

    /// The compact model segments every text exactly as `tokenizers`' own
    /// model over the same vocabulary does, and answers every question about
    /// its vocabulary the same way.
    ///
    /// The vocabulary is built to reach what a segmentation can turn on:
    /// scores from a short list, so that two segmentations often score the
    /// same and only the order ties are broken in decides; pieces listed twice
    /// with different scores; multi-byte pieces -- CJK, Thai, emoji,
    /// combining marks -- beside characters no piece covers, which have to be
    /// fused into one unknown token; the unknown piece's own text in the
    /// input; NUL bytes, which the trie cannot be asked about; and byte
    /// fallback both covering a run of unknown characters and not.
    #[test]
    fn the_compact_model_segments_as_tokenizers_own() {
        const COVERED: [&str; 14] = [
            "a", "b", "c", "▁", "e", "\u{301}", "中", "文", "ท", "ไ", "😀", " ", "<", "k",
        ];
        const UNCOVERED: [&str; 7] = ["ย", "👍🏽", "\0", "\t", "\u{200d}", "Ω", "\u{10ffff}"];
        const SCORES: [f64; 4] = [-1.0, -2.0, -3.0, -4.5];

        let mut random = Random(0x9e37_79b9_7f4a_7c15);
        let mut vocab: Vec<(String, f64)> = ["<s>", "<pad>", "</s>", "<unk>"]
            .iter()
            .map(|piece| (piece.to_string(), 0.0))
            .collect();
        for &piece in &COVERED {
            vocab.push((piece.into(), SCORES[random.below(SCORES.len())]));
        }
        while vocab.len() < 400 {
            let piece: String = (0..1 + random.below(4))
                .map(|_| COVERED[random.below(COVERED.len())])
                .collect();
            vocab.push((piece, SCORES[random.below(SCORES.len())]));
        }
        // Listed again under a different score: the later is the one used.
        for again in [10, 50, 200] {
            vocab.push((vocab[again].0.clone(), -0.5));
        }
        // Byte fallback covering "Ω" (CE A9) and not "ย" (E0 B8 A2).
        for byte in [0xCE, 0xA9, 0xE0, 0xB8] {
            vocab.push((format!("<0x{byte:02X}>"), -6.0));
        }

        let symbols: Vec<&str> = COVERED
            .iter()
            .chain(&UNCOVERED)
            .copied()
            .chain(["<unk>", "<s>"])
            .collect();
        let mut texts: Vec<String> = vec![String::new(), "<unk>".into(), "\0".into()];
        texts.extend((0..4000).map(|_| {
            (0..random.below(40))
                .map(|_| symbols[random.below(symbols.len())])
                .collect()
        }));

        let mut questions: Vec<String> = vocab.iter().map(|(piece, _)| piece.clone()).collect();
        questions.extend(["", "\0", "a\0", "zz", "<0xFF>"].map(String::from));
        questions.extend(texts.iter().take(200).cloned());

        for (unk_id, byte_fallback) in [(Some(3), false), (Some(3), true), (None, false)] {
            let theirs =
                tokenizers::models::unigram::Unigram::from(vocab.clone(), unk_id, byte_fallback)
                    .expect("their model");
            let ours = Shared(Arc::new(Vocabulary::Unigram(
                Unigram::new(&vocab, unk_id, byte_fallback).expect("our model"),
            )));
            let context = format!("unk_id {unk_id:?}, byte_fallback {byte_fallback}");

            let (mine, reference) = (segment(&ours, &texts), segment(&theirs, &texts));
            for ((text, mine), reference) in texts.iter().zip(&mine).zip(&reference) {
                assert!(
                    same(mine, reference),
                    "{context}: {text:?} segments as {mine:?}, not {reference:?}"
                );
            }
            if unk_id.is_some() {
                assert!(
                    reference.iter().all(|segmented| segmented.is_ok()),
                    "{context}: the reference failed"
                );
                assert!(
                    reference
                        .iter()
                        .flatten()
                        .flatten()
                        .any(|token| token.0 == 3),
                    "{context}: no text reached an unknown piece"
                );
                if byte_fallback {
                    assert!(
                        reference
                            .iter()
                            .flatten()
                            .flatten()
                            .any(|token| token.1 == "<0xCE>"),
                        "{context}: no text fell back to bytes"
                    );
                }
            } else {
                assert!(
                    reference.iter().any(|segmented| segmented.is_err()),
                    "{context}: no text needed the missing unknown piece"
                );
            }

            for question in &questions {
                assert_eq!(
                    ours.token_to_id(question),
                    theirs.token_to_id(question),
                    "{question:?}"
                );
            }
            for id in 0..vocab.len() as u32 + 2 {
                assert_eq!(ours.id_to_token(id), theirs.id_to_token(id), "{id}");
            }
            assert_eq!(ours.get_vocab(), theirs.get_vocab());
            assert_eq!(ours.get_vocab_size(), theirs.get_vocab_size());
        }
    }

    /// A vocabulary with a piece the trie cannot hold is left to
    /// `tokenizers`' own model rather than refused or held inexactly.
    #[test]
    fn a_vocabulary_the_trie_cannot_hold_is_tokenizers_own() {
        let section = |pieces: &[&str]| -> serde_json::Value {
            let vocab: Vec<(&str, f64)> = pieces.iter().map(|piece| (*piece, -1.0)).collect();
            canonical(serde_json::json!({"type": "Unigram", "unk_id": 0, "vocab": vocab}))
        };
        assert!(matches!(
            Vocabulary::new(section(&["<unk>", "a"])),
            Ok(Vocabulary::Unigram(_))
        ));
        for pieces in [&["<unk>", "a\0"][..], &["<unk>", ""]] {
            assert!(matches!(
                Vocabulary::new(section(pieces)),
                Ok(Vocabulary::Other(ModelWrapper::Unigram(_)))
            ));
        }
    }

    /// Every tokenizer `crate::encoder` loads with a Unigram vocabulary
    /// encodes real and constructed text to exactly the encodings
    /// `tokenizers`' own model gives, configured the same way.
    ///
    /// Ignored by default: it needs the tokenizer files of BGE-M3 and of the
    /// `accurate` and `fast` rerankers, and for its real text, corpora. Run
    /// with
    ///
    /// ```sh
    /// PAMIN_TEST_MODELS=/path/to/models PAMIN_PARITY_CORPORA=/path/a.json:/path/b.json \
    ///   cargo test --release -p pamin-index --lib -- --ignored --nocapture \
    ///   the_compact_vocabulary_encodes_as_tokenizers_own
    /// ```
    ///
    /// `PAMIN_TEST_MODELS` is a model cache in the hub's layout, written to
    /// only to fetch what it lacks; without it the files are fetched into a
    /// temporary one. `PAMIN_PARITY_CORPORA` lists JSON files, every string
    /// in which is a text -- the XQuAD-R, MuSiQue and LOCOMO files the
    /// evaluation harnesses read, for instance.
    #[test]
    #[ignore = "needs the models' tokenizer files and corpora; see the comment"]
    fn the_compact_vocabulary_encodes_as_tokenizers_own() {
        let temporary = tempfile::tempdir().expect("a temporary directory");
        let cache = std::env::var_os("PAMIN_TEST_MODELS")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| temporary.path().to_path_buf());

        let mut texts: Vec<String> = edge_cases();
        let constructed = texts.len();
        let short: Vec<String> = texts
            .iter()
            .filter(|text| text.len() <= 4096)
            .cloned()
            .collect();
        let mut seen = std::collections::HashSet::new();
        for path in
            std::env::split_paths(&std::env::var_os("PAMIN_PARITY_CORPORA").unwrap_or_default())
                .filter(|path| !path.as_os_str().is_empty())
        {
            let document: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).expect("a corpus"))
                    .expect("a JSON corpus");
            let before = texts.len();
            strings(document, &mut |text| {
                if seen.insert(text.clone()) {
                    texts.push(text);
                }
            });
            println!(
                "{}: {} distinct texts",
                path.display(),
                texts.len() - before
            );
        }
        let bytes: usize = texts.iter().map(String::len).sum();
        println!(
            "{} texts ({constructed} constructed), {:.1} MB",
            texts.len(),
            bytes as f64 / 1e6
        );

        // As `crate::reranking` and `crate::embedding` load them, and at a
        // length that truncates almost everything.
        let models = [
            ("gpahal/bge-m3-onnx-int8", 512),
            ("onnx-community/bge-reranker-v2-m3-ONNX", 256),
            ("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 256),
        ];
        let mut vocabularies = Vec::new();
        for (name, product_length) in models {
            let repository = Repository::open(&cache, name).expect("the repository");
            let read = |file: &str| std::fs::read(repository.get(file).expect(file)).expect(file);
            let files = [
                read("tokenizer.json"),
                read("config.json"),
                read("special_tokens_map.json"),
                read("tokenizer_config.json"),
            ];
            for max_length in [product_length, 8] {
                let ours: Tokenizer =
                    build(&files[0], &files[1], &files[2], &files[3], max_length).expect("ours");
                let theirs =
                    build::<ModelWrapper>(&files[0], &files[1], &files[2], &files[3], max_length)
                        .expect("theirs");
                assert!(
                    matches!(*ours.get_model().0, Vocabulary::Unigram(_)),
                    "{name}: not held as a compact Unigram vocabulary"
                );
                vocabularies.push(Arc::clone(&ours.get_model().0));

                // Real text at the product's length; constructed text, which
                // is where the truncation boundaries are, at both -- but not
                // the very long texts at 8, where a pair's overflow is every
                // combination of its two sides' pieces, in the millions.
                let texts = if max_length == product_length {
                    &texts[..]
                } else {
                    &short[..]
                };
                let mut tokens = 0;
                for (index, text) in texts.iter().enumerate() {
                    let encoding = ours.encode(text.as_str(), true).expect("encode");
                    assert_eq!(
                        encoding,
                        theirs.encode(text.as_str(), true).expect("encode"),
                        "{name} at {max_length}: {text:?}"
                    );
                    tokens += encoding.len();
                    // As the reranker pairs a query with a passage.
                    let query = &texts[(index * 7919 + 1) % texts.len()];
                    let pair = (query.as_str(), text.as_str());
                    assert_eq!(
                        ours.encode(pair, true).expect("encode"),
                        theirs.encode(pair, true).expect("encode"),
                        "{name} at {max_length}: {pair:?}"
                    );
                }
                // And in padded batches, as `crate::encoder` runs them.
                for batch in texts.chunks(16).take(2000) {
                    let batch: Vec<&str> = batch.iter().map(String::as_str).collect();
                    assert_eq!(
                        ours.encode_batch(batch.clone(), true).expect("encode"),
                        theirs.encode_batch(batch.clone(), true).expect("encode"),
                        "{name} at {max_length}: {batch:?}"
                    );
                }
                println!(
                    "{name} at {max_length}: {} texts, {tokens} tokens identical",
                    texts.len()
                );

                // The model alone, over text no normalizer has touched.
                // A chunk at a time, so this holds a chunk's tokens and not
                // the corpus's.
                if max_length == product_length {
                    for texts in texts.chunks(256) {
                        let mine = segment(ours.get_model(), texts);
                        let reference = segment(theirs.get_model(), texts);
                        for ((text, mine), reference) in texts.iter().zip(&mine).zip(&reference) {
                            assert!(
                                same(mine, reference),
                                "{name}: the model segments raw {text:?} as {mine:?}, not {reference:?}"
                            );
                        }
                    }
                }
            }
        }
        assert!(
            vocabularies
                .iter()
                .all(|held| Arc::ptr_eq(held, &vocabularies[0])),
            "the three tokenizers do not share one vocabulary"
        );
    }

    /// Every string in `value`, at any depth.
    fn strings(value: serde_json::Value, found: &mut impl FnMut(String)) {
        match value {
            serde_json::Value::String(text) => found(text),
            serde_json::Value::Array(values) => {
                values.into_iter().for_each(|value| strings(value, found))
            }
            serde_json::Value::Object(fields) => fields
                .into_iter()
                .for_each(|(_, value)| strings(value, found)),
            _ => {}
        }
    }

    /// Texts at the edges of what a tokenizer handles.
    fn edge_cases() -> Vec<String> {
        let mut texts: Vec<String> = [
            "",
            " ",
            "    ",
            "\t\n\r\u{b}\u{c}",
            "a",
            "  leading and trailing  ",
            "a    b\t\t c\n\n\nd     ",
            "\u{3000}ideographic\u{3000}\u{3000}space\u{a0}no-break\u{2009}thin",
            "😀 👍🏽 👨‍👩‍👧‍👦 🇯🇵🇫🇷 ❤️ 🏳️‍🌈 🧑🏿‍🚀",
            "記憶は検索できる。中文测试，한국어 텍스트입니다。",
            "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ และใช้สระประกอบ",
            "e\u{301} a\u{308} n\u{303} A\u{30a} ñ Å é ǅ ẞ ß",
            "Z\u{351}\u{34b}\u{35b}a\u{36d}\u{316}l\u{489}g\u{335}o\u{31b}",
            "ﬁ ﬂ ① ＡＢＣ ㍻ ½ ™ ℌ ⅷ ｶﾀｶﾅ",
            "\u{0}a\u{0}\u{0}b\u{0}",
            "\u{feff}BOM \u{200b}zero\u{200c}width\u{200d}joiner \u{2028}\u{2029}",
            "\u{e000}\u{f8ff}\u{10ffff}\u{fffd}",
            "<s> </s> <pad> <unk> <mask> ▁ ▁▁▁ ▁a",
            "مرحبا بالعالم، هذا نص عربي.",
            "नमस्ते दुनिया, यह हिंदी पाठ है।",
            "Ελληνικά και Русский текст и Tiếng Việt có dấu.",
            "https://example.com/path?q=1&x=%20 user@example.com #tag @mention",
            "fn main() { println!(\"{}\", 1 + 2); } // ```code```",
            "3.14159 1,000,000 1e-9 0x7fff -42 +7 12:30 2026-10-12",
        ]
        .map(String::from)
        .to_vec();
        // Very long: a paragraph many times over, and words no space breaks.
        let paragraph = texts[8..13].join(" ");
        texts.push(paragraph.repeat(400));
        texts.push("a".repeat(10_000));
        texts.push("中".repeat(5_000));
        texts.push("ไทย".repeat(3_000));
        // Around each truncation boundary: "▁a" is one token, and a
        // tokenizer adds two or three of its own.
        for words in (0..12).chain(250..262).chain(504..518) {
            texts.push(vec!["a"; words].join(" "));
            texts.push(vec!["中"; words].join(""));
        }
        texts
    }
}
