//! A second pass over the shortlist.
//!
//! Fusion orders candidates by the ranks four channels gave them, and it never
//! reads a candidate and the query together. A cross-encoder does, which is why
//! it can fix an ordering fusion got wrong and why it costs what it costs: the
//! query and the document go through one model jointly, so nothing about a
//! document can be computed before the query arrives. That is a property of the
//! architecture rather than of this implementation, and it is why the
//! literature's answers to cross-encoder latency all change the architecture --
//! precomputing part of the document's representation at index time, or moving
//! to late interaction. Both trade the cost for storage proportional to
//! documents times tokens times width, which for a store meant to hold millions
//! of memories is tens of gigabytes. So the cost is paid at query time or not
//! at all.
//!
//! ## What it is worth, measured
//!
//! On 13,014 sentences in eleven languages, reranking the fused shortlist:
//!
//! | | cross-lingual | same-language | per query |
//! |---|---|---|---|
//! | off | — | — | 0 ms |
//! | `fast` | **+0.0595** | 0.0000 | 151 ms |
//! | `accurate` | **+0.0852** | 0.0000 | 1795 ms |
//!
//! Two things in that table need saying.
//!
//! **The same-language column is zero by construction, not by luck.** Every
//! reranker tried -- three, across two orders of magnitude in size -- improves
//! cross-lingual ranking and damages same-language ranking, because the fused
//! list is already good at same-language and a cross-encoder reorders it worse.
//! So only the candidates written in a language other than the query's are
//! reranked, and they are placed back into the positions they already held.
//! A same-language candidate cannot move, and that group's score cannot change.
//! Unrestricted, the same two models score +0.1698 / -0.2109 and +0.1678 /
//! -0.0806: more cross-lingual, at a cost that is not this project's to take.
//!
//! **Smaller is not worse here.** The `fast` model is a twelve-layer distilled
//! MiniLM with 21M encoder parameters; `accurate` is XLM-RoBERTa-large with
//! 303M, fourteen times larger and twelve times slower, and it buys 0.026.
//! That is the finding of [Shallow Cross-Encoders for Low-Latency
//! Retrieval](https://arxiv.org/abs/2403.20222) arrived at independently: under
//! a latency budget a shallow model beats a full-scale one, because the budget
//! buys more candidates.
//!
//! ## What the numbers do not say
//!
//! They were measured on four cores. Published figures for a MiniLM
//! cross-encoder on CPU are 0.5 to 3 ms per pair; this measures 9.75, and the
//! gap is two layers' worth of depth and a quarter of the cores. On an ordinary
//! server the `fast` tier is tens of milliseconds rather than 151.
//!
//! They were also measured on parallel text, where every candidate is a
//! translation of every other. Real memories are not, and a corpus where the
//! same-language candidates are not near-duplicates of the foreign ones may
//! divide the gain differently.

use std::path::{Path, PathBuf};

use fastembed::{
    OnnxSource, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};
use serde::{Deserialize, Serialize};

use crate::error::{IndexError, Result};

/// How much to spend reordering the shortlist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rerank {
    /// Return what fusion ordered.
    ///
    /// The right setting for a workspace whose memories are all in one
    /// language: the measured gain there is exactly zero, because only
    /// candidates in another language are reordered and there are none.
    Off,
    /// A twelve-layer distilled multilingual MiniLM, 113 MB. The default.
    ///
    /// Chosen over the larger model on the shape of the trade rather than on
    /// the score: it reaches seven tenths of the gain for a twelfth of the
    /// latency and a fifth of the download.
    #[default]
    Fast,
    /// XLM-RoBERTa-large, 570 MB, twelve times slower for a quarter more gain.
    ///
    /// For a workspace where retrieval quality is worth a second of latency --
    /// an agent doing research rather than one answering interactively.
    Accurate,
}

/// How many of the fused results a tier looks at.
///
/// Twenty for both, which is where the gain flattens: measured on the `fast`
/// model, ten is worth +0.0147, fifteen +0.0417, twenty +0.0572, thirty
/// +0.0667 and fifty +0.0629 -- past thirty it goes backwards, and the last
/// ten candidates cost half the latency for a tenth of the gain.
const DEPTH: usize = 20;

/// How many candidates go through the model at once.
///
/// Eight, measured. A batch is padded to its longest member, so a large batch
/// pays for its longest candidate on every member: sixteen takes 191 ms and
/// eight takes 151 for the same candidates, and thirty-two and sixty-four are
/// slower again. Sorting by length before batching is the other half of the
/// same saving and is done below.
const BATCH: usize = 8;

/// The longest candidate the model reads.
///
/// Generous rather than binding: halving it to 128 changed latency by a fifth
/// and nothing else, which says the candidates are already shorter than this.
/// It is here so that one long memory cannot make one query slow.
const MAX_TOKENS: usize = 256;

impl Rerank {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "fast" => Some(Self::Fast),
            "accurate" => Some(Self::Accurate),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Fast => "fast",
            Self::Accurate => "accurate",
        }
    }

    /// How deep into the fused list this tier reaches.
    pub fn depth(self) -> usize {
        match self {
            Self::Off => 0,
            Self::Fast | Self::Accurate => DEPTH,
        }
    }

    fn repository(self) -> &'static str {
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            Self::Fast => "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            Self::Accurate => "onnx-community/bge-reranker-v2-m3-ONNX",
        }
    }

    /// Which export of the model to fetch.
    ///
    /// The fast model publishes one quantized export per instruction set, and
    /// they are not interchangeable in speed: on a machine with AVX-512 VNNI
    /// the VNNI build is 1.5x the AVX2 one for the same scores. Picking at
    /// runtime rather than at build time, because a binary is built once and
    /// run on whatever is there.
    fn onnx(self) -> &'static str {
        match self {
            Self::Off => unreachable!("nothing is loaded for the off tier"),
            Self::Accurate => "onnx/model_int8.onnx",
            Self::Fast => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("avx512vnni") {
                        return "onnx/model_qint8_avx512_vnni.onnx";
                    }
                    "onnx/model_quint8_avx2.onnx"
                }
                #[cfg(target_arch = "aarch64")]
                {
                    "onnx/model_qint8_arm64.onnx"
                }
                // No quantized build for this architecture; full precision runs
                // everywhere and is what the quantized ones were made from.
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    "onnx/model.onnx"
                }
            }
        }
    }
}

/// A loaded cross-encoder.
pub struct Reranker {
    model: TextRerank,
    tier: Rerank,
}

impl Reranker {
    /// Loads the tier's model, downloading it on first use.
    ///
    /// Through the same cache as the embedding model, so a workspace holds one
    /// directory of weights rather than two.
    pub fn load(tier: Rerank, cache_dir: &Path) -> Result<Self> {
        debug_assert!(tier != Rerank::Off, "the off tier loads nothing");
        std::fs::create_dir_all(cache_dir)?;

        let repository = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .with_progress(false)
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .model(tier.repository().to_string());

        let fetch = |name: &str| -> Result<PathBuf> {
            repository.get(name).map_err(|error| {
                IndexError::Engine(format!(
                    "fetching {name} for the {} reranker: {error}",
                    tier.name()
                ))
            })
        };
        let read = |name: &str| -> Result<Vec<u8>> { Ok(std::fs::read(fetch(name)?)?) };

        let model = TextRerank::try_new_from_user_defined(
            UserDefinedRerankingModel::new(
                // By path rather than by bytes: the session maps the file, and
                // handing it a copy of half a gigabyte first serves no purpose.
                OnnxSource::File(fetch(tier.onnx())?),
                TokenizerFiles {
                    tokenizer_file: read("tokenizer.json")?,
                    config_file: read("config.json")?,
                    special_tokens_map_file: read("special_tokens_map.json")?,
                    tokenizer_config_file: read("tokenizer_config.json")?,
                },
            ),
            RerankInitOptionsUserDefined::new().with_max_length(MAX_TOKENS),
        )
        .map_err(|error| IndexError::Engine(format!("loading the reranker: {error}")))?;

        Ok(Self { model, tier })
    }

    pub fn tier(&self) -> Rerank {
        self.tier
    }

    /// Orders `documents` best first, returning their original positions.
    ///
    /// Batched by length rather than in the order given, because a batch is
    /// padded to its longest member and the caller's order is by relevance,
    /// which says nothing about length. The permutation is undone before the
    /// result is returned, so a caller sees positions into what it passed.
    pub fn rank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<usize>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let mut by_length: Vec<usize> = (0..documents.len()).collect();
        by_length.sort_by_key(|position| documents[*position].len());
        let sorted: Vec<&str> = by_length
            .iter()
            .map(|position| documents[*position])
            .collect();

        let scored = self
            .model
            .rerank(query, &sorted, false, Some(BATCH))
            .map_err(|error| IndexError::Engine(format!("reranking: {error}")))?;

        Ok(scored
            .into_iter()
            .map(|result| by_length[result.index])
            .collect())
    }
}
