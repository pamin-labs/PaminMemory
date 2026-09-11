//! Local embeddings.
//!
//! Inference runs on this machine through ONNX Runtime. The default install
//! needs no API key and makes no network call at query time, which is what
//! keeps memory free to use and keeps evidence off third-party infrastructure.
//!
//! Two different operations are easy to conflate here. Quantizing model weights
//! buys a large CPU speedup for well under a percent of quality; storing output
//! vectors as int8 costs one and a half to three and a half percent and needs a
//! calibration set. The first is worth taking and the second is not,
//! particularly since the default reranker has no cross-encoder to recover the
//! loss.
//!
//! Stored vectors are float32. Weights are quantized where a quantized export
//! exists: BGE-M3 runs int8 weights, and the E5 pair runs full precision
//! because the model registry publishes no quantized variant for that family.

use fastembed::{
    Bgem3Embedding, Bgem3InitOptions, Bgem3Model, EmbeddingModel, TextEmbedding, TextInitOptions,
};
use serde::{Deserialize, Serialize};

use crate::error::{IndexError, Result};

/// Which embedding model to run.
///
/// Profiles rather than a single constant, because the right trade differs
/// between bulk ingestion on a laptop and answering one query well. All three
/// are permissively licensed. EmbeddingGemma scores well and would otherwise be
/// a candidate, but it carries usage restrictions that must be passed on to
/// downstream users, which is not a burden to attach to an open-source default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// 384 dimensions. Bulk ingestion and low-spec machines.
    ///
    /// 384 dimensions is generally held to be enough only alongside a
    /// cross-encoder reranker, and ours is deterministic and has none, so this
    /// pairs the weaker model with the weaker reranker. It is here for
    /// machines that cannot afford the others.
    Speed,
    /// 768 dimensions, full-precision weights.
    ///
    /// No longer the middle rung it was named for. The quantized BGE-M3 export
    /// beats it on retrieval by a factor of two, is half its size in memory,
    /// and costs nine milliseconds more per query -- so the only reason left
    /// to choose this is those nine milliseconds. Kept because a project
    /// indexed under it should not have to rebuild to keep working.
    Balanced,
    /// 1024 dimensions, int8 weights, and by a distance the best cross-lingual
    /// recall of the three. The default.
    ///
    /// Run through the joint BGE-M3 export, which produces dense, sparse and
    /// ColBERT representations in one forward pass. Only the dense one is
    /// kept. The sparse arm duplicates the two lexical channels already in
    /// place and is worth 0.2 points of cross-lingual nDCG by its own authors'
    /// ablation; the ColBERT arm is one 1024-wide vector per token, which for
    /// a project of seven million documents is terabytes.
    ///
    /// The default because it is not the trade its name implies. Against
    /// `balanced` it doubles cross-lingual nDCG@10, matches it monolingually,
    /// occupies 560 MB against 1.1 GB, and costs 35 ms per query against 26.
    /// The int8 export is what makes all of that true at once; the
    /// full-precision one is 2.2 GB and was the reason this profile used to be
    /// described as an order of magnitude more expensive.
    #[default]
    Accuracy,
}

impl Profile {
    /// Which E5 model this profile runs, for the two that run one.
    fn model(self) -> EmbeddingModel {
        match self {
            Self::Speed => EmbeddingModel::MultilingualE5Small,
            Self::Balanced => EmbeddingModel::MultilingualE5Base,
            Self::Accuracy => unreachable!("the accuracy profile runs the joint BGE-M3 export"),
        }
    }

    /// The prefixes this model expects on queries and on stored passages.
    ///
    /// E5 is an asymmetric retrieval family: it was trained with `query: ` and
    /// `passage: ` in front of the text, and omitting them degrades recall
    /// without failing. The embedding library does not add them, so we do.
    /// BGE-M3 uses none, which is why this belongs to the profile rather than
    /// to the embedder.
    fn prefixes(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Speed | Self::Balanced => Some(("query: ", "passage: ")),
            Self::Accuracy => None,
        }
    }

    /// The vector width this profile produces.
    ///
    /// The index is built for one width. Mixing embedding spaces yields
    /// distances that mean nothing, so changing profile reindexes.
    pub fn dimensions(self) -> u32 {
        match self {
            Self::Speed => 384,
            Self::Balanced => 768,
            Self::Accuracy => 1024,
        }
    }

    /// The identifier recorded alongside every stored vector, so a later read
    /// can tell which space a vector belongs to.
    pub fn model_id(self) -> &'static str {
        match self {
            // The suffix is an encoding revision, not part of the model name.
            // Vectors written before the E5 prefixes existed are in a different
            // space, and nothing about the resulting rankings would look wrong,
            // so the recorded identity has to change with the encoding.
            Self::Speed => "intfloat/multilingual-e5-small+p1",
            Self::Balanced => "intfloat/multilingual-e5-base+p1",
            // The quantized export rather than the base model: int8 weights
            // produce vectors close to the full-precision ones and not equal
            // to them, and the recorded identity is what stops two encodings
            // sharing one index.
            Self::Accuracy => "gpahal/bge-m3-onnx-int8",
        }
    }

    /// Parses a profile name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "speed" => Some(Self::Speed),
            "balanced" => Some(Self::Balanced),
            "accuracy" => Some(Self::Accuracy),
            _ => None,
        }
    }
}

/// Turns text into vectors.
pub struct Embedder {
    model: Model,
    profile: Profile,
}

/// The loaded model, which is not the same type for every profile.
///
/// BGE-M3 ships as a joint export producing three representations at once, and
/// the library loads it through its own type rather than the general text one.
/// That is also the only path to its int8 weights, which is most of why the
/// profile is usable at all.
///
/// Both boxed. Each is over a kilobyte of session and tokenizer state, and an
/// unboxed enum is the size of its largest variant everywhere it appears.
enum Model {
    Text(Box<TextEmbedding>),
    Joint(Box<Bgem3Embedding>),
}

impl Embedder {
    /// Loads the model, downloading it on first use.
    ///
    /// The model is fetched lazily rather than bundled: it is larger than the
    /// binary by an order of magnitude, and a user who never searches should
    /// not pay for it.
    pub fn load(profile: Profile, cache_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;

        let model = match profile {
            Profile::Accuracy => {
                let options = Bgem3InitOptions::new(Bgem3Model::BGEM3Q)
                    .with_cache_dir(cache_dir.to_path_buf())
                    .with_show_download_progress(false);
                Model::Joint(Box::new(Bgem3Embedding::try_new(options).map_err(
                    |error| IndexError::Engine(format!("loading embedding model: {error}")),
                )?))
            }
            _ => {
                let options = TextInitOptions::new(profile.model())
                    .with_cache_dir(cache_dir.to_path_buf())
                    .with_show_download_progress(false);
                Model::Text(Box::new(TextEmbedding::try_new(options).map_err(
                    |error| IndexError::Engine(format!("loading embedding model: {error}")),
                )?))
            }
        };

        Ok(Self { model, profile })
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Embeds one passage for storage.
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>> {
        match self.profile.prefixes() {
            Some((_, passage)) => self.embed_one(&format!("{passage}{text}")),
            None => self.embed_one(text),
        }
    }

    /// Embeds one query.
    ///
    /// Queries and passages take different prefixes, so this is not the same
    /// call as `embed_passage` even though both end in one forward pass.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        match self.profile.prefixes() {
            Some((query, _)) => self.embed_one(&format!("{query}{text}")),
            None => self.embed_one(text),
        }
    }

    /// Embeds many passages in one forward pass.
    pub fn embed_passages(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = match self.profile.prefixes() {
            Some((_, passage)) => texts.iter().map(|t| format!("{passage}{t}")).collect(),
            None => texts.iter().map(|t| (*t).to_string()).collect(),
        };
        self.run(prefixed)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.run(vec![text.to_string()])?;

        vectors
            .pop()
            .ok_or_else(|| IndexError::Engine("embedding produced no vector".into()))
    }

    /// One forward pass, whichever model this profile loaded.
    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match &mut self.model {
            Model::Text(model) => model.embed(texts, None),
            // The sparse and ColBERT representations come back from the same
            // pass and are dropped here. They are not free -- the pass
            // computes them -- but neither is wanted, and no cheaper export of
            // this model's int8 weights exists.
            Model::Joint(model) => model.embed(texts, None).map(|output| output.dense),
        }
        .map_err(|error| IndexError::Engine(format!("embedding text: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_names_round_trip() {
        for (name, profile) in [
            ("speed", Profile::Speed),
            ("balanced", Profile::Balanced),
            ("accuracy", Profile::Accuracy),
        ] {
            assert_eq!(Profile::parse(name), Some(profile));
        }
        assert_eq!(Profile::parse("enormous"), None);
    }

    #[test]
    fn each_profile_declares_a_distinct_width_and_identity() {
        // The width is what the index is built for and the identity is what a
        // stored vector is tagged with, so two profiles sharing either would
        // let incompatible vectors sit in one space undetected.
        let profiles = [Profile::Speed, Profile::Balanced, Profile::Accuracy];
        for (index, left) in profiles.iter().enumerate() {
            for right in &profiles[index + 1..] {
                assert_ne!(left.dimensions(), right.dimensions());
                assert_ne!(left.model_id(), right.model_id());
            }
        }
    }

    #[test]
    fn the_default_profile_is_accuracy() {
        assert_eq!(Profile::default(), Profile::Accuracy);
    }
}
