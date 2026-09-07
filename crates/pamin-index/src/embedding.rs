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
//! Neither is in force today. Stored vectors are float32 and will stay that
//! way. Weight quantization is unavailable rather than declined: the model
//! registry publishes quantized variants for several families but none for
//! multilingual E5, so both default profiles run full-precision weights.

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

use crate::error::{IndexError, Result};

/// Which embedding model to run.
///
/// Profiles rather than a single constant, because the right trade differs
/// between bulk ingestion on a laptop and answering one query well. All three
/// are permissively licensed. EmbeddingGemma scores well and would otherwise be
/// a candidate, but it carries usage restrictions that must be passed on to
/// downstream users, which is not a burden to attach to an open-source default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// 384 dimensions. Bulk ingestion and low-spec machines.
    Speed,
    /// 768 dimensions. The default.
    ///
    /// 384 dimensions is generally held to be enough only alongside a
    /// cross-encoder reranker, and ours is deterministic and has none, so
    /// defaulting to the smaller model would pair the weaker model with the
    /// weaker reranker.
    #[default]
    Balanced,
    /// 1024 dimensions, dense and sparse in one pass, longer context.
    ///
    /// Not the default: its main increment is a sparse arm that overlaps the
    /// two lexical channels already in place, and it costs an order of
    /// magnitude more per query.
    Accuracy,
}

impl Profile {
    fn model(self) -> EmbeddingModel {
        match self {
            Self::Speed => EmbeddingModel::MultilingualE5Small,
            Self::Balanced => EmbeddingModel::MultilingualE5Base,
            Self::Accuracy => EmbeddingModel::BGEM3,
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
            Self::Accuracy => "BAAI/bge-m3",
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
    model: TextEmbedding,
    profile: Profile,
}

impl Embedder {
    /// Loads the model, downloading it on first use.
    ///
    /// The model is fetched lazily rather than bundled: it is larger than the
    /// binary by an order of magnitude, and a user who never searches should
    /// not pay for it.
    pub fn load(profile: Profile, cache_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;

        let options = TextInitOptions::new(profile.model())
            .with_cache_dir(cache_dir.to_path_buf())
            .with_show_download_progress(false);

        let model = TextEmbedding::try_new(options)
            .map_err(|error| IndexError::Engine(format!("loading embedding model: {error}")))?;

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

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self
            .model
            .embed(vec![text], None)
            .map_err(|error| IndexError::Engine(format!("embedding text: {error}")))?;

        vectors
            .pop()
            .ok_or_else(|| IndexError::Engine("embedding produced no vector".into()))
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
    fn the_default_profile_is_balanced() {
        assert_eq!(Profile::default(), Profile::Balanced);
    }
}
