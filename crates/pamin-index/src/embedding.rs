//! Local embeddings.
//!
//! Inference runs on this machine through ONNX Runtime. The default install
//! needs no API key and makes no network call at query time, which is what
//! keeps memory free to use and keeps evidence off third-party infrastructure.
//!
//! Two different operations are easy to conflate here. Quantizing model weights
//! buys a large CPU speedup for well under a percent of quality; storing output
//! vectors as int8 costs one and a half to three and a half percent and needs a
//! calibration set. The first is worth taking and the second is not: a
//! reranker reorders the fused head, but it cannot recover a candidate the
//! vector channel ranked out of the list.
//!
//! Stored vectors are float32. Weights are quantized where a quantized export
//! exists: pplx-embed runs Perplexity's 8-bit export, and the E5 pair runs full
//! precision because the model registry publishes no quantized variant for
//! that family.

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};
use crate::hub::Repository;

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
    /// cross-encoder reranker. There is one now -- `accurate` by default --
    /// but it reorders what the channels found and cannot add what this
    /// narrower space missed, so this stays the profile for machines that
    /// cannot afford the others.
    Speed,
    /// 768 dimensions, full-precision weights.
    ///
    /// No longer the middle rung it was named for. The model `accuracy` ran
    /// before this one -- BGE-M3's int8 export -- already beat it on retrieval
    /// by a factor of two at half its memory for nine milliseconds more a
    /// query, and the one it runs now beats that. Kept because a project
    /// indexed under it should not have to rebuild to keep working.
    Balanced,
    /// 1024 dimensions: Perplexity's pplx-embed-v1-0.6b (MIT), run from its
    /// published 8-bit export. The default.
    ///
    /// It replaced BGE-M3 (`docs/adr/0001-tech-selection.md`, "pplx-embed-v1-0.6b
    /// on the shipped path"), the only model surveyed since that beat it on
    /// every corpus here with the vector channel alone and kept a gain through
    /// fusion and the reranker. An index built by BGE-M3 is not refused: it is
    /// served without its vector channel while it is embedded again beside
    /// itself (see [`Profile::replaces`]).
    ///
    /// Mean-pooled over the attention mask by the graph itself; no instruction
    /// on either side, because the model card says it was trained without
    /// one.
    #[default]
    Accuracy,
}

impl Profile {
    /// Which E5 model this profile runs, for the two that run one.
    fn model(self) -> EmbeddingModel {
        match self {
            Self::Speed => EmbeddingModel::MultilingualE5Small,
            Self::Balanced => EmbeddingModel::MultilingualE5Base,
            Self::Accuracy => {
                unreachable!("the accuracy profile runs through this crate's encoder")
            }
        }
    }

    /// The prefixes this model expects on queries and on stored passages.
    ///
    /// E5 is an asymmetric retrieval family: it was trained with `query: ` and
    /// `passage: ` in front of the text, and omitting them degrades recall
    /// without failing. The embedding library does not add them, so we do.
    /// pplx uses none, which is why this belongs to the profile rather than to
    /// the embedder.
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
            // The export rather than the base model: quantized weights
            // produce vectors close to the full-precision ones and not equal
            // to them, and the recorded identity is what stops two encodings
            // sharing one index.
            Self::Accuracy => PPLX.identity,
        }
    }

    /// Whether `model` is an identity this profile used to record, so that an
    /// index built by it is embedded again rather than refused.
    ///
    /// Only models this profile ran in an earlier release: a change of model
    /// the user did not ask for is an upgrade to carry out, and one they did
    /// ask for -- another profile's index, opened under this one -- is still a
    /// mistake to report.
    pub fn replaces(self, model: &str) -> bool {
        match self {
            // BGE-M3's int8 export, until pplx replaced it.
            Self::Accuracy => model == "gpahal/bge-m3-onnx-int8",
            Self::Speed | Self::Balanced => false,
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
    /// Query vectors already computed.
    ///
    /// Shared with every project on this profile, because the vector depends on
    /// the model and the text and on nothing else.
    remembered: Queries,
}

/// The loaded model, which is not the same type for every profile.
///
/// pplx runs through this crate's own [`Encoder`], because the output it is
/// read from is one `fastembed` does not know to read; the E5 pair run
/// through `fastembed`'s general text type.
///
/// Both boxed. Each is over a kilobyte of session and tokenizer state, and an
/// unboxed enum is the size of its largest variant everywhere it appears.
enum Model {
    Text(Box<TextEmbedding>),
    Pooled(Box<Encoder>),
}

impl Embedder {
    /// Loads the model, downloading it on first use.
    ///
    /// The model is fetched lazily rather than bundled: it is larger than the
    /// binary by an order of magnitude, and a user who never searches should
    /// not pay for it.
    pub fn load(profile: Profile, cache_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;

        // See `crate::inference`: unset leaves fastembed on one thread per
        // core, which is what this did before the setting existed.
        let threads = crate::inference::threads();

        let model = match profile {
            Profile::Accuracy => Model::Pooled(Box::new(pplx(cache_dir)?)),
            _ => {
                let mut options = TextInitOptions::new(profile.model())
                    .with_cache_dir(cache_dir.to_path_buf())
                    .with_show_download_progress(false)
                    .with_execution_providers(vec![crate::inference::cpu()]);
                if let Some(threads) = threads {
                    options = options.with_intra_threads(threads);
                }
                Model::Text(Box::new(TextEmbedding::try_new(options).map_err(
                    |error| IndexError::Engine(format!("loading embedding model: {error}")),
                )?))
            }
        };

        Ok(Self {
            model,
            profile,
            remembered: Queries::default(),
        })
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
    ///
    /// Remembered, because the pass is one of the most expensive things a
    /// search does -- 16.9 ms of one when BGE-M3 was the model, and pplx-embed
    /// measured about 4.5 times BGE-M3's -- and a query is a pure function of
    /// the model and the text. What makes it worth remembering is
    /// the same thing that makes the reranker remember its scores: an agent
    /// retries, widens a limit, and asks again after writing something. Only
    /// queries. A passage is embedded once in its life, so a cache of those
    /// would hold the corpus and never be read.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(known) = self.remembered.get(text) {
            return Ok(known);
        }

        let vector = match self.profile.prefixes() {
            Some((query, _)) => self.embed_one(&format!("{query}{text}")),
            None => self.embed_one(text),
        }?;
        self.remembered.put(text, &vector);
        Ok(vector)
    }

    /// Embeds many passages in one forward pass.
    pub fn embed_passages(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = match self.profile.prefixes() {
            Some((_, passage)) => texts.iter().map(|t| format!("{passage}{t}")).collect(),
            None => texts.iter().map(|t| (*t).to_string()).collect(),
        };
        self.run(prefixed)
    }

    /// How many queries this remembers, per profile.
    ///
    /// A vector is [`Profile::dimensions`] floats -- four kilobytes at the
    /// widest -- so this is a megabyte at the top of the range. Per process,
    /// like the reranker's, so it is `pamin serve` that makes it worth
    /// anything.
    const REMEMBERED_QUERIES: usize = 256;

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut vectors = self.run(vec![text.to_string()])?;

        vectors
            .pop()
            .ok_or_else(|| IndexError::Engine("embedding produced no vector".into()))
    }

    /// One forward pass, whichever model this profile loaded.
    ///
    /// pplx gives each text a pass of its own. Not batched, though batching
    /// would not change a vector -- the graph quantizes activations a row at a
    /// time, and 32 MuSiQue paragraphs came back bit for bit the same in a
    /// batch of 32 as alone -- because there is nothing for a batch to spread.
    /// Profiled through this crate's session (ONNX Runtime 1.28, four cores,
    /// shared with another run), 83% of a pass is the int8 matrix products and
    /// 7% attention, and a pass costs the same per token at 28 tokens as at
    /// 292: no fixed cost per pass, only the padding a batch adds. Those 32
    /// paragraphs, 3,245 tokens, took 1.75 times as long in batches of up to
    /// 2,048 padded tokens, sorted by length, and 4 times as long as one batch.
    ///
    /// A pass of its own also keeps what `reindex` and the cascade store for a
    /// text identical, whichever other texts were in flight beside it.
    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let failed = |error| IndexError::Engine(format!("embedding text: {error}"));
        match &mut self.model {
            Model::Text(model) => model.embed(texts, None).map_err(failed),
            Model::Pooled(model) => texts.iter().map(|text| pooled(model, text)).collect(),
        }
    }
}

/// Where the `accuracy` profile's weights come from.
///
/// One constant, so that fetching the model from somewhere else -- another of
/// Perplexity's exports, or one this project publishes -- is one edit, and the
/// identity every stored vector is tagged with changes in the same edit.
struct Export {
    /// The model repository on the hub.
    repository: &'static str,
    /// The commit read from it. Pinned: an export replaced upstream under the
    /// same file name would change every vector while the identity stayed, and
    /// nothing downstream would look wrong.
    revision: &'static str,
    /// The graph, and the files it keeps its weights in, which have to be
    /// beside it before it loads.
    graph: &'static str,
    weights: &'static [&'static str],
    /// Whether the graph's `MatMulNBits` nodes are set to compute in int8
    /// before it loads (see `crate::nbits`), which an export that states its
    /// own level does not need.
    int8_compute: bool,
    /// What an index built with these weights records. See
    /// [`Profile::model_id`].
    identity: &'static str,
}

/// pplx-embed-v1-0.6b, as Perplexity publishes it: the 8-bit export, set to
/// compute in int8.
///
/// Of the three exports Perplexity publishes, the only one both close to the
/// full-precision model and fast enough to embed a query with. Over 400 texts
/// the full-precision one is the reference; this one agrees with it at cosine
/// 0.99925 on average, 0.9986 at the lowest, where the 4-bit export agrees at
/// 0.906 -- a different model, near enough. Computing in int8 costs none of
/// that and runs a query 5 times faster than the export as published (see
/// `crate::nbits`).
const PPLX: Export = Export {
    repository: "perplexity-ai/pplx-embed-v1-0.6b",
    revision: "2c4d510dd4a732063c31a0f70193e35067b51fd8",
    graph: "onnx/model_quantized.onnx",
    weights: &["onnx/model_quantized.onnx_data"],
    int8_compute: true,
    identity: "perplexity-ai/pplx-embed-v1-0.6b@2c4d510d/onnx/model_quantized.onnx+int8",
};

/// Which of the graph's outputs is the embedding.
///
/// The graph returns four: the last hidden state, its mean over the attention
/// mask, and that mean quantized two ways. The int8 one is the output the
/// model card documents, `round(127 * tanh(mean))`, and what every figure for
/// this model was measured on.
const PPLX_OUTPUT: &str = "pooler_output_int8";

/// The longest text pplx reads here, in tokens.
///
/// The model reads 32K. 512 is what every figure for it was measured at, and
/// what the model it replaced read -- so a passage past it embeds as its first
/// 512 tokens, as it did before, and changing it would change the vectors of
/// exactly those passages with nothing to say so.
const PPLX_MAX_TOKENS: usize = 512;

/// Loads pplx on the CPU, from its prepared copy (see `crate::prepared`).
fn pplx(cache_dir: &std::path::Path) -> Result<Encoder> {
    let repository = Repository::pinned(cache_dir, PPLX.repository, PPLX.revision)?;
    let copy = || {
        let weights = PPLX
            .weights
            .iter()
            .map(|weights| repository.get(weights))
            .collect::<Result<Vec<_>>>()?;
        let mut source = repository.get(PPLX.graph)?;
        if PPLX.int8_compute {
            source = crate::nbits::int8_compute(&source, &weights, cache_dir)?;
        }
        Ok(crate::prepared::prepared(&source, cache_dir))
    };
    Encoder::load(
        copy,
        &repository,
        PPLX_MAX_TOKENS,
        vec![crate::inference::cpu()],
    )
    .map_err(|error| IndexError::Engine(format!("loading embedding model: {error}")))
}

/// One text's pplx vector, in one forward pass of its own, at unit length.
///
/// The documented output is int8 and is compared by cosine, so it is scaled
/// to unit length here; the vector channel's distance is the cosine, and a
/// stored vector at unit length is what every other profile stores.
fn pooled(model: &mut Encoder, text: &str) -> Result<Vec<f32>> {
    let outputs = model.run(vec![text])?;
    let output = outputs
        .get(PPLX_OUTPUT)
        .ok_or_else(|| IndexError::Engine(format!("the pplx export has no {PPLX_OUTPUT}")))?;
    let (shape, values) = output
        .try_extract_tensor::<i8>()
        .map_err(|error| IndexError::Engine(format!("reading the pooled vector: {error}")))?;
    match **shape {
        [1, width] if width > 0 => Ok(unit(values.iter().map(|value| f32::from(*value)).collect())),
        _ => Err(IndexError::Engine(format!(
            "an output of shape {shape:?} for one text"
        ))),
    }
}

/// `vector` scaled to unit length, or as it is when it has none.
fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

/// Query vectors already computed, oldest first.
///
/// Keyed by the query itself rather than by a hash of it. The reranker's cache
/// keys scores by a hash and accepts that a collision misorders a result; a
/// collision here would hand back another query's vector, and a search for one
/// thing would quietly answer another. A query is a few dozen bytes against a
/// four-kilobyte vector, so keeping it costs almost nothing next to what it
/// guards.
#[derive(Default)]
struct Queries {
    known: std::collections::HashMap<String, Vec<f32>>,
    order: std::collections::VecDeque<String>,
}

impl Queries {
    fn get(&self, query: &str) -> Option<Vec<f32>> {
        self.known.get(query).cloned()
    }

    /// Remembers a vector, forgetting the oldest once full.
    ///
    /// Insertion order rather than use order, for the reason the reranker's is:
    /// keeping a true LRU means writing on every hit, and a query asked twice
    /// is asked twice close together.
    fn put(&mut self, query: &str, vector: &[f32]) {
        if self
            .known
            .insert(query.to_string(), vector.to_vec())
            .is_some()
        {
            return;
        }
        self.order.push_back(query.to_string());
        while self.order.len() > Embedder::REMEMBERED_QUERIES {
            if let Some(oldest) = self.order.pop_front() {
                self.known.remove(&oldest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    /// A remembered query is the same vector, and a different one is not.
    ///
    /// Keyed by the text, so the thing worth proving is that two queries never
    /// share an entry -- a cache that returned one query's vector for another
    /// would answer a search for one thing with the results for a different
    /// one, and nothing downstream could tell.
    #[test]
    fn a_remembered_query_is_its_own() {
        let mut queries = super::Queries::default();
        assert!(queries.get("how does deployment work").is_none());

        queries.put("how does deployment work", &[1.0, 2.0, 3.0]);
        queries.put("how does deployment fail", &[4.0, 5.0, 6.0]);

        assert_eq!(
            queries.get("how does deployment work"),
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(
            queries.get("how does deployment fail"),
            Some(vec![4.0, 5.0, 6.0])
        );
        assert!(queries.get("how does deployment").is_none());
    }

    /// The oldest is forgotten, and the cache stays the size it says.
    #[test]
    fn the_oldest_query_is_forgotten_once_it_is_full() {
        let mut queries = super::Queries::default();
        let cap = super::Embedder::REMEMBERED_QUERIES;
        for i in 0..cap + 10 {
            queries.put(&format!("query {i}"), &[i as f32]);
        }
        assert_eq!(queries.known.len(), cap);
        assert!(queries.get("query 0").is_none(), "the oldest survived");
        assert_eq!(
            queries.get(&format!("query {}", cap + 9)),
            Some(vec![(cap + 9) as f32]),
            "the newest was lost"
        );
    }

    /// Rewriting a query does not grow the queue behind it.
    #[test]
    fn remembering_a_query_twice_does_not_shorten_the_cache() {
        let mut queries = super::Queries::default();
        for _ in 0..super::Embedder::REMEMBERED_QUERIES * 2 {
            queries.put("the same question", &[1.0]);
        }
        assert_eq!(queries.order.len(), 1);
        assert_eq!(queries.get("the same question"), Some(vec![1.0]));
    }

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

    /// The one model `accuracy` has replaced is an upgrade and every other is
    /// a mistake, including another profile's current one.
    #[test]
    fn only_a_replaced_model_is_embedded_again() {
        assert!(Profile::Accuracy.replaces("gpahal/bge-m3-onnx-int8"));
        for profile in [Profile::Speed, Profile::Balanced, Profile::Accuracy] {
            assert!(!profile.replaces(profile.model_id()));
            for other in [Profile::Speed, Profile::Balanced, Profile::Accuracy] {
                assert!(!profile.replaces(other.model_id()));
            }
        }
        assert!(!Profile::Speed.replaces("gpahal/bge-m3-onnx-int8"));
    }

    /// The vector pplx stores is its int8 output at unit length, pointing the
    /// same way.
    #[test]
    fn a_pooled_vector_is_scaled_to_unit_length() {
        assert_eq!(unit(vec![3.0, -4.0, 0.0]), vec![0.6, -0.8, 0.0]);
        assert_eq!(unit(vec![0.0; 3]), vec![0.0; 3], "nothing to scale");
    }

    #[test]
    fn the_default_profile_is_accuracy() {
        assert_eq!(Profile::default(), Profile::Accuracy);
    }
}
