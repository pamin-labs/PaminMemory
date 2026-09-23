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
//! exists: BGE-M3 runs int8 weights, and the E5 pair runs full precision
//! because the model registry publishes no quantized variant for that family.

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use serde::{Deserialize, Serialize};

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};
use crate::hub::Repository;

/// Which embedding model to run.
///
/// Profiles rather than a single constant, because the right trade differs
/// between bulk ingestion on a laptop and answering one query well. All of
/// them are permissively licensed. EmbeddingGemma scores well and would
/// otherwise be a candidate, but it carries usage restrictions that must be
/// passed on to downstream users, which is not a burden to attach to an
/// open-source default.
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
    /// 1024 dimensions: Perplexity's `pplx-embed-v1-0.6b` (MIT), on trial.
    /// Experimental, and chosen only by name.
    ///
    /// The one model surveyed since BGE-M3 that beat it on every corpus here
    /// with the vector channel alone (see the ADR's second embedder survey);
    /// this profile exists so the whole search path can be measured on it.
    ///
    /// Its weights are not downloaded. Perplexity's own 8-bit export was
    /// measured 8-10x slower on this project's four-core CPU (see the ADR),
    /// so this runs a quantization of their fp32 export made for the trial --
    /// dynamic int8 on every matrix product except the MLPs' `down_proj`,
    /// which stay fp32 -- and that export is published nowhere. It is read from the directory
    /// `PAMIN_PPLX_DIR` names, which holds `model.onnx` beside the model's
    /// four tokenizer files, and loading this profile without it fails.
    ///
    /// The graph mean-pools over the attention mask and returns the model's
    /// documented output, `round(127 * tanh(mean))` as int8; that is compared
    /// by cosine, so it is scaled to unit length here. No instruction on
    /// either side: the model card says it was trained without one.
    Pplx,
}

impl Profile {
    /// Which E5 model this profile runs, for the two that run one.
    fn model(self) -> EmbeddingModel {
        match self {
            Self::Speed => EmbeddingModel::MultilingualE5Small,
            Self::Balanced => EmbeddingModel::MultilingualE5Base,
            Self::Accuracy | Self::Pplx => {
                unreachable!("{self:?} runs through this crate's own encoder")
            }
        }
    }

    /// The prefixes this model expects on queries and on stored passages.
    ///
    /// E5 is an asymmetric retrieval family: it was trained with `query: ` and
    /// `passage: ` in front of the text, and omitting them degrades recall
    /// without failing. The embedding library does not add them, so we do.
    /// BGE-M3 and pplx use none, which is why this belongs to the profile
    /// rather than to the embedder.
    fn prefixes(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Speed | Self::Balanced => Some(("query: ", "passage: ")),
            Self::Accuracy | Self::Pplx => None,
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
            Self::Accuracy | Self::Pplx => 1024,
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
            // Not Perplexity's export but this trial's quantization of it, and
            // named for the quantization for the reason above. It shares its
            // width with BGE-M3's, so this is the only thing that keeps the
            // two spaces out of one index.
            Self::Pplx => "perplexity-ai/pplx-embed-v1-0.6b+int8-down-fp32",
        }
    }

    /// Parses a profile name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "speed" => Some(Self::Speed),
            "balanced" => Some(Self::Balanced),
            "accuracy" => Some(Self::Accuracy),
            "pplx" => Some(Self::Pplx),
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
/// BGE-M3 ships as a joint export producing three representations at once,
/// its int8 weights -- most of why the profile is usable at all -- only in
/// that export, and its vocabulary shared with the `accurate` reranker's. So
/// it runs through this crate's own [`Encoder`], which can share that
/// vocabulary, where the E5 pair run through `fastembed`'s general text type.
/// pplx runs through the same encoder, because its export is a directory on
/// this machine rather than anything `fastembed` knows how to load, and reads
/// a different output of its pass.
///
/// All boxed. Each is over a kilobyte of session and tokenizer state, and an
/// unboxed enum is the size of its largest variant everywhere it appears.
enum Model {
    Text(Box<TextEmbedding>),
    Joint(Box<Encoder>),
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
            Profile::Accuracy => Model::Joint(Box::new(joint(cache_dir)?)),
            Profile::Pplx => Model::Pooled(Box::new(pplx(
                std::env::var_os(PPLX_DIR).map(std::path::PathBuf::from),
                cache_dir,
            )?)),
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
    /// Remembered, because the pass is the most expensive thing a search does
    /// -- 16.9 ms of a search at the shipping profile -- and a query is a pure
    /// function of the model and the text. What makes it worth remembering is
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
    /// The joint export runs one text at a time, and that is a correctness
    /// choice rather than an oversight. Measured on this export: a text's
    /// vector changes when anything else shares its batch. Against the same
    /// text embedded alone, a batch of two returns cosine 0.9816 with a
    /// shorter neighbour and 0.9859 with a longer one, and the two neighbours
    /// disagree with each other at 0.9805 -- on 1024 dimensions that is a
    /// different vector, not a rounding difference. A batch of one is
    /// identical to a single call, so it is the presence of a neighbour that
    /// does it, not the batching API.
    ///
    /// It is not the tokenization: the tokenizer pads to the batch's
    /// longest member, so a text that *is* the longest gets byte-identical
    /// ids and mask either way, and the mask is passed to the session. Only
    /// the batch dimension differs, so what changes the answer is the export
    /// or the runtime's INT8 kernels. `speed` and `balanced` are unaffected --
    /// both return byte-identical vectors batched or alone -- which is why
    /// this is scoped to the joint model.
    ///
    /// What it costs is the batching win on this profile: thirty-two texts
    /// together take 190 ms against 409 ms one at a time, so `reindex` is
    /// roughly twice the wall clock here. What it buys is that a document's
    /// vector does not depend on which other documents happened to be in
    /// flight beside it -- so `reindex` and the cascade agree, and the same
    /// corpus written twice indexes to the same thing.
    ///
    /// pplx runs one text at a time for the same reason, and there it is a
    /// property of the scheme rather than of an export: its quantization is
    /// dynamic, and `DynamicQuantizeLinear` takes one scale for the whole
    /// input tensor, so every text in a batch would share the activation
    /// scale of whichever of them ranges widest.
    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let failed = |error| IndexError::Engine(format!("embedding text: {error}"));
        match &mut self.model {
            Model::Text(model) => model.embed(texts, None).map_err(failed),
            Model::Joint(model) => texts.iter().map(|text| dense(model, text)).collect(),
            Model::Pooled(model) => texts.iter().map(|text| pooled(model, text)).collect(),
        }
    }
}

/// Where BGE-M3's int8 export is published, and which of its files it is.
const JOINT_REPOSITORY: &str = "gpahal/bge-m3-onnx-int8";
const JOINT_FILE: &str = "model_quantized.onnx";

/// The longest text BGE-M3 reads, in tokens.
///
/// What `fastembed` truncated it at when it loaded this model, and so what
/// every vector stored under this profile was made with. A passage past it
/// embeds as its first 512 tokens; changing it would change the vectors of
/// exactly those passages and nothing would say so, so it is kept.
const JOINT_MAX_TOKENS: usize = 512;

/// Loads the joint BGE-M3 export on the CPU, from its prepared copy.
///
/// Its int8 export is 570 MB, and loaded from the file the hub serves it is
/// copied onto the heap whole and its matrix weights packed into a second copy
/// -- see `crate::prepared`, which writes a copy the runtime maps instead, and
/// falls back to the file itself when it cannot.
fn joint(cache_dir: &std::path::Path) -> Result<Encoder> {
    own(
        &Repository::open(cache_dir, JOINT_REPOSITORY)?,
        JOINT_FILE,
        JOINT_MAX_TOKENS,
        cache_dir,
    )
}

/// The variable naming the directory the `pplx` export is read from.
///
/// A sweep-only setting, and undocumented for the reason the profile is: the
/// export it points at exists only on the machine that made it.
const PPLX_DIR: &str = "PAMIN_PPLX_DIR";
const PPLX_FILE: &str = "model.onnx";

/// Which of the pplx graph's outputs is the embedding.
///
/// The graph returns four: the last hidden state, its mean over the attention
/// mask, and that mean quantized two ways. The int8 one is what the model card
/// documents and what the survey measured.
const PPLX_OUTPUT: &str = "pooler_output_int8";

/// The longest text pplx reads here, in tokens: BGE-M3's, so that a trial of
/// one against the other embeds the same text of every long passage. The
/// model itself reads 32K.
const PPLX_MAX_TOKENS: usize = 512;

/// Loads the `pplx` export from `dir`, the value of [`PPLX_DIR`].
fn pplx(dir: Option<std::path::PathBuf>, cache_dir: &std::path::Path) -> Result<Encoder> {
    let dir = dir.ok_or_else(|| {
        IndexError::Engine(format!(
            "the pplx profile reads its model from the directory {PPLX_DIR} names, and it is not set"
        ))
    })?;
    own(
        &Repository::directory(&dir),
        PPLX_FILE,
        PPLX_MAX_TOKENS,
        cache_dir,
    )
}

/// A model that runs through this crate's own [`Encoder`], on the CPU, from
/// its prepared copy.
fn own(
    repository: &Repository,
    file: &str,
    max_tokens: usize,
    cache_dir: &std::path::Path,
) -> Result<Encoder> {
    let source = repository.get(file)?;
    let copy = crate::prepared::prepared(&source, cache_dir);
    Encoder::load(&copy, repository, max_tokens, vec![crate::inference::cpu()])
        .map_err(|error| IndexError::Engine(format!("loading embedding model: {error}")))
}

/// One text's dense BGE-M3 vector, in one forward pass of its own.
///
/// The export's first output, as `fastembed` read it: the graph pools and
/// normalizes the vector itself, so it is used as it comes. The sparse and
/// ColBERT representations come back from the same pass and are dropped. They
/// are not free -- the pass computes them -- but neither is wanted, and no
/// cheaper export of this model's int8 weights exists.
fn dense(model: &mut Encoder, text: &str) -> Result<Vec<f32>> {
    let outputs = model.run(vec![text])?;
    let first = outputs
        .values()
        .next()
        .ok_or_else(|| IndexError::Engine("the joint export returned nothing".into()))?;
    let (shape, values) = first
        .try_extract_tensor::<f32>()
        .map_err(|error| IndexError::Engine(format!("reading the dense vector: {error}")))?;
    Ok(one_row(shape, values)?.to_vec())
}

/// One text's pplx vector, in one forward pass of its own, at unit length.
fn pooled(model: &mut Encoder, text: &str) -> Result<Vec<f32>> {
    let outputs = model.run(vec![text])?;
    let output = outputs
        .get(PPLX_OUTPUT)
        .ok_or_else(|| IndexError::Engine(format!("the pplx export has no {PPLX_OUTPUT}")))?;
    let (shape, values) = output
        .try_extract_tensor::<i8>()
        .map_err(|error| IndexError::Engine(format!("reading the pooled vector: {error}")))?;
    Ok(unit(
        one_row(shape, values)?
            .iter()
            .map(|value| f32::from(*value))
            .collect(),
    ))
}

/// The one row of an output computed for one text.
fn one_row<'a, T>(shape: &ort::value::Shape, values: &'a [T]) -> Result<&'a [T]> {
    match **shape {
        [1, width] if width > 0 => Ok(values),
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
            ("pplx", Profile::Pplx),
        ] {
            assert_eq!(Profile::parse(name), Some(profile));
        }
        assert_eq!(Profile::parse("enormous"), None);
    }

    #[test]
    fn each_profile_declares_a_distinct_identity() {
        // The identity is what a stored vector is tagged with, and what an
        // index refuses to open under another of. The width is not enough to
        // tell two spaces apart -- `accuracy` and `pplx` are both 1024 wide --
        // so two profiles sharing an identity would let incompatible vectors
        // sit in one space undetected.
        let profiles = [
            Profile::Speed,
            Profile::Balanced,
            Profile::Accuracy,
            Profile::Pplx,
        ];
        for (index, left) in profiles.iter().enumerate() {
            for right in &profiles[index + 1..] {
                assert_ne!(left.model_id(), right.model_id());
            }
        }
    }

    /// Without its directory the profile fails to load, and says which
    /// setting it wanted -- it never falls back to a download or to another
    /// model.
    #[test]
    fn pplx_without_its_directory_names_the_setting() {
        let cache = tempfile::tempdir().expect("temp dir");
        let unset = pplx(None, cache.path()).err().expect("loaded from nowhere");
        assert!(unset.to_string().contains(PPLX_DIR), "{unset}");

        let empty = tempfile::tempdir().expect("temp dir");
        let missing = pplx(Some(empty.path().to_path_buf()), cache.path())
            .err()
            .expect("loaded from an empty directory");
        assert!(missing.to_string().contains(PPLX_FILE), "{missing}");
    }

    /// The vector pplx stores is its int8 output at unit length, pointing the
    /// same way.
    #[test]
    fn a_pooled_vector_is_scaled_to_unit_length() {
        let scaled = unit(vec![3.0, -4.0, 0.0]);
        assert_eq!(scaled, vec![0.6, -0.8, 0.0]);
        assert_eq!(unit(vec![0.0; 3]), vec![0.0; 3], "nothing to scale");
    }

    /// How close a `pplx` vector must be to the Python pipeline's.
    ///
    /// Not 0.9999, and bit-identity is not on offer: that bar fails a correct
    /// pipeline. Measured on the reference texts below, the Python pipeline
    /// on ONNX Runtime 1.30, the same pipeline on 1.28, and this crate on its
    /// own 1.28 build agree pairwise at cosine 0.9967 to 0.9995 -- each build
    /// deterministic across optimization levels and thread counts, none
    /// equal to another. The export's activations are quantized afresh in
    /// every layer, so a float kernel that rounds differently flips an int8
    /// activation in the next one, and twenty-eight layers compound it.
    ///
    /// What a pipeline fault looks like against that: an end-of-text token
    /// appended moves a vector to 0.93-0.995, the last token dropped to
    /// 0.89-0.998 -- so ids are compared exactly rather than trusted to the
    /// cosine. Reading the float mean instead of the int8 output, 0.999, is
    /// refused by the output's type rather than by this.
    const ACROSS_BUILDS: f64 = 0.995;

    /// The `pplx` profile against the pipeline its trial was first scored on.
    ///
    /// The survey measured pplx through Python -- the `tokenizers` library,
    /// ONNX Runtime's Python session, the graph's int8 output at unit length
    /// -- and the end-to-end trial measures it through this crate. If the two
    /// pipelines disagreed, the second would be measuring the difference. So
    /// the export directory carries `reference.json`, written by the script
    /// that made the export: texts in six scripts, one past the 512-token
    /// window, with the ids and the vector that pipeline produced for each.
    ///
    /// Ignored: the export is published nowhere, so it needs
    /// `PAMIN_PPLX_DIR`.
    #[test]
    #[ignore = "needs the pplx export, which is not published: set PAMIN_PPLX_DIR"]
    fn pplx_embeds_as_the_pipeline_it_was_surveyed_with() {
        #[derive(serde::Deserialize)]
        struct Reference {
            text: String,
            ids: Vec<u32>,
            vector: Vec<f32>,
        }
        let dir = std::path::PathBuf::from(std::env::var_os(PPLX_DIR).expect("the export"));
        let references: Vec<Reference> = serde_json::from_slice(
            &std::fs::read(dir.join("reference.json")).expect("the export's reference.json"),
        )
        .expect("parse reference.json");
        assert!(
            references
                .iter()
                .any(|reference| reference.ids.len() == PPLX_MAX_TOKENS),
            "no reference text reaches the window, so truncation is not compared"
        );

        // Exactly: a special token added or a truncation off by one is a
        // pipeline fault the vectors alone cannot always show.
        let tokenizer = crate::tokenizer::load(&Repository::directory(&dir), PPLX_MAX_TOKENS)
            .expect("load the tokenizer");
        for reference in &references {
            let encoding = tokenizer
                .encode(reference.text.as_str(), true)
                .expect("encode");
            assert_eq!(
                encoding.get_ids(),
                reference.ids.as_slice(),
                "{:?} tokenized differently",
                reference.text
            );
        }

        let cache = tempfile::tempdir().expect("temp dir");
        let mut embedder = Embedder::load(Profile::Pplx, cache.path()).expect("load pplx");
        let mut lowest = 1.0_f64;
        for reference in &references {
            let passage = embedder
                .embed_passage(&reference.text)
                .expect("embed a passage");
            let dot = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            };
            let norm = dot(&passage, &passage).sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "not at unit length: {norm}");
            let cosine = dot(&passage, &reference.vector) / norm;
            lowest = lowest.min(cosine);
            assert!(
                cosine >= ACROSS_BUILDS,
                "{} tokens: cosine {cosine:.6} to the reference pipeline",
                reference.ids.len()
            );
            assert_eq!(
                embedder
                    .embed_query(&reference.text)
                    .expect("embed a query"),
                passage,
                "pplx takes no prefixes"
            );
        }
        println!(
            "  {} texts, ids identical, lowest cosine {lowest:.6}",
            references.len()
        );

        // And a batch is the same passes, in order.
        let texts: Vec<&str> = references.iter().map(|r| r.text.as_str()).collect();
        let alone: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| embedder.embed_passage(text).expect("embed one"))
            .collect();
        assert_eq!(
            embedder.embed_passages(&texts).expect("embed a batch"),
            alone,
            "a batch changed a vector"
        );
    }

    #[test]
    fn the_default_profile_is_accuracy() {
        assert_eq!(Profile::default(), Profile::Accuracy);
    }
}
