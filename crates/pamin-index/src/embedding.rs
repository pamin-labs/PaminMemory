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
//! Vectors leave here as float32 and the index stores them as half precision
//! (see `projection::VectorIndex`). Weights are quantized where a quantized export
//! exists: BGE-M3 runs int8 weights, and the E5 pair runs full precision
//! because the model registry publishes no quantized variant for that family.

use fastembed::{EmbeddingModel, OutputKey, Pooling, SingleBatchOutput, TextEmbedding};
use ndarray::Array2;
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
    /// Opt-in complementary dense spaces; primary BGE-M3 plus pinned PPLX.
    DualAccuracy,
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
}

impl Profile {
    /// Which E5 model this profile runs, for the two that run one.
    fn model(self) -> EmbeddingModel {
        match self {
            Self::Speed => EmbeddingModel::MultilingualE5Small,
            Self::Balanced => EmbeddingModel::MultilingualE5Base,
            Self::Accuracy | Self::DualAccuracy => {
                unreachable!("the accuracy profile runs the joint BGE-M3 export")
            }
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
            Self::Accuracy | Self::DualAccuracy => None,
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
            Self::Accuracy | Self::DualAccuracy => 1024,
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
            Self::DualAccuracy => {
                "bge-m3-int8@2b34e84df040034d4b9eabb62383a87c18955822+pplx-0.6b@2c4d510dd4a732063c31a0f70193e35067b51fd8:pool-int8-single-v2-level4"
            }
        }
    }

    pub fn secondary_dimensions(self) -> Option<u32> {
        (self == Self::DualAccuracy).then_some(1024)
    }

    pub const ALL: [Self; 4] = [
        Self::Speed,
        Self::Balanced,
        Self::Accuracy,
        Self::DualAccuracy,
    ];

    /// Parses a profile name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "speed" => Some(Self::Speed),
            "balanced" => Some(Self::Balanced),
            "accuracy" => Some(Self::Accuracy),
            "dual_accuracy" => Some(Self::DualAccuracy),
            _ => None,
        }
    }
}

/// All dense fields for one document/query; schema decides which are required.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    pub primary: Vec<f32>,
    pub secondary: Option<Vec<f32>>,
}

/// Turns text into vectors.
pub struct Embedder {
    model: Box<Encoder>,
    profile: Profile,
    device: crate::inference::Device,
    secondary: Option<(Box<Encoder>, crate::inference::Device)>,
    /// Query vectors already computed.
    ///
    /// Shared with every project on this profile, because the vector depends on
    /// the model and the text and on nothing else.
    remembered: Queries,
}

impl Embedder {
    /// Loads the model, downloading it on first use.
    ///
    /// The model is fetched lazily rather than bundled: it is larger than the
    /// binary by an order of magnitude, and a user who never searches should
    /// not pay for it.
    pub fn load(profile: Profile, cache_dir: &std::path::Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        let repository = if matches!(profile, Profile::Accuracy | Profile::DualAccuracy) {
            Repository::open_at(
                cache_dir,
                JOINT_REPOSITORY,
                if profile == Profile::DualAccuracy {
                    "2b34e84df040034d4b9eabb62383a87c18955822"
                } else {
                    "main"
                },
            )?
        } else {
            let model = profile.model();
            let info = TextEmbedding::get_model_info(&model)
                .map_err(|error| IndexError::Engine(format!("finding embedding model: {error}")))?;
            Repository::open(cache_dir, &info.model_code)?
        };
        let identity = format!(
            "embedding-query-v3:{}:{}",
            profile.model_id(),
            repository.identity(cache_dir)
        );
        let references = std::cell::RefCell::new(crate::inference::References::default());
        let long = "migration ".repeat(600);
        let fixtures: Vec<String> = if matches!(profile, Profile::Accuracy | Profile::DualAccuracy)
        {
            vec![
                "deployment rollback".into(),
                "数据库迁移失败后如何回滚？".into(),
                long,
            ]
        } else {
            // The accelerator's actual batch cap and token limit: large
            // cascade/reindex inputs are split into these validated chunks.
            (0..8)
                .map(|i| {
                    if i % 2 == 0 {
                        long.clone()
                    } else {
                        "数据库迁移失败后如何回滚？".into()
                    }
                })
                .collect()
        };
        let (model, device) = crate::inference::measured(
            &identity,
            cache_dir,
            &references,
            |device, target, _validated| load_on(profile, cache_dir, device, target),
            |model, device| {
                let mut reference = references.borrow_mut();
                // Reject an already incompatible short query before compiling
                // or allocating the much larger maximum-token/batch shape.
                let elapsed = crate::inference::time_calls(|| {
                    let mut vectors = Vec::new();
                    for query in ["deployment rollback", "数据库迁移失败后如何回滚？"]
                    {
                        let input = match profile.prefixes() {
                            Some((prefix, _)) => format!("{prefix}{query}"),
                            None => query.to_string(),
                        };
                        vectors.extend(profile_vectors(profile, model, device, vec![input])?);
                    }
                    check_vectors(
                        vectors,
                        &mut reference.queries,
                        profile.dimensions() as usize,
                    )
                })?;
                let vectors = profile_vectors(profile, model, device, fixtures.clone())?;
                check_vectors(
                    vectors,
                    &mut reference.vectors,
                    profile.dimensions() as usize,
                )?;
                Ok(elapsed)
            },
        )?;

        if device == crate::inference::Device::Cpu
            && matches!(profile, Profile::Accuracy | Profile::DualAccuracy)
        {
            crate::prepared::release(&repository.file(cache_dir, JOINT_FILE), cache_dir);
        }
        let secondary = if profile == Profile::DualAccuracy {
            let (model, device) = complementary(cache_dir)?;
            Some((Box::new(model), device))
        } else {
            None
        };
        tracing::info!(
            model = profile.model_id(),
            device = device.name(),
            "embedder loaded"
        );
        Ok(Self {
            model: Box::new(model),
            profile,
            device,
            secondary,
            remembered: Queries::default(),
        })
    }

    /// The provider selected for this model; graph assignment is logged separately.
    pub fn device(&self) -> crate::inference::Device {
        self.device
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
        Ok(self.encode_query(text)?.primary)
    }

    pub fn encode_query(&mut self, text: &str) -> Result<Encoded> {
        if let Some(known) = self.remembered.get(text) {
            return Ok(known);
        }
        let primary = match self.profile.prefixes() {
            Some((query, _)) => self.embed_one(&format!("{query}{text}")),
            None => self.embed_one(text),
        }?;
        let secondary = self
            .secondary
            .as_mut()
            .map(|(model, _)| complementary_vector(model, text))
            .transpose()?;
        let encoded = Encoded { primary, secondary };
        self.remembered.put(text, &encoded);
        Ok(encoded)
    }

    pub fn encode_passage(&mut self, text: &str) -> Result<Encoded> {
        let primary = self.embed_passage(text)?;
        let secondary = self
            .secondary
            .as_mut()
            .map(|(model, _)| complementary_vector(model, text))
            .transpose()?;
        Ok(Encoded { primary, secondary })
    }

    pub fn encode_passages(&mut self, texts: &[&str]) -> Result<Vec<Encoded>> {
        let primary = self.embed_passages(texts)?;
        primary
            .into_iter()
            .zip(texts)
            .map(|(primary, text)| {
                let secondary = self
                    .secondary
                    .as_mut()
                    .map(|(model, _)| complementary_vector(model, text))
                    .transpose()?;
                Ok(Encoded { primary, secondary })
            })
            .collect()
    }

    pub fn secondary_device(&self) -> Option<crate::inference::Device> {
        self.secondary.as_ref().map(|(_, device)| *device)
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
    /// A single space is at most four kilobytes of float payload per query;
    /// the dual profile keeps both 1024-dimensional vectors (eight kilobytes).
    /// At 256 entries that is up to one/two MiB of vector payload. Per process,
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
    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        profile_vectors(self.profile, &mut self.model, self.device, texts)
    }
}

fn profile_vectors(
    profile: Profile,
    model: &mut Encoder,
    device: crate::inference::Device,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>> {
    match profile {
        Profile::Accuracy | Profile::DualAccuracy => {
            texts.iter().map(|text| dense(model, text)).collect()
        }
        _ => e5_vectors(
            model,
            &texts,
            if device == crate::inference::Device::Cpu {
                256
            } else {
                8
            },
        ),
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

fn check_vectors(
    vectors: Vec<Vec<f32>>,
    expected: &mut Option<Vec<Vec<f32>>>,
    dimensions: usize,
) -> Result<()> {
    match expected {
        None => *expected = Some(vectors),
        Some(reference)
            if vectors.len() == reference.len()
                && vectors
                    .iter()
                    .zip(reference.iter())
                    .all(|(a, b)| compatible_vectors(a, b, dimensions)) => {}
        _ => {
            return Err(IndexError::Numerical(
                "embedding plan failed same-export compatibility".into(),
            ));
        }
    }
    Ok(())
}

/// Load an E5 model through the same scoring-session owner as BGE-M3.
/// FastEmbed still supplies its model registry and pooling implementation;
/// its private session cannot report which execution provider ran the graph.
fn load_on(
    profile: Profile,
    cache_dir: &std::path::Path,
    device: crate::inference::Device,
    target: crate::inference::Target,
) -> Result<Encoder> {
    if matches!(profile, Profile::Accuracy | Profile::DualAccuracy) {
        let repository = if profile == Profile::DualAccuracy {
            Repository::open_at(
                cache_dir,
                JOINT_REPOSITORY,
                "2b34e84df040034d4b9eabb62383a87c18955822",
            )?
        } else {
            Repository::open(cache_dir, JOINT_REPOSITORY)?
        };
        let weights = repository.file(cache_dir, JOINT_FILE);
        let encoder = joint_session(&repository, &weights, cache_dir, device, target)?;
        return Ok(encoder);
    }
    let model = profile.model();
    let info = TextEmbedding::get_model_info(&model)
        .map_err(|error| IndexError::Engine(format!("finding embedding model: {error}")))?;
    let repository = Repository::open(cache_dir, &info.model_code)?;
    let encoder = Encoder::load(
        || repository.get(&info.model_file),
        &repository,
        JOINT_MAX_TOKENS,
        target,
    )
    .map_err(|error| error.context("loading embedding model"))?;
    encoder.require_accelerator(device)?;
    Ok(encoder)
}

/// Numerical compatibility with an already indexed CPU export. This is the
/// same predeclared near-unit-cosine contract exercised by the provider test,
/// not a retrieval-quality score or a fit on a benchmark corpus.
fn compatible_vectors(actual: &[f32], expected: &[f32], dimensions: usize) -> bool {
    if actual.len() != dimensions
        || expected.len() != dimensions
        || !actual.iter().chain(expected).all(|v| v.is_finite())
    {
        return false;
    }
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let denominator = norm(actual) * norm(expected);
    denominator > 0.0
        && actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum::<f64>()
            / denominator
            >= 0.99999
}

/// E5's FastEmbed 6.1 contract: batches of 256, attention-masked mean pooling,
/// then an f32 L2 normalization. Use its pooling owner rather than another
/// implementation of the numerical reduction.
fn e5_vectors(model: &mut Encoder, texts: &[String], batch_size: usize) -> Result<Vec<Vec<f32>>> {
    let mut vectors = Vec::with_capacity(texts.len());
    let precedence: &[OutputKey] = &[
        OutputKey::OnlyOne,
        OutputKey::ByName("text_embeds"),
        OutputKey::ByName("last_hidden_state"),
        OutputKey::ByName("sentence_embedding"),
    ];
    for batch in texts.chunks(batch_size) {
        let encoded = model.encode(batch.iter().map(String::as_str).collect())?;
        let (padded, outputs) = model.run_padded(encoded)?;
        let tokens = padded[0].len();
        let masks = padded
            .iter()
            .flat_map(|encoding| {
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|value| i64::from(*value))
            })
            .collect();
        let attention_mask_array = Array2::from_shape_vec((padded.len(), tokens), masks)
            .map_err(|error| IndexError::Engine(format!("embedding mask: {error}")))?;
        let output = SingleBatchOutput {
            outputs: outputs
                .into_iter()
                .map(|(name, value)| (name.to_string(), value))
                .collect(),
            attention_mask_array,
        };
        let pooled = output
            .select_and_pool_output(&precedence, Some(Pooling::Mean))
            .map_err(|error| IndexError::Engine(format!("pooling embedding: {error}")))?;
        for row in pooled.rows() {
            let values = row
                .as_slice()
                .ok_or_else(|| IndexError::Engine("embedding row is not contiguous".into()))?;
            let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
            vectors.push(values.iter().map(|value| value / (norm + 1e-12)).collect());
        }
    }
    Ok(vectors)
}

/// Loads the same joint export through the shared provider policy.
/// Provider selection does not change the model revision or embedding prefixes.
///
/// Its int8 export is 570 MB, and loaded from the file the hub serves it is
/// copied onto the heap whole and its matrix weights packed into a second copy
/// -- see `crate::prepared`, which writes a copy the runtime maps instead, and
/// falls back to the file itself when it cannot. Once the copy has loaded the
/// download is removed, since nothing reads it again.
fn joint_session(
    repository: &Repository,
    weights: &impl crate::prepared::Download,
    cache_dir: &std::path::Path,
    device: crate::inference::Device,
    providers: impl Into<crate::inference::Target>,
) -> Result<Encoder> {
    let encoder = Encoder::load(
        || match device {
            crate::inference::Device::Cpu => crate::prepared::load_path(weights, cache_dir),
            _ => repository.get(JOINT_FILE),
        },
        repository,
        JOINT_MAX_TOKENS,
        providers,
    )
    .map_err(|error| error.context("loading embedding model"))?;
    encoder.require_accelerator(device)?;
    Ok(encoder)
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
    match **shape {
        [1, width] if width > 0 => Ok(values.to_vec()),
        _ => Err(IndexError::Engine(format!(
            "a dense output of shape {shape:?} for one text"
        ))),
    }
}

fn complementary(cache: &std::path::Path) -> Result<(Encoder, crate::inference::Device)> {
    let repository = Repository::open_at(
        cache,
        "perplexity-ai/pplx-embed-v1-0.6b",
        "2c4d510dd4a732063c31a0f70193e35067b51fd8",
    )?;
    let load = |device, target| -> Result<Encoder> {
        let model = Encoder::load(
            || {
                repository.get("onnx/model_quantized.onnx_data")?;
                crate::pplx::prepare(&repository.get("onnx/model_quantized.onnx")?, cache)
            },
            &repository,
            JOINT_MAX_TOKENS,
            target,
        )?;
        model.require_accelerator(device)?;
        Ok(model)
    };
    let references = std::cell::RefCell::new(crate::inference::References::default());
    crate::inference::measured(
        &format!(
            "complementary-query-v2:2c4d510dd4a732063c31a0f70193e35067b51fd8:{}",
            repository.identity(cache)
        ),
        cache,
        &references,
        |device, target, _validated| load(device, target),
        |model, _device| {
            let mut reference = references.borrow_mut();
            let elapsed = crate::inference::time_calls(|| {
                let vectors = ["deployment rollback", "数据库迁移失败后如何回滚？"]
                    .iter()
                    .map(|query| complementary_vector(model, query))
                    .collect::<Result<Vec<_>>>()?;
                check_vectors(vectors, &mut reference.queries, 1024)
            })?;
            let vectors = [
                "deployment rollback".to_string(),
                "数据库迁移失败后如何回滚？".to_string(),
                "migration ".repeat(600),
            ]
            .iter()
            .map(|text| complementary_vector(model, text))
            .collect::<Result<Vec<_>>>()?;
            check_vectors(vectors, &mut reference.vectors, 1024)?;
            Ok(elapsed)
        },
    )
}

fn complementary_vector(model: &mut Encoder, text: &str) -> Result<Vec<f32>> {
    let outputs = model.run(vec![text])?;
    let output = outputs.get("pooler_output_int8").ok_or_else(|| {
        IndexError::Engine("complementary export has no int8 pooled output".into())
    })?;
    let (shape, values) = output
        .try_extract_tensor::<i8>()
        .map_err(|e| IndexError::Engine(format!("complementary vector: {e}")))?;
    if **shape != [1, 1024] {
        return Err(IndexError::Engine(format!(
            "complementary vector shape {shape:?}"
        )));
    }
    normalized_pooled(values.try_into().expect("validated pooled shape"))
}

fn normalized_pooled(values: &[i8; 1024]) -> Result<Vec<f32>> {
    // Every squared i8 is an exact integer. The complete sum is at most
    // 1024 * 128^2 = 2^24, exact in f32 too. Integer reduction allows LLVM's
    // portable SIMD vectorization without reassociating floating additions
    // or changing the encoding's normalization.
    let squared: i32 = values.iter().map(|v| i32::from(*v).pow(2)).sum();
    if squared == 0 {
        return Err(IndexError::Engine("zero complementary vector".into()));
    }
    let norm = (squared as f32).sqrt();
    Ok(values.iter().map(|v| f32::from(*v) / norm).collect())
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
    known: std::collections::HashMap<String, Encoded>,
    order: std::collections::VecDeque<String>,
}

impl Queries {
    fn get(&self, query: &str) -> Option<Encoded> {
        self.known.get(query).cloned()
    }

    /// Remembers a vector, forgetting the oldest once full.
    ///
    /// Insertion order rather than use order, for the reason the reranker's is:
    /// keeping a true LRU means writing on every hit, and a query asked twice
    /// is asked twice close together.
    fn put(&mut self, query: &str, vector: &Encoded) {
        if self
            .known
            .insert(query.to_string(), vector.clone())
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
mod pooled_tests {
    use super::normalized_pooled;

    #[test]
    fn signed_pooled_normalization_matches_scalar_encoding_exactly() {
        for values in [
            [-128i8; 1024],
            [127i8; 1024],
            std::array::from_fn(|i| (i % 256) as u8 as i8),
        ] {
            let norm = values
                .iter()
                .map(|v| f32::from(*v).powi(2))
                .sum::<f32>()
                .sqrt();
            let expected: Vec<_> = values.iter().map(|v| f32::from(*v) / norm).collect();
            assert_eq!(normalized_pooled(&values).unwrap(), expected);
        }
        assert!(normalized_pooled(&[0; 1024]).is_err());
    }
}

#[cfg(test)]
mod tests {
    /// Changing an indexed embedding space silently corrupts retrieval, so
    /// both E5 profiles must retain FastEmbed's exact bytes on CPU.
    fn matches_fastembed(profile: Profile) {
        use fastembed::TextInitOptions;

        let _ = tracing_subscriber::fmt()
            .with_writer(std::io::stdout)
            .try_init();
        let models =
            std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").unwrap()).join("models");
        let options = TextInitOptions::new(profile.model())
            .with_cache_dir(models.clone())
            .with_show_download_progress(false)
            .with_execution_providers(vec![crate::inference::cpu()]);
        let mut old = TextEmbedding::try_new(options).unwrap();
        let mut current = Embedder::load(profile, &models).unwrap();
        let long = "migration rollback ".repeat(300);
        let texts = [
            "the deployment pipeline runs on continuous integration",
            "部署流水线运行在持续集成上",
            "the office coffee machine needs descaling",
            long.as_str(),
        ];
        let prefixed: Vec<String> = texts
            .iter()
            .map(|text| format!("passage: {text}"))
            .collect();
        let together = current.embed_passages(&texts).unwrap();
        let old_together = old.embed(prefixed.clone(), None).unwrap();
        assert!(
            together == old_together,
            "{profile:?}: batched output changed"
        );
        for (index, text) in texts.iter().enumerate() {
            let single = current.embed_passage(text).unwrap();
            let old_single = old.embed(vec![prefixed[index].as_str()], None).unwrap();
            assert!(
                single == old_single[0],
                "{profile:?}: single output changed at {index}"
            );
            if single != together[index] {
                let maximum = single
                    .iter()
                    .zip(&together[index])
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0f32, f32::max);
                println!("{profile:?} batch-versus-single at {index}: max {maximum}");
            }
        }
        let query = "how does deployment work";
        assert_eq!(
            current.embed_query(query).unwrap(),
            old.embed(vec![format!("query: {query}")], None).unwrap()[0],
        );
    }

    #[test]
    #[ignore = "loads the real multilingual E5 small model"]
    fn speed_shared_session_matches_fastembed() {
        matches_fastembed(Profile::Speed);
    }

    #[test]
    #[ignore = "loads the real multilingual E5 base model"]
    fn balanced_shared_session_matches_fastembed() {
        matches_fastembed(Profile::Balanced);
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "isolates same-export BGE-M3 CoreML compute units against CPU"]
    fn incompatible_coreml_joint_is_rejected() {
        let _ = tracing_subscriber::fmt()
            .with_writer(std::io::stdout)
            .try_init();
        let cache = std::path::PathBuf::from(std::env::var("PAMIN_TEST_MODEL_CACHE").unwrap());
        let repository = Repository::open(&cache, JOINT_REPOSITORY).unwrap();
        let weights = repository.file(&cache, JOINT_FILE);
        let inputs = [
            "How does a failed database migration roll back?",
            "数据库迁移失败后如何回滚？",
        ];
        let run = |device, provider| {
            let model = if let Some(path) = std::env::var_os("PAMIN_TEST_JOINT_GRAPH") {
                Encoder::load(
                    || Ok(std::path::PathBuf::from(path)),
                    &repository,
                    JOINT_MAX_TOKENS,
                    vec![provider],
                )
                .unwrap()
            } else {
                joint_session(&repository, &weights, &cache, device, vec![provider]).unwrap()
            };
            let mut embedder = Embedder {
                model: Box::new(model),
                profile: Profile::Accuracy,
                device,
                secondary: None,
                remembered: Queries::default(),
            };
            inputs
                .iter()
                .map(|text| embedder.embed_passage(text).unwrap())
                .collect::<Vec<_>>()
        };
        let cpu = run(crate::inference::Device::Cpu, crate::inference::cpu());
        let path = std::env::var_os("PAMIN_TEST_JOINT_GRAPH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| crate::prepared::load_path(&weights, &cache).unwrap());
        let session = crate::inference::options(vec![crate::inference::cpu()])
            .unwrap()
            .with_disabled_optimizers("MatMulAddFusion")
            .unwrap()
            .commit_from_file(&path)
            .unwrap();
        let model = Encoder::from_session(session, &repository, JOINT_MAX_TOKENS).unwrap();
        let mut control = Embedder {
            model: Box::new(model),
            profile: Profile::Accuracy,
            device: crate::inference::Device::Cpu,
            secondary: None,
            remembered: Queries::default(),
        };
        for (index, text) in inputs.iter().enumerate() {
            let actual = control.embed_passage(text).unwrap();
            let cosine = actual
                .iter()
                .zip(&cpu[index])
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>()
                / (actual
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    * cpu[index]
                        .iter()
                        .map(|v| f64::from(*v).powi(2))
                        .sum::<f64>()
                        .sqrt());
            println!("CPU without MatMulAddFusion case {index} cosine {cosine:.9}");
        }
        drop(control);
        let mut cosines = Vec::new();
        for units in [
            ort::ep::coreml::ComputeUnits::CPUOnly,
            ort::ep::coreml::ComputeUnits::CPUAndGPU,
            ort::ep::coreml::ComputeUnits::All,
        ] {
            let provider = ort::ep::CoreML::default()
                .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                .with_compute_units(units)
                .build()
                .error_on_failure();
            let actual = run(crate::inference::Device::CoreMl, provider);
            for (index, (left, right)) in actual.iter().zip(&cpu).enumerate() {
                let dot: f64 = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let norm = |values: &Vec<f32>| {
                    values
                        .iter()
                        .map(|v| f64::from(*v).powi(2))
                        .sum::<f64>()
                        .sqrt()
                };
                let cosine = dot / (norm(left) * norm(right));
                let maximum = left
                    .iter()
                    .zip(right)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "units {units:?} case {index} cosine {cosine:.9} max_abs {maximum:.9} norm_ratio {:.9}",
                    norm(left) / norm(right)
                );
                cosines.push(cosine);
            }
        }
        assert!(
            cosines.iter().any(|c| *c < 0.99999),
            "control must exercise the known incompatible CoreML graph"
        );
        let mut accepted = Embedder::load(Profile::Accuracy, &cache).unwrap();
        assert_eq!(
            accepted.device(),
            crate::inference::Device::Cpu,
            "incompatible graph accepted"
        );
        for (index, text) in inputs.iter().enumerate() {
            assert_eq!(accepted.embed_passage(text).unwrap(), cpu[index]);
        }
    }

    #[test]
    fn compatibility_refuses_invalid_or_changed_embedding_vectors() {
        assert!(compatible_vectors(&[1.0, 0.0], &[1.0, 0.0], 2));
        assert!(!compatible_vectors(&[0.0, 1.0], &[1.0, 0.0], 2));
        assert!(!compatible_vectors(&[f32::NAN, 0.0], &[1.0, 0.0], 2));
        assert!(!compatible_vectors(&[0.0, 0.0], &[1.0, 0.0], 2));
        assert!(!compatible_vectors(&[1.0], &[1.0, 0.0], 2));
    }

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

        queries.put(
            "how does deployment work",
            &Encoded {
                primary: vec![1.0, 2.0, 3.0],
                secondary: None,
            },
        );
        queries.put(
            "how does deployment fail",
            &Encoded {
                primary: vec![4.0, 5.0, 6.0],
                secondary: None,
            },
        );

        assert_eq!(
            queries.get("how does deployment work"),
            Some(Encoded {
                primary: vec![1.0, 2.0, 3.0],
                secondary: None
            })
        );
        assert_eq!(
            queries.get("how does deployment fail"),
            Some(Encoded {
                primary: vec![4.0, 5.0, 6.0],
                secondary: None
            })
        );
        assert!(queries.get("how does deployment").is_none());
    }

    /// The oldest is forgotten, and the cache stays the size it says.
    #[test]
    fn the_oldest_query_is_forgotten_once_it_is_full() {
        let mut queries = super::Queries::default();
        let cap = super::Embedder::REMEMBERED_QUERIES;
        for i in 0..cap + 10 {
            queries.put(
                &format!("query {i}"),
                &Encoded {
                    primary: vec![i as f32],
                    secondary: None,
                },
            );
        }
        assert_eq!(queries.known.len(), cap);
        assert!(queries.get("query 0").is_none(), "the oldest survived");
        assert_eq!(
            queries.get(&format!("query {}", cap + 9)),
            Some(Encoded {
                primary: vec![(cap + 9) as f32],
                secondary: None
            }),
            "the newest was lost"
        );
    }

    /// Rewriting a query does not grow the queue behind it.
    #[test]
    fn remembering_a_query_twice_does_not_shorten_the_cache() {
        let mut queries = super::Queries::default();
        for _ in 0..super::Embedder::REMEMBERED_QUERIES * 2 {
            queries.put(
                "the same question",
                &Encoded {
                    primary: vec![1.0],
                    secondary: None,
                },
            );
        }
        assert_eq!(queries.order.len(), 1);
        assert_eq!(
            queries.get("the same question"),
            Some(Encoded {
                primary: vec![1.0],
                secondary: None
            })
        );
    }

    use super::*;

    #[test]
    fn profile_names_round_trip() {
        for (name, profile) in [
            ("speed", Profile::Speed),
            ("balanced", Profile::Balanced),
            ("accuracy", Profile::Accuracy),
            ("dual_accuracy", Profile::DualAccuracy),
        ] {
            assert_eq!(Profile::parse(name), Some(profile));
        }
        assert_eq!(Profile::parse("enormous"), None);
    }

    #[test]
    fn each_profile_declares_a_distinct_encoding_identity() {
        // The width is what the index is built for and the identity is what a
        // stored vector is tagged with, so two profiles sharing either would
        // let incompatible vectors sit in one space undetected.
        let profiles = Profile::ALL;
        for (index, left) in profiles.iter().enumerate() {
            for right in &profiles[index + 1..] {
                assert_ne!(left.model_id(), right.model_id());
            }
        }
    }

    #[test]
    fn the_default_profile_is_accuracy() {
        assert_eq!(Profile::default(), Profile::Accuracy);
    }
}
