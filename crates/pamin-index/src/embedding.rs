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
//! exists: BGE-M3 runs int8 weights.

use serde::{Deserialize, Serialize};

use crate::encoder::Encoder;
use crate::error::{IndexError, Result};
use crate::hub::Repository;

/// Which embedding model to run.
///
/// One now. `speed` and `balanced` ran the multilingual E5 pair and are
/// retired: the quantized BGE-M3 export beat both on retrieval by a distance
/// and costs little more. It is permissively licensed. EmbeddingGemma scores
/// well and would otherwise be a candidate, but it carries usage restrictions
/// that must be passed on to downstream users, which is not a burden to attach
/// to an open-source default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    /// 1024 dimensions, int8 weights, and by a distance the best cross-lingual
    /// recall of the profiles there were.
    ///
    /// Run through the joint BGE-M3 export, which produces dense, sparse and
    /// ColBERT representations in one forward pass. Only the dense one is
    /// kept. The sparse arm duplicates the two lexical channels already in
    /// place and is worth 0.2 points of cross-lingual nDCG by its own authors'
    /// ablation; the ColBERT arm is one 1024-wide vector per token, which for
    /// a project of seven million documents is terabytes.
    ///
    /// It was the default before it was the only one, because it is not the
    /// trade its name implies. Against the retired `balanced` it doubles cross-lingual nDCG@10, matches it monolingually,
    /// occupies 560 MB against 1.1 GB, and costs 35 ms per query against 26.
    /// The int8 export is what makes all of that true at once; the
    /// full-precision one is 2.2 GB and was the reason this profile used to be
    /// described as an order of magnitude more expensive.
    #[default]
    Accuracy,
}

impl Profile {
    /// The vector width this profile produces.
    ///
    /// The index is built for one width. Mixing embedding spaces yields
    /// distances that mean nothing, so changing profile reindexes.
    pub fn dimensions(self) -> u32 {
        match self {
            Self::Accuracy => 1024,
        }
    }

    /// The identifier recorded alongside every stored vector, so a later read
    /// can tell which space a vector belongs to.
    pub fn model_id(self) -> &'static str {
        match self {
            // The quantized export rather than the base model: int8 weights
            // produce vectors close to the full-precision ones and not equal
            // to them, and the recorded identity is what stops two encodings
            // sharing one index.
            Self::Accuracy => "gpahal/bge-m3-onnx-int8",
        }
    }

    pub const ALL: [Self; 1] = [Self::Accuracy];

    /// Parses a profile name.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "accuracy" => Some(Self::Accuracy),
            _ => None,
        }
    }
}

/// Turns text into vectors.
pub struct Embedder {
    cache_dir: std::path::PathBuf,
    model: Box<Encoder>,
    profile: Profile,
    device: crate::inference::Device,
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
        let (model, device) = primary_model(profile, cache_dir)?;
        tracing::info!(
            model = profile.model_id(),
            device = device.name(),
            "embedder loaded"
        );
        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            model: Box::new(model),
            profile,
            device,
            remembered: Queries::default(),
        })
    }

    /// The provider selected for this model; graph assignment is logged separately.
    pub fn device(&self) -> crate::inference::Device {
        self.device
    }

    /// The execution target this model runs on, as its execution plans are
    /// keyed: `npu:<provider>:<id>` for an NPU, the device's name otherwise.
    pub fn target(&self) -> String {
        crate::inference::target_key(&self.model, self.device)
    }

    /// Every execution plan this model has run on in this process, oldest
    /// first, the current one last: more than one when a failure fell back
    /// or a revalidation moved it. See [`crate::PlanUse`].
    pub fn plans(&self) -> Vec<crate::PlanUse> {
        crate::inference::plans_of(&self.model, self.device)
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    /// Embeds one passage for storage.
    pub fn embed_passage(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_one(text)
    }

    /// Embeds one query.
    ///
    /// Remembered, because the pass is the most expensive thing a search does
    /// -- 16.9 ms of a search at the shipping profile -- and a query is a pure
    /// function of the model and the text. What makes it worth remembering is
    /// the same thing that makes the reranker remember its scores: an agent
    /// retries, widens a limit, and asks again after writing something. Only
    /// queries. A passage is embedded once in its life, so a cache of those
    /// would hold the corpus and never be read.
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.revalidate();
        if let Some(known) = self.remembered.get(text) {
            return Ok(known);
        }
        let vector = self.embed_one(text)?;
        self.remembered.put(text, &vector);
        Ok(vector)
    }

    /// Embeds many passages in one forward pass.
    pub fn embed_passages(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.run(texts.iter().map(|text| (*text).to_string()).collect())
    }

    /// How many queries this remembers, per profile.
    ///
    /// A vector is at most four kilobytes of float payload per query, so at
    /// 256 entries that is up to one MiB of vector payload. Per process,
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
    fn revalidate(&mut self) {
        if crate::inference::needs_revalidation(&self.model) {
            self.remembered = Queries::default();
        }
    }

    fn run(&mut self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.revalidate();
        let profile = self.profile;
        let cache = &self.cache_dir;
        let (vectors, replaced) = crate::inference::retry_model(
            &mut self.model,
            &mut self.device,
            cache,
            |model, _| vectors_of(model, &texts),
            || primary_model(profile, cache).map(|(model, device)| (Box::new(model), device)),
        )?;
        if replaced {
            self.remembered = Queries::default();
        }
        Ok(vectors)
    }
}

fn primary_model(
    profile: Profile,
    cache_dir: &std::path::Path,
) -> Result<(Encoder, crate::inference::Device)> {
    std::fs::create_dir_all(cache_dir)?;
    let repository = Repository::open_at(cache_dir, JOINT_REPOSITORY, "main")?;
    let identity = plan_key(profile.model_id(), &repository.identity(cache_dir));
    let references = std::cell::RefCell::new(crate::inference::References::default());
    let long = "migration ".repeat(600);
    let fixtures: Vec<String> = vec![
        "deployment rollback".into(),
        "数据库迁移失败后如何回滚？".into(),
        long,
    ];
    let (model, device) = crate::inference::measured(
        &identity,
        cache_dir,
        &references,
        crate::inference::ReferenceShape {
            vectors: Some((fixtures.len(), profile.dimensions() as usize)),
            queries: Some((2, profile.dimensions() as usize)),
            scores: None,
        },
        |device, target, _validated| load_on(cache_dir, device, target),
        |model, _device| {
            let mut reference = references.borrow_mut();
            // Reject an already incompatible short query before compiling
            // or allocating the much larger maximum-token/batch shape.
            let elapsed = crate::inference::time_calls(|| {
                let mut vectors = Vec::new();
                for query in ["deployment rollback", "数据库迁移失败后如何回滚？"] {
                    vectors.extend(vectors_of(model, &[query.to_string()])?);
                }
                check_vectors(
                    vectors,
                    &mut reference.queries,
                    profile.dimensions() as usize,
                )
            })?;
            let vectors = vectors_of(model, &fixtures)?;
            check_vectors(
                vectors,
                &mut reference.vectors,
                profile.dimensions() as usize,
            )?;
            Ok(elapsed)
        },
    )?;

    if device == crate::inference::Device::Cpu {
        crate::prepared::release(&repository.file(cache_dir, JOINT_FILE), cache_dir);
    }
    Ok((model, device))
}

/// The key the embedder's execution plans are measured and cached under.
///
/// `e5-cpu256-accelerator8` names how the retired E5 profiles batched. It
/// stays in the key, because every host's measured plan for this model is
/// cached under it and a new spelling would measure them all again.
fn plan_key(model_id: &str, repository: &str) -> String {
    format!(
        "embedding-query-v4:max512:joint-singleton:e5-cpu256-accelerator8:{model_id}:{repository}"
    )
}

/// One vector per text, each text run as a pass of its own -- see
/// [`Embedder::run`].
fn vectors_of(model: &mut Encoder, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    model.run_each(alone(model, texts)?, dense)
}

/// Each of `texts` tokenized as a batch of its own -- see [`Embedder::run`].
fn alone(model: &Encoder, texts: &[String]) -> Result<Vec<Vec<tokenizers::Encoding>>> {
    Ok(model
        .encode(texts.iter().map(String::as_str).collect())?
        .into_iter()
        .map(|encoding| vec![encoding])
        .collect())
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
    if vectors.is_empty()
        || vectors.iter().any(|vector| {
            vector.len() != dimensions
                || !vector.iter().all(|value| value.is_finite())
                || !vector.iter().any(|value| *value != 0.0)
        })
    {
        return Err(IndexError::Numerical(
            "invalid CPU-space embedding fixture".into(),
        ));
    }
    match expected {
        None => *expected = Some(vectors),
        Some(reference)
            if vectors.len() == reference.len()
                && vectors
                    .iter()
                    .zip(reference.iter())
                    .all(|(a, b)| compatible_vectors(a, b, dimensions))
                && preserves_fixture_retrieval(&vectors, reference) => {}
        _ => {
            return Err(IndexError::Numerical(
                "embedding plan failed same-export compatibility".into(),
            ));
        }
    }
    Ok(())
}

/// Loads the joint BGE-M3 export for `device`.
fn load_on(
    cache_dir: &std::path::Path,
    device: crate::inference::Device,
    target: crate::inference::Target,
) -> Result<Encoder> {
    let repository = Repository::open(cache_dir, JOINT_REPOSITORY)?;
    let weights = repository.file(cache_dir, JOINT_FILE);
    joint_session(&repository, &weights, cache_dir, device, target)
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
            >= 0.9999
}

/// Query the fixed CPU-space fixture index, rather than certifying a plan
/// from coordinate drift alone. This bounded startup smoke test is not a
/// corpus-quality claim; its tiny conservative reductions are diagnostic.
fn preserves_fixture_retrieval(actual: &[Vec<f32>], cpu: &[Vec<f32>]) -> bool {
    let similarity = |a: &[f32], b: &[f32]| {
        let dot = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (norm(a) * norm(b))
    };
    let nearest = |query: &[f32]| {
        cpu.iter()
            .enumerate()
            .max_by(|(left, a), (right, b)| {
                similarity(query, a)
                    .total_cmp(&similarity(query, b))
                    .then_with(|| right.cmp(left))
            })
            .map(|(index, _)| index)
    };
    actual
        .iter()
        .zip(cpu)
        .all(|(query, reference)| nearest(query) == nearest(reference))
}

/// Loads the same joint export through the shared provider policy.
/// Provider selection does not change the model revision or embedding prefixes.
///
/// Its int8 export is 570 MB, and loaded from the file the hub serves it is
/// copied onto the heap whole and its matrix weights packed into a second copy
/// -- see `crate::prepared`, which writes a copy the runtime maps instead, and
/// falls back to the file itself when it cannot. Once the copy has loaded the
/// download is removed, since nothing reads it again.
///
/// On the CPU loaded [`side_by_side`](Encoder::side_by_side): a query keeps a
/// pass of the whole budget, its fastest, and stored texts run several
/// one-threaded passes at once. Measured through `Embedder` on four cores
/// (ONNX Runtime 1.28, Xeon with AMX), a query's median was 37.7 ms at four
/// threads and 52.4 at one; and no vector changes: 500 texts as passages and
/// 100 as queries came out bit-identical at one thread and at four.
fn joint_session(
    repository: &Repository,
    weights: &impl crate::prepared::Download,
    cache_dir: &std::path::Path,
    device: crate::inference::Device,
    providers: impl Into<crate::inference::Target>,
) -> Result<Encoder> {
    if device == crate::inference::Device::Cpu {
        let copy = crate::prepared::load_path(weights, cache_dir)?;
        let tokenizer = crate::tokenizer::load(repository, JOINT_MAX_TOKENS)?;
        return Encoder::side_by_side(copy, crate::inference::budget()?, tokenizer)
            .map_err(|error| error.context("loading embedding model"));
    }
    let encoder = Encoder::load(
        || repository.get(JOINT_FILE),
        repository,
        JOINT_MAX_TOKENS,
        providers,
    )
    .map_err(|error| error.context("loading embedding model"))?;
    encoder.require_accelerator(device)?;
    Ok(encoder)
}

/// One text's dense BGE-M3 vector, from the outputs of a forward pass of its
/// own.
///
/// The export's first output, as `fastembed` read it: the graph pools and
/// normalizes the vector itself, so it is used as it comes. The sparse and
/// ColBERT representations come back from the same pass and are dropped. They
/// are not free -- the pass computes them -- but neither is wanted, and no
/// cheaper export of this model's int8 weights exists.
fn dense(outputs: ort::session::SessionOutputs<'_>) -> Result<Vec<f32>> {
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
                cache_dir: cache.clone(),
                model: Box::new(model),
                profile: Profile::Accuracy,
                device,
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
            cache_dir: cache.clone(),
            model: Box::new(model),
            profile: Profile::Accuracy,
            device: crate::inference::Device::Cpu,
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

    /// BGE-M3's stored texts, run side by side, are the vectors each gets
    /// alone on the query's session, bit for bit and in order: its query
    /// session has every core, and its vectors do not change with the thread
    /// count.
    ///
    /// Alone is one pass after another; side by side is as many passes at once
    /// as there are cores, each on a session of its own, taking texts in
    /// whatever order they finish. Lengths from a few tokens to the 512-token
    /// cap, more texts than cores, so every session runs several.
    /// The model is read from, or downloaded into, `PAMIN_TEST_MODEL_CACHE`.
    #[test]
    #[ignore = "downloads BGE-M3's int8 export; set PAMIN_TEST_MODEL_CACHE"]
    fn bge_side_by_side_is_each_text_alone() {
        let cache = std::path::PathBuf::from(
            std::env::var_os("PAMIN_TEST_MODEL_CACHE").expect("set PAMIN_TEST_MODEL_CACHE"),
        );
        let mut embedder = Embedder::load(Profile::Accuracy, &cache).expect("load the model");
        assert_eq!(embedder.device(), crate::inference::Device::Cpu);
        let long = "migration rollback ".repeat(300);
        let texts = [
            "deployment rollback",
            "数据库迁移失败后如何回滚？",
            "the office coffee machine needs descaling",
            long.as_str(),
            "Roll back a failed deployment to the previous application version.",
            "部署流水线运行在持续集成上",
            "Bake a chocolate cake and serve it with fresh strawberries.",
            "Die Datenbankmigration ist fehlgeschlagen.",
            "ok",
            "La migración de la base de datos falló y hubo que restaurar la copia de seguridad.",
        ];
        let alone: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| bits(&embedder.embed_passage(text).unwrap()))
            .collect();
        assert_eq!(embedder.model.side_by_side_sessions(), 0);
        let together: Vec<Vec<u32>> = embedder
            .embed_passages(&texts)
            .unwrap()
            .iter()
            .map(|vector| bits(vector))
            .collect();
        assert!(
            embedder.model.side_by_side_sessions() > 1,
            "nothing ran side by side; this needs more than one core"
        );
        assert!(alone == together, "a vector changed side by side");
    }

    fn bits(vector: &[f32]) -> Vec<u32> {
        vector.iter().map(|value| value.to_bits()).collect()
    }

    #[test]
    fn ordinary_drift_preserving_cpu_space_retrieval_is_accepted() {
        let cpu = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let actual = vec![vec![0.99998, 0.0063245], vec![0.0063245, 0.99998]];
        assert!(
            actual
                .iter()
                .zip(&cpu)
                .all(|(a, b)| compatible_vectors(a, b, 2))
        );
        assert!(preserves_fixture_retrieval(&actual, &cpu));
        assert!(!preserves_fixture_retrieval(
            &[cpu[1].clone(), cpu[0].clone()],
            &cpu
        ));
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
        assert_eq!(Profile::parse("accuracy"), Some(Profile::Accuracy));
        assert_eq!(Profile::parse("enormous"), None);
    }

    /// The recorded identity and the plan key stay what they were when there
    /// were several profiles: every existing index is tagged with the first,
    /// and every host's measured execution plan is cached under the second.
    #[test]
    fn the_model_keeps_its_recorded_identity_and_plan_key() {
        assert_eq!(Profile::Accuracy.model_id(), "gpahal/bge-m3-onnx-int8");
        assert_eq!(
            plan_key(Profile::Accuracy.model_id(), "gpahal/bge-m3-onnx-int8@abc"),
            "embedding-query-v4:max512:joint-singleton:e5-cpu256-accelerator8:\
             gpahal/bge-m3-onnx-int8:gpahal/bge-m3-onnx-int8@abc"
        );
    }

    #[test]
    fn the_default_profile_is_accuracy() {
        assert_eq!(Profile::default(), Profile::Accuracy);
    }
}
