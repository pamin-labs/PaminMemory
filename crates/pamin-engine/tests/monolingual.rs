//! Retrieval on real passages in one language, on somebody else's benchmark.
//!
//! [`crosslingual.rs`](crosslingual.rs) measures the thing this project
//! claims, on the corpus published for measuring it, and its own notes name
//! the limitation: XQuAD-R is parallel text, so the eleven versions of a
//! sentence are translations of each other and a question shares its literal
//! wording with the sentence that answers it. Both are properties of a
//! translation benchmark rather than of anything anyone stores, and both
//! flatter the lexical channels.
//!
//! MIRACL's Swahili dev split is the other shape: 131,924 Wikipedia passages
//! averaging a couple of hundred characters, 482 questions people actually
//! asked, relevance judged by people, one language throughout. Nothing is
//! parallel and nothing was translated, so a question and its answer share
//! whatever words they happen to share.
//!
//! **This harness exists because three numbers were published without one.**
//! `docs/measured.md`, `docs/benchmarks.md`, the ADR and
//! `pamin-index/src/reranking.rs` all quote MIRACL nDCG@10 for the three
//! rerank tiers, and `docs/measured.md` says the harness behind its figures is
//! in this repository -- which was true of the cross-lingual rows and false of
//! these. They came from a program nobody committed. Anyone re-checking them,
//! or breaking them, had no way to find out.
//!
//! ## Two tests, one corpus
//!
//! | test | what runs | what it answers |
//! |---|---|---|
//! | `the_model_ranks_real_passages` | the embedder alone, compared against every passage | how far the embedding space itself gets |
//! | `search_ranks_real_passages` | the shipped path: four channels, fusion, graph, reranker | how far the product gets |
//!
//! The difference between them is what fusing four channels does to a dense
//! ranking, and it is the reason to spend hours on this corpus rather than
//! reuse the other one. On XQuAD-R that difference is two numbers pointing
//! opposite ways -- fusion costs 0.0635 cross-lingual and gains 0.1213
//! same-language -- and both are artifacts of parallel text: the cross-lingual
//! group asks the lexical channels to find a sentence in a language the query
//! is not in, and the same-language group hands them a question built from the
//! answer's own words. Neither is a workload. Here there is one group, the
//! query is not the answer's translation, and the difference is a single
//! number about the shape of retrieval people do.
//!
//! The first test deliberately does not go through the vector index, for the
//! same reason [`crosslingual.rs`](crosslingual.rs) says: an approximate index
//! answers a slightly different question than the model does, and a loss that
//! could have come from either is a measurement of neither.
//!
//! ## Running it
//!
//! Ignored by default, and the most expensive thing in this repository: it
//! downloads 10 MB of corpus, downloads model weights, and embeds a hundred
//! and thirty-two thousand passages. On the `accuracy` profile that is a
//! forward pass of XLM-RoBERTa-large each, one at a time -- the joint export
//! is not batchable, see `pamin-index/src/embedding.rs` -- and it is hours.
//! Both the vector cache and the index resume, so losing the process costs
//! what it had not finished rather than everything.
//!
//! ```text
//! cargo test -p pamin-engine --test monolingual -- --ignored --nocapture
//! ```
//!
//! | variable | what it does |
//! |---|---|
//! | `MIRACL_DIR` | where the three dataset files are, instead of fetching them |
//! | `MIRACL_MAX_DOCS` | cap the corpus, for checking the harness runs. **Not the benchmark**: the floors are not asserted and the number is not comparable with anyone's |
//! | `PAMIN_PROFILE` | which embedding profile, default `accuracy`. The floors are asserted for that one only |
//! | `TIERS` | run all three rerank tiers instead of the default |
//! | `SWEEP` | run fusion settings instead of the shipped path |
//!
//! The dataset is not vendored. The corpus is Wikipedia text under
//! CC-BY-SA-3.0 and this repository is Apache-2.0, and it is 40 MB unpacked.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_core::{Channel, Fusion};
use pamin_engine::{Depths, Engine, Write};
use pamin_index::{Access, Embedder, Profile, Rerank};
use pamin_store::Workspace;

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

/// Where the three files come from.
///
/// Queries and judgements are in `miracl/miracl`; the passages are in
/// `miracl/miracl-corpus`, which is the split MIRACL publishes them in.
const TOPICS: &str = "https://huggingface.co/datasets/miracl/miracl/resolve/main/\
                      miracl-v1.0-sw/topics/topics.miracl-v1.0-sw-dev.tsv";
const QRELS: &str = "https://huggingface.co/datasets/miracl/miracl/resolve/main/\
                     miracl-v1.0-sw/qrels/qrels.miracl-v1.0-sw-dev.tsv";
const CORPUS: &str = "https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/\
                      miracl-corpus-v1.0-sw/docs-0.jsonl.gz";

/// Where nDCG is cut.
const NDCG_AT: usize = 10;

/// Where recall is cut.
const RECALL_AT: usize = 50;

/// How deep a ranking is taken before scoring, so recall@50 can be reached.
const DEPTH: usize = RECALL_AT + 1;

/// The channel and graph depths to search at: the shipped defaults.
const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

/// The profile the floors were measured against, and the product default.
const DEFAULT_PROFILE: &str = "accuracy";

/// The one group. MIRACL has no split inside a language and inventing one
/// would be reporting a number nobody else reports.
const GROUP: &str = "swahili";

/// One passage, as indexed.
struct Passage {
    /// MIRACL's own document id, `article#paragraph`. Used as the topic name,
    /// and deliberately unlike anything in the text: mention derivation looks
    /// for topic names inside content, and passages that named each other
    /// would measure the graph channel on relationships nobody asserted.
    docid: String,
    /// Title and body, which is what MIRACL's own baselines index.
    text: String,
}

/// One question and the passages judged relevant to it.
struct Query {
    text: String,
    relevant: HashSet<String>,
}

struct Corpus {
    passages: Vec<Passage>,
    queries: Vec<Query>,
    /// True when `MIRACL_MAX_DOCS` cut the corpus, which disqualifies every
    /// number this run produces from being compared with anybody's.
    capped: bool,
}

impl Corpus {
    fn load() -> Self {
        let dir = dataset_dir();
        fetch(&dir);

        let cap = std::env::var("MIRACL_MAX_DOCS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok());

        let raw = std::fs::read_to_string(dir.join("docs.jsonl")).expect("read the corpus");
        let mut passages = Vec::new();
        for line in raw.lines() {
            if let Some(cap) = cap
                && passages.len() >= cap
            {
                break;
            }
            let document: serde_json::Value =
                serde_json::from_str(line).expect("a corpus line is JSON");
            let docid = document["docid"].as_str().expect("docid").to_string();
            let title = document["title"].as_str().unwrap_or_default();
            let body = document["text"].as_str().unwrap_or_default();
            passages.push(Passage {
                docid,
                text: if title.is_empty() {
                    body.to_string()
                } else {
                    format!("{title}\n{body}")
                },
            });
        }
        assert!(!passages.is_empty(), "the corpus is empty");

        let indexed: HashSet<&str> = passages
            .iter()
            .map(|passage| passage.docid.as_str())
            .collect();

        // `qid \t Q0 \t docid \t relevance`, TREC's own shape. MIRACL judges
        // binary, so anything above zero is relevant.
        let mut judged: HashMap<String, HashSet<String>> = HashMap::new();
        let qrels = std::fs::read_to_string(dir.join("qrels.tsv")).expect("read the judgements");
        for line in qrels.lines() {
            let mut fields = line.split('\t');
            let (Some(qid), Some(_), Some(docid), Some(grade)) =
                (fields.next(), fields.next(), fields.next(), fields.next())
            else {
                panic!("a judgement line has four fields: {line:?}");
            };
            if grade.trim().parse::<i32>().unwrap_or(0) <= 0 || !indexed.contains(docid) {
                continue;
            }
            judged
                .entry(qid.to_string())
                .or_default()
                .insert(docid.to_string());
        }

        // A query whose every relevant passage was cut by the cap would score
        // zero and drag the mean down while measuring nothing, so it is
        // dropped rather than scored -- which is only reachable under the cap.
        let topics = std::fs::read_to_string(dir.join("topics.tsv")).expect("read the queries");
        let queries: Vec<Query> = topics
            .lines()
            .filter_map(|line| {
                let (qid, text) = line.split_once('\t')?;
                let relevant = judged.remove(qid)?;
                (!relevant.is_empty()).then(|| Query {
                    text: text.to_string(),
                    relevant,
                })
            })
            .collect();
        assert!(!queries.is_empty(), "no query has a judged passage");

        Self {
            passages,
            queries,
            capped: cap.is_some(),
        }
    }

    /// A short digest, so a corpus that has changed does not reuse a cache.
    fn fingerprint(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for passage in &self.passages {
            for byte in passage.docid.bytes().chain(passage.text.bytes()) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{hash:016x}")
    }

    fn describe(&self, named: &str) {
        println!(
            "  {} passages, {} judged queries, profile {named}{}",
            self.passages.len(),
            self.queries.len(),
            if self.capped {
                " -- CAPPED, not the benchmark"
            } else {
                ""
            }
        );
    }
}

/// Where the dataset lives.
fn dataset_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MIRACL_DIR") {
        return PathBuf::from(dir);
    }
    eval_home().join("miracl-sw")
}

fn eval_home() -> PathBuf {
    std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"))
}

/// Downloads the three files that are not there yet.
///
/// Through `curl` and `gzip` rather than an HTTP client and a decompressor,
/// for the reason [`crosslingual.rs`](crosslingual.rs) gives: a dependency
/// added for a test CI never runs is a dependency the whole project carries,
/// and `ci/budget.py` counts them. Each file lands under a partial name and is
/// renamed on success, so an interrupted download is not mistaken for a
/// complete one.
fn fetch(dir: &Path) {
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));

    for (name, url) in [("topics.tsv", TOPICS), ("qrels.tsv", QRELS)] {
        download(dir, name, url);
    }

    if !dir.join("docs.jsonl").exists() {
        download(dir, "docs-0.jsonl.gz", CORPUS);
        let unpacked = Command::new("gzip")
            .arg("-dkf")
            .arg(dir.join("docs-0.jsonl.gz"))
            .status();
        assert!(
            matches!(unpacked, Ok(status) if status.success()),
            "could not unpack the corpus with `gzip`"
        );
        std::fs::rename(dir.join("docs-0.jsonl"), dir.join("docs.jsonl"))
            .expect("name the unpacked corpus");
    }
}

fn download(dir: &Path, name: &str, url: &str) {
    let path = dir.join(name);
    if path.exists() {
        return;
    }
    let partial = dir.join(format!("{name}.part"));
    let status = Command::new("curl")
        .args(["-sSLf", "--max-time", "900", "-o"])
        .arg(&partial)
        .arg(url)
        .status();
    assert!(
        matches!(status, Ok(status) if status.success()),
        "could not fetch {url}\n\
         The dataset is not vendored: the corpus is Wikipedia text under CC-BY-SA-3.0 and\n\
         this repository is Apache-2.0. Download the three files by hand and point\n\
         MIRACL_DIR at the directory, or make `curl` and huggingface.co reachable."
    );
    std::fs::rename(&partial, &path).expect("name the downloaded file");
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// The running score over every query.
#[derive(Default)]
struct Scores {
    queries: usize,
    ndcg: f64,
    recall: f64,
    /// Relevant passages inside the shortlist but below rank ten -- the whole
    /// space a second pass over the shortlist could still fix.
    deep: usize,
    /// Queries with at least one of those.
    with_work: usize,
}

impl Scores {
    fn add(&mut self, ranked: &[String], relevant: &HashSet<String>) {
        let hit = |rank: usize| relevant.contains(&ranked[rank]);

        let gained: f64 = (0..ranked.len().min(NDCG_AT))
            .filter(|rank| hit(*rank))
            .map(|rank| 1.0 / ((rank + 2) as f64).log2())
            .sum();
        let ideal: f64 = (0..relevant.len().min(NDCG_AT))
            .map(|rank| 1.0 / ((rank + 2) as f64).log2())
            .sum();

        let found = (0..ranked.len().min(RECALL_AT))
            .filter(|rank| hit(*rank))
            .count();
        let deep = (NDCG_AT..ranked.len().min(RECALL_AT))
            .filter(|rank| hit(*rank))
            .count();

        self.queries += 1;
        self.ndcg += if ideal == 0.0 { 0.0 } else { gained / ideal };
        self.recall += found as f64 / relevant.len() as f64;
        self.deep += deep;
        self.with_work += usize::from(deep > 0);
    }

    fn mean_ndcg(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.ndcg / self.queries as f64
    }

    fn mean_recall(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.recall / self.queries as f64
    }
}

fn report(title: &str, scores: &Scores, per_query_ms: f64) {
    println!("\n  {title}");
    println!(
        "  queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}   below rank {NDCG_AT}   queries with any"
    );
    println!("  -----------------------------------------------------------------");
    println!(
        "  {:>7}   {:>7.4}   {:>9.4}   {:>13}   {:>16}",
        scores.queries,
        scores.mean_ndcg(),
        scores.mean_recall(),
        scores.deep,
        scores.with_work,
    );
    println!("  {per_query_ms:.0} ms per query\n");
}

/// Asserts the floors, unless this run is disqualified from carrying them.
fn assert_floors(named: &str, corpus: &Corpus, scores: &Scores, floors: (f64, f64)) {
    if named != DEFAULT_PROFILE {
        println!("  {named} is not the default profile, so the floors are not asserted\n");
        return;
    }
    if corpus.capped {
        println!("  the corpus was capped, so the floors are not asserted\n");
        return;
    }

    let (ndcg, recall) = floors;
    assert!(
        scores.mean_ndcg() >= ndcg,
        "{GROUP} nDCG@{NDCG_AT} fell to {:.4}, below the {ndcg:.4} floor",
        scores.mean_ndcg()
    );
    assert!(
        scores.mean_recall() >= recall,
        "{GROUP} recall@{RECALL_AT} fell to {:.4}, below the {recall:.4} floor",
        scores.mean_recall()
    );
}

fn profile() -> (String, Profile) {
    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).expect("a known profile");
    (named, profile)
}

// ---------------------------------------------------------------------------
// The embedding space alone
// ---------------------------------------------------------------------------

/// What the model reaches on its own, with no index and no fusion.
///
/// nDCG@10 0.0 / recall@50 0.0 until a run sets them: a floor invented rather
/// than measured is a floor that says nothing, and this is the one figure in
/// this file that had never been taken at all -- the published MIRACL rows are
/// all the whole search path, so "how much of the gap to BGE-M3's own
/// published 0.787 is fusion's" has been an inference and not a measurement.
const MODEL_FLOORS: (f64, f64) = (0.0, 0.0);

#[test]
#[ignore = "downloads a corpus and embeds a hundred and thirty-two thousand passages"]
fn the_model_ranks_real_passages() {
    let corpus = Corpus::load();
    let (named, profile) = profile();
    corpus.describe(&named);

    let home = eval_home();
    let mut embedder =
        Embedder::load(profile, &home.join("models")).expect("load the embedding model");

    let width = profile.dimensions() as usize;
    let vectors = embed_corpus(&mut embedder, &corpus, &home, width);

    let mut scores = Scores::default();
    let started = std::time::Instant::now();
    for query in &corpus.queries {
        let vector = unit(embedder.embed_query(&query.text).expect("embed the query"));

        let mut scored: Vec<(f32, usize)> = (0..corpus.passages.len())
            .map(|document| {
                let stored = &vectors[document * width..(document + 1) * width];
                let similarity = vector.iter().zip(stored).map(|(a, b)| a * b).sum();
                (similarity, document)
            })
            .collect();
        scored.select_nth_unstable_by(DEPTH, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(DEPTH);
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let ranked: Vec<String> = scored
            .into_iter()
            .map(|(_, document)| corpus.passages[document].docid.clone())
            .collect();
        scores.add(&ranked, &query.relevant);
    }

    report(
        &format!("the embedding space alone, {named}"),
        &scores,
        started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len() as f64,
    );
    assert_floors(&named, &corpus, &scores, MODEL_FLOORS);
}

/// Embeds every passage, reusing and resuming a cache on disk.
///
/// Resumable rather than merely cached: on the default profile this is hours
/// of forward passes, and a machine that loses the process halfway would
/// otherwise start again from nothing. Vectors are appended in batches, so the
/// length of the file is how many are done.
fn embed_corpus(embedder: &mut Embedder, corpus: &Corpus, home: &Path, width: usize) -> Vec<f32> {
    /// How many passages go into one call.
    ///
    /// A call and not a forward pass on the default profile: the joint export
    /// runs its texts one at a time, because a batched vector there depends on
    /// which texts it shared a batch with. This still bounds how much is lost
    /// when the process dies.
    const BATCH: usize = 32;

    let path = home.join(format!(
        "miracl-sw-{}-{}.f32",
        embedder.profile().model_id().replace('/', "-"),
        corpus.fingerprint()
    ));
    std::fs::create_dir_all(home).expect("the evaluation directory");

    let bytes = std::fs::metadata(&path)
        .map(|file| file.len() as usize)
        .unwrap_or(0);
    let mut done = bytes / 4 / width;
    if done > corpus.passages.len() {
        // The name carries the model and the corpus, so a mismatch cannot
        // happen -- but a truncated write can leave half a vector, and half a
        // vector is worse than none.
        done = 0;
    }
    let mut vectors: Vec<f32> = std::fs::read(&path)
        .map(|raw| {
            raw[..done * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect()
        })
        .unwrap_or_default();
    if done > 0 {
        println!("  reusing {done} embedded passages");
    }

    let resumed = done;
    let started = std::time::Instant::now();
    while done < corpus.passages.len() {
        let batch: Vec<&str> = corpus.passages[done..]
            .iter()
            .take(BATCH)
            .map(|passage| passage.text.as_str())
            .collect();
        let embedded = embedder.embed_passages(&batch).expect("embed the corpus");

        let mut appended = Vec::with_capacity(batch.len() * width * 4);
        for vector in embedded {
            let vector = unit(vector);
            for value in &vector {
                appended.extend_from_slice(&value.to_le_bytes());
            }
            vectors.extend_from_slice(&vector);
        }
        append(&path, &appended);

        done += batch.len();
        if done.is_multiple_of(BATCH * 50) {
            let rate = (done - resumed) as f64 / started.elapsed().as_secs_f64();
            println!(
                "  embedded {done}/{} at {rate:.1}/s, {:.0} minutes left",
                corpus.passages.len(),
                (corpus.passages.len() - done) as f64 / rate / 60.0,
            );
        }
    }

    vectors
}

fn append(path: &Path, bytes: &[u8]) {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|error| panic!("opening {}: {error}", path.display()));
    file.write_all(bytes).expect("write the vector cache");
}

/// Scales a vector to unit length, so a dot product is a cosine.
fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

// ---------------------------------------------------------------------------
// The whole search path
// ---------------------------------------------------------------------------

/// The floors for the search path the product calls.
///
/// Zero until a run sets them, for the same reason as [`MODEL_FLOORS`]. What
/// goes here has to come from this harness: `docs/measured.md` reports 0.7359
/// for the `fast` tier and 0.7158 for `off`, but those came from a program
/// that was never committed, so they are a target to reproduce rather than a
/// floor to inherit.
const SEARCH_FLOORS: (f64, f64) = (0.0, 0.0);

/// What the reranker has to be worth here to keep earning its 226 ms.
///
/// Zero until a run sets it. The published gain on this corpus is +0.0201 for
/// `fast`, half what it is worth on parallel sentences, and the ADR's reading
/// is that within one language the candidates no lexical channel found are a
/// fraction of the shortlist rather than nearly all of it.
const RERANK_IS_WORTH: f64 = 0.0;

/// What a search's shortlist is worth before and after fusion, and by tier.
#[test]
#[ignore = "provisions PostgreSQL, downloads a corpus, and indexes it"]
fn search_ranks_real_passages() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a runtime");
    runtime.block_on(search());
}

async fn search() {
    let corpus = Corpus::load();
    let (named, profile) = profile();
    corpus.describe(&named);

    let home = eval_home();
    let temporary = (std::env::var("PAMIN_EVAL_HOME").is_err())
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match &temporary {
        Some(dir) => Workspace::at(dir.path()),
        None => Workspace::at(&home),
    };

    // The profile is part of the workspace identity: an index records the
    // profile it was built with and refuses to open under another.
    let project = format!("miracl-sw-{named}-{}", corpus.fingerprint());
    let engine = Engine::open(&workspace, &project, profile, Access::ReadWrite)
        .await
        .expect("open the engine");

    write_corpus(&engine, &corpus).await;

    // Every arm asserts its own premise. A vector channel over an index with
    // no graph is a full scan of a different object, and a count that is short
    // means the corpus is not all there -- either one makes the rest of this
    // file a measurement of something nobody ships.
    assert_eq!(
        engine.indexed_documents().expect("count the documents") as usize,
        corpus.passages.len(),
        "the index does not hold the whole corpus"
    );
    let completeness = engine
        .vector_index_completeness()
        .expect("the vector index's completeness");
    println!("  the vector graph covers {completeness:.4} of the corpus");
    if corpus.capped {
        // Below the engine's unindexed-document budget the policy deliberately
        // does not build a graph, because a scan of that many documents is the
        // faster answer -- so a capped run measures a full scan and is honest
        // about it, which is the other reason its numbers are not comparable.
        println!("  a capped corpus is below the budget that builds one, so this is a scan\n");
    } else {
        assert!(
            completeness > 0.99,
            "the vector graph covers {completeness:.4} of the corpus, so this would \
             measure a full scan rather than the index the product serves"
        );
    }

    if let Some(settings) = sweep() {
        println!("\n  setting                nDCG@{NDCG_AT}   recall@{RECALL_AT}");
        println!("  ------------------------------------------------");
        for (label, fusion) in settings {
            let scores = run(&engine, &corpus, Route::Fused(fusion)).await;
            println!(
                "  {label:<20}   {:>7.4}   {:>9.4}",
                scores.mean_ndcg(),
                scores.mean_recall()
            );
        }
        println!();
        return;
    }

    if std::env::var("TIERS").is_ok() {
        for tier in [Rerank::Off, Rerank::Fast, Rerank::Accurate] {
            let started = std::time::Instant::now();
            let scores = run(&engine, &corpus, Route::Shipped(tier)).await;
            report(
                &format!("{tier:?}, {named}"),
                &scores,
                started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len() as f64,
            );
        }
        return;
    }

    let started = std::time::Instant::now();
    let shipped = run(&engine, &corpus, Route::Shipped(Rerank::default())).await;
    report(
        &format!("the shipped search path, {named}"),
        &shipped,
        started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len() as f64,
    );
    assert_floors(&named, &corpus, &shipped, SEARCH_FLOORS);

    // Fusion alone, to price the reranker on this corpus rather than assume
    // the other corpus's answer carries over. It does not: reranking touches
    // only what the lexical channels missed, and how much of the shortlist
    // that is differs by corpus more than by tier.
    let started = std::time::Instant::now();
    let fused = run(&engine, &corpus, Route::Fused(Fusion::default())).await;
    report(
        &format!("fusion alone, no reranking, {named}"),
        &fused,
        started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len() as f64,
    );
    if named == DEFAULT_PROFILE && !corpus.capped {
        let gain = shipped.mean_ndcg() - fused.mean_ndcg();
        println!("  reranking is worth {gain:+.4} nDCG@{NDCG_AT} here\n");
        assert!(
            gain >= RERANK_IS_WORTH,
            "reranking is worth {gain:.4} nDCG@{NDCG_AT}, under the {RERANK_IS_WORTH:.4} \
             it is supposed to be worth"
        );
    }
}

/// The fusion settings to try when `SWEEP` is set, each labelled as printed.
///
/// The same grid the other two harnesses sweep, so the three are readable
/// against each other, and for the same reason: a setting that suits one
/// corpus and ruins another is the outcome worth catching. This is the corpus
/// whose verdict should carry the most weight -- one language, real questions,
/// nothing parallel -- and until this harness existed it had no vote.
///
/// `SWEEP=1` runs every row; any other value keeps the rows whose label
/// contains it.
fn sweep() -> Option<Vec<(String, Fusion)>> {
    let wanted = std::env::var("SWEEP").ok()?;
    let filter = (wanted != "1").then_some(wanted);
    let mut settings = Vec::new();
    for k in [5.0, 10.0, 20.0, 60.0] {
        for weight in [0.0, 0.125, 0.25, 0.5, 1.0] {
            settings.push((
                format!("k={k:.0} lex {weight:.3}"),
                Fusion::default()
                    .with_k(k)
                    .with_weight(Channel::LexicalSegmented, weight)
                    .with_weight(Channel::LexicalNgram, weight),
            ));
        }
    }
    // `0.50-0.50` is a constant eighth of a weight wearing the rule's clothes,
    // and it is here so that a gain from the rule cannot be mistaken for a
    // gain from simply asking the lexical pair for less.
    for (floor, ceiling) in [(0.0, 1.0), (0.25, 1.0), (0.5, 1.0), (0.5, 0.5)] {
        settings.push((
            format!("k=10 adapt {floor:.2}-{ceiling:.2}"),
            Fusion::default().with_k(10.0).with_adaptive(floor, ceiling),
        ));
    }
    if let Some(filter) = &filter {
        settings.retain(|(label, _)| label.contains(filter.as_str()));
        assert!(!settings.is_empty(), "SWEEP={filter:?} matched no row");
    }
    Some(settings)
}

/// Which of the engine's search paths an arm runs.
enum Route {
    /// The product's own entry point, reranker and all.
    Shipped(Rerank),
    /// Fusion alone, at a weighting the caller chooses.
    Fused(Fusion),
}

async fn run(engine: &Engine, corpus: &Corpus, route: Route) -> Scores {
    let mut scores = Scores::default();
    for query in &corpus.queries {
        let hits = match &route {
            Route::Shipped(rerank) => {
                engine
                    .search_reranked(&query.text, DEPTH as u32, DEPTHS, *rerank)
                    .await
            }
            Route::Fused(fusion) => {
                engine
                    .search_fused(&query.text, DEPTH as u32, DEPTHS, fusion.clone())
                    .await
            }
        }
        .expect("search");
        let ranked: Vec<String> = hits.into_iter().map(|hit| hit.topic).collect();
        scores.add(&ranked, &query.relevant);
    }
    scores
}

/// Writes every passage that is not already a topic, then runs the queue.
async fn write_corpus(engine: &Engine, corpus: &Corpus) {
    let project = engine.project;
    let mut written = 0usize;

    for passage in &corpus.passages {
        let existing =
            pamin_store::repository::find_topic(engine.database.pool(), project, &passage.docid)
                .await
                .expect("look for the topic");
        if existing.is_some() {
            continue;
        }
        written += 1;
        engine
            .write(&Write {
                topic: &passage.docid,
                content: &passage.text,
                content_hash: &passage.text.len().to_string(),
                verdict: pamin_core::FilterDecision::Promoted,
                reason: "monolingual evaluation corpus",
                promoted: true,
                // The dataset's own language, as the other harness does it:
                // `pamin write` takes this from `detect_language`, and nothing
                // reads the column today.
                language: Some("sw"),
                language_confidence: None,
                observed_at: time::OffsetDateTime::now_utc(),
                validity: pamin_core::Validity::ALWAYS,
            })
            .await
            .unwrap_or_else(|error| panic!("writing {}: {error}", passage.docid));

        if written.is_multiple_of(5_000) {
            println!("  wrote {written} passages");
        }
    }

    if written > 0 {
        println!("  wrote {written} of {} passages", corpus.passages.len());
    }
    let started = std::time::Instant::now();
    let drained = engine
        .drain_cascade(pamin_engine::Owed::Everything)
        .await
        .expect("drain the cascade");
    assert_eq!(
        drained.pending, 0,
        "the corpus is not fully indexed: {} jobs still owed",
        drained.pending
    );
    if written > 0 {
        println!(
            "  ran {} cascade jobs in {:.0}s",
            drained.completed,
            started.elapsed().as_secs_f64()
        );
    }
}
