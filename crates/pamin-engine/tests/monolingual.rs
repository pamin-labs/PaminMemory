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
//! | `PASSAGES` | a second project whose vectors embed the topic name, paired against this one |
//! | `ROUTES` | a cascade and gates that spend less on the reranker, against the shipped pass |
//! | `CONTEXT` | price showing the reranker each candidate's name, from one run |
//! | `RERANK_RULES` | price blending the shipped tier's scores with fusion's, from one run |
//! | `SWEEP` | run fusion settings instead of the shipped path |
//!
//! The dataset is not vendored. The corpus is Wikipedia text under
//! CC-BY-SA-3.0 and this repository is Apache-2.0, and it is 40 MB unpacked.

mod channels;
mod memory;
mod reranking;
mod scoring;
mod statistics;

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

use scoring::{NDCG_AT, RECALL_AT, Scores};

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

/// What a topic's name is prefixed with, so it cannot occur in the corpus.
///
/// See [`Passage::docid`]: without it, 1,186 edges were derived from digits.
const KEY: &str = "miracl-sw:";

/// One passage, as indexed.
struct Passage {
    /// MIRACL's own document id, `article#paragraph`, behind a prefix.
    ///
    /// The prefix is the whole point and it was learned the hard way. Mention
    /// derivation looks for topic names inside content, so a key that can
    /// occur in prose measures the graph channel on relationships nobody
    /// asserted -- which is what the cross-lingual harness's own comment warns
    /// about, and which this copied and then ignored. A bare docid is
    /// `2#0`, whose tokens are `2` and `0`, and across 131,924 Wikipedia
    /// passages that matched: **1,186 edges, every one of them pointing at a
    /// short docid** like `2#3` or `30#5`, manufactured out of digits.
    ///
    /// `miracl-sw:2#0` tokenizes to a four-token run that no prose contains.
    /// The search arm asserts the result rather than trusting it: it fails if
    /// the graph channel credits any hit.
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
            let docid = format!("{KEY}{}", document["docid"].as_str().expect("docid"));
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
            let docid = format!("{KEY}{docid}");
            if grade.trim().parse::<i32>().unwrap_or(0) <= 0 || !indexed.contains(docid.as_str()) {
                continue;
            }
            judged.entry(qid.to_string()).or_default().insert(docid);
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

/// What the reranker did during a run, printed beside the scores.
///
/// A row of numbers nothing in this project had: how many candidates reached
/// the model against how many were offered it, how long they were, and how
/// often the score cache answered instead. Each of the three gates a decision
/// recorded as deferred -- see [`pamin_index::Reranked`] -- and each was an
/// inference from what the corpus is until this printed it.
///
/// This corpus is the one where the lengths matter most. `MAX_TOKENS` is 256
/// and these are Wikipedia passages rather than the sentences the other
/// harness holds, so it is here that truncation should be a lever rather than
/// a rounding error -- which was written down as an expectation long before
/// anything counted a character.
fn report_reranking(engine: &Engine, tier: Rerank, queries: usize) {
    let Some(counted) = engine.reranked(tier) else {
        println!("  the {} tier was never loaded\n", tier.name());
        return;
    };
    if counted.offered == 0 {
        println!("  the {} tier scored nothing\n", tier.name());
        return;
    }
    println!(
        "  {}: {:.1} candidates a query reached the model of {:.1} offered, \
         {:.0} characters each, longest {}, cache {:.1}% of {} lookups",
        tier.name(),
        counted.scored as f64 / queries as f64,
        counted.offered as f64 / queries as f64,
        counted.characters as f64 / counted.scored.max(1) as f64,
        counted.longest,
        100.0 * (counted.offered - counted.scored) as f64 / counted.offered as f64,
        counted.offered,
    );
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
        scores.add(&ranked, query.relevant.len(), |topic| {
            query.relevant.contains(topic)
        });
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
    // Not the CLI, so nothing has done this for us, and this corpus is the
    // one that proved it matters: 131,924 passages is more segment files than
    // the 1,024 descriptors a process starts with, and the first full run of
    // this harness died 65 minutes in with RocksDB unable to append.
    let (before, after) = pamin_index::raise_open_file_limit().expect("the open-file limit");
    println!("  open files: {before} raised to {after}");

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

    // `MEMORY`: where the resident memory goes, stage by stage. First, so no
    // other arm has loaded anything yet. See `memory`.
    if std::env::var("MEMORY").is_ok() {
        attribute_memory(&engine, &corpus).await;
        return;
    }

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

    // What makes skipping the graph jobs sound, asserted rather than argued.
    // If a topic name ever did appear in this corpus's prose, the graph channel
    // would credit a hit and this would fail -- which is the signal to drain
    // the whole queue rather than the part a search needs.
    let probe = engine
        .search_fused(
            &corpus.queries[0].text,
            DEPTH as u32,
            DEPTHS,
            Fusion::default(),
        )
        .await
        .expect("a probe search");
    let credited_graph = probe.iter().flat_map(|hit| &hit.result.why).any(|why| {
        matches!(
            why,
            pamin_core::Why::Channel {
                channel: Channel::Graph,
                ..
            }
        )
    });
    assert!(
        !credited_graph,
        "the graph channel credited a hit, so the graph jobs this run left owed \
         would have changed these numbers"
    );

    // `CHANNELS` reports what each channel is worth on its own, and what the
    // fused list looks like with each one taken away. One run, not four: the
    // trace carries every channel's rank for every candidate, so the whole
    // matrix comes out of a single pass. See `channels`.
    if std::env::var("CHANNELS").is_ok() {
        report_channels(&engine, &corpus, &named).await;
        return;
    }

    if let Some(settings) = sweep() {
        println!("\n  setting                nDCG@{NDCG_AT}   recall@{RECALL_AT}");
        println!("  ------------------------------------------------");
        let mut measured: Vec<(String, Scores)> = Vec::new();
        for (label, fusion) in settings {
            let scores = run(&engine, &corpus, Route::Fused(fusion)).await;
            println!(
                "  {label:<20}   {:>7.4}   {:>9.4}",
                scores.mean_ndcg(),
                scores.mean_recall()
            );
            measured.push((label, scores));
        }

        // A sweep that prints only means says which row is highest and not
        // whether it is distinguishable from the others -- and the rows here
        // are separated by thousandths. The default lexical weight was moved on
        // one such gap, so every row is now also priced against the best one,
        // query by query. A winner that beats nothing significantly is a winner
        // by luck of which queries the corpus happens to contain.
        if let Some((best, top)) = statistics::baseline(&measured, |scores| scores.mean_ndcg()) {
            println!("\n  against {best}:");
            for (label, scores) in &measured {
                if label == best {
                    continue;
                }
                println!(
                    "  {label:<20}   {}",
                    statistics::compare(&top.per_query, &scores.per_query)
                );
            }
        }
        println!();
        return;
    }

    // `PASSAGES`: the same memories in a second project whose vectors embed
    // the topic's name, asked every question alongside this one. See
    // `channels::Paired`.
    if std::env::var("PASSAGES").is_ok() {
        let other = Engine::open(
            &workspace,
            &format!("{project}-named"),
            profile,
            Access::ReadWrite,
        )
        .await
        .expect("open the named project");
        write_corpus(&other, &corpus).await;
        assert_eq!(
            engine.passage(),
            pamin_index::Passage::Content,
            "the baseline project was built from content"
        );
        assert_eq!(
            other.passage(),
            pamin_index::Passage::Named,
            "the new project embeds names"
        );
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        let mut paired = channels::Paired::default();
        for query in &corpus.queries {
            let before = engine
                .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            let after = other
                .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            paired.observe(GROUP, &before, &after, |into, ranking| {
                into.add(ranking, query.relevant.len(), |topic| {
                    query.relevant.contains(topic)
                });
            });
        }
        paired.report(&format!(
            "vectors embedding the topic name, MIRACL, {named}"
        ));
        return;
    }

    // `RERANK_RULES`: other rules for using the shipped tier's scores, priced
    // from one shipped run. See `reranking`.
    if std::env::var("RERANK_RULES").is_ok() {
        rerank_rules(&engine, &corpus, &named).await;
        return;
    }

    // `ROUTES`: spending less on the reranker -- a cascade, and gates that
    // skip it -- priced against the shipped pass. See `reranking::Routes`.
    if std::env::var("ROUTES").is_ok() {
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        let tier = Rerank::default();
        let mut small = pamin_index::Reranker::load(Rerank::Fast, &workspace.root().join("models"))
            .expect("load the small reranker");
        let mut routes = reranking::Routes::default();
        for query in &corpus.queries {
            let hits = engine
                .search_reranked(&query.text, WIDE, DEPTHS, tier)
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits, tier);
            routes.observe(
                GROUP,
                &query.text,
                &hits,
                &replayed,
                &mut small,
                &query.text,
                |into, ranking| {
                    into.add(ranking, query.relevant.len(), |topic| {
                        query.relevant.contains(topic)
                    });
                },
            );
        }
        routes.report(&format!(
            "spending less on the {} reranker, MIRACL, {named}",
            tier.name()
        ));
        return;
    }

    // `CONTEXT`: the shipped tier shown each candidate's name and seed. See
    // `reranking::in_context`.
    if std::env::var("CONTEXT").is_ok() {
        context(&engine, &workspace, &corpus, &named).await;
        return;
    }

    if std::env::var("TIERS").is_ok() {
        let mut priced: Vec<(Rerank, Scores)> = Vec::new();
        for tier in [Rerank::Off, Rerank::Fast, Rerank::Accurate] {
            let started = std::time::Instant::now();
            let scores = run(&engine, &corpus, Route::Shipped(tier)).await;
            report(
                &format!("{tier:?}, {named}"),
                &scores,
                started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len() as f64,
            );
            report_reranking(&engine, tier, corpus.queries.len());
            priced.push((tier, scores));
        }

        // Paired, against `off` and against the tier that ships. The means
        // above cannot say whether one tier beats another: that needs the two
        // compared query by query, and this corpus is the one where it matters
        // most -- real questions, judged by people, nothing parallel -- and
        // where reranking at the default tier has been a net loss.
        for against in [Rerank::Off, Rerank::default()] {
            let Some((_, base)) = priced.iter().find(|(tier, _)| *tier == against) else {
                continue;
            };
            println!("\n  every tier against {}, {named}", against.name());
            for (tier, scores) in &priced {
                if *tier == against {
                    continue;
                }
                println!(
                    "  {:<10}   {:>7.4}   {}",
                    tier.name(),
                    scores.mean_ndcg(),
                    statistics::compare(&base.per_query, &scores.per_query)
                );
            }
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
    report_reranking(&engine, Rerank::default(), corpus.queries.len());
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
    // Query by query, not mean against mean. This corpus is where the
    // reranker was measured *losing* 0.0152, and a loss that size is exactly
    // what a paired count can dissolve or confirm -- until now nothing here
    // could tell which. Printed on every profile, because the number is worth
    // having even when the floors are not asserted; see `statistics`.
    let paired = statistics::compare(&fused.per_query, &shipped.per_query);
    println!("  reranking is worth {paired} nDCG@{NDCG_AT} here\n");

    if named == DEFAULT_PROFILE && !corpus.capped {
        let gain = shipped.mean_ndcg() - fused.mean_ndcg();
        assert!(
            gain >= RERANK_IS_WORTH,
            "reranking is worth {gain:.4} nDCG@{NDCG_AT}, under the {RERANK_IS_WORTH:.4} \
             it is supposed to be worth ({paired})"
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
        scores.add(&ranked, query.relevant.len(), |topic| {
            query.relevant.contains(topic)
        });
    }
    scores
}

/// Every rule in `reranking::rules` over the shipped tier's own scores.
async fn rerank_rules(engine: &Engine, corpus: &Corpus, named: &str) {
    use std::collections::BTreeMap;

    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let rules = reranking::rules();
    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); rules.len()];
    for query in &corpus.queries {
        let hits = engine
            .search_reranked(&query.text, WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);
        for ((_, rule), into) in rules.iter().zip(&mut measured) {
            into.entry(GROUP.to_string()).or_default().add(
                &replayed.order(*rule),
                query.relevant.len(),
                |topic| query.relevant.contains(topic),
            );
        }
    }
    reranking::report(
        &format!(
            "rules for the {} reranker's scores, MIRACL, {named}",
            tier.name()
        ),
        &rules,
        reranking::shipped(&rules),
        &measured,
    );
}

/// What the shipped tier is worth when it is shown each candidate's topic
/// name, and the memory the graph reached it from. See `reranking::in_context`.
///
/// This corpus's names are MIRACL's own document ids behind a prefix, so the
/// named renderings ask what a name that carries nothing costs; it has no
/// edges, so the seeded renderings equal their unseeded counterparts.
async fn context(engine: &Engine, workspace: &Workspace, corpus: &Corpus, named: &str) {
    use std::collections::BTreeMap;

    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let mut model = pamin_index::Reranker::load(tier, &workspace.root().join("models"))
        .expect("load the reranker");
    let labels = reranking::context_labels();
    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); labels.len()];
    for query in &corpus.queries {
        let hits = engine
            .search_reranked(&query.text, WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);
        let (orders, _) = reranking::in_context(&hits, &replayed, &mut model, &query.text);
        for (order, into) in orders.iter().zip(&mut measured) {
            into.entry(GROUP.to_string())
                .or_default()
                .add(order, query.relevant.len(), |topic| {
                    query.relevant.contains(topic)
                });
        }
    }
    reranking::report(
        &format!(
            "what the {} reranker is shown, MIRACL, {named}",
            tier.name()
        ),
        &labels,
        reranking::shipped_context(),
        &measured,
    );
}

/// The channel diagnostic: each one alone, each one removed, and how far the
/// two lexical channels agree with each other.
///
/// `WIDE` rather than `DEPTH` because the point is an untruncated trace -- see
/// [`channels`] for what a truncated one silently does to these numbers.
async fn report_channels(engine: &Engine, corpus: &Corpus, named: &str) {
    use pamin_core::Channel;
    use std::collections::BTreeMap;

    /// Past four times the channel depth, so `take(limit)` cannot bite.
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;

    /// Every channel, so leaving one out is asked of all four.
    const CHANNELS: &[Channel] = &[
        Channel::LexicalSegmented,
        Channel::LexicalNgram,
        Channel::Vector,
        Channel::Graph,
    ];

    let mut alone: BTreeMap<Channel, Scores> = BTreeMap::new();
    let mut without: BTreeMap<Channel, Scores> = BTreeMap::new();
    let mut whole = Scores::default();
    let mut lexical_agreement: Vec<f64> = Vec::new();

    // Every fusion setting worth pricing, scored from the same traces as the
    // rows above. One pass, the whole grid.
    let variants = channels::variants();
    let mut offline: Vec<Scores> = variants.iter().map(|_| Scores::default()).collect();

    for query in &corpus.queries {
        let hits = engine
            .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
            .await
            .expect("search");

        // Both premises, on every query rather than once: a trace that was
        // truncated, or arithmetic that has drifted from the engine's, makes
        // every number below wrong in a way that looks like a finding.
        channels::enough_room(&hits, WIDE);
        channels::same_as_the_engine(&hits, &Fusion::default());

        let each = channels::each_alone(&hits);
        for (channel, ranking) in &each {
            alone
                .entry(*channel)
                .or_default()
                .add(ranking, query.relevant.len(), |topic| {
                    query.relevant.contains(topic)
                });
        }

        for missing in CHANNELS {
            let ranking = channels::as_if(&hits, &Fusion::default().without(*missing));
            without
                .entry(*missing)
                .or_default()
                .add(&ranking, query.relevant.len(), |topic| {
                    query.relevant.contains(topic)
                });
        }

        whole.add(
            &hits
                .iter()
                .map(|hit| hit.topic.clone())
                .collect::<Vec<String>>(),
            query.relevant.len(),
            |topic| query.relevant.contains(topic),
        );

        for ((_, fusion), into) in variants.iter().zip(&mut offline) {
            into.add(
                &channels::as_if(&hits, fusion),
                query.relevant.len(),
                |topic| query.relevant.contains(topic),
            );
        }

        if let (Some(segmented), Some(ngram)) = (
            each.get(&Channel::LexicalSegmented),
            each.get(&Channel::LexicalNgram),
        ) && let Some(tau) = channels::agreement(segmented, ngram)
        {
            lexical_agreement.push(tau);
        }
    }

    println!("\n  each channel on its own, {named}");
    println!("  channel               candidates   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
    println!("  ------------------------------------------------------------");
    for (channel, scores) in &alone {
        println!(
            "  {:<20}   {:>10}   {:>7.4}   {:>9.4}",
            format!("{channel:?}"),
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall()
        );
    }

    println!(
        "\n  all four fused: {:.4} nDCG@{NDCG_AT}",
        whole.mean_ndcg()
    );
    println!("\n  with one channel taken away, against all four:");
    for (channel, scores) in &without {
        println!(
            "  {:<20}   {}",
            format!("{channel:?}"),
            statistics::compare(&whole.per_query, &scores.per_query)
        );
    }

    channels::sweep_table(
        named,
        &whole,
        &variants,
        &offline.iter().map(Some).collect::<Vec<_>>(),
    );
    // One group, so the rule's mean over groups is this group's mean.
    let keyed = |scores: &Scores| BTreeMap::from([(GROUP.to_string(), scores.clone())]);
    channels::cross_validated(
        &format!("MIRACL, {named}"),
        &[GROUP],
        &keyed(&whole),
        &variants,
        channels::shipped_row(&variants),
        &offline.iter().map(keyed).collect::<Vec<_>>(),
    );

    println!(
        "\n  the two lexical channels agree at Kendall tau {:.4} over {} queries\n",
        channels::mean(&lexical_agreement),
        lexical_agreement.len()
    );
}

/// Writes every passage that is not already a topic, then runs the queue.
/// Resident memory after each thing a server loads, on the largest corpus
/// here, and what the allocator is holding that nothing uses.
async fn attribute_memory(engine: &Engine, corpus: &Corpus) {
    assert_eq!(
        engine.indexed_documents().expect("count the documents") as usize,
        corpus.passages.len(),
        "the index does not hold the whole corpus, so this would measure a smaller one"
    );
    const SEARCHES: usize = 100;
    println!(
        "\n  resident memory, stage by stage, {} passages",
        corpus.passages.len()
    );
    memory::Resident::now().print("the engine opened");
    let queries: Vec<&str> = corpus
        .queries
        .iter()
        .take(SEARCHES)
        .map(|query| query.text.as_str())
        .collect();
    for tier in [Rerank::Off, Rerank::Fast, Rerank::Accurate] {
        engine
            .search_reranked(queries[0], DEPTH as u32, DEPTHS, tier)
            .await
            .expect("search");
        memory::Resident::now().print(&format!("one search, {}", tier.name()));
        for query in &queries {
            engine
                .search_reranked(query, DEPTH as u32, DEPTHS, tier)
                .await
                .expect("search");
        }
        memory::Resident::now().print(&format!("{SEARCHES} searches, {}", tier.name()));
    }

    // What the allocator holds after it was freed: glibc keeps freed memory in
    // per-thread arenas, and the difference trimming makes is memory the
    // process holds and does not use.
    unsafe extern "C" {
        fn malloc_trim(pad: usize) -> i32;
    }
    // SAFETY: glibc's own function, no arguments that point anywhere.
    unsafe { malloc_trim(0) };
    memory::Resident::now().print("after malloc_trim(0)");
}

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
    // Drained until the index holds the whole corpus, rather than until the
    // queue is empty. Those are different, and on this corpus the difference
    // is hours.
    //
    // A promoted write owes three jobs and only one of them puts the memory in
    // the index; the other two derive graph edges from topic names. The engine
    // claims by priority, so indexing runs first -- and **on this corpus the
    // graph can contribute nothing at all**, because MIRACL's topic names are
    // its docids, `2#0` and the like, which do not appear in anyone's
    // Wikipedia prose. Two hundred and sixty thousand jobs to derive edges
    // from names no text contains is work whose result is provably empty here,
    // and waiting it out would cost about three hours to change nothing.
    //
    // That is an assertion rather than an argument: the arm below fails if the
    // graph channel ever credits a hit.
    let started = std::time::Instant::now();
    let mut completed = 0;
    loop {
        let drained = engine
            .drain_cascade(pamin_engine::Owed::Everything)
            .await
            .expect("drain the cascade");
        completed += drained.completed;

        let indexed = engine.indexed_documents().expect("count the documents") as usize;
        if indexed >= corpus.passages.len() {
            break;
        }
        assert!(
            drained.completed > 0 || drained.pending > 0,
            "the index holds {indexed} of {} passages and the queue is empty, \
             so nothing will ever finish it",
            corpus.passages.len()
        );
        if completed.is_multiple_of(20_000) {
            println!(
                "  indexed {indexed}/{} after {completed} jobs, {:.0} minutes",
                corpus.passages.len(),
                started.elapsed().as_secs_f64() / 60.0
            );
        }
    }

    let owed = pamin_store::jobs::pending(engine.database.pool(), project)
        .await
        .expect("what the queue still owes");
    println!(
        "  indexed the whole corpus with {completed} jobs in {:.0} minutes; \
         {owed} graph jobs left owed, which this corpus cannot use",
        started.elapsed().as_secs_f64() / 60.0
    );
}
