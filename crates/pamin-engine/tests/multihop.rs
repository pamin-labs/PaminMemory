//! Retrieval for questions that take more than one step: MuSiQue.
//!
//! **What this corpus is here to settle.** Every conclusion this repository has
//! drawn about the graph channel -- its weight, the corroboration rule, the
//! seed's relevance, showing the reranker the seed -- rests on the own corpus's
//! `relational` group: twenty queries, written here, by the people who wanted
//! to know whether the graph helps. The external corpora cannot check any of
//! it, because none of them has an edge. That is two problems at once: twenty
//! queries can only see a difference of about 0.08, and a question written by
//! the author of a mechanism is a question the mechanism was built to answer.
//!
//! MuSiQue is questions nobody here wrote, built so that no single paragraph
//! answers them: "who is the spouse of the Green performer" needs the album's
//! paragraph to find the performer and the performer's to find the spouse. Two
//! to four supporting paragraphs a question, among twenty. It is the benchmark
//! the graph-retrieval literature reports on (HippoRAG and its successors), so
//! a graph channel that is worth something should be worth something here.
//!
//! **How the graph gets its edges, which is also what it is testing.** Every
//! paragraph is a memory under its Wikipedia title, and nothing else is
//! supplied: the edges are whatever the write path's mention derivation finds
//! by looking for one title inside another paragraph's text. That is the
//! product's own mechanism, not an oracle. A bridge that is not itself a title
//! -- the performer, when no paragraph is titled after him -- is an edge this
//! engine cannot build, and the census below says how many it did.
//!
//! **Paragraphs that share a title are one memory.** A topic has one current
//! state, so two paragraphs under "Green" written separately would leave only
//! the second. They are joined instead, and relevance is judged by title. That
//! is coarser than the dataset's paragraph ids and is the grain the product
//! works at; it also means these figures are not comparable with published
//! paragraph-level recall.
//!
//! **Licence.** MuSiQue is CC BY 4.0 (Trivedi et al., TACL 2022). It is fetched
//! at run time from the HuggingFace datasets-server, not vendored.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p pamin-engine --test multihop -- --ignored --nocapture
//! ```
//!
//! | variable | what it does |
//! |---|---|
//! | `MUSIQUE_DIR` | where cached pages are, instead of fetching them |
//! | `PAMIN_PROFILE` | which embedding profile, default `accuracy` |
//! | `CHANNELS` | the channel diagnostic and the offline fusion sweep |
//! | `CONTEXT` | price what the reranker is shown, from one run |

mod channels;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_core::Fusion;
use pamin_engine::{Depths, Engine, Write};
use pamin_index::{Access, Profile, Rerank};
use pamin_store::Workspace;

use scoring::{NDCG_AT, RECALL_AT, Scores};

/// Where the rows come from: the answerable half of the dev split.
const ROWS: &str = "https://datasets-server.huggingface.co/rows\
                    ?dataset=bdsaglam%2Fmusique&config=answerable&split=validation";

/// The server's own maximum.
const PER_PAGE: usize = 100;

/// A thousand questions, which is the sample the graph-retrieval literature
/// reports on. Taken as the first thousand rows, so two runs read the same ones.
const QUESTIONS: usize = 1_000;

/// How deep to retrieve, so recall@50 can be scored.
const DEPTH: usize = RECALL_AT + 1;

const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

/// Past four times the channel depth, so a trace is never truncated.
const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;

const DEFAULT_PROFILE: &str = "accuracy";

/// A paragraph longer than this is cut, on a character boundary, at about what
/// the embedders' window holds; joined same-title paragraphs can run long.
const MAX_CHARS: usize = 2_000;

#[derive(serde::Deserialize)]
struct Row {
    id: String,
    question: String,
    paragraphs: Vec<Paragraph>,
}

#[derive(serde::Deserialize)]
struct Paragraph {
    title: String,
    paragraph_text: String,
    is_supporting: bool,
}

/// One memory: a title and every distinct paragraph under it.
struct Memory {
    title: String,
    text: String,
}

struct Query {
    text: String,
    /// `2hop`, `3hop` or `4hop`, read from the question's id.
    group: String,
    /// Titles of the supporting paragraphs.
    relevant: HashSet<String>,
}

struct Corpus {
    memories: Vec<Memory>,
    queries: Vec<Query>,
}

impl Corpus {
    fn load() -> Self {
        let dir = dataset_dir();
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));

        let mut rows: Vec<Row> = Vec::new();
        for offset in (0..QUESTIONS).step_by(PER_PAGE) {
            let path = dir.join(format!("page-{offset:05}.json"));
            download(&path, &format!("{ROWS}&offset={offset}&length={PER_PAGE}"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            let body: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));
            for row in body["rows"]
                .as_array()
                .unwrap_or_else(|| panic!("{} has no rows: {text:.200}", path.display()))
            {
                rows.push(
                    serde_json::from_value(row["row"].clone())
                        .unwrap_or_else(|error| panic!("a row of {}: {error}", path.display())),
                );
            }
        }
        rows.truncate(QUESTIONS);

        // Distinct paragraphs per title, in first-seen order so the joined
        // text is the same on every run.
        let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut queries = Vec::new();
        for row in &rows {
            for paragraph in &row.paragraphs {
                let seen = texts.entry(paragraph.title.clone()).or_default();
                if !seen.contains(&paragraph.paragraph_text) {
                    seen.push(paragraph.paragraph_text.clone());
                }
            }
            let hops = row.id.split("hop").next().unwrap_or("?");
            queries.push(Query {
                text: row.question.clone(),
                group: format!("{hops}hop"),
                relevant: row
                    .paragraphs
                    .iter()
                    .filter(|paragraph| paragraph.is_supporting)
                    .map(|paragraph| paragraph.title.clone())
                    .collect(),
            });
        }

        let memories = texts
            .into_iter()
            .map(|(title, paragraphs)| Memory {
                title,
                text: paragraphs.join("\n").chars().take(MAX_CHARS).collect(),
            })
            .collect();

        Self { memories, queries }
    }

    fn fingerprint(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for memory in &self.memories {
            for byte in memory.title.bytes().chain(memory.text.bytes()) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{hash:016x}")
    }

    fn describe(&self) {
        let mut by_group: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for query in &self.queries {
            let entry = by_group.entry(query.group.as_str()).or_default();
            entry.0 += 1;
            entry.1 += query.relevant.len();
        }
        println!(
            "  {} memories under distinct titles, {} questions",
            self.memories.len(),
            self.queries.len()
        );
        for (group, (queries, relevant)) in by_group {
            println!(
                "    {group}: {queries} questions, {:.1} supporting titles a question",
                relevant as f64 / queries.max(1) as f64
            );
        }
    }
}

fn dataset_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MUSIQUE_DIR") {
        return PathBuf::from(dir);
    }
    std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"))
        .join("musique")
}

/// Fetches one page if it is not already cached, through `curl` for the reason
/// the other harnesses give, under a partial name renamed on success.
fn download(path: &Path, url: &str) {
    if path.exists() {
        return;
    }
    let partial = path.with_extension("part");
    let status = Command::new("curl")
        .args(["-sSLf", "--max-time", "300", "-o"])
        .arg(&partial)
        .arg(url)
        .status();
    assert!(
        matches!(status, Ok(status) if status.success()),
        "could not fetch {url}\n\
         The dataset is not vendored: MuSiQue is CC BY 4.0 and this fetches it\n\
         at run time. Make `curl` and datasets-server.huggingface.co reachable,\n\
         or point MUSIQUE_DIR at a directory of cached pages."
    );
    std::fs::rename(&partial, path).expect("name the downloaded page");
}

fn score(into: &mut Scores, query: &Query, ranked: &[String]) {
    into.add(ranked, query.relevant.len(), |title| {
        query.relevant.contains(title)
    });
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads the dataset and model weights, and embeds the corpus"]
async fn search_answers_questions_that_take_several_steps() {
    let corpus = Corpus::load();
    corpus.describe();

    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).unwrap_or_else(|| panic!("unknown profile {named}"));
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    let project = format!("musique-{named}-{}", corpus.fingerprint());
    let engine = Engine::open(&workspace, &project, profile, Access::ReadWrite)
        .await
        .expect("open the engine");
    write_corpus(&engine, &corpus).await;

    let edges = channels::live_edges(&engine).await;
    let total: i64 = edges.iter().map(|(_, count)| count).sum();
    println!("  {total} live edges: {edges:?}");
    assert!(
        total > 0,
        "mention derivation built no edges, so nothing here can measure the graph channel"
    );

    if std::env::var("CHANNELS").is_ok() {
        let mut diagnosis = channels::Diagnosis::default();
        for query in &corpus.queries {
            let hits = engine
                .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            diagnosis.observe(&query.group, &hits, |into, ranking| {
                score(into, query, ranking)
            });
        }
        diagnosis.report(&format!("MuSiQue, {named}"), &edges);
        return;
    }

    if std::env::var("CONTEXT").is_ok() {
        let tier = Rerank::default();
        let mut model = pamin_index::Reranker::load(tier, &workspace.root().join("models"))
            .expect("load the reranker");
        let labels = reranking::context_labels();
        let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); labels.len()];
        let mut seeded = 0usize;
        for query in &corpus.queries {
            let hits = engine
                .search_reranked(&query.text, WIDE, DEPTHS, tier)
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits, tier);
            let (orders, shown) = reranking::in_context(&hits, &replayed, &mut model, &query.text);
            seeded += shown;
            for (order, into) in orders.iter().zip(&mut measured) {
                score(into.entry(query.group.clone()).or_default(), query, order);
            }
        }
        reranking::report(
            &format!(
                "what the {} reranker is shown, MuSiQue, {named}; {seeded} candidates shown a seed",
                tier.name()
            ),
            &labels,
            reranking::shipped_context(),
            &measured,
        );
        return;
    }

    // The shipped path, reranker and all, next to the same search with the
    // graph channel taken out -- the one comparison this corpus exists for,
    // made on the path a user gets rather than offline.
    let mut shipped: BTreeMap<String, Scores> = BTreeMap::new();
    let mut graphless: BTreeMap<String, Scores> = BTreeMap::new();
    for query in &corpus.queries {
        for (fusion, into) in [
            (Fusion::default(), &mut shipped),
            (
                Fusion::default().without(pamin_core::Channel::Graph),
                &mut graphless,
            ),
        ] {
            let hits = engine
                .search_reranked_with(&query.text, DEPTH as u32, DEPTHS, Rerank::default(), fusion)
                .await
                .expect("search");
            let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
            score(into.entry(query.group.clone()).or_default(), query, &ranked);
        }
    }

    println!("\n  the shipped search path, {named}, against the same path without the graph");
    println!("  group   questions   nDCG@{NDCG_AT}   recall@{RECALL_AT}   without the graph");
    println!("  ------------------------------------------------------------------------------");
    for (group, scores) in &shipped {
        println!(
            "  {group:<6}   {:>9}   {:>7.4}   {:>9.4}   {}",
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall(),
            statistics::compare(&scores.per_query, &graphless[group].per_query)
        );
    }
    println!("\n  no floors are asserted on this corpus yet: this is its first reading\n");
}

async fn write_corpus(engine: &Engine, corpus: &Corpus) {
    let mut written = 0usize;
    for memory in &corpus.memories {
        let existing = pamin_store::repository::find_topic(
            engine.database.pool(),
            engine.project,
            &memory.title,
        )
        .await
        .expect("look for the topic");
        if existing.is_some() {
            continue;
        }
        written += 1;
        engine
            .write(&Write {
                topic: &memory.title,
                content: &memory.text,
                content_hash: &memory.text.len().to_string(),
                verdict: pamin_core::FilterDecision::Promoted,
                reason: "evaluation corpus",
                promoted: true,
                language: Some("eng"),
                language_confidence: None,
                observed_at: time::OffsetDateTime::now_utc(),
                validity: pamin_core::Validity::ALWAYS,
            })
            .await
            .unwrap_or_else(|error| panic!("writing {}: {error}", memory.title));
        if written.is_multiple_of(2_000) {
            println!("  wrote {written} memories");
        }
    }
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
        println!("  wrote {written} of {} memories", corpus.memories.len());
    }
}
