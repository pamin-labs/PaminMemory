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
//! | `MUSIQUE_QUESTIONS` | read only the first this many questions |
//! | `PAMIN_PROFILE` | which embedding profile, default `accuracy` |
//! | `CHANNELS` | the channel diagnostic and the offline fusion sweep |
//! | `CONTEXT` | price what the reranker is shown, from one run |
//! | `PASSAGES` | a second project whose vectors embed the topic name, paired against this one |
//! | `REACH` | where the supporting titles sit, channel by channel, in the names-only and shared-name projects |
//! | `ENTITIES` | a second project with edges between memories that share a rare proper name, paired against this one |

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

/// Every answerable dev question, 2,417 of them.
///
/// It was the first thousand, the size the graph-retrieval literature reports
/// on -- and the dataset is ordered by hop count, so the first thousand are
/// all two-hop and the three- and four-hop questions, the ones a walk should
/// matter most for, were never asked. `QUESTIONS` is the whole split, read in
/// full so the groups are what the dataset has. `MUSIQUE_QUESTIONS` reads
/// fewer, which is how a run is paired with a project built on a prefix.
const QUESTIONS: usize = 2_417;

fn questions() -> usize {
    std::env::var("MUSIQUE_QUESTIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(QUESTIONS)
}

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
        let wanted = questions();
        for offset in (0..wanted).step_by(PER_PAGE) {
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
        rows.truncate(wanted);

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
            paired.observe(&query.group, &before, &after, |into, ranking| {
                score(into, query, ranking)
            });
        }
        paired.report(&format!(
            "vectors embedding the topic name, MuSiQue, {named}"
        ));
        return;
    }

    if std::env::var("REACH").is_ok() {
        reach(&engine, &corpus, "names only").await;
        let entities = Engine::open(
            &workspace,
            &format!("{project}-entities"),
            profile,
            Access::ReadOnly,
        )
        .await
        .expect("open the entity project; run ENTITIES first");
        reach(&entities, &corpus, "with shared names").await;
        return;
    }

    if std::env::var("ENTITIES").is_ok() {
        entities(&engine, &workspace, &corpus, &project, profile, &named).await;
        return;
    }

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

/// Edges from the names a memory's text shares with another's, priced against
/// edges from topic names alone.
///
/// **Why.** Mention derivation links two memories only when one's *topic name*
/// occurs in the other's text, and on this corpus that connects 39% of the
/// supporting pairs a question needs: the bridge -- "Steve Hillage", between
/// an album and his spouse -- is usually named in both paragraphs and titled
/// in neither. Sharing a capitalised name connects 72% of them, and sharing a
/// *rare* one (in at most 1% of memories) 63%. That is the premise; this arm
/// measures what it is worth.
///
/// **What it builds, and how the noise is held down.** Each memory's proper
/// names are its maximal runs of capitalised words (joined by the usual
/// particles), plus its own title. A name in more than 1% of the memories is a
/// hub and dropped -- "United States" links everything to everything, which
/// is the case HippoRAG's node specificity and SPRIG's hub pruning exist for.
/// Two memories sharing names are weighted by the sum of those names' inverse
/// document frequencies over `ln N`, which puts one shared name in two memories
/// near 0.9 and a common one near 0.5, the confidence a derived mention has;
/// each memory keeps its ten strongest. The edges are `related_to`, derived.
///
/// **Paired, in one run.** A second project holds the same memories with these
/// edges added to the mentions; every question is asked of both, so each row
/// below is a per-question comparison rather than two runs subtracted.
///
/// A prototype: the names are found here, in the harness, and asserted through
/// the store the way `pamin link` does. Only if it pays is it worth a job in
/// the write path, and a model instead of a regular pattern is a later step.
///
/// **Measured, and it does not pay.** 7,463 names, 57,486 edges beside the
/// 12,840 mentions. Fused nDCG@10 on the 1,000 two-hop questions goes from
/// 0.6499 to 0.6322, -0.0177 (156W/251L, p = 0.0001); the graph channel alone
/// falls from 0.1423 to 0.0770 and its net worth to the fused list from
/// +0.0159 to nothing (+0.0018 for removing it, n.s.). The corroboration rule
/// is not the cause -- relaxing it for the graph changes nothing on either
/// project. Coverage rose and precision fell further: a seed now has several
/// times as many neighbours, most unrelated to the question, and the channel's
/// fifty places go to them. That is HippoRAG's own ablation arrived at again
/// -- expanding to neighbours without asking which ones the query wants cost
/// it twelve points -- and it says the next edge set has to be chosen by the
/// query, not added to the graph.
async fn entities(
    base: &Engine,
    workspace: &Workspace,
    corpus: &Corpus,
    project: &str,
    profile: Profile,
    named: &str,
) {
    use pamin_core::{Derivation, EdgeKind, Validity};
    use pamin_store::graph::{EdgeClaim, assert_edges};
    use std::collections::HashMap;

    /// Past this share of the memories a name is a hub and says nothing.
    const HUB: f64 = 0.01;
    /// The strongest neighbours each memory keeps.
    const KEEP: usize = 10;

    let engine = Engine::open(
        workspace,
        &format!("{project}-entities"),
        profile,
        Access::ReadWrite,
    )
    .await
    .expect("open the entity project");
    write_corpus(&engine, corpus).await;

    let names: Vec<HashSet<String>> = corpus
        .memories
        .iter()
        .map(|memory| {
            let mut found = proper_names(&memory.text);
            found.insert(memory.title.to_lowercase());
            found
        })
        .collect();
    let total = names.len() as f64;
    let mut frequency: HashMap<&str, usize> = HashMap::new();
    for found in &names {
        for name in found {
            *frequency.entry(name.as_str()).or_default() += 1;
        }
    }
    let hub = ((HUB * total) as usize).max(2);
    let mut holders: HashMap<&str, Vec<usize>> = HashMap::new();
    for (memory, found) in names.iter().enumerate() {
        for name in found {
            let count = frequency[name.as_str()];
            if count >= 2 && count <= hub {
                holders.entry(name.as_str()).or_default().push(memory);
            }
        }
    }

    let mut weight: HashMap<(usize, usize), f64> = HashMap::new();
    for (name, memories) in &holders {
        let idf = (total / frequency[name] as f64).ln();
        for (at, left) in memories.iter().enumerate() {
            for right in &memories[at + 1..] {
                *weight.entry((*left, *right)).or_default() += idf;
            }
        }
    }
    let mut strongest: Vec<Vec<(usize, f64)>> = vec![Vec::new(); names.len()];
    for ((left, right), strength) in &weight {
        strongest[*left].push((*right, *strength));
        strongest[*right].push((*left, *strength));
    }
    let mut kept: HashMap<(usize, usize), f64> = HashMap::new();
    for (memory, neighbours) in strongest.iter_mut().enumerate() {
        neighbours.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (other, strength) in neighbours.iter().take(KEEP) {
            kept.insert((memory.min(*other), memory.max(*other)), *strength);
        }
    }

    let mut ids = Vec::with_capacity(corpus.memories.len());
    for memory in &corpus.memories {
        let topic = pamin_store::repository::find_topic(
            engine.database.pool(),
            engine.project,
            &memory.title,
        )
        .await
        .expect("look up a topic")
        .expect("every memory was written");
        ids.push(topic.id);
    }
    let scale = total.ln();
    let mut edges: Vec<_> = kept
        .iter()
        .map(|((left, right), strength)| {
            (
                ids[*left],
                ids[*right],
                EdgeClaim {
                    kind: EdgeKind::RelatedTo,
                    derivation: Derivation::Deterministic,
                    confidence: (strength / scale).clamp(0.05, 1.0) as f32,
                    validity: Validity::ALWAYS,
                    caused_by_topic_state: None,
                },
            )
        })
        .collect();
    edges.sort_by_key(|(left, right, _)| (left.0, right.0));
    for chunk in edges.chunks(1_000) {
        assert_edges(engine.database.pool(), engine.project, chunk)
            .await
            .expect("assert the entity edges");
    }
    println!(
        "  {} names in 2..={hub} memories, {} pairs share one, {} edges kept ({} a memory at most)",
        holders.len(),
        weight.len(),
        kept.len(),
        KEEP
    );

    let base_edges = channels::live_edges(base).await;
    let entity_edges = channels::live_edges(&engine).await;
    println!("  names only: {base_edges:?}\n  with shared names: {entity_edges:?}");

    let mut before: BTreeMap<String, Scores> = BTreeMap::new();
    let mut after: BTreeMap<String, Scores> = BTreeMap::new();
    let mut diagnosis = channels::Diagnosis::default();
    for query in &corpus.queries {
        for (searcher, into, observed) in [(base, &mut before, false), (&engine, &mut after, true)]
        {
            let hits = searcher
                .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
            score(into.entry(query.group.clone()).or_default(), query, &ranked);
            if observed {
                diagnosis.observe(&query.group, &hits, |into, ranking| {
                    score(into, query, ranking)
                });
            }
        }
    }

    println!("\n  fused, with edges from shared names against names only, {named}");
    println!(
        "  group   questions   nDCG@{NDCG_AT} before   after   recall@{RECALL_AT} before   after   paired"
    );
    for (group, scores) in &after {
        let was = &before[group];
        println!(
            "  {group:<6}   {:>9}   {:>12.4}   {:>5.4}   {:>14.4}   {:>5.4}   {}",
            scores.queries,
            was.mean_ndcg(),
            scores.mean_ndcg(),
            was.mean_recall(),
            scores.mean_recall(),
            statistics::compare(&was.per_query, &scores.per_query)
        );
    }
    diagnosis.report(
        &format!("MuSiQue with shared-name edges, {named}"),
        &entity_edges,
    );
}

/// A text's proper names, lowercased: maximal runs of capitalised words,
/// joined across the particles names carry ("Duke of York", "van Gogh").
///
/// A regular pattern rather than a model, which is the point of trying it
/// first: it costs nothing, and whether names are worth edges at all is the
/// question. It only sees scripts with case, which this corpus is.
fn proper_names(text: &str) -> HashSet<String> {
    const PARTICLES: [&str; 8] = ["of", "the", "de", "von", "van", "and", "la", "du"];
    let words: Vec<&str> = text
        .split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | ';' | ':' | '(' | ')' | '"' | '“' | '”')
        })
        .filter(|word| !word.is_empty())
        .collect();
    let capital = |word: &str| word.chars().next().is_some_and(char::is_uppercase);
    let clean = |word: &str| {
        word.trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase()
    };

    let mut names = HashSet::new();
    let mut at = 0;
    while at < words.len() {
        if !capital(words[at]) {
            at += 1;
            continue;
        }
        let mut run = vec![clean(words[at])];
        let mut end = at + 1;
        while end < words.len() {
            if capital(words[end]) {
                run.push(clean(words[end]));
                end += 1;
            } else if PARTICLES.contains(&words[end])
                && end + 1 < words.len()
                && capital(words[end + 1])
            {
                run.push(words[end].to_string());
                end += 1;
            } else {
                break;
            }
            // A sentence break ends a name even between capitals.
            if words[end - 1].ends_with('.') && words[end - 1].len() > 3 {
                break;
            }
        }
        let name = run.join(" ");
        if name.chars().count() > 2 {
            names.insert(name);
        }
        at = end;
    }
    names
}

#[test]
fn proper_names_are_runs_of_capitals_across_particles() {
    let names = proper_names(
        "Green is an album by Steve Hillage, recorded with the Duke of York in London.",
    );
    assert!(names.contains("steve hillage"), "{names:?}");
    assert!(names.contains("duke of york"), "{names:?}");
    assert!(names.contains("london"), "{names:?}");
    assert!(
        !names.iter().any(|name| name.contains("album")),
        "{names:?}"
    );
}

/// Where the supporting titles the fused list ranks badly actually are.
///
/// The question it answers decides what to change next. The reranker reads
/// the fused head -- the first `Rerank::default().depth()` -- and nothing
/// below it, so a supporting title that only the graph found helps a user only
/// if fusion puts it inside that head. If the graph finds such titles and
/// fusion leaves them below it, the bottleneck is the handoff to the reranker
/// and not the edges; if the graph never finds them, it is the edges.
async fn reach(engine: &Engine, corpus: &Corpus, label: &str) {
    use pamin_core::{Channel, Why};

    let head = Rerank::default().depth();
    let mut missed = 0usize;
    let mut graph_only = 0usize;
    let mut graph_only_in_head = 0usize;
    let mut graph_found_missed = 0usize;
    let mut vector_found_missed = 0usize;
    let mut fused_ranks: Vec<usize> = Vec::new();
    for query in &corpus.queries {
        let hits = engine
            .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        for (position, hit) in hits.iter().enumerate() {
            if !query.relevant.contains(&hit.topic) {
                continue;
            }
            let channels: Vec<Channel> = hit
                .result
                .why
                .iter()
                .filter_map(|why| match why {
                    Why::Channel { channel, .. } => Some(*channel),
                    _ => None,
                })
                .collect();
            let only_graph = channels == [Channel::Graph];
            if only_graph {
                graph_only += 1;
                fused_ranks.push(position + 1);
                if position < head {
                    graph_only_in_head += 1;
                }
            }
            if position >= NDCG_AT {
                missed += 1;
                if channels.contains(&Channel::Graph) {
                    graph_found_missed += 1;
                }
                if channels.contains(&Channel::Vector) {
                    vector_found_missed += 1;
                }
            }
        }
    }
    fused_ranks.sort_unstable();
    let median = fused_ranks.get(fused_ranks.len() / 2).copied().unwrap_or(0);
    let total: usize = corpus
        .queries
        .iter()
        .map(|query| query.relevant.len())
        .sum();
    println!("\n  where the supporting titles are, {label}: {total} supporting titles");
    println!(
        "  found by the graph alone: {graph_only}, of which {graph_only_in_head} inside the \
         reranker's head of {head}; median fused rank {median}"
    );
    println!(
        "  in the list but below rank {NDCG_AT}: {missed}, of which the graph found {graph_found_missed} \
         and the vector channel {vector_found_missed}"
    );
}
