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
//! | `MUSIQUE_QUESTIONS` | read only the first this many questions -- the split is ordered by hop count, so a short prefix is all two-hop; kept so a run can be paired with a project built on a prefix |
//! | `MUSIQUE_SAMPLE` | ask this many, drawn from the whole split by a seeded hash, stratified by hop count |
//! | `EVAL_SEED` | the seed `MUSIQUE_SAMPLE` draws with, default `harness::shape::SEED` |
//! | `CHANNELS` | the channel diagnostic and the offline fusion sweep |
//! | `FEATURES_OUT` | a path: every candidate fusion saw, one row each, for fitting a fusion offline |
//! | `CONTEXT` | price what the reranker is shown, from one run |
//! | `PASSAGES` | a second project whose vectors embed the topic name, paired against this one |
//! | `ROUTES` | a cascade and gates that spend less on the reranker, against the shipped pass |
//! | `REACH` | where the supporting titles sit, channel by channel, in the names-only and shared-name projects |
//! | `ENTITIES` | a second project with edges between memories that share a rare proper name, paired against this one |

mod channels;
mod features;
mod harness;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::process::Command;

use pamin_core::Fusion;
use pamin_engine::{Depths, Engine};
use pamin_index::{Access, VectorIndex};
use pamin_store::Workspace;

use harness::shape::Shape;
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

/// `MUSIQUE_SAMPLE`: how many questions to draw from the whole split.
///
/// The way to ask fewer than all of them and still ask every hop count. A
/// prefix of 600 -- what runs have used to stay short -- is 600 two-hop
/// questions, because the split is ordered by hop count; the three- and
/// four-hop questions, where a walk should matter most, were never in it.
fn sample() -> Option<usize> {
    let sample = std::env::var("MUSIQUE_SAMPLE").ok().map(|value| {
        value
            .parse()
            .expect("MUSIQUE_SAMPLE is a number of questions")
    });
    assert!(
        sample.is_none() || std::env::var("MUSIQUE_QUESTIONS").is_err(),
        "MUSIQUE_SAMPLE draws from the whole split and MUSIQUE_QUESTIONS takes a prefix; set one"
    );
    sample
}

/// How deep to retrieve, so recall@50 can be scored.
const DEPTH: usize = RECALL_AT + 1;

const DEPTHS: Depths = Depths::DEFAULT;

/// Past four times the channel depth, so a trace is never truncated.
const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;

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
    /// The dataset's own id.
    id: String,
    text: String,
    /// `2hop`, `3hop` or `4hop`, read from the question's id.
    group: String,
    /// Titles of the supporting paragraphs.
    relevant: HashSet<String>,
}

struct Corpus {
    memories: Vec<Memory>,
    queries: Vec<Query>,
    /// How the questions were chosen, and with what seed, for the run's shape.
    selection: String,
    seed: Option<u64>,
    /// Questions read and not asked: id, group, reason.
    excluded: Vec<(String, String, String)>,
}

impl Corpus {
    fn load() -> Self {
        let dir = harness::dataset_dir("MUSIQUE_DIR", "musique");
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));

        let mut rows: Vec<Row> = Vec::new();
        let mut reported: Option<usize> = None;
        let sample = sample();
        let wanted = if sample.is_some() {
            QUESTIONS
        } else {
            questions()
        };
        for offset in (0..wanted).step_by(PER_PAGE) {
            let path = dir.join(format!("page-{offset:05}.json"));
            download(&path, &format!("{ROWS}&offset={offset}&length={PER_PAGE}"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            let body: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));
            if let Some(total) = body["num_rows_total"].as_u64() {
                let total = total as usize;
                if let Some(earlier) = reported {
                    assert_eq!(
                        earlier,
                        total,
                        "{} reports {total} rows in the split where an earlier page reported \
                         {earlier}: the cached pages are from different versions of it",
                        path.display()
                    );
                }
                reported = Some(total);
            }
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
        complete(rows.len(), wanted, reported, &dir);
        rows.truncate(wanted);
        distinct(rows.iter().map(|row| row.id.as_str()), &dir);
        let (rows, selection, seed) = select(rows, sample, harness::shape::seed());

        // Distinct paragraphs per title, in first-seen order so the joined
        // text is the same on every run.
        let mut texts: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut queries = Vec::new();
        let mut excluded = Vec::new();
        for row in &rows {
            for paragraph in &row.paragraphs {
                let seen = texts.entry(paragraph.title.clone()).or_default();
                if !seen.contains(&paragraph.paragraph_text) {
                    seen.push(paragraph.paragraph_text.clone());
                }
            }
            let relevant: HashSet<String> = row
                .paragraphs
                .iter()
                .filter(|paragraph| paragraph.is_supporting)
                .map(|paragraph| paragraph.title.clone())
                .collect();
            // Scored, it would count zero whatever came back -- a question the
            // corpus cannot grade. Its paragraphs stay: they are documents
            // every other question is searched among.
            if !answerable(row) {
                excluded.push((row.id.clone(), group(row), "no supporting paragraph".into()));
                continue;
            }
            queries.push(Query {
                id: row.id.clone(),
                text: row.question.clone(),
                group: group(row),
                relevant,
            });
        }

        let memories = texts
            .into_iter()
            .map(|(title, paragraphs)| Memory {
                title,
                text: paragraphs.join("\n").chars().take(MAX_CHARS).collect(),
            })
            .collect();

        Self {
            memories,
            queries,
            selection,
            seed,
            excluded,
        }
    }

    /// What a run asks: every memory, every question with its supporting
    /// titles, and the questions excluded at load. See [`harness::shape`].
    fn shape(&self) -> Shape {
        let mut shape = Shape::new("musique", self.selection.clone(), self.seed);
        for memory in &self.memories {
            shape.document(&[&memory.title, &memory.text]);
        }
        for query in &self.queries {
            let mut relevant: Vec<&str> = query.relevant.iter().map(String::as_str).collect();
            relevant.sort_unstable();
            let mut fields = vec![query.text.as_str()];
            fields.extend(relevant);
            shape.query(&[&query.group], &query.id, &fields);
        }
        for (id, group, reason) in &self.excluded {
            shape.exclude(&[group], id, reason);
        }
        shape
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

/// Fails unless the pages held every question asked for and, when the whole
/// split was asked for -- which is what `MUSIQUE_SAMPLE` draws from -- unless
/// the dataset API reports that as the whole split.
///
/// A page that came back short, cached or fetched, drops the tail. The split
/// is ordered by hop count, so the tail is the four-hop questions, and a
/// sample drawn from what is left still asks every hop count it has.
///
/// The whole split needs the reported total: without it, a split that grew
/// past `QUESTIONS` fills the last page past it, and the rows beyond are
/// dropped as surplus, so a sample would be drawn from a prefix.
fn complete(loaded: usize, wanted: usize, reported: Option<usize>, dir: &Path) {
    if wanted == QUESTIONS {
        let total = reported.unwrap_or_else(|| {
            panic!(
                "no page under {} reports how many rows the split has, so reading {QUESTIONS} \
                 cannot be shown to be the whole of it. Delete the cached pages and rerun",
                dir.display()
            )
        });
        assert_eq!(
            total, QUESTIONS,
            "the dataset API reports {total} rows in the split, where the whole split is \
             taken to be {QUESTIONS}: the dataset has changed, and `QUESTIONS` with it"
        );
    }
    assert!(
        loaded >= wanted,
        "read {loaded} of the {wanted} MuSiQue questions asked for, {} short: a page under \
         {} came back short. Delete the cached pages and rerun",
        wanted - loaded,
        dir.display()
    );
}

/// Fails if any question id was read twice. Pages that overlap -- a cached
/// page from one offset beside a fetched page from another -- can hold the
/// right number of rows with some repeated and others missing, which the row
/// count alone does not see.
fn distinct<'a>(ids: impl Iterator<Item = &'a str>, dir: &Path) {
    let mut seen = HashSet::new();
    let repeated: BTreeSet<&str> = ids.filter(|id| !seen.insert(*id)).collect();
    assert!(
        repeated.is_empty(),
        "{} MuSiQue question ids were read more than once, e.g. {:?}: the pages under {} \
         overlap. Delete the cached pages and rerun",
        repeated.len(),
        repeated.iter().take(3).collect::<Vec<_>>(),
        dir.display()
    );
}

/// The loader's premise: a run reads every question it asks for, once each,
/// and the whole split is what the API says it is.
#[test]
fn a_short_read_of_the_split_fails() {
    let dir = Path::new("cache");
    distinct(["2hop__1", "3hop__1"].into_iter(), dir);
    let repeated = std::panic::catch_unwind(|| {
        distinct(["2hop__1", "3hop__1", "2hop__1"].into_iter(), dir);
    })
    .expect_err("a repeated id passed");
    let message = repeated
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("1 MuSiQue question ids"), "{message}");

    complete(QUESTIONS, QUESTIONS, Some(QUESTIONS), dir);
    complete(600, 600, Some(QUESTIONS), dir);
    complete(600, 600, None, dir);
    for (loaded, wanted, reported, expected) in [
        (QUESTIONS, QUESTIONS, None, "no page under"),
        (2_400, QUESTIONS, Some(QUESTIONS), "17 short"),
        (QUESTIONS, QUESTIONS, Some(2_500), "reports 2500 rows"),
        (500, 600, Some(QUESTIONS), "100 short"),
    ] {
        let failed = std::panic::catch_unwind(|| complete(loaded, wanted, reported, dir))
            .expect_err("a short read passed");
        let message = failed.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains(expected), "{message}");
    }
}

/// `2hop`, `3hop` or `4hop`, read from the question's id.
fn group(row: &Row) -> String {
    let hops = row.id.split("hop").next().unwrap_or("?");
    format!("{hops}hop")
}

/// The rows a run asks, how they were chosen in words, and the seed if one
/// chose them.
///
/// A sample is drawn per hop count in proportion to the split, by a hash of
/// the seed and the question id, and never by anything a run measured. Each
/// row carries its twenty paragraphs, and the corpus is built from the rows
/// drawn, so a question is asked among exactly the paragraphs it came with:
/// the linked group stays whole. The corpus is smaller than the whole split's,
/// so a sample's figures are not comparable with a prefix's or the whole
/// split's -- compare samples with samples, at the same seed.
fn select(rows: Vec<Row>, sample: Option<usize>, seed: u64) -> (Vec<Row>, String, Option<u64>) {
    let Some(wanted) = sample else {
        let selection = if std::env::var("MUSIQUE_QUESTIONS").is_ok() {
            format!(
                "the first {} questions in the dataset's order, which is by hop count",
                rows.len()
            )
        } else {
            "every answerable dev question".to_string()
        };
        return (rows, selection, None);
    };
    let read = rows.len();
    let hops: HashSet<String> = rows
        .iter()
        .filter(|row| answerable(row))
        .map(group)
        .collect();
    let drawn = harness::shape::stratified(rows, wanted, seed, group, |row| row.id.clone());
    // The premise of the mode: every hop count is asked. A sample too small
    // to reach one is a prefix by another name. Asked means scored: a drawn
    // question with no supporting paragraph is excluded at load, so it does
    // not reach its hop count.
    let reached: HashSet<String> = drawn
        .iter()
        .filter(|row| answerable(row))
        .map(group)
        .collect();
    assert_eq!(
        reached, hops,
        "a sample of {wanted} does not reach every hop count in the split"
    );
    let selection =
        format!("a seeded sample of {wanted} of {read} questions, stratified by hop count");
    (drawn, selection, Some(seed))
}

/// Whether a question can be graded: it has a supporting paragraph to find.
fn answerable(row: &Row) -> bool {
    row.paragraphs
        .iter()
        .any(|paragraph| paragraph.is_supporting)
}

/// Rows ordered by hop count, as the split is, each with one supporting
/// paragraph unless its id is in `unanswerable`.
fn rows(unanswerable: &[String]) -> Vec<Row> {
    [("2hop", 50), ("3hop", 30), ("4hop", 20)]
        .into_iter()
        .flat_map(|(hops, count)| {
            (0..count).map(move |at| {
                let id = format!("{hops}__{at}");
                Row {
                    paragraphs: vec![Paragraph {
                        title: id.clone(),
                        paragraph_text: String::new(),
                        is_supporting: !unanswerable.contains(&id),
                    }],
                    id,
                    question: String::new(),
                }
            })
        })
        .collect()
}

/// A hop count whose only drawn questions are excluded at load is not asked,
/// and the sample says so before anything is indexed.
#[test]
#[should_panic(expected = "does not reach every hop count")]
fn a_sample_whose_drawn_questions_cannot_be_graded_does_not_reach_their_hop_count() {
    let (drawn, _, _) = select(rows(&[]), Some(10), harness::shape::SEED);
    let four: Vec<String> = drawn
        .iter()
        .filter(|row| group(row) == "4hop")
        .map(|row| row.id.clone())
        .collect();
    assert!(!four.is_empty() && four.len() < 20);
    // Drawn by id alone, so the same rows are drawn again; the split still
    // has gradable four-hop questions, just not these.
    select(rows(&four), Some(10), harness::shape::SEED);
}

#[test]
fn a_sample_asks_every_hop_count_where_a_prefix_does_not() {
    let rows = rows(&[]);
    let prefix: HashSet<String> = rows[..10].iter().map(group).collect();
    assert_eq!(prefix.len(), 1, "{prefix:?}");

    let (drawn, _, seed) = select(rows, Some(10), harness::shape::SEED);
    assert_eq!(seed, Some(harness::shape::SEED));
    let count = |hops: &str| drawn.iter().filter(|row| group(row) == hops).count();
    assert_eq!((count("2hop"), count("3hop"), count("4hop")), (5, 3, 2));
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

/// Scores one ranking, and counts it toward the run's coverage. Every arm
/// that scores comes through here; the two that do not -- `FEATURES_OUT` and
/// `REACH` -- count each question themselves.
fn score(into: &mut Scores, query: &Query, ranked: &[String]) {
    harness::shape::tally(&query.group, &query.id);
    into.add(ranked, query.relevant.len(), |title| {
        query.relevant.contains(title)
    });
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads the dataset and model weights, and embeds the corpus"]
async fn search_answers_questions_that_take_several_steps() {
    let corpus = Corpus::load();
    let named = harness::profile();
    let mut shape = corpus.shape();
    // Published before anything is printed, so a failed write to stdout
    // still leaves the in-progress record.
    shape.publish("search_answers_questions_that_take_several_steps", &named);
    corpus.describe();
    // The run ends on a significance gate per hop count; a sample too small
    // to ever pass it fails here, before the corpus is indexed.
    let mut asked: BTreeMap<&str, usize> = BTreeMap::new();
    for query in &corpus.queries {
        *asked.entry(query.group.as_str()).or_default() += 1;
    }
    let gated = asked.iter().all(|(group, questions)| {
        statistics::gate(&format!("MuSiQue {group}, the graph's worth"), *questions)
    });
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
    shape.observe_models();
    let engine = Engine::open(
        &workspace,
        &project,
        VectorIndex::default(),
        Access::ReadWrite,
        &crate::harness::settings(),
    )
    .await
    .expect("open the engine");
    shape.ran_on(harness::shape::engines(&[("primary", &engine)]));
    if std::env::var("CHANNELS").is_ok() {
        assert_eq!(engine.passage(), pamin_index::Passage::Named);
    }
    write_corpus(&engine, &corpus, &mut shape, "primary", &workspace).await;
    if std::env::var("CHANNELS").is_ok() {
        assert_eq!(engine.passage(), pamin_index::Passage::Named);
        assert_eq!(
            engine.indexed_documents().expect("count documents") as usize,
            corpus.memories.len()
        );
    }

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
            VectorIndex::default(),
            Access::ReadWrite,
            &crate::harness::settings(),
        )
        .await
        .expect("open the named project");
        shape.ran_on(harness::shape::engines(&[
            ("primary", &engine),
            ("named", &other),
        ]));
        write_corpus(&other, &corpus, &mut shape, "named", &workspace).await;
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
                .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
                .await
                .expect("search");
            let after = other
                .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
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

    // `ROUTES`: spending less on the reranker -- gates that skip it, and a
    // learned router -- priced against the shipped pass. See
    // `reranking::Routes`.
    if std::env::var("ROUTES").is_ok() {
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        let mut routes = reranking::Routes::default();
        for query in &corpus.queries {
            let hits = engine
                .search_reranked(&query.text, WIDE, harness::depths())
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits);
            routes.observe(
                &query.group,
                &query.text,
                &hits,
                &replayed,
                &query.text,
                |into, ranking| score(into, query, ranking),
            );
        }
        routes.report(&format!(
            "spending less on the {} reranker, MuSiQue, {named}",
            pamin_index::reranking::TIER
        ));
        return;
    }

    // `RERANK_RULES`: every rule for using the reranker's scores, priced
    // against the shipped one from a single shipped run. See `reranking`.
    if std::env::var("RERANK_RULES").is_ok() {
        let mut priced = reranking::Priced::default();
        for query in &corpus.queries {
            let hits = engine
                .search_reranked(&query.text, WIDE, harness::depths())
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits);
            priced.observe(
                &engine,
                &query.group,
                &query.text,
                &hits,
                &replayed,
                None,
                |into, ranking| score(into.entry(query.group.clone()).or_default(), query, ranking),
            );
        }
        priced.report(&format!(
            "rules for the {} reranker's scores, MuSiQue, {named}",
            pamin_index::reranking::TIER
        ));
        return;
    }

    if std::env::var("REACH").is_ok() {
        harness::shape::tables(["names only", "with shared names"]);
        reach(&engine, &corpus, "names only").await;
        let entities = Engine::open(
            &workspace,
            &format!("{project}-entities"),
            VectorIndex::default(),
            Access::ReadOnly,
            &crate::harness::settings(),
        )
        .await
        .expect("open the entity project; run ENTITIES first");
        shape.ran_on(harness::shape::engines(&[
            ("primary", &engine),
            ("entities", &entities),
        ]));
        harness::index(
            &mut shape,
            "entities",
            &entities,
            workspace.root(),
            corpus.memories.len(),
        )
        .await;
        reach(&entities, &corpus, "with shared names").await;
        return;
    }

    if std::env::var("ENTITIES").is_ok() {
        entities(&engine, &mut shape, &workspace, &corpus, &project, &named).await;
        return;
    }

    // `FEATURES_OUT`: every candidate fusion saw, one row each, for fitting a
    // fusion offline. See `features`.
    if let Some(mut dump) = features::Features::from_env("musique") {
        harness::shape::tables(["features"]);
        harness::shape::table("features");
        for query in &corpus.queries {
            let hits = engine
                .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
                .await
                .expect("search");
            let asked = features::Asked {
                group: &query.group,
                question: &query.id,
                text: &query.text,
                // What `write_corpus` labels every memory with.
                language: Some("eng"),
                judged: query.relevant.len(),
            };
            dump.observe(&asked, &hits, WIDE, |title| {
                Some(f64::from(query.relevant.contains(title)))
            });
            harness::shape::tally(&query.group, &query.id);
        }
        dump.finish(corpus.queries.len());
        return;
    }

    if std::env::var("CHANNELS").is_ok() {
        let mut diagnosis = channels::Diagnosis::default();
        for query in &corpus.queries {
            let hits = engine
                .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
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

    if let Some(variants) = channels::requested_variants() {
        let questions: Vec<(String, String)> = corpus
            .queries
            .iter()
            .map(|query| (query.text.clone(), query.group.clone()))
            .collect();
        channels::compare_reranked(
            &engine,
            &format!("MuSiQue, {named}"),
            &questions,
            DEPTH as u32,
            harness::depths(),
            &variants,
            |index, into, hits| {
                let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
                score(into, &corpus.queries[index], &ranked);
            },
        )
        .await;
        return;
    }

    if std::env::var("CONTEXT").is_ok() {
        let model = harness::shape::owned(
            pamin_index::Reranker::load(
                &workspace.root().join("models"),
                &crate::harness::support::inference(),
            )
            .expect("load the reranker"),
        );
        shape.also_ran("context reranker", &model, pamin_index::Reranker::plans);
        let labels = reranking::context_labels();
        let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); labels.len()];
        harness::shape::tables(labels.iter().map(|(label, _)| label));
        let mut seeded = 0usize;
        for query in &corpus.queries {
            let hits = engine
                .search_reranked(&query.text, WIDE, harness::depths())
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits);
            let (orders, shown) = reranking::in_context(
                &hits,
                &replayed,
                &mut harness::shape::lock(&model),
                &query.text,
            );
            seeded += shown;
            for ((order, into), (label, _)) in orders.iter().zip(&mut measured).zip(&labels) {
                harness::shape::table(label);
                score(into.entry(query.group.clone()).or_default(), query, order);
            }
        }
        reranking::report(
            &format!(
                "what the {} reranker is shown, MuSiQue, {named}; {seeded} candidates shown a seed",
                pamin_index::reranking::TIER
            ),
            &labels,
            reranking::shipped_context(),
            &measured,
            channels::Units::Questions,
        );
        return;
    }

    // The shipped path, reranker and all, next to the same search with the
    // graph channel taken out -- the one comparison this corpus exists for,
    // made on the path a user gets rather than offline.
    let mut shipped: BTreeMap<String, Scores> = BTreeMap::new();
    let mut graphless: BTreeMap<String, Scores> = BTreeMap::new();
    harness::shape::tables(["shipped", "without the graph"]);
    for query in &corpus.queries {
        for (table, fusion, into) in [
            ("shipped", Fusion::default(), &mut shipped),
            (
                "without the graph",
                Fusion::default().without(pamin_core::Channel::Graph),
                &mut graphless,
            ),
        ] {
            harness::shape::table(table);
            let hits = engine
                .search_reranked_with(&query.text, DEPTH as u32, harness::depths(), fusion)
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
    if let Some(work) = engine.reranked() {
        println!(
            "  loaded reranker truncation: {} tokens",
            work.maximum_tokens
        );
        println!(
            "  both arms: {} scored pairs, {} tokens, {} padded tokens, {} batches; \
             encoding {:.1} ms, padding + inference {:.1} ms",
            work.scored,
            work.tokens,
            work.padded_tokens,
            work.batches,
            work.encode_us as f64 / 1_000.0,
            work.forward_us as f64 / 1_000.0,
        );
    }
    println!();

    // What this corpus exists to show. First read at 1,000 two-hop questions:
    // 0.6834 shipped, and the graph worth 0.0406 of it (114W/238L, p=0.0001).
    // Asserted as a direction rather than a figure, because the group's mean
    // moves with how many questions are asked.
    for (group, scores) in shipped.iter().filter(|_| gated) {
        let without = statistics::compare(&scores.per_query, &graphless[group].per_query);
        assert!(
            without.mean < 0.0 && without.is_significant(),
            "{group}: the shipped search no longer needs the graph ({without})"
        );
    }
}

/// Hands the index to `shape` under `label` (see `harness::index`), then
/// writes every memory not already a topic and runs the queue.
async fn write_corpus(
    engine: &Engine,
    corpus: &Corpus,
    shape: &mut Shape,
    label: &str,
    workspace: &Workspace,
) {
    harness::index(
        shape,
        label,
        engine,
        workspace.root(),
        corpus.memories.len(),
    )
    .await;
    let mut written = 0usize;
    for memory in &corpus.memories {
        if !harness::write_absent(
            engine,
            &memory.title,
            &memory.text,
            "eng",
            "evaluation corpus",
        )
        .await
        {
            continue;
        }
        written += 1;
        if written.is_multiple_of(2_000) {
            println!("  wrote {written} memories");
        }
    }
    harness::drain(engine).await;
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
/// was not the cause -- relaxing it for the graph changed nothing on either
/// project, and it has since been removed. Coverage rose and precision fell further: a seed now has several
/// times as many neighbours, most unrelated to the question, and the channel's
/// fifty places go to them. That is HippoRAG's own ablation arrived at again
/// -- expanding to neighbours without asking which ones the query wants cost
/// it twelve points -- and it says the next edge set has to be chosen by the
/// query, not added to the graph.
async fn entities(
    base: &Engine,
    shape: &mut Shape,
    workspace: &Workspace,
    corpus: &Corpus,
    project: &str,
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
        VectorIndex::default(),
        Access::ReadWrite,
        &crate::harness::settings(),
    )
    .await
    .expect("open the entity project");
    shape.ran_on(harness::shape::engines(&[
        ("primary", base),
        ("entities", &engine),
    ]));
    write_corpus(&engine, corpus, shape, "entities", workspace).await;

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
    harness::shape::tables(["names only", "with shared names"]);
    for query in &corpus.queries {
        for (searcher, into, observed) in [(base, &mut before, false), (&engine, &mut after, true)]
        {
            harness::shape::table(if observed {
                "with shared names"
            } else {
                "names only"
            });
            let hits = searcher
                .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
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
/// the fused head -- the first `crate::harness::depths().rerank` -- and nothing
/// below it, so a supporting title that only the graph found helps a user only
/// if fusion puts it inside that head. If the graph finds such titles and
/// fusion leaves them below it, the bottleneck is the handoff to the reranker
/// and not the edges; if the graph never finds them, it is the edges.
async fn reach(engine: &Engine, corpus: &Corpus, label: &str) {
    use pamin_core::{Channel, Why};

    let head = crate::harness::depths().rerank;
    let mut missed = 0usize;
    let mut graph_only = 0usize;
    let mut graph_only_in_head = 0usize;
    let mut graph_found_missed = 0usize;
    let mut vector_found_missed = 0usize;
    let mut fused_ranks: Vec<usize> = Vec::new();
    harness::shape::table(label);
    for query in &corpus.queries {
        let hits = engine
            .search_fused(&query.text, WIDE, harness::depths(), Fusion::default())
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        harness::shape::tally(&query.group, &query.id);
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
