//! Retrieval over LOCOMO: what the graph channel is worth on agent memory.
//!
//! **What this corpus is here to settle.** LOCOMO is ten long conversations
//! between two people and the questions the agent-memory systems in this
//! category publish on. Every question names the turns it was answered from
//! (`evidence`), so retrieval can be graded without a reader or a judge: did
//! the search return the turns the answer is in. That is what this harness
//! measures, and all it measures -- an answer's correctness is the Python
//! comparison under `benchmarks/`, which needs a model to read and to judge.
//!
//! **Two projects per conversation, paired.** Each conversation is written
//! twice: `flat`, every turn an independent memory, which is how the flat
//! benchmark arm ingests it; and `linked`, the same memories with consecutive
//! turns of a session joined by an explicit `related_to` edge, which is what
//! the ledger arm does with `pamin link`. The topic names are dialogue ids, so
//! mention derivation finds nothing in either and the only edges are those
//! links. Every question is asked of both, so each row is a per-question
//! comparison of the same search with and without the graph having anything
//! to walk -- the graph channel's worth on this corpus, not a guess at it.
//!
//! **Scored by category.** LOCOMO's five categories are reported separately
//! and never summed. Adversarial questions ask about something only the other
//! speaker said; their evidence is still the turn that says it, so they grade
//! retrieval the same way and are kept.
//!
//! **Licence.** LOCOMO is CC BY-NC 4.0 (Maharana et al., ACL 2024). It is
//! fetched at run time from the authors' repository, not vendored.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p pamin-engine --test locomo -- --ignored --nocapture
//! ```
//!
//! | variable | what it does |
//! |---|---|
//! | `LOCOMO_DIR` | where `locomo10.json` is cached, instead of fetching it |
//! | `LOCOMO_CONVERSATIONS` | read only the first this many conversations |
//! | `PAMIN_PROFILE` | which embedding profile, default `accuracy` |

mod channels;
mod harness;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process::Command;

use pamin_core::EdgeKind;
use pamin_engine::{Depths, Engine};
use pamin_index::{Access, Rerank, VectorIndex};
use pamin_store::Workspace;
use pamin_store::graph::{self, EdgeClaim};
use scoring::{NDCG_AT, RECALL_AT, Scores};

/// The upstream release, pinned to its file in the authors' repository.
const DATA: &str = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";

/// What the product's `pamin search` asks for by default.
const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

const CATEGORY: [&str; 5] = [
    "multi-hop",
    "temporal",
    "open-domain",
    "single-hop",
    "adversarial",
];

struct Turn {
    topic: String,
    session: String,
    record: String,
}

struct Question {
    text: String,
    category: String,
    evidence: HashSet<String>,
}

struct Conversation {
    id: String,
    turns: Vec<Turn>,
    questions: Vec<Question>,
}

/// A dialogue id as a topic name: `D1:3` becomes `D1_3`, the way the Python
/// arms write it, so both harnesses name the same memory the same way.
fn topic_of(dia_id: &str) -> String {
    dia_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn load() -> Vec<Conversation> {
    let dir = harness::dataset_dir("LOCOMO_DIR", "locomo");
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
    let path = dir.join("locomo10.json");
    download(&path);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let samples: Vec<serde_json::Value> = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));

    let wanted = std::env::var("LOCOMO_CONVERSATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(samples.len());
    samples
        .iter()
        .take(wanted)
        .map(|sample| {
            let id = sample["sample_id"].as_str().expect("sample_id").to_string();
            let conversation = sample["conversation"].as_object().expect("conversation");
            let mut sessions: Vec<(u32, &str)> = conversation
                .keys()
                .filter_map(|key| {
                    let number = key.strip_prefix("session_")?.parse().ok()?;
                    Some((number, key.as_str()))
                })
                .collect();
            sessions.sort_unstable();

            // The date goes into the text, as `benchmarks/datasets/locomo.py`
            // puts it, because two of the five categories ask when.
            let mut turns = Vec::new();
            for (_, key) in sessions {
                let when = conversation
                    .get(&format!("{key}_date_time"))
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                for turn in conversation[key].as_array().expect("a session's turns") {
                    let text = turn["text"].as_str().unwrap_or("").trim();
                    let Some(dia_id) = turn["dia_id"].as_str() else {
                        continue;
                    };
                    if text.is_empty() {
                        continue;
                    }
                    let speaker = turn["speaker"].as_str().unwrap_or("");
                    turns.push(Turn {
                        topic: topic_of(dia_id),
                        session: key.to_string(),
                        record: format!("[{when}] {speaker}: {text}"),
                    });
                }
            }

            let known: HashSet<&str> = turns.iter().map(|turn| turn.topic.as_str()).collect();
            let questions = sample["qa"]
                .as_array()
                .expect("qa")
                .iter()
                .filter_map(|qa| {
                    let category = qa["category"].as_u64()? as usize;
                    // Evidence ids are sometimes written with a typo upstream
                    // (`D:11:26`, `D30:05`); a question none of whose evidence
                    // names a turn that exists cannot be graded and is left out.
                    let evidence: HashSet<String> = qa["evidence"]
                        .as_array()?
                        .iter()
                        .filter_map(|id| id.as_str())
                        .map(topic_of)
                        .filter(|topic| known.contains(topic.as_str()))
                        .collect();
                    if evidence.is_empty() {
                        return None;
                    }
                    Some(Question {
                        text: qa["question"].as_str()?.to_string(),
                        category: CATEGORY.get(category.checked_sub(1)?)?.to_string(),
                        evidence,
                    })
                })
                .collect();
            Conversation {
                id,
                turns,
                questions,
            }
        })
        .collect()
}

/// Fetches the file if it is not already cached, through `curl` for the reason
/// the other harnesses give, under a partial name renamed on success.
fn download(path: &Path) {
    if path.exists() {
        return;
    }
    let partial = path.with_extension("part");
    let status = Command::new("curl")
        .args(["-sSLf", "--max-time", "300", "-o"])
        .arg(&partial)
        .arg(DATA)
        .status();
    assert!(
        matches!(status, Ok(status) if status.success()),
        "could not fetch {DATA}\n\
         The dataset is not vendored: LOCOMO is CC BY-NC 4.0 and this fetches it\n\
         at run time. Make `curl` and raw.githubusercontent.com reachable, or\n\
         point LOCOMO_DIR at a directory holding locomo10.json."
    );
    std::fs::rename(&partial, path).expect("name the downloaded file");
}

async fn write(engine: &Engine, conversation: &Conversation) {
    for turn in &conversation.turns {
        harness::write_absent(
            engine,
            &turn.topic,
            &turn.record,
            "eng",
            "evaluation corpus",
        )
        .await;
    }
    harness::drain(engine).await;
}

/// Joins consecutive turns of each session, as the ledger arm's `pamin link`
/// does. Idempotent: an identical live claim appends nothing.
async fn link(engine: &Engine, conversation: &Conversation) -> usize {
    let pool = engine.database.pool();
    let mut ids = BTreeMap::new();
    for turn in &conversation.turns {
        let topic = pamin_store::repository::find_topic(pool, engine.project, &turn.topic)
            .await
            .expect("look for the topic")
            .unwrap_or_else(|| panic!("{} was written", turn.topic));
        ids.insert(turn.topic.as_str(), topic.id);
    }
    let claim = EdgeClaim::explicit(EdgeKind::RelatedTo);
    let edges: Vec<_> = conversation
        .turns
        .windows(2)
        .filter(|pair| pair[0].session == pair[1].session)
        .map(|pair| {
            (
                ids[pair[0].topic.as_str()],
                ids[pair[1].topic.as_str()],
                claim.clone(),
            )
        })
        .collect();
    graph::assert_edges(pool, engine.project, &edges)
        .await
        .expect("link consecutive turns");
    edges.len()
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads the dataset and model weights, and embeds the corpus"]
async fn locomo_evidence_with_and_without_links() {
    let conversations = load();
    let (named, profile) = harness::profile();
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    // [flat, linked], by category; each question in the same order on both.
    let mut scores: [BTreeMap<String, Scores>; 2] = Default::default();
    // `RERANK_RULES`: the linked project's shipped searches replayed under
    // every rule for using the reranker's scores. See `reranking`.
    let rules = std::env::var("RERANK_RULES").is_ok().then(reranking::rules);
    let mut measured: Vec<BTreeMap<String, Scores>> =
        vec![BTreeMap::new(); rules.as_ref().map_or(0, Vec::len)];
    let mut questions = 0usize;
    let mut linked_edges = 0usize;
    for conversation in &conversations {
        let mut engines = Vec::new();
        for side in ["flat", "linked"] {
            let engine = Engine::open(
                &workspace,
                &format!("locomo-{named}-{}-{side}", conversation.id.to_lowercase()),
                profile,
                VectorIndex::default(),
                Access::ReadWrite,
            )
            .await
            .expect("open the engine");
            write(&engine, conversation).await;
            engines.push(engine);
        }
        linked_edges += link(&engines[1], conversation).await;

        for question in &conversation.questions {
            questions += 1;
            if let Some(rules) = &rules {
                const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
                let tier = Rerank::default();
                let hits = engines[1]
                    .search_reranked(&question.text, WIDE, DEPTHS, tier)
                    .await
                    .expect("search");
                channels::enough_room(&hits, WIDE);
                let replayed = reranking::replay(&hits, tier);
                for ((_, rule), into) in rules.iter().zip(&mut measured) {
                    into.entry(question.category.clone()).or_default().add(
                        &replayed.order(*rule),
                        question.evidence.len(),
                        |topic| question.evidence.contains(topic),
                    );
                }
                continue;
            }
            for (side, engine) in engines.iter().enumerate() {
                let hits = engine
                    .search_reranked(&question.text, RECALL_AT as u32, DEPTHS, Rerank::default())
                    .await
                    .expect("search");
                let ranked: Vec<String> = hits.into_iter().map(|hit| hit.topic).collect();
                scores[side]
                    .entry(question.category.clone())
                    .or_default()
                    .add(&ranked, question.evidence.len(), |topic| {
                        question.evidence.contains(topic)
                    });
            }
        }
        println!(
            "  {}: {} turns, {} questions",
            conversation.id,
            conversation.turns.len(),
            conversation.questions.len()
        );
    }
    assert!(questions > 0, "no question could be graded");
    if let Some(rules) = &rules {
        reranking::report(
            &format!("rules for the reranker's scores, LOCOMO linked, {named}"),
            rules,
            reranking::shipped(rules),
            &measured,
            false,
        );
        return;
    }
    assert!(
        linked_edges > 0,
        "the linked projects have no edges to walk"
    );

    println!(
        "\n  LOCOMO evidence retrieval, {named}, shipped search, {} conversations, \
         {questions} questions, {linked_edges} links",
        conversations.len()
    );
    println!(
        "  category        queries   flat nDCG@{NDCG_AT}  recall@{RECALL_AT}   linked nDCG@{NDCG_AT}  recall@{RECALL_AT}   paired nDCG@{NDCG_AT}"
    );
    for (category, flat) in &scores[0] {
        let linked = &scores[1][category];
        println!(
            "  {category:<14}  {:>7}   {:>14.4}  {:>9.4}   {:>16.4}  {:>9.4}   {}",
            flat.queries,
            flat.mean_ndcg(),
            flat.mean_recall(),
            linked.mean_ndcg(),
            linked.mean_recall(),
            statistics::compare(&flat.per_query, &linked.per_query)
        );
    }
}
