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
//! **One project per conversation, searched two ways.** Each conversation is
//! written once, every turn a memory, with consecutive turns of a session
//! joined by an explicit `related_to` edge, which is what the ledger arm does
//! with `pamin link`. The topic names are dialogue ids, so mention derivation
//! finds nothing and the only edges are those links. Every question is asked
//! twice of that one project: with the shipped fusion, and with the graph
//! channel taken out, which is what a project without the links -- the flat
//! benchmark arm -- can find. Each row is a per-question comparison of the
//! same search with and without the graph -- the graph channel's worth on
//! this corpus, not a guess at it.
//!
//! One project rather than a flat one beside a linked one, because two
//! projects built apart are not the same corpus twice: each gives its topics
//! fresh ids, fusion breaks ties by topic id, and so two independently built
//! projects can order a tie differently with no graph involved. Both arms
//! here read the same index and the same ids.
//!
//! **Scored by category.** LOCOMO's categories are reported separately and
//! never summed. Adversarial questions (category 5) are left out: their
//! `evidence` names the tempting wrong-speaker turn, not a turn holding the
//! answer, because abstaining is the correct answer to them (see
//! `benchmarks/datasets/locomo.py`). Rewarding retrieval of that turn would
//! score the trap as the key.
//!
//! **Significance is tested per conversation.** Every question of one
//! conversation shares its corpus, index and graph, so the questions are not
//! independent units. The paired test runs over each conversation's mean
//! nDCG per category, which on a few conversations cannot reach a small p;
//! that is the honest reading, not a defect of the test.
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
//!
//! Projects are named after a digest of the records and links they hold, so a
//! reused `PAMIN_EVAL_HOME` never scores a corpus built by older code or data.

mod harness;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process::Command;

use pamin_core::{Channel, EdgeKind, Fusion};
use pamin_engine::{Depths, Engine};
use pamin_index::{Access, Rerank, VectorIndex};
use pamin_store::Workspace;
use pamin_store::graph::{self, EdgeClaim};
use scoring::{NDCG_AT, RECALL_AT, Scores};
use sha2::{Digest, Sha256};

/// The upstream release, pinned to a commit of the authors' repository.
const DATA: &str = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json";

/// SHA-256 of `locomo10.json` at that commit; a cached copy must match it too.
const DATA_SHA256: &str = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4";

/// What the product's `pamin search` asks for by default. A constant because
/// the rule replay sizes a constant from it; the test asserts it still equals
/// [`Depths::default`].
const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

/// LOCOMO's adversarial category, whose evidence is the trap.
const ADVERSARIAL: usize = 5;

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

/// The dialogue ids an upstream evidence string names, canonicalised.
///
/// Upstream sometimes packs several ids into one string (`D8:6; D9:17`,
/// `D9:1 D4:4 D4:6`), zero-pads a turn (`D30:05`), or puts a colon after the
/// `D` (`D:11:26`, where conversation 5 has `D11:26` and another question
/// cites it). Those are split and written as `D{session}:{turn}`; a piece
/// that is still not `D<n>:<n>`, such as a bare `D`, is returned as is and
/// fails the existence check.
fn evidence_ids(evidence: &str) -> Vec<String> {
    evidence
        .split(|c: char| c == ';' || c == ',' || c.is_whitespace())
        .filter(|piece| !piece.is_empty())
        .map(|piece| {
            piece
                .strip_prefix('D')
                .map(|rest| rest.strip_prefix(':').unwrap_or(rest))
                .and_then(|rest| rest.split_once(':'))
                .and_then(|(session, turn)| {
                    Some(format!(
                        "D{}:{}",
                        session.parse::<u32>().ok()?,
                        turn.parse::<u32>().ok()?
                    ))
                })
                .unwrap_or_else(|| piece.to_string())
        })
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
            // puts it, because two of the five categories ask when. A shared
            // image's caption goes in too, the way LOCOMO's own textual
            // evaluation renders it, because some answers are only in it.
            // The Python arms do not add it yet.
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
                    let mut record = format!("[{when}] {speaker}: {text}");
                    if let Some(caption) = turn["blip_caption"]
                        .as_str()
                        .map(str::trim)
                        .filter(|caption| !caption.is_empty())
                    {
                        record.push_str(&format!(" [shared an image: {caption}]"));
                    }
                    turns.push(Turn {
                        topic: topic_of(dia_id),
                        session: key.to_string(),
                        record,
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
                    if category == ADVERSARIAL {
                        return None;
                    }
                    // A question none of whose evidence names a turn that
                    // exists, even after `evidence_ids` recovers what it can,
                    // cannot be graded and is left out.
                    let evidence: HashSet<String> = qa["evidence"]
                        .as_array()?
                        .iter()
                        .filter_map(|id| id.as_str())
                        .flat_map(evidence_ids)
                        .map(|id| topic_of(&id))
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

/// What a conversation's projects hold: every topic, record and session, in
/// order, which also fixes the links `link` derives from them.
fn corpus_digest(conversation: &Conversation) -> String {
    let mut hash = Sha256::new();
    for turn in &conversation.turns {
        for field in [&turn.topic, &turn.session, &turn.record] {
            hash.update(field.len().to_le_bytes());
            hash.update(field.as_bytes());
        }
    }
    format!("{:x}", hash.finalize())[..12].to_string()
}

/// Fetches the file if it is not already cached, through `curl` for the reason
/// the other harnesses give, under a partial name renamed on success, and
/// checks it is the pinned release either way.
fn download(path: &Path) {
    if !path.exists() {
        fetch(path);
    }
    let actual = format!(
        "{:x}",
        Sha256::digest(
            std::fs::read(path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()))
        )
    );
    assert_eq!(
        actual,
        DATA_SHA256,
        "{} is not the pinned LOCOMO release; remove it to fetch {DATA}",
        path.display()
    );
}

fn fetch(path: &Path) {
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
    let shipped = Depths::default();
    assert_eq!(
        (DEPTHS.channel, DEPTHS.graph),
        (shipped.channel, shipped.graph),
        "DEPTHS must be what `pamin search` ships"
    );
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

    // [without the graph, with it], by category; each question in the same
    // order on both.
    let mut scores: [BTreeMap<String, Scores>; 2] = Default::default();
    // [without the graph, with it], by category: each conversation's mean
    // nDCG, the unit the paired test is run over.
    let mut by_conversation: [BTreeMap<String, Vec<f64>>; 2] = Default::default();
    let mut questions = 0usize;
    let mut linked_edges = 0usize;
    for conversation in &conversations {
        let digest = corpus_digest(conversation);
        let engine = Engine::open(
            &workspace,
            &format!(
                "locomo-{named}-{}-linked-{digest}",
                conversation.id.to_lowercase()
            ),
            profile,
            VectorIndex::default(),
            Access::ReadWrite,
        )
        .await
        .expect("open the engine");
        write(&engine, conversation).await;
        linked_edges += link(&engine, conversation).await;

        let before: [BTreeMap<String, usize>; 2] = std::array::from_fn(|side| {
            scores[side]
                .iter()
                .map(|(category, scored)| (category.clone(), scored.per_query.len()))
                .collect()
        });
        for question in &conversation.questions {
            questions += 1;
            let fusions = [
                Fusion::default().without(Channel::Graph),
                Fusion::default(),
            ];
            for (side, fusion) in fusions.into_iter().enumerate() {
                let hits = engine
                    .search_reranked_with(
                        &question.text,
                        RECALL_AT as u32,
                        DEPTHS,
                        Rerank::default(),
                        fusion,
                    )
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
        for (side, scored_side) in scores.iter().enumerate() {
            for (category, scored) in scored_side {
                let from = before[side].get(category).copied().unwrap_or(0);
                let this = &scored.per_query[from..];
                if !this.is_empty() {
                    by_conversation[side]
                        .entry(category.clone())
                        .or_default()
                        .push(this.iter().sum::<f64>() / this.len() as f64);
                }
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
    assert!(linked_edges > 0, "the projects have no edges to walk");

    println!(
        "\n  LOCOMO evidence retrieval, {named}, shipped search, {} conversations, \
         {questions} questions, {linked_edges} links",
        conversations.len()
    );
    println!(
        "  category        queries   no graph nDCG@{NDCG_AT}  recall@{RECALL_AT}   graph nDCG@{NDCG_AT}  recall@{RECALL_AT}   paired nDCG@{NDCG_AT} by conversation"
    );
    for (category, without) in &scores[0] {
        let with = &scores[1][category];
        println!(
            "  {category:<14}  {:>7}   {:>18.4}  {:>9.4}   {:>15.4}  {:>9.4}   {}",
            without.queries,
            without.mean_ndcg(),
            without.mean_recall(),
            with.mean_ndcg(),
            with.mean_recall(),
            statistics::compare(&by_conversation[0][category], &by_conversation[1][category])
        );
    }
}

/// Every malformed id the pinned file holds that names a real turn is
/// recovered, and one that names none is left to fail the existence check.
#[test]
fn evidence_ids_recover_upstreams_malformed_ids() {
    assert_eq!(evidence_ids("D8:6; D9:17"), ["D8:6", "D9:17"]);
    assert_eq!(evidence_ids("D9:1 D4:4 D4:6"), ["D9:1", "D4:4", "D4:6"]);
    assert_eq!(evidence_ids("D30:05"), ["D30:5"]);
    assert_eq!(evidence_ids("D:11:26"), ["D11:26"]);
    assert_eq!(evidence_ids("D"), ["D"]);
}
