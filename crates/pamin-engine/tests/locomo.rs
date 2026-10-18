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
//! channel taken out and not walked (graph depth 0, so the reranker is shown
//! no graph seed either), which is what a project without the links -- the
//! flat benchmark arm -- can find. Each row is a per-question comparison of the
//! same search with and without the graph -- the graph channel's worth on
//! this corpus, not a guess at it.
//!
//! One project rather than a flat one beside a linked one, because two
//! projects built apart are not the same corpus twice: each gives its topics
//! fresh ids, fusion breaks ties by topic id, and so two independently built
//! projects can order a tie differently with no graph involved. Both arms
//! here read the same index and the same ids.
//!
//! **What it calls, and at what limit.** `Engine::search_reranked_with`, the
//! function `pamin search` reaches, at the shipped depths, reranker tier and
//! fusion, but asked for [`RECALL_AT`] results rather than the CLI's default
//! `--limit 5`, so that recall at that depth can be read. The limit is more
//! than a cut: the rerank pass runs only when a candidate it may move sits
//! above the limit, so the first five here need not be the five `pamin
//! search` prints. This is the shipped search path measured wider than a user
//! sees it, and says nothing on its own about the visible five.
//!
//! **Scored by category.** LOCOMO's categories are reported separately and
//! never summed. Adversarial questions (category 5) are left out: their
//! `evidence` names the tempting wrong-speaker turn, not a turn holding the
//! answer, because abstaining is the correct answer to them (see
//! `benchmarks/datasets/locomo.py`). Rewarding retrieval of that turn would
//! score the trap as the key.
//!
//! Questions whose cited evidence the audit replaces are left out too. The
//! locomo-audit project checked every non-adversarial question of this
//! pinned file and, for each error it finds, gives the evidence it cites and
//! the evidence that is correct. Where those two differ as sets of dialogue
//! ids, the cited turns are the wrong key for retrieval: grading against
//! them would reward retrieving the wrong turn, or penalise a right one. So
//! every such question is left out, whatever the error's type -- 94 of them:
//! 55 `WRONG_CITATION`, 18 `HALLUCINATION`, 7 `ATTRIBUTION_ERROR`, 7
//! `AMBIGUOUS`, 5 `TEMPORAL_ERROR` and 2 `INCOMPLETE`. The audit's other 62
//! errors keep the cited evidence -- they fault the answer or the question,
//! not which turn holds it, or, in two, only the way an id is written, which
//! [`evidence_ids`] already recovers -- so those questions stay. Its list is
//! fetched pinned to a commit and checked against its SHA-256, and each
//! entry must name the question the dataset asks at that position. Every
//! excluded question is recorded in the run's shape with its reason.
//!
//! **Significance is tested per conversation.** Every question of one
//! conversation shares its corpus, index and graph, so the questions are not
//! independent units. The paired test runs over each conversation's mean
//! nDCG per category, which on a few conversations cannot reach a small p;
//! that is the honest reading, not a defect of the test.
//!
//! **Licence.** LOCOMO is CC BY-NC 4.0 (Maharana et al., ACL 2024). It is
//! fetched at run time from the authors' repository, not vendored. So is the
//! audit's list (`dial481/locomo-audit`), under the same licence.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p pamin-engine --test locomo -- --ignored --nocapture
//! ```
//!
//! | variable | what it does |
//! |---|---|
//! | `LOCOMO_DIR` | where `locomo10.json` and `locomo-audit-errors.json` are cached, instead of fetching them |
//! | `LOCOMO_CONVERSATIONS` | read only the first this many conversations |
//! | `PAMIN_PROFILE` | which embedding profile, default `accuracy` |
//!
//! Any other `PAMIN_` variable fails the run before it starts, unless it
//! only says where something is or what to log, or chooses how a model
//! executes or an index is built, which the record names
//! (`harness::shape::shipped_path_only`): the rest are settings this
//! harness does not honour as `pamin search` would, or sweeps.
//!
//! Projects are named after a digest of the records and links they hold, and
//! of the execution plan and model revision that embedded them, so a reused
//! `PAMIN_EVAL_HOME` never scores a corpus built by older code or data, or
//! embedded on another provider, plan or by other weights. Each conversation's
//! index is also claimed through `harness::index` before it is written, so a
//! reused index built under other settings is reported rather than trusted.

mod channels;
mod harness;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::process::Command;

use pamin_core::{Channel, EdgeKind, Fusion};
use pamin_engine::{Depths, Engine, Models};
use pamin_index::{Access, Profile, Rerank, VectorIndex};
use pamin_store::graph::{self, EdgeClaim};
use pamin_store::{Connections, Database, Workspace};
use scoring::{NDCG_AT, RECALL_AT, Scores};
use sha2::{Digest, Sha256};

/// The upstream release, pinned to a commit of the authors' repository.
const DATA: &str = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json";

/// SHA-256 of `locomo10.json` at that commit; a cached copy must match it too.
const DATA_SHA256: &str = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4";

/// locomo-audit's list of this file's annotation errors, pinned to a commit.
const AUDIT: &str = "https://raw.githubusercontent.com/dial481/locomo-audit/9493fb4b4af4256ed17a18e8fd0b3cfdeec29539/errors.json";

/// SHA-256 of `errors.json` at that commit.
const AUDIT_SHA256: &str = "f298ef46263fada20688a1d672be37d4f488c6da3b19343eea55ae5c1b5fe55e";

/// Per error type, how many questions of the whole file the audit gives
/// other evidence than the dataset cites, and the reason the run's shape
/// records for them. Every audited question whose evidence changes is left
/// out; a type missing here that changes evidence fails the load.
const MISCITED: [(&str, usize, &str); 6] = [
    (
        "WRONG_CITATION",
        55,
        "miscited: locomo-audit finds its evidence does not hold the answer",
    ),
    (
        "HALLUCINATION",
        18,
        "re-cited: locomo-audit corrects a hallucinated answer and its evidence",
    ),
    (
        "ATTRIBUTION_ERROR",
        7,
        "re-cited: locomo-audit corrects a misattributed answer and its evidence",
    ),
    (
        "AMBIGUOUS",
        7,
        "re-cited: locomo-audit finds the question ambiguous and its evidence other",
    ),
    (
        "TEMPORAL_ERROR",
        5,
        "re-cited: locomo-audit corrects a misdated answer and its evidence",
    ),
    (
        "INCOMPLETE",
        2,
        "re-cited: locomo-audit finds its answer and evidence leave out turns",
    ),
];

/// How many of the audit's errors keep the evidence the dataset cites, and
/// so leave their question scored.
const RECITED_UNCHANGED: usize = 62;

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
    /// `<sample id>:qa<position in its qa list>`, as the shape records it.
    id: String,
    text: String,
    category: String,
    evidence: HashSet<String>,
}

/// A question the dataset asks and this harness does not score.
struct Excluded {
    id: String,
    category: String,
    reason: &'static str,
}

struct Conversation {
    id: String,
    turns: Vec<Turn>,
    questions: Vec<Question>,
    excluded: Vec<Excluded>,
}

/// What a run read, and how much of the file it is.
struct Loaded {
    conversations: Vec<Conversation>,
    selection: String,
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

fn load() -> Loaded {
    let dir = harness::dataset_dir("LOCOMO_DIR", "locomo");
    std::fs::create_dir_all(&dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
    let path = dir.join("locomo10.json");
    download(&path, DATA, DATA_SHA256);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let samples: Vec<serde_json::Value> = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));
    let audit = dir.join("locomo-audit-errors.json");
    download(&audit, AUDIT, AUDIT_SHA256);
    let miscited = miscited(&audit, &samples);

    let wanted = std::env::var("LOCOMO_CONVERSATIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(samples.len());
    let conversations = samples
        .iter()
        .take(wanted)
        .enumerate()
        .map(|(position, sample)| {
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
            let mut questions = Vec::new();
            let mut excluded = Vec::new();
            for (at, qa) in qa_of(sample).iter().enumerate() {
                let question_id = format!("{id}:qa{at}");
                let number = qa["category"].as_u64().expect("a question's category") as usize;
                let category = CATEGORY
                    .get(number.wrapping_sub(1))
                    .unwrap_or_else(|| panic!("{question_id}: unknown category {number}"))
                    .to_string();
                let mut exclude = |reason| {
                    excluded.push(Excluded {
                        id: question_id.clone(),
                        category: category.clone(),
                        reason,
                    });
                };
                if number == ADVERSARIAL {
                    exclude("adversarial: its evidence is the trap, and abstaining is the answer");
                    continue;
                }
                if let Some(reason) = miscited.get(&(position, at)) {
                    exclude(*reason);
                    continue;
                }
                // A question none of whose evidence names a turn that
                // exists, even after `evidence_ids` recovers what it can,
                // cannot be graded.
                let evidence: HashSet<String> = qa["evidence"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|id| id.as_str())
                    .flat_map(evidence_ids)
                    .map(|id| topic_of(&id))
                    .filter(|topic| known.contains(topic.as_str()))
                    .collect();
                if evidence.is_empty() {
                    exclude("no evidence id names a turn of this conversation");
                    continue;
                }
                questions.push(Question {
                    id: question_id.clone(),
                    text: qa["question"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{question_id}: no question"))
                        .to_string(),
                    category,
                    evidence,
                });
            }
            Conversation {
                id,
                turns,
                questions,
                excluded,
            }
        })
        .collect();
    Loaded {
        conversations,
        selection: format!(
            "the first {} of {} conversations in file order, every question",
            wanted.min(samples.len()),
            samples.len()
        ),
    }
}

/// The questions locomo-audit's list at `path` gives other evidence than the
/// dataset cites -- its `correct_evidence` and `cited_evidence` differ as
/// sets of dialogue ids, read through [`evidence_ids`] -- as (conversation,
/// question) positions in the file with the reason, by error type, they are
/// left out. Each entry must name the question `samples` asks at its
/// position, so a list that does not describe this file fails rather than
/// excluding the wrong questions.
fn miscited(path: &Path, samples: &[serde_json::Value]) -> HashMap<(usize, usize), &'static str> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let errors: Vec<serde_json::Value> = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));
    let ids = |error: &serde_json::Value, field: &str| -> HashSet<String> {
        error[field]
            .as_array()
            .unwrap_or_else(|| panic!("audit entry {} has no {field}", error["question_id"]))
            .iter()
            .map(|id| id.as_str().expect("a dialogue id"))
            .flat_map(evidence_ids)
            .collect()
    };
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut unchanged = 0;
    let found: HashMap<(usize, usize), &'static str> = errors
        .iter()
        .filter_map(|error| {
            let id = error["question_id"].as_str().expect("an audit entry's id");
            let (conversation, at) = id
                .strip_prefix("locomo_")
                .and_then(|rest| rest.split_once("_qa"))
                .and_then(|(conversation, at)| {
                    Some((
                        conversation.parse::<usize>().ok()?,
                        at.parse::<usize>().ok()?,
                    ))
                })
                .unwrap_or_else(|| panic!("unreadable audit id {id}"));
            let asked = samples
                .get(conversation)
                .and_then(|sample| qa_of(sample).get(at))
                .and_then(|qa| qa["question"].as_str());
            assert_eq!(
                asked,
                error["question"].as_str(),
                "audit entry {id} is not the question the dataset asks there"
            );
            if ids(error, "cited_evidence") == ids(error, "correct_evidence") {
                unchanged += 1;
                return None;
            }
            let kind = error["error_type"].as_str().expect("an audit entry's type");
            let (kind, _, reason) = MISCITED
                .iter()
                .find(|(name, _, _)| *name == kind)
                .unwrap_or_else(|| panic!("audit entry {id} changes evidence under {kind}"));
            *counts.entry(kind).or_default() += 1;
            Some(((conversation, at), *reason))
        })
        .collect();
    let expected: BTreeMap<&str, usize> = MISCITED
        .iter()
        .map(|(kind, count, _)| (*kind, *count))
        .collect();
    assert_eq!(
        counts,
        expected,
        "{} is not the pinned audit",
        path.display()
    );
    assert_eq!(
        unchanged,
        RECITED_UNCHANGED,
        "{} is not the pinned audit",
        path.display()
    );
    assert_eq!(
        found.len(),
        expected.values().sum::<usize>(),
        "{} names a question twice",
        path.display()
    );
    found
}

/// A conversation's questions, in file order.
fn qa_of(sample: &serde_json::Value) -> &[serde_json::Value] {
    sample["qa"].as_array().expect("qa")
}

/// What a run asks: every turn of every conversation, every question with
/// its evidence, and every question left out with why. See
/// [`harness::shape`].
fn shape(loaded: &Loaded) -> harness::shape::Shape {
    let mut shape = harness::shape::Shape::new("locomo", loaded.selection.clone(), None);
    for conversation in &loaded.conversations {
        for turn in &conversation.turns {
            shape.document(&[&conversation.id, &turn.topic, &turn.session, &turn.record]);
        }
        for question in &conversation.questions {
            let mut evidence: Vec<&str> = question.evidence.iter().map(String::as_str).collect();
            evidence.sort_unstable();
            let mut fields = vec![question.text.as_str()];
            fields.extend(evidence);
            shape.query(&[&question.category], &question.id, &fields);
        }
        for excluded in &conversation.excluded {
            shape.exclude(&[&excluded.category], &excluded.id, excluded.reason);
        }
    }
    shape
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

/// Fetches `url` to `path` if it is not already cached, through `curl` for
/// the reason the other harnesses give, under a partial name renamed on
/// success, and checks it is the pinned release either way.
fn download(path: &Path, url: &str, sha256: &str) {
    if !path.exists() {
        fetch(path, url);
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
        sha256,
        "{} is not the pinned release; remove it to fetch {url}",
        path.display()
    );
}

fn fetch(path: &Path, url: &str) {
    let partial = path.with_extension("part");
    let status = Command::new("curl")
        .args(["-sSLf", "--max-time", "300", "-o"])
        .arg(&partial)
        .arg(url)
        .status();
    assert!(
        matches!(status, Ok(status) if status.success()),
        "could not fetch {url}\n\
         The dataset and its audit are not vendored: both are CC BY-NC 4.0 and\n\
         this fetches them at run time. Make `curl` and\n\
         raw.githubusercontent.com reachable, or point LOCOMO_DIR at a\n\
         directory holding locomo10.json and locomo-audit-errors.json."
    );
    std::fs::rename(&partial, path).expect("name the downloaded file");
}

/// What embeds this run's documents, as a project name carries it: the first
/// twelve hex digits of a SHA-256 over each embedding plan's role, its full
/// execution plan as the shape records it (`harness::shape::provider`) and
/// its model revision (`repository@commit`).
///
/// A reused `PAMIN_EVAL_HOME` keeps the vectors an earlier run wrote, and
/// `write_absent` skips every topic it finds. Without this, a run on another
/// provider -- `PAMIN_DEVICE=auto` on another machine -- or after the hub's
/// `main` moved would search old document vectors with new query vectors.
/// The models load lazily, so an empty project is searched once to load the
/// embedder before any project is named.
async fn embedded_by(
    database: &Database,
    models: &Models,
    workspace: &Workspace,
    named: &str,
    profile: Profile,
) -> String {
    let probe = Engine::attached(
        database.clone(),
        models,
        workspace,
        &format!("locomo-{named}-models"),
        profile,
        VectorIndex::default(),
        Access::ReadWrite,
    )
    .await
    .expect("open the model probe");
    // An empty project still embeds the query, which loads the embedder.
    probe
        .search_fused("which model", 1, DEPTHS, Fusion::default())
        .await
        .expect("load the embedder");
    let mut plans: Vec<String> = probe
        .providers()
        .into_iter()
        .filter(|(role, _)| role.starts_with("embedding"))
        .map(|(role, plan)| {
            format!(
                "{role}={}@{}",
                harness::shape::provider(&plan),
                plan.revision
            )
        })
        .collect();
    plans.sort();
    plans.dedup();
    assert!(!plans.is_empty(), "the probe search loaded no embedder");
    for plan in &plans {
        println!("  embedded by {plan}");
    }
    format!("{:x}", Sha256::digest(plans.join("\n").as_bytes()))[..12].to_string()
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
    harness::shape::shipped_path_only();
    let loaded = load();
    let (named, profile) = harness::profile();
    // Published before anything is indexed, so a run that dies still said
    // what it was asked.
    let mut shape = shape(&loaded);
    shape.publish("locomo_evidence_with_and_without_links", &named);
    // Before the probe loads the first model, so the record says whether
    // this run fetched or prepared models, and a cold run is not named as a
    // warm one.
    shape.observe_models();
    let conversations = &loaded.conversations;
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    // One database and one model registry for every conversation, as a
    // resident server holds them: `Engine::open` makes a registry of its own,
    // which reloaded the weights for each conversation.
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");
    let models = Models::in_workspace(&workspace);
    let by = embedded_by(&database, &models, &workspace, &named, profile).await;

    // [without the graph, with it], by category; each question in the same
    // order on both.
    let mut scores: [BTreeMap<String, Scores>; 2] = Default::default();
    // [without the graph, with it], by category: each conversation's mean
    // nDCG, the unit the paired test is run over.
    let mut by_conversation: [BTreeMap<String, Vec<f64>>; 2] = Default::default();
    // `RERANK_RULES`: the shipped searches, with the graph, replayed under
    // every rule for using the reranker's scores. See `reranking`.
    let mut rules = std::env::var("RERANK_RULES")
        .is_ok()
        .then(reranking::Priced::default);
    let mut questions = 0usize;
    let mut linked_edges = 0usize;
    const TABLES: [&str; 2] = ["without the graph", "with the graph"];
    // A rules run scores only the tables `reranking::Priced` declares.
    if rules.is_none() {
        harness::shape::tables(TABLES);
    }
    for conversation in conversations {
        let digest = corpus_digest(conversation);
        let engine = Engine::attached(
            database.clone(),
            &models,
            &workspace,
            &format!(
                "locomo-{named}-{}-{digest}-{by}",
                conversation.id.to_lowercase()
            ),
            profile,
            VectorIndex::default(),
            Access::ReadWrite,
        )
        .await
        .expect("open the engine");
        // Claimed before writing, so a reused project's index is checked
        // against the settings and plans that built it, not trusted.
        harness::index(
            &mut shape,
            &conversation.id,
            &engine,
            workspace.root(),
            conversation.turns.len(),
        )
        .await;
        write(&engine, conversation).await;
        // `write_absent` skips a topic the ledger has, and `drain` sees
        // nothing owed, even when the index beside a reused project was
        // removed or half rebuilt; so count what the index holds.
        assert_eq!(
            engine.indexed_documents().expect("count the documents") as usize,
            conversation.turns.len(),
            "{}: the index does not hold every turn",
            conversation.id
        );
        linked_edges += link(&engine, conversation).await;
        shape.ran_on(harness::shape::engines(&[("primary", &engine)]));

        let before: [BTreeMap<String, usize>; 2] = std::array::from_fn(|side| {
            scores[side]
                .iter()
                .map(|(category, scored)| (category.clone(), scored.per_query.len()))
                .collect()
        });
        for question in &conversation.questions {
            questions += 1;
            if let Some(rules) = &mut rules {
                const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
                let tier = Rerank::default();
                let hits = engine
                    .search_reranked(&question.text, WIDE, DEPTHS, tier)
                    .await
                    .expect("search");
                channels::enough_room(&hits, WIDE);
                let replayed = reranking::replay(&hits, tier);
                rules.observe(
                    &engine,
                    &question.category,
                    &question.text,
                    &hits,
                    &replayed,
                    // The conversation's questions share its corpus, index
                    // and graph: one unit for the folds and the paired test.
                    Some(conversation.id.as_str()),
                    |into, ranking| {
                        harness::shape::tally(&question.category, &question.id);
                        into.entry(question.category.clone()).or_default().add(
                            ranking,
                            question.evidence.len(),
                            |topic| question.evidence.contains(topic),
                        );
                    },
                );
                continue;
            }
            // Without the graph, nothing walks it: a zero fusion weight alone
            // still recalls the graph and shows the reranker each reached
            // candidate's seed, so the links would still move the ranking.
            let arms = [
                (
                    Depths { graph: 0, ..DEPTHS },
                    Fusion::default().without(Channel::Graph),
                ),
                (DEPTHS, Fusion::default()),
            ];
            for (side, (depths, fusion)) in arms.into_iter().enumerate() {
                harness::shape::table(TABLES[side]);
                let hits = engine
                    .search_reranked_with(
                        &question.text,
                        RECALL_AT as u32,
                        depths,
                        Rerank::default(),
                        fusion,
                    )
                    .await
                    .expect("search");
                assert!(
                    side == 1 || hits.iter().all(|hit| hit.seed.is_none()),
                    "{}: the arm without the graph still used it",
                    question.id
                );
                let ranked: Vec<String> = hits.into_iter().map(|hit| hit.topic).collect();
                harness::shape::tally(&question.category, &question.id);
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
            "  {}: {} turns, {} questions, {} excluded",
            conversation.id,
            conversation.turns.len(),
            conversation.questions.len(),
            conversation.excluded.len()
        );
    }
    assert!(questions > 0, "no question could be graded");
    // Before either report: without edges the graph rules price a no-graph run.
    assert!(linked_edges > 0, "the projects have no edges to walk");
    if let Some(rules) = &mut rules {
        rules.report(&format!(
            "rules for the reranker's scores, LOCOMO with the graph, {named}"
        ));
        return;
    }

    println!(
        "\n  LOCOMO evidence retrieval, {named}, the shipped search path at limit {RECALL_AT} \
         (`pamin search` shows 5), {} conversations, {questions} questions, {linked_edges} links",
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
