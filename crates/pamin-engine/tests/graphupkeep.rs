//! Whether a drain leaves a vector graph behind.
//!
//! Ignored by default: it provisions PostgreSQL and downloads the embedding
//! model. Run with
//! `cargo test -p pamin-engine --test graphupkeep -- --ignored --nocapture`.
//!
//! `segment_documents` states the maintenance rule -- "whenever a segment has
//! sealed without a graph" -- and for a release the cascade asked it with the
//! wrong instrument. The only condition that could queue `optimize` was the
//! file budget, which was sound while every write flushed the index and files
//! grew about two a write. Once a write left the flush to the server, three
//! thousand writes left 136 files against a budget of 256, so the trigger
//! moved from every sixty writes to roughly every five and a half thousand and
//! a project below that had no graph at all. Nothing reported it: an
//! exhaustive scan of the write buffer returns the right neighbours, only
//! slower, so recall was right and the structure was absent.
//!
//! Two consequences make this worth a standing test rather than a fix and a
//! note. The published latency, throughput and resident-memory figures were
//! taken on corpora built by `import` and `reindex`, and `reindex` calls
//! `optimize` outright -- so the benchmarks held a graph that a user's
//! workspace did not, and the two were measuring different objects. And this
//! is the second time this project has shipped a vector channel with no graph
//! under it, which is the argument for asserting the structure rather than the
//! recall.
//!
//! The budget is lowered through `PAMIN_UNINDEXED_BUDGET` rather than by
//! writing twenty-five thousand memories. That is the same trade as the
//! reranker's sweep knobs: what is being asserted is that a drain acts on the
//! condition, not what the shipped number is.

use pamin_core::{FilterDecision, Validity};
use pamin_engine::{Engine, Owed, Write};
use pamin_index::{Access, Profile};
use pamin_store::{Connections, Database, Workspace, repository};

/// The default, because a graph built for a narrower profile is not the one
/// that ships.
const PROFILE: &str = "accuracy";

/// Enough to clear the lowered budget several times over, and few enough that
/// the embedding cost is a handful of forward passes.
const MEMORIES: usize = 12;

/// Low enough that twelve memories trip it, high enough that the first few do
/// not, so the "before" half of the assertion has something to observe.
const BUDGET: usize = 5;

fn request<'a>(topic: &'a str, content: &'a str, hash: &'a str) -> Write<'a> {
    Write {
        topic,
        content,
        content_hash: hash,
        verdict: FilterDecision::Promoted,
        reason: "graph upkeep test",
        promoted: true,
        language: Some("eng"),
        language_confidence: Some(1.0),
        observed_at: time::OffsetDateTime::now_utc(),
        validity: Validity::ALWAYS,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn a_drain_leaves_a_graph_over_what_it_wrote() {
    // SAFETY: set before any engine exists, and this test binary holds one
    // test, so nothing else in the process is reading it concurrently.
    unsafe { std::env::set_var("PAMIN_UNINDEXED_BUDGET", BUDGET.to_string()) };

    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    let profile = Profile::parse(PROFILE).expect("a known profile");
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    // A name nothing else uses, so a rerun builds a fresh projection rather
    // than reopening one a previous run already optimized.
    let name = format!("graphupkeep-{}", uuid::Uuid::new_v4());
    repository::ensure_project(database.pool(), &name)
        .await
        .expect("ensure project");

    let engine = Engine::open(&workspace, &name, profile, Access::ReadWrite)
        .await
        .expect("open the engine");

    for memory in 0..MEMORIES {
        let content = format!(
            "the deployment pipeline for service {memory} reviews its own \
             rollout before promoting a build"
        );
        let topic = format!("service {memory} rollout");
        engine
            .write(&request(&topic, &content, &format!("graphupkeep-{memory}")))
            .await
            .expect("write");
    }

    let drained = engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain the cascade");
    assert!(
        drained.completed > 0,
        "a drain that settled nothing cannot have queued maintenance"
    );

    // The premise, and it is not the one this test was first written with.
    // An *empty* projection reports a completeness of 1.0 -- everything it
    // holds is indexed, and it holds nothing -- so asserting 1.0 on its own
    // passes on a drain that indexed nothing at all, and asserting 0.0 on a
    // fresh project fails for that same reason. What has to be true first is
    // that the documents are in there.
    assert_eq!(
        engine
            .indexed_documents()
            .expect("documents in the projection"),
        MEMORIES as u64,
        "the drain did not index what it wrote, so the completeness below is \
         a statement about an empty index"
    );

    assert_eq!(
        engine
            .vector_index_completeness()
            .expect("completeness after the drain"),
        1.0,
        "the drain left documents outside the vector graph, so the vector \
         channel is answering from an exhaustive scan"
    );
}
