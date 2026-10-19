//! Whether a drain that owes everything leaves anything owed.
//!
//! Ignored by default: it provisions PostgreSQL and downloads the embedding
//! model. Run with
//! `cargo test -p pamin-engine --test drainowed -- --ignored --nocapture`.
//!
//! A resident server drains on nobody's behalf, owing only what a memory needs:
//! it applies the writes and leaves their flush, and the claims they hold, to
//! the next upkeep tick. `pamin cascade drain`, asked on the same engine in
//! between, claimed nothing -- every job was held -- and returned with those
//! jobs counted as still owed, done and waiting only to be recorded.

#[path = "../../pamin-index/tests/support/mod.rs"]
mod support;

use pamin_core::{FilterDecision, Validity};
use pamin_engine::{Engine, Owed, Write};
use pamin_index::{Access, VectorIndex};
use pamin_store::{Connections, Database, Workspace, jobs, repository};

const MEMORIES: usize = 4;

fn settings() -> pamin_engine::Settings {
    support::eval::check();
    pamin_engine::Settings {
        inference: support::inference(),
        store: pamin_store::StoreSettings {
            jit: support::eval::switch("PAMIN_EVAL_JIT").unwrap_or_default(),
            postgres_dir: support::eval::path("PAMIN_EVAL_POSTGRES_DIR"),
        },
        ..pamin_engine::Settings::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn a_drain_owing_everything_settles_what_another_round_applied() {
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    let database = Database::open(&workspace, Connections::PerCommand, &settings().store)
        .await
        .expect("open the database");
    let name = format!("drainowed-{}", uuid::Uuid::now_v7());
    let project = repository::ensure_project(database.pool(), &name)
        .await
        .expect("ensure project");
    let engine = Engine::open(
        &workspace,
        &name,
        VectorIndex::default(),
        Access::ReadWrite,
        &settings(),
    )
    .await
    .expect("open the engine");

    for memory in 0..MEMORIES {
        let content = format!("the release train for service {memory} leaves on tuesdays");
        let topic = format!("service {memory} releases");
        engine
            .write(&Write {
                topic: &topic,
                content: &content,
                content_hash: &format!("drainowed-{memory}"),
                verdict: FilterDecision::Promoted,
                reason: "drain test",
                promoted: true,
                language: Some("eng"),
                language_confidence: Some(1.0),
                observed_at: time::OffsetDateTime::now_utc(),
                validity: Validity::ALWAYS,
            })
            .await
            .expect("write");
    }

    // The server's round: applied, claims held, flush left for later.
    engine
        .drain_cascade(Owed::WhatAMemoryNeeds)
        .await
        .expect("drain what a memory needs");
    assert!(
        engine.awaiting_durability() > 0,
        "the premise: a round owing less left applied writes holding their claims"
    );
    assert!(
        jobs::any_in_flight(database.pool(), project.id)
            .await
            .expect("claimed"),
        "the premise: those claims are held"
    );

    let drained = engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain everything");
    assert_eq!(
        engine.awaiting_durability(),
        0,
        "applied writes left unflushed"
    );
    assert_eq!(
        drained.pending, 0,
        "a drain owing everything returned with work owed"
    );
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("pending"),
        0,
        "the queue still holds work"
    );
}
