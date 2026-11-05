//! What keeping span vectors in PostgreSQL buys: a rebuild that loads no model.
//!
//! Ignored by default: it provisions PostgreSQL and downloads the embedding
//! model. Run with `cargo test -p memory-engine --test spanvectors -- --ignored`.

#[path = "../../index/tests/support/mod.rs"]
mod support;

use memory_core::{Fusion, TopicStateId, Validity};
use memory_engine::{Depths, Engine, Owed};
use memory_index::{Access, VectorIndex};
use memory_store::{Connections, Database, Workspace, repository};

const ROOT: &str = "/notes";

const MEMORIES: [(&str, &str); 3] = [
    (
        "docs/release.md › Train",
        "The release train leaves every thursday at noon.",
    ),
    (
        "docs/oncall.md › Rota",
        "The oncall rota rotates weekly on mondays.",
    ),
    (
        "docs/backup.md › Restore",
        "Restoring a backup takes about an hour.",
    ),
];

fn settings() -> memory_engine::Settings {
    support::eval::check();
    memory_engine::Settings {
        inference: support::inference(),
        store: memory_store::StoreSettings {
            jit: support::eval::switch("PAMIN_MEMORY_EVAL_JIT").unwrap_or_default(),
            postgres_dir: support::eval::path("PAMIN_MEMORY_EVAL_POSTGRES_DIR"),
            database_url: None,
            database_connections: None,
        },
        ..memory_engine::Settings::default()
    }
}

async fn open(workspace: &Workspace, name: &str) -> Engine {
    Engine::open(
        workspace,
        name,
        VectorIndex::default(),
        Access::ReadWrite,
        &settings(),
    )
    .await
    .expect("open the engine")
}

async fn kept(engine: &Engine) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM span_vectors WHERE project_id = $1")
        .bind(engine.project.0)
        .fetch_one(engine.database.pool())
        .await
        .expect("count the kept vectors")
}

async fn found(engine: &Engine, query: &str) -> Vec<String> {
    engine
        .search_fused(query, 3, Depths::DEFAULT, Fusion::default())
        .await
        .expect("search")
        .into_iter()
        .map(|hit| hit.topic)
        .collect()
}

async fn seen(engine: &Engine, topic: &str) -> TopicStateId {
    let id: uuid::Uuid = sqlx::query_scalar(
        "SELECT current_state_id FROM topics WHERE project_id = $1 AND name = $2",
    )
    .bind(engine.project.0)
    .bind(topic)
    .fetch_one(engine.database.pool())
    .await
    .expect("a current state");
    id.into()
}

/// An index job keeps one vector per span it embeds; a rebuild of an index
/// that is gone writes them back without loading the model, and finds what
/// the index found; a rename embeds the renamed section alone.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn a_rebuild_from_kept_vectors_loads_no_model() {
    let scratch = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::at(scratch.path());
    let database = Database::open(&workspace, Connections::PerCommand, &settings().store)
        .await
        .expect("open the database");
    let name = format!("spanvectors-{}", uuid::Uuid::now_v7());
    let project = repository::ensure_project(database.pool(), &name)
        .await
        .expect("ensure project");

    let engine = open(&workspace, &name).await;
    for (topic, content) in MEMORIES {
        engine
            .remember_section(topic, content, Validity::ALWAYS, ROOT)
            .await
            .expect("ingest")
            .expect("recorded");
    }
    engine.drain_cascade(Owed::Everything).await.expect("drain");
    let documents = engine.indexed_documents().expect("count documents");
    assert!(
        documents >= MEMORIES.len() as u64,
        "the premise: {documents}"
    );
    assert_eq!(
        kept(&engine).await,
        documents as i64,
        "one vector kept per span embedded"
    );
    let before = found(&engine, "when does the release train leave").await;
    assert!(
        !before.is_empty(),
        "the premise: the search finds something"
    );
    drop(engine);

    std::fs::remove_dir_all(workspace.index_dir(project.id)).expect("remove the index");
    let engine = open(&workspace, &name).await;
    assert!(engine.needs_rebuild(), "the premise: the index is gone");
    let rebuilt = engine.reindex().await.expect("rebuild");
    assert_eq!(rebuilt.indexed, MEMORIES.len());
    assert_eq!(
        rebuilt.reused,
        MEMORIES.len(),
        "every topic from what was kept"
    );
    assert!(
        engine
            .providers()
            .iter()
            .all(|(role, _)| role != "embedding"),
        "the rebuild loaded the model: {:?}",
        engine.providers()
    );
    assert_eq!(engine.indexed_documents().expect("count"), documents);
    assert_eq!(
        found(&engine, "when does the release train leave").await,
        before
    );

    let (from, _) = MEMORIES[1];
    let renamed = "docs/oncall.md › Schedule";
    let state = seen(&engine, from).await;
    assert!(
        engine
            .rename_section(from, renamed, state)
            .await
            .expect("rename"),
        "the premise: the section was renamed"
    );
    engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain the rename");
    assert_eq!(
        kept(&engine).await,
        documents as i64 + 1,
        "the renamed section's one span is embedded again, and nothing else"
    );
    drop(engine);

    // A kept vector of another width is skipped, and the rest still serve.
    execute(
        &database,
        "UPDATE span_vectors SET vector = '\\x0100'::bytea
          WHERE (project_id, passage_hash) IN (
                SELECT project_id, passage_hash FROM span_vectors
                 WHERE project_id = $1 ORDER BY passage_hash LIMIT 2)",
        project.id,
    )
    .await;
    let rebuilt = rebuild(&workspace, &name, project.id).await;
    assert!(
        (1..MEMORIES.len()).contains(&rebuilt.reused),
        "one malformed vector costs its topic alone: {} reused",
        rebuilt.reused
    );

    // Embedding those topics again put their rows right, so the next rebuild
    // takes every topic from what was kept.
    let rebuilt = rebuild(&workspace, &name, project.id).await;
    assert_eq!(
        rebuilt.reused,
        MEMORIES.len(),
        "a rebuild that embeds a span repairs its row"
    );

    // A missing vector makes the rebuild load the model; vectors kept under a
    // plan it is not on stand in for none of the rest.
    execute(
        &database,
        "DELETE FROM span_vectors
          WHERE (project_id, passage_hash) IN (
                SELECT project_id, passage_hash FROM span_vectors
                 WHERE project_id = $1 ORDER BY passage_hash LIMIT 1)",
        project.id,
    )
    .await;
    execute(
        &database,
        "UPDATE span_vectors SET encoding = 'npu:another-plan' WHERE project_id = $1",
        project.id,
    )
    .await;
    let rebuilt = rebuild(&workspace, &name, project.id).await;
    assert_eq!(rebuilt.reused, 0, "another plan's vectors were reused");

    // Vectors of another revision of the model are not this one's.
    execute(
        &database,
        "UPDATE span_vectors SET revision = 'another/model@0' WHERE project_id = $1",
        project.id,
    )
    .await;
    let rebuilt = rebuild(&workspace, &name, project.id).await;
    assert_eq!(rebuilt.reused, 0, "another revision's vectors were reused");
}

async fn execute(database: &Database, sql: &'static str, project: memory_core::ProjectId) {
    sqlx::query(sql)
        .bind(project.0)
        .execute(database.pool())
        .await
        .expect("rewrite the kept vectors");
}

/// Rebuilds the project's index from nothing, as a node that lost it does.
async fn rebuild(
    workspace: &Workspace,
    name: &str,
    project: memory_core::ProjectId,
) -> memory_engine::Rebuilt {
    std::fs::remove_dir_all(workspace.index_dir(project)).expect("remove the index");
    let engine = open(workspace, name).await;
    let rebuilt = engine.reindex().await.expect("rebuild");
    assert_eq!(rebuilt.indexed, MEMORIES.len());
    rebuilt
}
