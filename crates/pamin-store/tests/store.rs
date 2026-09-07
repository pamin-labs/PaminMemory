//! Exercises the store against a real embedded PostgreSQL.
//!
//! Ignored by default: the first run downloads and installs a PostgreSQL
//! distribution, which is too slow and too network-dependent for the ordinary
//! test loop. Run with `cargo test -p pamin-store -- --ignored`.
//!
//! Everything lives in one test so a single cluster is installed, started, and
//! stopped. Splitting it across tests would install PostgreSQL once per
//! temporary workspace.

use pamin_core::{FilterDecision, SourceKind, VersionOffset, resolve};
use pamin_store::{Database, Workspace, repository};
use time::OffsetDateTime;

#[tokio::test]
#[ignore = "downloads and starts a real postgres cluster"]
async fn the_ledger_holds_its_promises() {
    let dir = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::at(dir.path());

    let mut database = Database::open(&workspace).await.expect("open workspace");

    migrations_create_every_table(&database).await;
    reopening_reuses_the_running_server(&workspace).await;
    appending_versions_builds_a_supersession_chain(&mut database).await;
    soft_deleting_the_current_version_promotes_its_predecessor(&mut database).await;
    filtered_evidence_is_still_stored(&mut database).await;

    drop(database);
    pamin_store::database::stop(&workspace)
        .await
        .expect("stop server");
}

async fn migrations_create_every_table(database: &Database) {
    for table in [
        "projects",
        "sources",
        "source_versions",
        "source_spans",
        "topics",
        "topic_states",
        "index_jobs",
    ] {
        let row = database
            .client()
            .query_one(&format!("SELECT count(*) FROM {table}"), &[])
            .await
            .unwrap_or_else(|error| panic!("querying {table}: {error}"));
        let count: i64 = row.get(0);
        assert_eq!(count, 0, "{table} should start empty");
    }
}

async fn reopening_reuses_the_running_server(workspace: &Workspace) {
    // Must not start a second cluster, and must not fail re-applying migrations.
    let reopened = Database::open(workspace).await.expect("reopen workspace");
    drop(reopened);
}

/// Writes evidence, a span over it, and a topic state derived from that span.
async fn write_state(
    database: &mut Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
    locator: &str,
    content: &str,
) -> pamin_core::TopicState {
    let source = repository::ensure_source(database.client(), project, SourceKind::Manual, locator)
        .await
        .expect("ensure source");
    let version = repository::append_source_version(
        database.client(),
        project,
        source,
        content,
        "hash",
        FilterDecision::Promoted,
        "test fixture",
    )
    .await
    .expect("append source version");
    let span = repository::append_source_span(
        database.client(),
        project,
        version.id,
        0,
        content.len() as u32,
        None,
        None,
    )
    .await
    .expect("append span");

    repository::append_topic_state(
        database.client_mut(),
        project,
        topic,
        content,
        span.id,
        OffsetDateTime::now_utc(),
    )
    .await
    .expect("append topic state")
}

async fn appending_versions_builds_a_supersession_chain(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "ledger")
        .await
        .expect("ensure project");
    let topic = repository::ensure_topic(database.client(), project.id, "deployment_pipeline")
        .await
        .expect("ensure topic");

    let first = write_state(database, project.id, topic.id, "note-1", "deploys via make").await;
    let second = write_state(database, project.id, topic.id, "note-2", "deploys via ci").await;

    assert_eq!(first.version, 1);
    assert_eq!(second.version, 2);
    assert_eq!(
        second.supersedes,
        Some(first.id),
        "a new version links back to the one it replaced"
    );

    let versions = repository::topic_versions(database.client(), topic.id)
        .await
        .expect("versions");
    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    assert_eq!(latest.version, 2);
    assert!(latest.is_current);

    let previous = resolve(&versions, VersionOffset(1)).expect("previous");
    assert_eq!(previous.version, 1);
    assert!(!previous.is_current);

    // Past the oldest, resolution clamps and says how far it actually reached.
    let clamped = resolve(&versions, VersionOffset(9)).expect("clamped");
    assert_eq!(clamped.version, 1);
    assert_eq!(clamped.actual_offset, VersionOffset(1));

    let loaded = repository::topic_state(database.client(), topic.id, 1)
        .await
        .expect("load state")
        .expect("state exists");
    assert_eq!(loaded.content, "deploys via make");
}

async fn soft_deleting_the_current_version_promotes_its_predecessor(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "ledger")
        .await
        .expect("ensure project");
    let topic = repository::find_topic(database.client(), project.id, "deployment_pipeline")
        .await
        .expect("find topic")
        .expect("topic exists");

    let deleted = repository::soft_delete_topic_state(database.client(), topic.id, 2)
        .await
        .expect("soft delete");
    assert!(deleted);

    let versions = repository::topic_versions(database.client(), topic.id)
        .await
        .expect("versions");
    assert_eq!(versions, vec![1], "deleted versions leave the live set");

    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    assert_eq!(latest.version, 1);
    assert!(latest.is_current, "the predecessor becomes current");

    // The row itself survives, so history and audit still reach it.
    let still_there = repository::topic_state(database.client(), topic.id, 2)
        .await
        .expect("load deleted state")
        .expect("deleted state is still stored");
    assert!(still_there.deleted_at.is_some());
    assert_eq!(still_there.content, "deploys via ci");

    // A new append continues the numbering rather than reusing the freed one.
    let next = write_state(database, project.id, topic.id, "note-3", "deploys via cd").await;
    assert_eq!(next.version, 3, "version numbers are never reused");
}

async fn filtered_evidence_is_still_stored(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "ledger")
        .await
        .expect("ensure project");
    let source = repository::ensure_source(
        database.client(),
        project.id,
        SourceKind::Manual,
        "noise-source",
    )
    .await
    .expect("ensure source");

    repository::append_source_version(
        database.client(),
        project.id,
        source,
        "ok",
        "hash",
        FilterDecision::Filtered,
        "no durable claim",
    )
    .await
    .expect("append filtered evidence");

    let stored = repository::latest_source_version(database.client(), source)
        .await
        .expect("read back")
        .expect("evidence exists despite being filtered");

    assert_eq!(stored.filter_decision, FilterDecision::Filtered);
    assert_eq!(stored.filter_reason, "no durable claim");
    assert_eq!(
        stored.content, "ok",
        "filtering gates promotion, never persistence"
    );
}
