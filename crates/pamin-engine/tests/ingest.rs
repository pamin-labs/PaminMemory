//! What recording a section of a file leaves in the ledger.
//!
//! Ignored by default: it provisions PostgreSQL. Only the reindex test loads a
//! model -- nothing else here drains the cascade -- so the rest run in
//! seconds. Run with `cargo test -p pamin-engine --test ingest -- --ignored`.
//!
//! A section differs from a manual memory in three ways, each of which is a
//! cost or a leak if it goes wrong and none of which a search would show: an
//! unchanged section writes nothing, a section's name is never filed in the
//! name index, and its source is the file's. Each test asserts the manual path
//! does the opposite, so it fails if the two paths ever become one.

#[path = "../../pamin-index/tests/support/mod.rs"]
mod support;

use pamin_core::Validity;
use pamin_engine::{Engine, Retired};
use pamin_index::{Access, VectorIndex};
use pamin_store::{Connections, Database, Workspace, repository};

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

/// A fresh project, in `PAMIN_EVAL_HOME`'s workspace when it is set -- where
/// the model a reindex loads is already downloaded -- or in a scratch one.
async fn engine() -> (Engine, Option<tempfile::TempDir>) {
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
    let name = format!("ingest-{}", uuid::Uuid::now_v7());
    repository::ensure_project(database.pool(), &name)
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
    (engine, scratch)
}

async fn count(engine: &Engine, sql: &'static str) -> i64 {
    sqlx::query_scalar(sql)
        .bind(engine.project.0)
        .fetch_one(engine.database.pool())
        .await
        .expect("count")
}

async fn versions(engine: &Engine) -> i64 {
    count(
        engine,
        "SELECT count(*) FROM source_versions WHERE project_id = $1",
    )
    .await
}

/// The kinds of job owed for `topic`, sorted.
async fn owed(engine: &Engine, topic: &str) -> Vec<String> {
    let mut kinds: Vec<String> = sqlx::query_scalar(
        "SELECT j.job_type FROM index_jobs j JOIN topics t ON t.id = j.subject
         WHERE t.project_id = $1 AND t.name = $2",
    )
    .bind(engine.project.0)
    .bind(topic)
    .fetch_all(engine.database.pool())
    .await
    .expect("jobs");
    kinds.sort();
    kinds
}

const SECTION: &str = "docs/ops.md › Deploy › Rollback";
const TEXT: &str = "## Rollback\n\nA rollback reverts the deployment pipeline to the previous tag.";

/// Re-ingesting an unchanged section writes nothing at all, where `remember`
/// records the same content again as held evidence.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn an_unchanged_section_is_not_recorded_again() {
    let (engine, _scratch) = engine().await;

    let first = engine
        .remember_section(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("first ingest");
    let (verdict, recorded) = first.expect("a new section is recorded");
    assert!(verdict.is_promoted());
    assert!(recorded.state.is_some());
    assert_eq!(versions(&engine).await, 1);

    // The same text, and the same text reflowed: both are what the topic
    // already says.
    for again in [
        TEXT.to_string(),
        TEXT.replace(' ', "  ").replace("\n\n", "\n"),
    ] {
        let outcome = engine
            .remember_section(SECTION, &again, Validity::ALWAYS)
            .await
            .expect("re-ingest");
        assert!(outcome.is_none(), "{again:?} was recorded again");
    }
    assert_eq!(
        versions(&engine).await,
        1,
        "an unchanged section wrote evidence"
    );

    // What the manual path does with the same restatement, so the assertion
    // above is about the difference and not about restatements in general.
    let (verdict, _) = engine
        .remember(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("manual restatement");
    assert!(!verdict.is_promoted());
    assert_eq!(versions(&engine).await, 2);

    // A changed section is a new version, as any edit is.
    let edited = format!("{TEXT} It takes about ten minutes.");
    let (verdict, recorded) = engine
        .remember_section(SECTION, &edited, Validity::ALWAYS)
        .await
        .expect("edited")
        .expect("a changed section is recorded");
    assert!(verdict.is_promoted());
    assert_eq!(recorded.state.expect("promoted").version, 2);

    // Cut down below the filter's minimum, the section is held -- and held
    // again on every ingest rather than reported unchanged, because its topic
    // still says what it said before and nothing else would say so.
    for _ in 0..2 {
        let (verdict, _) = engine
            .remember_section(SECTION, "ok", Validity::ALWAYS)
            .await
            .expect("cut down")
            .expect("a held section over live content is not unchanged");
        assert!(!verdict.is_promoted());
    }
}

/// A section's name stays out of the name index, so it neither widens the
/// window every write's mention derivation scans nor owes a backfill; a manual
/// topic is filed and owes one.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn a_section_is_kept_out_of_the_name_index() {
    let (engine, _scratch) = engine().await;

    engine
        .remember_section(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("ingest")
        .expect("recorded");
    engine
        .remember("rollback_plan", TEXT, Validity::ALWAYS)
        .await
        .expect("manual write");

    let named: Vec<String> = sqlx::query_scalar(
        "SELECT t.name FROM topic_name_tokens n JOIN topics t ON t.id = n.topic_id
         WHERE n.project_id = $1",
    )
    .bind(engine.project.0)
    .fetch_all(engine.database.pool())
    .await
    .expect("names");
    assert_eq!(named, ["rollback_plan"]);
    assert_eq!(
        repository::widest_topic_name(engine.database.pool(), engine.project)
            .await
            .expect("widest"),
        2,
        "the section's name widened the window"
    );

    assert_eq!(
        owed(&engine, SECTION).await,
        ["derive_mentions", "sync_topic_index"]
    );
    assert_eq!(
        owed(&engine, "rollback_plan").await,
        ["backfill_mentions", "derive_mentions", "sync_topic_index"]
    );

    // And a rebuild of the name index -- what `reindex` runs -- leaves it out
    // too, because it asks the source rather than the name.
    assert_eq!(to_name(&engine).await, ["rollback_plan"]);

    // A topic written by hand and then ingested under the same name keeps the
    // name it was filed with.
    engine
        .remember_section(
            "rollback_plan",
            "## Rollback\n\nIngested over a topic first written by hand.",
            Validity::ALWAYS,
        )
        .await
        .expect("ingest over a manual topic")
        .expect("recorded");
    assert_eq!(to_name(&engine).await, ["rollback_plan"]);
}

/// What a rebuild of the name index would restate.
async fn to_name(engine: &Engine) -> Vec<String> {
    repository::topics_to_name(engine.database.pool(), engine.project)
        .await
        .expect("topics to name")
        .into_iter()
        .map(|topic| topic.name)
        .collect()
}

/// A section's evidence is filed as a file's, one source per section, and the
/// listing an ingest compares against holds only sections on the retrieval
/// surface.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn a_sections_source_is_the_files() {
    let (engine, _scratch) = engine().await;

    let other = "docs/ops.md › Deploy";
    for (topic, content) in [
        (SECTION, TEXT),
        (other, "# Deploy\n\nThe pipeline runs on argo cd."),
    ] {
        engine
            .remember_section(topic, content, Validity::ALWAYS)
            .await
            .expect("ingest")
            .expect("recorded");
    }
    // Held by the filter, so on no retrieval surface and not a section to
    // report missing later.
    let held = engine
        .remember_section("docs/ops.md › Tiny", "ok", Validity::ALWAYS)
        .await
        .expect("ingest")
        .expect("recorded as held");
    assert!(!held.0.is_promoted());
    // And held once: its topic does not exist, so there is no current content
    // to restate, and ingesting it again must not copy it again.
    let before = versions(&engine).await;
    let again = engine
        .remember_section("docs/ops.md › Tiny", "ok", Validity::ALWAYS)
        .await
        .expect("re-ingest");
    assert!(again.is_none(), "a held section was recorded again");
    assert_eq!(versions(&engine).await, before);
    engine
        .remember(
            "manual_topic",
            "written by hand, not ingested",
            Validity::ALWAYS,
        )
        .await
        .expect("manual");

    let sources: Vec<(String, String)> =
        sqlx::query_as("SELECT kind, locator FROM sources WHERE project_id = $1 ORDER BY locator")
            .bind(engine.project.0)
            .fetch_all(engine.database.pool())
            .await
            .expect("sources");
    assert_eq!(
        sources,
        [
            ("file".to_string(), format!("file:{other}")),
            ("file".to_string(), format!("file:{SECTION}")),
            ("file".to_string(), "file:docs/ops.md › Tiny".to_string()),
            ("manual".to_string(), "manual:manual_topic".to_string()),
        ]
    );

    let mut listed = repository::file_topic_names(engine.database.pool(), engine.project)
        .await
        .expect("file topics");
    listed.sort();
    assert_eq!(listed, [other, SECTION]);

    // The column admits only the kinds a variant writes.
    let rejected = sqlx::query(
        "INSERT INTO sources (id, project_id, kind, locator, created_at)
         VALUES ($1, $2, 'web', 'web:x', now())",
    )
    .bind(uuid::Uuid::now_v7())
    .bind(engine.project.0)
    .execute(engine.database.pool())
    .await;
    assert!(rejected.is_err(), "an unknown source kind was admitted");
}

/// `reindex` restates every topic's name in the name index, and still leaves
/// the sections out. Run through the engine's own rebuild rather than the
/// query it reads, which is what would pass if the rebuild stopped asking it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn a_reindex_keeps_sections_out_of_the_name_index() {
    let (engine, _scratch) = engine().await;

    engine
        .remember_section(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("ingest")
        .expect("recorded");
    engine
        .remember("rollback_plan", TEXT, Validity::ALWAYS)
        .await
        .expect("manual write");
    sqlx::query("DELETE FROM topic_name_tokens WHERE project_id = $1")
        .bind(engine.project.0)
        .execute(engine.database.pool())
        .await
        .expect("empty the name index");

    let rebuilt = engine.reindex().await.expect("reindex");
    assert_eq!(rebuilt.indexed, 2, "both memories are in the projection");
    assert_eq!(rebuilt.names, 1);
    assert_eq!(
        count(
            &engine,
            "SELECT count(*) FROM topic_name_tokens WHERE project_id = $1"
        )
        .await,
        1
    );
}

/// A section a file no longer has is taken off the retrieval surface: every
/// version soft deleted, nothing current, and the projection and the graph
/// owed the work that removes it -- while its evidence stays, and ingesting it
/// again brings it back as a new version.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn a_retired_section_leaves_the_surface_and_can_come_back() {
    let (engine, _scratch) = engine().await;

    for content in [TEXT.to_string(), format!("{TEXT} Edited once.")] {
        engine
            .remember_section(SECTION, &content, Validity::ALWAYS)
            .await
            .expect("ingest")
            .expect("recorded");
    }
    sqlx::query("DELETE FROM index_jobs WHERE project_id = $1")
        .bind(engine.project.0)
        .execute(engine.database.pool())
        .await
        .expect("settle the queue");

    assert_eq!(
        engine.retire_section(SECTION).await.expect("retire"),
        Retired::Retired
    );
    assert_eq!(
        count(
            &engine,
            "SELECT count(*) FROM topic_states WHERE project_id = $1 AND deleted_at IS NULL"
        )
        .await,
        0,
        "a version survived"
    );
    assert_eq!(
        count(
            &engine,
            "SELECT count(*) FROM topics WHERE project_id = $1 AND current_state_id IS NOT NULL"
        )
        .await,
        0
    );
    assert_eq!(versions(&engine).await, 2, "evidence went with it");
    assert_eq!(
        owed(&engine, SECTION).await,
        ["derive_mentions", "sync_topic_index"]
    );
    assert!(
        repository::file_topic_names(engine.database.pool(), engine.project)
            .await
            .expect("file topics")
            .is_empty(),
        "a retired section is still listed as on the surface"
    );

    // Retiring again finds nothing to retire.
    assert_eq!(
        engine.retire_section(SECTION).await.expect("again"),
        Retired::Absent
    );
    assert_eq!(
        engine
            .retire_section("no such topic")
            .await
            .expect("absent"),
        Retired::Absent
    );

    // The section is back in the file: a new version, current again.
    let (verdict, recorded) = engine
        .remember_section(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("back")
        .expect("a returning section is recorded, not unchanged");
    assert!(verdict.is_promoted());
    assert_eq!(recorded.state.expect("promoted").version, 3);
}

/// A topic somebody also wrote by hand is theirs as well as the file's, and
/// is left on the surface.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn a_section_written_by_hand_too_is_kept() {
    let (engine, _scratch) = engine().await;
    engine
        .remember_section(SECTION, TEXT, Validity::ALWAYS)
        .await
        .expect("ingest")
        .expect("recorded");
    engine
        .remember(
            SECTION,
            "A note added by hand to the rollback section.",
            Validity::ALWAYS,
        )
        .await
        .expect("manual write");

    assert_eq!(
        engine.retire_section(SECTION).await.expect("retire"),
        Retired::Kept
    );
    assert_eq!(
        count(
            &engine,
            "SELECT count(*) FROM topics WHERE project_id = $1 AND current_state_id IS NOT NULL"
        )
        .await,
        1
    );
}
