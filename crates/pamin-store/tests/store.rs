//! Exercises the store against a real embedded PostgreSQL.
//!
//! Ignored by default: the first run downloads and installs a PostgreSQL
//! distribution, which is too slow and too network-dependent for the ordinary
//! test loop. Run with `cargo test -p pamin-store -- --ignored`.
//!
//! Everything lives in one test so a single cluster is installed, started, and
//! stopped. Splitting it across tests would install PostgreSQL once per
//! temporary workspace.

use pamin_core::{
    Derivation, EdgeKind, FilterDecision, JobKind, SourceKind, TombstoneReason, Validity,
    VersionOffset, resolve,
};
use pamin_store::graph::{EdgeClaim, Expansion};
use pamin_store::{Connections, Database, Workspace, graph, jobs, repository};
// The table name is a literal from the list above, not caller input; the
// assertion is what lets it be interpolated at all.
use sqlx::AssertSqlSafe;
use time::OffsetDateTime;

/// Runs one repository write inside its own transaction.
///
/// The writes that hold a row lock across several statements take a connection
/// rather than opening a transaction themselves, so that the whole write path
/// can be one transaction. A pooled connection in autocommit mode ends a
/// transaction at every statement, which releases the lock before the insert it
/// was taken for -- so the tests supply a real transaction, the way the engine
/// does.
macro_rules! committed {
    ($database:expr, $call:path, $($argument:expr),* $(,)?) => {{
        let mut transaction = $database.pool().begin().await.expect("begin");
        let outcome = $call(&mut *transaction, $($argument),*).await;
        transaction.commit().await.expect("commit");
        outcome
    }};
}

#[tokio::test]
#[ignore = "downloads and starts a real postgres cluster"]
async fn the_ledger_holds_its_promises() {
    // A fresh workspace, because one of the assertions below is that the
    // database starts empty. This pointed at a fixed path, which made the
    // premise true exactly once: a second run found the projects the first
    // one left and failed before testing anything. Paying `initdb` is what
    // buys a test that can be run twice, and every other harness in this
    // workspace already pays it.
    let home = tempfile::tempdir().expect("temp workspace");
    let workspace = Workspace::at(home.path());

    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open workspace");

    migrations_create_every_table(&database).await;
    the_cluster_forces_what_it_writes_to_disk(&database).await;
    reopening_reuses_the_running_server(&workspace).await;
    appending_versions_builds_a_supersession_chain(&database).await;
    soft_deleting_the_current_version_promotes_its_predecessor(&database).await;
    filtered_evidence_is_still_stored(&database).await;
    edges_are_versioned_rather_than_overwritten(&database).await;
    expansion_is_bounded_undirected_and_time_filtered(&database).await;
    grep_reaches_evidence_the_index_never_saw(&database).await;
    a_retraction_reason_decides_what_history_keeps(&database).await;
    a_seed_never_reaches_itself_however_deep_the_walk(&database).await;
    concurrent_writers_to_one_source_lose_no_evidence(&database, &workspace).await;
    concurrent_appends_to_one_topic_form_one_chain(&database, &workspace).await;
    ensuring_a_row_that_exists_does_not_rewrite_it(&database).await;
    derived_edges_are_asserted_together_or_not_at_all(&database).await;
    a_workspace_the_previous_runner_migrated_is_adopted(&database, &workspace).await;
    dropping_the_state_copy_loses_nothing_a_state_said(&database, &workspace).await;
    settled_jobs_go_and_owed_jobs_stay_through_the_migration(&database, &workspace).await;
    dropping_the_signal_columns_loses_nothing_written(&database, &workspace).await;
    a_queued_jobs_subject_survives_losing_its_key(&database, &workspace).await;
    the_current_state_pointer_follows_every_write(&database).await;
    two_adjacent_hubs_do_not_multiply(&database).await;
    the_outbox_coalesces_claims_and_survives_a_lost_worker(&database).await;
    each_kind_is_claimed_by_its_own_priority(&database).await;
    what_a_topic_says_now_is_one_lookup(&database).await;
    a_completion_names_the_claim_it_belongs_to(&database).await;
    one_projects_worker_never_takes_anothers_work(&database).await;
    a_derived_edge_the_content_stopped_making_is_closed(&database).await;
    several_topics_restate_their_mentions_at_once(&database).await;
    every_column_holds_what_was_written_to_it(&database).await;
    evidence_and_the_span_over_it_are_one_write(&database).await;
    a_promoted_write_is_one_statement_after_its_locks(&database).await;
    an_edge_reads_the_same_direction_from_either_end(&database).await;
    a_version_is_numbered_and_read_from_its_own_key(&database, &workspace).await;

    drop(database);
    a_stopped_server_is_started_again_without_waiting(&workspace).await;
}

/// Opening a workspace whose server was stopped starts it, promptly.
///
/// `stop` leaves the server record behind, as a reboot does. Asking whether
/// that server was up by connecting to it found the port refused and retried
/// for the pool's whole thirty-second acquire timeout before starting a new
/// one, so the first command after either paid half a minute for nothing.
/// Last, because it stops the cluster everything above shares.
async fn a_stopped_server_is_started_again_without_waiting(workspace: &Workspace) {
    pamin_store::database::stop(workspace)
        .await
        .expect("stop the server");
    assert!(
        workspace.read_server().expect("read the record").is_some(),
        "the premise: the record outlives the cluster"
    );

    let started = std::time::Instant::now();
    let reopened = Database::open(workspace, Connections::PerCommand)
        .await
        .expect("reopen a stopped workspace");
    let waited = started.elapsed();
    let (one,): (i32,) = sqlx::query_as("SELECT 1")
        .fetch_one(reopened.pool())
        .await
        .expect("the restarted server answers");
    assert_eq!(one, 1);
    assert!(
        waited < std::time::Duration::from_secs(20),
        "reopening a stopped workspace took {waited:?}; starting the server takes seconds"
    );
    drop(reopened);
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
        "relationships",
        "relationship_versions",
    ] {
        let (count,): (i64,) =
            sqlx::query_as(AssertSqlSafe(format!("SELECT count(*) FROM {table}")))
                .fetch_one(database.pool())
                .await
                .unwrap_or_else(|error| panic!("querying {table}: {error}"));
        assert_eq!(count, 0, "{table} should start empty");
    }
}

/// The running cluster flushes to disk, and does not make a commit wait for it.
///
/// Asked of the server rather than of the settings map, because the setting
/// that decides it is not in the map: `postgresql_embedded` passes `-F` on the
/// command line, which turns `fsync` off, and only an override given after it
/// turns it back on. A map entry the server never honoured would pass a test
/// that read the map.
async fn the_cluster_forces_what_it_writes_to_disk(database: &Database) {
    for (setting, expected) in [("fsync", "on"), ("synchronous_commit", "off")] {
        let value: String = sqlx::query_scalar(AssertSqlSafe(format!("SHOW {setting}")))
            .fetch_one(database.pool())
            .await
            .unwrap_or_else(|error| panic!("SHOW {setting}: {error}"));
        assert_eq!(value, expected, "the cluster runs with {setting} = {value}");
    }
}

async fn reopening_reuses_the_running_server(workspace: &Workspace) {
    // Must not start a second cluster, and must not fail re-applying migrations.
    let reopened = Database::open(workspace, Connections::PerCommand)
        .await
        .expect("reopen workspace");
    drop(reopened);
}

/// Writes a memory to a topic the way the write path does: the source locked,
/// the topic locked, and the evidence, its span and the state in one statement.
///
/// Queues nothing, so tests that count the queue see only what they queued.
async fn write_state(
    database: &Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
    locator: &str,
    content: &str,
) -> pamin_core::TopicState {
    let name = repository::topics_by_id(database.pool(), project, &[topic])
        .await
        .expect("name the topic")
        .pop()
        .expect("the topic exists")
        .1;
    let mut transaction = database.pool().begin().await.expect("begin");
    let source = repository::ensure_source(&mut transaction, project, SourceKind::Manual, locator)
        .await
        .expect("ensure source");
    let locked = repository::lock_topic(&mut transaction, project, &name)
        .await
        .expect("lock topic")
        .expect("the topic exists");
    let (_, _, state) = repository::append_promoted(
        &mut transaction,
        project,
        source,
        &repository::Evidence {
            content,
            content_hash: "hash",
            decision: FilterDecision::Promoted,
            reason: "test fixture",
            language: None,
            language_confidence: None,
        },
        &repository::Promotion {
            topic: &locked,
            observed_at: OffsetDateTime::now_utc(),
            validity: Validity::ALWAYS,
            owed: &[],
        },
    )
    .await
    .expect("append promoted");
    transaction.commit().await.expect("commit");
    state
}

/// Puts a state on a span that already exists, under the topic's lock, and
/// reads it back.
///
/// Nothing in the product writes a state this way: a write appends its
/// evidence, the span over all of it and the state in one statement
/// (`append_promoted`). The read path still cuts a state out of whatever part
/// of the evidence its span covers, and a span over part of the evidence is
/// what these tests need to check that -- so they build one here.
async fn state_over_span(
    database: &Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
    span: &pamin_core::SourceSpan,
) -> pamin_core::TopicState {
    let mut transaction = database.pool().begin().await.expect("begin");
    let previous: Option<uuid::Uuid> =
        sqlx::query_scalar("SELECT current_state_id FROM topics WHERE id = $1 FOR UPDATE")
            .bind(topic.0)
            .fetch_one(&mut *transaction)
            .await
            .expect("lock the topic");
    let version: i32 = sqlx::query_scalar(
        "WITH state AS (
             INSERT INTO topic_states (
                 id, project_id, topic_id, version, source_span_id,
                 observed_at, recorded_at, supersedes
             )
             SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, now(), now(), $5
               FROM topic_states WHERE project_id = $2 AND topic_id = $3
             RETURNING id, version
         ), pointer AS (
             UPDATE topics SET current_state_id = state.id, current_version = state.version
               FROM state WHERE topics.id = $3
         )
         SELECT version FROM state",
    )
    .bind(uuid::Uuid::now_v7())
    .bind(project.0)
    .bind(topic.0)
    .bind(span.id.0)
    .bind(previous)
    .fetch_one(&mut *transaction)
    .await
    .expect("append a state over the span");
    transaction.commit().await.expect("commit");
    repository::topic_state(database.pool(), project, topic, version as u32)
        .await
        .expect("read topic state")
        .expect("the state was written")
}

async fn appending_versions_builds_a_supersession_chain(database: &Database) {
    let project = repository::ensure_project(database.pool(), "ledger")
        .await
        .expect("ensure project");
    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "deployment_pipeline"
    )
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

    let versions = repository::topic_versions(database.pool(), topic.id)
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

    let loaded = repository::topic_state(database.pool(), project.id, topic.id, 1)
        .await
        .expect("load state")
        .expect("state exists");
    assert_eq!(loaded.content, "deploys via make");
}

async fn soft_deleting_the_current_version_promotes_its_predecessor(database: &Database) {
    let project = repository::ensure_project(database.pool(), "ledger")
        .await
        .expect("ensure project");
    let topic = repository::find_topic(database.pool(), project.id, "deployment_pipeline")
        .await
        .expect("find topic")
        .expect("topic exists");

    let deleted = committed!(database, repository::soft_delete_topic_state, topic.id, 2)
        .expect("soft delete");
    assert!(deleted);

    let versions = repository::topic_versions(database.pool(), topic.id)
        .await
        .expect("versions");
    assert_eq!(versions, vec![1], "deleted versions leave the live set");

    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    assert_eq!(latest.version, 1);
    assert!(latest.is_current, "the predecessor becomes current");

    // The row itself survives, so history and audit still reach it.
    let still_there = repository::topic_state(database.pool(), project.id, topic.id, 2)
        .await
        .expect("load deleted state")
        .expect("deleted state is still stored");
    assert!(still_there.deleted_at.is_some());
    assert_eq!(still_there.content, "deploys via ci");

    // A new append continues the numbering rather than reusing the freed one.
    let next = write_state(database, project.id, topic.id, "note-3", "deploys via cd").await;
    assert_eq!(next.version, 3, "version numbers are never reused");
    // And supersedes the newest state that survives, which is the one the
    // topic's pointer names, not the deleted version numbered before it.
    let survivor = repository::topic_state(database.pool(), project.id, topic.id, 1)
        .await
        .expect("load the survivor")
        .expect("version 1 is stored");
    assert_eq!(next.supersedes, Some(survivor.id));

    // The identifiers alone name exactly the topics whose states a rebuild
    // indexes, so a reshape and a rebuild agree on what the index holds.
    let states = repository::all_current_topic_states(database.pool(), project.id)
        .await
        .expect("current states");
    let ids = repository::current_topic_ids(database.pool(), project.id)
        .await
        .expect("current topic ids");
    assert!(ids.contains(&topic.id));
    assert_eq!(
        ids,
        states
            .iter()
            .map(|state| state.topic_id)
            .collect::<Vec<_>>()
    );
}

async fn filtered_evidence_is_still_stored(database: &Database) {
    let project = repository::ensure_project(database.pool(), "ledger")
        .await
        .expect("ensure project");
    let source = committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "noise-source"
    )
    .expect("ensure source");

    committed!(
        database,
        repository::append_source_version,
        project.id,
        source,
        "ok",
        "hash",
        FilterDecision::Filtered,
        "no durable claim"
    )
    .expect("append filtered evidence");

    let stored = repository::latest_source_version(database.pool(), project.id, source)
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

/// A project with three topics wired into a chain, for the graph checks.
async fn graph_fixture(
    database: &Database,
) -> (
    pamin_core::ProjectId,
    pamin_core::TopicId,
    pamin_core::TopicId,
    pamin_core::TopicId,
) {
    let project = repository::ensure_project(database.pool(), "graph")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["service", "database", "backup_job"] {
        let topic =
            committed!(database, repository::ensure_topic, project.id, name).expect("ensure topic");
        write_state(
            database,
            project.id,
            topic.id,
            &format!("graph-{name}"),
            &format!("a durable claim about {name}"),
        )
        .await;
        topics.push(topic.id);
    }

    (project.id, topics[0], topics[1], topics[2])
}

async fn edges_are_versioned_rather_than_overwritten(database: &Database) {
    let (project, service, db, _) = graph_fixture(database).await;

    let first = graph::assert_edge(
        database.pool(),
        project,
        service,
        db,
        &EdgeClaim::explicit(EdgeKind::DependsOn),
    )
    .await
    .expect("assert edge");
    assert!(first.is_new());
    assert_eq!(first.version().version, 1);
    assert_eq!(first.version().derivation, Derivation::Explicit);

    // Asserting the same claim again must not stack a version, or every
    // rewrite of unchanged content would grow the ledger without limit.
    let again = graph::assert_edge(
        database.pool(),
        project,
        service,
        db,
        &EdgeClaim::explicit(EdgeKind::DependsOn),
    )
    .await
    .expect("assert edge again");
    assert!(!again.is_new(), "an unchanged claim appends nothing");
    assert_eq!(again.version().id, first.version().id);

    // A changed claim closes the live version and appends a successor.
    let mut narrowed = EdgeClaim::explicit(EdgeKind::DependsOn);
    narrowed.validity = Validity::new(Some(OffsetDateTime::UNIX_EPOCH), None);
    let second = graph::assert_edge(database.pool(), project, service, db, &narrowed)
        .await
        .expect("assert changed edge");
    assert!(second.is_new());
    assert_eq!(second.version().version, 2);
    assert_eq!(second.version().supersedes, Some(first.version().id));

    let relationship =
        graph::find_relationship(database.pool(), project, service, db, EdgeKind::DependsOn)
            .await
            .expect("find relationship")
            .expect("relationship exists");

    let history = graph::edge_history(database.pool(), project, relationship.id)
        .await
        .expect("history");
    assert_eq!(history.len(), 2);
    assert_eq!(
        history[0].tombstone_reason,
        Some(TombstoneReason::Superseded),
        "a replaced version records why it was closed"
    );

    // Closing retracts the claim and leaves every row where it was.
    let closed = graph::close_edge(
        database.pool(),
        project,
        service,
        db,
        EdgeKind::DependsOn,
        TombstoneReason::Deleted,
    )
    .await
    .expect("close edge");
    assert!(closed);
    assert!(
        graph::live_version(database.pool(), relationship.id)
            .await
            .expect("live version")
            .is_none(),
        "nothing is believed after a retraction"
    );
    assert_eq!(
        graph::edge_history(database.pool(), project, relationship.id)
            .await
            .expect("history")
            .len(),
        2,
        "retraction removes no rows"
    );

    assert!(
        !graph::close_edge(
            database.pool(),
            project,
            service,
            db,
            EdgeKind::DependsOn,
            TombstoneReason::Deleted,
        )
        .await
        .expect("close again"),
        "closing an already closed edge reports that nothing was open"
    );
}

async fn expansion_is_bounded_undirected_and_time_filtered(database: &Database) {
    let (project, service, db, backup) = graph_fixture(database).await;

    // service -> database -> backup_job, so backup_job is two hops from
    // service and is only reachable by following the second edge backwards.
    let service_state = current_state(database, project, service).await;
    graph::assert_edge(
        database.pool(),
        project,
        service,
        db,
        &EdgeClaim::derived(EdgeKind::Mentions, service_state, 0.5),
    )
    .await
    .expect("service -> database");
    graph::assert_edge(
        database.pool(),
        project,
        backup,
        db,
        &EdgeClaim::explicit(EdgeKind::DependsOn),
    )
    .await
    .expect("backup_job -> database");

    let one_hop = graph::expand(
        &mut *connection(database).await,
        project,
        &[service],
        &Expansion::to_depth(1),
    )
    .await
    .expect("expand one hop");
    let reached: Vec<_> = one_hop.iter().map(|n| n.topic).collect();
    assert_eq!(
        reached,
        vec![db],
        "one hop reaches only the direct neighbour"
    );
    assert_eq!(one_hop[0].hops, 1);
    assert_eq!(one_hop[0].via, service);
    assert_eq!(one_hop[0].derivation, Derivation::Deterministic);

    let two_hops = graph::expand(
        &mut *connection(database).await,
        project,
        &[service],
        &Expansion::to_depth(2),
    )
    .await
    .expect("expand two hops");
    let backup_hit = two_hops
        .iter()
        .find(|n| n.topic == backup)
        .expect("two hops reaches backup_job");
    assert_eq!(backup_hit.hops, 2);
    assert_eq!(
        backup_hit.via, db,
        "the path names the topic it came through"
    );
    // The second edge points backup_job -> database, so reaching backup_job
    // from service means the walk crossed it against its direction.
    assert!(
        two_hops.iter().map(|n| n.topic).all(|t| t != service),
        "a seed with no independent path back to itself is not a neighbour"
    );

    // Restricting the edge kind removes the path that used the other kind.
    let mentions_only = graph::expand(
        &mut *connection(database).await,
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: Some(&[EdgeKind::Mentions]),
            at: None,
            keep: None,
        },
    )
    .await
    .expect("expand mentions only");
    assert_eq!(
        mentions_only.iter().map(|n| n.topic).collect::<Vec<_>>(),
        vec![db],
        "backup_job is only reachable through a depends_on edge"
    );

    // An edge bounded to the past is invisible to a query about now.
    let mut expired = EdgeClaim::explicit(EdgeKind::DependsOn);
    expired.validity = Validity::new(
        Some(OffsetDateTime::UNIX_EPOCH),
        Some(OffsetDateTime::UNIX_EPOCH + time::Duration::days(1)),
    );
    graph::assert_edge(database.pool(), project, backup, db, &expired)
        .await
        .expect("bound the edge to the past");

    let now = graph::expand(
        &mut *connection(database).await,
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: None,
            at: Some(OffsetDateTime::now_utc()),
            keep: None,
        },
    )
    .await
    .expect("expand at now");
    assert!(
        now.iter().all(|n| n.topic != backup),
        "an edge asserted only for a past interval does not hold now"
    );

    let back_then = graph::expand(
        &mut *connection(database).await,
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: None,
            at: Some(OffsetDateTime::UNIX_EPOCH + time::Duration::hours(1)),
            keep: None,
        },
    )
    .await
    .expect("expand inside the interval");
    assert!(
        back_then.iter().any(|n| n.topic == backup),
        "the same edge holds inside its own interval"
    );
}

/// The current state of a topic, for edges that cite what caused them.
async fn current_state(
    database: &Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
) -> pamin_core::TopicStateId {
    let versions = repository::topic_versions(database.pool(), topic)
        .await
        .expect("versions");
    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    repository::topic_state(database.pool(), project, topic, latest.version)
        .await
        .expect("load state")
        .expect("state exists")
        .id
}

async fn grep_reaches_evidence_the_index_never_saw(database: &Database) {
    let project = repository::ensure_project(database.pool(), "ledger")
        .await
        .expect("ensure project");
    let source = committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "grep-source"
    )
    .expect("ensure source");

    // Held by the filter, so it never became a topic state and never entered
    // the projection index. Reaching it is the entire reason this exists.
    committed!(
        database,
        repository::append_source_version,
        project.id,
        source,
        "the KILN reaches cone ten",
        "hash",
        FilterDecision::Filtered,
        "no durable claim"
    )
    .expect("append filtered evidence");

    let hits = repository::grep_evidence(database.pool(), project.id, "cone ten", true, 10)
        .await
        .expect("grep");
    let found = hits
        .iter()
        .find(|hit| hit.source_version.content.contains("cone ten"))
        .expect("filtered evidence is still reachable");
    assert_eq!(
        found.source_version.filter_decision,
        FilterDecision::Filtered,
        "the result says why it never reached the retrieval surface"
    );
    assert_eq!(found.locator, "grep-source");
    assert_eq!(
        &found.source_version.content[found.offset..found.offset + 8],
        "cone ten",
        "the offset points at the match"
    );

    // Case sensitivity is a choice the caller makes, not one made for them.
    assert!(
        repository::grep_evidence(database.pool(), project.id, "kiln", true, 10)
            .await
            .expect("grep")
            .is_empty(),
        "a case-sensitive search does not fold case"
    );
    assert!(
        !repository::grep_evidence(database.pool(), project.id, "kiln", false, 10)
            .await
            .expect("grep")
            .is_empty(),
        "a case-insensitive search does"
    );

    // The offset is in bytes, the unit every caller slices with. SQL's
    // `position` counts characters, and the two part company at the first
    // character outside ASCII -- which is most evidence, stored in whatever
    // language it arrived in. Either way of folding case has to agree.
    committed!(
        database,
        repository::append_source_version,
        project.id,
        source,
        "窑炉温度达到 Cone Twelve 之后保持",
        "hash-multibyte",
        FilterDecision::Filtered,
        "no durable claim"
    )
    .expect("append multi-byte evidence");
    for (needle, case_sensitive) in [("Cone Twelve", true), ("cone twelve", false)] {
        let hits =
            repository::grep_evidence(database.pool(), project.id, needle, case_sensitive, 10)
                .await
                .expect("grep");
        let [hit] = hits.as_slice() else {
            panic!(
                "{needle:?} should match one piece of evidence, got {}",
                hits.len()
            );
        };
        assert_eq!(
            hit.source_version
                .content
                .get(hit.offset..hit.offset + needle.len()),
            Some("Cone Twelve"),
            "the offset of {needle:?} is a byte offset into the evidence"
        );
    }

    // Superseded versions stay reachable, which is what makes this an audit
    // route rather than a second view of current state.
    let topic = repository::find_topic(database.pool(), project.id, "deployment_pipeline")
        .await
        .expect("find topic")
        .expect("topic exists");
    assert!(topic.name == "deployment_pipeline");
    assert!(
        !repository::grep_evidence(database.pool(), project.id, "deploys via make", true, 10)
            .await
            .expect("grep")
            .is_empty(),
        "the first version of a rewritten memory is still in evidence"
    );
}

async fn a_retraction_reason_decides_what_history_keeps(database: &Database) {
    let project = repository::ensure_project(database.pool(), "history")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["tenant_a", "tenant_b", "tenant_c"] {
        let topic =
            committed!(database, repository::ensure_topic, project.id, name).expect("ensure topic");
        write_state(
            database,
            project.id,
            topic.id,
            &format!("history-{name}"),
            &format!("a durable claim with no cross reference {name}"),
        )
        .await;
        topics.push(topic.id);
    }
    let (root, ended, wrong) = (topics[0], topics[1], topics[2]);

    for target in [ended, wrong] {
        graph::assert_edge(
            database.pool(),
            project.id,
            root,
            target,
            &EdgeClaim::explicit(EdgeKind::DependsOn),
        )
        .await
        .expect("assert edge");
    }

    // A second earlier, not "just now". PostgreSQL stores microseconds while
    // OffsetDateTime carries nanoseconds, so two calls close together can land
    // in the same stored microsecond and make a strict comparison false. The
    // question being asked is about an earlier instant, so it costs nothing to
    // pick one that is unambiguously earlier.
    let before_retraction = OffsetDateTime::now_utc() - time::Duration::seconds(1);

    // One relationship ended; the other was never true.
    graph::close_edge(
        database.pool(),
        project.id,
        root,
        ended,
        EdgeKind::DependsOn,
        TombstoneReason::Closed,
    )
    .await
    .expect("close ended");
    graph::close_edge(
        database.pool(),
        project.id,
        root,
        wrong,
        EdgeKind::DependsOn,
        TombstoneReason::Deleted,
    )
    .await
    .expect("close wrong");

    // Neither is believed now, so neither is traversed now.
    let now = graph::expand(
        &mut *connection(database).await,
        project.id,
        &[root],
        &Expansion::to_depth(1),
    )
    .await
    .expect("expand now");
    assert!(now.is_empty(), "nothing retracted is still asserted");

    // But a question about an earlier instant is a different question. A
    // relationship that ended did hold before it ended; one that was never
    // true never held. Treating both retractions alike erased that, which
    // meant retracting an edge deleted its history too.
    let earlier = graph::expand(
        &mut *connection(database).await,
        project.id,
        &[root],
        &Expansion {
            depth: 1,
            kinds: None,
            at: Some(before_retraction),
            keep: None,
        },
    )
    .await
    .expect("expand earlier");
    let reached: Vec<_> = earlier.iter().map(|n| n.topic).collect();
    assert!(
        reached.contains(&ended),
        "a relationship that ended still held before it ended: {reached:?}"
    );
    assert!(
        !reached.contains(&wrong),
        "a claim retracted as wrong never held at any instant: {reached:?}"
    );

    // The walk reports where it began, which at one hop is also the topic it
    // arrived through, and past that is not.
    assert_eq!(earlier[0].origin, root);
    assert_eq!(earlier[0].via, root);
}

async fn a_seed_never_reaches_itself_however_deep_the_walk(database: &Database) {
    let project = repository::ensure_project(database.pool(), "cycles")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["ring_a", "ring_b", "ring_c"] {
        let topic =
            committed!(database, repository::ensure_topic, project.id, name).expect("ensure topic");
        write_state(
            database,
            project.id,
            topic.id,
            &format!("cycle-{name}"),
            &format!("an isolated durable claim {name}"),
        )
        .await;
        topics.push(topic.id);
    }
    let (a, b, c) = (topics[0], topics[1], topics[2]);

    // A ring, which is the shape that makes depth matter: every node is
    // reachable from every other, and from itself.
    for (from, to) in [(a, b), (b, c), (c, a)] {
        graph::assert_edge(
            database.pool(),
            project.id,
            from,
            to,
            &EdgeClaim::explicit(EdgeKind::RelatedTo),
        )
        .await
        .expect("assert ring edge");
    }

    for depth in 1..=4 {
        let reached: Vec<_> = graph::expand(
            &mut *connection(database).await,
            project.id,
            &[a],
            &Expansion::to_depth(depth),
        )
        .await
        .expect("expand")
        .into_iter()
        .map(|neighbor| neighbor.topic)
        .collect();

        assert!(
            !reached.contains(&a),
            "at depth {depth} the seed came back as its own neighbour: {reached:?}"
        );
    }
}

/// Evidence written at the same time from several connections all survives.
///
/// Version numbers are allocated from the maximum already stored, so without a
/// lock on the source every concurrent writer reads the same maximum and every
/// one of them claims the version after it. The uniqueness constraint then
/// admits exactly one and the rest fail, which loses evidence -- the one thing
/// this store promises never to do.
///
/// Real connections rather than one: the race is between sessions, and a single
/// client serializes it away.
async fn concurrent_writers_to_one_source_lose_no_evidence(
    database: &Database,
    workspace: &Workspace,
) {
    const WRITERS: usize = 8;

    let project = repository::ensure_project(database.pool(), "contended")
        .await
        .expect("ensure project");
    let source = committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "contended-source"
    )
    .expect("ensure source");

    let server = workspace
        .read_server()
        .expect("read server record")
        .expect("workspace has a server");

    let writers: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let server = server.clone();
            tokio::spawn(async move {
                let database = Database::connect(&server, Connections::PerCommand)
                    .await
                    .expect("connect");
                // The write path's sequence: find the source, which locks it,
                // then append under that lock in the same transaction.
                let mut transaction = database.pool().begin().await.expect("begin");
                let found = repository::ensure_source(
                    &mut transaction,
                    project.id,
                    SourceKind::Manual,
                    "contended-source",
                )
                .await
                .expect("ensure source");
                assert_eq!(found, source);
                let content = format!("evidence from writer {writer}");
                let appended = repository::append_evidence(
                    &mut transaction,
                    project.id,
                    source,
                    &repository::Evidence {
                        content: &content,
                        content_hash: "hash",
                        decision: FilterDecision::Promoted,
                        reason: "test fixture",
                        language: None,
                        language_confidence: None,
                    },
                )
                .await
                .map(|(version, _)| version);
                transaction.commit().await.expect("commit");
                appended
            })
        })
        .collect();

    let mut versions = Vec::new();
    for writer in writers {
        let appended = writer
            .await
            .expect("writer task")
            .expect("every writer keeps its evidence");
        versions.push(appended.version);
    }

    versions.sort_unstable();
    assert_eq!(
        versions,
        (1..=WRITERS as u32).collect::<Vec<_>>(),
        "concurrent writers should take consecutive versions"
    );
}

/// Concurrent appends to one topic each supersede the state before them.
///
/// An append reads its predecessor from the topic's current-state pointer, in
/// the statement that locks the topic. That is right only if a writer that
/// waited for the lock reads the pointer the writer before it moved: if it
/// read the one from before the wait, two states would name the same
/// predecessor and the chain would fork -- no error, just a history that says
/// two things replaced one. So this checks the chain, not only the version
/// numbers.
///
/// Each writer has a source of its own, so the source lock serializes nothing
/// here and the topic lock is the only thing that can. The write path also
/// queues its work in the same statement, and eight requests for one subject
/// have to leave one row.
async fn concurrent_appends_to_one_topic_form_one_chain(
    database: &Database,
    workspace: &Workspace,
) {
    const WRITERS: usize = 8;

    let name = "contended_topic".to_string();
    let project = repository::ensure_project(database.pool(), &name)
        .await
        .expect("ensure project");
    let topic =
        committed!(database, repository::ensure_topic, project.id, &name).expect("ensure topic");
    let server = workspace
        .read_server()
        .expect("read server record")
        .expect("workspace has a server");

    let writers: Vec<_> = (0..WRITERS)
        .map(|writer| {
            let server = server.clone();
            let name = name.clone();
            tokio::spawn(async move {
                let database = Database::connect(&server, Connections::PerCommand)
                    .await
                    .expect("connect");
                let mut transaction = database.pool().begin().await.expect("begin");
                let source = repository::ensure_source(
                    &mut transaction,
                    project.id,
                    SourceKind::Manual,
                    &format!("{name}-{writer}"),
                )
                .await
                .expect("ensure source");
                let content = format!("state from writer {writer}");
                let evidence = repository::Evidence {
                    content: &content,
                    content_hash: "hash",
                    decision: FilterDecision::Promoted,
                    reason: "test fixture",
                    language: None,
                    language_confidence: None,
                };
                let locked = repository::lock_topic(&mut transaction, project.id, &name)
                    .await
                    .expect("lock topic")
                    .expect("the topic exists");
                let (_, _, state) = repository::append_promoted(
                    &mut transaction,
                    project.id,
                    source,
                    &evidence,
                    &repository::Promotion {
                        topic: &locked,
                        observed_at: OffsetDateTime::now_utc(),
                        validity: Validity::ALWAYS,
                        owed: &[JobKind::SyncTopicIndex],
                    },
                )
                .await
                .expect("every writer keeps its state");
                transaction.commit().await.expect("commit");
                state
            })
        })
        .collect();
    for writer in writers {
        writer.await.expect("writer task");
    }

    let chain: Vec<(uuid::Uuid, i32, Option<uuid::Uuid>)> = sqlx::query_as(
        "SELECT id, version, supersedes FROM topic_states
         WHERE topic_id = $1 ORDER BY version",
    )
    .bind(topic.id.0)
    .fetch_all(database.pool())
    .await
    .expect("read the chain");
    assert_eq!(
        chain.iter().map(|link| link.1).collect::<Vec<_>>(),
        (1..=WRITERS as i32).collect::<Vec<_>>(),
        "concurrent writers should take consecutive versions"
    );
    let mut previous = None;
    for (id, version, supersedes) in &chain {
        assert_eq!(
            *supersedes, previous,
            "version {version} supersedes a state other than the one before it"
        );
        previous = Some(*id);
    }
    let newest = chain.last().map(|link| link.1 as u32);
    assert_pointer_matches_the_ledger(database, project.id, topic.id, newest).await;

    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("count pending"),
        1,
        "{WRITERS} requests for one subject should coalesce onto one row"
    );
}

/// Re-ensuring a project, source, topic or relationship leaves the row alone.
///
/// Returning the existing row from a conflict clause requires `DO UPDATE`, and
/// with a uniqueness constraint as the target the only assignment available is
/// the key to itself. That reads as a no-op and is not one: PostgreSQL takes a
/// row lock and writes a new tuple version anyway. Every command begins by
/// ensuring the project, so the cost landed on the one row all of them share.
///
/// `ctid` locates a row's current tuple version, so it moves exactly when the
/// row is rewritten. That is the difference this test is for; counting rows
/// would pass either way.
async fn ensuring_a_row_that_exists_does_not_rewrite_it(database: &Database) {
    let project = repository::ensure_project(database.pool(), "idempotent")
        .await
        .expect("ensure project");
    let source = committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "idempotent-source"
    )
    .expect("ensure source");
    let from =
        committed!(database, repository::ensure_topic, project.id, "from").expect("ensure topic");
    let to =
        committed!(database, repository::ensure_topic, project.id, "to").expect("ensure topic");
    graph::assert_edge(
        database.pool(),
        project.id,
        from.id,
        to.id,
        &EdgeClaim::explicit(EdgeKind::RelatedTo),
    )
    .await
    .expect("assert edge");

    let rows = [
        ("projects", "id", project.id.0),
        ("sources", "id", source.0),
        ("topics", "id", from.id.0),
        ("relationships", "from_topic", from.id.0),
    ];

    let before = tuple_versions(database, &rows).await;

    repository::ensure_project(database.pool(), "idempotent")
        .await
        .expect("re-ensure project");
    committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "idempotent-source"
    )
    .expect("re-ensure source");
    committed!(database, repository::ensure_topic, project.id, "from").expect("re-ensure topic");
    graph::assert_edge(
        database.pool(),
        project.id,
        from.id,
        to.id,
        &EdgeClaim::explicit(EdgeKind::RelatedTo),
    )
    .await
    .expect("re-assert edge");

    assert_eq!(
        before,
        tuple_versions(database, &rows).await,
        "ensuring an existing row rewrote it"
    );
}

/// Where each named row's current tuple version sits.
async fn tuple_versions(database: &Database, rows: &[(&str, &str, uuid::Uuid)]) -> Vec<String> {
    let mut versions = Vec::new();
    for (table, column, id) in rows {
        let (placement,): (String,) = sqlx::query_as(AssertSqlSafe(format!(
            "SELECT ctid::TEXT FROM {table} WHERE {column} = $1"
        )))
        .bind(id)
        .fetch_one(database.pool())
        .await
        .unwrap_or_else(|error| panic!("reading {table}: {error}"));
        versions.push(placement);
    }
    versions
}

/// The edges one memory derives land together or not at all.
///
/// They are a single statement about what that memory says, and they used to be
/// asserted a transaction each: a failure partway through committed the earlier
/// ones and dropped the rest, leaving a memory that names fewer topics than it
/// does and no record that anything went wrong. The schema rejects an edge from
/// a topic to itself, so one at the end of a batch is a failure the database
/// supplies rather than one the test has to fake.
async fn derived_edges_are_asserted_together_or_not_at_all(database: &Database) {
    let project = repository::ensure_project(database.pool(), "atomic")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["first", "second", "third"] {
        topics.push(
            committed!(database, repository::ensure_topic, project.id, name)
                .expect("ensure topic")
                .id,
        );
    }

    let doomed = vec![
        (
            topics[0],
            topics[1],
            EdgeClaim::explicit(EdgeKind::RelatedTo),
        ),
        (
            topics[1],
            topics[2],
            EdgeClaim::explicit(EdgeKind::RelatedTo),
        ),
        // Refused by `relationships_no_self_edge`.
        (
            topics[2],
            topics[2],
            EdgeClaim::explicit(EdgeKind::RelatedTo),
        ),
    ];

    graph::assert_edges(database.pool(), project.id, &doomed)
        .await
        .expect_err("a self edge should be refused");

    for (from, to, _) in &doomed[..2] {
        assert!(
            graph::find_relationship(database.pool(), project.id, *from, *to, EdgeKind::RelatedTo)
                .await
                .expect("look up edge")
                .is_none(),
            "an edge from a batch that failed was left behind"
        );
    }

    // The same batch without the refused edge writes every one of them, and
    // asserting it a second time writes none: a batch is as idempotent as the
    // single assertion it is built from.
    let sound = &doomed[..2];
    let appended = graph::assert_edges(database.pool(), project.id, sound)
        .await
        .expect("assert edges");
    assert_eq!(
        appended.iter().filter(|edge| edge.is_new()).count(),
        sound.len(),
        "every edge in a batch should be appended"
    );

    let again = graph::assert_edges(database.pool(), project.id, sound)
        .await
        .expect("re-assert edges");
    assert_eq!(
        again.iter().filter(|edge| edge.is_new()).count(),
        0,
        "re-asserting an unchanged batch should append nothing"
    );
}

/// A database the previous migration runner had migrated is adopted, not redone.
///
/// `refinery` and `sqlx` keep unrelated books: different table, different
/// checksum function, different columns. Pointed at a database `refinery` had
/// already migrated, `sqlx` finds nothing applied and tries to apply
/// everything -- against a schema that already has every table.
///
/// This is the one code path in the change that only existing workspaces reach.
/// Anyone who deletes their workspace and starts over never runs it, which is
/// most of the ways it would be tried by hand, so it is built here instead: a
/// scratch database is migrated, its books are rewritten as the old runner kept
/// them, and the runner is pointed at it again. Without adoption the second run
/// fails trying to create `projects` a second time.
async fn a_workspace_the_previous_runner_migrated_is_adopted(
    database: &Database,
    workspace: &Workspace,
) {
    // Built to the schema the old runner left, which is the three migrations
    // that existed while it was in use -- not to today's schema.
    let scratch = database_left_at(database, workspace, "pamin_adoption_check", 3).await;

    pamin_store::migrate::run(&scratch)
        .await
        .expect("a database the previous runner migrated should be adopted");

    let adopted: Vec<(i64, i64)> =
        sqlx::query_as("SELECT version, execution_time FROM _sqlx_migrations ORDER BY version")
            .fetch_all(&scratch)
            .await
            .expect("read adopted migrations");

    assert_eq!(
        adopted
            .iter()
            .filter(|(_, execution_time)| *execution_time < 0)
            .map(|(version, _)| *version)
            .collect::<Vec<_>>(),
        vec![1, 2, 3],
        "exactly the migrations the old runner applied should carry the \
         placeholder execution time that marks an adopted row"
    );
    assert!(
        adopted.len() > 3,
        "migrations added after the old runner should still have been applied"
    );

    // And adoption is not a one-time trick that breaks the next start.
    pamin_store::migrate::run(&scratch)
        .await
        .expect("an adopted database still starts");

    scratch.close().await;
    sqlx::query("DROP DATABASE pamin_adoption_check")
        .execute(database.pool())
        .await
        .expect("drop scratch database");
}

/// V9 drops `topic_states.content` because every state's content is its span's
/// text -- and refuses to, rather than lose anything, when one is not.
///
/// Both halves run against rows written *before* the migration, which is the
/// only place it matters and the one a fresh workspace never reaches: a new
/// workspace is created at the newest schema and has no column to drop. So a
/// scratch database is left at V8, given states the old schema could hold, and
/// migrated.
///
/// The refusal is the half that would fail without its guard. Dropping the
/// column unconditionally succeeds on the disagreeing row too, and takes with
/// it the only copy of what that state said.
async fn dropping_the_state_copy_loses_nothing_a_state_said(
    database: &Database,
    workspace: &Workspace,
) {
    const NAME: &str = "pamin_state_content_check";

    /// One state over a span of one piece of evidence, the way V8 stored it.
    async fn state_at_v8(pool: &sqlx::PgPool, evidence: &str, span: (i32, i32), content: &str) {
        let (project, source, version, span_id, topic, state) = (
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
        );
        sqlx::raw_sql(AssertSqlSafe(format!(
            "INSERT INTO projects VALUES ('{project}', '{project}', now());
             INSERT INTO sources VALUES ('{source}', '{project}', 'manual', 'm', now());
             INSERT INTO source_versions VALUES ('{version}', '{project}', '{source}', 1,
                 $e${evidence}$e$, 'h', 'promoted', 'r', now());
             INSERT INTO source_spans VALUES ('{span_id}', '{project}', '{version}',
                 {}, {}, NULL, NULL);
             INSERT INTO topics (id, project_id, name, created_at)
                 VALUES ('{topic}', '{project}', 't', now());
             INSERT INTO topic_states (id, project_id, topic_id, version, content,
                 source_span_id, observed_at, recorded_at)
                 VALUES ('{state}', '{project}', '{topic}', 1, $c${content}$c$,
                 '{span_id}', now(), now());
             UPDATE topics SET current_state_id = '{state}', current_version = 1
              WHERE id = '{topic}';",
            span.0, span.1
        )))
        .execute(pool)
        .await
        .expect("write a state the way V8 stored it");
    }

    async fn has_content_column(pool: &sqlx::PgPool) -> bool {
        sqlx::query_scalar(
            "SELECT EXISTS (
                 SELECT 1 FROM information_schema.columns
                  WHERE table_schema = current_schema()
                    AND table_name = 'topic_states' AND column_name = 'content'
             )",
        )
        .fetch_one(pool)
        .await
        .expect("read the catalogue")
    }

    // A state that says something its span does not.
    let scratch = database_left_at(database, workspace, NAME, 8).await;
    state_at_v8(&scratch, "Grüße: the content", (9, 20), "the content").await;
    state_at_v8(
        &scratch,
        "what the evidence says",
        (0, 22),
        "something else",
    )
    .await;

    let refused = pamin_store::migrate::run(&scratch).await;
    assert!(
        refused.is_err(),
        "a state whose content is not its span's text must stop the migration"
    );
    assert!(
        has_content_column(&scratch).await,
        "a refused migration must leave the column, and the only copy of that state's content"
    );
    let kept: Vec<String> = sqlx::query_scalar("SELECT content FROM topic_states ORDER BY content")
        .fetch_all(&scratch)
        .await
        .expect("read the states back");
    assert_eq!(kept, vec!["something else", "the content"]);
    scratch.close().await;

    // The same data with the disagreement taken out migrates, and every state
    // reads back what it said -- including one whose span starts past a
    // character outside ASCII, which is where a cut by characters and a cut by
    // bytes part company.
    let scratch = database_left_at(database, workspace, NAME, 8).await;
    state_at_v8(&scratch, "Grüße: the content", (9, 20), "the content").await;
    state_at_v8(
        &scratch,
        "what the evidence says",
        (0, 22),
        "what the evidence says",
    )
    .await;

    pamin_store::migrate::run(&scratch)
        .await
        .expect("states that are their spans' text should migrate");
    assert!(
        !has_content_column(&scratch).await,
        "the copy should be gone once nothing depends on it"
    );

    let mut read: Vec<String> = Vec::new();
    let projects: Vec<uuid::Uuid> = sqlx::query_scalar("SELECT id FROM projects")
        .fetch_all(&scratch)
        .await
        .expect("projects");
    for project in projects {
        let states = repository::all_current_topic_states(&scratch, project.into())
            .await
            .expect("read states through the store");
        read.extend(states.into_iter().map(|state| state.content));
    }
    read.sort();
    assert_eq!(read, vec!["the content", "what the evidence says"]);

    scratch.close().await;
    sqlx::query(AssertSqlSafe(format!("DROP DATABASE {NAME}")))
        .execute(database.pool())
        .await
        .expect("drop scratch database");
}

/// V10 deletes the settled rows an earlier build kept, and nothing else.
///
/// The migration identifies them by the column it then drops, so it is the
/// last chance to tell a finished job from an owed one -- and getting it wrong
/// in the other direction loses work silently: an owed job deleted here is a
/// memory the index never hears about, with nothing left to say so. So a
/// scratch database is left at V9 holding one of each, plus one held by a
/// worker, and migrated.
async fn settled_jobs_go_and_owed_jobs_stay_through_the_migration(
    database: &Database,
    workspace: &Workspace,
) {
    const NAME: &str = "pamin_settled_jobs_check";
    let scratch = database_left_at(database, workspace, NAME, 9).await;

    let project = uuid::Uuid::now_v7();
    sqlx::raw_sql(AssertSqlSafe(format!(
        "INSERT INTO projects VALUES ('{project}', '{project}', now());
         INSERT INTO index_jobs (id, project_id, job_type, payload, idempotency_key,
                                 available_at, created_at, completed_at,
                                 claimed_at, claimed_by, priority)
         VALUES
           (gen_random_uuid(), '{project}', 'sync_topic_index', '{{}}', 'settled',
            now(), now(), now(), NULL, NULL, 10),
           (gen_random_uuid(), '{project}', 'sync_topic_index', '{{}}', 'owed',
            now(), now(), NULL, NULL, NULL, 10),
           (gen_random_uuid(), '{project}', 'derive_mentions', '{{}}', 'held',
            now(), now(), NULL, now(), 'a worker', 20);"
    )))
    .execute(&scratch)
    .await
    .expect("write the queue the way V9 kept it");

    pamin_store::migrate::run(&scratch)
        .await
        .expect("migrate past V10");

    // Read by kind: V12, which the run also applies, drops the key the rows
    // were written with.
    let mut left: Vec<String> = sqlx::query_scalar("SELECT job_type FROM index_jobs")
        .fetch_all(&scratch)
        .await
        .expect("read the queue back");
    left.sort();
    assert_eq!(
        left,
        vec!["derive_mentions", "sync_topic_index"],
        "only the settled row should go; owed and in-flight work must survive"
    );

    scratch.close().await;
    sqlx::query(AssertSqlSafe(format!("DROP DATABASE {NAME}")))
        .execute(database.pool())
        .await
        .expect("drop scratch database");
}

/// V12 moves a queued job's subject into a column and drops the key and the
/// payload that spelled it, and the work still coalesces afterwards.
///
/// The queue as V11 left it: a job with a subject, a project-wide job, and a
/// row whose subject does not parse -- which the old reader made nothing of.
/// After the migration each reads back through the store with the subject it
/// had, and enqueueing the same work again finds the migrated row rather
/// than adding a second, the project-wide one included: that is the
/// uniqueness the key used to carry. Then the refusal: two rows that would
/// name the same work once their keys are gone stop the migration, and the
/// queue is left as it was.
async fn a_queued_jobs_subject_survives_losing_its_key(database: &Database, workspace: &Workspace) {
    const NAME: &str = "pamin_job_subject_check";
    let project = uuid::Uuid::now_v7();
    let topic = uuid::Uuid::now_v7();
    let queue_at_v11 = |extra: &str| {
        format!(
            "INSERT INTO projects VALUES ('{project}', '{project}', now());
             INSERT INTO index_jobs (id, project_id, job_type, payload, idempotency_key,
                                     available_at, created_at, priority)
             VALUES
               (gen_random_uuid(), '{project}', 'sync_topic_index',
                '{{\"subject\": \"{topic}\"}}', 'sync_topic_index:{topic}', now(), now(), 10),
               (gen_random_uuid(), '{project}', 'optimize_index',
                '{{\"subject\": null}}', 'optimize_index:', now(), now(), 100),
               (gen_random_uuid(), '{project}', 'derive_mentions',
                '{{\"subject\": \"not a uuid\"}}', 'derive_mentions:not a uuid', now(), now(), 20)
               {extra};"
        )
    };

    let scratch = database_left_at(database, workspace, NAME, 11).await;
    sqlx::raw_sql(AssertSqlSafe(queue_at_v11("")))
        .execute(&scratch)
        .await
        .expect("write the queue the way V11 kept it");
    pamin_store::migrate::run(&scratch)
        .await
        .expect("migrate past V12");

    let claimed = jobs::claim(&scratch, project.into(), "migrated", 10, &JobKind::ALL)
        .await
        .expect("claim the migrated jobs");
    let mut read: Vec<(JobKind, Option<uuid::Uuid>)> =
        claimed.iter().map(|job| (job.kind, job.subject)).collect();
    read.sort_by_key(|(kind, _)| kind.as_str());
    assert_eq!(
        read,
        vec![
            (JobKind::DeriveMentions, None),
            (JobKind::OptimizeIndex, None),
            (JobKind::SyncTopicIndex, Some(topic)),
        ]
    );

    jobs::enqueue(
        &scratch,
        project.into(),
        JobKind::SyncTopicIndex,
        Some(topic),
    )
    .await
    .expect("enqueue a subject's work again");
    jobs::enqueue(&scratch, project.into(), JobKind::OptimizeIndex, None)
        .await
        .expect("enqueue project-wide work again");
    assert_eq!(
        jobs::pending(&scratch, project.into())
            .await
            .expect("count"),
        3,
        "enqueueing work already queued should find the migrated row"
    );
    scratch.close().await;

    // Two rows the key told apart and the columns do not.
    let scratch = database_left_at(database, workspace, NAME, 11).await;
    sqlx::raw_sql(AssertSqlSafe(queue_at_v11(&format!(
        ", (gen_random_uuid(), '{project}', 'derive_mentions',
            '{{\"subject\": \"also not a uuid\"}}', 'derive_mentions:also not a uuid',
            now(), now(), 20)"
    ))))
    .execute(&scratch)
    .await
    .expect("write the colliding queue");
    assert!(
        pamin_store::migrate::run(&scratch).await.is_err(),
        "two rows naming the same work must stop the migration"
    );
    let keys: i64 = sqlx::query_scalar("SELECT count(DISTINCT idempotency_key) FROM index_jobs")
        .fetch_one(&scratch)
        .await
        .expect("the key column is still there");
    assert_eq!(
        keys, 4,
        "a refused migration must leave the queue as it was"
    );

    scratch.close().await;
    sqlx::query(AssertSqlSafe(format!("DROP DATABASE {NAME}")))
        .execute(database.pool())
        .await
        .expect("drop scratch database");
}
/// V11 drops the retrieval-signal columns because nothing ever wrote them --
/// and refuses to, rather than lose anything, when a state holds one.
///
/// Like V9's test, this runs against rows written before the migration, in a
/// scratch database left at V10. The refusal is the half that would fail
/// without its guard: dropping the columns unconditionally succeeds on the
/// row that holds a value too, and takes the value with it.
async fn dropping_the_signal_columns_loses_nothing_written(
    database: &Database,
    workspace: &Workspace,
) {
    const NAME: &str = "pamin_signal_columns_check";

    /// One state over the whole of one piece of evidence, the way V10 stored
    /// it, with `signal` -- an assignment such as `access_count = 3` -- set on
    /// it when given.
    async fn state_at_v10(pool: &sqlx::PgPool, evidence: &str, signal: Option<&str>) {
        let (project, source, version, span, topic, state) = (
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
            uuid::Uuid::now_v7(),
        );
        let set = signal
            .map(|signal| format!("UPDATE topic_states SET {signal} WHERE id = '{state}';"))
            .unwrap_or_default();
        sqlx::raw_sql(AssertSqlSafe(format!(
            "INSERT INTO projects VALUES ('{project}', '{project}', now());
             INSERT INTO sources VALUES ('{source}', '{project}', 'manual', 'm', now());
             INSERT INTO source_versions VALUES ('{version}', '{project}', '{source}', 1,
                 $e${evidence}$e$, 'h', 'promoted', 'r', now());
             INSERT INTO source_spans VALUES ('{span}', '{project}', '{version}',
                 0, {}, NULL, NULL);
             INSERT INTO topics (id, project_id, name, created_at)
                 VALUES ('{topic}', '{project}', 't', now());
             INSERT INTO topic_states (id, project_id, topic_id, version,
                 source_span_id, observed_at, recorded_at)
                 VALUES ('{state}', '{project}', '{topic}', 1, '{span}', now(), now());
             UPDATE topics SET current_state_id = '{state}', current_version = 1
              WHERE id = '{topic}';
             {set}",
            evidence.len()
        )))
        .execute(pool)
        .await
        .expect("write a state the way V10 stored it");
    }

    async fn signal_columns(pool: &sqlx::PgPool) -> i64 {
        sqlx::query_scalar(
            "SELECT count(*) FROM information_schema.columns
              WHERE table_schema = current_schema() AND table_name = 'topic_states'
                AND column_name IN ('importance', 'worth_positive', 'worth_negative',
                                    'access_count', 'last_accessed_at')",
        )
        .fetch_one(pool)
        .await
        .expect("read the catalogue")
    }

    // A state that holds a signal, beside one that does not.
    let scratch = database_left_at(database, workspace, NAME, 10).await;
    state_at_v10(&scratch, "never touched", None).await;
    state_at_v10(&scratch, "read three times", Some("access_count = 3")).await;

    assert!(
        pamin_store::migrate::run(&scratch).await.is_err(),
        "a state holding a signal must stop the migration"
    );
    assert_eq!(
        signal_columns(&scratch).await,
        5,
        "a refused migration must leave every column in place"
    );
    let kept: i32 = sqlx::query_scalar("SELECT max(access_count) FROM topic_states")
        .fetch_one(&scratch)
        .await
        .expect("read the signal back");
    assert_eq!(kept, 3);
    scratch.close().await;

    // The same data with the signal left at its default migrates, and every
    // state still reads back.
    let scratch = database_left_at(database, workspace, NAME, 10).await;
    state_at_v10(&scratch, "never touched", None).await;
    state_at_v10(&scratch, "read three times", None).await;

    pamin_store::migrate::run(&scratch)
        .await
        .expect("states holding only defaults should migrate");
    assert_eq!(
        signal_columns(&scratch).await,
        0,
        "the columns should be gone once nothing is in them"
    );

    let mut read: Vec<String> = Vec::new();
    let projects: Vec<uuid::Uuid> = sqlx::query_scalar("SELECT id FROM projects")
        .fetch_all(&scratch)
        .await
        .expect("projects");
    for project in projects {
        let states = repository::all_current_topic_states(&scratch, project.into())
            .await
            .expect("read states through the store");
        read.extend(states.into_iter().map(|state| state.content));
    }
    read.sort();
    assert_eq!(read, vec!["never touched", "read three times"]);

    scratch.close().await;
    sqlx::query(AssertSqlSafe(format!("DROP DATABASE {NAME}")))
        .execute(database.pool())
        .await
        .expect("drop scratch database");
}

/// A scratch database migrated through `through` by an earlier build, and no
/// further.
///
/// The migration files are applied directly and recorded the way `refinery`
/// recorded them, which is the one way this crate accepts a database it did not
/// migrate itself: the runner adopts those books and applies only what comes
/// after. So this is both a database `refinery` could have produced and one a
/// build that stopped at `through` would have left -- which is what testing a
/// migration against existing data needs, and what a fresh workspace, already
/// at the newest schema, cannot give.
async fn database_left_at(
    database: &Database,
    workspace: &Workspace,
    name: &str,
    through: i32,
) -> sqlx::PgPool {
    sqlx::query(AssertSqlSafe(format!("DROP DATABASE IF EXISTS {name}")))
        .execute(database.pool())
        .await
        .expect("drop scratch database");
    sqlx::query(AssertSqlSafe(format!("CREATE DATABASE {name}")))
        .execute(database.pool())
        .await
        .expect("create scratch database");

    let mut server = workspace
        .read_server()
        .expect("read server record")
        .expect("workspace has a server");
    server.database = name.to_string();

    let scratch = sqlx::PgPool::connect(&server.url())
        .await
        .expect("connect to scratch database");

    // Applying the files directly is what makes this a database an earlier
    // runner produced, rather than one this runner produced and relabelled.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations");
    let mut files: Vec<(i32, String)> = std::fs::read_dir(&dir)
        .expect("migrations directory")
        .map(|entry| entry.expect("directory entry").file_name())
        .filter_map(|file| {
            let file = file.to_string_lossy().to_string();
            let version = file.strip_prefix('V')?.split_once("__")?.0.parse().ok()?;
            Some((version, file))
        })
        .filter(|(version, _)| *version <= through)
        .collect();
    files.sort();
    assert_eq!(
        files.len(),
        through as usize,
        "expected every migration up to V{through}"
    );

    for (_, file) in &files {
        let sql = std::fs::read_to_string(dir.join(file))
            .unwrap_or_else(|error| panic!("reading {file}: {error}"));
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(&scratch)
            .await
            .unwrap_or_else(|error| panic!("applying {file}: {error}"));
    }

    sqlx::query(
        "CREATE TABLE refinery_schema_history (
             version    INTEGER PRIMARY KEY,
             name       VARCHAR(255),
             applied_on VARCHAR(255),
             checksum   VARCHAR(255)
         )",
    )
    .execute(&scratch)
    .await
    .expect("create the old bookkeeping");

    for (version, file) in &files {
        sqlx::query(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)
             VALUES ($1, $2, '2026-01-01T00:00:00Z', '1234567890')",
        )
        .bind(version)
        .bind(file)
        .execute(&scratch)
        .await
        .expect("record an applied migration");
    }

    scratch
}

/// Every value written comes back from the column it was written to.
///
/// Most of these functions bind several arguments of one type in a row: three
/// timestamps on a topic state, two optional identifiers and two intervals on
/// an edge, three strings on a piece of evidence. Two of those swapped compiles,
/// runs, and returns a row -- so a test that asserts a row came back, or that a
/// count went up, passes just as happily with the values in each other's
/// columns.
///
/// So every value here is distinguishable from every other value of its type,
/// and every one is read back on its own. That is what makes this a check on
/// the mapping rather than on the plumbing, which is what it is for: the
/// mapping is being rewritten onto a different driver, one whose arguments are
/// positional and whose ordering the compiler cannot check.
async fn every_column_holds_what_was_written_to_it(database: &Database) {
    // Distinct, ordered, and none of them equal to now.
    let observed = OffsetDateTime::from_unix_timestamp(1_000_000_000).expect("observed");
    let valid_from = OffsetDateTime::from_unix_timestamp(1_100_000_000).expect("valid from");
    let valid_to = OffsetDateTime::from_unix_timestamp(1_200_000_000).expect("valid to");
    let edge_from = OffsetDateTime::from_unix_timestamp(1_300_000_000).expect("edge from");
    let edge_to = OffsetDateTime::from_unix_timestamp(1_400_000_000).expect("edge to");

    let project = repository::ensure_project(database.pool(), "columns")
        .await
        .expect("ensure project");
    let source = committed!(
        database,
        repository::ensure_source,
        project.id,
        SourceKind::Manual,
        "columns-locator"
    )
    .expect("ensure source");

    let evidence = committed!(
        database,
        repository::append_source_version,
        project.id,
        source,
        "the content",
        "the-hash",
        FilterDecision::Filtered,
        "the reason"
    )
    .expect("append source version");

    let read_back = repository::latest_source_version(database.pool(), project.id, source)
        .await
        .expect("latest source version")
        .expect("a version was written");
    assert_eq!(read_back.content, "the content");
    assert_eq!(read_back.content_hash, "the-hash");
    assert_eq!(read_back.filter_reason, "the reason");
    assert_eq!(read_back.filter_decision, FilterDecision::Filtered);
    assert_eq!(read_back.source_id, source);
    assert_eq!(read_back.project_id, project.id);

    let span = repository::append_source_span(
        database.pool(),
        project.id,
        evidence.id,
        3,
        11,
        Some("eng"),
        Some(0.75),
    )
    .await
    .expect("append source span");
    assert_eq!(span.byte_start, 3);
    assert_eq!(span.byte_end, 11);
    assert_eq!(span.detected_language.as_deref(), Some("eng"));

    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "columns_topic"
    )
    .expect("ensure topic");
    let partial = state_over_span(database, project.id, topic.id, &span).await;
    // The span's text, read back through the evidence it points into -- the
    // state has no copy of its own to read instead, so a span that is not the
    // whole evidence reads back as exactly the part it covers.
    assert_eq!(partial.content, " content");
    assert_eq!(partial.source_span_id, span.id);
    // The span's language, read back through the join -- and the first time
    // anything reads `source_spans` at all. The assertion above on `span` is on
    // the struct `append_source_span` built and handed back, so an INSERT that
    // dropped this column would have passed it; this one would not.
    assert_eq!(partial.language.as_deref(), Some("eng"));

    // The write path binds twenty-one arguments by position into one
    // statement: a version, a span, a state and the jobs. Every value of a
    // type here differs from every other of that type, and the content does
    // not say "content", which the grep below counts on.
    let mut transaction = database.pool().begin().await.expect("begin");
    let source = repository::ensure_source(
        &mut transaction,
        project.id,
        SourceKind::Manual,
        "columns-promoted",
    )
    .await
    .expect("ensure source");
    let locked = repository::lock_topic(&mut transaction, project.id, "columns_topic")
        .await
        .expect("lock topic")
        .expect("the topic exists");
    assert_eq!(locked.current, Some(partial.id));
    let (promoted, promoted_span, state) = repository::append_promoted(
        &mut transaction,
        project.id,
        source,
        &repository::Evidence {
            content: "a promoted memory",
            content_hash: "promoted-hash",
            decision: FilterDecision::Promoted,
            reason: "promoted reason",
            language: Some("swe"),
            language_confidence: Some(0.375),
        },
        &repository::Promotion {
            topic: &locked,
            observed_at: observed,
            validity: Validity {
                from: Some(valid_from),
                to: Some(valid_to),
            },
            owed: &[],
        },
    )
    .await
    .expect("append promoted");
    transaction.commit().await.expect("commit");

    let read_back = repository::latest_source_version(database.pool(), project.id, source)
        .await
        .expect("latest source version")
        .expect("a version was written");
    assert_eq!(read_back.id, promoted.id);
    assert_eq!(read_back.content, "a promoted memory");
    assert_eq!(read_back.content_hash, "promoted-hash");
    assert_eq!(read_back.filter_reason, "promoted reason");
    assert_eq!(read_back.filter_decision, FilterDecision::Promoted);
    assert_eq!(read_back.source_id, source);
    let stored_span: (uuid::Uuid, i32, i32, Option<String>, Option<f32>) = sqlx::query_as(
        "SELECT source_version_id, byte_start, byte_end, detected_language, language_confidence
           FROM source_spans WHERE id = $1",
    )
    .bind(promoted_span.id.0)
    .fetch_one(database.pool())
    .await
    .expect("the span was written");
    assert_eq!(
        stored_span,
        (
            promoted.id.0,
            0,
            "a promoted memory".len() as i32,
            Some("swe".to_string()),
            Some(0.375)
        )
    );

    let stored = repository::topic_state(database.pool(), project.id, topic.id, state.version)
        .await
        .expect("read topic state")
        .expect("the state was written");
    assert_eq!(stored.id, state.id);
    assert_eq!(stored.content, "a promoted memory");
    assert_eq!(stored.source_span_id, promoted_span.id);
    assert_eq!(stored.language.as_deref(), Some("swe"));
    assert_eq!(stored.observed_at, observed);
    assert_eq!(stored.validity.from, Some(valid_from));
    assert_eq!(stored.validity.to, Some(valid_to));
    assert!(
        stored.recorded_at > valid_to,
        "recorded_at should be now, not one of the stated instants"
    );
    assert_eq!(stored.supersedes, Some(partial.id));
    assert_eq!(stored.deleted_at, None);

    // An edge carrying every field that could be transposed with another.
    let other = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "columns_other"
    )
    .expect("ensure topic");
    let claim = EdgeClaim {
        kind: EdgeKind::DependsOn,
        derivation: Derivation::Model,
        confidence: 0.625,
        validity: Validity {
            from: Some(edge_from),
            to: Some(edge_to),
        },
        caused_by_topic_state: Some(state.id),
    };
    graph::assert_edge(database.pool(), project.id, topic.id, other.id, &claim)
        .await
        .expect("assert edge");

    let relationship = graph::find_relationship(
        database.pool(),
        project.id,
        topic.id,
        other.id,
        EdgeKind::DependsOn,
    )
    .await
    .expect("find relationship")
    .expect("the edge was asserted");
    assert_eq!(relationship.from_topic, topic.id);
    assert_eq!(relationship.to_topic, other.id);

    let version = graph::live_version(database.pool(), relationship.id)
        .await
        .expect("live version")
        .expect("the edge is live");
    assert_eq!(version.derivation, Derivation::Model);
    assert_eq!(version.confidence, 0.625);
    assert_eq!(version.validity.from, Some(edge_from));
    assert_eq!(version.validity.to, Some(edge_to));
    assert_eq!(version.caused_by_topic_state, Some(state.id));
    assert_eq!(version.invalidated_at, None);
    assert_eq!(version.tombstone_reason, None);
    assert_eq!(version.supersedes, None);

    // `needle` and `limit` are the other pair that would compile transposed.
    let matches = repository::grep_evidence(database.pool(), project.id, "content", false, 1)
        .await
        .expect("grep evidence");
    assert_eq!(matches.len(), 1, "the limit is the limit, not the needle");
    assert_eq!(matches[0].source_version.content, "the content");
    assert_eq!(matches[0].locator, "columns-locator");
    assert_eq!(
        matches[0].offset, 4,
        "the offset is a zero-based byte offset of the needle, not SQL's one-based position"
    );
}

/// The stored current-state pointer agrees with the ledger after every write.
///
/// V1 argued against storing which state is current, and against a flag on
/// `topic_states` it was right: every write path has to clear the old row and
/// set the new one, and one that forgets leaves two rows both claiming to be
/// current. Moving the pointer to the parent removes the contradiction -- one
/// column on one row cannot disagree with itself -- but not the obligation.
/// Two write paths maintain it, and a third that forgot would show up as a
/// search returning content the topic no longer has, with nothing to say so.
///
/// So this walks every transition and compares the pointer against the answer
/// computed from the ledger each time: append, append again, delete the
/// current one, delete a middle one, delete the last surviving one, append
/// after that. Reading the column back is the point -- a helper that recomputed
/// it would agree with itself and prove nothing.
async fn the_current_state_pointer_follows_every_write(database: &Database) {
    let project = repository::ensure_project(database.pool(), "pointer")
        .await
        .expect("ensure project");
    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "pointer_topic"
    )
    .expect("ensure topic");

    // A topic with no states yet points nowhere.
    assert_pointer_matches_the_ledger(database, project.id, topic.id, None).await;

    let first = write_state(database, project.id, topic.id, "pointer-1", "first").await;
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(first.version)).await;

    let second = write_state(database, project.id, topic.id, "pointer-2", "second").await;
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(second.version)).await;

    let third = write_state(database, project.id, topic.id, "pointer-3", "third").await;
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(third.version)).await;

    // Deleting the current one falls back to the newest survivor.
    assert!(
        committed!(
            database,
            repository::soft_delete_topic_state,
            topic.id,
            third.version
        )
        .expect("soft delete the current state")
    );
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(second.version)).await;

    // Deleting one that is not current leaves the pointer alone -- which the
    // naive fix of "step back to the predecessor" would get wrong.
    assert!(
        committed!(
            database,
            repository::soft_delete_topic_state,
            topic.id,
            first.version
        )
        .expect("soft delete a state that is not current")
    );
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(second.version)).await;

    // Deleting the last survivor leaves the topic resolving to nothing.
    assert!(
        committed!(
            database,
            repository::soft_delete_topic_state,
            topic.id,
            second.version
        )
        .expect("soft delete the last survivor")
    );
    assert_pointer_matches_the_ledger(database, project.id, topic.id, None).await;

    // And an append brings it back.
    let fourth = write_state(database, project.id, topic.id, "pointer-4", "fourth").await;
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(fourth.version)).await;

    // Deleting a version that is already deleted changes nothing.
    assert!(
        !committed!(
            database,
            repository::soft_delete_topic_state,
            topic.id,
            first.version
        )
        .expect("soft delete an already deleted state")
    );
    assert_pointer_matches_the_ledger(database, project.id, topic.id, Some(fourth.version)).await;

    // Nothing above should have left work for the repair path.
    let repaired = repository::repair_current_state_pointers(database.pool(), project.id)
        .await
        .expect("repair pointers");
    assert_eq!(
        repaired, 0,
        "the write paths left {repaired} topics pointing at the wrong state"
    );
}

/// `append_evidence` writes the version and the span over all of it, and both
/// read back as written.
///
/// The span is inserted by the same statement as the version it points into,
/// so this reads the span's row itself rather than the struct handed back, and
/// reads the state cut from it: multi-byte content, so a span measured in
/// characters rather than bytes would cut it short.
async fn evidence_and_the_span_over_it_are_one_write(database: &Database) {
    let project = repository::ensure_project(database.pool(), "evidence")
        .await
        .expect("ensure project");
    let content = "ugnen når kon tolv, och håller den";
    let mut transaction = database.pool().begin().await.expect("begin");
    let source = repository::ensure_source(
        &mut transaction,
        project.id,
        SourceKind::Manual,
        "evidence-locator",
    )
    .await
    .expect("ensure source");
    let (evidence, span) = repository::append_evidence(
        &mut transaction,
        project.id,
        source,
        &repository::Evidence {
            content,
            content_hash: "evidence-hash",
            decision: FilterDecision::Promoted,
            reason: "evidence reason",
            language: Some("swe"),
            language_confidence: Some(0.5),
        },
    )
    .await
    .expect("append evidence");
    transaction.commit().await.expect("commit");

    let stored: (
        uuid::Uuid,
        uuid::Uuid,
        i32,
        i32,
        Option<String>,
        Option<f32>,
    ) = sqlx::query_as(
        "SELECT id, source_version_id, byte_start, byte_end, detected_language,
                language_confidence
         FROM source_spans WHERE source_version_id = $1",
    )
    .bind(evidence.id.0)
    .fetch_one(database.pool())
    .await
    .expect("the span was written with its version");
    assert_eq!(
        stored,
        (
            span.id.0,
            evidence.id.0,
            0,
            content.len() as i32,
            Some("swe".to_string()),
            Some(0.5)
        )
    );
    assert_eq!(span.byte_end as usize, content.len());

    let read_back = repository::latest_source_version(database.pool(), project.id, source)
        .await
        .expect("latest source version")
        .expect("the version was written");
    assert_eq!(read_back.id, evidence.id);
    assert_eq!(read_back.version, 1);
    assert_eq!(read_back.content, content);
    assert_eq!(read_back.content_hash, "evidence-hash");
    assert_eq!(read_back.filter_decision, FilterDecision::Promoted);
    assert_eq!(read_back.filter_reason, "evidence reason");

    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "evidence_topic"
    )
    .expect("ensure topic");
    let stored = state_over_span(database, project.id, topic.id, &span).await;
    assert_eq!(stored.content, content);
    assert_eq!(stored.language.as_deref(), Some("swe"));
}

/// `append_promoted` writes what the write path's four statements wrote, and
/// queues its work with the conflict behaviour `enqueue` has.
///
/// The rows are read back rather than the structs handed back, so a state
/// pointing at a span the statement did not write, or a pointer left behind,
/// fails here. The queue half is the part the shared fragment exists for: a
/// write for a subject whose job a worker already holds must take the claim
/// away, or the worker completes work requested after it started reading.
async fn a_promoted_write_is_one_statement_after_its_locks(database: &Database) {
    let project = repository::ensure_project(database.pool(), "promoted")
        .await
        .expect("ensure project");

    let mut transaction = database.pool().begin().await.expect("begin");
    assert!(
        repository::lock_topic(&mut transaction, project.id, "promoted_topic")
            .await
            .expect("lock topic")
            .is_none(),
        "a topic nobody created was found"
    );
    let created = repository::create_topic_named(
        &mut transaction,
        project.id,
        "promoted_topic",
        "promoted topic",
        2,
    )
    .await
    .expect("create topic");
    assert!(created.current.is_none(), "a new topic points at a state");
    transaction.commit().await.expect("commit");
    let topic = created.topic.id;
    assert_eq!(
        repository::topics_named_by(database.pool(), project.id, &["promoted topic".to_string()])
            .await
            .expect("names"),
        vec![topic],
        "the topic was created without its name row"
    );

    // Racing a creation that already happened finds the winner, locked.
    let mut transaction = database.pool().begin().await.expect("begin");
    let again = repository::create_topic_named(
        &mut transaction,
        project.id,
        "promoted_topic",
        "promoted topic",
        2,
    )
    .await
    .expect("create topic again");
    transaction.commit().await.expect("commit");
    assert_eq!(again.topic.id, topic, "a second topic took the name");

    // A worker holds the topic's sync job when the write arrives.
    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::SyncTopicIndex,
        Some(topic.0),
    )
    .await
    .expect("enqueue");
    let held = jobs::claim(database.pool(), project.id, "worker", 1, &JobKind::ALL)
        .await
        .expect("claim");
    assert_eq!(held.len(), 1);

    let mut states = Vec::new();
    for (round, content) in ["först", "sedan"].into_iter().enumerate() {
        let mut transaction = database.pool().begin().await.expect("begin");
        let source = repository::ensure_source(
            &mut transaction,
            project.id,
            SourceKind::Manual,
            "manual:promoted_topic",
        )
        .await
        .expect("ensure source");
        let locked = repository::lock_topic(&mut transaction, project.id, "promoted_topic")
            .await
            .expect("lock topic")
            .expect("the topic exists");
        assert_eq!(
            locked.current,
            states.last().map(|state: &pamin_core::TopicState| state.id),
            "the lock read a pointer other than the newest state"
        );
        let (version, span, state) = repository::append_promoted(
            &mut transaction,
            project.id,
            source,
            &repository::Evidence {
                content,
                content_hash: "promoted-hash",
                decision: FilterDecision::Promoted,
                reason: "promoted reason",
                language: Some("swe"),
                language_confidence: Some(0.5),
            },
            &repository::Promotion {
                topic: &locked,
                observed_at: OffsetDateTime::now_utc(),
                validity: Validity::ALWAYS,
                owed: &[JobKind::SyncTopicIndex, JobKind::DeriveMentions],
            },
        )
        .await
        .expect("append promoted");
        transaction.commit().await.expect("commit");

        assert_eq!(version.version, round as u32 + 1);
        assert_eq!(state.version, round as u32 + 1);
        assert_eq!(state.source_span_id, span.id);
        assert_eq!(state.supersedes, locked.current);
        let (span_version, byte_end): (uuid::Uuid, i32) =
            sqlx::query_as("SELECT source_version_id, byte_end FROM source_spans WHERE id = $1")
                .bind(span.id.0)
                .fetch_one(database.pool())
                .await
                .expect("the span was written");
        assert_eq!(span_version, version.id.0);
        assert_eq!(byte_end as usize, content.len());
        let stored = repository::topic_state(database.pool(), project.id, topic, state.version)
            .await
            .expect("read topic state")
            .expect("the state was written");
        assert_eq!(stored.id, state.id);
        assert_eq!(stored.content, content);
        assert_eq!(stored.language.as_deref(), Some("swe"));
        assert_eq!(stored.supersedes, state.supersedes);
        assert_pointer_matches_the_ledger(database, project.id, topic, Some(state.version)).await;
        states.push(state);
    }

    assert!(
        jobs::complete(database.pool(), &[&held[0]], "worker")
            .await
            .expect("complete")
            .is_empty(),
        "a job requested again by a write was completed by the attempt before it"
    );
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("count pending"),
        2,
        "two writes owing two kinds for one subject should leave two rows"
    );
}

/// Reads the stored pointer and checks it against the version it should hold.
async fn assert_pointer_matches_the_ledger(
    database: &Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
    expected: Option<u32>,
) {
    let stored: (Option<uuid::Uuid>, Option<i32>) =
        sqlx::query_as("SELECT current_state_id, current_version FROM topics WHERE id = $1")
            .bind(topic.0)
            .fetch_one(database.pool())
            .await
            .expect("read the current-state pointer");

    let (pointed_at, version) = stored;

    // Through the reader the search path uses, not only the column. The graph
    // channel resolves every neighbour it finds through `current_states_of`,
    // and that statement joins `topics`, where an unqualified column list is
    // ambiguous -- an error PostgreSQL raises when the statement runs, so
    // nothing short of running it says so.
    let resolved = repository::current_states_of(database.pool(), project, &[topic])
        .await
        .expect("resolve the topic to its current state");
    assert_eq!(
        resolved.first().map(|state| state.id.0),
        pointed_at,
        "the topic resolves to a different state than its pointer names"
    );
    // And through the search path's form, which also names the topic: the
    // same state, beside the name the topic row holds.
    let named = repository::current_states_named(database.pool(), project, &[topic])
        .await
        .expect("resolve the topic to its current state and name");
    let name: String = sqlx::query_scalar("SELECT name FROM topics WHERE id = $1")
        .bind(topic.0)
        .fetch_one(database.pool())
        .await
        .expect("read the topic's name");
    assert_eq!(
        named
            .first()
            .map(|(named, state)| (named.clone(), state.id.0)),
        pointed_at.map(|state| (name, state)),
        "the named lookup disagrees with the pointer or the name"
    );
    assert_eq!(
        version.map(|version| version as u32),
        expected,
        "the topic points at version {version:?}, expected {expected:?}"
    );

    match expected {
        None => assert!(
            pointed_at.is_none(),
            "a topic resolving to nothing still points at a state"
        ),
        Some(expected) => {
            let state = repository::topic_state(database.pool(), project, topic, expected)
                .await
                .expect("load the expected state")
                .expect("the expected state exists");
            assert_eq!(
                pointed_at,
                Some(state.id.0),
                "the pointer names a different state than its version does"
            );
            assert!(
                state.deleted_at.is_none(),
                "a topic points at a state that has been deleted"
            );
        }
    }
}

/// A walk through two adjacent hubs returns each topic once, at its distance.
///
/// This guards the rewrite, not the reason for it. Moving the walk out of a
/// recursive query and into a hop-at-a-time loop introduced a visited set and a
/// per-hop merge, which is where a rewrite like this goes wrong: a topic
/// reachable by several routes coming back several times, or coming back at the
/// distance of the route that happened to be found first rather than the
/// shortest. Neither shows up in the fixtures the other expansion tests use,
/// because they are too small for a topic to have two routes.
///
/// What it does not show is the cost the rewrite is for. The recursive form
/// materialised every edge in the project, twice, before the walk began, and
/// had nowhere to put a bound on the frontier -- but its output was deduplicated
/// at the end, so at any size a test can build it answers the same thing. That
/// difference is real and is not visible from here.
async fn two_adjacent_hubs_do_not_multiply(database: &Database) {
    const SPOKES: usize = 120;

    let project = repository::ensure_project(database.pool(), "hubs")
        .await
        .expect("ensure project");

    let topic = |name: String| async move {
        committed!(database, repository::ensure_topic, project.id, &name)
            .expect("ensure topic")
            .id
    };

    let left = topic("hub_left".to_string()).await;
    let right = topic("hub_right".to_string()).await;

    let mut edges = vec![(left, right, EdgeClaim::explicit(EdgeKind::RelatedTo))];
    for spoke in 0..SPOKES {
        let on_left = topic(format!("left_spoke_{spoke}")).await;
        let on_right = topic(format!("right_spoke_{spoke}")).await;
        edges.push((left, on_left, EdgeClaim::explicit(EdgeKind::RelatedTo)));
        edges.push((right, on_right, EdgeClaim::explicit(EdgeKind::RelatedTo)));
    }

    graph::assert_edges(database.pool(), project.id, &edges)
        .await
        .expect("build the hubs");

    let walked = graph::expand(
        &mut *connection(database).await,
        project.id,
        &[left],
        &Expansion::to_depth(2),
    )
    .await
    .expect("walk two hops from a hub");

    // Every topic once, at its shortest distance: the far hub and this hub's
    // own spokes at one, the far hub's spokes at two. The near hub itself is
    // the seed and nothing reaches it independently, so it is absent.
    assert_eq!(
        walked.len(),
        1 + SPOKES * 2,
        "the walk returned {} entries for {} reachable topics",
        walked.len(),
        1 + SPOKES * 2
    );

    let mut seen = std::collections::HashSet::new();
    for neighbor in &walked {
        assert!(
            seen.insert(neighbor.topic),
            "a topic came back more than once, so paths were enumerated rather than nodes"
        );
        assert!(
            neighbor.topic != left,
            "the seed came back as its own neighbour"
        );
    }

    assert_eq!(
        walked.iter().filter(|n| n.hops == 1).count(),
        1 + SPOKES,
        "one hop reaches the far hub and this hub's own spokes"
    );
    assert_eq!(
        walked.iter().filter(|n| n.hops == 2).count(),
        SPOKES,
        "two hops reaches the far hub's spokes and nothing further"
    );
}

/// What a topic says now comes from the pointer, at any length of history.
///
/// The sensory filter asks this before every write, to decide whether the
/// content it was handed is what the topic already says. It used to be answered
/// by reading *every* version of the topic, in order, into memory and taking
/// the last -- three round trips and a scan proportional to how often that
/// topic has been edited, to compare one string. `topics.current_state_id`
/// answers it directly, which is what it was added for.
///
/// The two differ once a state is soft deleted, because the pointer is not
/// repaired when one is: reading the versions skipped the deleted state and
/// returned the newest survivor, while the pointer still names the deleted one.
/// Following the pointer is what `current_states_of` does, so this is the
/// reading the search path already has, and having the filter and the search
/// disagree about what a topic says is worse than either answer.
async fn what_a_topic_says_now_is_one_lookup(database: &Database) {
    let project = repository::ensure_project(database.pool(), "saysnow")
        .await
        .expect("ensure project");

    assert_eq!(
        repository::current_content(database.pool(), project.id, "never_written")
            .await
            .expect("look up a topic that does not exist"),
        None,
        "a topic nobody has written has nothing to compare against"
    );

    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "edited_often"
    )
    .expect("ensure topic")
    .id;

    // Enough history that reading all of it would be a different answer from
    // reading none of it.
    for round in 0..25 {
        write_state(
            database,
            project.id,
            topic,
            &format!("note-{round}"),
            &format!("what the topic said in round {round}"),
        )
        .await;
    }

    assert_eq!(
        repository::current_content(database.pool(), project.id, "edited_often")
            .await
            .expect("look up the current content")
            .as_deref(),
        Some("what the topic said in round 24"),
        "the newest state is what the topic says now"
    );
}

/// A completion belongs to one claim, not to whoever happens to hold the job.
///
/// The worker is one string per process -- host and pid -- so two attempts by
/// the same process are indistinguishable by it. That matters because a lease
/// expires on a timer rather than on the worker going away: a job that outruns
/// its minute is claimed again while the first attempt is still working, and
/// when that attempt finishes it must not mark the second one done.
///
/// Not a hypothetical. Building the vector graph over a sealed segment takes
/// longer than the lease, and the server's upkeep loop comes round every five
/// seconds, so a compaction is re-claimed by the same process as a matter of
/// course.
///
/// Before the completion named its claim, the last assertion here failed: the
/// stale attempt completed the fresh one, and the work the fresh claim stood
/// for was recorded as done without being run.
async fn a_completion_names_the_claim_it_belongs_to(database: &Database) {
    let project = repository::ensure_project(database.pool(), "reclaim")
        .await
        .expect("ensure project");
    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "reclaimed_topic"
    )
    .expect("ensure topic")
    .id;

    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::SyncTopicIndex,
        Some(topic.0),
    )
    .await
    .expect("enqueue");

    // One process, one worker string, for both attempts.
    const WORKER: &str = "the-only-worker";

    let first = jobs::claim(database.pool(), project.id, WORKER, 1, &JobKind::ALL)
        .await
        .expect("claim");
    assert_eq!(first.len(), 1);

    // The lease running out, without waiting a minute for it. The lease is
    // `available_at` and nothing else, so this is exactly what expiry is.
    sqlx::query("UPDATE index_jobs SET available_at = $1 WHERE id = $2")
        .bind(time::OffsetDateTime::now_utc() - std::time::Duration::from_secs(1))
        .bind(first[0].id.0)
        .execute(database.pool())
        .await
        .expect("expire the lease");

    let second = jobs::claim(database.pool(), project.id, WORKER, 1, &JobKind::ALL)
        .await
        .expect("claim again after the lease expired");
    assert_eq!(second.len(), 1, "an expired claim is claimable again");
    assert_eq!(second[0].id, first[0].id);
    assert_eq!(
        second[0].attempts, 2,
        "the second claim is a second attempt"
    );
    assert!(
        first[0].claimed_at.is_some() && second[0].claimed_at.is_some(),
        "a claimed job is held"
    );
    assert_ne!(
        second[0].claimed_at, first[0].claimed_at,
        "two claims of one job are two different claims"
    );

    assert!(
        jobs::complete(database.pool(), &[&first[0]], WORKER)
            .await
            .expect("complete")
            .is_empty(),
        "an attempt whose lease expired completed the attempt that replaced it"
    );

    assert_eq!(
        jobs::complete(database.pool(), &[&second[0]], WORKER)
            .await
            .expect("complete"),
        vec![second[0].id],
        "the claim that still holds the job could not complete it"
    );
}

/// A claim for some kinds takes those kinds' work and nobody else's.
///
/// The claim names each kind's priority beside the kind, so the queue's index
/// can find a kind's rows without reading everyone else's. That holds only
/// while the priority a row was queued with is the one the claim asks for, and
/// a claim that asked for the wrong number would find nothing and report the
/// queue empty -- so each kind is claimed alone, with every other kind owed
/// beside it.
async fn each_kind_is_claimed_by_its_own_priority(database: &Database) {
    let project = repository::ensure_project(database.pool(), "claim-by-kind")
        .await
        .expect("ensure project");
    let subject = uuid::Uuid::now_v7();
    for kind in JobKind::ALL {
        let subject = (kind != JobKind::OptimizeIndex).then_some(subject);
        jobs::enqueue(database.pool(), project.id, kind, subject)
            .await
            .expect("enqueue");
    }

    for kind in JobKind::ALL {
        let claimed = jobs::claim(database.pool(), project.id, "by-kind", 64, &[kind])
            .await
            .expect("claim one kind");
        assert_eq!(
            claimed.iter().map(|job| job.kind).collect::<Vec<_>>(),
            vec![kind],
            "a claim for {kind} alone"
        );
    }
}

/// What the outbox has to get right for the projection to stay correct.
///
/// Four properties, each of which fails silently if it is wrong -- the queue
/// keeps working and the projection quietly stops matching the ledger:
///
///   * repeated requests for one subject coalesce, or fourteen edits to one
///     topic cost fourteen embeddings;
///   * a request made *while* a job is running is not swallowed by that job's
///     completion, or the last write before a completion is never indexed;
///   * a claim expires, or a worker that dies holding a job takes the work with
///     it;
///   * attempts run out, or one poisoned job becomes a worker that never does
///     anything else.
async fn the_outbox_coalesces_claims_and_survives_a_lost_worker(database: &Database) {
    let project = repository::ensure_project(database.pool(), "outbox")
        .await
        .expect("ensure project");
    let topic = committed!(
        database,
        repository::ensure_topic,
        project.id,
        "outbox_topic"
    )
    .expect("ensure topic")
    .id;

    // Coalescing: three requests for the same subject are one row.
    for _ in 0..3 {
        jobs::enqueue(
            database.pool(),
            project.id,
            JobKind::SyncTopicIndex,
            Some(topic.0),
        )
        .await
        .expect("enqueue");
    }
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("count pending"),
        1,
        "requests for one subject should coalesce onto one row"
    );

    // A different kind for the same subject is different work.
    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::DeriveMentions,
        Some(topic.0),
    )
    .await
    .expect("enqueue a second kind");
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("count pending"),
        2
    );
    // The write path's count stops at the bound it compares against, and is
    // exact below it.
    for (cap, counted) in [(0, 0), (1, 1), (2, 2), (5, 2)] {
        assert_eq!(
            jobs::pending_up_to(database.pool(), project.id, cap)
                .await
                .expect("count pending up to a cap"),
            counted,
            "two owed, counted up to {cap}"
        );
    }

    // Priority decides what a worker sees first: syncing the index for a memory
    // just written comes before deriving its edges.
    let claimed = jobs::claim(database.pool(), project.id, "worker-a", 1, &JobKind::ALL)
        .await
        .expect("claim");
    assert_eq!(claimed.len(), 1);
    assert_eq!(claimed[0].kind, JobKind::SyncTopicIndex);
    assert_eq!(claimed[0].subject, Some(topic.0));
    assert_eq!(
        claimed[0].attempts, 1,
        "attempts count at claim, not at failure"
    );

    // A claimed job is not handed to anyone else.
    let contended = jobs::claim(database.pool(), project.id, "worker-b", 10, &JobKind::ALL)
        .await
        .expect("claim again");
    assert!(
        contended.iter().all(|job| job.id != claimed[0].id),
        "a claimed job was handed to a second worker"
    );

    // The second worker finishes what it did get, so the rest of this is about
    // one job rather than two.
    let finished = jobs::complete(
        database.pool(),
        &contended.iter().collect::<Vec<_>>(),
        "worker-b",
    )
    .await
    .expect("complete");
    assert_eq!(finished.len(), contended.len());

    // A request arriving while the job runs is not swallowed by its completion.
    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::SyncTopicIndex,
        Some(topic.0),
    )
    .await
    .expect("enqueue during processing");

    assert!(
        jobs::complete(database.pool(), &[&claimed[0]], "worker-a")
            .await
            .expect("complete")
            .is_empty(),
        "a job requested again mid-flight must not be completed by the attempt \
         that was already running"
    );

    // So it is still there, and claimable.
    let requeued = jobs::claim(database.pool(), project.id, "worker-a", 1, &JobKind::ALL)
        .await
        .expect("claim the re-requested job");
    assert_eq!(requeued.len(), 1);
    assert_eq!(requeued[0].id, claimed[0].id);
    assert_eq!(
        requeued[0].attempts, 1,
        "reviving a job resets its attempts"
    );

    assert_eq!(
        jobs::complete(database.pool(), &[&requeued[0]], "worker-a")
            .await
            .expect("complete"),
        vec![requeued[0].id],
        "a job nobody re-requested completes"
    );

    // Completing deletes the row: nothing is owed, so nothing is kept.
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("pending"),
        0,
        "a completed job should leave no row behind"
    );

    // And a later request is not swallowed by the work already done: it owes
    // the work again, as a new row in the state a fresh request starts in.
    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::SyncTopicIndex,
        Some(topic.0),
    )
    .await
    .expect("enqueue after completion");
    let revived = jobs::claim(database.pool(), project.id, "worker-a", 1, &JobKind::ALL)
        .await
        .expect("claim revived");
    assert_eq!(
        revived.len(),
        1,
        "a request after completion should owe the work again"
    );
    assert_ne!(
        revived[0].id, claimed[0].id,
        "the completed row should be gone, so this is a new one"
    );
    assert_eq!(
        revived[0].attempts, 1,
        "a new request starts its attempts afresh"
    );

    // Attempts run out, and the job is then left pending with its error rather
    // than coming round again. Every round here is a real claim and a real
    // failure; the update in between stands for the retry delay elapsing, which
    // is an hour and not something to wait for.
    let mut failing = revived;
    for round in 1..=pamin_core::MAX_ATTEMPTS {
        assert_eq!(
            failing.len(),
            1,
            "round {round}: a job below its attempt limit should still be claimable"
        );
        jobs::fail(
            database.pool(),
            &failing[0],
            "worker-a",
            "the index was unreachable",
        )
        .await
        .expect("record a failure");

        sqlx::query("UPDATE index_jobs SET available_at = now() WHERE id = $1")
            .bind(failing[0].id.0)
            .execute(database.pool())
            .await
            .expect("let the retry delay elapse");

        failing = jobs::claim(database.pool(), project.id, "worker-a", 1, &JobKind::ALL)
            .await
            .expect("claim after a failure");
    }

    assert!(
        failing.is_empty(),
        "a job that has used its attempts should not be handed out again"
    );

    let stuck = jobs::exhausted(database.pool(), project.id)
        .await
        .expect("read exhausted jobs");
    assert_eq!(
        stuck.len(),
        1,
        "the job that used its attempts should be listed"
    );
    assert_eq!(stuck[0].1, "the index was unreachable");

    assert_eq!(
        jobs::replay(database.pool(), project.id)
            .await
            .expect("replay"),
        1
    );
    assert!(
        jobs::exhausted(database.pool(), project.id)
            .await
            .expect("read exhausted jobs")
            .is_empty(),
        "replaying should put the job back in the ordinary queue"
    );

    // Leave the project clean for anything that counts pending work later.
    let outstanding = jobs::claim(database.pool(), project.id, "worker-a", 100, &JobKind::ALL)
        .await
        .expect("drain");
    jobs::complete(
        database.pool(),
        &outstanding.iter().collect::<Vec<_>>(),
        "worker-a",
    )
    .await
    .expect("complete");
    assert_eq!(
        jobs::pending(database.pool(), project.id)
            .await
            .expect("count pending"),
        0
    );
}

/// What a memory no longer says stops being claimed, and nothing else moves.
///
/// Deriving edges only ever asserted them. Rewriting a memory from "uses
/// argo_cd" to "uses flux" therefore kept the edge to `argo_cd` for ever, and
/// `why[]` cited a path the content it quotes does not support. Nothing in the
/// suite could see it, because every existing assertion is about an edge being
/// added.
///
/// Three things have to hold at once, and each is a way the obvious fix goes
/// wrong: the name that went away is closed, the name that stayed is not
/// touched -- closing and re-asserting it would churn the ledger on every
/// write -- and an edge somebody asserted by hand survives, because it is their
/// claim and not this memory's.
/// A worker draining one project leaves every other project's queue alone.
///
/// The handlers that run a claimed job read the ledger under the project their
/// engine was opened for, not the project the row names, so a job taken from
/// somewhere else is run against the wrong project: the topic resolves to
/// nothing, the handler concludes it has no current state and unindexes it, and
/// the row is marked complete. The queue drains and the memory is never
/// indexed. Nothing raises, and only a second project makes it reachable —
/// which is why it survived until one process began serving many.
async fn one_projects_worker_never_takes_anothers_work(database: &Database) {
    let mine = repository::ensure_project(database.pool(), "tenant-mine")
        .await
        .expect("ensure project");
    let theirs = repository::ensure_project(database.pool(), "tenant-theirs")
        .await
        .expect("ensure project");

    let mut queued = Vec::new();
    for (project, name) in [(mine.id, "mine_topic"), (theirs.id, "theirs_topic")] {
        let topic = committed!(database, repository::ensure_topic, project, name)
            .expect("ensure topic")
            .id;
        jobs::enqueue(
            database.pool(),
            project,
            JobKind::SyncTopicIndex,
            Some(topic.0),
        )
        .await
        .expect("enqueue");
        queued.push((project, topic));
    }

    // A batch far larger than what this project owes, so anything it is allowed
    // to see it takes.
    let claimed = jobs::claim(database.pool(), mine.id, "worker-mine", 100, &JobKind::ALL)
        .await
        .expect("claim");
    assert!(
        claimed.iter().all(|job| job.project_id == mine.id),
        "a worker for one project claimed another project's job"
    );
    assert!(
        claimed.iter().any(|job| job.subject == Some(queued[0].1.0)),
        "the worker did not claim its own project's job"
    );

    // And the other project's work is still there to be done, unclaimed.
    assert_eq!(
        jobs::pending(database.pool(), theirs.id)
            .await
            .expect("count pending"),
        1,
        "another project's queue was drained by this project's worker"
    );
    let left = jobs::claim(
        database.pool(),
        theirs.id,
        "worker-theirs",
        10,
        &JobKind::ALL,
    )
    .await
    .expect("claim");
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].subject, Some(queued[1].1.0));

    for (worker, owned) in [("worker-mine", &claimed), ("worker-theirs", &left)] {
        jobs::complete(database.pool(), &owned.iter().collect::<Vec<_>>(), worker)
            .await
            .expect("complete");
    }
}

/// The batched forms answer each topic as if it had been asked alone.
///
/// A cascade round restates many memories at once: their name lookups go in
/// one statement, told apart by the run each name matched, and their
/// retractions in another, each topic closing only what its own content
/// stopped naming. Two topics here keep different targets out of the same
/// three, so a retraction that mixed their lists up would close the wrong
/// edge of one of them.
async fn several_topics_restate_their_mentions_at_once(database: &Database) {
    let project = repository::ensure_project(database.pool(), "restate")
        .await
        .expect("ensure project");
    let mut topics = Vec::new();
    for name in ["left", "right", "alpha", "beta", "gamma"] {
        let topic = committed!(database, repository::ensure_topic, project.id, name)
            .expect("ensure topic")
            .id;
        repository::record_topic_name(database.pool(), project.id, topic, name, 1)
            .await
            .expect("record name");
        topics.push(topic);
    }
    let (left, right, alpha, beta, gamma) = (topics[0], topics[1], topics[2], topics[3], topics[4]);

    let matched = repository::names_matching(
        database.pool(),
        project.id,
        &[
            "alpha".to_string(),
            "gamma".to_string(),
            "nobody".to_string(),
        ],
    )
    .await
    .expect("names matching");
    let mut matched: Vec<(String, pamin_core::TopicId)> = matched;
    matched.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        matched,
        vec![("alpha".to_string(), alpha), ("gamma".to_string(), gamma)]
    );

    let mut edges = Vec::new();
    for from in [left, right] {
        let state = write_state(
            database,
            project.id,
            from,
            &format!("restate-{from}"),
            "alpha beta gamma",
        )
        .await;
        for to in [alpha, beta, gamma] {
            edges.push((
                from,
                to,
                EdgeClaim::derived(EdgeKind::Mentions, state.id, 0.5),
            ));
        }
    }
    graph::assert_edges(database.pool(), project.id, &edges)
        .await
        .expect("assert the derived edges");

    // `left` now names alpha alone; `right` names beta and gamma.
    let closed = graph::retract_derived_all(
        database.pool(),
        project.id,
        EdgeKind::Mentions,
        &[(left, vec![alpha]), (right, vec![beta, gamma])],
    )
    .await
    .expect("retract for both topics");
    assert_eq!(closed, 3, "left's beta and gamma, and right's alpha");

    for (from, to, live) in [
        (left, alpha, true),
        (left, beta, false),
        (left, gamma, false),
        (right, alpha, false),
        (right, beta, true),
        (right, gamma, true),
    ] {
        let relationship =
            graph::find_relationship(database.pool(), project.id, from, to, EdgeKind::Mentions)
                .await
                .expect("find relationship")
                .expect("the edge was asserted");
        let version = graph::live_version(database.pool(), relationship.id)
            .await
            .expect("live version");
        assert_eq!(
            version.is_some(),
            live,
            "the edge {from} -> {to} should be {}",
            if live { "live" } else { "closed" }
        );
    }
}

async fn a_derived_edge_the_content_stopped_making_is_closed(database: &Database) {
    let project = repository::ensure_project(database.pool(), "retraction")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["deploy", "argo_cd", "flux", "runbook"] {
        topics.push(
            committed!(database, repository::ensure_topic, project.id, name)
                .expect("ensure topic")
                .id,
        );
    }
    let (deploy, argo, flux, runbook) = (topics[0], topics[1], topics[2], topics[3]);

    let state = write_state(
        database,
        project.id,
        deploy,
        "retraction-1",
        "goes out through argo",
    )
    .await;
    let derived = |to| {
        (
            deploy,
            to,
            EdgeClaim::derived(EdgeKind::Mentions, state.id, 0.5),
        )
    };

    graph::assert_edges(database.pool(), project.id, &[derived(argo), derived(flux)])
        .await
        .expect("assert the derived edges");
    // Somebody's own claim, of the same kind and out of the same topic.
    graph::assert_edge(
        database.pool(),
        project.id,
        deploy,
        runbook,
        &EdgeClaim::explicit(EdgeKind::Mentions),
    )
    .await
    .expect("assert the explicit edge");

    let live = |to| async move {
        let relationship =
            graph::find_relationship(database.pool(), project.id, deploy, to, EdgeKind::Mentions)
                .await
                .expect("find relationship")
                .expect("the edge was asserted");
        graph::live_version(database.pool(), relationship.id)
            .await
            .expect("live version")
    };

    let kept_before = live(flux).await.expect("the kept edge is live");

    // The memory now names only `flux`.
    let closed = graph::retract_derived(
        database.pool(),
        project.id,
        deploy,
        EdgeKind::Mentions,
        &[flux],
    )
    .await
    .expect("retract what the content no longer says");
    assert_eq!(closed, 1, "exactly the edge that went away should close");

    assert!(
        live(argo).await.is_none(),
        "an edge the content stopped making is still claimed"
    );
    assert!(
        live(runbook).await.is_some(),
        "retracting derived edges closed one somebody asserted by hand"
    );

    let kept_after = live(flux).await.expect("the kept edge is still live");
    assert_eq!(
        kept_after.id, kept_before.id,
        "a name that is still there should not be closed and re-asserted"
    );

    // Closed, not deleted: the claim is retracted from here on rather than
    // declared never to have held, so a walk asked about a moment before the
    // retraction still reaches it.
    let relationship = graph::find_relationship(
        database.pool(),
        project.id,
        deploy,
        argo,
        EdgeKind::Mentions,
    )
    .await
    .expect("find relationship")
    .expect("the edge was asserted");
    let history = graph::edge_history(database.pool(), project.id, relationship.id)
        .await
        .expect("edge history");
    let retracted = history.last().expect("the edge has a version");
    assert_eq!(retracted.tombstone_reason, Some(TombstoneReason::Closed));

    let before = retracted
        .invalidated_at
        .expect("a closed version records when")
        - time::Duration::seconds(1);
    let reached = graph::expand(
        &mut *connection(database).await,
        project.id,
        &[deploy],
        &Expansion {
            depth: 1,
            at: Some(before),
            kinds: None,
            keep: None,
        },
    )
    .await
    .expect("expand at a moment before the retraction");
    assert!(
        reached.iter().any(|neighbor| neighbor.topic == argo),
        "what the memory said before is still true of before: {reached:?}"
    );
}

/// The walk is undirected, so an edge is reached from both ends. What it says
/// about itself must not depend on which end the walk started at, and that has
/// to hold past the first hop, where the arrival's `via` is no longer the seed.
///
/// Before `Neighbor::outbound` existed the only direction an arrival carried
/// was the traversal's, so every edge printed as pointing away from wherever
/// the walk happened to be standing. On this fixture that is wrong for three of
/// the four edges from at least one seed, and the failure is not visible from a
/// single walk -- each answer looks coherent on its own and contradicts the
/// others.
///
/// The fixture mixes directions deliberately:
///
/// ```text
///   rota  --depends_on-->  pipeline  <--depends_on--  scheduler
///                              |
///                          part_of
///                              v
///                          platform  --supersedes-->  legacy
/// ```
///
/// Two edges point into `pipeline` and one out of it, so a walk seeded there
/// meets both orientations in the same hop; `platform` and `legacy` put a
/// second and third hop behind it.
async fn an_edge_reads_the_same_direction_from_either_end(database: &Database) {
    let project = repository::ensure_project(database.pool(), "direction")
        .await
        .expect("ensure project");

    let mut id = std::collections::HashMap::new();
    for name in ["rota", "pipeline", "scheduler", "platform", "legacy"] {
        let topic =
            committed!(database, repository::ensure_topic, project.id, name).expect("ensure topic");
        write_state(
            database,
            project.id,
            topic.id,
            &format!("direction-{name}"),
            &format!("a durable claim about {name}"),
        )
        .await;
        id.insert(name, topic.id);
    }

    let edges = [
        ("rota", "pipeline", EdgeKind::DependsOn),
        ("scheduler", "pipeline", EdgeKind::DependsOn),
        ("pipeline", "platform", EdgeKind::PartOf),
        ("platform", "legacy", EdgeKind::Supersedes),
    ];
    for (from, to, kind) in edges {
        graph::assert_edge(
            database.pool(),
            project.id,
            id[from],
            id[to],
            &EdgeClaim::explicit(kind),
        )
        .await
        .unwrap_or_else(|e| panic!("{from} -> {to}: {e}"));
    }

    // What each arrival claims the edge is, resolved to a pair of endpoints.
    let ends = |n: &graph::Neighbor| {
        if n.outbound {
            (n.via, n.topic)
        } else {
            (n.topic, n.via)
        }
    };

    // Every seed, to the full depth of the component. Each walk meets a
    // different subset of the edges, and from a different side.
    let mut seen: std::collections::HashMap<(uuid::Uuid, uuid::Uuid), EdgeKind> =
        std::collections::HashMap::new();
    for seed in ["rota", "pipeline", "scheduler", "platform", "legacy"] {
        let reached = graph::expand(
            &mut *connection(database).await,
            project.id,
            &[id[seed]],
            &Expansion::to_depth(4),
        )
        .await
        .unwrap_or_else(|e| panic!("expand from {seed}: {e}"));

        assert!(
            reached.iter().any(|n| n.hops > 1),
            "the walk from {seed} should reach past one hop, or it tests nothing"
        );

        for neighbour in &reached {
            let (from, to) = ends(neighbour);
            let key = (from.0.min(to.0), from.0.max(to.0));
            // Whichever seed met this edge first fixes what it claims; every
            // later walk must agree, orientation included.
            if let Some(kind) = seen.insert(key, neighbour.kind) {
                assert_eq!(
                    kind, neighbour.kind,
                    "walking from {seed} changed what kind of edge this is"
                );
            }
            let expected = edges
                .iter()
                .find(|(f, t, _)| (id[f], id[t]) == (from, to))
                .map(|(f, t, k)| (*f, *t, *k));
            assert!(
                expected.is_some(),
                "walking from {seed} reported an edge {from:?} -> {to:?} that was \
                 never asserted; the reverse of it probably was"
            );
            assert_eq!(
                expected.unwrap().2,
                neighbour.kind,
                "walking from {seed}, the edge kind does not match the assertion"
            );
        }
    }
    assert_eq!(seen.len(), edges.len(), "every edge should have been met");

    // The case a caller actually asks: what depends on the pipeline. Both
    // answers are one hop away and the edges point opposite ways, so reading
    // direction off the traversal returns the pipeline depending on them.
    let around_pipeline = graph::expand(
        &mut *connection(database).await,
        project.id,
        &[id["pipeline"]],
        &Expansion::to_depth(1),
    )
    .await
    .expect("expand from pipeline");

    let mut dependents: Vec<_> = around_pipeline
        .iter()
        .filter(|n| n.kind == EdgeKind::DependsOn && ends(n).1 == id["pipeline"])
        .map(|n| ends(n).0)
        .collect();
    dependents.sort();
    let mut expected = vec![id["rota"], id["scheduler"]];
    expected.sort();
    assert_eq!(
        dependents, expected,
        "rota and scheduler depend on the pipeline; the pipeline depends on neither"
    );
    assert!(
        around_pipeline
            .iter()
            .any(|n| n.kind == EdgeKind::PartOf && ends(n) == (id["pipeline"], id["platform"])),
        "and the one edge that does point away from the pipeline still does"
    );
}

/// How many versions another project holds of one topic, one source and one
/// edge, for [`a_version_is_numbered_and_read_from_its_own_key`].
///
/// Far more than any one key's own rows: a hundred thousand index entries are
/// several hundred pages, where one key's are one leaf.
const CROWD: i64 = 100_000;

/// Numbering a version, and reading one back, touches that key's rows alone.
///
/// Every version table is keyed `(project_id, <owner>, version)` since V3, and
/// PostgreSQL 17 cannot seek a b-tree on a later column alone: a statement
/// that names the owner and leaves the project out walks the whole index, or
/// the whole table, which is every project's rows. `MAX(version)` on the write
/// path did exactly that -- once for the evidence, once for the state and once
/// per appended edge -- so a write cost more the more anybody had ever written.
///
/// Counted rather than timed. The server's own statistics say how many pages
/// of each table and its indexes a call touched, and another project holds
/// [`CROWD`] versions of one topic, one source and one edge, so a call that
/// leaves the project out walks all of theirs. Pages rather than rows, because
/// a b-tree tests a condition on a later column inside the scan and returns
/// only what passes: the rows-read counter says zero for a walk of the whole
/// index. The calls run on a pool of exactly one connection because
/// statistics are flushed per backend, and asking that backend to flush is
/// what makes the count exact rather than a second late.
async fn a_version_is_numbered_and_read_from_its_own_key(
    database: &Database,
    workspace: &Workspace,
) {
    let crowd = repository::ensure_project(database.pool(), "crowd")
        .await
        .expect("ensure the crowded project");
    let crowded =
        committed!(database, repository::ensure_topic, crowd.id, "crowded").expect("ensure topic");
    let other =
        committed!(database, repository::ensure_topic, crowd.id, "other").expect("ensure topic");
    let first = write_state(database, crowd.id, crowded.id, "crowd-source", "v1").await;
    write_state(database, crowd.id, other.id, "crowd-other", "v1").await;
    graph::assert_edge(
        database.pool(),
        crowd.id,
        crowded.id,
        other.id,
        &EdgeClaim::explicit(EdgeKind::RelatedTo),
    )
    .await
    .expect("assert the crowded edge");
    let relationship = graph::find_relationship(
        database.pool(),
        crowd.id,
        crowded.id,
        other.id,
        EdgeKind::RelatedTo,
    )
    .await
    .expect("find relationship")
    .expect("the edge exists");

    // Setup, so in bulk: later versions of the one source, the one topic and
    // the one edge, each numbered on from the first.
    for statement in [
        "INSERT INTO source_versions (id, project_id, source_id, version, content,
             content_hash, filter_decision, filter_reason, recorded_at)
         SELECT gen_random_uuid(), $1, sv.source_id, g, 'crowd', 'crowd', 'promoted',
                'crowd', now()
           FROM source_versions sv
           JOIN source_spans sp ON sp.source_version_id = sv.id
          CROSS JOIN generate_series(2, $4 + 1) AS g
          WHERE sp.id = $2",
        "INSERT INTO topic_states (id, project_id, topic_id, version, source_span_id,
             observed_at, recorded_at)
         SELECT gen_random_uuid(), $1, ts.topic_id, g, ts.source_span_id, now(), now()
           FROM topic_states ts
          CROSS JOIN generate_series(2, $4 + 1) AS g
          WHERE ts.source_span_id = $2",
        "INSERT INTO relationship_versions (id, project_id, relationship_id, version,
             created_at, invalidated_at, tombstone_reason, confidence, derivation,
             from_topic, to_topic, kind)
         SELECT gen_random_uuid(), $1, $3, g, now(), now(), 'closed', 1, 'explicit',
                r.from_topic, r.to_topic, r.kind
           FROM generate_series(2, $4 + 1) AS g
           JOIN relationships r ON r.id = $3",
    ] {
        sqlx::query(statement)
            .bind(crowd.id.0)
            .bind(first.source_span_id.0)
            .bind(relationship.id.0)
            .bind(CROWD)
            .execute(database.pool())
            .await
            .expect("crowd the version tables");
    }
    sqlx::query("ANALYZE source_versions, topic_states, relationship_versions")
        .execute(database.pool())
        .await
        .expect("analyze");

    let server = workspace
        .read_server()
        .expect("read server record")
        .expect("workspace has a server");
    let probe = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&server.url())
        .await
        .expect("a one-connection pool");

    // At most this many pages of a table and its indexes for any one call
    // below. Each touches a leaf or two of its own key, the pages it writes,
    // and whatever its foreign keys check; leaving the project out walks
    // hundreds.
    const OWN: i64 = 100;

    // The premise, asserted: a statement that does leave the project out walks
    // the crowd, far enough past the bound that the two cannot be confused.
    let before = pages_touched(&probe, "topic_states").await;
    let _: Option<i32> =
        sqlx::query_scalar("SELECT MAX(version) FROM topic_states WHERE topic_id = $1")
            .bind(uuid::Uuid::now_v7())
            .fetch_one(&probe)
            .await
            .expect("an unscoped maximum");
    let unscoped = pages_touched(&probe, "topic_states").await - before;
    assert!(
        unscoped > 4 * OWN,
        "numbering a topic without its project touched {unscoped} pages; the \
         crowd is not where this test thinks it is"
    );

    let project = repository::ensure_project(database.pool(), "uncrowded")
        .await
        .expect("ensure project")
        .id;
    let mut transaction = probe.begin().await.expect("begin");
    let topic = repository::ensure_topic(&mut transaction, project, "sparse")
        .await
        .expect("ensure topic");
    let target = repository::ensure_topic(&mut transaction, project, "target")
        .await
        .expect("ensure topic");
    let source = repository::ensure_source(&mut transaction, project, SourceKind::Manual, "sparse")
        .await
        .expect("ensure source");
    transaction.commit().await.expect("commit");

    macro_rules! reads_its_own_key {
        ($table:literal, $call:literal, $work:expr) => {{
            let before = pages_touched(&probe, $table).await;
            let outcome = $work;
            let touched = pages_touched(&probe, $table).await - before;
            assert!(
                touched < OWN,
                "{} touched {touched} pages of {} with {CROWD} of another \
                 project's rows there; it is not reading by the project's key",
                $call,
                $table,
            );
            outcome
        }};
    }

    for round in 0..2u32 {
        // Both tables the write numbers from, around one write: the version
        // and the state are each numbered by a maximum over their own key.
        let state = reads_its_own_key!(
            "source_versions",
            "append_promoted",
            reads_its_own_key!("topic_states", "append_promoted", {
                let mut transaction = probe.begin().await.expect("begin");
                let source = repository::ensure_source(
                    &mut transaction,
                    project,
                    SourceKind::Manual,
                    "sparse",
                )
                .await
                .expect("ensure source");
                let locked = repository::lock_topic(&mut transaction, project, "sparse")
                    .await
                    .expect("lock topic")
                    .expect("the topic exists");
                let (_, _, state) = repository::append_promoted(
                    &mut transaction,
                    project,
                    source,
                    &repository::Evidence {
                        content: "sparse evidence",
                        content_hash: "hash",
                        decision: FilterDecision::Promoted,
                        reason: "test fixture",
                        language: None,
                        language_confidence: None,
                    },
                    &repository::Promotion {
                        topic: &locked,
                        observed_at: OffsetDateTime::now_utc(),
                        validity: Validity::ALWAYS,
                        owed: &[],
                    },
                )
                .await
                .expect("append promoted");
                transaction.commit().await.expect("commit");
                state
            })
        );
        assert_eq!(state.version, round + 1, "numbered from its own topic");

        // A different claim each round, so each appends a version.
        let mut claim = EdgeClaim::explicit(EdgeKind::RelatedTo);
        claim.confidence = 1.0 - round as f32 / 4.0;
        let asserted = reads_its_own_key!(
            "relationship_versions",
            "assert_edge",
            graph::assert_edge(&probe, project, topic.id, target.id, &claim)
                .await
                .expect("assert edge")
        );
        assert_eq!(
            asserted.version().version,
            round + 1,
            "numbered from its own edge"
        );
    }

    let latest = reads_its_own_key!(
        "source_versions",
        "latest_source_version",
        repository::latest_source_version(&probe, project, source)
            .await
            .expect("latest source version")
            .expect("evidence exists")
    );
    assert_eq!(latest.version, 2);
    let state = reads_its_own_key!(
        "topic_states",
        "topic_state",
        repository::topic_state(&probe, project, topic.id, 1)
            .await
            .expect("topic state")
            .expect("version one exists")
    );
    assert_eq!(state.version, 1);
    let edge = graph::find_relationship(&probe, project, topic.id, target.id, EdgeKind::RelatedTo)
        .await
        .expect("find relationship")
        .expect("the edge exists");
    let history = reads_its_own_key!(
        "relationship_versions",
        "edge_history",
        graph::edge_history(&probe, project, edge.id)
            .await
            .expect("edge history")
    );
    assert_eq!(history.len(), 2);
}

/// Pages of `table` and its indexes the server has touched so far, whether
/// found in its buffers or read in.
///
/// A backend holds its counts until it is idle and a second has passed since
/// it last reported, so they are flushed here first. That flushes the probe's
/// own backend, which is why the calls being counted run on the probe.
async fn pages_touched(probe: &sqlx::PgPool, table: &str) -> i64 {
    sqlx::query("SELECT pg_stat_force_next_flush()")
        .execute(probe)
        .await
        .expect("flush the statistics");
    sqlx::query_scalar(
        "SELECT heap_blks_read + heap_blks_hit
              + COALESCE(idx_blks_read, 0) + COALESCE(idx_blks_hit, 0)
           FROM pg_statio_user_tables WHERE relname = $1",
    )
    .bind(table)
    .fetch_one(probe)
    .await
    .expect("read the table statistics")
}

/// One connection, which `graph::expand` asks every hop on.
async fn connection(database: &Database) -> sqlx::pool::PoolConnection<sqlx::Postgres> {
    database
        .pool()
        .acquire()
        .await
        .expect("acquire a connection")
}
