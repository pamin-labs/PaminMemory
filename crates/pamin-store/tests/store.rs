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
use pamin_store::{Database, Workspace, graph, jobs, repository};
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
    let workspace = Workspace::at("/tmp/pamin-ws");

    let database = Database::open(&workspace).await.expect("open workspace");

    migrations_create_every_table(&database).await;
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
    ensuring_a_row_that_exists_does_not_rewrite_it(&database).await;
    derived_edges_are_asserted_together_or_not_at_all(&database).await;
    a_workspace_the_previous_runner_migrated_is_adopted(&database, &workspace).await;
    the_current_state_pointer_follows_every_write(&database).await;
    two_adjacent_hubs_do_not_multiply(&database).await;
    the_outbox_coalesces_claims_and_survives_a_lost_worker(&database).await;
    a_derived_edge_the_content_stopped_making_is_closed(&database).await;
    every_column_holds_what_was_written_to_it(&database).await;

    drop(database);
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

async fn reopening_reuses_the_running_server(workspace: &Workspace) {
    // Must not start a second cluster, and must not fail re-applying migrations.
    let reopened = Database::open(workspace).await.expect("reopen workspace");
    drop(reopened);
}

/// Writes evidence, a span over it, and a topic state derived from that span.
async fn write_state(
    database: &Database,
    project: pamin_core::ProjectId,
    topic: pamin_core::TopicId,
    locator: &str,
    content: &str,
) -> pamin_core::TopicState {
    let source = committed!(
        database,
        repository::ensure_source,
        project,
        SourceKind::Manual,
        locator
    )
    .expect("ensure source");
    let version = committed!(
        database,
        repository::append_source_version,
        project,
        source,
        content,
        "hash",
        FilterDecision::Promoted,
        "test fixture"
    )
    .expect("append source version");
    let span = repository::append_source_span(
        database.pool(),
        project,
        version.id,
        0,
        content.len() as u32,
        None,
        None,
    )
    .await
    .expect("append span");

    committed!(
        database,
        repository::append_topic_state,
        project,
        topic,
        content,
        span.id,
        OffsetDateTime::now_utc(),
        Validity::ALWAYS
    )
    .expect("append topic state")
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

    let loaded = repository::topic_state(database.pool(), topic.id, 1)
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
    let still_there = repository::topic_state(database.pool(), topic.id, 2)
        .await
        .expect("load deleted state")
        .expect("deleted state is still stored");
    assert!(still_there.deleted_at.is_some());
    assert_eq!(still_there.content, "deploys via ci");

    // A new append continues the numbering rather than reusing the freed one.
    let next = write_state(database, project.id, topic.id, "note-3", "deploys via cd").await;
    assert_eq!(next.version, 3, "version numbers are never reused");
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

    let stored = repository::latest_source_version(database.pool(), source)
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

    let history = graph::edge_history(database.pool(), relationship.id)
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
        graph::edge_history(database.pool(), relationship.id)
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
    let service_state = current_state(database, service).await;
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
        database.pool(),
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
        database.pool(),
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
        database.pool(),
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: Some(&[EdgeKind::Mentions]),
            at: None,
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
        database.pool(),
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: None,
            at: Some(OffsetDateTime::now_utc()),
        },
    )
    .await
    .expect("expand at now");
    assert!(
        now.iter().all(|n| n.topic != backup),
        "an edge asserted only for a past interval does not hold now"
    );

    let back_then = graph::expand(
        database.pool(),
        project,
        &[service],
        &Expansion {
            depth: 2,
            kinds: None,
            at: Some(OffsetDateTime::UNIX_EPOCH + time::Duration::hours(1)),
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
    topic: pamin_core::TopicId,
) -> pamin_core::TopicStateId {
    let versions = repository::topic_versions(database.pool(), topic)
        .await
        .expect("versions");
    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    repository::topic_state(database.pool(), topic, latest.version)
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
        database.pool(),
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
        database.pool(),
        project.id,
        &[root],
        &Expansion {
            depth: 1,
            kinds: None,
            at: Some(before_retraction),
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
            database.pool(),
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
                let database = Database::connect(&server).await.expect("connect");
                committed!(
                    &database,
                    repository::append_source_version,
                    project.id,
                    source,
                    &format!("evidence from writer {writer}"),
                    "hash",
                    FilterDecision::Promoted,
                    "test fixture"
                )
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
    sqlx::query("DROP DATABASE IF EXISTS pamin_adoption_check")
        .execute(database.pool())
        .await
        .expect("drop scratch database");
    sqlx::query("CREATE DATABASE pamin_adoption_check")
        .execute(database.pool())
        .await
        .expect("create scratch database");

    let mut server = workspace
        .read_server()
        .expect("read server record")
        .expect("workspace has a server");
    server.database = "pamin_adoption_check".to_string();

    let scratch = sqlx::PgPool::connect(&server.url())
        .await
        .expect("connect to scratch database");

    // Built to the schema the old runner left, which is the three migrations
    // that existed while it was in use -- not to today's schema. Applying the
    // files directly is what makes this a database `refinery` could have
    // produced, rather than one this runner produced and then relabelled.
    for file in [
        "V1__initial.sql",
        "V2__relationships.sql",
        "V3__shard_key_and_indexes.sql",
    ] {
        let sql = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("migrations")
                .join(file),
        )
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

    for (version, name) in [
        (1, "initial"),
        (2, "relationships"),
        (3, "shard_key_and_indexes"),
    ] {
        sqlx::query(
            "INSERT INTO refinery_schema_history (version, name, applied_on, checksum)
             VALUES ($1, $2, '2026-01-01T00:00:00Z', '1234567890')",
        )
        .bind(version)
        .bind(name)
        .execute(&scratch)
        .await
        .expect("record an applied migration");
    }

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

    let read_back = repository::latest_source_version(database.pool(), source)
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
    let state = committed!(
        database,
        repository::append_topic_state,
        project.id,
        topic.id,
        "the state content",
        span.id,
        observed,
        Validity {
            from: Some(valid_from),
            to: Some(valid_to),
        }
    )
    .expect("append topic state");

    let stored = repository::topic_state(database.pool(), topic.id, state.version)
        .await
        .expect("read topic state")
        .expect("the state was written");
    assert_eq!(stored.content, "the state content");
    assert_eq!(stored.source_span_id, span.id);
    assert_eq!(stored.observed_at, observed);
    assert_eq!(stored.validity.from, Some(valid_from));
    assert_eq!(stored.validity.to, Some(valid_to));
    assert!(
        stored.recorded_at > valid_to,
        "recorded_at should be now, not one of the stated instants"
    );
    assert_eq!(stored.supersedes, None);
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
            let state = repository::topic_state(database.pool(), topic, expected)
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
        database.pool(),
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

    // Priority decides what a worker sees first: syncing the index for a memory
    // just written comes before deriving its edges.
    let claimed = jobs::claim(database.pool(), "worker-a", 1)
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
    let contended = jobs::claim(database.pool(), "worker-b", 10)
        .await
        .expect("claim again");
    assert!(
        contended.iter().all(|job| job.id != claimed[0].id),
        "a claimed job was handed to a second worker"
    );

    // The second worker finishes what it did get, so the rest of this is about
    // one job rather than two.
    for job in &contended {
        assert!(
            jobs::complete(database.pool(), job, "worker-b")
                .await
                .expect("complete")
        );
    }

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
        !jobs::complete(database.pool(), &claimed[0], "worker-a")
            .await
            .expect("complete"),
        "a job requested again mid-flight must not be completed by the attempt \
         that was already running"
    );

    // So it is still there, and claimable.
    let requeued = jobs::claim(database.pool(), "worker-a", 1)
        .await
        .expect("claim the re-requested job");
    assert_eq!(requeued.len(), 1);
    assert_eq!(requeued[0].id, claimed[0].id);
    assert_eq!(
        requeued[0].attempts, 1,
        "reviving a job resets its attempts"
    );

    assert!(
        jobs::complete(database.pool(), &requeued[0], "worker-a")
            .await
            .expect("complete"),
        "a job nobody re-requested completes"
    );

    // Completing does not delete, and a later request revives the same row.
    jobs::enqueue(
        database.pool(),
        project.id,
        JobKind::SyncTopicIndex,
        Some(topic.0),
    )
    .await
    .expect("enqueue after completion");
    let revived = jobs::claim(database.pool(), "worker-a", 1)
        .await
        .expect("claim revived");
    assert_eq!(
        revived[0].id, claimed[0].id,
        "a completed row should be revived rather than left to swallow the request"
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

        failing = jobs::claim(database.pool(), "worker-a", 1)
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
    let outstanding = jobs::claim(database.pool(), "worker-a", 100)
        .await
        .expect("drain");
    for job in &outstanding {
        jobs::complete(database.pool(), job, "worker-a")
            .await
            .expect("complete");
    }
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
    let history = graph::edge_history(database.pool(), relationship.id)
        .await
        .expect("edge history");
    let retracted = history.last().expect("the edge has a version");
    assert_eq!(retracted.tombstone_reason, Some(TombstoneReason::Closed));

    let before = retracted
        .invalidated_at
        .expect("a closed version records when")
        - time::Duration::seconds(1);
    let reached = graph::expand(
        database.pool(),
        project.id,
        &[deploy],
        &Expansion {
            depth: 1,
            at: Some(before),
            kinds: None,
        },
    )
    .await
    .expect("expand at a moment before the retraction");
    assert!(
        reached.iter().any(|neighbor| neighbor.topic == argo),
        "what the memory said before is still true of before: {reached:?}"
    );
}
