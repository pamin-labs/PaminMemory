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
    Derivation, EdgeKind, FilterDecision, SourceKind, TombstoneReason, Validity, VersionOffset,
    resolve,
};
use pamin_store::graph::{EdgeClaim, Expansion};
use pamin_store::{Database, Workspace, graph, repository};
// The table name is a literal from the list above, not caller input; the
// assertion is what lets it be interpolated at all.
use sqlx::AssertSqlSafe;
use time::OffsetDateTime;

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
    let source = repository::ensure_source(database.pool(), project, SourceKind::Manual, locator)
        .await
        .expect("ensure source");
    let version = repository::append_source_version(
        database.pool(),
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

    repository::append_topic_state(
        database.pool(),
        project,
        topic,
        content,
        span.id,
        OffsetDateTime::now_utc(),
        Validity::ALWAYS,
    )
    .await
    .expect("append topic state")
}

async fn appending_versions_builds_a_supersession_chain(database: &Database) {
    let project = repository::ensure_project(database.pool(), "ledger")
        .await
        .expect("ensure project");
    let topic = repository::ensure_topic(database.pool(), project.id, "deployment_pipeline")
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

    let deleted = repository::soft_delete_topic_state(database.pool(), topic.id, 2)
        .await
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
    let source = repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "noise-source",
    )
    .await
    .expect("ensure source");

    repository::append_source_version(
        database.pool(),
        project.id,
        source,
        "ok",
        "hash",
        FilterDecision::Filtered,
        "no durable claim",
    )
    .await
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
        let topic = repository::ensure_topic(database.pool(), project.id, name)
            .await
            .expect("ensure topic");
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
    let source = repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "grep-source",
    )
    .await
    .expect("ensure source");

    // Held by the filter, so it never became a topic state and never entered
    // the projection index. Reaching it is the entire reason this exists.
    repository::append_source_version(
        database.pool(),
        project.id,
        source,
        "the KILN reaches cone ten",
        "hash",
        FilterDecision::Filtered,
        "no durable claim",
    )
    .await
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
        let topic = repository::ensure_topic(database.pool(), project.id, name)
            .await
            .expect("ensure topic");
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
        let topic = repository::ensure_topic(database.pool(), project.id, name)
            .await
            .expect("ensure topic");
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
    let source = repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "contended-source",
    )
    .await
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
                repository::append_source_version(
                    database.pool(),
                    project.id,
                    source,
                    &format!("evidence from writer {writer}"),
                    "hash",
                    FilterDecision::Promoted,
                    "test fixture",
                )
                .await
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
    let source = repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "idempotent-source",
    )
    .await
    .expect("ensure source");
    let from = repository::ensure_topic(database.pool(), project.id, "from")
        .await
        .expect("ensure topic");
    let to = repository::ensure_topic(database.pool(), project.id, "to")
        .await
        .expect("ensure topic");
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
    repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "idempotent-source",
    )
    .await
    .expect("re-ensure source");
    repository::ensure_topic(database.pool(), project.id, "from")
        .await
        .expect("re-ensure topic");
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
            repository::ensure_topic(database.pool(), project.id, name)
                .await
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

    pamin_store::migrate::run(&scratch)
        .await
        .expect("migrate the scratch database");

    // Rewrite the books the way the previous runner kept them, leaving the
    // schema it produced in place.
    sqlx::query("DROP TABLE _sqlx_migrations")
        .execute(&scratch)
        .await
        .expect("drop the new bookkeeping");
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

    let (adopted,): (i64,) = sqlx::query_as("SELECT count(*) FROM _sqlx_migrations")
        .fetch_one(&scratch)
        .await
        .expect("count adopted migrations");
    assert_eq!(
        adopted, 3,
        "every applied migration should have been adopted"
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
    let source = repository::ensure_source(
        database.pool(),
        project.id,
        SourceKind::Manual,
        "columns-locator",
    )
    .await
    .expect("ensure source");

    let evidence = repository::append_source_version(
        database.pool(),
        project.id,
        source,
        "the content",
        "the-hash",
        FilterDecision::Filtered,
        "the reason",
    )
    .await
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

    let topic = repository::ensure_topic(database.pool(), project.id, "columns_topic")
        .await
        .expect("ensure topic");
    let state = repository::append_topic_state(
        database.pool(),
        project.id,
        topic.id,
        "the state content",
        span.id,
        observed,
        Validity {
            from: Some(valid_from),
            to: Some(valid_to),
        },
    )
    .await
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
    let other = repository::ensure_topic(database.pool(), project.id, "columns_other")
        .await
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
