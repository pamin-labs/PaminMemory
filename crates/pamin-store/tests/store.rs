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
    edges_are_versioned_rather_than_overwritten(&mut database).await;
    expansion_is_bounded_undirected_and_time_filtered(&mut database).await;
    grep_reaches_evidence_the_index_never_saw(&mut database).await;
    a_retraction_reason_decides_what_history_keeps(&mut database).await;
    a_seed_never_reaches_itself_however_deep_the_walk(&mut database).await;

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
        "relationships",
        "relationship_versions",
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
        Validity::ALWAYS,
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

/// A project with three topics wired into a chain, for the graph checks.
async fn graph_fixture(
    database: &mut Database,
) -> (
    pamin_core::ProjectId,
    pamin_core::TopicId,
    pamin_core::TopicId,
    pamin_core::TopicId,
) {
    let project = repository::ensure_project(database.client(), "graph")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["service", "database", "backup_job"] {
        let topic = repository::ensure_topic(database.client(), project.id, name)
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

async fn edges_are_versioned_rather_than_overwritten(database: &mut Database) {
    let (project, service, db, _) = graph_fixture(database).await;

    let first = graph::assert_edge(
        database.client_mut(),
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
        database.client_mut(),
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
    let second = graph::assert_edge(database.client_mut(), project, service, db, &narrowed)
        .await
        .expect("assert changed edge");
    assert!(second.is_new());
    assert_eq!(second.version().version, 2);
    assert_eq!(second.version().supersedes, Some(first.version().id));

    let relationship =
        graph::find_relationship(database.client(), project, service, db, EdgeKind::DependsOn)
            .await
            .expect("find relationship")
            .expect("relationship exists");

    let history = graph::edge_history(database.client(), relationship.id)
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
        database.client(),
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
        graph::live_version(database.client(), relationship.id)
            .await
            .expect("live version")
            .is_none(),
        "nothing is believed after a retraction"
    );
    assert_eq!(
        graph::edge_history(database.client(), relationship.id)
            .await
            .expect("history")
            .len(),
        2,
        "retraction removes no rows"
    );

    assert!(
        !graph::close_edge(
            database.client(),
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

async fn expansion_is_bounded_undirected_and_time_filtered(database: &mut Database) {
    let (project, service, db, backup) = graph_fixture(database).await;

    // service -> database -> backup_job, so backup_job is two hops from
    // service and is only reachable by following the second edge backwards.
    let service_state = current_state(database, service).await;
    graph::assert_edge(
        database.client_mut(),
        project,
        service,
        db,
        &EdgeClaim::derived(EdgeKind::Mentions, service_state, 0.5),
    )
    .await
    .expect("service -> database");
    graph::assert_edge(
        database.client_mut(),
        project,
        backup,
        db,
        &EdgeClaim::explicit(EdgeKind::DependsOn),
    )
    .await
    .expect("backup_job -> database");

    let one_hop = graph::expand(
        database.client(),
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
        database.client(),
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
        database.client(),
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
    graph::assert_edge(database.client_mut(), project, backup, db, &expired)
        .await
        .expect("bound the edge to the past");

    let now = graph::expand(
        database.client(),
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
        database.client(),
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
    let versions = repository::topic_versions(database.client(), topic)
        .await
        .expect("versions");
    let latest = resolve(&versions, VersionOffset::LATEST).expect("latest");
    repository::topic_state(database.client(), topic, latest.version)
        .await
        .expect("load state")
        .expect("state exists")
        .id
}

async fn grep_reaches_evidence_the_index_never_saw(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "ledger")
        .await
        .expect("ensure project");
    let source = repository::ensure_source(
        database.client(),
        project.id,
        SourceKind::Manual,
        "grep-source",
    )
    .await
    .expect("ensure source");

    // Held by the filter, so it never became a topic state and never entered
    // the projection index. Reaching it is the entire reason this exists.
    repository::append_source_version(
        database.client(),
        project.id,
        source,
        "the KILN reaches cone ten",
        "hash",
        FilterDecision::Filtered,
        "no durable claim",
    )
    .await
    .expect("append filtered evidence");

    let hits = repository::grep_evidence(database.client(), project.id, "cone ten", true, 10)
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
        repository::grep_evidence(database.client(), project.id, "kiln", true, 10)
            .await
            .expect("grep")
            .is_empty(),
        "a case-sensitive search does not fold case"
    );
    assert!(
        !repository::grep_evidence(database.client(), project.id, "kiln", false, 10)
            .await
            .expect("grep")
            .is_empty(),
        "a case-insensitive search does"
    );

    // Superseded versions stay reachable, which is what makes this an audit
    // route rather than a second view of current state.
    let topic = repository::find_topic(database.client(), project.id, "deployment_pipeline")
        .await
        .expect("find topic")
        .expect("topic exists");
    assert!(topic.name == "deployment_pipeline");
    assert!(
        !repository::grep_evidence(database.client(), project.id, "deploys via make", true, 10)
            .await
            .expect("grep")
            .is_empty(),
        "the first version of a rewritten memory is still in evidence"
    );
}

async fn a_retraction_reason_decides_what_history_keeps(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "history")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["tenant_a", "tenant_b", "tenant_c"] {
        let topic = repository::ensure_topic(database.client(), project.id, name)
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
            database.client_mut(),
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
        database.client(),
        project.id,
        root,
        ended,
        EdgeKind::DependsOn,
        TombstoneReason::Closed,
    )
    .await
    .expect("close ended");
    graph::close_edge(
        database.client(),
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
        database.client(),
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
        database.client(),
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

async fn a_seed_never_reaches_itself_however_deep_the_walk(database: &mut Database) {
    let project = repository::ensure_project(database.client(), "cycles")
        .await
        .expect("ensure project");

    let mut topics = Vec::new();
    for name in ["ring_a", "ring_b", "ring_c"] {
        let topic = repository::ensure_topic(database.client(), project.id, name)
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
            database.client_mut(),
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
            database.client(),
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
