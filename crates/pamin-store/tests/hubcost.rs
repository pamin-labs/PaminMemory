//! What the graph walk costs when the graph has hubs in it.
//!
//! The expansion's per-hop statement has no `LIMIT`, fetches every edge
//! incident on the frontier in both directions, and puts each of them into a
//! map twice; the frontier cap applies to the *next* hop's positions, not to
//! the rows this one pulled. None of that is visible on a project small enough
//! for an ordinary test, which is why the hub test in `store.rs` says in its own
//! comment that the difference "is not visible from here". This is where it is
//! visible.
//!
//! Topics and edges are inserted in bulk rather than through `assert_edges`,
//! because building the graph is setup and `assert_edges` costs about five
//! round trips an edge. What is measured is `graph::expand` only.

use pamin_core::{ProjectId, TopicId};
use pamin_store::graph::Expansion;
use pamin_store::{Connections, Database, Workspace, graph, repository};
use std::time::Instant;

/// Builds two hubs, each with `spokes` spokes, and an edge between the hubs.
async fn hubs(database: &Database, project: ProjectId, spokes: usize) -> TopicId {
    let left = TopicId(uuid::Uuid::new_v4());
    let right = TopicId(uuid::Uuid::new_v4());

    let mut ids: Vec<uuid::Uuid> = vec![left.0, right.0];
    let mut names: Vec<String> = vec!["hub_left".into(), "hub_right".into()];
    let mut from: Vec<uuid::Uuid> = vec![left.0];
    let mut to: Vec<uuid::Uuid> = vec![right.0];

    for spoke in 0..spokes {
        let on_left = uuid::Uuid::new_v4();
        let on_right = uuid::Uuid::new_v4();
        ids.push(on_left);
        names.push(format!("left_{spoke}"));
        ids.push(on_right);
        names.push(format!("right_{spoke}"));
        from.push(left.0);
        to.push(on_left);
        from.push(right.0);
        to.push(on_right);
    }

    let now = time::OffsetDateTime::now_utc();
    sqlx::query(
        "INSERT INTO topics (id, project_id, name, created_at)
         SELECT id, $2, name, $4 FROM unnest($1::uuid[], $3::text[]) AS t (id, name)",
    )
    .bind(&ids)
    .bind(project.0)
    .bind(&names)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert topics");

    let relationships: Vec<uuid::Uuid> = (0..from.len()).map(|_| uuid::Uuid::new_v4()).collect();
    sqlx::query(
        "INSERT INTO relationships (id, project_id, from_topic, to_topic, kind, created_at)
         SELECT id, $4, f, t, 'related_to', $5
           FROM unnest($1::uuid[], $2::uuid[], $3::uuid[]) AS e (id, f, t)",
    )
    .bind(&relationships)
    .bind(&from)
    .bind(&to)
    .bind(project.0)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert relationships");

    let versions: Vec<uuid::Uuid> = (0..relationships.len())
        .map(|_| uuid::Uuid::new_v4())
        .collect();
    sqlx::query(
        "INSERT INTO relationship_versions
             (id, project_id, relationship_id, version, created_at, confidence, derivation)
         SELECT id, $3, r, 1, $4, 1.0, 'explicit'
           FROM unnest($1::uuid[], $2::uuid[]) AS v (id, r)",
    )
    .bind(&versions)
    .bind(&relationships)
    .bind(project.0)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert relationship versions");

    left
}

/// How many rows the second hop's own statement pulls back.
async fn edges_touching(database: &Database, project: ProjectId, frontier: &[uuid::Uuid]) -> i64 {
    sqlx::query_scalar(
        "SELECT count(*)
           FROM relationships r
           JOIN relationship_versions v ON v.relationship_id = r.id
          WHERE r.project_id = $1
            AND (r.from_topic = ANY($2) OR r.to_topic = ANY($2))
            AND v.invalidated_at IS NULL",
    )
    .bind(project.0)
    .bind(frontier)
    .fetch_one(database.pool())
    .await
    .expect("count the edges a hop touches")
}

/// A walk stops once the caller has enough, and stops at the same answer.
///
/// Two claims, and both need saying because either alone would be misleading.
///
/// The bound is not an approximation. Arrivals are breadth first, the ranking
/// sorts on hops before anything else, and a later hop only improves an entry
/// from the same hop -- so once a hop has produced what the caller keeps, every
/// arrival still to come sorts behind all of them. The head of the list is
/// asserted identical at every size here.
///
/// And it is worth having. Off a twenty-thousand-degree hub the unbounded walk
/// reaches forty thousand topics, sorts all of them, and hands back fifty:
///
/// ```text
///     spokes    edges   unbounded   bounded   reached
///        120      241       7.1       1.8       241 -> 121
///      1,000    2,001      31.6      10.4     2,001 -> 1,001
///      5,000   10,001     125.5      51.0    10,001 -> 5,001
///     20,000   40,001     446.8     201.5    40,001 -> 20,001
/// ```
///
/// What is left after the bound is the seed's own degree, which no bound can
/// avoid: knowing a hub's neighbours means reading its edges. Getting under
/// that needs a `LIMIT` in the statement, and that is not free -- the ranking
/// breaks ties on the neighbour's identifier, which the statement cannot order
/// by, so a limited fetch would return a different fifty rather than the same
/// fifty sooner.
#[tokio::test]
#[ignore = "needs a real postgres cluster, and builds graphs with tens of thousands of edges"]
async fn a_walk_stops_once_the_caller_has_enough() {
    let workspace = Workspace::at("/tmp/pamin-ws");
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    println!(
        "{:>7}  {:>8}  {:>9}  {:>12}  {:>10}  {:>9}  {:>9}  {:>8}",
        "spokes", "topics", "edges", "hop-2 rows", "loose ms", "bounded", "returned", "bounded"
    );

    for spokes in [120usize, 1_000, 5_000, 20_000] {
        let project = repository::ensure_project(database.pool(), &format!("hub-{spokes}"))
            .await
            .expect("ensure project");
        let left = hubs(&database, project.id, spokes).await;

        // The frontier entering hop two is the left hub's neighbours: every
        // left spoke, plus the right hub.
        let frontier: Vec<uuid::Uuid> = sqlx::query_scalar(
            "SELECT to_topic FROM relationships WHERE project_id = $1 AND from_topic = $2",
        )
        .bind(project.id.0)
        .bind(left.0)
        .fetch_all(database.pool())
        .await
        .expect("the first hop's arrivals");

        let rows = edges_touching(&database, project.id, &frontier).await;

        let started = Instant::now();
        let unbounded = graph::expand(
            database.pool(),
            project.id,
            &[left],
            &Expansion::to_depth(2),
        )
        .await
        .expect("expand");
        let loose = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let bounded = graph::expand(
            database.pool(),
            project.id,
            &[left],
            &Expansion::to_depth(2).keeping(50),
        )
        .await
        .expect("expand");
        let tight = started.elapsed().as_secs_f64() * 1000.0;

        // The top fifty must be the same list either way, which is the whole
        // claim the bound rests on.
        let head = |walk: &[pamin_store::graph::Neighbor]| -> Vec<uuid::Uuid> {
            walk.iter().take(50).map(|n| n.topic.0).collect()
        };
        assert_eq!(
            head(&unbounded),
            head(&bounded),
            "stopping early changed the top fifty at {spokes} spokes"
        );

        // And it stopped. Without this the assertion above passes on a walk
        // that never bounded anything, which is the shape the bound is for.
        assert!(
            bounded.len() < unbounded.len(),
            "the bounded walk reached as much as the unbounded one at {spokes} spokes"
        );

        println!(
            "{spokes:>7}  {:>8}  {:>9}  {rows:>12}  {loose:>10.1}  {tight:>9.1}  {:>9}  {:>8}",
            2 + spokes * 2,
            1 + spokes * 2,
            unbounded.len(),
            bounded.len()
        );
    }
}
