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
//! `graph::expand_reading` is the walk that does bound what it pulls, and the
//! engine's; it is measured beside the unbounded one here, and held to it by
//! the second test below.
//!
//! Topics and edges are inserted in bulk rather than through `assert_edges`,
//! because building the graph is setup and `assert_edges` costs about five
//! round trips an edge. What is measured is the walk only.

use pamin_core::{ProjectId, TopicId};
use pamin_store::graph::{Expansion, Neighbor};
use pamin_store::{Connections, Database, Workspace, graph, repository};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Builds two hubs, each with `spokes` spokes, and an edge between the hubs.
async fn hubs(database: &Database, project: ProjectId, spokes: usize) -> TopicId {
    let left = TopicId(uuid::Uuid::now_v7());
    let right = TopicId(uuid::Uuid::now_v7());

    let mut ids: Vec<uuid::Uuid> = vec![left.0, right.0];
    let mut names: Vec<String> = vec!["hub_left".into(), "hub_right".into()];
    let mut from: Vec<uuid::Uuid> = vec![left.0];
    let mut to: Vec<uuid::Uuid> = vec![right.0];

    for spoke in 0..spokes {
        let on_left = uuid::Uuid::now_v7();
        let on_right = uuid::Uuid::now_v7();
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

    let relationships: Vec<uuid::Uuid> = (0..from.len()).map(|_| uuid::Uuid::now_v7()).collect();
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
        .map(|_| uuid::Uuid::now_v7())
        .collect();
    sqlx::query(
        "INSERT INTO relationship_versions
             (id, project_id, relationship_id, version, created_at, confidence, derivation,
              from_topic, to_topic, kind)
         SELECT id, $3, r, 1, $4, 1.0, 'explicit', f, t, 'related_to'
           FROM unnest($1::uuid[], $2::uuid[], $5::uuid[], $6::uuid[]) AS v (id, r, f, t)",
    )
    .bind(&versions)
    .bind(&relationships)
    .bind(project.0)
    .bind(now)
    .bind(&from)
    .bind(&to)
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
/// What is left after the bound is the seed's own degree: knowing all of a
/// hub's neighbours means reading all of its edges. Getting under that needed
/// a `LIMIT` in the statement, and this paragraph used to say why that was not
/// free -- the ranking breaks ties on the neighbour's identifier, which the
/// statement could not order by, so a limited fetch would have returned a
/// different fifty rather than the same fifty sooner. Since V13 a version
/// carries its identity's endpoints and an index holds a topic's live edges in
/// the walk's own order, confidence and then that identifier, so the statement
/// can, and `graph::expand_reading` does. Taking the head of each topic's
/// edges is still not enough on its own -- see
/// `a_bounded_walk_ranks_as_the_whole_walk_does` for why, and for the test
/// that holds it to the old walk -- so the third column below is that read,
/// ranked the way the engine ranks it, and asserted to be the same fifty.
#[tokio::test]
#[ignore = "needs a real postgres cluster, and builds graphs with tens of thousands of edges"]
async fn a_walk_stops_once_the_caller_has_enough() {
    // A path rather than a tempdir, because `initdb` is twenty seconds and
    // this test's subject is milliseconds -- the cluster is worth reusing. The
    // project inside it is not: the names below are fixed, so a second run on
    // a reused project inserted `hub_left` twice and died on the unique
    // constraint. This test could therefore be run once, which is the same
    // defect as a guard that only covers the path it was written on.
    let workspace = Workspace::at(
        std::env::var("PAMIN_EVAL_HOME").unwrap_or_else(|_| "/tmp/pamin-ws".to_string()),
    );
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");
    let run = uuid::Uuid::now_v7();

    println!(
        "{:>7}  {:>8}  {:>9}  {:>12}  {:>10}  {:>9}  {:>9}  {:>9}  {:>8}",
        "spokes",
        "topics",
        "edges",
        "hop-2 rows",
        "loose ms",
        "bounded",
        "read",
        "returned",
        "bounded"
    );

    for spokes in [120usize, 1_000, 5_000, 20_000] {
        let project = repository::ensure_project(database.pool(), &format!("hub-{spokes}-{run}"))
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
            &mut *connection(&database).await,
            project.id,
            &[left],
            &Expansion::to_depth(2),
        )
        .await
        .expect("expand");
        let loose = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let bounded = graph::expand(
            &mut *connection(&database).await,
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

        // Reading fifty of each topic's edges, ranked as the engine ranks: the
        // same fifty, and settled without walking again -- a read that fell
        // back every time would pass the first half and cost more than either.
        let score = |confidence: f32, hops: u8, _| {
            confidence * 0.5f32.powi(i32::from(hops.saturating_sub(1)))
        };
        let started = Instant::now();
        let read = graph::expand_reading(
            &mut *connection(&database).await,
            project.id,
            &[left],
            &Expansion::to_depth(2).keeping(50),
            50,
        )
        .await
        .expect("expand reading");
        let reading = started.elapsed().as_secs_f64() * 1000.0;
        assert!(
            !read.unread.is_empty(),
            "nothing was left unread at {spokes} spokes, so the read cut nothing"
        );
        assert_eq!(
            read.strongest(50, score),
            Some(graph::strongest(unbounded.clone(), 50, score)),
            "reading the strongest edges changed the top fifty at {spokes} spokes"
        );

        println!(
            "{spokes:>7}  {:>8}  {:>9}  {rows:>12}  {loose:>10.1}  {tight:>9.1}  {reading:>9.1}  {:>9}  {:>8}",
            2 + spokes * 2,
            1 + spokes * 2,
            unbounded.len(),
            bounded.len()
        );
    }
}

/// One connection, which `graph::expand` asks every hop on.
async fn connection(database: &Database) -> sqlx::pool::PoolConnection<sqlx::Postgres> {
    database
        .pool()
        .acquire()
        .await
        .expect("acquire a connection")
}

/// The walk as it was before it could be cut, kept here as the oracle.
///
/// The statement and the loop are the ones `graph::expand` ran until the
/// bounded read, copied rather than called, so that a mistake in the new walk
/// cannot also be a mistake in the thing checking it. One thing is added: an
/// `ORDER BY`. The old statement had none, so which of two tied arrivals it
/// kept was whatever order PostgreSQL returned rows in; the order given here
/// is one it could have returned, and the one the new walk takes on purpose.
mod oracle {
    use pamin_core::{Derivation, EdgeKind, ProjectId, TopicId};
    use pamin_store::graph::Neighbor;
    use sqlx::Row;
    use std::collections::{HashMap, HashSet};

    const MAX_FRONTIER: usize = 2_000;

    const OLD: &str = "SELECT r.from_topic, r.to_topic, r.kind, v.confidence, v.derivation
         FROM relationships r
         JOIN relationship_versions v
           ON v.project_id = r.project_id AND v.relationship_id = r.id
         WHERE r.project_id = $1
           AND (r.from_topic = ANY($2) OR r.to_topic = ANY($2))
           AND ($3::TEXT[] IS NULL OR r.kind = ANY ($3))
           AND v.invalidated_at IS NULL
         ORDER BY v.confidence DESC,
                  LEAST(r.from_topic, r.to_topic), GREATEST(r.from_topic, r.to_topic),
                  array_position(ARRAY['mentions', 'supports', 'contradicts', 'supersedes',
                                       'related_to', 'part_of', 'derived_from', 'same_as',
                                       'depends_on'], r.kind),
                  r.from_topic";

    #[derive(Clone, Copy)]
    struct Step {
        topic: TopicId,
        origin: TopicId,
        via: TopicId,
    }

    #[derive(Clone, Copy)]
    struct Crossing {
        kind: EdgeKind,
        derivation: Derivation,
        confidence: f32,
        outbound: bool,
    }

    pub async fn expand(
        connection: &mut sqlx::PgConnection,
        project: ProjectId,
        seeds: &[TopicId],
        depth: u8,
        keep: Option<usize>,
    ) -> Vec<Neighbor> {
        let mut seen: HashSet<(TopicId, TopicId)> = HashSet::new();
        let mut reached: HashMap<TopicId, Neighbor> = HashMap::new();
        let mut frontier: Vec<Step> = seeds
            .iter()
            .map(|seed| Step {
                topic: *seed,
                origin: *seed,
                via: *seed,
            })
            .collect();

        for hop in 1..=depth {
            let positions: Vec<uuid::Uuid> = {
                let mut positions: Vec<uuid::Uuid> =
                    frontier.iter().map(|step| step.topic.0).collect();
                positions.sort_unstable();
                positions.dedup();
                positions
            };
            let edges = sqlx::query(OLD)
                .bind(project.0)
                .bind(&positions)
                .bind(None::<Vec<String>>)
                .fetch_all(&mut *connection)
                .await
                .expect("the old statement");

            let mut neighbours: HashMap<TopicId, Vec<(TopicId, Crossing)>> = HashMap::new();
            for row in &edges {
                let from = TopicId::from(row.get::<uuid::Uuid, _>("from_topic"));
                let to = TopicId::from(row.get::<uuid::Uuid, _>("to_topic"));
                let crossing = Crossing {
                    kind: EdgeKind::parse(row.get("kind")).expect("a known kind"),
                    derivation: match row.get::<&str, _>("derivation") {
                        "explicit" => Derivation::Explicit,
                        "deterministic" => Derivation::Deterministic,
                        "model" => Derivation::Model,
                        _ => Derivation::Imported,
                    },
                    confidence: row.get("confidence"),
                    outbound: true,
                };
                neighbours.entry(from).or_default().push((
                    to,
                    Crossing {
                        outbound: true,
                        ..crossing
                    },
                ));
                neighbours.entry(to).or_default().push((
                    from,
                    Crossing {
                        outbound: false,
                        ..crossing
                    },
                ));
            }

            let mut next: Vec<(Step, f32)> = Vec::new();
            for step in &frontier {
                let Some(crossings) = neighbours.get(&step.topic) else {
                    continue;
                };
                for (neighbour, crossing) in crossings {
                    if hop > 1 && (*neighbour == step.origin || *neighbour == step.via) {
                        continue;
                    }
                    if !seen.insert((*neighbour, step.origin)) {
                        continue;
                    }
                    let arrival = Neighbor {
                        topic: *neighbour,
                        origin: step.origin,
                        hops: hop,
                        via: step.topic,
                        kind: crossing.kind,
                        derivation: crossing.derivation,
                        confidence: crossing.confidence,
                        outbound: crossing.outbound,
                    };
                    reached
                        .entry(*neighbour)
                        .and_modify(|held| {
                            if held.hops == hop && crossing.confidence > held.confidence {
                                *held = arrival.clone();
                            }
                        })
                        .or_insert_with(|| arrival.clone());
                    next.push((
                        Step {
                            topic: *neighbour,
                            origin: step.origin,
                            via: step.topic,
                        },
                        crossing.confidence,
                    ));
                }
            }

            if next.is_empty() {
                break;
            }
            next.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            next.truncate(MAX_FRONTIER);
            frontier = next.into_iter().map(|(step, _)| step).collect();
            if keep.is_some_and(|keep| reached.len() >= keep) {
                break;
            }
        }

        let mut neighbors: Vec<Neighbor> = reached.into_values().collect();
        neighbors.sort_by(|left, right| {
            left.hops
                .cmp(&right.hops)
                .then_with(|| {
                    right
                        .confidence
                        .partial_cmp(&left.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.topic.0.cmp(&right.topic.0))
        });
        neighbors
    }
}

/// A small deterministic generator, so a failing graph can be built again
/// from its number.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    fn id(&mut self, run: uuid::Uuid) -> uuid::Uuid {
        let drawn = u128::from(self.next()) << 64 | u128::from(self.next());
        uuid::Uuid::from_u128(drawn ^ run.as_u128())
    }
}

/// Confidences that tie a lot, because ties are where a cut goes wrong: the
/// two the product writes, and a few more.
const CONFIDENCES: &[f32] = &[1.0, 0.5, 0.5, 0.5, 0.25, 0.75];
const KINDS: &[&str] = &["mentions", "related_to", "depends_on"];
const DERIVATIONS: &[&str] = &["explicit", "deterministic", "model"];

/// A graph with hubs in it, parallel edges of several kinds both ways round,
/// and history: edges closed, superseded by a live version at another
/// confidence, or deleted.
///
/// Identifiers are drawn and then mixed with `run`, so a cluster kept between
/// runs does not see the same key twice; a graph is rebuilt from its number
/// and the run's identifier, which a failure prints.
async fn random_graph(
    database: &Database,
    project: ProjectId,
    run: uuid::Uuid,
    draw: &mut Draw,
) -> Vec<TopicId> {
    let topics: Vec<uuid::Uuid> = (0..20 + draw.below(180)).map(|_| draw.id(run)).collect();
    let names: Vec<String> = (0..topics.len()).map(|i| format!("t{i}")).collect();
    let now = time::OffsetDateTime::now_utc();
    sqlx::query(
        "INSERT INTO topics (id, project_id, name, created_at)
         SELECT id, $2, name, $4 FROM unnest($1::uuid[], $3::text[]) AS t (id, name)",
    )
    .bind(&topics)
    .bind(project.0)
    .bind(&names)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert topics");

    let mut pairs: HashSet<(uuid::Uuid, uuid::Uuid, &str)> = HashSet::new();
    let hubs = draw.below(4);
    for hub in 0..hubs {
        let degree = 30 + draw.below(170);
        for _ in 0..degree {
            let other = topics[draw.below(topics.len())];
            if other == topics[hub] {
                continue;
            }
            let kind = KINDS[draw.below(KINDS.len())];
            if draw.chance(50) {
                pairs.insert((topics[hub], other, kind));
            } else {
                pairs.insert((other, topics[hub], kind));
            }
        }
    }
    for _ in 0..topics.len() * 2 {
        let from = topics[draw.below(topics.len())];
        let to = topics[draw.below(topics.len())];
        if from != to {
            pairs.insert((from, to, KINDS[draw.below(KINDS.len())]));
        }
    }

    let (mut rel_id, mut rel_from, mut rel_to, mut rel_kind) = (vec![], vec![], vec![], vec![]);
    let (mut ver_id, mut ver_rel, mut ver_n, mut ver_conf, mut ver_der, mut ver_closed) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut ver_from, mut ver_to, mut ver_kind) = (vec![], vec![], vec![]);
    let mut pairs: Vec<_> = pairs.into_iter().collect();
    pairs.sort_unstable();
    for (from, to, kind) in pairs {
        let id = draw.id(run);
        rel_id.push(id);
        rel_from.push(from);
        rel_to.push(to);
        rel_kind.push(kind.to_string());
        // Most edges are one live version. Some have a closed one before it,
        // some were closed and nothing replaced them.
        let history = match draw.below(10) {
            0 => vec![Some("superseded"), None],
            1 => vec![Some("closed")],
            2 => vec![Some("deleted")],
            _ => vec![None],
        };
        for (version, closed) in history.into_iter().enumerate() {
            ver_id.push(draw.id(run));
            ver_rel.push(id);
            ver_n.push(version as i32 + 1);
            ver_conf.push(CONFIDENCES[draw.below(CONFIDENCES.len())]);
            ver_der.push(DERIVATIONS[draw.below(DERIVATIONS.len())].to_string());
            ver_closed.push(closed.map(str::to_string));
            ver_from.push(from);
            ver_to.push(to);
            ver_kind.push(kind.to_string());
        }
    }

    sqlx::query(
        "INSERT INTO relationships (id, project_id, from_topic, to_topic, kind, created_at)
         SELECT id, $5, f, t, k, $6
           FROM unnest($1::uuid[], $2::uuid[], $3::uuid[], $4::text[]) AS e (id, f, t, k)",
    )
    .bind(&rel_id)
    .bind(&rel_from)
    .bind(&rel_to)
    .bind(&rel_kind)
    .bind(project.0)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert relationships");

    sqlx::query(
        "INSERT INTO relationship_versions
             (id, project_id, relationship_id, version, created_at, invalidated_at,
              tombstone_reason, confidence, derivation, from_topic, to_topic, kind)
         SELECT id, $10, r, n, $11, CASE WHEN c IS NULL THEN NULL ELSE $11 END,
                c, conf, d, f, t, k
           FROM unnest($1::uuid[], $2::uuid[], $3::int[], $4::real[], $5::text[],
                       $6::text[], $7::uuid[], $8::uuid[], $9::text[])
                AS v (id, r, n, conf, d, c, f, t, k)",
    )
    .bind(&ver_id)
    .bind(&ver_rel)
    .bind(&ver_n)
    .bind(&ver_conf)
    .bind(&ver_der)
    .bind(&ver_closed)
    .bind(&ver_from)
    .bind(&ver_to)
    .bind(&ver_kind)
    .bind(project.0)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert relationship versions");

    sqlx::query("ANALYZE relationships, relationship_versions")
        .execute(database.pool())
        .await
        .expect("analyze");

    // Hubs first among the topics, so seeds drawn from the front include them.
    topics.into_iter().map(TopicId).collect()
}

/// A bounded walk, and the ranking a caller makes of it, agree with the walk
/// that reads everything.
///
/// Three claims, each checked against the old statement and the old loop:
///
/// * The unbounded walk is the old walk. Only the order ties are settled in
///   changed, and it is one the old statement could have returned.
/// * Every topic a bounded walk holds, it holds exactly as the old walk does,
///   and every topic it misses is one its `unread` bounds.
/// * Whenever `strongest` accepts a bounded walk, its head is the old walk's
///   head under the same ranking.
///
/// And the control: ranking what a bounded walk holds *without* asking
/// `strongest` -- a plain per-topic `LIMIT` -- gets heads wrong on these same
/// graphs. Without that, this test would pass on a cut that never cut
/// anything that mattered.
#[tokio::test]
#[ignore = "needs a real postgres cluster"]
async fn a_bounded_walk_ranks_as_the_whole_walk_does() {
    let workspace = Workspace::at(
        std::env::var("PAMIN_EVAL_HOME").unwrap_or_else(|_| "/tmp/pamin-ws".to_string()),
    );
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");
    let run = uuid::Uuid::now_v7();
    let graphs: u64 = std::env::var("GRAPHS")
        .ok()
        .and_then(|graphs| graphs.parse().ok())
        .unwrap_or(120);

    let (mut walks, mut cut, mut settled, mut refused) = (0, 0, 0, 0);
    // The engine's setting, a cut as deep as the head: how often it is
    // settled, and how often it has to be walked again.
    let (mut settled_at_keep, mut refused_at_keep) = (0, 0);
    // How often a plain LIMIT gets the head wrong, at any cut and at the
    // engine's own, where the cut is as deep as the head.
    let (mut naive_wrong, mut naive_wrong_at_keep) = (0, 0);
    for graph_number in 0..graphs {
        let mut draw = Draw(0x9e37_79b9_7f4a_7c15 ^ (graph_number + 1));
        let project =
            repository::ensure_project(database.pool(), &format!("strongest-{graph_number}-{run}"))
                .await
                .expect("ensure project");
        let topics = random_graph(&database, project.id, run, &mut draw).await;

        for _ in 0..6 {
            let seeds: Vec<TopicId> = {
                let mut seeds = Vec::new();
                for _ in 0..1 + draw.below(8) {
                    // Half the time from the front, where the hubs are.
                    let seed = if draw.chance(50) {
                        topics[draw.below(4.min(topics.len()))]
                    } else {
                        topics[draw.below(topics.len())]
                    };
                    if !seeds.contains(&seed) {
                        seeds.push(seed);
                    }
                }
                seeds
            };
            let depth = 1 + draw.below(3) as u8;
            let keep = [3, 10, 50][draw.below(3)];
            let per_topic = [1, 3, keep][draw.below(3)];
            let expansion = Expansion::to_depth(depth).keeping(keep);
            let mut connection = connection(&database).await;

            let whole =
                oracle::expand(&mut connection, project.id, &seeds, depth, Some(keep)).await;
            let now = graph::expand(&mut connection, project.id, &seeds, &expansion)
                .await
                .expect("expand");
            assert_eq!(
                now, whole,
                "graph {graph_number} of run {run}: the unbounded walk is not the old walk"
            );

            let walk =
                graph::expand_reading(&mut connection, project.id, &seeds, &expansion, per_topic)
                    .await
                    .expect("expand reading");
            walks += 1;
            let by_topic: HashMap<TopicId, &Neighbor> = whole
                .iter()
                .map(|neighbor| (neighbor.topic, neighbor))
                .collect();
            let held: HashSet<TopicId> = walk.neighbors.iter().map(|n| n.topic).collect();
            for neighbor in &walk.neighbors {
                assert_eq!(
                    Some(&neighbor),
                    by_topic.get(&neighbor.topic),
                    "graph {graph_number} of run {run}: a bounded walk holds a topic differently"
                );
            }
            for missing in whole.iter().filter(|n| !held.contains(&n.topic)) {
                assert!(
                    walk.unread.iter().any(|unread| unread.via == missing.via
                        && unread.hops == missing.hops
                        && unread.origins.contains(&missing.origin)
                        && (missing.confidence < unread.confidence
                            || (missing.confidence == unread.confidence
                                && missing.topic.0 > unread.after.0))),
                    "graph {graph_number} of run {run}: {missing:?} is missing and nothing unread bounds it"
                );
            }
            if walk.unread.is_empty() {
                assert_eq!(
                    walk.neighbors, whole,
                    "graph {graph_number} of run {run}: nothing was cut"
                );
                continue;
            }
            cut += 1;

            // Rankings the way the engine makes them: confidence, halved per
            // hop, times the seed's relevance -- here drawn from few values,
            // so that ties across seeds are common.
            for _ in 0..4 {
                let relevance: HashMap<TopicId, f32> = seeds
                    .iter()
                    .map(|seed| (*seed, [1.0, 0.8, 0.5, 0.49][draw.below(4)]))
                    .collect();
                let score = |confidence: f32, hops: u8, origin: TopicId| {
                    confidence
                        * 0.5f32.powi(i32::from(hops.saturating_sub(1)))
                        * relevance.get(&origin).copied().unwrap_or(0.0)
                };
                let truth = graph::strongest(whole.clone(), keep, score);
                if graph::strongest(walk.neighbors.clone(), keep, score) != truth {
                    naive_wrong += 1;
                    naive_wrong_at_keep += usize::from(per_topic == keep);
                }
                match walk.clone().strongest(keep, score) {
                    Some(head) => {
                        settled += 1;
                        settled_at_keep += usize::from(per_topic == keep);
                        assert_eq!(
                            head, truth,
                            "graph {graph_number} of run {run}: a settled head is not the whole walk's"
                        );
                    }
                    None => {
                        refused += 1;
                        refused_at_keep += usize::from(per_topic == keep);
                    }
                }
            }
        }
    }

    println!(
        "{walks} walks, {cut} cut; rankings of cut walks: {settled} settled, {refused} refused, \
         {naive_wrong} a plain LIMIT gets wrong\n\
         cut at the head's depth: {settled_at_keep} settled, {refused_at_keep} refused, \
         {naive_wrong_at_keep} a plain LIMIT gets wrong"
    );
    assert!(cut > 0, "no walk was cut, so nothing here tested the cut");
    assert!(
        settled > 0,
        "no cut walk was settled, so the fast path never ran"
    );
    assert!(
        naive_wrong_at_keep > 0,
        "a plain LIMIT as deep as the head never got it wrong, so these graphs cannot tell"
    );
}
