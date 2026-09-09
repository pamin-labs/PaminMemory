//! The relationship graph.
//!
//! These rows are the one recall channel the projection index cannot see, which
//! is why fusion has to happen above both of them rather than inside the index.
//!
//! Edges are append-only on the same terms as topic states. Asserting a changed
//! edge closes the live version and appends a new one pointing back at it;
//! nothing is overwritten, so what we believed before a relationship changed
//! stays readable.

use pamin_core::{
    Derivation, EdgeKind, ProjectId, Relationship, RelationshipId, RelationshipVersion,
    RelationshipVersionId, TombstoneReason, TopicId, TopicStateId, Validity,
};
use std::collections::{HashMap, HashSet};

use sqlx::postgres::PgRow;
use sqlx::{PgExecutor, PgPool, Row};
use time::OffsetDateTime;

use crate::error::Result;
use crate::sql::SqlLabel;

/// What is being asserted about an edge.
///
/// Grouped rather than passed positionally because the interesting fields are
/// all optional timestamps, and a caller swapping two of those would compile.
#[derive(Clone, Debug)]
pub struct EdgeClaim {
    pub kind: EdgeKind,
    pub derivation: Derivation,
    /// Orders neighbours at equal graph distance.
    pub confidence: f32,
    /// When the relationship is asserted to hold.
    pub validity: Validity,
    /// The topic state that caused the claim. Absent when a caller asserted it
    /// directly, since no state produced it.
    pub caused_by_topic_state: Option<TopicStateId>,
}

impl EdgeClaim {
    /// An edge a caller asserted directly.
    pub fn explicit(kind: EdgeKind) -> Self {
        Self {
            kind,
            derivation: Derivation::Explicit,
            confidence: 1.0,
            validity: Validity::ALWAYS,
            caused_by_topic_state: None,
        }
    }

    /// An edge a rule derived, with the state that produced it.
    ///
    /// Derived edges are less certain than asserted ones by construction: a
    /// rule matched text, where an explicit edge is somebody saying so. The
    /// weight is provisional and belongs to the evaluation harness.
    pub fn derived(kind: EdgeKind, caused_by: TopicStateId, confidence: f32) -> Self {
        Self {
            kind,
            derivation: Derivation::Deterministic,
            confidence,
            // A rule that matched a name learns that a relationship exists, not
            // when it holds, so a derived edge asserts no interval.
            validity: Validity::ALWAYS,
            caused_by_topic_state: Some(caused_by),
        }
    }

    /// Whether an existing version already says exactly this.
    fn matches(&self, version: &RelationshipVersion) -> bool {
        version.derivation == self.derivation
            && version.confidence == self.confidence
            && version.validity == self.validity
    }
}

/// What asserting an edge did.
///
/// The distinction is the idempotency contract: re-deriving the same edge from
/// the same content must not append a second version, and a caller that cannot
/// tell the two apart cannot check that it did not.
#[derive(Clone, Debug)]
pub enum Assertion {
    /// A live version already said this. Nothing was written.
    Unchanged(RelationshipVersion),
    /// A new version was appended, superseding any live one.
    Appended(RelationshipVersion),
}

impl Assertion {
    pub fn version(&self) -> &RelationshipVersion {
        match self {
            Self::Unchanged(version) | Self::Appended(version) => version,
        }
    }

    pub fn is_new(&self) -> bool {
        matches!(self, Self::Appended(_))
    }
}

/// The columns `row_to_version` reads.
///
/// A macro rather than a constant so the statements below can be assembled with
/// `concat!` and stay `&'static str`, which is what the driver accepts without
/// an explicit assertion that a built string is safe.
macro_rules! version_columns {
    () => {
        "id, relationship_id, version, valid_from, valid_to, created_at, \
         invalidated_at, supersedes, caused_by_topic_state, confidence, derivation, \
         tombstone_reason"
    };
}

fn row_to_version(row: &PgRow) -> RelationshipVersion {
    RelationshipVersion {
        id: row.get::<uuid::Uuid, _>("id").into(),
        relationship_id: row.get::<uuid::Uuid, _>("relationship_id").into(),
        version: row.get::<i32, _>("version") as u32,
        validity: Validity::new(row.get("valid_from"), row.get("valid_to")),
        created_at: row.get("created_at"),
        invalidated_at: row.get("invalidated_at"),
        supersedes: row
            .get::<Option<uuid::Uuid>, _>("supersedes")
            .map(Into::into),
        caused_by_topic_state: row
            .get::<Option<uuid::Uuid>, _>("caused_by_topic_state")
            .map(Into::into),
        confidence: row.get("confidence"),
        derivation: Derivation::from_label(row.get("derivation"))
            // The column's CHECK constraint admits nothing else.
            .unwrap_or(Derivation::Imported),
        tombstone_reason: row
            .get::<Option<String>, _>("tombstone_reason")
            .as_deref()
            .and_then(TombstoneReason::from_label),
    }
}

const FIND_RELATIONSHIP: &str = "SELECT id, created_at FROM relationships
                                 WHERE project_id = $1 AND from_topic = $2
                                   AND to_topic = $3 AND kind = $4";

/// Returns the edge identity for this pair and kind, creating it if absent.
///
/// One identity per (pair, kind): two topics can be related several ways at
/// once, and each way carries its own history.
/// Takes a connection rather than any executor because it uses it three times,
/// and a pooled connection is not something that can be handed out twice.
pub async fn ensure_relationship(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Relationship> {
    // Read first: an edge is created once and re-asserted on every rewrite of
    // the memory that derives it, so the insert is the rare path. See
    // `repository::ensure_project` for why the conflict clause does not update.
    if let Some(relationship) = find_relationship(&mut *connection, project, from, to, kind).await?
    {
        return Ok(relationship);
    }

    let inserted = sqlx::query(
        "INSERT INTO relationships
             (id, project_id, from_topic, to_topic, kind, created_at)
         VALUES ($1, $2, $3, $4, $5, $6)
         ON CONFLICT (project_id, from_topic, to_topic, kind) DO NOTHING
         RETURNING id, created_at",
    )
    .bind(RelationshipId::new().0)
    .bind(project.0)
    .bind(from.0)
    .bind(to.0)
    .bind(kind.label())
    .bind(OffsetDateTime::now_utc())
    .fetch_optional(&mut *connection)
    .await?;

    let row = match inserted {
        Some(row) => row,
        // Another writer created it in between.
        None => {
            sqlx::query(FIND_RELATIONSHIP)
                .bind(project.0)
                .bind(from.0)
                .bind(to.0)
                .bind(kind.label())
                .fetch_one(&mut *connection)
                .await?
        }
    };

    Ok(Relationship {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: project,
        from_topic: from,
        to_topic: to,
        kind,
        created_at: row.get("created_at"),
    })
}

/// Loads the version of an edge currently believed, if any.
pub async fn live_version(
    executor: impl PgExecutor<'_>,
    relationship: RelationshipId,
) -> Result<Option<RelationshipVersion>> {
    let row = sqlx::query(concat!(
        "SELECT ",
        version_columns!(),
        " FROM relationship_versions
          WHERE relationship_id = $1 AND invalidated_at IS NULL
          ORDER BY version DESC LIMIT 1"
    ))
    .bind(relationship.0)
    .fetch_optional(executor)
    .await?;

    Ok(row.as_ref().map(row_to_version))
}

/// Loads every version of an edge, oldest first.
pub async fn edge_history(
    executor: impl PgExecutor<'_>,
    relationship: RelationshipId,
) -> Result<Vec<RelationshipVersion>> {
    let rows = sqlx::query(concat!(
        "SELECT ",
        version_columns!(),
        " FROM relationship_versions
          WHERE relationship_id = $1 ORDER BY version ASC"
    ))
    .bind(relationship.0)
    .fetch_all(executor)
    .await?;

    Ok(rows.iter().map(row_to_version).collect())
}

/// Asserts an edge, appending a version only if the claim is new.
///
/// Runs in a transaction that locks the edge identity first. Without the lock,
/// two writers can read the same maximum version and race to insert it; one
/// loses on the unique constraint and its claim is dropped rather than queued.
/// This is the same protocol topic states use, for the same reason.
pub async fn assert_edge(
    pool: &PgPool,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Assertion> {
    if let Some(unchanged) = already_asserted(pool, project, from, to, claim).await? {
        return Ok(Assertion::Unchanged(unchanged));
    }

    let mut transaction = pool.begin().await?;
    let asserted = assert_within(&mut transaction, project, from, to, claim).await?;
    transaction.commit().await?;
    Ok(asserted)
}

/// Asserts several edges in one transaction.
///
/// Deriving the edges of one memory means asserting every topic it names, and
/// a transaction each meant four to six round trips per edge plus a commit.
/// Here they share one, so the cost of a write grows with the number of names
/// in it rather than with that number times the depth of the protocol.
///
/// Atomic as well as cheaper, which is the more important half: a memory's
/// derived edges are one statement about what it says, and a crash partway
/// through used to leave that statement half told.
pub async fn assert_edges(
    pool: &PgPool,
    project: ProjectId,
    edges: &[(TopicId, TopicId, EdgeClaim)],
) -> Result<Vec<Assertion>> {
    // Answered outside the transaction, and for the common case that is the
    // whole call: rewriting a memory re-derives the edges it already had, and
    // an unchanged claim writes nothing, so no lock is needed to decide it.
    let mut asserted: Vec<Option<Assertion>> = Vec::with_capacity(edges.len());
    let mut pending = Vec::new();
    for (index, (from, to, claim)) in edges.iter().enumerate() {
        let unchanged = already_asserted(pool, project, *from, *to, claim).await?;
        if unchanged.is_none() {
            pending.push(index);
        }
        asserted.push(unchanged.map(Assertion::Unchanged));
    }

    if !pending.is_empty() {
        let mut transaction = pool.begin().await?;
        for index in pending {
            let (from, to, claim) = &edges[index];
            asserted[index] =
                Some(assert_within(&mut transaction, project, *from, *to, claim).await?);
        }
        transaction.commit().await?;
    }

    Ok(asserted
        .into_iter()
        .map(|assertion| assertion.expect("every edge was either read or written"))
        .collect())
}

/// The live version of this edge when the claim would not change it.
///
/// A read, so it can run before any lock is taken. Racing it is harmless: the
/// answer is that nothing needs writing, which is as true a moment later as it
/// is a moment after a commit.
async fn already_asserted(
    pool: &PgPool,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Option<RelationshipVersion>> {
    let Some(relationship) = find_relationship(pool, project, from, to, claim.kind).await? else {
        return Ok(None);
    };

    Ok(live_version(pool, relationship.id)
        .await?
        .filter(|version| claim.matches(version)))
}

async fn assert_within(
    transaction: &mut sqlx::PgTransaction<'_>,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Assertion> {
    let relationship = ensure_relationship(transaction, project, from, to, claim.kind).await?;

    sqlx::query("SELECT id FROM relationships WHERE id = $1 FOR UPDATE")
        .bind(relationship.id.0)
        .execute(&mut **transaction)
        .await?;

    let live = live_version(&mut **transaction, relationship.id).await?;
    if let Some(existing) = live.as_ref().filter(|version| claim.matches(version)) {
        // Re-deriving the same edge from unchanged content must not stack
        // versions, or every rewrite of a memory would grow the ledger.
        return Ok(Assertion::Unchanged(existing.clone()));
    }

    let now = OffsetDateTime::now_utc();
    if let Some(previous) = live.as_ref() {
        sqlx::query(
            "UPDATE relationship_versions
             SET invalidated_at = $2, tombstone_reason = $3
             WHERE id = $1",
        )
        .bind(previous.id.0)
        .bind(now)
        .bind(TombstoneReason::Superseded.label())
        .execute(&mut **transaction)
        .await?;
    }

    let row = sqlx::query(concat!(
        "INSERT INTO relationship_versions (
             id, project_id, relationship_id, version, valid_from, valid_to,
             created_at, supersedes, caused_by_topic_state, confidence, derivation
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8, $9, $10
         FROM relationship_versions WHERE relationship_id = $3
         RETURNING ",
        version_columns!()
    ))
    .bind(RelationshipVersionId::new().0)
    .bind(project.0)
    .bind(relationship.id.0)
    .bind(claim.validity.from)
    .bind(claim.validity.to)
    .bind(now)
    .bind(live.as_ref().map(|version| version.id.0))
    .bind(claim.caused_by_topic_state.map(|id| id.0))
    .bind(claim.confidence)
    .bind(claim.derivation.label())
    .fetch_one(&mut **transaction)
    .await?;

    Ok(Assertion::Appended(row_to_version(&row)))
}

/// Closes the derived edges of one kind out of a topic that are no longer
/// claimed, and returns how many.
///
/// Deriving edges only ever asserted them, so a memory rewritten from "uses
/// argo_cd" to "uses flux" kept the edge to `argo_cd` for ever, and `why[]`
/// reported a path the content it cites does not support. The graph's average
/// degree could only grow, which is also the mechanism that turns a traversal
/// from affordable into unfinishable.
///
/// `keep` is what the content says now, so this closes the difference. Derived
/// edges only: an edge somebody asserted with `pamin link` is a claim of theirs
/// and is not answered by what a memory happens to say.
///
/// Closing rather than deleting, and `closed` rather than `deleted`: the claim
/// is retracted from here on, not declared never to have held. What the memory
/// said before is still true of before, which is what `--at` reads.
pub async fn retract_derived(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    from: TopicId,
    kind: EdgeKind,
    keep: &[TopicId],
) -> Result<u64> {
    let kept: Vec<uuid::Uuid> = keep.iter().map(|topic| topic.0).collect();

    let closed = sqlx::query(
        "UPDATE relationship_versions
            SET invalidated_at = $1, tombstone_reason = $2
          WHERE invalidated_at IS NULL
            AND derivation = $3
            AND relationship_id IN (
                SELECT id FROM relationships
                 WHERE project_id = $4 AND from_topic = $5 AND kind = $6
                   AND NOT (to_topic = ANY($7))
            )",
    )
    .bind(OffsetDateTime::now_utc())
    .bind(TombstoneReason::Closed.label())
    .bind(Derivation::Deterministic.label())
    .bind(project.0)
    .bind(from.0)
    .bind(kind.label())
    .bind(&kept)
    .execute(executor)
    .await?;

    Ok(closed.rows_affected())
}

/// Closes the live version of an edge, leaving every row in place.
///
/// Returns whether anything was open to close. Closing is a retraction of the
/// claim, not a statement that the relationship ended at this instant: the
/// truth interval is untouched.
pub async fn close_edge(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
    reason: TombstoneReason,
) -> Result<bool> {
    let affected = sqlx::query(
        "UPDATE relationship_versions
         SET invalidated_at = $1, tombstone_reason = $2
         WHERE invalidated_at IS NULL
           AND relationship_id IN (
               SELECT id FROM relationships
               WHERE project_id = $3 AND from_topic = $4 AND to_topic = $5 AND kind = $6
           )",
    )
    .bind(OffsetDateTime::now_utc())
    .bind(reason.label())
    .bind(project.0)
    .bind(from.0)
    .bind(to.0)
    .bind(kind.label())
    .execute(executor)
    .await?;

    Ok(affected.rows_affected() > 0)
}

/// Looks up an edge identity without creating one.
pub async fn find_relationship(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Option<Relationship>> {
    let row = sqlx::query(FIND_RELATIONSHIP)
        .bind(project.0)
        .bind(from.0)
        .bind(to.0)
        .bind(kind.label())
        .fetch_optional(executor)
        .await?;

    Ok(row.map(|row| Relationship {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: project,
        from_topic: from,
        to_topic: to,
        kind,
        created_at: row.get("created_at"),
    }))
}

/// One topic reached from a seed, and the edge that reached it.
#[derive(Clone, Debug, PartialEq)]
pub struct Neighbor {
    pub topic: TopicId,
    /// The seed this walk started from.
    ///
    /// At one hop this is also `via`. Past that they diverge, and without it a
    /// two-hop result names the topic it arrived through but not where the
    /// walk began, which is half an explanation.
    pub origin: TopicId,
    /// Edges traversed to get here. Never zero: a seed is not its own
    /// neighbour.
    pub hops: u8,
    /// The topic on the other end of the final edge, which is what makes the
    /// connection explainable rather than merely asserted.
    pub via: TopicId,
    pub kind: EdgeKind,
    pub derivation: Derivation,
    pub confidence: f32,
}

/// How many positions the walk carries into the next hop.
///
/// A hub topic can carry tens of thousands of edges, so without a cap the cost
/// of a hop is set by the shape of the graph rather than by the question. The
/// recursive form this replaced had no way to say this at all.
const MAX_FRONTIER: usize = 2_000;

/// One walk's position: where it is, where it started, what it came through.
#[derive(Clone, Copy)]
struct Step {
    topic: TopicId,
    /// The seed this walk began at. A walk never returns to its own.
    origin: TopicId,
    /// The topic this position was reached through.
    via: TopicId,
}

/// What an edge contributes to an arrival, once its direction is discarded.
#[derive(Clone, Copy)]
struct Crossing {
    kind: EdgeKind,
    derivation: Derivation,
    confidence: f32,
}

/// The deepest walk this channel will make.
///
/// Not a preference. A topic's neighbourhood grows multiplicatively with each
/// hop, and hub topics in a real project reach five figures of degree, so one
/// hop further is not a slower query but a differently sized one. A caller who
/// wants to reach further wants a different question.
pub const MAX_DEPTH: u8 = 4;

/// How the neighbourhood query is bounded.
#[derive(Clone, Debug)]
pub struct Expansion<'a> {
    /// Maximum edges to traverse.
    pub depth: u8,
    /// Restricts traversal to these edge kinds. `None` traverses all of them.
    pub kinds: Option<&'a [EdgeKind]>,
    /// Keeps only edges asserted to hold at this instant. `None` ignores truth
    /// validity and considers every edge we still stand behind.
    pub at: Option<OffsetDateTime>,
}

impl Expansion<'_> {
    /// Bounds a walk at `depth`, or at [`MAX_DEPTH`] when that is smaller.
    ///
    /// The clamp is a floor under the library rather than the interface a
    /// caller sees: the CLI rejects an out-of-range depth so the operator
    /// learns their number was ignored rather than wondering why the walk
    /// stopped early.
    pub fn to_depth(depth: u8) -> Self {
        Self {
            depth: depth.min(MAX_DEPTH),
            kinds: None,
            at: None,
        }
    }
}

/// Walks outward from `seeds` through live edges.
///
/// Seeds themselves are returned only when something else reaches them, which
/// is the whole discipline of this channel. Seeds arrive from the lexical and
/// vector channels; handing them back as graph results would make this channel
/// a restatement of those, counting one piece of evidence twice under two
/// names. A seed that is genuinely reached from elsewhere in the graph carries
/// evidence the other channels did not supply, and only then does it belong
/// here.
///
/// The walk therefore never steps back along the edge it just took. Because
/// direction is ignored, every edge is walkable both ways, so without that rule
/// each seed would reach itself at two hops through its own first edge — the
/// double counting this channel exists to avoid, arriving through the back
/// door. Genuine cycles of three or more are still traversed.
///
/// Traversal ignores edge direction. Both ends of a `depends_on` are relevant
/// to recall, and which way the arrow points is a fact about the relationship
/// rather than about who may find whom. The direction taken is reported in
/// `via` so the path stays explainable.
///
/// A query per hop rather than one recursive pass.
///
/// The recursive form read well and did not scale. Its two intermediate views
/// were each referenced twice, so PostgreSQL materialised every edge in the
/// project and then every edge again in both directions, before the walk began.
/// The recursion itself carried no visited set, so it enumerated paths rather
/// than nodes: two adjacent topics with twenty thousand edges each are four
/// hundred million rows at two hops. And the two indexes on `relationships`
/// that exist for exactly this traversal were never touched.
///
/// A hop at a time uses them, reads only the edges of the current frontier, and
/// puts the walk where a bound can be stated: the frontier is capped, which the
/// recursive form had no way to express. The extra round trips buy that, and
/// there are at most [`MAX_DEPTH`] of them.
pub async fn expand(
    executor: impl PgExecutor<'_> + Copy,
    project: ProjectId,
    seeds: &[TopicId],
    options: &Expansion<'_>,
) -> Result<Vec<Neighbor>> {
    if seeds.is_empty() || options.depth == 0 {
        return Ok(Vec::new());
    }

    let kind_labels: Option<Vec<String>> = options
        .kinds
        .map(|kinds| kinds.iter().map(|kind| kind.label().to_string()).collect());

    // Keyed by the seed the walk began at, not by topic alone. Two seeds are
    // two walks: one may legitimately reach the other, and each is barred only
    // from returning to its own start. Merging them into one visited set would
    // silently drop the case that carries the most evidence.
    let mut seen: HashSet<(TopicId, TopicId)> = HashSet::new();
    let mut reached: HashMap<TopicId, Neighbor> = HashMap::new();

    // Each entry is one walk's position: where it is, where it started, and
    // what it came through.
    let mut frontier: Vec<Step> = seeds
        .iter()
        .map(|seed| Step {
            topic: *seed,
            origin: *seed,
            via: *seed,
        })
        .collect();

    for hop in 1..=options.depth {
        let positions: Vec<uuid::Uuid> = {
            let mut positions: Vec<uuid::Uuid> = frontier.iter().map(|step| step.topic.0).collect();
            positions.sort_unstable();
            positions.dedup();
            positions
        };

        let edges = sqlx::query(
            "SELECT r.from_topic, r.to_topic, r.kind, v.confidence, v.derivation
             FROM relationships r
             JOIN relationship_versions v ON v.relationship_id = r.id
             WHERE r.project_id = $1
               AND (r.from_topic = ANY($2) OR r.to_topic = ANY($2))
               AND ($3::TEXT[] IS NULL OR r.kind = ANY ($3))
               -- Which edges are visible depends on whether a moment was asked
               -- about. Without `--at` the question is what we still stand
               -- behind, so only uninvalidated versions count. With `--at` the
               -- question is what held then, which a later retraction does not
               -- answer on its own: an edge closed because the relationship
               -- ended still held before it was closed, one deleted because the
               -- claim was wrong never held at all, and a superseded one is
               -- answered by its successor. That distinction is what
               -- tombstone_reason records, and ignoring it made every
               -- retraction erase its own history.
               AND CASE WHEN $4::TIMESTAMPTZ IS NULL
                   THEN v.invalidated_at IS NULL
                   ELSE (v.valid_from IS NULL OR v.valid_from <= $4)
                        AND (v.valid_to IS NULL OR $4 < v.valid_to)
                        AND (v.invalidated_at IS NULL
                             OR (v.tombstone_reason = 'closed'
                                 AND $4 < v.invalidated_at))
                   END",
        )
        .bind(project.0)
        .bind(&positions)
        .bind(kind_labels.as_deref())
        .bind(options.at)
        .fetch_all(executor)
        .await?;

        // Undirected: both ends of a `depends_on` are relevant to recall, and
        // which way the arrow points is a fact about the relationship rather
        // than about who may find whom, so each edge is entered from both ends.
        let mut neighbours: HashMap<TopicId, Vec<(TopicId, Crossing)>> = HashMap::new();
        for row in &edges {
            let from = TopicId::from(row.get::<uuid::Uuid, _>("from_topic"));
            let to = TopicId::from(row.get::<uuid::Uuid, _>("to_topic"));
            let crossing = Crossing {
                kind: EdgeKind::from_label(row.get("kind")).unwrap_or(EdgeKind::RelatedTo),
                derivation: Derivation::from_label(row.get("derivation"))
                    .unwrap_or(Derivation::Imported),
                confidence: row.get("confidence"),
            };

            neighbours.entry(from).or_default().push((to, crossing));
            neighbours.entry(to).or_default().push((from, crossing));
        }

        let mut next: Vec<(Step, f32)> = Vec::new();
        for step in &frontier {
            let Some(crossings) = neighbours.get(&step.topic) else {
                continue;
            };

            for (neighbour, crossing) in crossings {
                // Never back to where this walk started, and never straight
                // back along the edge just taken. Both would return a topic
                // already reached at a shorter distance; the first is the seed
                // itself, which this channel must not hand back.
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
                };

                // Breadth first, so the first arrival is the shortest. Among
                // arrivals at the same distance the most confident one wins,
                // which is the order the ranking below expects.
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

        // The bound the recursive form could not state. A hub topic can carry
        // tens of thousands of edges, so without this the next hop's query
        // grows with the graph rather than with the question; the walk keeps
        // the arrivals it is most confident in.
        next.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next.truncate(MAX_FRONTIER);

        frontier = next.into_iter().map(|(step, _)| step).collect();
    }

    let mut neighbors: Vec<Neighbor> = reached.into_values().collect();

    // Collected by topic, so the ranking has to be imposed here. The
    // identifier tie-break keeps the order stable across identical inputs,
    // which is what lets an assembled context be reused rather than rebuilt.
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

    Ok(neighbors)
}
