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
        version_columns!("")
    };
    // Qualified, for the one query that joins another table alongside. Written
    // out rather than assembled from the unqualified list, because a macro that
    // pastes a prefix onto a comma-separated string cannot be read at the call
    // site and this is a list the reader has to be able to check.
    ("v.") => {
        "v.id, v.relationship_id, v.version, v.valid_from, v.valid_to, v.created_at, \
         v.invalidated_at, v.supersedes, v.caused_by_topic_state, v.confidence, \
         v.derivation, v.tombstone_reason"
    };
    ("") => {
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

/// The same lookup, locking the row it finds.
///
/// [`assert_within`] is about to append a version under that identity, and the
/// lock used to be a second statement: the same row read again by id, for
/// nothing but `FOR UPDATE`. Two round trips where the work is one. Taking it
/// in the lookup is also the stricter order -- the old form read the row and
/// *then* locked it, so the read was outside the lock it exists to be inside.
///
/// Derived from the constant above rather than written out, so the two cannot
/// drift into asking different questions. The unlocked form stays public: a
/// lookup is a lookup, and `FOR UPDATE` outside a transaction locks a row for
/// no longer than the statement.
const LOCK_RELATIONSHIP: &str = concat!(
    "SELECT id, created_at FROM relationships
                                 WHERE project_id = $1 AND from_topic = $2
                                   AND to_topic = $3 AND kind = $4",
    " FOR UPDATE"
);

/// Returns the edge identity for this pair and kind, creating it if absent.
///
/// One identity per (pair, kind): two topics can be related several ways at
/// once, and each way carries its own history.
/// Takes a connection rather than any executor because it uses it twice, and a
/// pooled connection is not something that can be handed out twice.
///
/// Private: `assert_within` is the only caller, and the lock the lookup takes
/// only means anything inside that transaction.
async fn ensure_relationship(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Relationship> {
    // Read first: an edge is created once and re-asserted on every rewrite of
    // the memory that derives it, so the insert is the rare path. See
    // `repository::ensure_project` for why the conflict clause does not update.
    if let Some(relationship) =
        locked_relationship(&mut *connection, project, from, to, kind).await?
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
        // Another writer created it in between, and it has to be locked here
        // too: this is the path where two writers raced, which is exactly when
        // the lock matters.
        None => {
            sqlx::query(LOCK_RELATIONSHIP)
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
///
/// Takes the project because the only index holding every version is keyed
/// `(project_id, relationship_id, version)`; the one keyed by relationship
/// alone holds live versions only.
pub async fn edge_history(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    relationship: RelationshipId,
) -> Result<Vec<RelationshipVersion>> {
    let rows = sqlx::query(concat!(
        "SELECT ",
        version_columns!(),
        " FROM relationship_versions
          WHERE project_id = $1 AND relationship_id = $2 ORDER BY version ASC"
    ))
    .bind(project.0)
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
    // Through the batched read with one edge in it, so that "would this claim
    // change anything" has one implementation. It had two, and the one on this
    // path was the one no test measured.
    if let Some(unchanged) = live_versions_of(pool, project, &[(from, to, claim.clone())])
        .await?
        .remove(&(from, to, claim.kind))
        .filter(|version| claim.matches(version))
    {
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
    //
    // In one round trip rather than two per edge. This was a loop calling
    // `already_asserted`, which is itself two queries -- find the identity,
    // read its live version -- so a memory naming ten topics asked twenty
    // questions to find out that the answer to all of them was "nothing to
    // do". That is the case the transaction below is skipped for, so it was
    // the cheap path paying the most.
    let live = live_versions_of(pool, project, edges).await?;
    let mut asserted: Vec<Option<Assertion>> = Vec::with_capacity(edges.len());
    let mut pending = Vec::new();
    for (index, (from, to, claim)) in edges.iter().enumerate() {
        let unchanged = live
            .get(&(*from, *to, claim.kind))
            .filter(|version| claim.matches(version));
        if unchanged.is_none() {
            pending.push(index);
        }
        asserted.push(unchanged.cloned().map(Assertion::Unchanged));
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

/// The live version of each of these edges that has one, in one round trip.
///
/// Keyed by the triple the caller asked about rather than by relationship id,
/// because that is what the caller has; an edge with no identity yet, or an
/// identity whose every version has been invalidated, is simply absent.
///
/// `DISTINCT ON` is what makes this one query instead of one per edge:
/// `live_version` reads the newest surviving version of one relationship with
/// an `ORDER BY ... LIMIT 1`, and the same order per relationship is exactly
/// what `DISTINCT ON (r.id)` keeps.
///
/// The three arrays are unnested into rows rather than built into the SQL,
/// which is what lets this stay a `'static` statement -- the driver takes one
/// of those or an explicit assertion that a built string is safe, and building
/// SQL from a caller's topic ids is the shape that assertion exists to
/// discourage.
async fn live_versions_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    edges: &[(TopicId, TopicId, EdgeClaim)],
) -> Result<HashMap<(TopicId, TopicId, EdgeKind), RelationshipVersion>> {
    if edges.is_empty() {
        return Ok(HashMap::new());
    }

    let from: Vec<uuid::Uuid> = edges.iter().map(|(from, _, _)| from.0).collect();
    let to: Vec<uuid::Uuid> = edges.iter().map(|(_, to, _)| to.0).collect();
    let kinds: Vec<String> = edges
        .iter()
        .map(|(_, _, claim)| claim.kind.label().to_string())
        .collect();

    let rows = sqlx::query(concat!(
        "SELECT DISTINCT ON (r.id) r.from_topic, r.to_topic, r.kind, ",
        version_columns!("v."),
        " FROM unnest($2::uuid[], $3::uuid[], $4::text[]) AS wanted(from_topic, to_topic, kind)
           JOIN relationships r
             ON r.project_id = $1 AND r.from_topic = wanted.from_topic
            AND r.to_topic = wanted.to_topic AND r.kind = wanted.kind
           JOIN relationship_versions v
             ON v.relationship_id = r.id AND v.invalidated_at IS NULL
          ORDER BY r.id, v.version DESC"
    ))
    .bind(project.0)
    .bind(&from)
    .bind(&to)
    .bind(&kinds)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .filter_map(|row| {
            let kind = EdgeKind::parse(row.get::<&str, _>("kind"))?;
            Some((
                (
                    TopicId(row.get::<uuid::Uuid, _>("from_topic")),
                    TopicId(row.get::<uuid::Uuid, _>("to_topic")),
                    kind,
                ),
                row_to_version(row),
            ))
        })
        .collect())
}

async fn assert_within(
    transaction: &mut sqlx::PgTransaction<'_>,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Assertion> {
    // Locked by the lookup inside this, so the version append below is
    // serialised against another writer asserting the same edge.
    let relationship = ensure_relationship(transaction, project, from, to, claim.kind).await?;

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
             created_at, supersedes, caused_by_topic_state, confidence, derivation,
             from_topic, to_topic, kind
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13
         FROM relationship_versions WHERE project_id = $2 AND relationship_id = $3
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
    // The identity's own, copied so a walk can read a topic's strongest live
    // edges from one index. See `V13__edge_endpoints_on_versions.sql`.
    .bind(relationship.from_topic.0)
    .bind(relationship.to_topic.0)
    .bind(relationship.kind.label())
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
    retract_derived_all(executor, project, kind, &[(from, keep.to_vec())]).await
}

/// [`retract_derived`] for several topics, in one statement.
///
/// Each entry is a topic and what its content says now. What a topic keeps is
/// passed as pairs, since the lists differ in length and an array of arrays
/// in PostgreSQL has to be rectangular.
pub async fn retract_derived_all(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    kind: EdgeKind,
    keep: &[(TopicId, Vec<TopicId>)],
) -> Result<u64> {
    if keep.is_empty() {
        return Ok(0);
    }
    let froms: Vec<uuid::Uuid> = keep.iter().map(|(from, _)| from.0).collect();
    let (kept_from, kept_to): (Vec<uuid::Uuid>, Vec<uuid::Uuid>) = keep
        .iter()
        .flat_map(|(from, to)| to.iter().map(|to| (from.0, to.0)))
        .unzip();

    let closed = sqlx::query(
        "UPDATE relationship_versions
            SET invalidated_at = $1, tombstone_reason = $2
          WHERE invalidated_at IS NULL
            AND derivation = $3
            AND relationship_id IN (
                SELECT r.id FROM relationships r
                 WHERE r.project_id = $4 AND r.from_topic = ANY($5) AND r.kind = $6
                   AND NOT EXISTS (
                       SELECT 1 FROM unnest($7::uuid[], $8::uuid[]) AS kept (from_topic, to_topic)
                        WHERE kept.from_topic = r.from_topic AND kept.to_topic = r.to_topic
                   )
            )",
    )
    .bind(OffsetDateTime::now_utc())
    .bind(TombstoneReason::Closed.label())
    .bind(Derivation::Deterministic.label())
    .bind(project.0)
    .bind(&froms)
    .bind(kind.label())
    .bind(&kept_from)
    .bind(&kept_to)
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

    Ok(row.map(|row| relationship_row(&row, project, from, to, kind)))
}

/// The same lookup, holding the row until the transaction ends.
async fn locked_relationship(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Option<Relationship>> {
    let row = sqlx::query(LOCK_RELATIONSHIP)
        .bind(project.0)
        .bind(from.0)
        .bind(to.0)
        .bind(kind.label())
        .fetch_optional(executor)
        .await?;

    Ok(row.map(|row| relationship_row(&row, project, from, to, kind)))
}

fn relationship_row(
    row: &sqlx::postgres::PgRow,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Relationship {
    Relationship {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: project,
        from_topic: from,
        to_topic: to,
        kind,
        created_at: row.get("created_at"),
    }
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
    /// Whether the final edge was asserted from `via` to `topic`.
    ///
    /// The walk ignores direction, because both ends of a `depends_on` are
    /// relevant to recall. But which end asserted it is the claim itself for
    /// `depends_on`, `supersedes`, `contradicts`, `derived_from` and
    /// `part_of`, and without this the same edge reads one way walked from one
    /// end and the opposite way walked from the other -- so a caller asking
    /// what depends on what could not answer from the result.
    pub outbound: bool,
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

/// What an edge contributes to an arrival.
#[derive(Clone, Copy)]
struct Crossing {
    kind: EdgeKind,
    derivation: Derivation,
    confidence: f32,
    /// Whether the edge points away from the position holding this crossing.
    ///
    /// The walk is undirected, so each edge is entered from both ends and the
    /// same edge yields one crossing with this set and one without. Keeping it
    /// is what lets an arrival report the direction the edge was asserted in,
    /// rather than the direction the walk happened to take.
    outbound: bool,
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
    /// How many neighbours the caller is going to keep, if it knows.
    ///
    /// The walk stops once it has that many, which costs nothing and is not an
    /// approximation: arrivals are breadth first, the ranking sorts on hops
    /// before anything else, and a later hop only ever improves an entry from
    /// the *same* hop. So once a hop has produced this many, every arrival
    /// still to come sorts behind all of them and cannot enter the result.
    ///
    /// `None` walks to the depth whatever it finds, which is what
    /// `pamin neighbors` wants -- it is asking what the neighbourhood *is*,
    /// not for the best few of it.
    pub keep: Option<usize>,
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
            keep: None,
        }
    }

    /// The same walk, stopped once `keep` neighbours have been found.
    pub fn keeping(self, keep: usize) -> Self {
        Self {
            keep: Some(keep),
            ..self
        }
    }
}

/// The edges incident on a frontier, `$2`, of the kinds in `$3` (all of them
/// when it is null), before the question of which versions count.
///
/// The version is joined on its project as well as its relationship, which is
/// the prefix of its unique key.
macro_rules! edges_touching {
    () => {
        "SELECT r.from_topic, r.to_topic, r.kind, v.confidence, v.derivation
         FROM relationships r
         JOIN relationship_versions v
           ON v.project_id = r.project_id AND v.relationship_id = r.id
         WHERE r.project_id = $1
           AND (r.from_topic = ANY($2) OR r.to_topic = ANY($2))
           AND ($3::TEXT[] IS NULL OR r.kind = ANY ($3))"
    };
}

/// The `$4` strongest live edges each way of every topic in `$2`, of the kinds
/// in `$3` (all of them when it is null).
///
/// Strongest is most confident, then the topic on the other end in ascending
/// order, which is the order [`crossing_order`] takes crossings in -- so what a
/// topic's edges are cut to is exactly the head of the list the walk would
/// have read, and what is left out is exactly its tail. Each side says which
/// way it was read, so the walk can tell a side that ran out of edges from one
/// the `LIMIT` cut.
///
/// Each side is served by a partial index in that order
/// (`relationship_versions_live_from`, `relationship_versions_live_to`), so a
/// hub's side costs `$4` index entries rather than its degree. Said as the
/// index's own predicate, `invalidated_at IS NULL`, for the reason the live
/// statement below says it: a generic plan can only use a partial index whose
/// predicate the statement states.
const STRONGEST: &str = "
    SELECT p.topic AS standing, TRUE AS outward,
           e.from_topic, e.to_topic, e.kind, e.confidence, e.derivation
      FROM unnest($2::uuid[]) AS p (topic)
     CROSS JOIN LATERAL (
           SELECT v.from_topic, v.to_topic, v.kind, v.confidence, v.derivation
             FROM relationship_versions v
            WHERE v.project_id = $1 AND v.from_topic = p.topic
              AND v.invalidated_at IS NULL
              AND ($3::TEXT[] IS NULL OR v.kind = ANY ($3))
            ORDER BY v.confidence DESC, v.to_topic
            LIMIT $4) e
    UNION ALL
    SELECT p.topic, FALSE,
           e.from_topic, e.to_topic, e.kind, e.confidence, e.derivation
      FROM unnest($2::uuid[]) AS p (topic)
     CROSS JOIN LATERAL (
           SELECT v.from_topic, v.to_topic, v.kind, v.confidence, v.derivation
             FROM relationship_versions v
            WHERE v.project_id = $1 AND v.to_topic = p.topic
              AND v.invalidated_at IS NULL
              AND ($3::TEXT[] IS NULL OR v.kind = ANY ($3))
            ORDER BY v.confidence DESC, v.from_topic
            LIMIT $4) e";

/// Every live edge between a topic in `$2` and a topic in `$3`, either way
/// round, of the kinds in `$4`.
///
/// Through the identity's unique key, `(project_id, from_topic, to_topic,
/// kind)`, so this costs the pairs asked about and not the degree of the hub
/// that made them worth asking about.
const BETWEEN: &str = "
    SELECT r.from_topic, r.to_topic, r.kind, v.confidence, v.derivation
      FROM relationships r
      JOIN relationship_versions v
        ON v.project_id = r.project_id AND v.relationship_id = r.id
     WHERE r.project_id = $1
       AND ((r.from_topic = ANY ($2) AND r.to_topic = ANY ($3))
         OR (r.to_topic = ANY ($2) AND r.from_topic = ANY ($3)))
       AND ($4::TEXT[] IS NULL OR r.kind = ANY ($4))
       AND v.invalidated_at IS NULL";

/// One live edge, as a hop reads it.
#[derive(Clone, Copy)]
struct Edge {
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
    derivation: Derivation,
    confidence: f32,
}

impl Edge {
    fn read(row: &PgRow) -> Self {
        Self {
            from: TopicId::from(row.get::<uuid::Uuid, _>("from_topic")),
            to: TopicId::from(row.get::<uuid::Uuid, _>("to_topic")),
            kind: EdgeKind::from_label(row.get("kind")).unwrap_or(EdgeKind::RelatedTo),
            derivation: Derivation::from_label(row.get("derivation"))
                .unwrap_or(Derivation::Imported),
            confidence: row.get("confidence"),
        }
    }
}

/// The order a topic's crossings are taken in: most confident first, then by
/// the topic on the other end, then by kind, then by which end asserted it.
///
/// Every rule of the walk that chooses between two arrivals keeps the first it
/// meets, so this order is what settles every tie. It used to be the order
/// PostgreSQL happened to return rows in, which a statement without an
/// `ORDER BY` does not promise: one topic reached from one seed over two edges
/// kept whichever row came first, not the more confident one.
///
/// It is also an order one statement could have returned every row in -- by
/// confidence, then the lesser endpoint, the greater, the kind, and
/// `from_topic` -- so a walk taking it is a walk the unordered statement could
/// have produced. And its first two keys are the `ORDER BY` of [`STRONGEST`],
/// which is what makes cutting a topic's edges there cut the tail of this list.
fn crossing_order(
    standing: TopicId,
) -> impl Fn(&(TopicId, Crossing), &(TopicId, Crossing)) -> std::cmp::Ordering {
    move |(left, one), (right, other)| {
        let asserted_from = |neighbour: &TopicId, crossing: &Crossing| {
            if crossing.outbound {
                standing.0
            } else {
                neighbour.0
            }
        };
        other
            .confidence
            .total_cmp(&one.confidence)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| one.kind.cmp(&other.kind))
            .then_with(|| asserted_from(left, one).cmp(&asserted_from(right, other)))
    }
}

/// The edges a frontier stands on, of every kind asked for and valid when
/// asked about. The statement the walk had before it could be bounded.
async fn every_edge(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    positions: &[uuid::Uuid],
    kinds: Option<&[String]>,
    at: Option<OffsetDateTime>,
) -> Result<Vec<Edge>> {
    // Which edges are visible depends on whether a moment was asked about,
    // and the two questions are two statements rather than one with a
    // `CASE` on the parameter. Without `--at` the question is what we still
    // stand behind, so only uninvalidated versions count -- and said
    // literally, that is the predicate of the partial index
    // `relationship_versions_live`, which a prepared statement's generic plan
    // can use only if the statement says it rather than a branch on a
    // parameter the plan has not seen. With `--at` the question is what held
    // then, which a later retraction does not answer on its own: an edge
    // closed because the relationship ended still held before it was closed,
    // one deleted because the claim was wrong never held at all, and a
    // superseded one is answered by its successor. That distinction is what
    // tombstone_reason records, and ignoring it made every retraction erase
    // its own history.
    const LIVE: &str = concat!(edges_touching!(), " AND v.invalidated_at IS NULL");
    const AT: &str = concat!(
        edges_touching!(),
        " AND (v.valid_from IS NULL OR v.valid_from <= $4)
          AND (v.valid_to IS NULL OR $4 < v.valid_to)
          AND (v.invalidated_at IS NULL
               OR (v.tombstone_reason = 'closed' AND $4 < v.invalidated_at))"
    );
    let query = sqlx::query(if at.is_some() { AT } else { LIVE })
        .bind(project.0)
        .bind(positions)
        .bind(kinds);
    let query = match at {
        Some(at) => query.bind(at),
        None => query,
    };
    Ok(query
        .fetch_all(&mut *connection)
        .await?
        .iter()
        .map(Edge::read)
        .collect())
}

/// Where a bounded read stopped short on one topic: the last crossing it read
/// on the side that was cut, and the earlier of the two when both were.
struct Cut {
    standing: TopicId,
    confidence: f32,
    after: TopicId,
}

/// A frontier's edges, at most `per_topic` of them each way per topic, and
/// where that left something unread.
///
/// Cutting a topic's crossings to the head of their order drops arrivals that
/// are not only its own. A topic the head *does* reach may be reached by
/// another seed as well, and which of the two arrivals the walk keeps depends
/// on both -- so the cut could change a topic that is in the result, not just
/// leave one out. That is closed here: for every topic that was cut, every
/// edge between it and any topic this hop read is read too, by pair, which
/// costs the pairs rather than the degree. After that, every topic the walk
/// reaches at this hop is reached exactly as it would have been without the
/// cut, and what the cut can still hide is a topic nothing read reached.
async fn strongest_edges(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    positions: &[uuid::Uuid],
    kinds: Option<&[String]>,
    per_topic: usize,
) -> Result<(Vec<Edge>, Vec<Cut>)> {
    let rows = sqlx::query(STRONGEST)
        .bind(project.0)
        .bind(positions)
        .bind(kinds)
        .bind(i64::try_from(per_topic).unwrap_or(i64::MAX))
        .fetch_all(&mut *connection)
        .await?;

    // The last crossing each side read, and how many it read.
    let mut sides: HashMap<(TopicId, bool), (usize, f32, TopicId)> = HashMap::new();
    let mut edges = Vec::with_capacity(rows.len());
    for row in &rows {
        let edge = Edge::read(row);
        let standing = TopicId::from(row.get::<uuid::Uuid, _>("standing"));
        let outward: bool = row.get("outward");
        let other = if outward { edge.to } else { edge.from };
        let side = sides
            .entry((standing, outward))
            .or_insert((0, edge.confidence, other));
        side.0 += 1;
        // Later in the order: less confident, or as confident and further on.
        if edge.confidence < side.1 || (edge.confidence == side.1 && other.0 > side.2.0) {
            side.1 = edge.confidence;
            side.2 = other;
        }
        edges.push(edge);
    }

    // A side that read as many as it was allowed may have had more. One that
    // read exactly that many and no more is counted as cut, which costs a
    // lookup and loses nothing.
    let mut cuts: HashMap<TopicId, Cut> = HashMap::new();
    for ((standing, _), (read, confidence, after)) in sides {
        if read < per_topic {
            continue;
        }
        let cut = cuts.entry(standing).or_insert(Cut {
            standing,
            confidence,
            after,
        });
        // The earlier of two cut sides bounds both: what either left unread
        // comes after its own last crossing, so after the earlier one.
        if confidence > cut.confidence || (confidence == cut.confidence && after.0 < cut.after.0) {
            cut.confidence = confidence;
            cut.after = after;
        }
    }
    let mut cuts: Vec<Cut> = cuts.into_values().collect();
    cuts.sort_by_key(|cut| cut.standing.0);

    if !cuts.is_empty() {
        let cut: Vec<uuid::Uuid> = cuts.iter().map(|cut| cut.standing.0).collect();
        let touched: Vec<uuid::Uuid> = {
            let mut touched: Vec<uuid::Uuid> = edges
                .iter()
                .flat_map(|edge| [edge.from.0, edge.to.0])
                .collect();
            touched.sort_unstable();
            touched.dedup();
            touched
        };
        let rows = sqlx::query(BETWEEN)
            .bind(project.0)
            .bind(&cut)
            .bind(&touched)
            .bind(kinds)
            .fetch_all(&mut *connection)
            .await?;
        edges.extend(rows.iter().map(Edge::read));
    }

    // The same edge arrives once per side that read it, and again from the
    // pairs above. One live version per identity, so the identity is the key.
    let mut once = HashSet::new();
    edges.retain(|edge| once.insert((edge.from, edge.to, edge.kind)));

    Ok((edges, cuts))
}

/// Crosses one hop's edges from every position of the frontier.
///
/// Records what each walk reaches in `seen` and `reached`, and returns the
/// positions the next hop would start from, each with the confidence of the
/// edge that led there.
fn cross(
    hop: u8,
    frontier: &[Step],
    edges: &[Edge],
    seen: &mut HashSet<(TopicId, TopicId)>,
    reached: &mut HashMap<TopicId, Neighbor>,
) -> Vec<(Step, f32)> {
    // Undirected: both ends of a `depends_on` are relevant to recall, and
    // which way the arrow points is a fact about the relationship rather than
    // about who may find whom, so each edge is entered from both ends.
    let mut neighbours: HashMap<TopicId, Vec<(TopicId, Crossing)>> = HashMap::new();
    for edge in edges {
        let crossing = Crossing {
            kind: edge.kind,
            derivation: edge.derivation,
            confidence: edge.confidence,
            outbound: true,
        };

        neighbours.entry(edge.from).or_default().push((
            edge.to,
            Crossing {
                outbound: true,
                ..crossing
            },
        ));
        neighbours.entry(edge.to).or_default().push((
            edge.from,
            Crossing {
                outbound: false,
                ..crossing
            },
        ));
    }
    for (standing, crossings) in &mut neighbours {
        crossings.sort_by(crossing_order(*standing));
    }

    let mut next: Vec<(Step, f32)> = Vec::new();
    for step in frontier {
        let Some(crossings) = neighbours.get(&step.topic) else {
            continue;
        };

        for (neighbour, crossing) in crossings {
            // Never back to where this walk started, and never straight back
            // along the edge just taken. Both would return a topic already
            // reached at a shorter distance; the first is the seed itself,
            // which this channel must not hand back.
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

            // Breadth first, so the first arrival is the shortest. Among
            // arrivals at the same distance the most confident one wins, which
            // is the order the ranking below expects.
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
    next
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
/// rather than about who may find whom. `via` reports the topic on the other
/// end of the final edge, and `outbound` reports which of the two asserted the
/// edge -- both, because `via` alone says how the walk arrived and not what was
/// claimed, and for `depends_on` the claim is the whole content.
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
///
/// Takes one connection and asks every hop on it: a statement run on the pool
/// returns its connection afterwards, and sqlx checks a returned connection
/// with a round trip of its own.
///
/// Reads every edge of every topic it stands on. [`expand_reading`] is the
/// same walk reading only the strongest few, for a caller that ranks what it
/// gets and keeps the head.
pub async fn expand(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    seeds: &[TopicId],
    options: &Expansion<'_>,
) -> Result<Vec<Neighbor>> {
    Ok(walk(connection, project, seeds, options, None)
        .await?
        .neighbors)
}

/// A walk that read at most `per_topic` of each topic's edges each way, and
/// what it left unread.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    /// Every topic the walk reached, each exactly as [`expand`] reaches it:
    /// same hop, same origin, same final edge.
    pub neighbors: Vec<Neighbor>,
    /// Where a topic's edges were cut. Empty when nothing was, and then
    /// `neighbors` is everything [`expand`] returns.
    pub unread: Vec<Unread>,
}

/// One topic whose edges a bounded walk did not read to the end.
///
/// The claim is about the topics [`expand`] would reach and [`Walk`] does not
/// hold: every one of them was reached at `hops`, from `via`, by the walk of
/// one of `origins`, over an edge less confident than `confidence` -- or as
/// confident, to a topic whose identifier sorts after `after`. Nothing else is
/// missing, and nothing present differs.
#[derive(Clone, Debug, PartialEq)]
pub struct Unread {
    pub hops: u8,
    /// The topic whose edges were cut.
    pub via: TopicId,
    pub confidence: f32,
    pub after: TopicId,
    /// The seeds whose walks stood on `via` when it was cut.
    pub origins: Vec<TopicId>,
}

/// [`expand`], reading at most `per_topic` of each topic's edges each way,
/// strongest first.
///
/// **What this is for.** A hub carries thousands of edges and a caller keeping
/// fifty ranked arrivals needs few of them, but reading them was the walk's
/// cost: every edge of the hub, fetched, mapped twice and sorted, for a list
/// cut to fifty afterwards.
///
/// **Why it is not a plain `LIMIT`.** What a caller ranks by -- confidence
/// times how relevant the arrival's seed was, in the engine -- is not known
/// here, and which of two seeds' arrivals at a topic the walk keeps is decided
/// by confidence alone, so a cut on one seed can change a topic that another
/// seed reached. So the cut is made where it can be stated and checked
/// instead of assumed: every topic in [`Walk::neighbors`] is exactly what
/// [`expand`] makes of it, and [`Walk::unread`] bounds every one that is
/// missing. [`Walk::strongest`] then decides, for the caller's own ranking,
/// whether anything missing could have ranked, and says so rather than guess.
///
/// The walk only keeps a cut hop when it stops there -- at `options.depth`, or
/// with `options.keep` reached, which a cut topic's own edges nearly always
/// see to. A cut hop the walk would have gone on from is read again in full:
/// a hop's frontier is made of everything it reached, and past a cut that is
/// not known.
///
/// With `--at` the bound is not used and every edge is read: the partial
/// indexes this rests on hold live versions only, and what held at a moment
/// is a different set.
pub async fn expand_reading(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    seeds: &[TopicId],
    options: &Expansion<'_>,
    per_topic: usize,
) -> Result<Walk> {
    // At least one, so a cut side always has a last crossing to bound it by.
    walk(connection, project, seeds, options, Some(per_topic.max(1))).await
}

async fn walk(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    seeds: &[TopicId],
    options: &Expansion<'_>,
    per_topic: Option<usize>,
) -> Result<Walk> {
    if seeds.is_empty() || options.depth == 0 {
        return Ok(Walk::default());
    }

    let kind_labels: Option<Vec<String>> = options
        .kinds
        .map(|kinds| kinds.iter().map(|kind| kind.label().to_string()).collect());
    let per_topic = per_topic.filter(|_| options.at.is_none());

    // Keyed by the seed the walk began at, not by topic alone. Two seeds are
    // two walks: one may legitimately reach the other, and each is barred only
    // from returning to its own start. Merging them into one visited set would
    // silently drop the case that carries the most evidence.
    let mut seen: HashSet<(TopicId, TopicId)> = HashSet::new();
    let mut reached: HashMap<TopicId, Neighbor> = HashMap::new();
    let mut unread = Vec::new();

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

        let (edges, cuts) = match per_topic {
            Some(per_topic) => {
                strongest_edges(
                    &mut *connection,
                    project,
                    &positions,
                    kind_labels.as_deref(),
                    per_topic,
                )
                .await?
            }
            None => (
                every_edge(
                    &mut *connection,
                    project,
                    &positions,
                    kind_labels.as_deref(),
                    options.at,
                )
                .await?,
                Vec::new(),
            ),
        };

        let mut next = if cuts.is_empty() {
            cross(hop, &frontier, &edges, &mut seen, &mut reached)
        } else {
            let (seen_before, reached_before) = (seen.clone(), reached.clone());
            let next = cross(hop, &frontier, &edges, &mut seen, &mut reached);
            let stops =
                hop == options.depth || options.keep.is_some_and(|keep| reached.len() >= keep);
            if stops {
                tracing::debug!(
                    hop,
                    cut = cuts.len(),
                    "the walk read only the strongest edges of some topics"
                );
                // Every walk that stood on a cut topic, so the bound can be
                // priced for each seed it might have credited.
                for cut in &cuts {
                    let mut origins: Vec<TopicId> = frontier
                        .iter()
                        .filter(|step| step.topic == cut.standing)
                        .map(|step| step.origin)
                        .collect();
                    origins.sort_unstable_by_key(|origin| origin.0);
                    origins.dedup();
                    unread.push(Unread {
                        hops: hop,
                        via: cut.standing,
                        confidence: cut.confidence,
                        after: cut.after,
                        origins,
                    });
                }
                next
            } else {
                // The walk goes on from here, and the next frontier is made
                // of what this hop reached -- which past a cut is not known.
                seen = seen_before;
                reached = reached_before;
                let edges = every_edge(
                    &mut *connection,
                    project,
                    &positions,
                    kind_labels.as_deref(),
                    options.at,
                )
                .await?;
                cross(hop, &frontier, &edges, &mut seen, &mut reached)
            }
        };

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

        // Enough for whoever asked. Going further can only add arrivals at a
        // greater distance, and distance is the first thing the ranking sorts
        // on, so none of them could displace what is already here. On a graph
        // with hubs in it this is the difference between walking the
        // neighbourhood and walking the project: two hops off a
        // twenty-thousand-degree hub reaches forty thousand topics, sorts all
        // of them, and hands back fifty.
        if options.keep.is_some_and(|keep| reached.len() >= keep) {
            break;
        }
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

    Ok(Walk { neighbors, unread })
}

/// The `keep` arrivals `score` rates highest, highest first, ties by topic.
///
/// `score` is given an arrival's edge confidence, its hops and the seed it
/// came from.
pub fn strongest(
    mut neighbors: Vec<Neighbor>,
    keep: usize,
    score: impl Fn(f32, u8, TopicId) -> f32,
) -> Vec<Neighbor> {
    let rate = |neighbor: &Neighbor| score(neighbor.confidence, neighbor.hops, neighbor.origin);
    neighbors.sort_by(|left, right| {
        rate(right)
            .total_cmp(&rate(left))
            .then_with(|| left.topic.0.cmp(&right.topic.0))
    });
    neighbors.truncate(keep);
    neighbors
}

impl Walk {
    /// [`strongest`], when what the walk left unread cannot change it, and
    /// `None` when it might.
    ///
    /// `score` must depend on nothing but the three things it is given, and
    /// must not fall as confidence rises. Then every topic missing from this
    /// walk rates no higher than its [`Unread`] at that entry's confidence --
    /// and, rating exactly that, sorts after `after` -- or no higher than the
    /// next confidence down. So if the last arrival kept outranks all of that,
    /// nothing missing could have been kept, and nothing kept differs from
    /// what [`expand`] would have given.
    ///
    /// Fewer than `keep` arrivals is `None` whenever anything was cut, since
    /// an unread edge could have filled the gap.
    pub fn strongest(
        self,
        keep: usize,
        score: impl Fn(f32, u8, TopicId) -> f32,
    ) -> Option<Vec<Neighbor>> {
        let ranked = strongest(self.neighbors, keep, &score);
        if self.unread.is_empty() || keep == 0 {
            return Some(ranked);
        }
        let last = ranked.get(keep - 1)?;
        let floor = score(last.confidence, last.hops, last.origin);
        let settled = self.unread.iter().all(|unread| {
            unread.origins.iter().all(|origin| {
                let bound = score(unread.confidence, unread.hops, *origin);
                let below = score(unread.confidence.next_down(), unread.hops, *origin);
                floor > bound || (floor == bound && last.topic.0 <= unread.after.0 && below < floor)
            })
        });
        settled.then_some(ranked)
    }
}
