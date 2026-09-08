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
use time::OffsetDateTime;
use tokio_postgres::{Client, GenericClient, Row};

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

const VERSION_COLUMNS: &str = "id, relationship_id, version, valid_from, valid_to, created_at, \
     invalidated_at, supersedes, caused_by_topic_state, confidence, derivation, tombstone_reason";

fn row_to_version(row: &Row) -> RelationshipVersion {
    RelationshipVersion {
        id: row.get::<_, uuid::Uuid>("id").into(),
        relationship_id: row.get::<_, uuid::Uuid>("relationship_id").into(),
        version: row.get::<_, i32>("version") as u32,
        validity: Validity::new(row.get("valid_from"), row.get("valid_to")),
        created_at: row.get("created_at"),
        invalidated_at: row.get("invalidated_at"),
        supersedes: row
            .get::<_, Option<uuid::Uuid>>("supersedes")
            .map(Into::into),
        caused_by_topic_state: row
            .get::<_, Option<uuid::Uuid>>("caused_by_topic_state")
            .map(Into::into),
        confidence: row.get("confidence"),
        derivation: Derivation::from_label(row.get("derivation"))
            // The column's CHECK constraint admits nothing else.
            .unwrap_or(Derivation::Imported),
        tombstone_reason: row
            .get::<_, Option<&str>>("tombstone_reason")
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
pub async fn ensure_relationship<C: GenericClient>(
    client: &C,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Relationship> {
    // Read first: an edge is created once and re-asserted on every rewrite of
    // the memory that derives it, so the insert is the rare path. See
    // `repository::ensure_project` for why the conflict clause does not update.
    if let Some(relationship) = find_relationship(client, project, from, to, kind).await? {
        return Ok(relationship);
    }

    let inserted = client
        .query_opt(
            "INSERT INTO relationships
                 (id, project_id, from_topic, to_topic, kind, created_at)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (project_id, from_topic, to_topic, kind) DO NOTHING
             RETURNING id, created_at",
            &[
                &RelationshipId::new().0,
                &project.0,
                &from.0,
                &to.0,
                &kind.label(),
                &OffsetDateTime::now_utc(),
            ],
        )
        .await?;

    let row = match inserted {
        Some(row) => row,
        // Another writer created it in between.
        None => {
            client
                .query_one(
                    FIND_RELATIONSHIP,
                    &[&project.0, &from.0, &to.0, &kind.label()],
                )
                .await?
        }
    };

    Ok(Relationship {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: project,
        from_topic: from,
        to_topic: to,
        kind,
        created_at: row.get("created_at"),
    })
}

/// Loads the version of an edge currently believed, if any.
pub async fn live_version<C: GenericClient>(
    client: &C,
    relationship: RelationshipId,
) -> Result<Option<RelationshipVersion>> {
    let sql = format!(
        "SELECT {VERSION_COLUMNS} FROM relationship_versions
         WHERE relationship_id = $1 AND invalidated_at IS NULL
         ORDER BY version DESC LIMIT 1"
    );
    let row = client.query_opt(&sql, &[&relationship.0]).await?;
    Ok(row.as_ref().map(row_to_version))
}

/// Loads every version of an edge, oldest first.
pub async fn edge_history<C: GenericClient>(
    client: &C,
    relationship: RelationshipId,
) -> Result<Vec<RelationshipVersion>> {
    let sql = format!(
        "SELECT {VERSION_COLUMNS} FROM relationship_versions
         WHERE relationship_id = $1 ORDER BY version ASC"
    );
    let rows = client.query(&sql, &[&relationship.0]).await?;
    Ok(rows.iter().map(row_to_version).collect())
}

/// Asserts an edge, appending a version only if the claim is new.
///
/// Runs in a transaction that locks the edge identity first. Without the lock,
/// two writers can read the same maximum version and race to insert it; one
/// loses on the unique constraint and its claim is dropped rather than queued.
/// This is the same protocol topic states use, for the same reason.
pub async fn assert_edge(
    client: &mut Client,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Assertion> {
    if let Some(unchanged) = already_asserted(client, project, from, to, claim).await? {
        return Ok(Assertion::Unchanged(unchanged));
    }

    let transaction = client.transaction().await?;
    let asserted = assert_within(&transaction, project, from, to, claim).await?;
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
    client: &mut Client,
    project: ProjectId,
    edges: &[(TopicId, TopicId, EdgeClaim)],
) -> Result<Vec<Assertion>> {
    // Answered outside the transaction, and for the common case that is the
    // whole call: rewriting a memory re-derives the edges it already had, and
    // an unchanged claim writes nothing, so no lock is needed to decide it.
    let mut asserted: Vec<Option<Assertion>> = Vec::with_capacity(edges.len());
    let mut pending = Vec::new();
    for (index, (from, to, claim)) in edges.iter().enumerate() {
        let unchanged = already_asserted(client, project, *from, *to, claim).await?;
        if unchanged.is_none() {
            pending.push(index);
        }
        asserted.push(unchanged.map(Assertion::Unchanged));
    }

    if !pending.is_empty() {
        let transaction = client.transaction().await?;
        for index in pending {
            let (from, to, claim) = &edges[index];
            asserted[index] = Some(assert_within(&transaction, project, *from, *to, claim).await?);
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
async fn already_asserted<C: GenericClient>(
    client: &C,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Option<RelationshipVersion>> {
    let Some(relationship) = find_relationship(client, project, from, to, claim.kind).await? else {
        return Ok(None);
    };

    Ok(live_version(client, relationship.id)
        .await?
        .filter(|version| claim.matches(version)))
}

async fn assert_within<C: GenericClient>(
    client: &C,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    claim: &EdgeClaim,
) -> Result<Assertion> {
    let relationship = ensure_relationship(client, project, from, to, claim.kind).await?;

    client
        .execute(
            "SELECT id FROM relationships WHERE id = $1 FOR UPDATE",
            &[&relationship.id.0],
        )
        .await?;

    let live = live_version(client, relationship.id).await?;
    if let Some(existing) = live.as_ref().filter(|version| claim.matches(version)) {
        // Re-deriving the same edge from unchanged content must not stack
        // versions, or every rewrite of a memory would grow the ledger.
        return Ok(Assertion::Unchanged(existing.clone()));
    }

    let now = OffsetDateTime::now_utc();
    if let Some(previous) = live.as_ref() {
        client
            .execute(
                "UPDATE relationship_versions
                 SET invalidated_at = $2, tombstone_reason = $3
                 WHERE id = $1",
                &[&previous.id.0, &now, &TombstoneReason::Superseded.label()],
            )
            .await?;
    }

    let sql = format!(
        "INSERT INTO relationship_versions (
             id, project_id, relationship_id, version, valid_from, valid_to,
             created_at, supersedes, caused_by_topic_state, confidence, derivation
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8, $9, $10
         FROM relationship_versions WHERE relationship_id = $3
         RETURNING {VERSION_COLUMNS}"
    );
    let row = client
        .query_one(
            &sql,
            &[
                &RelationshipVersionId::new().0,
                &project.0,
                &relationship.id.0,
                &claim.validity.from,
                &claim.validity.to,
                &now,
                &live.as_ref().map(|version| version.id.0),
                &claim.caused_by_topic_state.map(|id| id.0),
                &claim.confidence,
                &claim.derivation.label(),
            ],
        )
        .await?;

    Ok(Assertion::Appended(row_to_version(&row)))
}

/// Closes the live version of an edge, leaving every row in place.
///
/// Returns whether anything was open to close. Closing is a retraction of the
/// claim, not a statement that the relationship ended at this instant: the
/// truth interval is untouched.
pub async fn close_edge(
    client: &Client,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
    reason: TombstoneReason,
) -> Result<bool> {
    let affected = client
        .execute(
            "UPDATE relationship_versions
             SET invalidated_at = $1, tombstone_reason = $2
             WHERE invalidated_at IS NULL
               AND relationship_id IN (
                   SELECT id FROM relationships
                   WHERE project_id = $3 AND from_topic = $4 AND to_topic = $5 AND kind = $6
               )",
            &[
                &OffsetDateTime::now_utc(),
                &reason.label(),
                &project.0,
                &from.0,
                &to.0,
                &kind.label(),
            ],
        )
        .await?;
    Ok(affected > 0)
}

/// Looks up an edge identity without creating one.
pub async fn find_relationship<C: GenericClient>(
    client: &C,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Option<Relationship>> {
    let row = client
        .query_opt(
            FIND_RELATIONSHIP,
            &[&project.0, &from.0, &to.0, &kind.label()],
        )
        .await?;

    Ok(row.map(|row| Relationship {
        id: row.get::<_, uuid::Uuid>("id").into(),
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
/// One round trip. A query per hop would multiply latency by the depth for a
/// traversal the database can do in a single recursive pass.
pub async fn expand(
    client: &Client,
    project: ProjectId,
    seeds: &[TopicId],
    options: &Expansion<'_>,
) -> Result<Vec<Neighbor>> {
    if seeds.is_empty() || options.depth == 0 {
        return Ok(Vec::new());
    }

    let seed_ids: Vec<uuid::Uuid> = seeds.iter().map(|topic| topic.0).collect();
    let kind_labels: Option<Vec<&str>> = options
        .kinds
        .map(|kinds| kinds.iter().map(|kind| kind.label()).collect());

    // The CTE walks edges in both directions, tracking the seed it started from
    // and the end it arrived through. `depth < $3` bounds the recursion;
    // without it a cycle never terminates. Ordering by hops then confidence
    // before DISTINCT ON keeps the shortest, most confident path to each topic.
    //
    // Which edges are visible depends on whether a moment was asked about.
    // Without `--at` the question is what we still stand behind, so only
    // uninvalidated versions count. With `--at` the question is what held then,
    // which a later retraction does not answer on its own: an edge closed
    // because the relationship ended still held before it was closed, while one
    // deleted because the claim was wrong never held at all, and a superseded
    // one is answered by its successor. That distinction is exactly what
    // tombstone_reason records, and ignoring it made every retraction erase its
    // own history.
    let rows = client
        .query(
            "WITH RECURSIVE visible_edges AS (
                 SELECT r.from_topic, r.to_topic, r.kind, v.confidence, v.derivation
                 FROM relationships r
                 JOIN relationship_versions v ON v.relationship_id = r.id
                 WHERE r.project_id = $1
                   AND ($4::TEXT[] IS NULL OR r.kind = ANY ($4))
                   AND CASE WHEN $5::TIMESTAMPTZ IS NULL
                       THEN v.invalidated_at IS NULL
                       ELSE (v.valid_from IS NULL OR v.valid_from <= $5)
                            AND (v.valid_to IS NULL OR $5 < v.valid_to)
                            AND (v.invalidated_at IS NULL
                                 OR (v.tombstone_reason = 'closed'
                                     AND $5 < v.invalidated_at))
                       END
             ),
             undirected AS (
                 SELECT from_topic AS source, to_topic AS target, kind, confidence, derivation
                 FROM visible_edges
                 UNION ALL
                 SELECT to_topic AS source, from_topic AS target, kind, confidence, derivation
                 FROM visible_edges
             ),
             walk AS (
                 SELECT e.source AS origin, e.target AS topic, 1 AS hops, e.source AS via,
                        e.kind, e.confidence, e.derivation
                 FROM undirected e
                 WHERE e.source = ANY ($2)
                 UNION ALL
                 SELECT w.origin, e.target, w.hops + 1, e.source,
                        e.kind, e.confidence, e.derivation
                 FROM walk w
                 JOIN undirected e ON e.source = w.topic
                 -- Never arrive back at the seed this walk started from.
                 -- Seeds come from the other channels, so a seed handed back
                 -- as its own graph result is one piece of evidence counted
                 -- twice under two names. Excluding the topic just left is
                 -- not enough: it happens to be the seed at two hops and
                 -- stops being it at three, where a ring walks straight back
                 -- to the start. Another seed reaching this one is still
                 -- allowed, which is the case that carries real evidence.
                 WHERE w.hops < $3
                   AND e.target <> w.origin
                   AND e.target <> w.via
             )
             SELECT DISTINCT ON (topic) topic, origin, hops, via, kind, confidence, derivation
             FROM walk
             ORDER BY topic, hops ASC, confidence DESC",
            &[
                &project.0,
                &seed_ids,
                &(options.depth as i32),
                &kind_labels,
                &options.at,
            ],
        )
        .await?;

    let mut neighbors: Vec<Neighbor> = rows
        .iter()
        .map(|row| Neighbor {
            topic: row.get::<_, uuid::Uuid>("topic").into(),
            origin: row.get::<_, uuid::Uuid>("origin").into(),
            hops: row.get::<_, i32>("hops") as u8,
            via: row.get::<_, uuid::Uuid>("via").into(),
            kind: EdgeKind::from_label(row.get("kind")).unwrap_or(EdgeKind::RelatedTo),
            derivation: Derivation::from_label(row.get("derivation"))
                .unwrap_or(Derivation::Imported),
            confidence: row.get("confidence"),
        })
        .collect();

    // DISTINCT ON orders by topic, so the ranking has to be reimposed. The
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
