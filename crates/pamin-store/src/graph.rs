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
    RelationshipVersionId, TombstoneReason, TopicId, TopicStateId,
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
    pub valid_from: Option<OffsetDateTime>,
    pub valid_to: Option<OffsetDateTime>,
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
            valid_from: None,
            valid_to: None,
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
            valid_from: None,
            valid_to: None,
            caused_by_topic_state: Some(caused_by),
        }
    }

    /// Whether an existing version already says exactly this.
    fn matches(&self, version: &RelationshipVersion) -> bool {
        version.derivation == self.derivation
            && version.confidence == self.confidence
            && version.valid_from == self.valid_from
            && version.valid_to == self.valid_to
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
        valid_from: row.get("valid_from"),
        valid_to: row.get("valid_to"),
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
    let row = client
        .query_one(
            "INSERT INTO relationships (id, project_id, from_topic, to_topic, kind, created_at)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (project_id, from_topic, to_topic, kind)
                 DO UPDATE SET kind = EXCLUDED.kind
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
    let transaction = client.transaction().await?;
    let relationship = ensure_relationship(&transaction, project, from, to, claim.kind).await?;

    transaction
        .execute(
            "SELECT id FROM relationships WHERE id = $1 FOR UPDATE",
            &[&relationship.id.0],
        )
        .await?;

    let live = live_version(&transaction, relationship.id).await?;
    if let Some(existing) = live.as_ref().filter(|version| claim.matches(version)) {
        // Re-deriving the same edge from unchanged content must not stack
        // versions, or every rewrite of a memory would grow the ledger.
        let unchanged = existing.clone();
        transaction.commit().await?;
        return Ok(Assertion::Unchanged(unchanged));
    }

    let now = OffsetDateTime::now_utc();
    if let Some(previous) = live.as_ref() {
        transaction
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
    let row = transaction
        .query_one(
            &sql,
            &[
                &RelationshipVersionId::new().0,
                &project.0,
                &relationship.id.0,
                &claim.valid_from,
                &claim.valid_to,
                &now,
                &live.as_ref().map(|version| version.id.0),
                &claim.caused_by_topic_state.map(|id| id.0),
                &claim.confidence,
                &claim.derivation.label(),
            ],
        )
        .await?;

    let appended = row_to_version(&row);
    transaction.commit().await?;
    Ok(Assertion::Appended(appended))
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
pub async fn find_relationship(
    client: &Client,
    project: ProjectId,
    from: TopicId,
    to: TopicId,
    kind: EdgeKind,
) -> Result<Option<Relationship>> {
    let row = client
        .query_opt(
            "SELECT id, created_at FROM relationships
             WHERE project_id = $1 AND from_topic = $2 AND to_topic = $3 AND kind = $4",
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
