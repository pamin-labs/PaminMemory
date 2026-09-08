//! Reads and writes against the authority store.
//!
//! Every write appends. Nothing here updates a row in place except a soft
//! delete, which sets `deleted_at` and leaves the content intact.

use pamin_core::{
    Derivation, EdgeKind, FilterDecision, Project, ProjectId, RetrievalSignals, SourceId,
    SourceKind, SourceSpan, SourceSpanId, SourceVersion, SourceVersionId, TombstoneReason, Topic,
    TopicId, TopicState, TopicStateId, Validity,
};
use time::OffsetDateTime;
use tokio_postgres::{Client, Row};

use crate::error::Result;
use crate::sql::{SqlLabel, sql_enum};

/// Version counters are `INTEGER`, which bounds a topic at two billion versions.
/// Converting through `i32` is therefore lossless in practice, and the cast is
/// kept in one place rather than scattered through each query.
fn to_sql_version(version: u32) -> i32 {
    version as i32
}

fn from_sql_version(version: i32) -> u32 {
    version as u32
}

sql_enum!(FilterDecision {
    Promoted => "promoted",
    Filtered => "filtered",
});

sql_enum!(SourceKind {
    Manual => "manual",
    File => "file",
    Directory => "directory",
    ChatLog => "chat_log",
});

sql_enum!(EdgeKind {
    Mentions => "mentions",
    Supports => "supports",
    Contradicts => "contradicts",
    Supersedes => "supersedes",
    RelatedTo => "related_to",
    PartOf => "part_of",
    DerivedFrom => "derived_from",
    SameAs => "same_as",
    DependsOn => "depends_on",
});

sql_enum!(Derivation {
    Explicit => "explicit",
    Deterministic => "deterministic",
    Model => "model",
    Imported => "imported",
});

sql_enum!(TombstoneReason {
    Closed => "closed",
    Superseded => "superseded",
    Deleted => "deleted",
});

/// Returns the project with this name, creating it if it does not exist.
pub async fn ensure_project(client: &Client, name: &str) -> Result<Project> {
    let row = client
        .query_one(
            "INSERT INTO projects (id, name, created_at)
             VALUES ($1, $2, $3)
             ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
             RETURNING id, name, created_at",
            &[&ProjectId::new().0, &name, &OffsetDateTime::now_utc()],
        )
        .await?;

    Ok(Project {
        id: row.get::<_, uuid::Uuid>("id").into(),
        name: row.get("name"),
        created_at: row.get("created_at"),
    })
}

/// Returns the source with this locator, creating it if it does not exist.
///
/// Re-ingesting the same locator appends a version to the existing source
/// rather than forking a second one, which is what keeps a file's history in a
/// single chain.
pub async fn ensure_source(
    client: &Client,
    project: ProjectId,
    kind: SourceKind,
    locator: &str,
) -> Result<SourceId> {
    let row = client
        .query_one(
            "INSERT INTO sources (id, project_id, kind, locator, created_at)
             VALUES ($1, $2, $3, $4, $5)
             ON CONFLICT (project_id, locator) DO UPDATE SET locator = EXCLUDED.locator
             RETURNING id",
            &[
                &SourceId::new().0,
                &project.0,
                &kind.label(),
                &locator,
                &OffsetDateTime::now_utc(),
            ],
        )
        .await?;

    Ok(row.get::<_, uuid::Uuid>("id").into())
}

/// Appends evidence, along with the filter's verdict on it.
///
/// The verdict rides on a row that exists either way: the filter decides
/// whether content reaches the retrieval surface, never whether it is kept.
///
/// The version number is read and written under a lock on the source row, for
/// the same reason `append_topic_state` locks the topic. Two agents writing to
/// one source otherwise both read the same maximum and both claim the version
/// after it, and only one of the two rows survives the uniqueness constraint.
/// Losing the other is losing evidence, which is the one thing this store
/// promises never to do.
pub async fn append_source_version(
    client: &mut Client,
    project: ProjectId,
    source: SourceId,
    content: &str,
    content_hash: &str,
    decision: FilterDecision,
    reason: &str,
) -> Result<SourceVersion> {
    let transaction = client.transaction().await?;

    transaction
        .execute(
            "SELECT id FROM sources WHERE id = $1 FOR UPDATE",
            &[&source.0],
        )
        .await?;

    let row = transaction
        .query_one(
            "INSERT INTO source_versions (
                 id, project_id, source_id, version, content, content_hash,
                 filter_decision, filter_reason, recorded_at
             )
             SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8
             FROM source_versions WHERE source_id = $3
             RETURNING id, version, recorded_at",
            &[
                &SourceVersionId::new().0,
                &project.0,
                &source.0,
                &content,
                &content_hash,
                &decision.label(),
                &reason,
                &OffsetDateTime::now_utc(),
            ],
        )
        .await?;

    transaction.commit().await?;

    Ok(SourceVersion {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: project,
        source_id: source,
        version: from_sql_version(row.get("version")),
        content: content.to_string(),
        content_hash: content_hash.to_string(),
        filter_decision: decision,
        filter_reason: reason.to_string(),
        recorded_at: row.get("recorded_at"),
    })
}

/// Records a byte range into a source version, with any language detected for it.
pub async fn append_source_span(
    client: &Client,
    project: ProjectId,
    source_version: SourceVersionId,
    byte_start: u32,
    byte_end: u32,
    detected_language: Option<&str>,
    language_confidence: Option<f32>,
) -> Result<SourceSpan> {
    let id = SourceSpanId::new();
    client
        .execute(
            "INSERT INTO source_spans (
                 id, project_id, source_version_id, byte_start, byte_end,
                 detected_language, language_confidence
             ) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            &[
                &id.0,
                &project.0,
                &source_version.0,
                &(byte_start as i32),
                &(byte_end as i32),
                &detected_language,
                &language_confidence,
            ],
        )
        .await?;

    Ok(SourceSpan {
        id,
        project_id: project,
        source_version_id: source_version,
        byte_start,
        byte_end,
        detected_language: detected_language.map(str::to_string),
        language_confidence,
    })
}

/// Returns the topic with this name, creating it if it does not exist.
pub async fn ensure_topic(client: &Client, project: ProjectId, name: &str) -> Result<Topic> {
    let row = client
        .query_one(
            "INSERT INTO topics (id, project_id, name, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (project_id, name) DO UPDATE SET name = EXCLUDED.name
             RETURNING id, name, path, created_at",
            &[
                &TopicId::new().0,
                &project.0,
                &name,
                &OffsetDateTime::now_utc(),
            ],
        )
        .await?;

    Ok(Topic {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: project,
        name: row.get("name"),
        path: row.get("path"),
        created_at: row.get("created_at"),
    })
}

/// Appends a new state to a topic.
///
/// Runs in a transaction that first locks the topic row. Without that lock, two
/// concurrent writers can both read the same maximum version and race to insert
/// it; one loses on the unique constraint, and the loser's content is dropped
/// rather than queued behind the winner. Locking the topic serializes appends
/// per topic while leaving different topics free to proceed in parallel.
pub async fn append_topic_state(
    client: &mut Client,
    project: ProjectId,
    topic: TopicId,
    content: &str,
    source_span: SourceSpanId,
    observed_at: OffsetDateTime,
    validity: Validity,
) -> Result<TopicState> {
    let transaction = client.transaction().await?;

    transaction
        .execute(
            "SELECT id FROM topics WHERE id = $1 FOR UPDATE",
            &[&topic.0],
        )
        .await?;

    let previous = transaction
        .query_opt(
            "SELECT id FROM topic_states
             WHERE topic_id = $1 AND deleted_at IS NULL
             ORDER BY version DESC LIMIT 1",
            &[&topic.0],
        )
        .await?
        .map(|row| TopicStateId::from(row.get::<_, uuid::Uuid>("id")));

    let row = transaction
        .query_one(
            "INSERT INTO topic_states (
                 id, project_id, topic_id, version, content, source_span_id,
                 observed_at, recorded_at, supersedes, valid_from, valid_to
             )
             SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8, $9, $10
             FROM topic_states WHERE topic_id = $3
             RETURNING id, version, recorded_at",
            &[
                &TopicStateId::new().0,
                &project.0,
                &topic.0,
                &content,
                &source_span.0,
                &observed_at,
                &OffsetDateTime::now_utc(),
                &previous.map(|id| id.0),
                &validity.from,
                &validity.to,
            ],
        )
        .await?;

    let state = TopicState {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: project,
        topic_id: topic,
        version: from_sql_version(row.get("version")),
        content: content.to_string(),
        source_span_id: source_span,
        observed_at,
        recorded_at: row.get("recorded_at"),
        validity,
        supersedes: previous,
        deleted_at: None,
        signals: RetrievalSignals::default(),
    };

    transaction.commit().await?;
    Ok(state)
}

fn row_to_topic_state(row: &Row) -> TopicState {
    TopicState {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: row.get::<_, uuid::Uuid>("project_id").into(),
        topic_id: row.get::<_, uuid::Uuid>("topic_id").into(),
        version: from_sql_version(row.get("version")),
        content: row.get("content"),
        source_span_id: row.get::<_, uuid::Uuid>("source_span_id").into(),
        observed_at: row.get("observed_at"),
        recorded_at: row.get("recorded_at"),
        validity: Validity::new(row.get("valid_from"), row.get("valid_to")),
        supersedes: row
            .get::<_, Option<uuid::Uuid>>("supersedes")
            .map(Into::into),
        deleted_at: row.get("deleted_at"),
        signals: RetrievalSignals {
            importance: row.get("importance"),
            worth_positive: row.get::<_, i32>("worth_positive") as u32,
            worth_negative: row.get::<_, i32>("worth_negative") as u32,
            access_count: row.get::<_, i32>("access_count") as u32,
            last_accessed_at: row.get("last_accessed_at"),
        },
    }
}

const STATE_COLUMNS: &str = "id, project_id, topic_id, version, content, source_span_id, \
     observed_at, recorded_at, valid_from, valid_to, supersedes, deleted_at, \
     importance, worth_positive, worth_negative, access_count, last_accessed_at";

/// Returns a topic's undeleted version numbers, oldest first.
///
/// This is what version resolution runs against, so soft-deleted versions are
/// excluded here rather than filtered afterwards.
pub async fn topic_versions(client: &Client, topic: TopicId) -> Result<Vec<u32>> {
    let rows = client
        .query(
            "SELECT version FROM topic_states
             WHERE topic_id = $1 AND deleted_at IS NULL
             ORDER BY version ASC",
            &[&topic.0],
        )
        .await?;

    Ok(rows
        .iter()
        .map(|row| from_sql_version(row.get("version")))
        .collect())
}

/// Loads one version of a topic.
pub async fn topic_state(
    client: &Client,
    topic: TopicId,
    version: u32,
) -> Result<Option<TopicState>> {
    let sql =
        format!("SELECT {STATE_COLUMNS} FROM topic_states WHERE topic_id = $1 AND version = $2");
    let row = client
        .query_opt(&sql, &[&topic.0, &to_sql_version(version)])
        .await?;
    Ok(row.as_ref().map(row_to_topic_state))
}

/// Loads every undeleted state in a project, oldest first.
///
/// Used by reindex, which rebuilds the projection from the authority store.
pub async fn all_live_topic_states(client: &Client, project: ProjectId) -> Result<Vec<TopicState>> {
    let sql = format!(
        "SELECT {STATE_COLUMNS} FROM topic_states
         WHERE project_id = $1 AND deleted_at IS NULL
         ORDER BY topic_id, version ASC"
    );
    let rows = client.query(&sql, &[&project.0]).await?;
    Ok(rows.iter().map(row_to_topic_state).collect())
}

/// Soft deletes one version of a topic.
///
/// The row and its content stay: deletion removes a state from the default
/// retrieval surface, not from the ledger. If the deleted state was current,
/// the previous surviving version becomes current, which falls out of computing
/// latest rather than needing a flag to be moved.
pub async fn soft_delete_topic_state(
    client: &Client,
    topic: TopicId,
    version: u32,
) -> Result<bool> {
    let affected = client
        .execute(
            "UPDATE topic_states SET deleted_at = $3
             WHERE topic_id = $1 AND version = $2 AND deleted_at IS NULL",
            &[
                &topic.0,
                &to_sql_version(version),
                &OffsetDateTime::now_utc(),
            ],
        )
        .await?;
    Ok(affected > 0)
}

/// Loads the newest evidence version for a source.
///
/// Reads back the filter verdict, which is how a caller confirms that filtered
/// content was still stored rather than discarded.
pub async fn latest_source_version(
    client: &Client,
    source: SourceId,
) -> Result<Option<SourceVersion>> {
    let row = client
        .query_opt(
            "SELECT id, project_id, source_id, version, content, content_hash,
                    filter_decision, filter_reason, recorded_at
             FROM source_versions WHERE source_id = $1
             ORDER BY version DESC LIMIT 1",
            &[&source.0],
        )
        .await?;

    Ok(row.map(|row| SourceVersion {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: row.get::<_, uuid::Uuid>("project_id").into(),
        source_id: row.get::<_, uuid::Uuid>("source_id").into(),
        version: from_sql_version(row.get("version")),
        content: row.get("content"),
        content_hash: row.get("content_hash"),
        filter_decision: FilterDecision::from_label(row.get("filter_decision"))
            // The column's CHECK constraint admits nothing else, so this
            // fallback stands only so a corrupted row degrades to the
            // conservative reading rather than aborting the command.
            .unwrap_or(FilterDecision::Promoted),
        filter_reason: row.get("filter_reason"),
        recorded_at: row.get("recorded_at"),
    }))
}

/// One piece of evidence matching a literal search, with where it came from.
#[derive(Clone, Debug)]
pub struct EvidenceMatch {
    pub source_version: SourceVersion,
    pub locator: String,
    /// Byte offset of the first occurrence, for rendering context around it.
    pub offset: usize,
}

/// Finds evidence containing `needle`, verbatim.
///
/// Searches `source_versions`, which is the authority: it holds every version
/// ever written, in the language it arrived in, **including content the sensory
/// filter held**. That content never enters the projection index, so this is
/// the only way to reach it — and reaching it is the point. A filter mistake
/// has to stay recoverable, and a recovery route nobody can take is not one.
///
/// `position` rather than a regular expression or a similarity operator.
/// PostgreSQL's regex operators are outside the portable subset, and a literal
/// match is what an exact-string question actually asks for. Nothing here
/// ranks: this is the primitive an agent reaches for when it does not want a
/// ranking model in the path.
pub async fn grep_evidence(
    client: &Client,
    project: ProjectId,
    needle: &str,
    case_sensitive: bool,
    limit: u32,
) -> Result<Vec<EvidenceMatch>> {
    // Folding case in SQL keeps the match and the offset consistent: computing
    // one here and the other in Rust would drift on any multi-byte casing rule.
    let matched = if case_sensitive {
        "position($2 IN v.content)"
    } else {
        "position(lower($2) IN lower(v.content))"
    };

    let sql = format!(
        "SELECT v.id, v.project_id, v.source_id, v.version, v.content, v.content_hash,
                v.filter_decision, v.filter_reason, v.recorded_at,
                s.locator, {matched} AS match_position
         FROM source_versions v
         JOIN sources s ON s.id = v.source_id
         WHERE v.project_id = $1 AND {matched} > 0
         ORDER BY v.recorded_at DESC, v.id
         LIMIT $3"
    );

    let rows = client
        .query(&sql, &[&project.0, &needle, &(limit as i64)])
        .await?;

    Ok(rows
        .iter()
        .map(|row| EvidenceMatch {
            source_version: SourceVersion {
                id: row.get::<_, uuid::Uuid>("id").into(),
                project_id: row.get::<_, uuid::Uuid>("project_id").into(),
                source_id: row.get::<_, uuid::Uuid>("source_id").into(),
                version: from_sql_version(row.get("version")),
                content: row.get("content"),
                content_hash: row.get("content_hash"),
                filter_decision: FilterDecision::from_label(row.get("filter_decision"))
                    .unwrap_or(FilterDecision::Promoted),
                filter_reason: row.get("filter_reason"),
                recorded_at: row.get("recorded_at"),
            },
            locator: row.get("locator"),
            // SQL positions are one-based; byte offsets are not.
            offset: (row.get::<_, i32>("match_position") as usize).saturating_sub(1),
        })
        .collect())
}

/// Lists every topic in a project.
///
/// Used to derive relationships from one topic's content naming another, which
/// needs the whole set rather than a candidate list: any topic may be named.
pub async fn all_topics(client: &Client, project: ProjectId) -> Result<Vec<Topic>> {
    let rows = client
        .query(
            "SELECT id, name, path, created_at FROM topics
             WHERE project_id = $1 ORDER BY name ASC",
            &[&project.0],
        )
        .await?;

    Ok(rows
        .iter()
        .map(|row| Topic {
            id: row.get::<_, uuid::Uuid>("id").into(),
            project_id: project,
            name: row.get("name"),
            path: row.get("path"),
            created_at: row.get("created_at"),
        })
        .collect())
}

/// Looks up a topic by name within a project.
pub async fn find_topic(client: &Client, project: ProjectId, name: &str) -> Result<Option<Topic>> {
    let row = client
        .query_opt(
            "SELECT id, name, path, created_at FROM topics
             WHERE project_id = $1 AND name = $2",
            &[&project.0, &name],
        )
        .await?;

    Ok(row.map(|row| Topic {
        id: row.get::<_, uuid::Uuid>("id").into(),
        project_id: project,
        name: row.get("name"),
        path: row.get("path"),
        created_at: row.get("created_at"),
    }))
}
