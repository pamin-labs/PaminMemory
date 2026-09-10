//! Reads and writes against the authority store.
//!
//! Every write appends. Nothing here updates a row in place except a soft
//! delete, which sets `deleted_at` and leaves the content intact.

use pamin_core::{
    Derivation, EdgeKind, FilterDecision, JobKind, Project, ProjectId, RetrievalSignals, SourceId,
    SourceKind, SourceSpan, SourceSpanId, SourceVersion, SourceVersionId, TombstoneReason, Topic,
    TopicId, TopicState, TopicStateId, Validity,
};
use sqlx::postgres::PgRow;
use sqlx::{PgExecutor, PgPool, Row};
use time::OffsetDateTime;

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

sql_enum!(JobKind {
    SyncTopicIndex => "sync_topic_index",
    UnindexState => "unindex_state",
    DeriveMentions => "derive_mentions",
    BackfillMentions => "backfill_mentions",
    OptimizeIndex => "optimize_index",
});

/// Returns the project with this name, creating it if it does not exist.
///
/// Reads before it writes. Every command opens with this call, and a project is
/// created once and then found forever after, so the write is the rare case.
/// Reaching it through `ON CONFLICT DO UPDATE` -- which is what a conflict
/// clause has to do to return the existing row -- made every command take a row
/// lock on the one row all of them share, and leave a dead tuple behind for
/// autovacuum. `DO NOTHING` returns nothing on conflict, so the losing side of
/// the race reads the winner's row instead.
pub async fn ensure_project(pool: &PgPool, name: &str) -> Result<Project> {
    const FIND: &str = "SELECT id, name, created_at FROM projects WHERE name = $1";

    let row = match sqlx::query(FIND).bind(name).fetch_optional(pool).await? {
        Some(row) => row,
        None => {
            let inserted = sqlx::query(
                "INSERT INTO projects (id, name, created_at)
                 VALUES ($1, $2, $3)
                 ON CONFLICT (name) DO NOTHING
                 RETURNING id, name, created_at",
            )
            .bind(ProjectId::new().0)
            .bind(name)
            .bind(OffsetDateTime::now_utc())
            .fetch_optional(pool)
            .await?;

            match inserted {
                Some(row) => row,
                // Another writer created it in between.
                None => sqlx::query(FIND).bind(name).fetch_one(pool).await?,
            }
        }
    };

    Ok(Project {
        id: row.get::<uuid::Uuid, _>("id").into(),
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
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    kind: SourceKind,
    locator: &str,
) -> Result<SourceId> {
    const FIND: &str = "SELECT id FROM sources WHERE project_id = $1 AND locator = $2";

    let found = sqlx::query(FIND)
        .bind(project.0)
        .bind(locator)
        .fetch_optional(&mut *connection)
        .await?;
    if let Some(row) = found {
        return Ok(row.get::<uuid::Uuid, _>("id").into());
    }

    let inserted = sqlx::query(
        "INSERT INTO sources (id, project_id, kind, locator, created_at)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (project_id, locator) DO NOTHING
         RETURNING id",
    )
    .bind(SourceId::new().0)
    .bind(project.0)
    .bind(kind.label())
    .bind(locator)
    .bind(OffsetDateTime::now_utc())
    .fetch_optional(&mut *connection)
    .await?;

    let row = match inserted {
        Some(row) => row,
        None => {
            sqlx::query(FIND)
                .bind(project.0)
                .bind(locator)
                .fetch_one(&mut *connection)
                .await?
        }
    };

    Ok(row.get::<uuid::Uuid, _>("id").into())
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
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    source: SourceId,
    content: &str,
    content_hash: &str,
    decision: FilterDecision,
    reason: &str,
) -> Result<SourceVersion> {
    sqlx::query("SELECT id FROM sources WHERE id = $1 FOR UPDATE")
        .bind(source.0)
        .execute(&mut *connection)
        .await?;

    let row = sqlx::query(
        "INSERT INTO source_versions (
             id, project_id, source_id, version, content, content_hash,
             filter_decision, filter_reason, recorded_at
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8
         FROM source_versions WHERE source_id = $3
         RETURNING id, version, recorded_at",
    )
    .bind(SourceVersionId::new().0)
    .bind(project.0)
    .bind(source.0)
    .bind(content)
    .bind(content_hash)
    .bind(decision.label())
    .bind(reason)
    .bind(OffsetDateTime::now_utc())
    .fetch_one(connection)
    .await?;

    Ok(SourceVersion {
        id: row.get::<uuid::Uuid, _>("id").into(),
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
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    source_version: SourceVersionId,
    byte_start: u32,
    byte_end: u32,
    detected_language: Option<&str>,
    language_confidence: Option<f32>,
) -> Result<SourceSpan> {
    let id = SourceSpanId::new();
    sqlx::query(
        "INSERT INTO source_spans (
             id, project_id, source_version_id, byte_start, byte_end,
             detected_language, language_confidence
         ) VALUES ($1, $2, $3, $4, $5, $6, $7)",
    )
    .bind(id.0)
    .bind(project.0)
    .bind(source_version.0)
    .bind(byte_start as i32)
    .bind(byte_end as i32)
    .bind(detected_language)
    .bind(language_confidence)
    .execute(executor)
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
///
/// Finds before it inserts, for the reason given on `ensure_project`: a topic
/// is created once and named on every write afterwards, and rewriting the row
/// to read it back is a lock and a dead tuple bought for nothing.
pub async fn ensure_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
) -> Result<Topic> {
    if let Some(topic) = find_topic(&mut *connection, project, name).await? {
        return Ok(topic);
    }

    let inserted = sqlx::query(
        "INSERT INTO topics (id, project_id, name, created_at)
         VALUES ($1, $2, $3, $4)
         ON CONFLICT (project_id, name) DO NOTHING
         RETURNING id, name, path, created_at",
    )
    .bind(TopicId::new().0)
    .bind(project.0)
    .bind(name)
    .bind(OffsetDateTime::now_utc())
    .fetch_optional(&mut *connection)
    .await?;

    let row = match inserted {
        Some(row) => row,
        // Another writer created it in between.
        None => {
            sqlx::query(FIND_TOPIC)
                .bind(project.0)
                .bind(name)
                .fetch_one(&mut *connection)
                .await?
        }
    };

    Ok(Topic {
        id: row.get::<uuid::Uuid, _>("id").into(),
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
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    topic: TopicId,
    content: &str,
    source_span: SourceSpanId,
    observed_at: OffsetDateTime,
    validity: Validity,
) -> Result<TopicState> {
    sqlx::query("SELECT id FROM topics WHERE id = $1 FOR UPDATE")
        .bind(topic.0)
        .execute(&mut *connection)
        .await?;

    let previous = sqlx::query(
        "SELECT id FROM topic_states
         WHERE topic_id = $1 AND deleted_at IS NULL
         ORDER BY version DESC LIMIT 1",
    )
    .bind(topic.0)
    .fetch_optional(&mut *connection)
    .await?
    .map(|row| TopicStateId::from(row.get::<uuid::Uuid, _>("id")));

    let row = sqlx::query(
        "INSERT INTO topic_states (
             id, project_id, topic_id, version, content, source_span_id,
             observed_at, recorded_at, supersedes, valid_from, valid_to
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8, $9, $10
         FROM topic_states WHERE topic_id = $3
         RETURNING id, version, recorded_at",
    )
    .bind(TopicStateId::new().0)
    .bind(project.0)
    .bind(topic.0)
    .bind(content)
    .bind(source_span.0)
    .bind(observed_at)
    .bind(OffsetDateTime::now_utc())
    .bind(previous.map(|id| id.0))
    .bind(validity.from)
    .bind(validity.to)
    .fetch_one(&mut *connection)
    .await?;

    let state = TopicState {
        id: row.get::<uuid::Uuid, _>("id").into(),
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

    // The appended state is the newest surviving one by construction, so the
    // pointer moves here rather than being recomputed. Same transaction, same
    // lock: there is no window where the topic points at the state before this
    // one.
    point_at(connection, topic, Some(&state)).await?;

    Ok(state)
}

/// Sets the topic's current state, or clears it when nothing survives.
///
/// Both callers already hold the topic row locked, which is what keeps this a
/// write rather than a read-modify-write.
async fn point_at(
    connection: &mut sqlx::PgConnection,
    topic: TopicId,
    state: Option<&TopicState>,
) -> Result<()> {
    sqlx::query("UPDATE topics SET current_state_id = $2, current_version = $3 WHERE id = $1")
        .bind(topic.0)
        .bind(state.map(|state| state.id.0))
        .bind(state.map(|state| to_sql_version(state.version)))
        .execute(connection)
        .await?;

    Ok(())
}

fn row_to_topic_state(row: &PgRow) -> TopicState {
    TopicState {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: row.get::<uuid::Uuid, _>("project_id").into(),
        topic_id: row.get::<uuid::Uuid, _>("topic_id").into(),
        version: from_sql_version(row.get("version")),
        content: row.get("content"),
        source_span_id: row.get::<uuid::Uuid, _>("source_span_id").into(),
        observed_at: row.get("observed_at"),
        recorded_at: row.get("recorded_at"),
        validity: Validity::new(row.get("valid_from"), row.get("valid_to")),
        supersedes: row
            .get::<Option<uuid::Uuid>, _>("supersedes")
            .map(Into::into),
        deleted_at: row.get("deleted_at"),
        signals: RetrievalSignals {
            importance: row.get("importance"),
            worth_positive: row.get::<i32, _>("worth_positive") as u32,
            worth_negative: row.get::<i32, _>("worth_negative") as u32,
            access_count: row.get::<i32, _>("access_count") as u32,
            last_accessed_at: row.get("last_accessed_at"),
        },
    }
}

/// The columns `row_to_topic_state` reads.
///
/// A macro rather than a constant so the statements below can be assembled with
/// `concat!` and stay `&'static str`. sqlx accepts only a statement that is
/// static or explicitly asserted safe, which is a deliberate obstacle in front
/// of building SQL with `format!`; this keeps the column list in one place
/// without stepping over it.
///
/// Takes the table's alias, because a statement that joins `topics` has two
/// `id` and two `project_id` columns in scope and an unqualified list is
/// ambiguous there -- an error PostgreSQL raises at execution, so only a query
/// that actually runs finds it.
macro_rules! state_columns {
    () => {
        state_columns!("")
    };
    ($alias:literal) => {
        concat!(
            $alias,
            "id, ",
            $alias,
            "project_id, ",
            $alias,
            "topic_id, ",
            $alias,
            "version, ",
            $alias,
            "content, ",
            $alias,
            "source_span_id, ",
            $alias,
            "observed_at, ",
            $alias,
            "recorded_at, ",
            $alias,
            "valid_from, ",
            $alias,
            "valid_to, ",
            $alias,
            "supersedes, ",
            $alias,
            "deleted_at, ",
            $alias,
            "importance, ",
            $alias,
            "worth_positive, ",
            $alias,
            "worth_negative, ",
            $alias,
            "access_count, ",
            $alias,
            "last_accessed_at"
        )
    };
}

/// Returns a topic's undeleted version numbers, oldest first.
///
/// This is what version resolution runs against, so soft-deleted versions are
/// excluded here rather than filtered afterwards.
pub async fn topic_versions(executor: impl PgExecutor<'_>, topic: TopicId) -> Result<Vec<u32>> {
    let versions: Vec<(i32,)> = sqlx::query_as(
        "SELECT version FROM topic_states
         WHERE topic_id = $1 AND deleted_at IS NULL
         ORDER BY version ASC",
    )
    .bind(topic.0)
    .fetch_all(executor)
    .await?;

    Ok(versions
        .into_iter()
        .map(|(version,)| from_sql_version(version))
        .collect())
}

/// Loads one version of a topic.
pub async fn topic_state(
    executor: impl PgExecutor<'_>,
    topic: TopicId,
    version: u32,
) -> Result<Option<TopicState>> {
    let row = sqlx::query(concat!(
        "SELECT ",
        state_columns!(),
        " FROM topic_states WHERE topic_id = $1 AND version = $2"
    ))
    .bind(topic.0)
    .bind(to_sql_version(version))
    .fetch_optional(executor)
    .await?;

    Ok(row.as_ref().map(row_to_topic_state))
}

/// Loads every undeleted state in a project, oldest first.
///
/// Used by reindex, which rebuilds the projection from the authority store.
pub async fn all_live_topic_states(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<TopicState>> {
    let rows = sqlx::query(concat!(
        "SELECT ",
        state_columns!(),
        " FROM topic_states
          WHERE project_id = $1 AND deleted_at IS NULL
          ORDER BY topic_id, version ASC"
    ))
    .bind(project.0)
    .fetch_all(executor)
    .await?;

    Ok(rows.iter().map(row_to_topic_state).collect())
}

/// Loads these states, skipping any the ledger has soft deleted.
///
/// The search path's replacement for reading the project. What it needs is the
/// states the recall channels actually returned, which is a few hundred rows
/// whatever the project holds; loading every live state to answer that was the
/// single largest thing a query did.
///
/// Soft-deleted states are dropped here rather than after ranking. Filtering
/// afterwards meant a state the ledger had removed still occupied a place in
/// each channel's candidate budget, so deleting content quietly reduced how
/// much a search could find.
pub async fn topic_states_by_id(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    states: &[TopicStateId],
) -> Result<Vec<TopicState>> {
    if states.is_empty() {
        return Ok(Vec::new());
    }

    let ids: Vec<uuid::Uuid> = states.iter().map(|state| state.0).collect();
    let rows = sqlx::query(concat!(
        "SELECT ",
        state_columns!(),
        " FROM topic_states
          WHERE project_id = $1 AND id = ANY($2) AND deleted_at IS NULL"
    ))
    .bind(project.0)
    .bind(&ids)
    .fetch_all(executor)
    .await?;

    Ok(rows.iter().map(row_to_topic_state).collect())
}

/// Loads the states these topics currently resolve to.
///
/// Through the pointer on `topics`, so this is a primary key lookup per topic
/// rather than a search for each one's newest surviving version. The graph
/// channel walks topic identities and can only return states, so every
/// neighbour it finds comes through here.
///
/// A topic whose every state has been soft deleted resolves to nothing and is
/// simply absent from the result.
pub async fn current_states_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topics: &[TopicId],
) -> Result<Vec<TopicState>> {
    if topics.is_empty() {
        return Ok(Vec::new());
    }

    let ids: Vec<uuid::Uuid> = topics.iter().map(|topic| topic.0).collect();
    let rows = sqlx::query(concat!(
        "SELECT ",
        state_columns!("ts."),
        " FROM topic_states ts
          JOIN topics t ON t.current_state_id = ts.id
          WHERE t.project_id = $1 AND t.id = ANY($2)"
    ))
    .bind(project.0)
    .bind(&ids)
    .fetch_all(executor)
    .await?;

    Ok(rows.iter().map(row_to_topic_state).collect())
}

/// Names these topics, and says which state each currently resolves to.
///
/// One lookup for the two things the search path needs about a topic once it
/// has a result from it: what to call it, and whether the state in hand is the
/// one the topic stands for now.
pub async fn topics_by_id(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topics: &[TopicId],
) -> Result<Vec<(TopicId, String, Option<TopicStateId>)>> {
    if topics.is_empty() {
        return Ok(Vec::new());
    }

    let ids: Vec<uuid::Uuid> = topics.iter().map(|topic| topic.0).collect();
    let rows = sqlx::query(
        "SELECT id, name, current_state_id FROM topics
         WHERE project_id = $1 AND id = ANY($2)",
    )
    .bind(project.0)
    .bind(&ids)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .map(|row| {
            (
                row.get::<uuid::Uuid, _>("id").into(),
                row.get("name"),
                row.get::<Option<uuid::Uuid>, _>("current_state_id")
                    .map(Into::into),
            )
        })
        .collect())
}

/// Soft deletes one version of a topic.
///
/// The row and its content stay: deletion removes a state from the default
/// retrieval surface, not from the ledger. If the deleted state was current,
/// the previous surviving version becomes current -- and since that is now a
/// stored pointer rather than a computed one, moving it is part of the same
/// transaction, under the same lock the append path takes.
pub async fn soft_delete_topic_state(
    connection: &mut sqlx::PgConnection,
    topic: TopicId,
    version: u32,
) -> Result<bool> {
    sqlx::query("SELECT id FROM topics WHERE id = $1 FOR UPDATE")
        .bind(topic.0)
        .execute(&mut *connection)
        .await?;

    let affected = sqlx::query(
        "UPDATE topic_states SET deleted_at = $3
         WHERE topic_id = $1 AND version = $2 AND deleted_at IS NULL",
    )
    .bind(topic.0)
    .bind(to_sql_version(version))
    .bind(OffsetDateTime::now_utc())
    .execute(&mut *connection)
    .await?;

    if affected.rows_affected() == 0 {
        return Ok(false);
    }

    // Recomputed rather than stepped back to the predecessor: the deleted state
    // need not have been the current one, and the version before it need not
    // have survived either. This asks the ledger the same question the pointer
    // is an answer to.
    let surviving = sqlx::query(concat!(
        "SELECT ",
        state_columns!(),
        " FROM topic_states
          WHERE topic_id = $1 AND deleted_at IS NULL
          ORDER BY version DESC LIMIT 1"
    ))
    .bind(topic.0)
    .fetch_optional(&mut *connection)
    .await?
    .as_ref()
    .map(row_to_topic_state);

    point_at(connection, topic, surviving.as_ref()).await?;

    Ok(true)
}

/// Recomputes every topic's current-state pointer from the ledger.
///
/// Returns how many topics were pointing somewhere else. Nothing should be, and
/// the two writers that move the pointer both do it under the topic's lock in
/// the transaction that changed the ledger. But a stored pointer is derived
/// data, and derived data needs a route back: without one, a third write path
/// that forgets shows up as a search returning content the topic no longer has,
/// with nothing to say so.
///
/// This is that route, and `reindex` is where it belongs -- the command whose
/// whole promise is that everything outside the ledger can be rebuilt from it.
pub async fn repair_current_state_pointers(pool: &PgPool, project: ProjectId) -> Result<u64> {
    let repaired = sqlx::query(
        "UPDATE topics t
            SET current_state_id = latest.id,
                current_version  = latest.version
           FROM (
               SELECT topics.id AS topic_id, live.id, live.version
               FROM topics
               LEFT JOIN LATERAL (
                   SELECT id, version FROM topic_states
                   WHERE topic_id = topics.id AND deleted_at IS NULL
                   ORDER BY version DESC LIMIT 1
               ) AS live ON TRUE
               WHERE topics.project_id = $1
           ) AS latest
          WHERE latest.topic_id = t.id
            AND (t.current_state_id, t.current_version)
                IS DISTINCT FROM (latest.id, latest.version)",
    )
    .bind(project.0)
    .execute(pool)
    .await?;

    Ok(repaired.rows_affected())
}

/// Loads the newest evidence version for a source.
///
/// Reads back the filter verdict, which is how a caller confirms that filtered
/// content was still stored rather than discarded.
pub async fn latest_source_version(
    executor: impl PgExecutor<'_>,
    source: SourceId,
) -> Result<Option<SourceVersion>> {
    let row = sqlx::query(
        "SELECT id, project_id, source_id, version, content, content_hash,
                filter_decision, filter_reason, recorded_at
         FROM source_versions WHERE source_id = $1
         ORDER BY version DESC LIMIT 1",
    )
    .bind(source.0)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|row| SourceVersion {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: row.get::<uuid::Uuid, _>("project_id").into(),
        source_id: row.get::<uuid::Uuid, _>("source_id").into(),
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
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    needle: &str,
    case_sensitive: bool,
    limit: u32,
) -> Result<Vec<EvidenceMatch>> {
    // Folding case in SQL keeps the match and the offset consistent: computing
    // one here and the other in Rust would drift on any multi-byte casing rule.
    //
    // Two whole statements rather than one assembled around the comparison. The
    // driver takes a statement that is `'static` or explicitly asserted safe,
    // which is a deliberate obstacle in front of building SQL with `format!`,
    // and the way past it that keeps the guarantee is to write both out.
    // The match is computed once, in a lateral, and both selected and filtered
    // from there. Written twice it read as one expression and was two, on the
    // column holding every byte ever written.
    macro_rules! grep {
        ($matched:literal) => {
            concat!(
                "SELECT v.id, v.project_id, v.source_id, v.version, v.content, v.content_hash,
                        v.filter_decision, v.filter_reason, v.recorded_at,
                        s.locator, m.match_position
                 FROM source_versions v
                 JOIN sources s ON s.id = v.source_id
                 CROSS JOIN LATERAL (SELECT ",
                $matched,
                " AS match_position) m
                 WHERE v.project_id = $1 AND m.match_position > 0
                 ORDER BY v.recorded_at DESC, v.id
                 LIMIT $3"
            )
        };
    }

    let sql = if case_sensitive {
        grep!("position($2 IN v.content)")
    } else {
        grep!("position(lower($2) IN lower(v.content))")
    };

    let rows = sqlx::query(sql)
        .bind(project.0)
        .bind(needle)
        .bind(i64::from(limit))
        .fetch_all(executor)
        .await?;

    Ok(rows
        .iter()
        .map(|row| EvidenceMatch {
            source_version: SourceVersion {
                id: row.get::<uuid::Uuid, _>("id").into(),
                project_id: row.get::<uuid::Uuid, _>("project_id").into(),
                source_id: row.get::<uuid::Uuid, _>("source_id").into(),
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
            offset: (row.get::<i32, _>("match_position") as usize).saturating_sub(1),
        })
        .collect())
}

/// Lists every topic in a project.
///
/// Used to derive relationships from one topic's content naming another, which
/// needs the whole set rather than a candidate list: any topic may be named.
pub async fn all_topics(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<Vec<Topic>> {
    let rows = sqlx::query(
        "SELECT id, name, path, created_at FROM topics
         WHERE project_id = $1 ORDER BY name ASC",
    )
    .bind(project.0)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .map(|row| Topic {
            id: row.get::<uuid::Uuid, _>("id").into(),
            project_id: project,
            name: row.get("name"),
            path: row.get("path"),
            created_at: row.get("created_at"),
        })
        .collect())
}

const FIND_TOPIC: &str = "SELECT id, name, path, created_at FROM topics
                          WHERE project_id = $1 AND name = $2";

/// Looks up a topic by name within a project.
pub async fn find_topic(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    name: &str,
) -> Result<Option<Topic>> {
    let row = sqlx::query(FIND_TOPIC)
        .bind(project.0)
        .bind(name)
        .fetch_optional(executor)
        .await?;

    Ok(row.map(|row| Topic {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: project,
        name: row.get("name"),
        path: row.get("path"),
        created_at: row.get("created_at"),
    }))
}
