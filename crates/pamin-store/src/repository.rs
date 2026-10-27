//! Reads and writes against the authority store.
//!
//! Every write appends. Nothing here updates a row in place except a soft
//! delete, which sets `deleted_at` and leaves the content intact.

use pamin_core::{
    Derivation, EdgeKind, FilterDecision, JobKind, Project, ProjectId, SourceId, SourceKind,
    SourceSpan, SourceSpanId, SourceVersion, SourceVersionId, TombstoneReason, Topic, TopicId,
    TopicState, TopicStateId, Validity,
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

/// Returns the source with this locator, creating it if it does not exist,
/// and holds it locked until the transaction ends.
///
/// Re-ingesting the same locator appends a version to the existing source
/// rather than forking a second one, which is what keeps a file's history in a
/// single chain.
///
/// The lock is what [`append_evidence`] and [`append_promoted`] number under, taken here
/// because this is the statement that finds the row: locking it again by id
/// was a second round trip for the same row. A row this call inserts is
/// already held by the inserting transaction, so both paths leave it locked.
/// Outside a transaction the lock lasts one statement and means nothing.
///
/// The lookup and the insert are one statement. The insert runs only when the
/// lookup found nothing, which is a question the statement can ask itself, so
/// a new source -- every first write to a topic -- no longer pays a round trip
/// to learn that it has to ask a second question.
pub async fn ensure_source(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    kind: SourceKind,
    locator: &str,
) -> Result<SourceId> {
    Ok(ensure_source_created(connection, project, kind, locator)
        .await?
        .0)
}

/// [`ensure_source`], saying whether this call created the source -- which is
/// the first time anything was filed under `locator`.
pub async fn ensure_source_created(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    kind: SourceKind,
    locator: &str,
) -> Result<(SourceId, bool)> {
    const FIND: &str = "SELECT id, false AS created FROM sources WHERE project_id = $1 AND locator = $2 FOR UPDATE";

    let row = sqlx::query(
        "WITH found AS (
             SELECT id FROM sources WHERE project_id = $1 AND locator = $2 FOR UPDATE
         ), inserted AS (
             INSERT INTO sources (id, project_id, kind, locator, created_at)
             SELECT $3, $1, $4, $2, $5
              WHERE NOT EXISTS (SELECT 1 FROM found)
             ON CONFLICT (project_id, locator) DO NOTHING
             RETURNING id
         )
         SELECT id, false AS created FROM found
         UNION ALL
         SELECT id, true AS created FROM inserted",
    )
    .bind(project.0)
    .bind(locator)
    .bind(SourceId::new().0)
    .bind(kind.label())
    .bind(OffsetDateTime::now_utc())
    .fetch_optional(&mut *connection)
    .await?;

    let row = match row {
        Some(row) => row,
        // Another writer created it after this statement's snapshot was taken:
        // the insert waited for that writer and then did nothing, and the
        // lookup had already looked. A statement of its own sees the row.
        None => {
            sqlx::query(FIND)
                .bind(project.0)
                .bind(locator)
                .fetch_one(&mut *connection)
                .await?
        }
    };

    Ok((row.get::<uuid::Uuid, _>("id").into(), row.get("created")))
}

/// Evidence as the write path records it: the content, the filter's verdict
/// on it, and the language detected over it.
pub struct Evidence<'a> {
    pub content: &'a str,
    pub content_hash: &'a str,
    pub decision: FilterDecision,
    pub reason: &'a str,
    pub language: Option<&'a str>,
    pub language_confidence: Option<f32>,
}

/// Appends evidence and the span that covers all of it, in one statement,
/// along with the filter's verdict on it.
///
/// What a write the filter holds records. The verdict rides on a row that
/// exists either way: the filter decides whether content reaches the retrieval
/// surface, never whether it is kept.
///
/// **Call it after [`ensure_source`], in the same transaction.** The version
/// number is read and written under the lock that call takes on the source
/// row. Two agents writing to one source otherwise both read the same maximum
/// and both claim the version after it, and only one of the two rows survives
/// the uniqueness constraint. Losing the other is losing evidence, which is
/// the one thing this store promises never to do. The lock has to be taken in
/// a statement before this one: a statement reads with the snapshot it began
/// with, so one that waited for the lock inside itself would still number from
/// what it saw before the wait.
pub async fn append_evidence(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    source: SourceId,
    evidence: &Evidence<'_>,
) -> Result<(SourceVersion, SourceSpan)> {
    let (version, span, _) = insert_evidence(connection, project, source, evidence, None).await?;
    Ok((version, span))
}

/// What a promoted write records beyond its evidence: the topic it is
/// promoted into, when and for how long it holds, and the work it owes.
pub struct Promotion<'a> {
    /// Locked by an earlier statement of the same transaction, which is what
    /// holding one says.
    pub topic: &'a LockedTopic,
    pub observed_at: OffsetDateTime,
    pub validity: Validity,
    /// The cascade's work, queued the way [`crate::jobs::enqueue_all`] queues
    /// it, against the topic.
    pub owed: &'a [JobKind],
}

/// Appends evidence, the span over all of it, the topic's state cut from that
/// span, the topic's pointer to it, and the work the cascade owes it -- all in
/// one statement.
///
/// Everything a promoted write records once its locks are held. They were
/// four statements, and each needed nothing from the one before it that the
/// caller could not choose first: the version's id, the span's id, the state's
/// id are all picked here, and the predecessor the state supersedes was read
/// by the statement that locked the topic. A data-modifying `WITH` runs
/// whether or not the outer query reads it, and every foreign key between
/// these rows is checked at the end of the statement, by which point all of
/// them exist.
///
/// **Call it after [`ensure_source`] and after [`lock_topic`] or
/// [`create_topic_named`], in the same transaction.** Both numbers -- the
/// evidence's version and the state's -- are read and written under those
/// locks, and a lock only protects a number read by a statement that began
/// after it was granted.
pub async fn append_promoted(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    source: SourceId,
    evidence: &Evidence<'_>,
    promotion: &Promotion<'_>,
) -> Result<(SourceVersion, SourceSpan, TopicState)> {
    let (version, span, state) =
        insert_evidence(connection, project, source, evidence, Some(promotion)).await?;
    Ok((version, span, state.expect("asked for the state")))
}

/// Numbers and inserts a source version, the query every append of evidence
/// starts from.
macro_rules! insert_source_version {
    () => {
        "INSERT INTO source_versions (
             id, project_id, source_id, version, content, content_hash,
             filter_decision, filter_reason, recorded_at
         )
         SELECT $1, $2, $3, COALESCE(MAX(version), 0) + 1, $4, $5, $6, $7, $8
         FROM source_versions WHERE project_id = $2 AND source_id = $3
         RETURNING id, version, recorded_at"
    };
}

/// The span over the whole of the version the `version` query inserted.
macro_rules! insert_whole_span {
    () => {
        "INSERT INTO source_spans (
             id, project_id, source_version_id, byte_start, byte_end,
             detected_language, language_confidence
         )
         SELECT $9, $2, version.id, 0, $10, $11, $12 FROM version"
    };
}

/// A topic's next state and the topic's pointer to it, as the two `WITH`
/// queries `state` and `pointer` of the statement that appends one.
///
/// The state and the pointer to it in one statement: the appended state is
/// the newest surviving one by construction, so the pointer moves with it
/// rather than being recomputed, and there is no moment at which the topic
/// points at the state before this one. The arguments name the placeholders
/// each value is bound to.
#[rustfmt::skip] // One argument per line would bury the statement.
macro_rules! append_state {
    (
        id = $id:literal,
        project = $project:literal,
        topic = $topic:literal,
        span = $span:literal,
        observed = $observed:literal,
        recorded = $recorded:literal,
        supersedes = $supersedes:literal,
        valid_from = $from:literal,
        valid_to = $to:literal $(,)?
    ) => {
        concat!(
            "state AS (
                 INSERT INTO topic_states (
                     id, project_id, topic_id, version, source_span_id,
                     observed_at, recorded_at, supersedes, valid_from, valid_to
                 )
                 SELECT ", $id, ", ", $project, ", ", $topic, ", COALESCE(MAX(version), 0) + 1,
                        ", $span, ", ", $observed, ", ", $recorded, ", ", $supersedes, ",
                        ", $from, ", ", $to, "
                   FROM topic_states
                  WHERE project_id = ", $project, " AND topic_id = ", $topic, "
                 RETURNING id, version, recorded_at
             ), pointer AS (
                 UPDATE topics SET current_state_id = state.id, current_version = state.version
                   FROM state WHERE topics.id = ", $topic, "
             )"
        )
    };
}

/// The version, the span over all of it, and with a `promotion` the state cut
/// from that span, written by one statement.
async fn insert_evidence(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    source: SourceId,
    evidence: &Evidence<'_>,
    promotion: Option<&Promotion<'_>>,
) -> Result<(SourceVersion, SourceSpan, Option<TopicState>)> {
    const WITH_SPAN: &str = concat!(
        "WITH version AS (",
        insert_source_version!(),
        "), span AS (",
        insert_whole_span!(),
        ")
         SELECT id, version, recorded_at FROM version"
    );
    const PROMOTED: &str = concat!(
        "WITH version AS (",
        insert_source_version!(),
        "), span AS (",
        insert_whole_span!(),
        "), ",
        append_state!(
            id = "$13",
            project = "$2",
            topic = "$14",
            span = "$9",
            observed = "$15",
            recorded = "$8",
            supersedes = "$16",
            valid_from = "$17",
            valid_to = "$18",
        ),
        ", owed AS (",
        crate::jobs::enqueue_jobs!(
            project = "$2",
            subject = "$14",
            at = "$8",
            rows = ["$19", "$20", "$21"]
        ),
        ")
         SELECT version.id, version.version, version.recorded_at,
                state.id AS state_id, state.version AS state_version
           FROM version, state"
    );

    let span_id = SourceSpanId::new();
    let byte_end = evidence.content.len() as u32;
    let query = sqlx::query(match promotion {
        None => WITH_SPAN,
        Some(_) => PROMOTED,
    })
    .bind(SourceVersionId::new().0)
    .bind(project.0)
    .bind(source.0)
    .bind(evidence.content)
    .bind(evidence.content_hash)
    .bind(evidence.decision.label())
    .bind(evidence.reason)
    .bind(OffsetDateTime::now_utc())
    .bind(span_id.0)
    .bind(byte_end as i32)
    .bind(evidence.language)
    .bind(evidence.language_confidence);
    let query = match promotion {
        None => query,
        Some(promotion) => crate::jobs::Queued::of(promotion.owed).bind(
            query
                .bind(TopicStateId::new().0)
                .bind(promotion.topic.topic.id.0)
                .bind(promotion.observed_at)
                .bind(promotion.topic.current.map(|id| id.0))
                .bind(promotion.validity.from)
                .bind(promotion.validity.to),
        ),
    };
    let row = query.fetch_one(connection).await?;

    let version = SourceVersion {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: project,
        source_id: source,
        version: from_sql_version(row.get("version")),
        content: evidence.content.to_string(),
        content_hash: evidence.content_hash.to_string(),
        filter_decision: evidence.decision,
        filter_reason: evidence.reason.to_string(),
        recorded_at: row.get("recorded_at"),
    };
    let span = SourceSpan {
        id: span_id,
        project_id: project,
        source_version_id: version.id,
        byte_start: 0,
        byte_end,
        detected_language: evidence.language.map(str::to_string),
        language_confidence: evidence.language_confidence,
    };
    let state = promotion.map(|promotion| TopicState {
        id: row.get::<uuid::Uuid, _>("state_id").into(),
        project_id: project,
        topic_id: promotion.topic.topic.id,
        version: from_sql_version(row.get("state_version")),
        // The span is the whole of the evidence, so the state says all of it.
        content: evidence.content.to_string(),
        source_span_id: span_id,
        language: evidence.language.map(str::to_string),
        observed_at: promotion.observed_at,
        recorded_at: version.recorded_at,
        validity: promotion.validity,
        supersedes: promotion.topic.current,
        deleted_at: None,
    });
    Ok((version, span, state))
}

/// A topic held locked by the transaction that found it, with the state it
/// pointed at when the lock was granted.
///
/// What [`append_promoted`] requires: holding one is the proof that the lock
/// the append numbers under was taken by an earlier statement.
pub struct LockedTopic {
    pub topic: Topic,
    /// The newest surviving state, which the next one supersedes.
    pub current: Option<TopicStateId>,
}

const LOCK_TOPIC: &str = "SELECT id, name, path, created_at, current_state_id FROM topics
                          WHERE project_id = $1 AND name = $2
                          FOR UPDATE";

/// Finds a topic by name and locks it, reading its current state under the
/// lock. `None` when no topic has the name, and then nothing is locked.
///
/// The write path found a topic by name and then locked it by id, and the
/// second statement needed nothing from the first but the id. Asked together,
/// the lock is on the row the name found and the pointer is read under it: a
/// statement that waits for a row lock reads the row as the holder left it, so
/// this sees the pointer a concurrent append just moved, not the one before
/// it. Every path that changes which states survive moves the pointer under
/// this same lock ([`append_promoted`] and [`soft_delete_topic_state`]).
pub async fn lock_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
) -> Result<Option<LockedTopic>> {
    let row = sqlx::query(LOCK_TOPIC)
        .bind(project.0)
        .bind(name)
        .fetch_optional(connection)
        .await?;
    Ok(row.map(|row| locked_topic(project, &row)))
}

/// Creates a topic, files its name in the name index, and returns it locked --
/// or returns, locked, the topic another writer created first.
///
/// For a caller that has just asked [`lock_topic`] and found nothing. A row
/// this inserts is held by the inserting transaction, so it is locked without
/// asking again. The name row goes in by the same statement: a topic that
/// exists and is missing from the name index is a topic no memory will ever
/// derive an edge to, and it was a round trip of its own that needed only the
/// id this chose. `key` and `tokens` are the name as the segmenter reads it,
/// which is why the caller computes them.
///
/// When another writer created the topic in between, the insert waits for it
/// and does nothing, the name row with it -- that writer filed its own -- and
/// the topic is found and locked by a statement of its own.
pub async fn create_topic_named(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
    key: &str,
    tokens: usize,
) -> Result<LockedTopic> {
    create_topic(connection, project, name, Some((key, tokens))).await
}

/// Files the name of the topic called `name` in the name index, if the topic
/// exists and is not filed there yet.
///
/// For a topic `pamin ingest` created, which is left out of the index, when
/// somebody writes it by hand: from then on it is theirs too, and a name
/// somebody wrote is one a memory may name.
///
/// Returns the topic when this call filed it.
pub async fn name_topic(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    name: &str,
    key: &str,
    tokens: usize,
) -> Result<Option<TopicId>> {
    let inserted: Option<(uuid::Uuid,)> = sqlx::query_as(
        "INSERT INTO topic_name_tokens (project_id, topic_id, name_key, token_count)
         SELECT project_id, id, $3, $4 FROM topics WHERE project_id = $1 AND name = $2
         ON CONFLICT (project_id, topic_id) DO NOTHING
         RETURNING topic_id",
    )
    .bind(project.0)
    .bind(name)
    .bind(key)
    .bind(i16::try_from(tokens).unwrap_or(i16::MAX))
    .fetch_optional(executor)
    .await?;
    Ok(inserted.map(|(topic,)| topic.into()))
}

/// [`create_topic_named`] for a topic no memory is expected to name: one cut
/// out of a file, whose name is its path and headings.
///
/// Left out of the name index rather than filed there, because a name in it is
/// a cost every later write pays and not only its own: mention derivation
/// compares every run of a memory's tokens up to the widest name in the
/// project, so one name of twenty tokens makes every memory written afterwards
/// build four times the runs one of five did. A path no prose quotes in full
/// buys nothing for that.
pub async fn create_topic_unnamed(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
) -> Result<LockedTopic> {
    create_topic(connection, project, name, None).await
}

async fn create_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
    named: Option<(&str, usize)>,
) -> Result<LockedTopic> {
    let inserted = sqlx::query(
        "WITH topic AS (
             INSERT INTO topics (id, project_id, name, created_at)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (project_id, name) DO NOTHING
             RETURNING id, name, path, created_at, current_state_id
         ), named AS (
             INSERT INTO topic_name_tokens (project_id, topic_id, name_key, token_count)
             SELECT $2, id, $5, $6 FROM topic WHERE $5::text IS NOT NULL
         )
         SELECT id, name, path, created_at, current_state_id FROM topic",
    )
    .bind(TopicId::new().0)
    .bind(project.0)
    .bind(name)
    .bind(OffsetDateTime::now_utc())
    .bind(named.map(|(key, _)| key))
    .bind(named.map(|(_, tokens)| tokens as i16))
    .fetch_optional(&mut *connection)
    .await?;

    let row = match inserted {
        Some(row) => row,
        None => {
            sqlx::query(LOCK_TOPIC)
                .bind(project.0)
                .bind(name)
                .fetch_one(&mut *connection)
                .await?
        }
    };
    Ok(locked_topic(project, &row))
}

fn locked_topic(project: ProjectId, row: &PgRow) -> LockedTopic {
    LockedTopic {
        topic: Topic {
            id: row.get::<uuid::Uuid, _>("id").into(),
            project_id: project,
            name: row.get("name"),
            path: row.get("path"),
            created_at: row.get("created_at"),
        },
        current: row
            .get::<Option<uuid::Uuid>, _>("current_state_id")
            .map(TopicStateId::from),
    }
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
        content: content_of_span(row),
        source_span_id: row.get::<uuid::Uuid, _>("source_span_id").into(),
        language: row.get("detected_language"),
        observed_at: row.get("observed_at"),
        recorded_at: row.get("recorded_at"),
        validity: Validity::new(row.get("valid_from"), row.get("valid_to")),
        supersedes: row
            .get::<Option<uuid::Uuid>, _>("supersedes")
            .map(Into::into),
        deleted_at: row.get("deleted_at"),
    }
}

/// The text a span covers: `byte_start..byte_end` of its evidence.
///
/// Byte offsets, as `SourceSpan` defines them, which is why the cut is made
/// here rather than with SQL's `substring` -- that counts characters, and the
/// two part company at the first character outside ASCII.
fn span_text(evidence: &str, byte_start: u32, byte_end: u32) -> &str {
    &evidence[byte_start as usize..byte_end as usize]
}

/// A state's content, cut from the evidence the row joined in through
/// `span_columns!`.
///
/// Every span the write path records covers its evidence whole, so the usual
/// case hands the string over rather than copying it.
fn content_of_span(row: &PgRow) -> String {
    let evidence: String = row.get("evidence");
    let (start, end) = (
        row.get::<i32, _>("byte_start") as u32,
        row.get::<i32, _>("byte_end") as u32,
    );
    if start == 0 && end as usize == evidence.len() {
        evidence
    } else {
        span_text(&evidence, start, end).to_string()
    }
}

/// The span columns every state read needs, spelled the way every statement
/// below spells them: its language, and the evidence its content is cut from.
///
/// `topic_states.source_span_id` is `NOT NULL` and references `source_spans`,
/// which references `source_versions` the same way, so the inner joins that
/// bring these in cannot drop a state. What is nullable is the language:
/// detection declines on content too short to be sure about.
///
/// Both joins are on a primary key. The first was measured on
/// `current_states_of` at its hundred-and-fifty-candidate ceiling, same rows,
/// same process, alternating: 1.00 ms median without it and 1.04 ms with, over
/// three runs. The second costs more, because it is where the text is read:
/// on XQuAD-R's 2,640 paragraphs, builds before and after it alternated over
/// five rounds, `current_states_of` at 150 topics went from 1.70 ms to 2.06.
/// That is the price of keeping each memory's text once; `docs/measured.md`
/// has the rest.
macro_rules! span_columns {
    () => {
        ", sp.detected_language, sp.byte_start, sp.byte_end, sv.content AS evidence"
    };
}

/// The joins `span_columns!` reads from, off a `topic_states` aliased `ts`.
macro_rules! span_joins {
    () => {
        " JOIN source_spans sp ON sp.id = ts.source_span_id
          JOIN source_versions sv ON sv.id = sp.source_version_id"
    };
}

/// The columns `row_to_topic_state` reads from `topic_states` itself.
///
/// A macro rather than a constant so the statements below can be assembled with
/// `concat!` and stay `&'static str`. sqlx accepts only a statement that is
/// static or explicitly asserted safe, which is a deliberate obstacle in front
/// of building SQL with `format!`; this keeps the column list in one place
/// without stepping over it.
///
/// Takes the table's alias, because every statement below joins `source_spans`
/// -- which has its own `id` and `project_id` -- and an unqualified list is
/// ambiguous there. PostgreSQL raises that at execution, so only a query that
/// actually runs finds it.
///
/// The content is not in here. It is the span's text, so it comes from the
/// evidence through `span_columns!`.
macro_rules! state_columns {
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
            "deleted_at"
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
///
/// Takes the project because the key it reads is `(project_id, topic_id,
/// version)`, and without its first column that is a walk of the whole index
/// -- every project's states -- to find one row.
pub async fn topic_state(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topic: TopicId,
    version: u32,
) -> Result<Option<TopicState>> {
    let row = sqlx::query(concat!(
        "SELECT ",
        state_columns!("ts."),
        span_columns!(),
        " FROM topic_states ts",
        span_joins!(),
        " WHERE ts.project_id = $1 AND ts.topic_id = $2 AND ts.version = $3"
    ))
    .bind(project.0)
    .bind(topic.0)
    .bind(to_sql_version(version))
    .fetch_optional(executor)
    .await?;

    Ok(row.as_ref().map(row_to_topic_state))
}

/// Every topic's current state, one row per topic.
///
/// What a rebuild indexes. The projection holds one document per topic, so
/// feeding it every live state would write a topic's fourteen versions onto one
/// key and leave whichever the scan reached last -- which is not the same thing
/// as the one the topic stands for.
pub async fn all_current_topic_states(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<TopicState>> {
    let rows = sqlx::query(concat!(
        "SELECT ",
        state_columns!("ts."),
        span_columns!(),
        " FROM topics
          JOIN topic_states ts ON ts.id = topics.current_state_id",
        span_joins!(),
        " WHERE topics.project_id = $1 AND ts.deleted_at IS NULL
          ORDER BY ts.topic_id ASC"
    ))
    .bind(project.0)
    .fetch_all(executor)
    .await?;

    Ok(rows.iter().map(row_to_topic_state).collect())
}

/// The topics [`all_current_topic_states`] would return, without their states.
///
/// For a caller that needs to know which topics the projection should hold
/// and not what they say -- a reshape copies each document from the index it
/// already has, so reading every topic's content out of the ledger to learn
/// its identifier would load the whole project to use none of it. Filtered the
/// same way, so the two can never disagree about which topics are live.
pub async fn current_topic_ids(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<TopicId>> {
    let ids: Vec<uuid::Uuid> = sqlx::query_scalar(
        "SELECT topics.id FROM topics
          JOIN topic_states ts ON ts.id = topics.current_state_id
          WHERE topics.project_id = $1 AND ts.deleted_at IS NULL
          ORDER BY topics.id ASC",
    )
    .bind(project.0)
    .fetch_all(executor)
    .await?;

    Ok(ids.into_iter().map(TopicId::from).collect())
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
    Ok(current_states_named(executor, project, topics)
        .await?
        .into_iter()
        .map(|(_, state)| state)
        .collect())
}

/// [`current_states_of`], with each topic's name beside its state.
///
/// The statement already joins `topics` to follow the pointer, so the name is
/// a column away. The search path needs both -- the state to rank and the name
/// to show -- and asked for the name in a second round trip over the same
/// rows.
pub async fn current_states_named(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topics: &[TopicId],
) -> Result<Vec<(String, TopicState)>> {
    if topics.is_empty() {
        return Ok(Vec::new());
    }

    let ids: Vec<uuid::Uuid> = topics.iter().map(|topic| topic.0).collect();
    let rows = sqlx::query(concat!(
        "SELECT ",
        state_columns!("ts."),
        span_columns!(),
        ", t.name AS topic_name
          FROM topic_states ts
          JOIN topics t ON t.current_state_id = ts.id",
        span_joins!(),
        " WHERE t.project_id = $1 AND t.id = ANY($2)"
    ))
    .bind(project.0)
    .bind(&ids)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .map(|row| (row.get("topic_name"), row_to_topic_state(row)))
        .collect())
}

/// What one topic currently says, found by name.
///
/// Through the pointer on `topics`, like [`current_states_of`], and for the
/// same reason: the alternative is a search for the topic's newest surviving
/// version. The write path used to do exactly that -- read *every* version of
/// the topic, ordered, into memory, and take the last one -- to answer a
/// question the pointer answers directly. That is three round trips and a scan
/// that grows with the topic's edit history, on every write, to compare one
/// string.
///
/// Only the content, because that is the whole question the caller has: is what
/// is being written what the topic already says.
///
/// A topic that does not exist and a topic whose every state has been soft
/// deleted both resolve to nothing, and the caller treats them the same -- in
/// both cases there is nothing for the new content to be identical to.
pub async fn current_content(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    name: &str,
) -> Result<Option<String>> {
    let row = sqlx::query(concat!(
        "SELECT sp.byte_start, sp.byte_end, sv.content AS evidence
           FROM topics t
           JOIN topic_states ts ON ts.id = t.current_state_id",
        span_joins!(),
        " WHERE t.project_id = $1 AND t.name = $2"
    ))
    .bind(project.0)
    .bind(name)
    .fetch_optional(executor)
    .await?;

    Ok(row.as_ref().map(content_of_span))
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
        state_columns!("ts."),
        span_columns!(),
        " FROM topic_states ts",
        span_joins!(),
        " WHERE ts.topic_id = $1 AND ts.deleted_at IS NULL
          ORDER BY ts.version DESC LIMIT 1"
    ))
    .bind(topic.0)
    .fetch_optional(&mut *connection)
    .await?
    .as_ref()
    .map(row_to_topic_state);

    point_at(connection, topic, surviving.as_ref()).await?;

    Ok(true)
}

/// Soft deletes every surviving version of a topic, so that it stands for
/// nothing: off the retrieval surface, its pointer cleared, and the work that
/// removes its document from the projection and closes the edges its content
/// derived owed in the same transaction. Returns whether anything survived to
/// delete.
///
/// What [`soft_delete_topic_state`] does to one version, done to all of them --
/// for a memory whose source no longer says anything at all, which is what a
/// section of a file becomes when the file stops having it. The rows and their
/// evidence stay, so the history reads as it did; a later write to the topic
/// appends a version as any write does, and brings it back.
pub async fn retire_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    topic: TopicId,
) -> Result<bool> {
    sqlx::query("SELECT id FROM topics WHERE id = $1 FOR UPDATE")
        .bind(topic.0)
        .execute(&mut *connection)
        .await?;

    let affected = sqlx::query(
        "UPDATE topic_states SET deleted_at = $2
         WHERE topic_id = $1 AND deleted_at IS NULL",
    )
    .bind(topic.0)
    .bind(OffsetDateTime::now_utc())
    .execute(&mut *connection)
    .await?;

    if affected.rows_affected() == 0 {
        return Ok(false);
    }

    point_at(connection, topic, None).await?;
    crate::jobs::enqueue_all(
        &mut *connection,
        project,
        &[JobKind::SyncTopicIndex, JobKind::DeriveMentions],
        Some(topic.0),
    )
    .await?;

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

/// Whether any evidence is filed under `locator`.
pub async fn source_at(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    locator: &str,
) -> Result<bool> {
    let found: Option<(i32,)> =
        sqlx::query_as("SELECT 1 FROM sources WHERE project_id = $1 AND locator = $2")
            .bind(project.0)
            .bind(locator)
            .fetch_optional(executor)
            .await?;
    Ok(found.is_some())
}

/// The content hash of the newest evidence filed under `locator`, if any.
///
/// For a caller that has to know whether it is about to record what is
/// already there, and has the locator rather than the source's id.
pub async fn latest_hash_at(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    locator: &str,
) -> Result<Option<String>> {
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT sv.content_hash FROM sources s
           JOIN source_versions sv ON sv.project_id = s.project_id AND sv.source_id = s.id
         WHERE s.project_id = $1 AND s.locator = $2
         ORDER BY sv.version DESC LIMIT 1",
    )
    .bind(project.0)
    .bind(locator)
    .fetch_optional(executor)
    .await?;

    Ok(row.map(|(hash,)| hash))
}

/// Loads the newest evidence version for a source.
///
/// Reads back the filter verdict, which is how a caller confirms that filtered
/// content was still stored rather than discarded.
///
/// Takes the project for the reason [`topic_state`] does: the key it reads,
/// `(project_id, source_id, version)`, leads with it.
pub async fn latest_source_version(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    source: SourceId,
) -> Result<Option<SourceVersion>> {
    let row = sqlx::query(
        "SELECT id, project_id, source_id, version, content, content_hash,
                filter_decision, filter_reason, recorded_at
         FROM source_versions WHERE project_id = $1 AND source_id = $2
         ORDER BY version DESC LIMIT 1",
    )
    .bind(project.0)
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
        .map(|row| {
            let content: String = row.get("content");
            let offset = byte_offset(&content, row.get::<i32, _>("match_position") as usize);
            EvidenceMatch {
                source_version: SourceVersion {
                    id: row.get::<uuid::Uuid, _>("id").into(),
                    project_id: row.get::<uuid::Uuid, _>("project_id").into(),
                    source_id: row.get::<uuid::Uuid, _>("source_id").into(),
                    version: from_sql_version(row.get("version")),
                    content,
                    content_hash: row.get("content_hash"),
                    filter_decision: FilterDecision::from_label(row.get("filter_decision"))
                        .unwrap_or(FilterDecision::Promoted),
                    filter_reason: row.get("filter_reason"),
                    recorded_at: row.get("recorded_at"),
                },
                locator: row.get("locator"),
                offset,
            }
        })
        .collect())
}

/// Where the match SQL's `position` found starts, in bytes of `content`.
///
/// `position` answers in characters, counting from one, and an offset here is
/// bytes -- the unit `span_text` cuts in and every caller slices with. The two
/// agree only while everything before the match is ASCII. Counting the
/// characters again in Rust is exact because the cluster is initialised UTF-8,
/// where PostgreSQL's character is a code point and so is a Rust `char`. It
/// holds for the folded search too: under the default libc provider `lower`
/// maps one character to one, so a position in the folded text is the same
/// position in the original.
fn byte_offset(content: &str, position: usize) -> usize {
    content
        .char_indices()
        .nth(position.saturating_sub(1))
        .map_or(content.len(), |(byte, _)| byte)
}

/// Records how many topics' names tokenize, in one statement.
///
/// The key is computed by the caller because tokenizing is the segmenter's
/// job and the segmenter lives above this layer. A write files its one name
/// with the topic, in [`create_topic_named`]. This is what a rebuild uses,
/// because a rebuild records all of them -- and doing that a row at a time is
/// one round trip per topic, which on the corpora this project measures is
/// thirteen thousand of them for a table with no more rows than that.
///
/// `unnest` over three arrays, which is the same shape [`topics_named_by`] and
/// `graph::live_versions_of` already use for their reads. A conflict updates
/// the row, so a rebuild over a table that already has these rows rewrites
/// them rather than failing -- which is what a rebuild is.
pub async fn record_topic_names(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    names: &[(TopicId, String, usize)],
) -> Result<()> {
    if names.is_empty() {
        return Ok(());
    }

    let topics: Vec<uuid::Uuid> = names.iter().map(|(topic, _, _)| topic.0).collect();
    let keys: Vec<String> = names.iter().map(|(_, key, _)| key.clone()).collect();
    // `i16` because that is the column, and a name with more tokens than a
    // `smallint` holds is not a name.
    let counts: Vec<i16> = names
        .iter()
        .map(|(_, _, tokens)| i16::try_from(*tokens).unwrap_or(i16::MAX))
        .collect();

    sqlx::query(
        "INSERT INTO topic_name_tokens (project_id, topic_id, name_key, token_count)
         SELECT $1, topic_id, name_key, token_count
           FROM unnest($2::uuid[], $3::text[], $4::smallint[])
             AS incoming(topic_id, name_key, token_count)
         ON CONFLICT (project_id, topic_id) DO UPDATE
             SET name_key = EXCLUDED.name_key, token_count = EXCLUDED.token_count",
    )
    .bind(project.0)
    .bind(&topics)
    .bind(&keys)
    .bind(&counts)
    .execute(executor)
    .await?;

    Ok(())
}

/// How many tokens the longest topic name in this project has.
///
/// Bounds the lookup: a run of tokens wider than the widest name cannot be a
/// name, so there is no point asking about it. Zero when the project has no
/// topics, which means there is nothing to ask about at all.
pub async fn widest_topic_name(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<usize> {
    let row: (Option<i16>,) =
        sqlx::query_as("SELECT MAX(token_count) FROM topic_name_tokens WHERE project_id = $1")
            .bind(project.0)
            .fetch_one(executor)
            .await?;

    Ok(row.0.unwrap_or(0).max(0) as usize)
}

/// The topics whose names appear among these token runs.
///
/// The runs are every window of the text being examined, at every width up to
/// [`widest_topic_name`]. A name matches only as a contiguous run, so equality
/// against the stored key is the whole test -- there is no candidate set to
/// re-check afterwards.
pub async fn topics_named_by(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    runs: &[String],
) -> Result<Vec<TopicId>> {
    Ok(names_matching(executor, project, runs)
        .await?
        .into_iter()
        .map(|(_, topic)| topic)
        .collect())
}

/// [`topics_named_by`], with the run each topic's name matched.
///
/// For a caller asking on behalf of several texts at once: each text's runs
/// go into one question, and the run beside each answer is what says which
/// text it belongs to.
pub async fn names_matching(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    runs: &[String],
) -> Result<Vec<(String, TopicId)>> {
    if runs.is_empty() {
        return Ok(Vec::new());
    }

    let rows = sqlx::query(
        "SELECT name_key, topic_id FROM topic_name_tokens
         WHERE project_id = $1 AND name_key = ANY($2)",
    )
    .bind(project.0)
    .bind(runs)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .map(|row| {
            (
                row.get("name_key"),
                row.get::<uuid::Uuid, _>("topic_id").into(),
            )
        })
        .collect())
}

/// Every topic in a project that belongs in the name index: all of them but
/// the sections `pamin ingest` cut out of files, which
/// [`create_topic_unnamed`] keeps out of it.
///
/// Told apart by their sources rather than their name. A section's evidence is
/// filed under `file:{name}` and a manual memory's under `manual:{name}`, and
/// the unique key on `(project_id, locator)` answers both for each topic,
/// where a rule about what a name looks like would also catch a topic someone
/// wrote by hand. A topic with both was written by hand as well as ingested,
/// and keeps its name.
pub async fn topics_to_name(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<Topic>> {
    let rows = sqlx::query(
        "SELECT t.id, t.name, t.path, t.created_at FROM topics t
         WHERE t.project_id = $1
           AND (NOT EXISTS (
                    SELECT 1 FROM sources s
                     WHERE s.project_id = t.project_id
                       AND s.locator = 'file:' || t.name)
                OR EXISTS (
                    SELECT 1 FROM sources s
                     WHERE s.project_id = t.project_id
                       AND s.locator = 'manual:' || t.name))
         ORDER BY t.name ASC",
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

/// A topic `pamin ingest` has put on the retrieval surface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileTopic {
    pub name: String,
    /// What it says now, which an ingest that retires it checks is still what
    /// it says when it gets there.
    pub current: TopicStateId,
    /// The directory it was last ingested from.
    pub root: Option<String>,
}

/// Every topic `pamin ingest` has put on the retrieval surface in this project.
///
/// One read for a whole ingest, rather than one per file: an ingest compares
/// each file's sections against what an earlier ingest left under that file,
/// and asking once per file would scan the project's sources once per file.
pub async fn file_topic_names(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<FileTopic>> {
    let rows: Vec<(String, uuid::Uuid, Option<String>)> = sqlx::query_as(
        "SELECT t.name, t.current_state_id, s.root FROM sources s
           JOIN topics t ON t.project_id = s.project_id AND t.name = substr(s.locator, 6)
         WHERE s.project_id = $1 AND s.kind = 'file' AND t.current_state_id IS NOT NULL",
    )
    .bind(project.0)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(name, current, root)| FileTopic {
            name,
            current: current.into(),
            root,
        })
        .collect())
}

/// [`file_topic_names`] for the files an ingest is looking at, rather than
/// every file the project has: the topics whose name starts with one of
/// `prefixes` -- a file's name, or a directory's followed by `/`.
///
/// A prefix is a match on the start of the name and nothing more, so `a.md`
/// also finds `a.md.bak`'s sections; the caller keeps what belongs to it.
/// Each prefix is a range of the `C`-collated locator index, given as bounds
/// rather than a `LIKE` pattern: a parameter's pattern is unknown when the
/// statement is planned, and a range of parameters is not.
pub async fn file_topics_under(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    prefixes: &[String],
) -> Result<Vec<FileTopic>> {
    if prefixes.is_empty() {
        return Ok(Vec::new());
    }
    let (low, high): (Vec<String>, Vec<String>) = prefixes
        .iter()
        .map(|prefix| {
            let low = format!("file:{prefix}");
            // Above every string that starts with `low`: no character sorts
            // after the last one Unicode has, in UTF-8 bytes or code points.
            let high = format!("{low}{}", char::MAX);
            (low, high)
        })
        .unzip();
    let rows: Vec<(String, uuid::Uuid, Option<String>)> = sqlx::query_as(
        r#"SELECT DISTINCT t.name, t.current_state_id, s.root
             FROM unnest($2::text[], $3::text[]) AS r (low, high)
             JOIN sources s
               ON s.project_id = $1 AND s.kind = 'file'
              AND (s.locator COLLATE "C") >= r.low AND (s.locator COLLATE "C") < r.high
             JOIN topics t ON t.project_id = s.project_id AND t.name = substr(s.locator, 6)
            WHERE t.current_state_id IS NOT NULL"#,
    )
    .bind(project.0)
    .bind(&low)
    .bind(&high)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(name, current, root)| FileTopic {
            name,
            current: current.into(),
            root,
        })
        .collect())
}

/// Which of `topics` are sections of a file, each with the directory its file
/// was last ingested from.
pub async fn file_roots_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topics: &[String],
) -> Result<Vec<(String, String)>> {
    if topics.is_empty() {
        return Ok(Vec::new());
    }
    let locators: Vec<String> = topics.iter().map(|topic| format!("file:{topic}")).collect();
    let rows: Vec<(String, String)> = sqlx::query_as(
        "SELECT substr(locator, 6), root FROM sources
          WHERE project_id = $1 AND locator = ANY($2) AND root IS NOT NULL",
    )
    .bind(project.0)
    .bind(&locators)
    .fetch_all(executor)
    .await?;
    Ok(rows)
}

/// What a file looked like when an ingest read it. See `V18__file_records`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileRecord {
    pub root: String,
    pub file: String,
    pub size: u64,
    pub modified_ns: i64,
    pub changed_ns: i64,
    pub inode: u64,
    pub read_ns: i64,
    pub content_hash: String,
    pub sections: u32,
    pub build: String,
}

/// The records of these files, each named from the root beside it.
pub async fn file_records_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    roots: &[String],
    files: &[String],
) -> Result<Vec<FileRecord>> {
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let rows = sqlx::query(
        "SELECT f.root, f.file, f.size, f.modified_ns, f.changed_ns, f.inode, f.read_ns,
                f.content_hash, f.sections, f.build
           FROM unnest($2::text[], $3::text[]) AS k (root, file)
           JOIN file_records f ON f.project_id = $1 AND f.root = k.root AND f.file = k.file",
    )
    .bind(project.0)
    .bind(roots)
    .bind(files)
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().map(file_record).collect())
}

/// Every record under `root` whose file starts with `prefix`: what a tree held
/// when it was last ingested, for a caller asking what has left it.
pub async fn file_records_under(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    root: &str,
    prefix: &str,
) -> Result<Vec<FileRecord>> {
    let rows = sqlx::query(
        "SELECT root, file, size, modified_ns, changed_ns, inode, read_ns, content_hash, sections,
                build
           FROM file_records
          WHERE project_id = $1 AND root = $2 AND starts_with(file, $3)",
    )
    .bind(project.0)
    .bind(root)
    .bind(prefix)
    .fetch_all(executor)
    .await?;
    Ok(rows.iter().map(file_record).collect())
}

fn file_record(row: &sqlx::postgres::PgRow) -> FileRecord {
    // Stored in a BIGINT's bits; see the migration.
    FileRecord {
        root: row.get("root"),
        file: row.get("file"),
        size: row.get::<i64, _>("size") as u64,
        modified_ns: row.get("modified_ns"),
        changed_ns: row.get("changed_ns"),
        inode: row.get::<i64, _>("inode") as u64,
        read_ns: row.get("read_ns"),
        content_hash: row.get("content_hash"),
        sections: row.get::<i32, _>("sections").max(0) as u32,
        build: row.get("build"),
    }
}

/// Records these files as read, replacing what was recorded of them.
pub async fn put_file_records(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    records: &[FileRecord],
) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    let column = |f: fn(&FileRecord) -> i64| records.iter().map(f).collect::<Vec<i64>>();
    let roots: Vec<&str> = records.iter().map(|r| r.root.as_str()).collect();
    let files: Vec<&str> = records.iter().map(|r| r.file.as_str()).collect();
    let hashes: Vec<&str> = records.iter().map(|r| r.content_hash.as_str()).collect();
    let builds: Vec<&str> = records.iter().map(|r| r.build.as_str()).collect();
    let sections: Vec<i32> = records
        .iter()
        .map(|r| i32::try_from(r.sections).unwrap_or(i32::MAX))
        .collect();
    sqlx::query(
        "INSERT INTO file_records AS f
                (project_id, root, file, size, modified_ns, changed_ns, inode, read_ns,
                 content_hash, sections, build)
         SELECT $1, * FROM unnest($2::text[], $3::text[], $4::bigint[], $5::bigint[],
                                  $6::bigint[], $7::bigint[], $8::bigint[], $9::text[],
                                  $10::int[], $11::text[])
         ON CONFLICT (project_id, root, file) DO UPDATE
            SET size = excluded.size, modified_ns = excluded.modified_ns,
                changed_ns = excluded.changed_ns, inode = excluded.inode,
                read_ns = excluded.read_ns, content_hash = excluded.content_hash,
                sections = excluded.sections, build = excluded.build",
    )
    .bind(project.0)
    .bind(&roots)
    .bind(&files)
    .bind(column(|r| r.size as i64))
    .bind(column(|r| r.modified_ns))
    .bind(column(|r| r.changed_ns))
    .bind(column(|r| r.inode as i64))
    .bind(column(|r| r.read_ns))
    .bind(&hashes)
    .bind(&sections)
    .bind(&builds)
    .execute(executor)
    .await?;
    Ok(())
}

/// Forgets what was recorded of these files: each `files[i]` under `roots[i]`
/// when `elsewhere` is false, or under every root but `roots[i]` when it is
/// true -- for an ingest that rewrote sections another tree's file of the same
/// name had recorded as its own.
pub async fn forget_file_records(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    roots: &[String],
    files: &[String],
    elsewhere: bool,
) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "DELETE FROM file_records f
          USING unnest($2::text[], $3::text[]) AS k (root, file)
          WHERE f.project_id = $1 AND f.file = k.file AND (f.root = k.root) <> $4",
    )
    .bind(project.0)
    .bind(roots)
    .bind(files)
    .bind(elsewhere)
    .execute(executor)
    .await?;
    Ok(())
}

/// Forgets the record of each `files[i]` under `roots[i]` one of whose
/// sections, `topics[i]`, was written by hand since `since`: a write that may
/// have landed while an ingest was recording that file, after its own
/// [`forget_file_record_of`] found no record yet to forget.
pub async fn forget_file_records_written_since(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    roots: &[String],
    files: &[String],
    topics: &[String],
    since: OffsetDateTime,
) -> Result<()> {
    if topics.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "DELETE FROM file_records f
          USING unnest($2::text[], $3::text[], $4::text[]) AS k (root, file, topic)
           JOIN sources s ON s.project_id = $1 AND s.locator = 'manual:' || k.topic
           JOIN source_versions v
             ON v.project_id = $1 AND v.source_id = s.id AND v.recorded_at >= $5
          WHERE f.project_id = $1 AND f.root = k.root AND f.file = k.file",
    )
    .bind(project.0)
    .bind(roots)
    .bind(files)
    .bind(topics)
    .bind(since)
    .execute(executor)
    .await?;
    Ok(())
}

/// How many sections a file's prefix range holds on the retrieval surface:
/// every file whose name starts with `prefix`, for a caller comparing it with
/// what the records under that prefix account for.
pub async fn count_file_topics_under(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    prefix: &str,
) -> Result<i64> {
    let low = format!("file:{prefix}");
    let high = format!("{low}{}", char::MAX);
    let (count,): (i64,) = sqlx::query_as(
        r#"SELECT count(*) FROM sources s
             JOIN topics t ON t.project_id = s.project_id AND t.name = substr(s.locator, 6)
            WHERE s.project_id = $1 AND s.kind = 'file'
              AND (s.locator COLLATE "C") >= $2 AND (s.locator COLLATE "C") < $3
              AND t.current_state_id IS NOT NULL"#,
    )
    .bind(project.0)
    .bind(&low)
    .bind(&high)
    .fetch_one(executor)
    .await?;
    Ok(count)
}

/// Forgets the record of the file `topic` was cut from, if it is a section of
/// one: a write to it other than its file's leaves the file saying something
/// its section no longer does, which only reading the file again can settle.
///
/// The file is the record whose name the topic's starts with, as a section's
/// does: the whole name, or the name and then ` › ` or ` #`.
pub async fn forget_file_record_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    topic: &str,
) -> Result<()> {
    sqlx::query(
        "DELETE FROM file_records f
          USING sources s
          WHERE s.project_id = $1 AND s.locator = 'file:' || $2
            AND f.project_id = $1 AND f.root = s.root
            AND ($2 = f.file OR starts_with($2, f.file || ' › ')
                 OR starts_with($2, f.file || ' #'))",
    )
    .bind(project.0)
    .bind(topic)
    .execute(executor)
    .await?;
    Ok(())
}

/// A path `pamin ingest` keeps in sync. See `V19__synced_paths`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncedPath {
    /// The project's name.
    pub project: String,
    pub path: String,
    pub keep_missing: bool,
    pub vector_index: String,
}

/// Records `paths` as kept in sync for this project, replacing what an
/// earlier ingest of them said.
pub async fn sync_paths(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    paths: &[String],
    keep_missing: bool,
    vector_index: &str,
) -> Result<()> {
    if paths.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "INSERT INTO synced_paths (project_id, path, keep_missing, vector_index, added_at)
         SELECT $1, path, $3, $4, now() FROM unnest($2::text[]) AS path
         ON CONFLICT (project_id, path) DO UPDATE
            SET keep_missing = excluded.keep_missing, vector_index = excluded.vector_index",
    )
    .bind(project.0)
    .bind(paths)
    .bind(keep_missing)
    .bind(vector_index)
    .execute(executor)
    .await?;
    Ok(())
}

/// Stops keeping `paths` in sync for this project, returning those that were.
pub async fn unsync_paths(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    paths: &[String],
) -> Result<Vec<String>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let removed: Vec<(String,)> = sqlx::query_as(
        "DELETE FROM synced_paths WHERE project_id = $1 AND path = ANY($2) RETURNING path",
    )
    .bind(project.0)
    .bind(paths)
    .fetch_all(executor)
    .await?;
    Ok(removed.into_iter().map(|(path,)| path).collect())
}

/// The paths kept in sync in one project, by path.
pub async fn synced_paths_of(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<SyncedPath>> {
    let rows: Vec<(String, String, bool, String)> = sqlx::query_as(
        "SELECT p.name, s.path, s.keep_missing, s.vector_index
           FROM synced_paths s JOIN projects p ON p.id = s.project_id
          WHERE s.project_id = $1
          ORDER BY s.path",
    )
    .bind(project.0)
    .fetch_all(executor)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(project, path, keep_missing, vector_index)| SyncedPath {
            project,
            path,
            keep_missing,
            vector_index,
        })
        .collect())
}

/// Every path kept in sync, in every project, by project and path.
pub async fn synced_paths(executor: impl PgExecutor<'_>) -> Result<Vec<SyncedPath>> {
    let rows: Vec<(String, String, bool, String)> = sqlx::query_as(
        "SELECT p.name, s.path, s.keep_missing, s.vector_index
           FROM synced_paths s JOIN projects p ON p.id = s.project_id
          ORDER BY p.name, s.path",
    )
    .fetch_all(executor)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(project, path, keep_missing, vector_index)| SyncedPath {
            project,
            path,
            keep_missing,
            vector_index,
        })
        .collect())
}

/// A topic as it stands now, found by name.
#[derive(Clone, Debug)]
pub struct Named {
    pub name: String,
    pub topic: TopicId,
    pub current: TopicStateId,
    pub content: String,
}

/// Each of the topics `names` as it stands now; a name with no topic, or none
/// on the retrieval surface, is left out.
///
/// For an ingest, which asks it about many sections at once: to number a
/// file's sections against what they said before, and to resolve the
/// sections its outline relates.
pub async fn current_named(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    names: &[String],
) -> Result<Vec<Named>> {
    if names.is_empty() {
        return Ok(Vec::new());
    }
    let rows = sqlx::query(concat!(
        "SELECT t.name, t.id, ts.id AS state_id, sp.byte_start, sp.byte_end,
                sv.content AS evidence
           FROM topics t
           JOIN topic_states ts ON ts.id = t.current_state_id",
        span_joins!(),
        " WHERE t.project_id = $1 AND t.name = ANY($2)"
    ))
    .bind(project.0)
    .bind(names)
    .fetch_all(executor)
    .await?;
    Ok(rows
        .iter()
        .map(|row| Named {
            name: row.get("name"),
            topic: row.get::<uuid::Uuid, _>("id").into(),
            current: row.get::<uuid::Uuid, _>("state_id").into(),
            content: content_of_span(row),
        })
        .collect())
}

/// Renames a topic, keeping the name it had as an alias that
/// [`find_topic`] resolves, and files the source at `from_locator` under
/// `to_locator`; then owes the projection the topic again, since its name is
/// read into every span's header.
///
/// Returns false, changing nothing, when another topic or source already holds
/// the new name. For a topic locked by the caller's transaction.
pub async fn rename_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    topic: TopicId,
    (from, to): (&str, &str),
    (from_locator, to_locator): (&str, &str),
) -> Result<bool> {
    let taken: Option<(i32,)> = sqlx::query_as(
        "SELECT 1 FROM topics WHERE project_id = $1 AND name = $2
         UNION ALL
         SELECT 1 FROM sources WHERE project_id = $1 AND locator = $3
         LIMIT 1",
    )
    .bind(project.0)
    .bind(to)
    .bind(to_locator)
    .fetch_optional(&mut *connection)
    .await?;
    if taken.is_some() {
        return Ok(false);
    }
    sqlx::query("UPDATE topics SET name = $2 WHERE id = $1")
        .bind(topic.0)
        .bind(to)
        .execute(&mut *connection)
        .await?;
    sqlx::query("UPDATE sources SET locator = $3 WHERE project_id = $1 AND locator = $2")
        .bind(project.0)
        .bind(from_locator)
        .bind(to_locator)
        .execute(&mut *connection)
        .await?;
    // The new name is a topic's own now, and the old one is this topic's.
    sqlx::query(
        "WITH gone AS (DELETE FROM topic_aliases WHERE project_id = $1 AND name = $3)
         INSERT INTO topic_aliases (project_id, name, topic_id, renamed_at)
         VALUES ($1, $2, $4, $5)
         ON CONFLICT (project_id, name)
         DO UPDATE SET topic_id = EXCLUDED.topic_id, renamed_at = EXCLUDED.renamed_at",
    )
    .bind(project.0)
    .bind(from)
    .bind(to)
    .bind(topic.0)
    .bind(OffsetDateTime::now_utc())
    .execute(&mut *connection)
    .await?;
    crate::jobs::enqueue_all(
        &mut *connection,
        project,
        &[JobKind::SyncTopicIndex],
        Some(topic.0),
    )
    .await?;
    Ok(true)
}

/// [`set_source_root`] for every source filed under `locators`, in one
/// statement: for the sections an ingest found unchanged, which write nothing
/// else, so that each records the tree it was last ingested from all the same.
pub async fn set_roots_at(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    locators: &[String],
    root: &str,
) -> Result<()> {
    if locators.is_empty() {
        return Ok(());
    }
    sqlx::query(
        "UPDATE sources SET root = $3
         WHERE project_id = $1 AND locator = ANY($2) AND root IS DISTINCT FROM $3",
    )
    .bind(project.0)
    .bind(locators)
    .bind(root)
    .execute(executor)
    .await?;
    Ok(())
}

/// Records `root` as the directory the file source was last ingested from.
pub async fn set_source_root(
    executor: impl PgExecutor<'_>,
    source: SourceId,
    root: &str,
) -> Result<()> {
    sqlx::query("UPDATE sources SET root = $2 WHERE id = $1 AND root IS DISTINCT FROM $2")
        .bind(source.0)
        .bind(root)
        .execute(executor)
        .await?;
    Ok(())
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

/// The topics written most recently, newest first.
///
/// Bounded, and ordered by an index rather than by sorting the project: the
/// caller wants to know what is here, and a caller that wanted all of it would
/// be asking for something no context window can hold anyway.
pub async fn recent_topics(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    limit: u32,
) -> Result<Vec<Topic>> {
    let rows = sqlx::query(
        "SELECT id, name, path, created_at FROM topics
         WHERE project_id = $1 ORDER BY created_at DESC LIMIT $2",
    )
    .bind(project.0)
    .bind(i64::from(limit))
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

/// How many topics a project holds.
///
/// The projection is keyed by topic, so this is how many documents it will
/// hold once it has caught up -- which is what the index needs to size its
/// segments before it is opened.
pub async fn topic_count(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<u64> {
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM topics WHERE project_id = $1")
        .bind(project.0)
        .fetch_one(executor)
        .await?;
    Ok(count.max(0) as u64)
}

// A topic by its name, or by a name it went by before a rename when no topic
// holds that name now: the topic's own row, read under its current name.
const FIND_TOPIC: &str = "SELECT id, name, path, created_at FROM topics
                          WHERE project_id = $1 AND name = $2
                          UNION ALL
                          SELECT t.id, t.name, t.path, t.created_at
                            FROM topic_aliases a JOIN topics t ON t.id = a.topic_id
                           WHERE a.project_id = $1 AND a.name = $2
                             AND NOT EXISTS (
                                 SELECT 1 FROM topics WHERE project_id = $1 AND name = $2
                             )
                          LIMIT 1";

/// Looks up a topic by name within a project, or by a name it was renamed
/// from that no topic holds now; the topic comes back under its current name.
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
