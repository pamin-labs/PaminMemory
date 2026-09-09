//! The cascade outbox: durable work a write leaves behind.
//!
//! A write commits evidence, a span, a topic state, the topic's current
//! pointer, and a row here, in one transaction that touches nothing outside
//! PostgreSQL. Everything derived from it happens afterwards, driven from that
//! row. So a write does not depend on the index being reachable, and the
//! derived work outlives the process that scheduled it -- which is the whole
//! point, and the part a `tokio::spawn` cannot give.
//!
//! Jobs are addressed by subject rather than by event: "bring topic T up to
//! date", not "T changed at 12:04". That makes them idempotent by construction
//! and makes them coalesce -- fourteen edits to one topic leave one pending row
//! and thirteen embeddings never computed.

use std::time::Duration;

use pamin_core::{IndexJobId, JobKind, ProjectId};
use sqlx::{PgExecutor, PgPool, Row};
use time::OffsetDateTime;

use crate::error::Result;
use crate::sql::SqlLabel;

/// How long a claim is good for.
///
/// A worker that dies holding a job leaves the row claimed, and nothing would
/// pick it up again without this. Long enough that an ordinary job finishes
/// inside it -- the slowest is an embedding -- and short enough that a crash
/// does not strand work for the rest of the session.
pub const LEASE: Duration = Duration::from_secs(60);

/// How long a failed job waits before it is tried again.
pub const RETRY_AFTER: Duration = Duration::from_secs(3600);

/// One row of pending work.
#[derive(Clone, Debug)]
pub struct Job {
    pub id: IndexJobId,
    pub project_id: ProjectId,
    pub kind: JobKind,
    /// The topic or state the job is about. Absent for project-wide work.
    pub subject: Option<uuid::Uuid>,
    /// How many times this job has been claimed, including now.
    pub attempts: i32,
}

/// What runs first when several jobs are pending.
///
/// Deriving the edges of a memory somebody just wrote is worth more than
/// rebuilding a vector index in the background. Without an ordering the
/// background work goes first as often as not, and the write the user is
/// waiting on is behind it.
fn priority(kind: JobKind) -> i32 {
    match kind {
        JobKind::SyncTopicIndex | JobKind::UnindexState => 10,
        JobKind::DeriveMentions => 20,
        JobKind::BackfillMentions => 50,
        JobKind::OptimizeIndex => 100,
    }
}

/// Schedules work, coalescing with anything already scheduled for it.
///
/// One row per subject and kind, forever. A conflict means the same work is
/// either still pending -- in which case this call is already represented by it
/// -- or was completed earlier and is being asked for again, in which case the
/// row is revived. Either way the queue never holds two rows saying the same
/// thing.
///
/// Reviving clears the claim as well as the completion. That is what keeps a
/// worker from marking work done that was requested after it started reading:
/// [`complete`] only completes a job it still holds, and a claim cleared out
/// from under it is exactly the signal that the state it read is stale.
pub async fn enqueue(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
    kind: JobKind,
    subject: Option<uuid::Uuid>,
) -> Result<()> {
    let key = match subject {
        Some(subject) => format!("{kind}:{subject}"),
        None => format!("{kind}:"),
    };
    let payload = serde_json::json!({ "subject": subject });
    let now = OffsetDateTime::now_utc();

    sqlx::query(
        "INSERT INTO index_jobs
             (id, project_id, job_type, payload, idempotency_key,
              available_at, created_at, priority)
         VALUES ($1, $2, $3, $4, $5, $6, $6, $7)
         ON CONFLICT (project_id, idempotency_key) DO UPDATE
             SET completed_at = NULL,
                 available_at = $6,
                 claimed_at   = NULL,
                 claimed_by   = NULL,
                 last_error   = NULL,
                 attempts     = 0",
    )
    .bind(IndexJobId::new().0)
    .bind(project.0)
    .bind(kind.label())
    .bind(&payload)
    .bind(&key)
    .bind(now)
    .bind(priority(kind))
    .execute(executor)
    .await?;

    Ok(())
}

/// Takes up to `batch` jobs that are due, and holds them for [`LEASE`].
///
/// `FOR UPDATE SKIP LOCKED` is what lets several workers drain the queue at
/// once without queueing behind each other on the same row -- the construct
/// `V1` named when it created this table.
///
/// Attempts are counted at claim rather than at failure, so a job that panics
/// the worker still runs out of attempts. Counting on the way out would let one
/// poisoned job take down every worker that reaches it, for ever.
///
/// Ordered by priority alone. Fairness between projects belongs with the
/// process that serves many of them at once; here the worker runs inside a
/// command that is already about one project.
pub async fn claim(pool: &PgPool, worker: &str, batch: i32) -> Result<Vec<Job>> {
    let now = OffsetDateTime::now_utc();
    let rows = sqlx::query(
        "UPDATE index_jobs
            SET claimed_at   = $1,
                claimed_by   = $2,
                attempts     = attempts + 1,
                available_at = $3
          WHERE id IN (
              SELECT id FROM index_jobs
               WHERE completed_at IS NULL
                 AND available_at <= $1
                 -- A job that has used its attempts stays pending with its
                 -- error rather than coming round again. Retrying for ever
                 -- turns one poisoned job into a worker that never does
                 -- anything else; `cascade replay` is how it comes back.
                 AND attempts < $5
               ORDER BY priority, available_at
                 FOR UPDATE SKIP LOCKED
               LIMIT $4
          )
      RETURNING id, project_id, job_type, payload, attempts",
    )
    .bind(now)
    .bind(worker)
    .bind(now + LEASE)
    .bind(i64::from(batch))
    .bind(pamin_core::MAX_ATTEMPTS)
    .fetch_all(pool)
    .await?;

    Ok(rows.iter().map(row_to_job).collect())
}

/// Marks a job done, if this worker still holds it.
///
/// Returns false when it does not, which happens two ways and means the same
/// thing both times: the job was requested again while this attempt was
/// running, or the lease expired and another worker took it. In either case the
/// state this attempt read is not the state the queue is now asking about, so
/// the row stays pending and runs again.
pub async fn complete(executor: impl PgExecutor<'_>, job: &Job, worker: &str) -> Result<bool> {
    let completed = sqlx::query(
        "UPDATE index_jobs
            SET completed_at = $3, claimed_at = NULL, claimed_by = NULL, last_error = NULL
          WHERE id = $1 AND claimed_by = $2 AND completed_at IS NULL",
    )
    .bind(job.id.0)
    .bind(worker)
    .bind(OffsetDateTime::now_utc())
    .execute(executor)
    .await?;

    Ok(completed.rows_affected() > 0)
}

/// Records a failure and schedules a retry, until the attempts run out.
///
/// A job that has used its attempts is left pending with its error, which is
/// what `pamin cascade failed` reads. It is not retried again on its own:
/// retrying for ever turns one poisoned job into a worker that never does
/// anything else.
pub async fn fail(
    executor: impl PgExecutor<'_>,
    job: &Job,
    worker: &str,
    error: &str,
) -> Result<()> {
    let exhausted = job.attempts >= pamin_core::MAX_ATTEMPTS;
    let retry_at = OffsetDateTime::now_utc() + RETRY_AFTER;

    sqlx::query(
        "UPDATE index_jobs
            SET last_error   = $3,
                claimed_at   = NULL,
                claimed_by   = NULL,
                available_at = CASE WHEN $4 THEN available_at ELSE $5 END
          WHERE id = $1 AND claimed_by = $2",
    )
    .bind(job.id.0)
    .bind(worker)
    .bind(error)
    .bind(exhausted)
    .bind(retry_at)
    .execute(executor)
    .await?;

    Ok(())
}

/// How many jobs are waiting, whether or not they are due yet.
pub async fn pending(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<i64> {
    let (waiting,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM index_jobs
          WHERE project_id = $1 AND completed_at IS NULL",
    )
    .bind(project.0)
    .fetch_one(executor)
    .await?;

    Ok(waiting)
}

/// Jobs that have used their attempts, with the error that stopped them.
pub async fn exhausted(
    executor: impl PgExecutor<'_>,
    project: ProjectId,
) -> Result<Vec<(Job, String)>> {
    let rows = sqlx::query(
        "SELECT id, project_id, job_type, payload, attempts, last_error
           FROM index_jobs
          WHERE project_id = $1 AND completed_at IS NULL AND attempts >= $2
          ORDER BY priority, available_at",
    )
    .bind(project.0)
    .bind(pamin_core::MAX_ATTEMPTS)
    .fetch_all(executor)
    .await?;

    Ok(rows
        .iter()
        .map(|row| {
            (
                row_to_job(row),
                row.get::<Option<String>, _>("last_error")
                    .unwrap_or_default(),
            )
        })
        .collect())
}

/// Makes exhausted jobs due again, and returns how many.
///
/// The manual counterpart to the retry the worker gives up on: whatever was
/// wrong has presumably been fixed, and this says so.
pub async fn replay(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<u64> {
    let revived = sqlx::query(
        "UPDATE index_jobs
            SET attempts = 0, available_at = $2, claimed_at = NULL, claimed_by = NULL
          WHERE project_id = $1 AND completed_at IS NULL AND attempts >= $3",
    )
    .bind(project.0)
    .bind(OffsetDateTime::now_utc())
    .bind(pamin_core::MAX_ATTEMPTS)
    .execute(executor)
    .await?;

    Ok(revived.rows_affected())
}

/// Abandons exhausted jobs, and returns how many.
///
/// Completing them rather than deleting them keeps the row, so a later write to
/// the same subject revives it rather than being coalesced onto a row that no
/// longer means anything.
pub async fn discard(executor: impl PgExecutor<'_>, project: ProjectId) -> Result<u64> {
    let discarded = sqlx::query(
        "UPDATE index_jobs
            SET completed_at = $2, claimed_at = NULL, claimed_by = NULL
          WHERE project_id = $1 AND completed_at IS NULL AND attempts >= $3",
    )
    .bind(project.0)
    .bind(OffsetDateTime::now_utc())
    .bind(pamin_core::MAX_ATTEMPTS)
    .execute(executor)
    .await?;

    Ok(discarded.rows_affected())
}

fn row_to_job(row: &sqlx::postgres::PgRow) -> Job {
    let payload: serde_json::Value = row.get("payload");

    Job {
        id: row.get::<uuid::Uuid, _>("id").into(),
        project_id: row.get::<uuid::Uuid, _>("project_id").into(),
        // The column's CHECK constraint admits nothing else, and the drift test
        // holds it to the same list this parses from.
        kind: JobKind::from_label(row.get("job_type")).unwrap_or(JobKind::SyncTopicIndex),
        subject: payload
            .get("subject")
            .and_then(serde_json::Value::as_str)
            .and_then(|subject| uuid::Uuid::parse_str(subject).ok()),
        attempts: row.get("attempts"),
    }
}
