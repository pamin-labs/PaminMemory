//! What the outbox keeps once work is done: nothing.
//!
//! Settled jobs used to be kept, first for ever and then for an hour, on a
//! reason that was real: `enqueue` upserts on the idempotency key, so a
//! completed row was revived by the next write to the same subject rather than
//! joined by a second row saying the same thing. Keeping them made the queue
//! the largest object in the database -- 26 MB of a 62 MB workspace over 13,014
//! documents, and 632 MB of a 1.7 GB one on the evaluation workspace, where
//! 651,128 of its 1,054,646 rows were settled. Nothing read them.
//!
//! A job is now deleted when it completes. That is equivalent rather than a
//! behaviour change, because the insert takes its other branch and produces a
//! row in the same state the revival did. What must not happen is a row being
//! deleted out from under a worker, and what must still happen is a subject
//! owing work again once it is asked for; the assertions below are those two
//! and the count in between.
//!
//! Run with `cargo test -p pamin-store --test jobsettle -- --ignored`.

use pamin_core::{JobKind, ProjectId};
use pamin_store::{Connections, Database, Workspace, jobs};

/// Every row the queue holds for this project, owed or not.
async fn rows(database: &Database, project: ProjectId) -> i64 {
    sqlx::query_scalar("SELECT count(*) FROM index_jobs WHERE project_id = $1")
        .bind(project.0)
        .fetch_one(database.pool())
        .await
        .expect("count the rows")
}

#[tokio::test]
#[ignore = "needs a real postgres cluster"]
async fn finished_work_leaves_no_row_and_live_work_stays() {
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    let project = pamin_store::repository::ensure_project(
        database.pool(),
        &format!("jobsettle-{}", uuid::Uuid::new_v4()),
    )
    .await
    .expect("ensure project")
    .id;

    // Three subjects owe work. Two will be finished, one will not.
    let subjects: Vec<uuid::Uuid> = (0..3).map(|_| uuid::Uuid::new_v4()).collect();
    for subject in &subjects {
        jobs::enqueue(
            database.pool(),
            project,
            JobKind::SyncTopicIndex,
            Some(*subject),
        )
        .await
        .expect("enqueue");
    }
    assert_eq!(rows(&database, project).await, 3, "three owed");

    // Finished the way a worker finishes them, through claim and complete, so
    // the rows carry a real claim for the completion to match.
    let claimed = jobs::claim(
        database.pool(),
        project,
        "jobsettle",
        2,
        &[JobKind::SyncTopicIndex],
    )
    .await
    .expect("claim");
    assert_eq!(claimed.len(), 2, "two claimed");
    let held: Vec<&jobs::Job> = claimed.iter().collect();
    let completed = jobs::complete(database.pool(), &held, "jobsettle")
        .await
        .expect("complete");
    assert_eq!(completed.len(), 2, "both claims were still held");

    // The whole change: finished work is gone, not kept for somebody to read.
    assert_eq!(
        rows(&database, project).await,
        1,
        "a completed job should leave no row behind"
    );
    assert_eq!(
        jobs::pending(database.pool(), project)
            .await
            .expect("pending"),
        1,
        "and the one never finished is still owed"
    );

    // A worker that lost its claim completes nothing, so a row cannot be
    // deleted out from under the worker that holds it now.
    let mut stale = claimed[0].clone();
    stale.id = pamin_core::IndexJobId(
        sqlx::query_scalar("SELECT id FROM index_jobs WHERE project_id = $1")
            .bind(project.0)
            .fetch_one(database.pool())
            .await
            .expect("the remaining row"),
    );
    let none = jobs::complete(database.pool(), &[&stale], "jobsettle")
        .await
        .expect("complete a job nobody holds");
    assert!(
        none.is_empty(),
        "a job this worker never claimed was deleted"
    );
    assert_eq!(rows(&database, project).await, 1, "live work must survive");

    // And the subject owes work again as soon as it is asked for: the insert
    // takes the branch a revival used to.
    jobs::enqueue(
        database.pool(),
        project,
        JobKind::SyncTopicIndex,
        Some(subjects[0]),
    )
    .await
    .expect("re-enqueue a finished subject");
    assert_eq!(
        jobs::pending(database.pool(), project)
            .await
            .expect("pending"),
        2,
        "a finished subject can owe work again"
    );
}
