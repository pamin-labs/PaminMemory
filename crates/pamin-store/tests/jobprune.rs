//! What the outbox keeps, and what it lets go of.
//!
//! Settled jobs used to be kept forever, on a reason that was real: `enqueue`
//! upserts on the idempotency key, so a completed row is revived by the next
//! write to the same subject rather than joined by a second row saying the
//! same thing. Keeping it made the queue the largest object in the database --
//! measured on a workspace of 13,014 documents, `index_jobs` held 39,043 rows,
//! every one of them completed, at 14 MB of heap and 12 MB of indexes. That is
//! 26 MB of a 62 MB database, none of it reachable work, over content that
//! came to 3.2 MB.
//!
//! Deleting a settled row is equivalent rather than a behaviour change,
//! because the insert then takes its other branch and produces a row in the
//! same state. What must not happen is a row being deleted out from under a
//! worker, and that is what the second assertion is for.
//!
//! Run with `cargo test -p pamin-store --test jobprune -- --ignored`.

use pamin_core::{JobKind, ProjectId};
use pamin_store::{Connections, Database, Workspace, jobs};

/// Backdates every settled row so the retention window has something to act
/// on. The window is an hour and a test cannot wait for it.
async fn settle_into_the_past(database: &Database, project: ProjectId) {
    sqlx::query(
        "UPDATE index_jobs SET completed_at = now() - interval '2 hours'
          WHERE project_id = $1 AND completed_at IS NOT NULL",
    )
    .bind(project.0)
    .execute(database.pool())
    .await
    .expect("backdate the settled rows");
}

async fn count(database: &Database, project: ProjectId, settled: bool) -> i64 {
    let sql = if settled {
        "SELECT count(*) FROM index_jobs WHERE project_id = $1 AND completed_at IS NOT NULL"
    } else {
        "SELECT count(*) FROM index_jobs WHERE project_id = $1 AND completed_at IS NULL"
    };
    sqlx::query_scalar(sql)
        .bind(project.0)
        .fetch_one(database.pool())
        .await
        .expect("count the rows")
}

#[tokio::test]
#[ignore = "needs a real postgres cluster"]
async fn pruning_takes_settled_work_and_leaves_live_work() {
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
        &format!("jobprune-{}", uuid::Uuid::new_v4()),
    )
    .await
    .expect("ensure project")
    .id;

    // Three subjects owe work. Two will be settled, one will not.
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
    assert_eq!(count(&database, project, false).await, 3, "three owed");

    // Settle two of them the way a worker does, through claim and complete, so
    // the rows carry a real claim rather than a hand-written completion.
    let claimed = jobs::claim(
        database.pool(),
        project,
        "jobprune",
        2,
        &[JobKind::SyncTopicIndex],
    )
    .await
    .expect("claim");
    assert_eq!(claimed.len(), 2, "two claimed");
    let held: Vec<&jobs::Job> = claimed.iter().collect();
    jobs::complete(database.pool(), &held, "jobprune")
        .await
        .expect("complete");

    assert_eq!(count(&database, project, true).await, 2, "two settled");
    assert_eq!(count(&database, project, false).await, 1, "one still owed");

    // Inside the window, nothing goes: `cascade status` still has to be able
    // to show what the drain just did.
    let fresh = jobs::prune(database.pool(), project)
        .await
        .expect("prune inside the window");
    assert_eq!(fresh, 0, "a job settled a moment ago is not old enough");
    assert_eq!(count(&database, project, true).await, 2);

    settle_into_the_past(&database, project).await;

    let pruned = jobs::prune(database.pool(), project)
        .await
        .expect("prune past the window");
    assert_eq!(pruned, 2, "both settled rows go");
    assert_eq!(count(&database, project, true).await, 0);

    // The one that was never settled is untouched. A claimed job has no
    // completion, so this is also what stops a row being deleted out from
    // under a worker mid-flight.
    assert_eq!(
        count(&database, project, false).await,
        1,
        "live work must survive a prune"
    );

    // And the whole point of keeping the rows was that `enqueue` revives
    // them. After a prune the insert takes its other branch instead, and the
    // subject owes work again exactly as it would have.
    jobs::enqueue(
        database.pool(),
        project,
        JobKind::SyncTopicIndex,
        Some(subjects[0]),
    )
    .await
    .expect("re-enqueue a pruned subject");
    assert_eq!(
        count(&database, project, false).await,
        2,
        "a pruned subject can owe work again"
    );
}
