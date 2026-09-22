//! Whether a drain with nothing to do still clears the settled queue.
//!
//! Ignored by default: it provisions PostgreSQL. Run with
//! `cargo test -p pamin-engine --test jobretention -- --ignored --nocapture`.
//!
//! `jobs::prune` has existed and worked for a while, and `pamin-store`'s own
//! `jobprune` test covers it thoroughly. What nothing covered is the condition
//! it was called under, and that condition was wrong in the way that is worst
//! for a queue: `drain_cascade` pruned only when the drain had completed
//! something, so it skipped the cleanup in exactly the state that needs it.
//!
//! A workspace that finishes importing and goes quiet has nothing left to
//! claim. Every drain after the last write therefore took the cheap path and
//! left `index_jobs` at its high-water mark, indefinitely -- and "indefinitely"
//! is not a figure of speech. Measured on the evaluation workspace this project
//! runs its corpora in: `index_jobs` at 632 MB of a 1.7 GB database, 38.7% of
//! it, holding 651,128 settled rows out of 1,054,646, completed the previous
//! day and pruned by nothing since, because nothing had been written since.
//!
//! So the test is about the *empty* drain, which is the one the guard hid. It
//! settles rows by hand rather than by writing memories, because that keeps the
//! embedding model out of a test about a `DELETE`, and because the drain under
//! test has to find nothing to claim -- which is easier to arrange when nothing
//! was ever claimed by the engine in the first place.

use pamin_core::{JobKind, ProjectId};
use pamin_engine::{Engine, Owed};
use pamin_index::{Access, Profile};
use pamin_store::{Connections, Database, Workspace, jobs, repository};

/// The default profile. Nothing here embeds anything, but the engine opens a
/// projection for one and it should be the shipped one.
const PROFILE: &str = "accuracy";

/// Enough rows that a count is unambiguous, few enough that the setup is fast.
const JOBS: usize = 8;

async fn settled(database: &Database, project: ProjectId) -> i64 {
    sqlx::query_scalar(
        "SELECT count(*) FROM index_jobs WHERE project_id = $1 AND completed_at IS NOT NULL",
    )
    .bind(project.0)
    .fetch_one(database.pool())
    .await
    .expect("count the settled rows")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn a_drain_with_nothing_to_do_still_clears_the_settled_queue() {
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    let profile = Profile::parse(PROFILE).expect("a known profile");
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    // A name nothing else uses, so the counts below are this test's rows and
    // not whatever a previous run left in a shared workspace.
    let name = format!("jobretention-{}", uuid::Uuid::new_v4());
    let project = repository::ensure_project(database.pool(), &name)
        .await
        .expect("ensure project")
        .id;

    let engine = Engine::open(&workspace, &name, profile, Access::ReadWrite)
        .await
        .expect("open the engine");

    // Enqueued, claimed and completed without the engine, so the drain below
    // is the first thing the engine does to this queue and has nothing left to
    // claim.
    for _ in 0..JOBS {
        jobs::enqueue(
            database.pool(),
            project,
            JobKind::SyncTopicIndex,
            Some(uuid::Uuid::new_v4()),
        )
        .await
        .expect("enqueue");
    }
    let claimed = jobs::claim(
        database.pool(),
        project,
        "jobretention",
        JOBS as i32,
        &[JobKind::SyncTopicIndex],
    )
    .await
    .expect("claim");
    assert_eq!(
        claimed.len(),
        JOBS,
        "the setup did not claim what it queued"
    );
    let held: Vec<&jobs::Job> = claimed.iter().collect();
    jobs::complete(database.pool(), &held, "jobretention")
        .await
        .expect("complete");
    assert_eq!(settled(&database, project).await, JOBS as i64);

    // Past the retention window, which is what makes them eligible. Inside it
    // they are deliberately kept, so a test that skipped this would be
    // asserting that the window does not work.
    sqlx::query(
        "UPDATE index_jobs SET completed_at = now() - interval '2 hours'
          WHERE project_id = $1 AND completed_at IS NOT NULL",
    )
    .bind(project.0)
    .execute(database.pool())
    .await
    .expect("backdate the settled rows");

    let drained = engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain the cascade");

    // The premise. If this drain completed anything then it is not the drain
    // under test: the old guard would have pruned too, and the test would pass
    // while the defect it is named after remained.
    assert_eq!(
        drained.completed, 0,
        "this drain found work to do, so it does not exercise the empty case"
    );

    assert_eq!(
        settled(&database, project).await,
        0,
        "a drain with nothing to claim left {JOBS} settled rows behind, which is how \
         `index_jobs` became 38.7% of a 1.7 GB database"
    );
}
