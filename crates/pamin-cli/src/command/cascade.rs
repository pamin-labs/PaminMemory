//! `pamin cascade` — run and inspect the work a write left behind.
//!
//! A write commits what the projection owes and stops there, so something has
//! to pay it. Ordinarily that is the drain at the end of `pamin write`, and
//! what that leaves -- deferred writes, a queue a process that died left
//! behind -- the server catches up on between requests. These commands are
//! for making the index catch up at once, and for jobs that failed often
//! enough to be set aside for a person to look at.

use anyhow::Result;
use pamin_client::command::cascade::{Args, Command, Drained, Failure, Failures, Moved, Segments};
use pamin_engine::Owed;
use pamin_index::VectorIndex;
use pamin_store::jobs;

use crate::session::Session;

/// Runs one subcommand and returns its result as JSON.
///
/// The subcommands answer with different types, so the server cannot hand back
/// one struct the way every other command does; it hands back the JSON each of
/// them would have printed.
pub async fn answer(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<serde_json::Value> {
    let value = match args.command {
        Command::Drain => serde_json::to_value(drain(session, project, vector_index).await?)?,
        Command::Failed => serde_json::to_value(failed(session, project).await?)?,
        Command::Replay => serde_json::to_value(replay(session, project).await?)?,
        Command::Discard => serde_json::to_value(discard(session, project).await?)?,
    };

    Ok(value)
}

pub async fn drain(session: &Session, project: &str, vector_index: VectorIndex) -> Result<Drained> {
    let engine = session.engine(project, vector_index).await?;
    let drained = engine.drain_cascade(Owed::Everything).await?;

    let shape = engine.segmentation()?;
    Ok(Drained {
        completed: drained.completed,
        failed: drained.failed,
        // Counted in full: the drain's own figure stops at the lag bound,
        // which is all a write needs and less than this command reports.
        pending: jobs::pending(engine.database.pool(), engine.project).await?,
        segments: shape.is_worth_rebuilding().then(|| Segments {
            holds: shape.segments(),
            wants: shape.wanted(),
        }),
    })
}

pub async fn failed(session: &Session, project: &str) -> Result<Failures> {
    // Straight to the ledger: listing what failed should not load a model or
    // take the index's exclusive lock, and it must work while a worker holds
    // both.
    let project = session.project(project).await?;
    let exhausted = jobs::exhausted(session.database().pool(), project).await?;

    let result = Failures {
        failed: exhausted
            .iter()
            .map(|(job, error)| Failure {
                job: job.kind.to_string(),
                subject: job.subject.map(|subject| subject.to_string()),
                attempts: job.attempts,
                error: error.clone(),
            })
            .collect(),
    };
    Ok(result)
}

pub async fn replay(session: &Session, project: &str) -> Result<Moved> {
    let project = session.project(project).await?;
    let revived = jobs::replay(session.database().pool(), project).await?;

    Ok(Moved { jobs: revived })
}

pub async fn discard(session: &Session, project: &str) -> Result<Moved> {
    let project = session.project(project).await?;
    let discarded = jobs::discard(session.database().pool(), project).await?;

    Ok(Moved { jobs: discarded })
}
