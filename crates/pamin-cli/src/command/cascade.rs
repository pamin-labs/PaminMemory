//! `pamin cascade` — run and inspect the work a write left behind.
//!
//! A write commits what the projection owes and stops there, so something has
//! to pay it. Ordinarily that is the drain at the end of `pamin write`, and
//! these commands are for the cases it does not cover: a queue left behind by a
//! process that died, work deferred because the index was unreachable, and
//! jobs that failed often enough to be set aside for a person to look at.

use anyhow::Result;
use pamin_index::{Access, Profile};
use pamin_store::{Database, Workspace, jobs, repository};
use serde::Serialize;

use crate::output::Format;
use pamin_engine::Engine;

/// How long `run` waits before looking again when it finds nothing.
///
/// There is no wake-up signal between processes -- `LISTEN`/`NOTIFY` is outside
/// the portable subset this store holds itself to -- so an idle worker polls.
/// A quarter of a second is short enough to feel immediate and long enough that
/// an idle worker is not a load.
const IDLE: std::time::Duration = std::time::Duration::from_millis(250);

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand)]
pub enum Command {
    /// Run every job that is due, then stop.
    Drain,

    /// Keep running jobs as they arrive, until interrupted.
    Run,

    /// List the jobs that used their attempts, with the error that stopped them.
    Failed,

    /// Make the failed jobs due again, for when what broke them is fixed.
    Replay,

    /// Abandon the failed jobs.
    ///
    /// The work is dropped, not the record: a later write to the same subject
    /// queues it again rather than being coalesced onto a row nothing will run.
    Discard,
}

#[derive(Serialize)]
struct Drained {
    completed: usize,
    failed: usize,
    /// Jobs still owed, including any not yet due.
    pending: i64,
}

#[derive(Serialize)]
struct Failure {
    job: String,
    subject: Option<String>,
    attempts: i32,
    error: String,
}

#[derive(Serialize)]
struct Failures {
    failed: Vec<Failure>,
}

#[derive(Serialize)]
struct Moved {
    jobs: u64,
}

pub async fn run(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
    args: Args,
) -> Result<()> {
    match args.command {
        Command::Drain => drain(workspace, project, profile, format).await,
        Command::Run => keep_running(workspace, project, profile).await,
        Command::Failed => failed(workspace, project, format).await,
        Command::Replay => replay(workspace, project, format).await,
        Command::Discard => discard(workspace, project, format).await,
    }
}

async fn drain(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
) -> Result<()> {
    let mut engine = Engine::open(workspace, project, profile, Access::ReadWrite).await?;
    let drained = engine.drain_cascade().await?;

    let result = Drained {
        completed: drained.completed,
        failed: drained.failed,
        pending: drained.pending,
    };
    format.emit(&result, || {
        format!(
            "Ran {} jobs, {} failed, {} still owed",
            result.completed, result.failed, result.pending
        )
    });
    Ok(())
}

/// Drains, then waits, then drains again, for as long as it is left running.
///
/// Holds the index open for writing the whole time, which is the point: this is
/// the shape a worker has before there is a server to hold it, and the reason
/// it cannot run beside a `pamin write` in another terminal.
async fn keep_running(workspace: &Workspace, project: &str, profile: Profile) -> Result<()> {
    let mut engine = Engine::open(workspace, project, profile, Access::ReadWrite).await?;

    loop {
        let drained = engine.drain_cascade().await?;
        if drained.completed > 0 || drained.failed > 0 {
            tracing::info!(
                completed = drained.completed,
                failed = drained.failed,
                pending = drained.pending,
                "cascade round"
            );
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(()),
            _ = tokio::time::sleep(IDLE) => {}
        }
    }
}

async fn failed(workspace: &Workspace, project: &str, format: Format) -> Result<()> {
    // Straight to the ledger: listing what failed should not load a model or
    // take the index's exclusive lock, and it must work while a worker holds
    // both.
    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;
    let exhausted = jobs::exhausted(database.pool(), project.id).await?;

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
    format.emit(&result, || {
        if result.failed.is_empty() {
            return "Nothing has failed".to_string();
        }
        result
            .failed
            .iter()
            .map(|failure| {
                format!(
                    "{} {} after {} attempts: {}",
                    failure.job,
                    failure.subject.as_deref().unwrap_or("(project)"),
                    failure.attempts,
                    failure.error
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(())
}

async fn replay(workspace: &Workspace, project: &str, format: Format) -> Result<()> {
    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;
    let revived = jobs::replay(database.pool(), project.id).await?;

    let result = Moved { jobs: revived };
    format.emit(&result, || {
        format!("Queued {} failed jobs to run again", result.jobs)
    });
    Ok(())
}

async fn discard(workspace: &Workspace, project: &str, format: Format) -> Result<()> {
    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;
    let discarded = jobs::discard(database.pool(), project.id).await?;

    let result = Moved { jobs: discarded };
    format.emit(&result, || format!("Abandoned {} failed jobs", result.jobs));
    Ok(())
}
