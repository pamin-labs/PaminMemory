//! `pamin cascade` — run and inspect the work a write left behind.
//!
//! A write commits what the projection owes and stops there, so something has
//! to pay it. Ordinarily that is the drain at the end of `pamin write`, and
//! these commands are for the cases it does not cover: a queue left behind by a
//! process that died, work deferred because the index was unreachable, and
//! jobs that failed often enough to be set aside for a person to look at.

use anyhow::Result;
use pamin_engine::Owed;
use pamin_index::Profile;
use pamin_store::jobs;
use serde::{Deserialize, Serialize};

use crate::output::Format;
use crate::session::Session;

/// How long `run` waits before looking again when it finds nothing.
///
/// There is no wake-up signal between processes -- `LISTEN`/`NOTIFY` is outside
/// the portable subset this store holds itself to -- so an idle worker polls.
/// A quarter of a second is short enough to feel immediate and long enough that
/// an idle worker is not a load.
const IDLE: std::time::Duration = std::time::Duration::from_millis(250);

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
pub struct Drained {
    completed: usize,
    failed: usize,
    /// Jobs still owed, including any not yet due.
    pending: i64,
    /// Segments the index holds, and how many its own policy wants, when the
    /// two differ enough to be worth a rebuild.
    ///
    /// Absent otherwise, so a healthy index says nothing. Here rather than in
    /// a command of its own because whoever has just drained the cascade is
    /// whoever cares what shape the index is in, and this is the only place
    /// the answer was reachable from: a collection records its segment size
    /// when it is created, a workspace is created empty, and nothing has ever
    /// told anyone what that left them with.
    #[serde(skip_serializing_if = "Option::is_none")]
    segments: Option<Segments>,
}

#[derive(Serialize, Deserialize)]
pub struct Segments {
    holds: u64,
    wants: u64,
}

#[derive(Serialize, Deserialize)]
pub struct Failure {
    job: String,
    subject: Option<String>,
    attempts: i32,
    error: String,
}

#[derive(Serialize, Deserialize)]
pub struct Failures {
    failed: Vec<Failure>,
}

#[derive(Serialize, Deserialize)]
pub struct Moved {
    jobs: u64,
}

/// Runs one subcommand and returns its result as JSON.
///
/// The subcommands answer with different types, so the server cannot hand back
/// one struct the way every other command does; it hands back the JSON each of
/// them would have printed. `run` is the exception inside the exception -- a
/// foreground loop with no result -- and a client that asks a server for it
/// gets told to run it itself, because the loop belongs to the process that
/// wants to hold the index, and that is the server already.
pub async fn answer(
    session: &Session,
    project: &str,
    profile: Profile,
    args: Args,
) -> Result<serde_json::Value> {
    let value = match args.command {
        Command::Drain => serde_json::to_value(drain(session, project, profile).await?)?,
        Command::Failed => serde_json::to_value(failed(session, project).await?)?,
        Command::Replay => serde_json::to_value(replay(session, project).await?)?,
        Command::Discard => serde_json::to_value(discard(session, project).await?)?,
        Command::Run => anyhow::bail!(
            "`pamin cascade run` holds the index for as long as it runs, so it cannot be \
             served by the process already holding it; run it against a workspace with \
             PAMIN_NO_SERVER=1, or let the server drain on its own"
        ),
    };

    Ok(value)
}

/// Prints a subcommand's result, given the request that produced it.
///
/// Which type the JSON is depends on which subcommand was asked for, so the
/// request has to be in hand to read the response. That is the cost of one
/// command answering with four shapes, and it is paid here rather than by
/// flattening them into one shape nobody wanted.
pub fn render_value(
    args: &Args,
    value: &serde_json::value::RawValue,
    format: crate::output::Format,
) -> Result<()> {
    match args.command {
        Command::Drain => {
            let result: Drained = serde_json::from_str(value.get())?;
            format.emit(&result, || render_drained(&result, Served::Yes));
        }
        Command::Failed => {
            let result: Failures = serde_json::from_str(value.get())?;
            format.emit(&result, || render_failures(&result));
        }
        Command::Replay => {
            let result: Moved = serde_json::from_str(value.get())?;
            format.emit(&result, || {
                format!("Queued {} failed jobs to run again", result.jobs)
            });
        }
        Command::Discard => {
            let result: Moved = serde_json::from_str(value.get())?;
            format.emit(&result, || format!("Abandoned {} failed jobs", result.jobs));
        }
        Command::Run => unreachable!("the server refuses `run` rather than answering it"),
    }

    Ok(())
}

/// Runs one of the subcommands and prints it.
///
/// The only command that still owns its own printing, because `run` is the one
/// that has nothing to print: it is a foreground loop rather than a request
/// with an answer.
pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
    format: Format,
    args: Args,
) -> Result<()> {
    match args.command {
        Command::Drain => {
            let result = drain(session, project, profile).await?;
            format.emit(&result, || render_drained(&result, Served::No));
        }
        Command::Run => keep_running(session, project, profile).await?,
        Command::Failed => {
            let result = failed(session, project).await?;
            format.emit(&result, || render_failures(&result));
        }
        Command::Replay => {
            let result = replay(session, project).await?;
            format.emit(&result, || {
                format!("Queued {} failed jobs to run again", result.jobs)
            });
        }
        Command::Discard => {
            let result = discard(session, project).await?;
            format.emit(&result, || format!("Abandoned {} failed jobs", result.jobs));
        }
    }
    Ok(())
}

/// Whether a server answered, which decides what fixes a badly shaped index.
#[derive(Clone, Copy)]
enum Served {
    Yes,
    No,
}

/// What a drain did, and what shape it left the index in if that is worth
/// saying.
///
/// One rendering for both paths. The served one used to be the only one that
/// mentioned the shape, so the same drain said less without a server -- where
/// nothing reshapes the index on its own and the advice mattered most.
fn render_drained(result: &Drained, served: Served) -> String {
    let mut rendered = format!(
        "Ran {} jobs, {} failed, {} still owed",
        result.completed, result.failed, result.pending
    );
    if let Some(segments) = &result.segments {
        let fix = match served {
            Served::Yes => {
                "the server reshapes it in the background, copying what it holds \
                 rather than embedding it again; `pamin reindex` rebuilds it now"
            }
            Served::No => {
                "`pamin reindex` rebuilds it at the right size, and a running server \
                 reshapes it in the background on its own"
            }
        };
        rendered.push_str(&format!(
            "\nThis index is spread over {} segments where {} would do, because it \
             recorded its segment size when it was empty. Searches pay for the extra \
             segments; {fix}.",
            segments.holds, segments.wants
        ));
    }
    rendered
}

pub async fn drain(session: &Session, project: &str, profile: Profile) -> Result<Drained> {
    let engine = session.engine(project, profile).await?;
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

/// Drains, then waits, then drains again, for as long as it is left running.
///
/// Holds the index open for writing the whole time, which is the point: this is
/// the shape a worker has before there is a server to hold it, and the reason
/// it cannot run beside a `pamin write` in another terminal.
async fn keep_running(session: &Session, project: &str, profile: Profile) -> Result<()> {
    let engine = session.engine(project, profile).await?;

    loop {
        let drained = engine.drain_cascade(Owed::Everything).await?;
        if drained.completed > 0 || drained.failed > 0 {
            tracing::info!(
                completed = drained.completed,
                failed = drained.failed,
                pending = jobs::pending(engine.database.pool(), engine.project).await?,
                "cascade round"
            );
        }

        tokio::select! {
            _ = tokio::signal::ctrl_c() => return Ok(()),
            _ = tokio::time::sleep(IDLE) => {}
        }
    }
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

/// Renders the failed jobs for a person reading them.
fn render_failures(result: &Failures) -> String {
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
