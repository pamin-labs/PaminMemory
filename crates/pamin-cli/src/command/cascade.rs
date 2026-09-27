//! `pamin cascade` — run and inspect the work a write left behind.
//!
//! A write commits what the projection owes and stops there, so something has
//! to pay it. Ordinarily that is the drain at the end of `pamin write`, and
//! what that leaves -- deferred writes, a queue a process that died left
//! behind -- the server catches up on between requests. These commands are
//! for making the index catch up at once, and for jobs that failed often
//! enough to be set aside for a person to look at.

use anyhow::Result;
use pamin_engine::Owed;
use pamin_index::{Profile, VectorIndex};
use pamin_store::jobs;
use serde::{Deserialize, Serialize};

use crate::session::Session;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Serialize, Deserialize)]
pub enum Command {
    /// Run every job that is due, then stop.
    Drain,

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
/// them would have printed.
pub async fn answer(
    session: &Session,
    project: &str,
    profile: Profile,
    vector_index: VectorIndex,
    args: Args,
) -> Result<serde_json::Value> {
    let value = match args.command {
        Command::Drain => {
            serde_json::to_value(drain(session, project, profile, vector_index).await?)?
        }
        Command::Failed => serde_json::to_value(failed(session, project).await?)?,
        Command::Replay => serde_json::to_value(replay(session, project).await?)?,
        Command::Discard => serde_json::to_value(discard(session, project).await?)?,
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
            format.emit(&result, || render_drained(&result));
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
    }

    Ok(())
}

/// What a drain did, and what shape it left the index in if that is worth
/// saying.
fn render_drained(result: &Drained) -> String {
    let mut rendered = format!(
        "Ran {} jobs, {} failed, {} still owed",
        result.completed, result.failed, result.pending
    );
    if let Some(segments) = &result.segments {
        rendered.push_str(&format!(
            "\nThis index is spread over {} segments where {} would do, because it \
             recorded its segment size when it was empty. Searches pay for the extra \
             segments; the server reshapes it in the background, copying what it holds \
             rather than embedding it again; `pamin reindex` rebuilds it now.",
            segments.holds, segments.wants
        ));
    }
    rendered
}

pub async fn drain(
    session: &Session,
    project: &str,
    profile: Profile,
    vector_index: VectorIndex,
) -> Result<Drained> {
    let engine = session.engine(project, profile, vector_index).await?;
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
