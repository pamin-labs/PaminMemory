//! `pamin cascade` — run and inspect the work a write left behind.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::output::Format;

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
    pub completed: usize,
    pub failed: usize,
    /// Jobs still owed, including any not yet due.
    pub pending: i64,
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
    pub segments: Option<Segments>,
}

#[derive(Serialize, Deserialize)]
pub struct Segments {
    pub holds: u64,
    pub wants: u64,
}

#[derive(Serialize, Deserialize)]
pub struct Failure {
    pub job: String,
    pub subject: Option<String>,
    pub attempts: i32,
    pub error: String,
}

#[derive(Serialize, Deserialize)]
pub struct Failures {
    pub failed: Vec<Failure>,
}

#[derive(Serialize, Deserialize)]
pub struct Moved {
    pub jobs: u64,
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
    format: Format,
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
