//! `pamin-memory cascade` — run and inspect the work a write left behind.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Serialize, Deserialize, JsonSchema)]
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

#[derive(Serialize, Deserialize, JsonSchema)]
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

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Segments {
    pub holds: u64,
    pub wants: u64,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Failure {
    pub job: String,
    pub subject: Option<String>,
    pub attempts: i32,
    pub error: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Failures {
    pub failed: Vec<Failure>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Moved {
    pub jobs: u64,
}

/// What a drain did, and what shape it left the index in if that is worth
/// saying.
pub fn render_drained(result: &Drained) -> String {
    let mut rendered = format!(
        "Ran {} jobs, {} failed, {} still owed",
        result.completed, result.failed, result.pending
    );
    if let Some(segments) = &result.segments {
        rendered.push_str(&format!(
            "\nThis index is spread over {} segments where {} would do, because it \
             recorded its segment size when it was empty. Searches pay for the extra \
             segments; the server reshapes it in the background, copying what it holds \
             rather than embedding it again; `pamin-memory reindex` rebuilds it now.",
            segments.holds, segments.wants
        ));
    }
    rendered
}

/// Renders the failed jobs for a person reading them.
pub fn render_failures(result: &Failures) -> String {
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
