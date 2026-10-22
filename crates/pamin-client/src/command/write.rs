//! `pamin write` — record a memory.

use serde::{Deserialize, Serialize};

use crate::command::validity;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic this memory belongs to.
    #[arg(long)]
    pub topic: String,

    /// The memory content. Reads standard input when omitted.
    pub content: Option<String>,

    /// Record the memory without waiting for the index to catch up.
    ///
    /// The work is queued rather than skipped. The server runs it once no
    /// request is being answered, and `pamin cascade drain` runs it at once.
    #[arg(long)]
    pub defer: bool,

    #[command(flatten)]
    pub validity: validity::Flags,
}

#[derive(Serialize, Deserialize)]
pub struct Written {
    pub topic: String,
    /// Absent when the filter held the content in the evidence layer.
    pub version: Option<u32>,
    pub promoted: bool,
    /// Why the filter decided as it did, promoted or not.
    pub reason: String,
    /// Always set: evidence is recorded whatever the filter decides.
    pub source_version: u32,
    /// Whether the projection caught up before this command returned, or the
    /// work is still owed. Either way the memory is recorded.
    pub cascade: String,
    /// Set when the projection has fallen far enough behind to say so. The
    /// queue is unbounded on purpose -- a write must not fail because the
    /// index is slow -- but a backlog nobody reports looks like searches
    /// quietly missing the newest memories.
    pub cascade_lagging: bool,
    /// The truth interval this state was asserted for, if one was given.
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Written) -> String {
    match result.version {
        Some(version) => format!("Wrote {} v{}", result.topic, version),
        None => format!(
            "Held in evidence only: {}\nStored as {} source version {}",
            result.reason, result.topic, result.source_version
        ),
    }
}
