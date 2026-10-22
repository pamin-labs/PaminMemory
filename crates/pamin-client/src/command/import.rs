//! `pamin import` — record many memories in one call.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::command::validity;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// A file of memories, one JSON object per line: `{"topic": …, "content": …}`.
    ///
    /// Read by whichever process holds the workspace, which is the server when
    /// one is running. Both are on this machine and run as you.
    #[arg(long)]
    pub from: PathBuf,

    #[command(flatten)]
    pub validity: validity::Flags,
}

#[derive(Serialize, Deserialize)]
pub struct Imported {
    /// Memories read from the file.
    pub memories: usize,
    /// Of those, the ones the filter put on the retrieval surface.
    pub promoted: usize,
    /// The rest: recorded as evidence, not indexed. Re-importing the same file
    /// is how this fills up, and it is the right answer rather than an error.
    pub held: usize,
    /// Whether the projection caught up before this returned.
    pub cascade: String,
    /// Set when the projection was behind at any point during the import.
    pub cascade_lagging: bool,
}

/// Renders the result for a person reading it.
pub fn render(result: &Imported) -> String {
    format!(
        "Imported {} memories: {} written, {} held in evidence only",
        result.memories, result.promoted, result.held
    )
}
