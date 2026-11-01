//! `pamin unsync` — stop keeping ingested paths in sync with their files.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// Paths an ingest was given, as `pamin sync` lists them. Made absolute
    /// here, as `ingest` makes its paths.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
}

#[derive(Serialize, Deserialize)]
pub struct Unsynced {
    /// The paths no longer kept in sync. Their sections stay as they are.
    pub removed: Vec<String>,
    /// The paths that were not kept in sync to begin with.
    pub unknown: Vec<String>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Unsynced) -> String {
    let mut lines = Vec::new();
    for path in &result.removed {
        lines.push(format!("No longer kept in sync: {path}"));
    }
    for path in &result.unknown {
        lines.push(format!("Not kept in sync: {path}"));
    }
    lines.join("\n")
}
