//! `pamin sync` — list the paths the server keeps in sync with their files.

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {}

#[derive(Serialize, Deserialize)]
pub struct Synced {
    pub paths: Vec<SyncedPath>,
}

/// A path an ingest was given, which the server keeps in sync.
#[derive(Serialize, Deserialize)]
pub struct SyncedPath {
    pub path: String,
    /// How the server is following it: `watching`, `polling` when the
    /// operating system would not watch it any more, `missing` when it is not
    /// there, or `starting` until the server has caught up on it.
    pub state: String,
    /// When the server last finished ingesting what changed under it.
    pub synced_at: Option<String>,
    /// Why the last ingest of it failed, if it did.
    pub error: Option<String>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Synced) -> String {
    if result.paths.is_empty() {
        return "No paths are kept in sync: `pamin ingest` keeps what it is given".to_string();
    }
    let mut out = format!("{} paths kept in sync:", result.paths.len());
    for synced in &result.paths {
        out.push_str(&format!("\n  {} ({}", synced.path, synced.state));
        if let Some(at) = &synced.synced_at {
            out.push_str(&format!(", synced {at}"));
        }
        out.push(')');
        if let Some(error) = &synced.error {
            out.push_str(&format!("\n    last ingest failed: {error}"));
        }
    }
    out
}
