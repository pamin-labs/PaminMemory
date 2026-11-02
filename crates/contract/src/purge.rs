//! `pamin-memory purge` — delete a project and everything it holds, for good.
//!
//! For privacy requests. A retraction (`unlink --reason deleted`, a write that
//! supersedes) keeps history; this keeps nothing: the project's rows and its
//! index directory are gone, and no query at any instant finds them. Not an
//! MCP tool, so a model cannot reach it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Delete the project and everything it holds. It cannot be undone.
#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {
    /// Confirm that the project is to be deleted for good. Refused without it.
    #[arg(long)]
    #[serde(default)]
    pub yes: bool,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Purged {
    /// Whether there was a project of that name to purge.
    pub existed: bool,
    /// Topics it held, every version of each now gone.
    pub topics: u64,
    /// Sources it held, with all of their evidence.
    pub sources: u64,
    /// Paths that were kept in sync and are no longer watched.
    pub unsynced: Vec<String>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Purged) -> String {
    if !result.existed {
        return "No such project; nothing was purged".to_string();
    }
    let mut rendered = format!(
        "Purged the project: {} topics and {} sources deleted, with their index",
        result.topics, result.sources
    );
    if !result.unsynced.is_empty() {
        rendered.push_str(&format!(
            "\nNo longer syncing {}",
            result.unsynced.join(", ")
        ));
    }
    rendered
}
