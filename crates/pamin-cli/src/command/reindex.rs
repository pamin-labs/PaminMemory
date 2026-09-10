//! `pamin reindex` — rebuild the projection index from PostgreSQL.
//!
//! This command is the guarantee that makes a pre-1.0 index engine acceptable:
//! whatever the index does, the authority store can reproduce it. It exists
//! from the first release so the guarantee is executable rather than stated.

use anyhow::Result;
use pamin_index::Profile;

use serde::{Deserialize, Serialize};

use crate::session::Session;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {}

#[derive(Serialize, Deserialize)]
pub struct Reindexed {
    indexed: usize,
    /// Topics whose current-state pointer disagreed with the ledger and was
    /// corrected. Zero unless something stopped maintaining it.
    repaired_pointers: u64,
}

pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
    _args: Args,
) -> Result<Reindexed> {
    // Rebuilding discards this project's index first, and clears the shared
    // pre-split layout if the workspace still has one.
    let engine = session.rebuilding(project, profile).await?;
    let rebuilt = engine.reindex().await?;

    let result = Reindexed {
        indexed: rebuilt.indexed,
        repaired_pointers: rebuilt.repaired_pointers,
    };
    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(result: &Reindexed) -> String {
    let mut rendered = format!("Rebuilt the index from postgres: {} states", result.indexed);
    if result.repaired_pointers > 0 {
        rendered.push_str(&format!(
            "\nRepaired {} topics pointing at the wrong current state",
            result.repaired_pointers
        ));
    }
    rendered
}
