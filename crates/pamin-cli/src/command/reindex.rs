//! `pamin reindex` — rebuild the projection index from PostgreSQL.
//!
//! This command is the guarantee that makes a pre-1.0 index engine acceptable:
//! whatever the index does, the authority store can reproduce it. It exists
//! from the first release so the guarantee is executable rather than stated.

use anyhow::Result;
use pamin_index::Profile;
use pamin_store::Workspace;
use serde::Serialize;

use crate::output::Format;
use pamin_engine::Engine;

#[derive(clap::Args)]
pub struct Args {}

#[derive(Serialize)]
struct Reindexed {
    indexed: usize,
    /// Topics whose current-state pointer disagreed with the ledger and was
    /// corrected. Zero unless something stopped maintaining it.
    repaired_pointers: u64,
}

pub async fn run(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
    _args: Args,
) -> Result<()> {
    // Rebuilding discards this project's index first, and clears the shared
    // pre-split layout if the workspace still has one.
    let mut engine = Engine::rebuilding(workspace, project, profile).await?;
    let rebuilt = engine.reindex().await?;

    let result = Reindexed {
        indexed: rebuilt.indexed,
        repaired_pointers: rebuilt.repaired_pointers,
    };
    format.emit(&result, || {
        let mut rendered = format!("Rebuilt the index from postgres: {} states", result.indexed);
        if result.repaired_pointers > 0 {
            rendered.push_str(&format!(
                "\nRepaired {} topics pointing at the wrong current state",
                result.repaired_pointers
            ));
        }
        rendered
    });
    Ok(())
}
