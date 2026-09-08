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
    let indexed = engine.reindex().await?;

    let result = Reindexed { indexed };
    format.emit(&result, || {
        format!("Rebuilt the index from postgres: {} states", result.indexed)
    });
    Ok(())
}
