//! `pamin reindex` — rebuild the projection index from PostgreSQL.
//!
//! This command is the guarantee that makes a pre-1.0 index engine acceptable:
//! whatever the index does, the authority store can reproduce it. It exists
//! from the first release so the guarantee is executable rather than stated.

use anyhow::Result;
use pamin_index::{Profile, ProjectionIndex};
use pamin_store::Workspace;
use serde::Serialize;

use crate::engine::Engine;
use crate::output::Format;

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
    // Discard before rebuilding rather than overwriting in place. An overwrite
    // leaves behind anything the ledger no longer has, which is exactly the
    // drift a rebuild is supposed to eliminate.
    ProjectionIndex::discard(&workspace.index_dir())?;

    let mut engine = Engine::open(workspace, project, profile).await?;
    let indexed = engine.reindex().await?;

    let result = Reindexed { indexed };
    format.emit(&result, || {
        format!("Rebuilt the index from postgres: {} states", result.indexed)
    });
    Ok(())
}
