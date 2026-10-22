//! `pamin reindex` — rebuild the projection index from PostgreSQL.
//!
//! This command is the guarantee that makes a pre-1.0 index engine acceptable:
//! whatever the index does, the authority store can reproduce it. It exists
//! from the first release so the guarantee is executable rather than stated.

use anyhow::Result;
use pamin_client::command::reindex::{Args, Reindexed};
use pamin_index::VectorIndex;

use crate::session::Session;

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    _args: Args,
) -> Result<Reindexed> {
    // Rebuilding discards this project's index first, and clears the shared
    // pre-split layout if the workspace still has one.
    let engine = session.rebuilding(project, vector_index).await?;
    let rebuilt = engine.reindex().await?;

    let result = Reindexed {
        indexed: rebuilt.indexed,
        reused: rebuilt.reused,
        repaired_pointers: rebuilt.repaired_pointers,
        names: rebuilt.names,
    };
    Ok(result)
}
