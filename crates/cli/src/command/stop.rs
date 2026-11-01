//! `pamin stop` — shut down the local database server.

use anyhow::Result;
use memory_contract::stop::Stopped;
use memory_store::Workspace;

pub async fn execute(workspace: &Workspace) -> Result<Stopped> {
    memory_store::database::stop(workspace).await?;

    let result = Stopped { stopped: true };
    Ok(result)
}
