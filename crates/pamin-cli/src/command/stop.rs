//! `pamin stop` — shut down the local database server.

use anyhow::Result;
use pamin_contract::stop::Stopped;
use pamin_store::Workspace;

pub async fn execute(workspace: &Workspace) -> Result<Stopped> {
    pamin_store::database::stop(workspace).await?;

    let result = Stopped { stopped: true };
    Ok(result)
}
