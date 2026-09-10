//! `pamin stop` — shut down the local database server.
//!
//! The server is deliberately left running between commands so an agent does
//! not pay cluster startup on every call. That makes an explicit way to stop it
//! part of the contract rather than an extra.

use anyhow::Result;
use pamin_store::Workspace;
use serde::Serialize;

#[derive(Serialize)]
pub struct Stopped {
    stopped: bool,
}

pub async fn execute(workspace: &Workspace) -> Result<Stopped> {
    pamin_store::database::stop(workspace).await?;

    let result = Stopped { stopped: true };
    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(_result: &Stopped) -> String {
    "Stopped the local database server".to_string()
}
