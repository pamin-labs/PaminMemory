//! `pamin stop` — shut down the local database server.
//!
//! The server is deliberately left running between commands so an agent does
//! not pay cluster startup on every call. That makes an explicit way to stop it
//! part of the contract rather than an extra.

use anyhow::Result;
use pamin_store::Workspace;
use serde::Serialize;

use crate::output::Format;

#[derive(Serialize)]
struct Stopped {
    stopped: bool,
}

pub async fn run(workspace: &Workspace, format: Format) -> Result<()> {
    pamin_store::database::stop(workspace).await?;

    let result = Stopped { stopped: true };
    format.emit(&result, || "Stopped the local database server".to_string());
    Ok(())
}
