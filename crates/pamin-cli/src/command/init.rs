//! `pamin init` — provision the local database.

use anyhow::Result;
use pamin_store::{Database, Workspace, repository};
use serde::Serialize;

use crate::output::Format;

#[derive(Serialize)]
struct Initialized {
    project: String,
    home: String,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format) -> Result<()> {
    // Provisioning, starting, and migrating all happen here, so the quickstart
    // is one command with no database to install and no configuration to write.
    let database = Database::open(workspace).await?;
    repository::ensure_project(database.client(), project).await?;

    let result = Initialized {
        project: project.to_string(),
        home: workspace.root().display().to_string(),
    };

    format.emit(&result, || {
        format!("Initialized project {} in {}", result.project, result.home)
    });
    Ok(())
}
