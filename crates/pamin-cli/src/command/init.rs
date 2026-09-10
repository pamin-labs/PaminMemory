//! `pamin init` — provision the local database.

use anyhow::Result;
use pamin_store::{Database, Workspace, repository};
use serde::Serialize;

#[derive(Serialize)]
pub struct Initialized {
    project: String,
    home: String,
}

pub async fn execute(workspace: &Workspace, project: &str) -> Result<Initialized> {
    // Provisioning, starting, and migrating all happen here, so the quickstart
    // is one command with no database to install and no configuration to write.
    let database = Database::open(workspace).await?;
    repository::ensure_project(database.pool(), project).await?;

    let result = Initialized {
        project: project.to_string(),
        home: workspace.root().display().to_string(),
    };

    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(result: &Initialized) -> String {
    format!("Initialized project {} in {}", result.project, result.home)
}
