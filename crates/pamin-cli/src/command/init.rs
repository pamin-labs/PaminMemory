//! `pamin init` — provision the local database.

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::session::Session;

#[derive(Serialize, Deserialize)]
pub struct Initialized {
    project: String,
    home: String,
}

pub async fn execute(session: &Session, project: &str) -> Result<Initialized> {
    // Provisioning, starting, and migrating all happen when the session opens,
    // so the quickstart is one command with no database to install and no
    // configuration to write. All that is left here is the project row.
    session.project(project).await?;

    let result = Initialized {
        project: project.to_string(),
        home: session.workspace().root().display().to_string(),
    };

    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(result: &Initialized) -> String {
    format!("Initialized project {} in {}", result.project, result.home)
}
