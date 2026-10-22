//! `pamin init` — provision the local database.

use anyhow::Result;
use pamin_client::command::init::Initialized;

use crate::session::Session;

pub async fn execute(session: &Session, project: &str) -> Result<Initialized> {
    // Provisioning, starting, and migrating all happen when the session opens,
    // so the quickstart is one command with no database to install and no
    // configuration to write. All that is left here is the project row.
    session.project(project).await?;

    let result = Initialized {
        project: project.to_string(),
        home: pamin_index::shown_path(session.workspace().root()),
    };

    Ok(result)
}
