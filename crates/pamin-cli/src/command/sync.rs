//! `pamin sync` and `pamin unsync` — the paths the server keeps in sync.

use anyhow::Result;
use pamin_client::command::sync::{Synced, SyncedPath};
use pamin_client::command::unsync::{Args as UnsyncArgs, Unsynced};
use pamin_client::command::validity;

use crate::session::Session;

/// Every path of this project kept in sync, and how each stands.
pub async fn list(session: &Session, project: &str) -> Result<Synced> {
    let project_id = session.project(project).await?;
    let synced =
        pamin_store::repository::synced_paths_of(session.database().pool(), project_id).await?;
    let paths = synced
        .into_iter()
        .map(|synced| {
            let status = session
                .syncing()
                .map(|syncing| syncing.status(project, &synced.path))
                .unwrap_or_default();
            SyncedPath {
                state: status.state.name().to_string(),
                synced_at: status.synced_at.map(validity::render),
                error: status.error,
                path: synced.path,
            }
        })
        .collect();
    Ok(Synced { paths })
}

/// Stops keeping these paths in sync. Each is matched as given and as it
/// resolves, since a path kept in sync is kept as it resolved when it was
/// ingested and may no longer resolve at all.
pub async fn unsync(session: &Session, project: &str, args: UnsyncArgs) -> Result<Unsynced> {
    let project_id = session.project(project).await?;
    let mut removed = Vec::new();
    let mut unknown = Vec::new();
    for path in &args.paths {
        let mut spellings = vec![path.display().to_string()];
        if let Ok(resolved) = tokio::fs::canonicalize(path).await {
            spellings.push(resolved.display().to_string());
        }
        let gone = pamin_store::repository::unsync_paths(
            session.database().pool(),
            project_id,
            &spellings,
        )
        .await?;
        if gone.is_empty() {
            unknown.push(path.display().to_string());
        }
        for path in gone {
            if let Some(syncing) = session.syncing() {
                syncing.remove(project, &path);
            }
            removed.push(path);
        }
    }
    Ok(Unsynced { removed, unknown })
}
