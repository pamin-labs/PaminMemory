//! `pamin-memory purge` — delete a project and everything it holds, for good.

use anyhow::Result;
use memory_contract::purge::{Args, Purged};

use crate::session::Session;

pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Purged> {
    anyhow::ensure!(
        args.yes,
        "purging deletes project {project:?} and everything it holds, and cannot be undone; \
         pass --yes to confirm"
    );
    let purged = session.purge(project).await?;
    Ok(match purged {
        Some(purged) => Purged {
            existed: true,
            topics: purged.topics,
            sources: purged.sources,
            unsynced: purged.synced,
        },
        None => Purged {
            existed: false,
            topics: 0,
            sources: 0,
            unsynced: Vec::new(),
        },
    })
}
