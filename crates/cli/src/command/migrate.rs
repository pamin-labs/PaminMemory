//! `pamin-memory migrate` — migrate the external database, once per deploy.

use anyhow::{Context, Result};
use memory_client::settings::DATABASE_URL;
use memory_contract::migrate::Migrated;

pub async fn execute() -> Result<Migrated> {
    let url = crate::settings::database_url()?.with_context(|| {
        format!(
            "migrate applies to the database named by {DATABASE_URL}, which is not set; \
             a workspace's own cluster migrates when it opens"
        )
    })?;
    let applied = memory_store::Database::migrate(&url)
        .await
        .with_context(|| format!("migrating the PostgreSQL server named by {DATABASE_URL}"))?;
    Ok(Migrated { applied })
}
