//! Schema migrations.
//!
//! Migrations are embedded into the binary and applied by `refinery`, which
//! also owns the applied-version table. Nothing here is hand-rolled: a
//! home-grown runner would have to re-solve ordering, checksums, and partial
//! application, and would get one of them wrong.

use crate::error::Result;

mod embedded {
    refinery::embed_migrations!("migrations");
}

/// Applies every migration the database has not seen yet.
///
/// Safe to call on every start: already-applied migrations are skipped.
pub async fn run(client: &mut tokio_postgres::Client) -> Result<()> {
    let report = embedded::migrations::runner().run_async(client).await?;

    for migration in report.applied_migrations() {
        tracing::info!(
            version = migration.version(),
            name = migration.name(),
            "migration applied"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    /// Constructs that would strand us on a single PostgreSQL node.
    ///
    /// The rule is cheap to hold now and expensive to retrofit, so it is checked
    /// rather than trusted to review. Each entry is rejected for a reason:
    ///
    ///   * `SERIAL` and `BIGSERIAL` are monotonic sequences, a coordination
    ///     point under sharding;
    ///   * `LISTEN` and `NOTIFY` are per-connection and do not survive a
    ///     distributed deployment;
    ///   * advisory locks are node-local;
    ///   * `CREATE EXTENSION` ties the schema to one engine's extension set.
    const NON_PORTABLE: &[&str] = &[
        "serial",
        "bigserial",
        "listen ",
        "notify ",
        "pg_advisory",
        "create extension",
    ];

    #[test]
    fn migrations_stay_within_the_portable_sql_subset() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");

        let mut checked = 0;
        for entry in std::fs::read_dir(dir).expect("migrations directory") {
            let path = entry.expect("directory entry").path();
            if path.extension().is_none_or(|ext| ext != "sql") {
                continue;
            }

            let sql = std::fs::read_to_string(&path).expect("readable migration");
            // Comments explain why a construct is banned and would otherwise
            // trip the scan that enforces it.
            let statements: String = sql
                .lines()
                .filter(|line| !line.trim_start().starts_with("--"))
                .collect::<Vec<_>>()
                .join("\n")
                .to_lowercase();

            for banned in NON_PORTABLE {
                assert!(
                    !statements.contains(banned),
                    "{} uses non-portable construct {banned:?}",
                    path.display()
                );
            }
            checked += 1;
        }

        assert!(checked > 0, "no migrations were checked");
    }
}
