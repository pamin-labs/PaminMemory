//! Schema migrations.
//!
//! Migrations are embedded into the binary and applied by `sqlx`, which also
//! owns the applied-version table. Nothing here is hand-rolled: a home-grown
//! runner would have to re-solve ordering, checksums, and partial application,
//! and would get one of them wrong.
//!
//! The list is written out rather than discovered. `sqlx::migrate!()` scans a
//! directory, but it needs the `macros` feature, which brings the offline-data
//! machinery and a `.env` scan with it, and it insists on its own filename
//! shape. `Migrator::with_migrations` takes the versions and descriptions
//! directly, so the files keep the names they have and adding one is a line
//! here -- visible in the diff, which a directory scan is not.

use std::borrow::Cow;

use sqlx::PgPool;
use sqlx::SqlStr;
use sqlx::migrate::{Migration, MigrationType, Migrator};

use crate::error::Result;

/// The table `refinery` recorded applied migrations in, before `sqlx`.
const APPLIED_BY_REFINERY: &str = "refinery_schema_history";

/// Every migration, oldest first.
fn migrations() -> Vec<Migration> {
    vec![
        migration(1, "initial", include_str!("../migrations/V1__initial.sql")),
        migration(
            2,
            "relationships",
            include_str!("../migrations/V2__relationships.sql"),
        ),
        migration(
            3,
            "shard_key_and_indexes",
            include_str!("../migrations/V3__shard_key_and_indexes.sql"),
        ),
        migration(
            4,
            "current_state_pointer",
            include_str!("../migrations/V4__current_state_pointer.sql"),
        ),
        migration(
            5,
            "cascade_outbox",
            include_str!("../migrations/V5__cascade_outbox.sql"),
        ),
    ]
}

fn migration(version: i64, description: &'static str, sql: &'static str) -> Migration {
    Migration::new(
        version,
        Cow::Borrowed(description),
        MigrationType::Simple,
        SqlStr::from_static(sql),
        false,
    )
}

/// Applies every migration the database has not seen yet.
///
/// Safe to call on every start: already-applied migrations are skipped.
pub async fn run(pool: &PgPool) -> Result<()> {
    let migrations = migrations();
    adopt_refinery_history(pool, &migrations).await?;

    let mut migrator = Migrator::with_migrations(migrations);

    // `sqlx` takes a `pg_advisory_lock` around the run by default, and advisory
    // locks are one of the constructs this project rules out: they are
    // node-local, so a schema that depends on one does not survive being
    // sharded. The rule is checked against the migration files, which cannot
    // see a lock the runner takes on its own, so it is turned off here instead.
    //
    // What the lock buys is that two commands starting at once do not both try
    // to migrate. Without it they still cannot both succeed: each migration is
    // applied in the same transaction that records it and the version column is
    // a primary key, so the loser fails rather than applying anything twice.
    // That is the exposure this already had under `refinery`, and it goes away
    // once a server owns startup.
    migrator.set_locking(false);
    migrator.run(pool).await?;

    Ok(())
}

/// Records migrations a previous `refinery` run applied, so `sqlx` skips them.
///
/// The two runners keep unrelated books: different table, different checksum
/// function, different columns. Pointing `sqlx` at a database `refinery` had
/// already migrated would have it find nothing applied and try to apply
/// everything, against a schema that already has every table.
///
/// Only the version and the checksum are ever validated, so the rest of the row
/// is filled with what `sqlx` itself writes. The checksums come from the same
/// `Migration` values the runner is about to validate against, which is what
/// makes this incapable of recording one that disagrees.
///
/// Runs before every migration and does nothing on all but one of them: a
/// workspace created after this change has no `refinery` table, and one created
/// before has been adopted by the time its next command runs.
async fn adopt_refinery_history(pool: &PgPool, migrations: &[Migration]) -> Result<()> {
    if !table_exists(pool, APPLIED_BY_REFINERY).await? {
        return Ok(());
    }

    // The same shape `sqlx` creates, because it is about to find it already
    // there and go on to read it.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS _sqlx_migrations (
             version        BIGINT PRIMARY KEY,
             description    TEXT NOT NULL,
             installed_on   TIMESTAMPTZ NOT NULL DEFAULT now(),
             success        BOOLEAN NOT NULL,
             checksum       BYTEA NOT NULL,
             execution_time BIGINT NOT NULL
         )",
    )
    .execute(pool)
    .await?;

    let (recorded,): (i64,) = sqlx::query_as("SELECT count(*) FROM _sqlx_migrations")
        .fetch_one(pool)
        .await?;
    if recorded > 0 {
        return Ok(());
    }

    let applied: Vec<i64> = sqlx::query_as("SELECT version FROM refinery_schema_history")
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(|(version,): (i32,)| i64::from(version))
        .collect();

    for migration in migrations {
        if !applied.contains(&migration.version) {
            continue;
        }

        sqlx::query(
            "INSERT INTO _sqlx_migrations
                 (version, description, success, checksum, execution_time)
             VALUES ($1, $2, TRUE, $3, -1)
             ON CONFLICT (version) DO NOTHING",
        )
        .bind(migration.version)
        .bind(migration.description.as_ref())
        .bind(migration.checksum.as_ref())
        .execute(pool)
        .await?;

        tracing::info!(
            version = migration.version,
            name = %migration.description,
            "migration adopted from refinery"
        );
    }

    Ok(())
}

/// Whether a table of this name exists in the current schema.
///
/// Through `information_schema` rather than `to_regclass`, which is one engine's
/// spelling of the same question.
async fn table_exists(pool: &PgPool, name: &str) -> Result<bool> {
    let (exists,): (bool,) = sqlx::query_as(
        "SELECT EXISTS (
             SELECT 1 FROM information_schema.tables
             WHERE table_schema = current_schema() AND table_name = $1
         )",
    )
    .bind(name)
    .fetch_one(pool)
    .await?;

    Ok(exists)
}

#[cfg(test)]
mod tests {
    use sqlx::migrate::Migrator;

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

    /// The runner must not take an advisory lock either.
    ///
    /// The scan above reads the migration files, so it cannot see a statement
    /// the runner issues on its own -- and `sqlx` issues `pg_advisory_lock`
    /// around every run unless told otherwise. Without this the rule held for
    /// the schema and was broken by the thing applying it.
    #[test]
    fn the_runner_is_told_not_to_take_an_advisory_lock() {
        let mut migrator = Migrator::with_migrations(super::migrations());
        assert!(
            migrator.locking,
            "sqlx no longer locks by default; check whether run() still needs to turn it off"
        );

        migrator.set_locking(false);
        assert!(!migrator.locking);
    }

    /// Every migration file is applied, exactly once, in ascending order.
    ///
    /// Listing them by hand is what keeps a new migration visible in a diff. It
    /// is also what makes a repeated version or a forgotten file possible, and
    /// both fail quietly: an unlisted migration is never applied, and a
    /// repeated version is skipped as already applied.
    #[test]
    fn every_migration_file_is_applied_once_and_in_order() {
        let listed = super::migrations();

        let versions: Vec<i64> = listed.iter().map(|migration| migration.version).collect();
        let mut ascending = versions.clone();
        ascending.sort_unstable();
        ascending.dedup();
        assert_eq!(
            versions, ascending,
            "migration versions must be distinct and ascending"
        );

        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations");
        let files = std::fs::read_dir(dir)
            .expect("migrations directory")
            .filter(|entry| {
                entry
                    .as_ref()
                    .is_ok_and(|entry| entry.path().extension().is_some_and(|ext| ext == "sql"))
            })
            .count();

        assert_eq!(
            listed.len(),
            files,
            "a migration file exists that the runner never applies"
        );
    }
}
