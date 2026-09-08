//! Bringing up and connecting to the local PostgreSQL server.
//!
//! PostgreSQL is bundled rather than brought by the user. The CLI is meant to
//! be a zero-integration path for any agent with a shell, and that promise does
//! not survive a prerequisite that starts with installing a database.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::time::Duration;

use postgresql_embedded::{PostgreSQL, Settings, VersionReq};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use crate::error::{Result, StoreError};
use crate::workspace::{LocalServer, Workspace};

/// The database name created inside the embedded cluster.
const DATABASE: &str = "pamin";

/// A connected client, plus the background task driving its connection.
pub struct Database {
    pool: PgPool,
    client: tokio_postgres::Client,
    connection: tokio::task::JoinHandle<()>,
}

impl Database {
    /// Ensures a server is running for this workspace, connects, and migrates.
    ///
    /// Safe to call repeatedly. The first call installs and initializes the
    /// cluster; later calls reuse the running server.
    pub async fn open(workspace: &Workspace) -> Result<Self> {
        let server = match workspace.read_server()? {
            Some(existing) if can_connect(&existing).await => existing,
            _ => start_server(workspace).await?,
        };

        let database = Self::connect(&server).await?;
        crate::migrate::run(&database.pool).await?;
        Ok(database)
    }

    /// Connects to an already running server without touching its lifecycle.
    pub async fn connect(server: &LocalServer) -> Result<Self> {
        let (client, driver) = tokio_postgres::connect(&server.url(), tokio_postgres::NoTls)
            .await
            .map_err(StoreError::Database)?;

        // tokio-postgres splits the client from the connection: the returned
        // future must be polled for the client to make progress.
        let connection = tokio::spawn(async move {
            if let Err(error) = driver.await {
                tracing::error!(%error, "database connection ended");
            }
        });

        let pool = pool(&server.url()).await?;

        Ok(Self {
            pool,
            client,
            connection,
        })
    }

    /// The connection pool. Every query goes through this.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    pub fn client(&self) -> &tokio_postgres::Client {
        &self.client
    }

    pub fn client_mut(&mut self) -> &mut tokio_postgres::Client {
        &mut self.client
    }
}

/// Opens a pool sized for one short-lived command.
///
/// Every `pamin` invocation is its own process with its own pool, all pointing
/// at one cluster, so the pool's size is multiplied by however many agents are
/// running. The default of ten connections each means thirty agents ask for
/// three hundred, against a server that allows a hundred, and what they get is
/// `too many clients` after a thirty-second wait. Four is enough for the three
/// recall channels to run at once and small enough to multiply.
///
/// `test_before_acquire` is off. It costs a full round trip on every acquire to
/// detect connections dropped by a proxy or an idle timer, and there is neither
/// between here and a cluster on this machine that this process just started.
async fn pool(url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .min_connections(1)
        .test_before_acquire(false)
        // A command outlives neither, so recycling connections underneath it
        // only adds reconnects.
        .idle_timeout(None)
        .max_lifetime(None)
        // Long enough to outlast a busy moment, short enough that a caller
        // learns the server is unreachable rather than appearing to hang.
        .acquire_timeout(Duration::from_secs(10))
        .connect(url)
        .await?;

    Ok(pool)
}

impl Drop for Database {
    fn drop(&mut self) {
        // Drops the client's half of the connection; the server keeps running.
        self.connection.abort();
    }
}

/// Returns true when a server is already listening with these credentials.
async fn can_connect(server: &LocalServer) -> bool {
    let Ok((_client, driver)) = tokio_postgres::connect(&server.url(), tokio_postgres::NoTls).await
    else {
        return false;
    };
    tokio::spawn(driver);
    true
}

/// Installs if needed, starts the server, and leaves it running.
async fn start_server(workspace: &Workspace) -> Result<LocalServer> {
    std::fs::create_dir_all(workspace.root())?;

    let mut settings = Settings {
        version: VersionReq::parse("=17.6.0").expect("valid version requirement"),
        installation_dir: workspace.postgres_install_dir(),
        data_dir: workspace.postgres_data_dir(),
        password_file: workspace.postgres_password_file(),
        // Not temporary: the cluster outlives the process that created it, so a
        // workspace survives between commands.
        temporary: false,
        // A hard limit on `initdb` and `pg_ctl start`, not a poll interval. The
        // default is five seconds, which a first `initdb` on a cold filesystem
        // exceeds routinely -- and the failure lands on whoever is setting the
        // project up for the first time, which is the worst audience for it.
        timeout: Some(Duration::from_secs(60)),
        // PostgreSQL allows a hundred clients by default, and every `pamin`
        // command is a process with its own pool pointing at this one cluster.
        // A few dozen agents working at once exhaust that, and what they see is
        // a connection timeout rather than anything naming the limit.
        //
        // Passed to the server as it starts, so an already-initialised
        // workspace keeps whatever it was started with until `pamin stop`.
        configuration: HashMap::from([("max_connections".to_string(), "300".to_string())]),
        ..Settings::default()
    };

    // Reuse the credentials and port from a previous run when the cluster was
    // already initialized, since initdb fixed the superuser password then.
    if let Some(existing) = workspace.read_server()? {
        settings.username = existing.username.clone();
        settings.password = existing.password.clone();
        settings.port = existing.port;
    }

    let mut postgres = PostgreSQL::new(settings);
    postgres.setup().await?;
    postgres.start().await?;

    if !postgres.database_exists(DATABASE).await? {
        postgres.create_database(DATABASE).await?;
    }

    let settings = postgres.settings();
    let server = LocalServer {
        host: settings.host.clone(),
        port: settings.port,
        username: settings.username.clone(),
        password: settings.password.clone(),
        database: DATABASE.to_string(),
        // setup() qualifies the install directory with the resolved version, so
        // record what it settled on rather than what we asked for.
        installation_dir: settings.installation_dir.clone(),
    };
    workspace.write_server(&server)?;

    // Suppress the handle's Drop, which would stop the server we just started.
    // Leaving it running is the point: an agent invoking the CLI repeatedly
    // should not pay cluster startup on every call.
    let _ = ManuallyDrop::new(postgres);

    Ok(server)
}

/// Stops the server for this workspace, if one is running.
pub async fn stop(workspace: &Workspace) -> Result<()> {
    let Some(server) = workspace.read_server()? else {
        return Ok(());
    };

    let settings = Settings {
        installation_dir: server.installation_dir,
        data_dir: workspace.postgres_data_dir(),
        password_file: workspace.postgres_password_file(),
        username: server.username,
        password: server.password,
        port: server.port,
        temporary: false,
        // Same reason as starting: this is a hard limit on `pg_ctl stop`, and
        // a shutdown that waits for a long checkpoint is not a hung one.
        timeout: Some(Duration::from_secs(60)),
        ..Settings::default()
    };

    PostgreSQL::new(settings).stop().await?;
    Ok(())
}
