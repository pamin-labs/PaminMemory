//! Bringing up and connecting to the local PostgreSQL server.
//!
//! PostgreSQL is bundled rather than brought by the user. The CLI is meant to
//! be a zero-integration path for any agent with a shell, and that promise does
//! not survive a prerequisite that starts with installing a database.

use std::mem::ManuallyDrop;

use postgresql_embedded::{PostgreSQL, Settings, VersionReq};

use crate::error::{Result, StoreError};
use crate::workspace::{LocalServer, Workspace};

/// The database name created inside the embedded cluster.
const DATABASE: &str = "pamin";

/// A connected client, plus the background task driving its connection.
pub struct Database {
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

        let mut database = Self::connect(&server).await?;
        crate::migrate::run(&mut database.client).await?;
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

        Ok(Self { client, connection })
    }

    pub fn client(&self) -> &tokio_postgres::Client {
        &self.client
    }

    pub fn client_mut(&mut self) -> &mut tokio_postgres::Client {
        &mut self.client
    }
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
        ..Settings::default()
    };

    PostgreSQL::new(settings).stop().await?;
    Ok(())
}
