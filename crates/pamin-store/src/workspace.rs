//! On-disk layout and connection settings for a local install.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Where PaminMemory keeps its database, indexes, and models.
///
/// Defaults to `~/.pamin`, overridable with `PAMIN_HOME` so a test or a second
/// checkout can run against its own state.
#[derive(Clone, Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Opens the workspace named by `PAMIN_HOME`, or `~/.pamin` otherwise.
    pub fn discover() -> std::io::Result<Self> {
        let root = match std::env::var_os("PAMIN_HOME") {
            Some(path) => PathBuf::from(path),
            None => {
                let home = std::env::home_dir().ok_or_else(|| {
                    std::io::Error::other("cannot locate a home directory; set PAMIN_HOME")
                })?;
                home.join(".pamin")
            }
        };
        Ok(Self::at(root))
    }

    /// Opens a workspace at an explicit path.
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the embedded PostgreSQL binaries are installed.
    pub fn postgres_install_dir(&self) -> PathBuf {
        self.root.join("postgres").join("install")
    }

    /// Where the embedded PostgreSQL cluster keeps its data.
    pub fn postgres_data_dir(&self) -> PathBuf {
        self.root.join("postgres").join("data")
    }

    pub fn postgres_password_file(&self) -> PathBuf {
        self.root.join("postgres").join("pgpass")
    }

    /// Where the projection index lives. Derived data: safe to delete, and
    /// rebuilt from PostgreSQL by `pamin reindex`.
    pub fn index_dir(&self) -> PathBuf {
        self.root.join("index")
    }

    fn connection_file(&self) -> PathBuf {
        self.root.join("server.json")
    }

    /// Reads the stored server record, if this workspace was initialized.
    pub fn read_server(&self) -> std::io::Result<Option<LocalServer>> {
        match std::fs::read(self.connection_file()) {
            Ok(bytes) => Ok(Some(
                serde_json::from_slice(&bytes).map_err(std::io::Error::other)?,
            )),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Records the server so later commands reuse it instead of starting a
    /// second one.
    pub fn write_server(&self, server: &LocalServer) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.root)?;
        let json = serde_json::to_vec_pretty(server).map_err(std::io::Error::other)?;
        std::fs::write(self.connection_file(), json)
    }
}

/// The PostgreSQL server belonging to this workspace.
///
/// Records both how to reach the server and where its binaries ended up.
/// The install directory is version-qualified by the installer, so stopping the
/// server later needs the resolved path rather than the directory we asked for.
///
/// The password is generated at setup and stored here because the server is
/// local, listens on loopback, and belongs to the user who created it. Treating
/// it as a secret to prompt for would add friction without adding a boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalServer {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub database: String,
    pub installation_dir: PathBuf,
}

impl LocalServer {
    /// The libpq connection string for this workspace's database.
    pub fn url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.username, self.password, self.host, self.port, self.database
        )
    }
}
