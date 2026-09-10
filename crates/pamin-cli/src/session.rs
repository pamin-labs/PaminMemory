//! What a command needs from the process running it.
//!
//! Every command used to open its own: a database (which probes the cluster and
//! runs the migrations), a project row, and for some of them an index and a
//! model. That is the right thing to do once and the wrong thing to do per
//! request, and the difference between those two is the whole reason there is
//! a server.
//!
//! Both callers build one of these. A server builds one and keeps it; a command
//! running without a server builds one, uses it, and drops it, which is exactly
//! what it did before. The commands cannot tell the difference, which is what
//! keeps the two paths honest about producing the same answers.

use std::collections::HashMap;

use anyhow::Result;
use pamin_core::ProjectId;
use pamin_engine::Engine;
use pamin_index::{Access, Profile};
use pamin_store::{Database, Workspace, repository};
use tokio::sync::Mutex;

/// The database, and whatever has been opened against it so far.
pub struct Session {
    workspace: Workspace,
    database: Database,
    /// Project names resolved to rows. `ensure_project` is an upsert, so this
    /// saves a round trip rather than changing what happens.
    projects: Mutex<HashMap<String, ProjectId>>,
    /// One engine per project and profile.
    ///
    /// Keyed by profile as well as project because an index records the
    /// profile it was built with and refuses to open under another; two
    /// profiles against one project are two indexes, and the engine holding
    /// one cannot answer for the other.
    engines: Mutex<HashMap<(String, Profile), Engine>>,
}

impl Session {
    /// Connects, migrates, and holds the result.
    pub async fn open(workspace: &Workspace) -> Result<Self> {
        Ok(Self {
            workspace: workspace.clone(),
            database: Database::open(workspace).await?,
            projects: Mutex::default(),
            engines: Mutex::default(),
        })
    }

    pub fn database(&self) -> &Database {
        &self.database
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The project row for this name, creating it if it is new.
    pub async fn project(&self, name: &str) -> Result<ProjectId> {
        let mut projects = self.projects.lock().await;
        if let Some(project) = projects.get(name) {
            return Ok(*project);
        }

        let project = repository::ensure_project(self.database.pool(), name).await?;
        projects.insert(name.to_string(), project.id);
        Ok(project.id)
    }

    /// An engine for this project and profile, opening one if there is none.
    ///
    /// Always for writing. Read-only opens existed so that two *processes*
    /// could search one project at once, by taking a shared lock on the
    /// collection instead of an exclusive one. Inside a server there is one
    /// process, it is both the reader and the writer, and what separates
    /// readers from writers is the lock the engine holds rather than the mode
    /// the collection was opened in.
    ///
    /// The lock is held across the open, so twenty requests arriving at a cold
    /// project load one model rather than twenty. That costs the second caller
    /// the first caller's wait, which is the same wait it would have paid
    /// loading its own.
    pub async fn engine(&self, project: &str, profile: Profile) -> Result<Engine> {
        let key = (project.to_string(), profile);

        let mut engines = self.engines.lock().await;
        if let Some(engine) = engines.get(&key) {
            return Ok(engine.clone());
        }

        let engine = Engine::attached(
            self.database.clone(),
            &self.workspace,
            project,
            profile,
            Access::ReadWrite,
        )
        .await?;

        engines.insert(key, engine.clone());
        Ok(engine)
    }

    /// An engine with this project's index discarded first, for a rebuild.
    ///
    /// Evicts rather than reuses: the rebuild throws the collection away and
    /// opens a new one, so an engine held from before points at a directory
    /// that is gone.
    pub async fn rebuilding(&self, project: &str, profile: Profile) -> Result<Engine> {
        let mut engines = self.engines.lock().await;
        engines.remove(&(project.to_string(), profile));

        let engine =
            Engine::rebuilding_attached(self.database.clone(), &self.workspace, project, profile)
                .await?;

        engines.insert((project.to_string(), profile), engine.clone());
        Ok(engine)
    }
}
