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
use std::sync::Arc;

use anyhow::Result;
use pamin_core::ProjectId;
use pamin_engine::{Engine, Models};
use pamin_index::{Access, Profile};
use pamin_store::{Connections, Database, Workspace, repository};

use crate::registry::Registry;
use tokio::sync::Mutex;

/// The database, and whatever has been opened against it so far.
pub struct Session {
    workspace: Workspace,
    database: Database,
    /// Project names resolved to rows. `ensure_project` is an upsert, so this
    /// saves a round trip rather than changing what happens.
    projects: Mutex<HashMap<String, ProjectId>>,
    /// The models this process has loaded, shared by every engine below.
    models: Models,
    /// The engines this process is holding open, most recently used last.
    ///
    /// Keyed by profile as well as project because an index records the
    /// profile it was built with and refuses to open under another; two
    /// profiles against one project are two indexes, and the engine holding
    /// one cannot answer for the other.
    engines: Registry<(String, Profile), Engine>,
}

/// How many indexes stay open at once.
///
/// An open index is not free and does not become free by being idle: measured
/// on the smallest profile, each one holds 27 file descriptors and about 100 MB
/// resident. Against the 1024-descriptor limit a process usually starts with,
/// that runs out at the thirty-seventh project -- which is what happened, with
/// the engine refusing to create its lexical indexer, before there was a bound
/// here at all.
///
/// Sixteen leaves both quantities a wide margin and is large enough that a
/// server round-robining a working set keeps it. The seventeenth project
/// closes the one nobody has touched for longest rather than failing.
const OPEN_INDEXES: usize = 16;

impl Session {
    /// Connects, migrates, and holds the result.
    ///
    /// How many connections it may hold is the caller's to say, because that
    /// is a question about the process rather than about the workspace: a
    /// server is the only one talking to the cluster, and a command is one of
    /// however many an agent is running.
    pub async fn open(workspace: &Workspace, connections: Connections) -> Result<Self> {
        Ok(Self {
            workspace: workspace.clone(),
            database: Database::open(workspace, connections).await?,
            models: Models::in_workspace(workspace),
            projects: Mutex::default(),
            engines: Registry::with_capacity(OPEN_INDEXES),
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
    /// Twenty requests arriving at a cold project load one model rather than
    /// twenty: they queue on that project's slot, and one of them opens it.
    /// That costs the second caller the first caller's wait, which is the same
    /// wait it would have paid loading its own.
    ///
    /// What they no longer cost is every *other* project. The exclusion used to
    /// be one lock over the whole registry, held across the open -- and an open
    /// downloads the profile's weights the first time anyone wants them, so a
    /// cold project stalled every project this process was serving, including
    /// ones already open. Two projects on one profile still wait for each other
    /// inside [`Models`], which is where waiting for weights belongs.
    pub async fn engine(&self, project: &str, profile: Profile) -> Result<Arc<Engine>> {
        self.engines
            .get_or_open((project.to_string(), profile), || {
                Engine::attached(
                    self.database.clone(),
                    &self.models,
                    &self.workspace,
                    project,
                    profile,
                    Access::ReadWrite,
                )
            })
            .await
    }

    /// The engines this process currently holds open.
    ///
    /// For the maintenance loop, which has no project of its own to work on:
    /// upkeep is owed by whatever has been written, and what has been written
    /// recently is what is open. A project evicted before its upkeep ran keeps
    /// the job -- nothing is lost, it waits until the project is wanted again,
    /// which is also when it starts mattering again.
    /// Handed out one at a time rather than all at once, because holding every
    /// engine for the length of a sweep makes all of them look busy and stops
    /// eviction finding anything to close while it runs.
    pub fn opened_projects(&self) -> Vec<(String, Profile)> {
        self.engines.keys()
    }

    /// One open engine, if it is open and nobody else is inside it.
    ///
    /// Never opens one: a caller working through what is open should not be
    /// what reopens a project nobody asked for.
    pub fn opened_engine(&self, key: &(String, Profile)) -> Option<Arc<Engine>> {
        self.engines.opened(key)
    }

    /// An engine with this project's index discarded first, for a rebuild.
    ///
    /// Evicts rather than reuses: the rebuild throws the collection away and
    /// opens a new one, so an engine held from before points at a directory
    /// that is gone.
    pub async fn rebuilding(&self, project: &str, profile: Profile) -> Result<Arc<Engine>> {
        self.engines
            .reopen((project.to_string(), profile), || {
                Engine::rebuilding_attached(
                    self.database.clone(),
                    &self.models,
                    &self.workspace,
                    project,
                    profile,
                )
            })
            .await
    }
}
