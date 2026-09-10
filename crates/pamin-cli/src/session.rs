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
    engines: Mutex<Open>,
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

/// The open engines, and which was used when.
#[derive(Default)]
struct Open {
    engines: HashMap<(String, Profile), Entry>,
    /// Counts uses rather than reading a clock: what matters is the order they
    /// were last wanted in, and a counter cannot go backwards.
    uses: u64,
}

struct Entry {
    /// Behind an `Arc` so that eviction can tell an idle engine from one a
    /// request is still using. Dropping the registry's handle to a busy engine
    /// would leave it open anyway, and the next request for that project would
    /// try to open its index a second time and be refused by its own lock.
    engine: Arc<Engine>,
    used: u64,
}

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
    pub async fn engine(&self, project: &str, profile: Profile) -> Result<Arc<Engine>> {
        let key = (project.to_string(), profile);

        let mut open = self.engines.lock().await;
        open.uses += 1;
        let now = open.uses;

        if let Some(entry) = open.engines.get_mut(&key) {
            entry.used = now;
            return Ok(Arc::clone(&entry.engine));
        }

        open.make_room();

        let engine = Arc::new(
            Engine::attached(
                self.database.clone(),
                &self.models,
                &self.workspace,
                project,
                profile,
                Access::ReadWrite,
            )
            .await?,
        );

        open.engines.insert(
            key,
            Entry {
                engine: Arc::clone(&engine),
                used: now,
            },
        );
        Ok(engine)
    }

    /// An engine with this project's index discarded first, for a rebuild.
    ///
    /// Evicts rather than reuses: the rebuild throws the collection away and
    /// opens a new one, so an engine held from before points at a directory
    /// that is gone.
    pub async fn rebuilding(&self, project: &str, profile: Profile) -> Result<Arc<Engine>> {
        let key = (project.to_string(), profile);

        let mut open = self.engines.lock().await;
        open.engines.remove(&key);
        open.uses += 1;
        let now = open.uses;
        open.make_room();

        let engine = Arc::new(
            Engine::rebuilding_attached(
                self.database.clone(),
                &self.models,
                &self.workspace,
                project,
                profile,
            )
            .await?,
        );

        open.engines.insert(
            key,
            Entry {
                engine: Arc::clone(&engine),
                used: now,
            },
        );
        Ok(engine)
    }
}

impl Open {
    /// Closes least-recently-used engines until there is room for one more.
    ///
    /// Only ones nothing is using. An engine a request still holds stays open
    /// whether or not this drops its handle, so evicting it would buy nothing
    /// and cost the next request for that project a refusal from the index's
    /// own lock. When every open engine is busy the bound gives way rather
    /// than the request: it exists to stop idle indexes accumulating, not to
    /// cap how many projects can be served at once.
    fn make_room(&mut self) {
        while self.engines.len() >= OPEN_INDEXES {
            let idle = self
                .engines
                .iter()
                .filter(|(_, entry)| Arc::strong_count(&entry.engine) == 1)
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());

            let Some(idle) = idle else { return };
            self.engines.remove(&idle);
        }
    }
}
