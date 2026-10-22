//! What a command needs from the process running it.
//!
//! Every command used to open its own: a database (which probes the cluster and
//! runs the migrations), a project row, and for some of them an index and a
//! model. That is the right thing to do once and the wrong thing to do per
//! request, and the difference between those two is the whole reason there is
//! a server.
//!
//! The server builds one of these and keeps it. It is the only thing that
//! does: every command reaches the workspace through the server, so there is
//! no second path to keep producing the same answers.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use pamin_core::ProjectId;
use pamin_engine::{Engine, Models};
use pamin_index::{Access, IndexError, VectorIndex};
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
    /// The engines this process is holding open, most recently used last,
    /// keyed by project.
    engines: Registry<String, Engine>,
    /// Projects being warmed in the background, so a burst of requests at one
    /// cold project starts one warm-up rather than one each. See
    /// [`Session::warm`].
    warming: std::sync::Mutex<std::collections::HashSet<String>>,
    /// How many requests are being answered right now.
    ///
    /// What the server's upkeep asks before it spends a model on work nobody
    /// is waiting for. See [`Session::serving`].
    serving: AtomicUsize,
}

/// A request being answered, for as long as this is held.
pub struct Serving<'a>(&'a AtomicUsize);

impl Drop for Serving<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Session {
    /// Connects, migrates, and holds the result.
    ///
    /// With a resident's pool: the server holding this is the only process
    /// the CLI talks to the cluster from.
    /// `open_indexes` is how many project indexes it keeps open at once; see
    /// [`OPEN_INDEXES`](pamin_client::settings::OPEN_INDEXES).
    pub async fn open(
        workspace: &Workspace,
        settings: pamin_engine::Settings,
        open_indexes: usize,
    ) -> Result<Self> {
        Ok(Self {
            workspace: workspace.clone(),
            database: Database::open(workspace, Connections::Resident, &settings.store).await?,
            models: Models::new(workspace, settings),
            projects: Mutex::default(),
            engines: Registry::with_capacity(open_indexes),
            warming: std::sync::Mutex::default(),
            serving: AtomicUsize::new(0),
        })
    }

    /// Counts a request as being answered until the result is dropped.
    ///
    /// A guard rather than a pair of calls, so a request that fails or panics
    /// half way is still uncounted -- a count left one too high would stop
    /// the upkeep catching up for as long as the server lives.
    pub fn serving(&self) -> Serving<'_> {
        self.serving.fetch_add(1, Ordering::SeqCst);
        Serving(&self.serving)
    }

    /// Whether no request is being answered.
    ///
    /// Asked between rounds of work the server does on nobody's behalf. A
    /// request that arrives just after this answers yes waits for at most the
    /// round that started, which is what the round's size is chosen against.
    pub fn is_quiet(&self) -> bool {
        self.serving.load(Ordering::SeqCst) == 0
    }

    /// Opens this project's index and loads its models in the background, if
    /// this process holds nothing open for it -- without making the caller
    /// wait for any of it.
    ///
    /// A server that has released a project, or never held it, answers the
    /// first search by opening the index and loading both models in front of
    /// it: 2,386 ms at the median of the `COLD` arm against 618 for the next
    /// search. Whatever an agent asks first -- a `read`, a `write`, the search
    /// itself -- is the first sign it is working on the project, so that is
    /// when this starts. A search that arrives before it finishes waits for
    /// the loads already in flight rather than starting its own, so it never
    /// pays more than it would have, and one that arrives after pays nothing.
    ///
    /// Nothing here changes what is given back. The index is opened into the
    /// same registry and the models into the same [`Models`], stamped as used
    /// now, so a project warmed and then left alone is closed and its models
    /// released one idle window later, as if a search had opened them.
    pub fn warm(self: &Arc<Self>, project: &str, vector_index: VectorIndex) {
        let key = project.to_string();
        if self.engines.holds(&key) {
            return;
        }
        if !self
            .warming
            .lock()
            .expect("the warming set is poisoned")
            .insert(key.clone())
        {
            return;
        }
        let session = Arc::clone(self);
        tokio::spawn(async move {
            let models = session.models.clone();
            let loading = tokio::task::spawn_blocking(move || models.warm());
            if let Err(error) = session.engine(&key, vector_index).await {
                tracing::debug!(project = %key, %error, "warming: opening the project failed");
            }
            match loading.await {
                Ok(Ok(())) => tracing::debug!(project = %key, "warmed"),
                Ok(Err(error)) => {
                    tracing::debug!(project = %key, %error, "warming: loading a model failed");
                }
                Err(error) => tracing::warn!(%error, "warming: the load panicked"),
            }
            session
                .warming
                .lock()
                .expect("the warming set is poisoned")
                .remove(&key);
        });
    }

    pub fn database(&self) -> &Database {
        &self.database
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Gives back what this process has held open without being asked.
    ///
    /// Returns what it closed: the projects, then whether the embedder and
    /// the reranker went, so a caller can say so in a log rather than guess.
    ///
    /// **The order is the whole of it and cannot be swapped.** An engine holds
    /// the embedder for as long as it is open, so releasing models
    /// first releases nothing; the indexes have to go before the weights they
    /// pin become releasable. Doing it in one pass means an idle server gives
    /// everything back on one tick rather than over two.
    ///
    /// What each half is worth, measured on this machine: an open index is
    /// about 205 MB with the shipping model, and the constant above says what
    /// sixteen of them did to two benchmark servers; the embedder is 1,625 MB
    /// against 29 MB for a server that has not loaded one.
    ///
    /// Only the server calls this. A command that exits after one search gives
    /// everything back by exiting, and closing an index it is about to use
    /// again would be the opposite of the point.
    pub fn close_what_is_idle(&self) -> (Vec<String>, bool, bool) {
        let idle = self.models.settings().idle();
        let engines = self.engines.close_idle(idle);
        let embedder = self.models.release_idle_embedder();
        let reranker = self.models.release_idle_reranker();
        (engines, embedder, reranker)
    }

    /// How many indexes the open-index bound has closed since the last call.
    ///
    /// The bound closes one on the way to opening another, inside a request,
    /// so giving the freed heap back is left to the server's upkeep.
    pub fn take_evicted(&self) -> usize {
        self.engines.take_evicted()
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

    /// An engine for this project, opening one if there is none.
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
    /// downloads the model's weights the first time anyone wants them, so a
    /// cold project stalled every project this process was serving, including
    /// ones already open. Two cold projects still wait for each other
    /// inside [`Models`], which is where waiting for weights belongs.
    ///
    /// Asked for under the vector index its index was built with: an open
    /// engine asked for under the other one is
    /// refused the way opening its index would be, rather than answering as
    /// what it is not.
    pub async fn engine(&self, project: &str, vector_index: VectorIndex) -> Result<Arc<Engine>> {
        let engine = self
            .engines
            .get_or_open(project.to_string(), || {
                Engine::attached(
                    self.database.clone(),
                    &self.models,
                    &self.workspace,
                    project,
                    vector_index,
                    Access::ReadWrite,
                )
            })
            .await?;
        if engine.vector_index() != vector_index {
            return Err(IndexError::VectorIndexMismatch {
                indexed: engine.vector_index().label().to_string(),
                requested: vector_index.label().to_string(),
            }
            .into());
        }
        Ok(engine)
    }

    /// The engines this process currently holds open.
    ///
    /// For the maintenance loop, which has no project of its own to work on:
    /// upkeep is owed by whatever has been written, and what has been written
    /// recently is what is open. A project evicted before its upkeep ran keeps
    /// the job -- nothing is lost, it waits until the project is wanted again,
    /// which is also when it starts mattering again. That is maintenance
    /// only: work that changes what a search finds is caught up whether the
    /// project is open or not, which the server finds from the queue instead.
    /// Handed out one at a time rather than all at once, because holding every
    /// engine for the length of a sweep makes all of them look busy and stops
    /// eviction finding anything to close while it runs.
    pub fn opened_projects(&self) -> Vec<String> {
        self.engines.keys()
    }

    /// One open engine, if it is open and nobody else is inside it.
    ///
    /// Never opens one: a caller working through what is open should not be
    /// what reopens a project nobody asked for.
    pub fn opened_engine(&self, project: &str) -> Option<Arc<Engine>> {
        self.engines.opened(&project.to_string())
    }

    /// One open engine to do owed work on, if nobody else is inside it.
    ///
    /// [`Session::opened_engine`], except that it counts as a use: work owed
    /// is a reason to keep the index, so the idle sweep waits for the work to
    /// be done and then an idle window, as it would after a request.
    pub fn opened_engine_for_work(&self, project: &str) -> Option<Arc<Engine>> {
        self.engines.opened_for_work(&project.to_string())
    }

    /// Whether this project has a place here, open or opening.
    ///
    /// Not a use, like [`Session::opened_engine`]: asking must not keep a
    /// project open that nothing is using.
    pub fn holds(&self, project: &str) -> bool {
        self.engines.holds(&project.to_string())
    }

    /// An engine with this project's index discarded first, for a rebuild.
    ///
    /// Evicts rather than reuses: the rebuild throws the collection away and
    /// opens a new one, so an engine held from before points at a directory
    /// that is gone.
    pub async fn rebuilding(
        &self,
        project: &str,
        vector_index: VectorIndex,
    ) -> Result<Arc<Engine>> {
        self.engines
            .reopen(project.to_string(), || {
                Engine::rebuilding_attached(
                    self.database.clone(),
                    &self.models,
                    &self.workspace,
                    project,
                    vector_index,
                )
            })
            .await
    }
}
