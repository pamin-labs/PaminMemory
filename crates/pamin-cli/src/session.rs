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
use pamin_index::{Access, Profile, Rerank};
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
    /// Projects being warmed in the background, so a burst of requests at one
    /// cold project starts one warm-up rather than one each. See
    /// [`Session::warm`].
    warming: std::sync::Mutex<std::collections::HashSet<(String, Profile)>>,
    /// The reranking tier a warm-up loads: the one the last search asked for,
    /// or before any has, what `pamin search` would pass by default.
    tier: std::sync::Mutex<Rerank>,
}

/// How many indexes stay open at once.
///
/// An open index is not free and does not become free by being idle. It holds
/// 27 file descriptors, which is why the bound exists: against the
/// 1024-descriptor limit a process usually starts with, an unbounded server
/// ran out at the thirty-seventh project and the engine refused to create its
/// lexical indexer. Sixteen leaves that a wide margin and is large enough that
/// a server round-robining a working set keeps it; the seventeenth project
/// closes the one nobody has touched for longest rather than failing.
///
/// The descriptors are the only quantity this number bounds, and they are not
/// the one that hurts. Memory is, and it does not follow the count:
///
/// | profile    | per index | sixteen of them |
/// | ---------- | --------- | --------------- |
/// | `speed`    | ~100 MB   | ~1.6 GB         |
/// | `accuracy` | ~205 MB   | ~3.3 GB         |
///
/// The second row was measured the hard way. Two benchmark servers on the
/// shipping profile reached 5.6 GB resident apiece -- sixteen indexes on top
/// of a 2.3 GB floor of model and runtime -- and the kernel killed both. The
/// figure the paragraph above used to quote, 100 MB, was the smallest
/// profile's, and nothing said so.
///
/// Most of the first row is a floor rather than a function of size. The
/// full-text store is a RocksDB with twelve column families, and zvec gives
/// each one's memtable a hash index of a million buckets, an 8,000,000-byte
/// array allocated and zeroed when the memtable is: 96 MB before the first
/// memory is written. A one-memory `speed` index measured 93 MiB anonymous
/// (98 MB), and nothing pamin configures changes the bucket count. In the
/// hundred-project end-to-end test, one-memory `accuracy` indexes added 103 to
/// 152 MiB apiece while the first sixteen opened.
///
/// A count cannot be made to mean bytes here: what an index costs depends on
/// the profile, the quantization, and how much has been written, and this
/// process learns the last of those only by opening it. So the number stays a
/// count, and [`OPEN_INDEXES_VAR`] lets whoever knows their machine say a
/// smaller one. Sizing it from a memory budget is [ADR 0001]'s hot-set
/// residency work, which is where it belongs.
const OPEN_INDEXES: usize = 16;

/// Overrides [`OPEN_INDEXES`], for a machine too small for the default.
///
/// Read once, at open. A value that is not a positive number is ignored rather
/// than refused: this is a tuning knob on a long-running process, and failing
/// to start over a typo in an environment variable is the worse failure.
const OPEN_INDEXES_VAR: &str = "PAMIN_OPEN_INDEXES";

/// The tier `pamin search` passes when a caller names none: `PAMIN_RERANK`
/// where the server was started with it, as a client started from the same
/// environment reads it, and the shipped default otherwise.
fn default_tier() -> Rerank {
    std::env::var("PAMIN_RERANK")
        .ok()
        .and_then(|name| Rerank::parse(&name))
        .unwrap_or_default()
}

fn open_indexes() -> usize {
    parse_open_indexes(std::env::var(OPEN_INDEXES_VAR).ok().as_deref())
}

/// Split from the lookup so it can be tested without setting a variable the
/// rest of the process shares.
fn parse_open_indexes(raw: Option<&str>) -> usize {
    raw.and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(OPEN_INDEXES)
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
            engines: Registry::with_capacity(open_indexes()),
            warming: std::sync::Mutex::default(),
            tier: std::sync::Mutex::new(default_tier()),
        })
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
    pub fn warm(self: &Arc<Self>, project: &str, profile: Profile) {
        let key = (project.to_string(), profile);
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
        let tier = *self.tier.lock().expect("the tier lock is poisoned");
        let session = Arc::clone(self);
        tokio::spawn(async move {
            let models = session.models.clone();
            let loading = tokio::task::spawn_blocking(move || models.warm(profile, tier));
            if let Err(error) = session.engine(&key.0, profile).await {
                tracing::debug!(project = %key.0, %error, "warming: opening the project failed");
            }
            match loading.await {
                Ok(Ok(())) => tracing::debug!(project = %key.0, tier = tier.name(), "warmed"),
                Ok(Err(error)) => {
                    tracing::debug!(project = %key.0, %error, "warming: loading a model failed");
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

    /// Records the tier a search asked for, as the one to warm next.
    pub fn searched_at(&self, tier: Rerank) {
        *self.tier.lock().expect("the tier lock is poisoned") = tier;
    }

    pub fn database(&self) -> &Database {
        &self.database
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// Gives back what this process has held open without being asked.
    ///
    /// Returns what it closed: the project-and-profile keys, then the model
    /// profiles and reranker tiers, so a caller can say so in a log rather
    /// than guess.
    ///
    /// **The order is the whole of it and cannot be swapped.** An engine holds
    /// its profile's embedder for as long as it is open, so releasing models
    /// first releases nothing; the indexes have to go before the weights they
    /// pin become releasable. Doing it in one pass means an idle server gives
    /// everything back on one tick rather than over two.
    ///
    /// What each half is worth, measured on this machine: an open index is
    /// about 205 MB on the shipping profile, and the constant above says what
    /// sixteen of them did to two benchmark servers; the embedder is 1,625 MB
    /// against 29 MB for a server that has not loaded one, and the `fast`
    /// reranker another 380 to 645 MB.
    ///
    /// Only the server calls this. A command that exits after one search gives
    /// everything back by exiting, and closing an index it is about to use
    /// again would be the opposite of the point.
    pub fn close_what_is_idle(&self) -> (Vec<(String, Profile)>, Vec<Profile>, Vec<Rerank>) {
        let idle = pamin_engine::model_idle();
        let engines = self.engines.close_idle(idle);
        let embedders = self.models.release_idle_embedders();
        let rerankers = self.models.release_idle_rerankers();
        (engines, embedders, rerankers)
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

#[cfg(test)]
mod tests {
    use super::{OPEN_INDEXES, parse_open_indexes};

    #[test]
    fn a_smaller_machine_can_ask_for_fewer_open_indexes() {
        assert_eq!(parse_open_indexes(Some("4")), 4);
        assert_eq!(parse_open_indexes(Some("  4 ")), 4);
    }

    #[test]
    fn anything_that_is_not_a_count_leaves_the_default_alone() {
        // Zero would evict whatever was just opened, and a long-running server
        // that refuses to start over a typo in an environment variable has
        // failed worse than one that ignores it.
        for raw in [None, Some(""), Some("0"), Some("-1"), Some("lots")] {
            assert_eq!(
                parse_open_indexes(raw),
                OPEN_INDEXES,
                "{raw:?} should not have changed the bound"
            );
        }
    }
}
