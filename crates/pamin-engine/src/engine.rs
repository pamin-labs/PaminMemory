//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection are kept apart everywhere else in the
//! codebase; this is the one place that holds both, so it is also the only
//! place where the two can drift out of step.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Result;
use pamin_core::{
    Channel, ChannelResults, EdgeKind, FilterDecision, FusedResult, Fusion, JobKind, ProjectId,
    Scored, SourceKind, Topic, TopicId, TopicState, TopicStateId, Validity, Why,
};
use pamin_index::{
    Access, Embedder, Previous, Profile, Projection, ProjectionIndex, Rerank, Reranker,
};
use pamin_store::graph::{EdgeClaim, Expansion, Neighbor};
use pamin_store::{
    Connections, Database, PgConnection, PgExecutor, Workspace, graph, jobs, repository,
};
use time::OffsetDateTime;

/// How deep each channel reaches before fusion.
///
/// These are inputs rather than constants because they are provisional: the
/// evaluation harness exists to settle them, and it cannot sweep a value that
/// is compiled in. The defaults are the ones the architecture specifies, so
/// nothing changes for a caller that does not ask.
#[derive(Clone, Copy, Debug)]
pub struct Depths {
    /// Candidates each channel contributes.
    ///
    /// Deep enough for rank fusion to find agreement between channels, shallow
    /// enough that reranking stays cheap.
    pub channel: u32,
    /// Edges the graph channel walks out from its seeds.
    ///
    /// Two hops reaches a topic's neighbours and their neighbours, which is
    /// where "related to something related to this" stops being informative
    /// and starts being most of the project.
    pub graph: u8,
}

impl Default for Depths {
    fn default() -> Self {
        Self {
            channel: 50,
            graph: 2,
        }
    }
}

/// How much weight a derived mention carries against an asserted edge.
///
/// A rule matching a name is weaker evidence than somebody saying two things
/// are related, and the gap has to be expressed somewhere or the two become
/// interchangeable. Provisional, like every other retrieval constant here.
const MENTION_CONFIDENCE: f32 = 0.5;

/// How many topics the graph channel is willing to walk out from.
///
/// Not a query count -- the walk asks one question per hop for the whole
/// frontier, however many seeds it started from. What a seed costs is a place
/// in that frontier: walks are tracked per (topic, seed) pair, so one neighbour
/// reached from sixty-four seeds takes sixty-four of the frontier's places and
/// still produces one result. Without a bound here, how far the walk reaches is
/// decided by how many topics the other channels happened to surface.
const MAX_SEEDS: usize = 64;

/// What one more hop does to how strongly the graph vouches for a topic.
///
/// The graph channel is the one channel with no score against the query -- it
/// reaches a topic across edges rather than matching it -- but it is not
/// without a measure of quality. Every edge carries the confidence whoever or
/// whatever asserted it, and every arrival carries how far the walk went. A
/// topic one asserted edge away from something the query found is a stronger
/// claim than one three derived edges away, and until now fusion could not tell
/// those apart: both arrived as a position in a list and were scored `weight /
/// (k + rank)` on that alone. The edge strength never entered fusion at all.
///
/// A half per hop, and it is a stated choice rather than a measured one. The
/// walk stops at two hops by default, so the only comparison this constant
/// decides is a one-hop arrival against a two-hop one at the same confidence,
/// and halving is the plainest way to say that the second is worth less.
const HOP_DECAY: f32 = 0.5;

/// How many states a rebuild embeds and writes at a time.
///
/// Bounded so the peak memory of a rebuild follows the batch rather than the
/// project: at a thousand states the embeddings alone are already megabytes,
/// and the whole point of the batch is that it does not have to be the whole
/// project.
const REINDEX_BATCH: usize = 256;

/// How many candidates the index proposes when a new topic is backfilled.
///
/// Every one costs a row read and an exact comparison, and the probe is a
/// conjunction of the name's own words, so a real match that does not make this
/// list is a memory outranked by hundreds of others carrying the same full
/// name. Generous rather than tuned: this runs once in a topic's life.
const BACKFILL_CANDIDATES: u32 = 500;

/// Runs synchronous index and embedder work off the async path.
///
/// The projection engine and ONNX Runtime are both synchronous C libraries. A
/// call into either can take tens of milliseconds -- a forward pass, a lexical
/// scan, taking the index's file lock -- and calling one directly from an
/// async function runs it on a runtime thread, where it stalls every other task
/// that thread was driving. Today that is one command's own work. Once a server
/// holds the runtime it is every caller's.
///
/// `block_in_place` rather than `spawn_blocking` because the work borrows the
/// index and the embedder, and neither is `'static`. It requires the
/// multi-threaded runtime, which is what `pamin` runs on.
pub(crate) fn off_the_runtime<T>(work: impl FnOnce() -> T) -> T {
    tokio::task::block_in_place(work)
}

/// The store, the index, and the embedder, wired together.
///
/// Cheap to clone, and every method takes `&self`, so one engine serves many
/// requests at once rather than one at a time. That is what a resident server
/// needs and what a short-lived command never did: before, each of these was
/// opened, used once, and dropped.
#[derive(Clone)]
pub struct Engine {
    pub database: Database,
    /// Who this process is when it claims cascade work. Distinct per process so
    /// a claim identifies its holder, which is what completion checks against.
    pub(crate) worker: String,
    /// Behind the trait rather than the concrete type, so the composition layer
    /// names what it needs from a projection and not which engine provides it.
    ///
    /// `Send + Sync` on the object as well as on the concrete type: erasing the
    /// type erases the auto traits with it, and a resident server serves one
    /// engine from whichever runtime thread takes the request.
    ///
    /// Behind a lock even though every method on the trait takes `&self`. The
    /// engine declares `Sync` and does not honour it: a reader takes an
    /// unsynchronized snapshot of the segments a writer is in the middle of
    /// changing, reported upstream as alibaba/zvec#714. One caller at a time,
    /// not readers sharing -- see [`Engine::index`] for what decided that.
    ///
    /// Not a precaution. Taking this lock out makes searches fail inside a
    /// minute under the concurrency `readers_and_writers_share_one_index_
    /// without_bringing_it_down` puts through it, and that is the mild form --
    /// upstream reports the same race faulting. On macOS it is latent, so it
    /// looks like a precaution there.
    ///
    /// **Merged upstream and not released.** #714 closed with alibaba/zvec#715,
    /// merged as `515c11a`, which gives the segment locks shared readers; the
    /// sibling report #724 closed with #731 as `31d88ea`. The newest published
    /// version is 0.7.0, which predates both, so what this depends on is still
    /// the code that needs the lock. Neither touched a public header, so
    /// picking them up is a version bump rather than a binding change -- see
    /// the deferred entry in `docs/adr/0001-tech-selection.md` for the trigger
    /// and for what has to be measured before the lock comes off.
    ///
    /// A reshape takes the same lock a batch at a time and, for the length of
    /// one, replaces what it holds -- see [`reshape`](Self::reshape).
    ///
    /// [`Engine::index`]: Self::index
    pub(crate) index: Arc<pamin_index::Held>,
    /// Where this project's index lives, which a reshape builds beside and
    /// swaps into.
    pub(crate) dir: std::path::PathBuf,
    /// How the index was opened. Only a writer may reshape it: a reshape moves
    /// the directory, and a reader shares it with another process.
    pub(crate) access: Access,
    /// What a rebuild holds until [`reindex`](Self::reindex) has finished.
    rebuilding: Arc<Mutex<Option<Rebuilding>>>,
    /// Index writes that are applied but not yet on disk, with their claims.
    ///
    /// The projection buffers a write in memory and a query reads that buffer,
    /// so a memory is findable the moment it is upserted -- verified against
    /// every channel, and against a reopen. What the buffer is not is durable:
    /// `upsert` reaches the engine's log with no `fsync` behind it, and only
    /// `flush` calls one. So a write survives this process being killed and
    /// would not survive the machine losing power.
    ///
    /// That is the whole of what a flush buys, and paying for it per write is
    /// the most expensive thing in the write path: 39 ms, against 0.15 ms a
    /// document when the engine is left to materialize on its own schedule.
    /// Worse than the latency, it interrupts that schedule -- two thousand
    /// memories flushed one at a time leave 10,031 index files and 2.2 GB
    /// resident, and left alone leave 25 files and 4 MB.
    ///
    /// So the flush is amortized, and a job stays claimed until one covers it.
    /// That is what keeps the outbox's promise without relying on the engine's:
    /// power lost here leaves these jobs owed and the ledger replays them. The
    /// same is true of this process going away with the list non-empty, or of
    /// the project being evicted from the registry -- the claims lapse and the
    /// work comes round again.
    unflushed: Arc<Mutex<Vec<jobs::Job>>>,
    /// The longest topic name in this project, in tokens, as far as this
    /// process has seen.
    ///
    /// A search asks for it before anything else it does, to know how wide a
    /// window of the query could be a name. The answer changes only when a
    /// topic is created whose name is wider than any before it -- names are
    /// immutable and nothing deletes them -- so it only ever grows, and a
    /// remembered value is either right or too low.
    ///
    /// Too low is not symmetrical with too high, which is why this is only
    /// consulted by the search path. Too high costs windows that match
    /// nothing, because a run is compared against stored names by equality.
    /// Too low means a name is never looked for -- and on the write path that
    /// is not a missed edge but a **retracted** one, because deriving mentions
    /// asserts what it found and then closes everything it did not. A search
    /// that misses a graph seed is right again on the next query; an edge
    /// retracted against a name nobody looked for stays gone.
    widest_name: Arc<std::sync::atomic::AtomicUsize>,
    /// How this splits text, held here rather than reached through the index.
    ///
    /// It is the index's segmenter -- taken from it at open, so the two cannot
    /// drift -- but splitting text touches nothing the index owns, and the lock
    /// above is there for an engine defect that has no bearing on it. Reaching
    /// it through the index meant five callers took that lock for work the
    /// index was not doing, and two of them held it a long time: segmenting up
    /// to [`BACKFILL_CANDIDATES`] documents, and segmenting every topic name in
    /// the project. Every search on the project queued behind them.
    segmenter: Arc<pamin_index::segmentation::Segmenter>,
    /// One model, and one caller into it at a time -- once anything wants one.
    ///
    /// Inference wants `&mut`, which is the only reason anything here ever
    /// needed `&mut self`. Putting it behind its own lock rather than the
    /// index's is what lets several searches read the index at once while one
    /// of them is embedding.
    ///
    /// **Empty until something embeds, which is why `pamin link` can go
    /// through this crate at all.** Loading was part of opening an engine, so
    /// opening one cost 1,625 MB of resident memory and about four and a half
    /// seconds -- and `read`, `grep`, `topics`, `link`, `unlink` and
    /// `neighbors` reached around the engine into the store rather than pay it,
    /// which is how `pamin-cli` came to hold both layers that
    /// `docs/architecture.md` says only this crate holds. A command that never
    /// embeds now never loads: a server serving those alone stays at 29 MB.
    ///
    /// Shared with every other engine on the same profile, which is why it
    /// comes from [`Models`] rather than being built here -- two projects are
    /// two indexes and one model, and the registry is where two callers
    /// arriving at a cold profile queue rather than both loading it. This cell
    /// is a per-engine cache of that lookup, so a search does not take the
    /// registry's lock to find what it already has.
    embedder: std::sync::OnceLock<Arc<Mutex<Embedder>>>,
    /// Which model this engine's index was built with, and therefore the only
    /// one it may embed with. Held because the load is deferred and the
    /// deferred load has to ask for the same profile the index recorded.
    pub(crate) profile: Profile,
    /// Where a reranker comes from, if a search asks for one.
    ///
    /// The registry rather than a loaded model: most searches do not rerank,
    /// most workspaces never will, and half a gigabyte should not be read off
    /// disk by opening a project.
    models: Models,
    pub project: ProjectId,
}

/// What a rebuild holds from opening its index until
/// [`reindex`](Engine::reindex) has finished with it.
struct Rebuilding {
    /// The index it replaced, to lend its vectors. See [`Previous`].
    previous: Option<Previous>,
    /// This project's index directory, held against a reshape.
    _exclusive: tokio::sync::OwnedMutexGuard<()>,
}

/// The embedding models this process has loaded, one per profile.
///
/// Shared between projects rather than held by each. The weights are the same
/// hundreds of megabytes whichever project asks for an embedding, so a process
/// serving a hundred projects on one profile holds one model, not a hundred.
/// Keyed by profile because that is what decides which weights these are; the
/// project decides nothing about them.
#[derive(Clone)]
pub struct Models {
    dir: std::path::PathBuf,
    loaded: Arc<Mutex<std::collections::HashMap<Profile, Held<Embedder>>>>,
    /// The same arrangement for rerankers, keyed by tier for the same reason:
    /// the tier is what decides which weights these are.
    ///
    /// Each carries when it was last handed out, because unlike an embedder a
    /// reranker can stop being wanted. Every search needs a query vector; a
    /// reranker is a tier a caller chose once, and a workspace in one language
    /// is told by `docs/cli.md` to choose `off`.
    rerankers: Arc<Mutex<std::collections::HashMap<Rerank, Held<Reranker>>>>,
}

/// A loaded model and when it was last handed out.
type Held<T> = (Instant, Arc<Mutex<T>>);

/// How long a model may sit unused before the process gives it back.
///
/// It is worth giving back because of what it costs while it sits there, and
/// because the giving back was measured rather than assumed -- dropping a
/// model frees it to the allocator, which is not the same as to the operating
/// system. On a resident server over a 13,014-document project, four cores:
///
/// | | resident |
/// | --- | --- |
/// | serving `--rerank off` | 1,625-1,626 MB, stable to a megabyte |
/// | after one `fast` search | 2,007 MB or 2,271 MB |
/// | after this releases it | 1,627-1,903 MB |
///
/// **One `fast` search costs either about 380 MB or about 645 MB of resident
/// memory for a 130 MB model cache**, and the bimodality is the runtime's
/// arenas rather than the weights -- six samples over one fixed query pair,
/// three at each value, so it is not the shortlist either. Releasing gives
/// back 368 to 380 MB of it, three samples, consistently: the model's own
/// footprint returns to the operating system and the arena growth above it
/// does not. So this reclaims about 380 MB reliably and sometimes all 645,
/// against a server that otherwise holds it for as long as it lives.
///
/// `accurate` is four and a half times the weights and was not measurable
/// here, its download being unreachable from the machine this ran on.
///
/// Thirty minutes, and the number moved once the other side of the trade was
/// measured rather than guessed. Five was written here first, on the strength
/// of "a caller pays about a second" -- **it is 4,528 ms**, median of three,
/// against 116 ms for a warm search and 5,123 ms for a server starting from
/// nothing. The sweep puts the next caller back to 88% of cold, so this is not
/// a second of latency for a gigabyte, it is four and a half.
///
/// Thirty is where that trade stops being close. An agent working in bursts
/// almost never waits half an hour between searches, so it almost never pays;
/// a server left running overnight pays once and gives back 1.6 GB for the
/// hours in between. Five minutes would have charged 4.5 seconds to anyone who
/// stopped to read what they found.
///
/// One window for models and for the indexes that pin them, because the
/// model dominates: reopening an index is part of the 4,528 ms and reloading
/// the weights is most of it.
const MODEL_IDLE: Duration = Duration::from_secs(30 * 60);

impl Models {
    /// Reads and writes the weights a workspace caches.
    pub fn in_workspace(workspace: &Workspace) -> Self {
        Self {
            dir: workspace.root().join("models"),
            loaded: Arc::default(),
            rerankers: Arc::default(),
        }
    }

    /// The model for a profile, loading it the first time it is asked for.
    ///
    /// Blocking, and the registry lock is held across the load. That makes a
    /// second caller for the same profile wait out the first one's download
    /// instead of starting its own, which is the whole point; the wait it pays
    /// is the wait it would have paid loading its own copy.
    fn get(&self, profile: Profile) -> Result<Arc<Mutex<Embedder>>, pamin_index::IndexError> {
        let mut loaded = self
            .loaded
            .lock()
            .expect("the model registry lock is poisoned");

        if let Some((last_used, embedder)) = loaded.get_mut(&profile) {
            *last_used = Instant::now();
            return Ok(Arc::clone(embedder));
        }

        let embedder = Arc::new(Mutex::new(Embedder::load(profile, &self.dir)?));
        loaded.insert(profile, (Instant::now(), Arc::clone(&embedder)));
        Ok(embedder)
    }

    /// The reranker for a tier, loading it the first time it is asked for.
    ///
    /// Lazily rather than with the project: a workspace that never reranks
    /// never downloads one, and the tier is chosen per search.
    fn reranker(&self, tier: Rerank) -> Result<Arc<Mutex<Reranker>>, pamin_index::IndexError> {
        let mut rerankers = self
            .rerankers
            .lock()
            .expect("the reranker registry lock is poisoned");

        if let Some((last_used, reranker)) = rerankers.get_mut(&tier) {
            *last_used = Instant::now();
            return Ok(Arc::clone(reranker));
        }

        let reranker = Arc::new(Mutex::new(Reranker::load(tier, &self.dir)?));
        rerankers.insert(tier, (Instant::now(), Arc::clone(&reranker)));
        Ok(reranker)
    }

    /// What a loaded reranker has been asked to do, or `None` if this process
    /// never loaded that tier.
    ///
    /// The handle is cloned out from under the registry lock before the
    /// reranker's own lock is taken, so asking this while a search is in the
    /// middle of a forward pass waits for that pass rather than for every
    /// other tier as well.
    fn counted(&self, tier: Rerank) -> Option<pamin_index::Reranked> {
        let held = {
            let rerankers = self
                .rerankers
                .lock()
                .expect("the reranker registry lock is poisoned");
            let (_, reranker) = rerankers.get(&tier)?;
            Arc::clone(reranker)
        };
        Some(
            held.lock()
                .expect("the reranker lock is poisoned")
                .counted(),
        )
    }

    /// Gives back the embedders no open engine is holding any more.
    ///
    /// **Costs 1.6 GB to hold and 29 MB not to.** Measured on a server over
    /// the 13,014-document project, four cores: it is 29 MB resident having
    /// answered a command that touches no model, 1,625 MB after the first
    /// embedding, and 1,628 MB after five more -- so the whole of it is paid
    /// when the weights load and inference adds nothing. A server that served
    /// a burst this morning holds all of it for as long as it lives.
    ///
    /// Releasing it is only half the saving, and the half that does not show
    /// up in `smaps` without the other. Dropping the model returns its pages
    /// to the C heap rather than to the kernel: resident falls from 2,263 MB
    /// to 1,000, and the gigabyte that stays is free heap the allocator is
    /// holding. `server.rs` asks for that back with `malloc_trim`, after which
    /// it is 88 to 101 MB over three runs -- 96% of it returned.
    ///
    /// The timestamp here is only a lower bound on idleness, and the strong
    /// count is what makes this correct. An embedder is handed out when an
    /// *engine* opens rather than per search, so a busy server can have a
    /// last-used of hours ago -- but every engine holding one keeps the count
    /// above one, so the release cannot take a model anything is using. What
    /// actually decides this is therefore the engine registry closing its idle
    /// entries first; this is the second half of that, and on its own it
    /// releases nothing.
    pub fn release_idle_embedders(&self) -> Vec<Profile> {
        let mut loaded = self
            .loaded
            .lock()
            .expect("the model registry lock is poisoned");

        let now = Instant::now();
        let idle: Vec<Profile> = loaded
            .iter()
            .filter(|(_, (last_used, embedder))| {
                is_idle(*last_used, now, model_idle()) && Arc::strong_count(embedder) == 1
            })
            .map(|(profile, _)| *profile)
            .collect();

        for profile in &idle {
            loaded.remove(profile);
        }
        idle
    }

    /// Gives back the rerankers nothing has asked for lately.
    ///
    /// For a resident process to call on a schedule; a command that exits
    /// after one search has nothing to give back. Returns what it released, so
    /// the caller can say so in a log rather than guess.
    ///
    /// A tier a search is *using* is never released, even if the last hand-out
    /// was long enough ago -- a long rerank over a large shortlist is exactly
    /// that case. Dropping the registry's handle while a search holds its own
    /// would not free anything, it would only make the next search load a
    /// second copy alongside the first, which is the opposite of the point.
    pub fn release_idle_rerankers(&self) -> Vec<Rerank> {
        let mut rerankers = self
            .rerankers
            .lock()
            .expect("the reranker registry lock is poisoned");

        let now = Instant::now();
        let idle: Vec<Rerank> = rerankers
            .iter()
            .filter(|(_, (last_used, reranker))| {
                is_idle(*last_used, now, model_idle()) && Arc::strong_count(reranker) == 1
            })
            .map(|(tier, _)| *tier)
            .collect();

        for tier in &idle {
            rerankers.remove(tier);
        }
        idle
    }
}

/// Whether something last wanted at `last_used` has been unwanted long enough.
///
/// Its own function because it is the whole of the policy, and a policy inside
/// a closure inside a lock is a policy with no test.
fn is_idle(last_used: Instant, now: Instant, idle: Duration) -> bool {
    now.saturating_duration_since(last_used) >= idle
}

/// The idle window, or what [`MODEL_IDLE`] says.
///
/// `PAMIN_MODEL_IDLE`, in seconds, so that a machine where memory is scarcer
/// than a second of latency can say so, and so that a test can shorten it to
/// something it can wait out. Zero is ignored rather than honoured: a zero
/// window releases a model the tick after it loads and turns every search into
/// a model load.
pub fn model_idle() -> Duration {
    std::env::var(MODEL_IDLE_VAR)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(MODEL_IDLE)
}

/// Overrides [`MODEL_IDLE`], in seconds.
pub const MODEL_IDLE_VAR: &str = "PAMIN_MODEL_IDLE";

/// A memory's content hash, as the ledger stores it.
///
/// Part of what recording a memory means rather than a detail of the caller:
/// the ledger uses it to recognise the same content arriving twice, so two
/// callers computing it two ways would be two callers disagreeing about what
/// the same memory is.
fn content_hash(content: &str) -> String {
    use sha2::Digest as _;
    format!("{:x}", sha2::Sha256::digest(content.as_bytes()))
}

impl Engine {
    /// Opens everything a search or a write needs.
    ///
    /// The access mode is the caller's to state. A read-write handle excludes
    /// every other one, readers included, so a command that only queries and
    /// asks for one turns two simultaneous searches into one search and one
    /// wait.
    pub async fn open(
        workspace: &Workspace,
        project: &str,
        profile: Profile,
        access: Access,
    ) -> Result<Self> {
        let database = Database::open(workspace, Connections::PerCommand).await?;
        let models = Models::in_workspace(workspace);
        Self::assemble(
            database, &models, workspace, project, profile, access, false,
        )
        .await
    }

    /// Opens against a database that is already up.
    ///
    /// `Database::open` probes the cluster and runs the migrations, which is
    /// right once per process and wasteful once per request. A caller holding
    /// a database -- which is what a resident server is -- passes it in.
    pub async fn attached(
        database: Database,
        models: &Models,
        workspace: &Workspace,
        project: &str,
        profile: Profile,
        access: Access,
    ) -> Result<Self> {
        Self::assemble(database, models, workspace, project, profile, access, false).await
    }

    /// Rebuilding, against a database that is already up.
    ///
    /// Discarding before opening rather than overwriting in place is what
    /// makes a rebuild a rebuild: an overwrite leaves behind anything the
    /// ledger no longer has, which is the drift the rebuild exists to remove.
    pub async fn rebuilding_attached(
        database: Database,
        models: &Models,
        workspace: &Workspace,
        project: &str,
        profile: Profile,
    ) -> Result<Self> {
        Self::assemble(
            database,
            models,
            workspace,
            project,
            profile,
            Access::ReadWrite,
            true,
        )
        .await
    }

    async fn assemble(
        database: Database,
        models: &Models,
        workspace: &Workspace,
        project: &str,
        profile: Profile,
        access: Access,
        discard: bool,
    ) -> Result<Self> {
        let project = repository::ensure_project(database.pool(), project).await?;

        // The index is per project, so the identity has to be resolved before
        // the index can be located at all.
        let dir = workspace.index_dir(project.id);
        let legacy = workspace.legacy_index_dir();

        // How large a segment should be follows how much there is to hold, and
        // a collection records the answer when it is created, so the count has
        // to be in hand before the index is opened.
        let documents = repository::topic_count(database.pool(), project.id).await?;

        // A rebuild waits out a reshape of this index, and holds it off until
        // `reindex` finishes: both move the directory. Every open first puts
        // back what a reshape that died mid-swap left aside, so a rebuild can
        // lend its vectors and anything else finds the index rather than
        // creating an empty one -- but not while a reshape is running here,
        // when the directory is that reshape's to move.
        let exclusion = crate::reshape::exclusive(&dir);
        let exclusive = if discard {
            Some(exclusion.lock_owned().await)
        } else {
            exclusion.try_lock_owned().ok()
        };
        if exclusive.is_some() {
            off_the_runtime(|| pamin_index::Reshape::recover(&dir))?;
        }
        // Only a rebuild keeps it past the open.
        let exclusive = exclusive.filter(|_| discard);

        let (index, previous) = off_the_runtime(|| {
            // Set aside rather than deleted, so the rebuild can take the
            // vectors it still holds instead of embedding every memory again.
            let previous = if discard {
                let previous = Previous::set_aside(&dir, profile)?;
                // A rebuild is also the migration off the shared layout, which
                // is what the error about it tells the caller to run.
                ProjectionIndex::discard(&legacy)?;
                previous
            } else {
                None
            };

            let index = ProjectionIndex::open(&dir, &legacy, profile, access, documents)?;
            // The model is not loaded here. It was, and that made opening an
            // engine cost the weights -- see the `embedder` field. What is lost
            // is that a cold profile's download used to surface at open rather
            // than at the first embedding; what is gained is every command
            // that does not embed.
            Ok::<_, pamin_index::IndexError>((
                Arc::new(index) as Arc<dyn Projection + Send + Sync>,
                previous,
            ))
        })?;

        Ok(Self {
            database,
            worker: format!("{}:{}", hostname(), std::process::id()),
            // Taken from the index rather than built here, so what this
            // tokenizes with is what the index tokenized with. Two segmenters
            // would be the same code today and a divergence the first time
            // either side changed.
            segmenter: index.segmenter(),
            widest_name: Arc::default(),
            unflushed: Arc::default(),
            index: Arc::new(Mutex::new(index)),
            dir,
            access,
            rebuilding: Arc::new(Mutex::new(exclusive.map(|exclusive| Rebuilding {
                previous,
                _exclusive: exclusive,
            }))),
            embedder: std::sync::OnceLock::new(),
            profile,
            models: models.clone(),
            project: project.id,
        })
    }

    /// The projection. One caller at a time.
    ///
    /// This was a read-write lock, on the reading that queries are read-only
    /// and may as well run together. The engine does not agree. Twenty writers
    /// and twenty readers against one index wedged it inside the engine's own
    /// code -- forty-six of its threads asleep on futexes with no caller of
    /// ours above them, four of ours stopped inside a query, and no processor
    /// time being used by any of them for thirty-five minutes. The same run
    /// with writers alone passes, and the same run with this lock made
    /// exclusive passes for the full five minutes. So concurrent queries are
    /// the thing it cannot do, and an exclusive lock is what it costs to say
    /// so.
    ///
    /// It costs less than it sounds. A search spends about half its time in a
    /// forward pass, and the model is behind a mutex already, so two searches
    /// were never going to overlap by much; what is given up is a few
    /// milliseconds of index work per query, and only between callers sharing
    /// one project.
    ///
    /// # Lock order
    ///
    /// Anything taking both the model and the index takes the **model first**.
    /// Reversing it is what deadlocks: a rebuild holding the index and waiting
    /// for the model, against a search holding the model and waiting for the
    /// index. Neither is doing anything wrong on its own, which is why the
    /// order is written here rather than left to be noticed.
    ///
    /// A poisoned lock means a previous request panicked while holding the
    /// index, so what it holds is whatever that panic left. Failing here is
    /// the honest outcome: the alternative is serving from state nobody
    /// finished writing.
    pub(crate) fn index(&self) -> MutexGuard<'_, Arc<dyn Projection + Send + Sync>> {
        self.index.lock().expect("the index lock is poisoned")
    }

    /// What text this project's vectors are embedded from. Public because a
    /// measurement comparing two encodings has to be able to assert which one
    /// each side holds.
    pub fn passage(&self) -> pamin_index::Passage {
        self.index().passage()
    }

    /// What share of this project's documents the vector graph covers.
    ///
    /// Public because it is the premise of every retrieval measurement taken
    /// through this engine, and an unasserted premise is how a vector channel
    /// came to answer from an exhaustive scan for a whole release while the
    /// HNSW parameters it was configured with described a structure nothing
    /// had built. A latency or recall figure taken below 1.0 here is a figure
    /// for a different object, so a harness that cannot read this cannot say
    /// what it measured.
    pub fn vector_index_completeness(&self) -> Result<f32> {
        Ok(off_the_runtime(|| {
            self.index().vector_index_completeness()
        })?)
    }

    /// How many documents this project's projection holds.
    ///
    /// Public for the same reason as
    /// [`vector_index_completeness`](Self::vector_index_completeness), and it
    /// is the half that stops the other one passing vacuously: an *empty*
    /// projection reports a completeness of 1.0, because everything it holds
    /// is indexed and it holds nothing. So "the graph covers everything" is
    /// only a claim about a graph once something is in there.
    /// How this project's index is segmented, against what the policy wants.
    ///
    /// Resegmenting means recreating the collection --
    /// `set_max_doc_count_per_segment` on an open one returns `Ok` and changes
    /// nothing, which ADR 0001 records. [`reshape`](Self::reshape) does that
    /// by copying the index while it is served, and a server runs it on its
    /// own; without a server, `pamin reindex` is what fixes it.
    pub fn segmentation(&self) -> Result<pamin_index::Segmentation> {
        Ok(self.index().segmentation()?)
    }

    pub fn indexed_documents(&self) -> Result<u64> {
        Ok(off_the_runtime(|| self.index().document_count())?)
    }

    /// How wide the widest topic name in this project is, in tokens.
    ///
    /// Read once and remembered. It is asked at the top of every search -- to
    /// decide how long a run of the query could be a name -- and it answers a
    /// `MAX` over a column that only grows: names cannot be renamed and nothing
    /// deletes them, so a value this process has seen can only become stale by
    /// being too low, and only when some other writer creates a wider name.
    ///
    /// Too low costs a search one graph seed, which the next search gets right
    /// once this process sees the wider name itself. That is the whole reason
    /// this is on the search path and not on the write path, where the same
    /// staleness would retract edges rather than miss them.
    async fn widest_name(&self) -> Result<usize> {
        use std::sync::atomic::Ordering;

        let known = self.widest_name.load(Ordering::Relaxed);
        if known > 0 {
            return Ok(known);
        }

        let widest = repository::widest_topic_name(self.database.pool(), self.project).await?;
        self.remember_widest_name(widest);
        Ok(widest)
    }

    /// Raises what this process believes the widest name to be.
    ///
    /// Never lowers it. Two writers racing here both win: the larger stands,
    /// which is the direction that cannot lose a lookup.
    pub(crate) fn remember_widest_name(&self, tokens: usize) {
        self.widest_name
            .fetch_max(tokens, std::sync::atomic::Ordering::Relaxed);
    }

    /// The claims held open by writes the index has and the disk does not.
    pub(crate) fn unflushed(
        &self,
    ) -> std::sync::LockResult<MutexGuard<'_, Vec<pamin_store::jobs::Job>>> {
        self.unflushed.lock()
    }

    /// The projection, for the one thing that may run while others use it.
    ///
    /// Compaction and graph building are the long operation in this system --
    /// a third of a second for a few hundred files, minutes for a graph over a
    /// sealed segment -- and they make nothing more correct, only faster. That
    /// is what makes them safe to run alongside a query, and the engine agrees:
    /// alibaba/zvec#614 turned Optimize into a brief exclusive seal, a long
    /// phase holding no schema lock, and a brief exclusive commit, so reads and
    /// writes proceed through the middle of it. Unlike the concurrency fix this
    /// index is still waiting on, **that one shipped**, in the 0.7.0 this
    /// depends on.
    ///
    /// So holding [`index`] across it was our own exclusion, not the engine's,
    /// and it was the whole reason moving upkeep off the writer bought nothing:
    /// a write waited out a compaction whether or not it was the one running
    /// it. Taking a handle instead lets the upkeep worker compact while the
    /// searches it is speeding up are still being answered.
    ///
    /// Only this operation gets it. Every other call goes through [`index`],
    /// because #714 is what that lock is for and it is still unfixed here.
    ///
    /// [`index`]: Self::index
    pub(crate) fn index_for_upkeep(&self) -> Arc<dyn Projection + Send + Sync> {
        Arc::clone(&self.index())
    }

    /// The model, loading it the first time anything asks.
    ///
    /// One caller at a time, because inference wants `&mut`. Fallible now that
    /// the load is deferred, and every caller is already inside
    /// [`off_the_runtime`] -- which it has to be, since loading hundreds of
    /// megabytes of weights is not something to do on the async runtime.
    pub(crate) fn embedding(
        &self,
    ) -> std::result::Result<std::sync::MutexGuard<'_, Embedder>, pamin_index::IndexError> {
        let held = match self.embedder.get() {
            Some(held) => held,
            None => {
                // Loaded outside the cell's initializer because that cannot
                // fail. Two callers racing here both ask the registry, which
                // hands out the same model to both -- so the loser discards an
                // `Arc` clone rather than a second copy of the weights.
                let loaded = self.models.get(self.profile)?;
                self.embedder.get_or_init(|| loaded)
            }
        };
        Ok(held.lock().expect("the embedder lock is poisoned"))
    }

    /// Adds one topic state to the projection index, without flushing.
    ///
    /// The flush belongs to whoever is running a group of these, not here.
    /// Every buffered write costs one flush and one set of files, and an index
    /// built a document at a time is measurably a different object than the
    /// same documents written in batches: 384 sentences cost 1,161 files and
    /// 1,893 MB flushed one at a time, and 35 files and 61 MB flushed every
    /// thirty-two. Rate went with it, 13.6 documents a second against 328.
    ///
    /// So this leaves the writes buffered and [`drain_cascade`] flushes the
    /// round. `pub(crate)` because that contract cannot be honoured by a caller
    /// outside this crate, which would get an index that never became visible.
    ///
    /// [`drain_cascade`]: Self::drain_cascade
    pub(crate) async fn index_state(&self, state: &TopicState) -> Result<()> {
        let passage = self
            .passages(std::slice::from_ref(state))
            .await?
            .pop()
            .expect("one passage for one state");
        off_the_runtime(|| {
            let embedding = self.embedding()?.embed_passage(&passage)?;
            self.index()
                .upsert(state.topic_id, &state.content, &embedding)
        })?;
        Ok(())
    }

    /// The text each state's vector is embedded from, in the encoding this
    /// project's index was built with ([`pamin_index::Passage`]).
    ///
    /// Asks for the names only when the encoding uses them, and for a whole
    /// batch at once. What the lexical channels index is the content either
    /// way: this is what the vector is shown, not what a word is matched in.
    async fn passages(&self, states: &[TopicState]) -> Result<Vec<String>> {
        let passage = self.index().passage();
        if passage == pamin_index::Passage::Content {
            return Ok(states.iter().map(|state| state.content.clone()).collect());
        }
        let ids: Vec<TopicId> = states.iter().map(|state| state.topic_id).collect();
        let names: std::collections::HashMap<TopicId, String> =
            repository::topics_by_id(self.database.pool(), self.project, &ids)
                .await?
                .into_iter()
                .map(|(topic, name, _)| (topic, name))
                .collect();
        Ok(states
            .iter()
            .map(|state| {
                let name = names.get(&state.topic_id).map_or("", String::as_str);
                passage.render(name, &state.content)
            })
            .collect())
    }

    /// The same for many states, in one forward pass.
    ///
    /// A cascade round claims up to sixty-four jobs and used to index them one
    /// at a time, which is sixty-four forward passes where the model can do one
    /// -- and `Embedder::embed_passages` has been there the whole time, used
    /// only by the rebuild. The flush in the same round was already batched,
    /// with a comment explaining why batching it mattered, so the pass was the
    /// half that got left.
    ///
    /// **Score-neutral, and that is asserted rather than assumed.**
    /// `pamin-index`'s `a_batch_changes_nothing_on` compares `embed_passages`
    /// against `embed_passage` position by position across profiles and
    /// requires the vectors to be identical, so an index built in batches holds
    /// what an index built one at a time would have held. (The reranker's
    /// length-sorted batching is *not* score-neutral and says so where it
    /// lives; that is a different model and a different code path.)
    ///
    /// One model lock and one index lock for the whole batch rather than one
    /// each per state, which is the other saving and the reason this is one
    /// closure rather than a loop over the single-state form.
    pub(crate) async fn index_states(&self, states: &[TopicState]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        let passages = self.passages(states).await?;
        let contents: Vec<&str> = passages.iter().map(String::as_str).collect();
        off_the_runtime(|| {
            let embeddings = self.embedding()?.embed_passages(&contents)?;
            let documents: Vec<(TopicId, &str, &[f32])> = states
                .iter()
                .zip(&embeddings)
                .map(|(state, embedding)| {
                    (state.topic_id, state.content.as_str(), embedding.as_slice())
                })
                .collect();
            self.index().upsert_batch(&documents)
        })?;
        Ok(())
    }

    /// Records one memory: the filter's verdict, its language, and one
    /// transaction.
    ///
    /// Everything a memory costs except the index, which the outbox catches up
    /// on afterwards. This is the write API, and it is here rather than in the
    /// CLI because it *was* in the CLI:
    /// `pamin-cli::command::write::record` was what `import` called too, so a
    /// module of the command layer had become the only way to record a memory
    /// correctly. The interfaces `README.md` lists as unbuilt would each have
    /// had to call it or copy it, and copying it is how two callers start
    /// recording memories by different rules.
    ///
    /// [`write`](Self::write) stays what it was and is what this calls: the
    /// transaction, given a verdict somebody else reached. The difference
    /// between the two is the whole of what this adds -- looking up what the
    /// topic says now, judging the content against it, and detecting the
    /// language.
    pub async fn remember(
        &self,
        topic: &str,
        content: &str,
        validity: pamin_core::Validity,
    ) -> Result<(pamin_core::Verdict, Recorded)> {
        // Looked up rather than created: a write the filter holds should leave
        // no trace on the retrieval surface, and an empty topic is a trace.
        // Promotion is what creates one, inside the write transaction.
        let current =
            repository::current_content(self.database.pool(), self.project, topic).await?;
        let verdict = pamin_core::SensoryFilter::default().judge(content, current.as_deref());

        let (language, confidence) = match pamin_index::detect_language(content) {
            Some((language, confidence)) => (Some(language), Some(confidence)),
            None => (None, None),
        };

        let recorded = self
            .write(&Write {
                topic,
                content,
                content_hash: &content_hash(content),
                verdict: verdict.decision,
                reason: verdict.reason(),
                promoted: verdict.is_promoted(),
                language: language.as_deref(),
                language_confidence: confidence,
                observed_at: OffsetDateTime::now_utc(),
                validity,
            })
            .await?;

        Ok((verdict, recorded))
    }

    /// Records a memory: evidence, the span over it, and -- when the filter
    /// promotes it -- a topic state and the work the projection owes it.
    ///
    /// All of it in one transaction, and that transaction touches nothing
    /// outside PostgreSQL. Deriving the embedding and writing the projection
    /// used to happen here, inline, which meant a write depended on the index
    /// being reachable and on a model being loaded; now the write commits a row
    /// in the outbox saying what is owed, and the cascade pays it. That is the
    /// first thing an outbox buys, before any question of throughput: the write
    /// succeeds or fails on the ledger alone.
    ///
    /// The topic is created only when something is promoted under it, and only
    /// inside this transaction, so a held write leaves no name behind.
    pub async fn write(&self, request: &Write<'_>) -> Result<Recorded> {
        let mut transaction = self.database.pool().begin().await?;

        // Manual writes to one topic share a source, so their evidence forms a
        // single chain rather than a new source per write.
        let source = repository::ensure_source(
            &mut transaction,
            self.project,
            SourceKind::Manual,
            &format!("manual:{}", request.topic),
        )
        .await?;

        // Evidence first, always, and before the filter's verdict is acted on.
        // That ordering is what makes a rejection recoverable instead of a loss.
        let (evidence, span) = repository::append_evidence(
            &mut transaction,
            self.project,
            source,
            &repository::Evidence {
                content: request.content,
                content_hash: request.content_hash,
                decision: request.verdict,
                reason: request.reason,
                language: request.language,
                language_confidence: request.language_confidence,
            },
        )
        .await?;

        let state = if request.promoted {
            let existed =
                repository::find_topic(&mut *transaction, self.project, request.topic).await?;
            let topic = match existed.clone() {
                Some(topic) => topic,
                None => {
                    let topic =
                        repository::create_topic(&mut transaction, self.project, request.topic)
                            .await?;
                    // In the same transaction as the topic. A topic that exists
                    // and is missing from the name index is a topic no memory
                    // will ever derive an edge to, and nothing would report it.
                    self.record_name(&mut *transaction, &topic).await?;
                    topic
                }
            };

            let state = repository::append_topic_state(
                &mut transaction,
                self.project,
                topic.id,
                &evidence,
                &span,
                request.observed_at,
                request.validity,
            )
            .await?;

            // Committed with the state rather than after it. A crash between
            // the two would otherwise leave a memory the ledger knows about and
            // the projection never hears of -- which is the failure an outbox
            // exists to make impossible, and the one a `tokio::spawn` here
            // would leave wide open.
            //
            // All of them in one statement, because this is inside the write
            // transaction: three rows is the right number of rows and was
            // three round trips with the transaction held open across them.
            //
            // The third is for a topic that did not exist a moment ago, which
            // may already be named by memories written before it. Finding them
            // is a scan, so it is scheduled rather than paid for by whoever
            // created the topic -- and it is skipped entirely for a rewrite,
            // which is why the three cannot simply be one kind.
            let mut owed = vec![JobKind::SyncTopicIndex, JobKind::DeriveMentions];
            if existed.is_none() {
                owed.push(JobKind::BackfillMentions);
            }
            jobs::enqueue_all(&mut *transaction, self.project, &owed, Some(topic.id.0)).await?;

            Some(state)
        } else {
            None
        };

        transaction.commit().await?;

        Ok(Recorded {
            source_version: evidence.version,
            state,
        })
    }

    /// Files a topic's name in the index that answers "who is named here".
    ///
    /// Tokenized here rather than in the store because the segmenter is what
    /// decides where a name begins and ends, and both sides of the eventual
    /// comparison have to have gone through it.
    async fn record_name(&self, executor: impl PgExecutor<'_>, topic: &Topic) -> Result<()> {
        let tokens = off_the_runtime(|| self.segmenter.name_sequence(&topic.name));
        repository::record_topic_name(
            executor,
            self.project,
            topic.id,
            &tokens.join(" "),
            tokens.len(),
        )
        .await?;
        // This process now knows of a name at least this wide, whether or not
        // it had asked. Raising it here is what keeps the search path's
        // remembered value from going stale against writes made through it.
        self.remember_widest_name(tokens.len());
        Ok(())
    }

    /// Restates the edges a topic's current content implies.
    ///
    /// In a topic-centred graph the topics are the entities, so a memory naming
    /// another topic is entity linking with no model in the path. Returns how
    /// many edges this call actually added, which is zero when the content is
    /// unchanged: asserting an edge is idempotent, so rewriting a memory does
    /// not grow the ledger.
    ///
    /// Restates rather than adds: what the content no longer names is closed.
    /// The two are separate statements, and a crash between them leaves edges
    /// this content does not claim -- which the next run of the job removes,
    /// because the job asks what the topic says now rather than what changed.
    /// Asserting first is what keeps it cheap: a name still present is found
    /// unchanged and written nowhere, and only then is the rest closed, so
    /// re-deriving an unaltered memory still touches no row.
    pub async fn derive_mentions(&self, state: &TopicState) -> Result<usize> {
        let mut connection = self.database.pool().acquire().await?;
        self.restate_mentions(&mut connection, std::slice::from_ref(state))
            .await
    }

    /// [`derive_mentions`](Self::derive_mentions) for many states, asking each
    /// of its questions once for all of them.
    ///
    /// A cascade round restates up to sixty-four memories, and one at a time
    /// that was five statements each -- the widest name, the names in the
    /// memory, the edges already there, and the retraction -- where each
    /// question is the same for every memory in the round apart from its
    /// arguments. So the runs of every memory go into one lookup, the edges
    /// into one assertion and the retractions into one statement, and the run
    /// each name matched is what says which memory it belongs to.
    pub(crate) async fn restate_mentions(
        &self,
        connection: &mut PgConnection,
        states: &[TopicState],
    ) -> Result<usize> {
        if states.is_empty() {
            return Ok(0);
        }

        // Every run of tokens each memory contains that is short enough to be
        // somebody's name. A name matches only as a contiguous run, so this is
        // the complete set of things it could be naming -- and asking the index
        // for these is the same question the old loop asked of every topic in
        // the project one at a time, with the cost following the length of the
        // memory rather than the size of the project.
        // Asked fresh, never from what this process remembers. A remembered
        // value can only be too low, and too low here does not mean a missed
        // edge -- what is not found below is closed as no longer named. See
        // [`Engine::widest_name`].
        let widest = repository::widest_topic_name(&mut *connection, self.project).await?;
        let runs: Vec<Vec<String>> = off_the_runtime(|| {
            states
                .iter()
                .map(|state| runs_of_tokens(&self.segmenter.name_sequence(&state.content), widest))
                .collect()
        });
        let mut asked: Vec<String> = runs.iter().flatten().cloned().collect();
        asked.sort_unstable();
        asked.dedup();
        let mut naming: std::collections::HashMap<String, Vec<TopicId>> =
            std::collections::HashMap::new();
        for (run, topic) in
            repository::names_matching(&mut *connection, self.project, &asked).await?
        {
            naming.entry(run).or_default().push(topic);
        }

        let mut edges = Vec::new();
        let mut named_now = Vec::with_capacity(states.len());
        for (state, runs) in states.iter().zip(&runs) {
            let mut named: Vec<TopicId> = runs
                .iter()
                .filter_map(|run| naming.get(run))
                .flatten()
                .copied()
                // A topic naming itself is not a relationship, and the schema
                // rejects the edge anyway.
                .filter(|topic| *topic != state.topic_id)
                .collect();
            named.sort_unstable();
            named.dedup();
            edges.extend(named.iter().map(|target| {
                (
                    state.topic_id,
                    *target,
                    EdgeClaim::derived(EdgeKind::Mentions, state.id, MENTION_CONFIDENCE),
                )
            }));
            named_now.push((state.topic_id, named));
        }

        // One transaction: the edges a memory derives are one statement about
        // what it says, and asserting them separately both cost a commit each
        // and let a crash tell half of it.
        let asserted = graph::assert_edges(self.database.pool(), self.project, &edges).await?;

        graph::retract_derived_all(
            &mut *connection,
            self.project,
            EdgeKind::Mentions,
            &named_now,
        )
        .await?;

        Ok(asserted
            .iter()
            .filter(|assertion| assertion.is_new())
            .count())
    }

    /// Topics whose own name appears in this text.
    ///
    /// The other half of "which topic should this go under". `search` answers
    /// it by content, so it finds a topic only when something written under it
    /// matches; this asks the name index, so a topic nobody has written much
    /// about is still findable by what it is called.
    ///
    /// Exact on the segmenter's tokens, not a prefix and not a stem: every run
    /// of words in the text is looked up whole, so `deployment pipeline` finds
    /// `deployment_pipeline` and `deploy pipeline` does not. That is the same
    /// question `derive_mentions` asks of a memory, answered against the same
    /// index, and it is deliberately the strict half of this pair -- the
    /// forgiving half is the content search beside it.
    pub async fn topics_named_like(&self, text: &str, limit: u32) -> Result<Vec<String>> {
        let widest = repository::widest_topic_name(self.database.pool(), self.project).await?;
        let runs = off_the_runtime(|| runs_of_tokens(&self.segmenter.name_sequence(text), widest));
        let candidates =
            repository::topics_named_by(self.database.pool(), self.project, &runs).await?;
        let mut named: Vec<String> =
            repository::topics_by_id(self.database.pool(), self.project, &candidates)
                .await?
                .into_iter()
                .map(|(_, name, _)| name)
                .take(limit as usize)
                .collect();
        named.sort_unstable();
        named.dedup();
        Ok(named)
    }

    /// Links a newly created topic to memories that already named it.
    ///
    /// Without it an edge would appear only when one of those older memories
    /// happened to be rewritten, so a topic would know less about the past the
    /// longer that past was.
    ///
    /// The name is asked of the index rather than compared against every memory
    /// in the project. A scan is exact, and at the size this store is built for
    /// it is also the one operation whose cost is the whole project -- creating
    /// a topic would segment millions of states to find the handful that name
    /// it. So the index proposes candidates and [`segmentation::names`]
    /// disposes: the answer is still the exact one for everything the probe
    /// returns, and what it can miss is a memory ranked below
    /// [`BACKFILL_CANDIDATES`] on a conjunction of the name's own words.
    ///
    /// Only current states are considered. A mentions edge says what a topic
    /// says now -- that is what `derive_mentions` restates and what makes the
    /// edge retractable -- and an edge derived from a superseded version would
    /// be a claim nothing later revisits.
    pub(crate) async fn backfill_mentions(&self, topic: TopicId, name: &str) -> Result<usize> {
        let candidates = off_the_runtime(|| self.index().recall_naming(name, BACKFILL_CANDIDATES))?;

        let states =
            repository::current_states_of(self.database.pool(), self.project, &candidates).await?;

        // The probe returns states; the edge is about topics, and only the
        // state a topic stands for now can support one.
        let topics: Vec<TopicId> = states.iter().map(|state| state.topic_id).collect();
        let current: std::collections::HashSet<pamin_core::TopicStateId> =
            repository::topics_by_id(self.database.pool(), self.project, &topics)
                .await?
                .into_iter()
                .filter_map(|(_, _, current)| current)
                .collect();

        // Off the runtime because this segments every candidate the probe
        // returned -- up to `BACKFILL_CANDIDATES` documents -- and that is tens
        // of milliseconds of ICU work that would otherwise run on a runtime
        // thread and stall every task sharing it.
        let naming: Vec<(TopicId, pamin_core::TopicStateId)> = off_the_runtime(|| {
            let segmenter = &self.segmenter;
            // The fixed side here is the name, so that is the side prepared.
            let name = segmenter.name_sequence(name);
            states
                .iter()
                .filter(|state| state.topic_id != topic)
                .filter(|state| current.contains(&state.id))
                .filter(|state| {
                    pamin_index::segmentation::names(
                        &segmenter.name_sequence(&state.content),
                        &name,
                    )
                })
                .map(|state| (state.topic_id, state.id))
                .collect()
        });

        let edges: Vec<_> = naming
            .into_iter()
            .map(|(from, caused_by)| {
                (
                    from,
                    topic,
                    EdgeClaim::derived(EdgeKind::Mentions, caused_by, MENTION_CONFIDENCE),
                )
            })
            .collect();

        let asserted = graph::assert_edges(self.database.pool(), self.project, &edges).await?;

        Ok(asserted
            .iter()
            .filter(|assertion| assertion.is_new())
            .count())
    }

    /// Search, then reorder the head of the result with a cross-encoder.
    ///
    /// Only the candidates no lexical channel found, and only into the
    /// positions those candidates already hold.
    ///
    /// Fused deeper than it returns, because a reranker that only sees what the
    /// caller asked for has nothing to work with. See [`fused_for`]: the tier's
    /// depth was measured over a list of fifty and the default `--limit` is
    /// five, so cutting first left it reordering five candidates and usually
    /// declining to reorder at all.
    ///
    /// Every cross-encoder measured improves cross-lingual ranking and damages
    /// same-language ranking by about as much: fusion is already good at
    /// placing a memory that shares words with the query, and a second pass
    /// reorders it worse. So the pass is confined to the candidates the
    /// lexical channels did not find -- the ones fusion ordered on the vector
    /// channel alone. Everything else keeps the rank it had, which makes the
    /// damage arithmetically impossible rather than merely unlikely.
    ///
    /// This was a language comparison first, since "written in another
    /// language" is what the case really is. The two pick the same candidates
    /// -- they agree on 93% of a shortlist and score within 0.002 of each
    /// other -- and the language test needed the query's language, which for a
    /// short query is exactly what a detector will not commit to:
    /// `detect_language` returns nothing for "how does deployment work". A
    /// rule that quietly does nothing on the commonest shape of query is worse
    /// than a slightly different rule, and this one asks only what the search
    /// already recorded.
    /// What the reranker at `tier` has done in this process, or `None` if it
    /// was never loaded.
    ///
    /// Here because three deferred decisions turn on these counters and none
    /// of them had a value -- see [`pamin_index::Reranked`]. A caller that
    /// wants them across a run reads them once at the end: they are lifetime
    /// totals for the loaded model and are lost when an idle tier is released.
    pub fn reranked(&self, tier: Rerank) -> Option<pamin_index::Reranked> {
        self.models.counted(tier)
    }

    pub async fn search_reranked(
        &self,
        query: &str,
        limit: u32,
        depths: Depths,
        rerank: Rerank,
    ) -> Result<Vec<SearchHit>> {
        self.search_reranked_with(query, limit, depths, rerank, Fusion::default())
            .await
    }

    /// [`search_reranked`](Self::search_reranked) over a fusion other than
    /// the shipped one, for a measurement that has to hold everything else
    /// the product does fixed -- the reranker above all -- while one channel
    /// is taken out or reweighted.
    pub async fn search_reranked_with(
        &self,
        query: &str,
        limit: u32,
        depths: Depths,
        rerank: Rerank,
        fusion: Fusion,
    ) -> Result<Vec<SearchHit>> {
        // The reranker is needed after retrieval and was loaded only then, so
        // a search that found no model resident paid the embedder's load and
        // then the reranker's, one after the other -- the first search after
        // an idle release was measured at 4,528 ms against 116, at the `fast`
        // tier, and most of that is the two loads. Started here, it loads
        // while the query is
        // embedded and the channels run. The registry holds its lock across a
        // load, so the call below either finds the model resident or waits for
        // this one to finish -- it is never loaded twice. Nothing about the
        // ranking changes; a failure here is the same failure the call below
        // reports, so it is left to that one.
        if rerank != Rerank::Off {
            let models = self.models.clone();
            drop(tokio::task::spawn_blocking(move || models.reranker(rerank)));
        }

        let hits = self
            .search_fused(query, fused_for(limit, rerank), depths, fusion)
            .await?;
        if rerank == Rerank::Off || hits.is_empty() {
            return Ok(hits);
        }

        let traces: Vec<&[Why]> = hits.iter().map(|hit| hit.result.why.as_slice()).collect();
        let unlexical = rerankable(&traces, rerank);
        if !can_be_seen(&unlexical, limit) {
            return Ok(only(hits, limit));
        }

        let shown: Vec<String> = unlexical
            .iter()
            .map(|position| {
                let hit = &hits[*position];
                shown(&hit.topic, &hit.state.content, hit.seed.as_deref())
            })
            .collect();
        let documents: Vec<&str> = shown.iter().map(String::as_str).collect();
        // Finding the reranker is inside this too, not just using it. The
        // first search of a tier downloads its weights, holding the registry
        // lock so that twenty concurrent searches fetch one model rather than
        // twenty -- and a lock held across a download is a lock held for a long
        // time. Taken on a runtime thread, that blocks the thread rather than
        // yielding it, and the callers waiting behind it block their threads
        // too, until the runtime has no thread left to finish the download with
        // and the server stops answering anything at all. Measured: twenty
        // readers against a cold cache wedged it indefinitely.
        let ordered = off_the_runtime(|| {
            self.models
                .reranker(rerank)?
                .lock()
                .expect("the reranker lock is poisoned")
                .rank(query, &documents)
        })?;

        // Recorded on the candidate rather than returned beside the list,
        // because it is an answer to "why is this here" and belongs with the
        // channel entries that answer the same question. A candidate the
        // reranker never saw carries no entry, which is how a reader -- and
        // the threshold sweep this unblocks -- tells the two cases apart.
        let mut hits = hits;
        for ranked in &ordered {
            hits[unlexical[ranked.position]]
                .result
                .why
                .push(Why::Reranked {
                    score: ranked.score,
                });
        }
        let best_first: Vec<usize> = ordered.iter().map(|ranked| ranked.position).collect();
        Ok(only(
            place(hits, &unlexical, rerank.depth(), &best_first),
            limit,
        ))
    }

    /// The same search, with the fusion settings supplied.
    ///
    /// Exists for the same reason [`Depths`] is a parameter: the constants
    /// fusion runs on were settled by measurement and are re-settled the same
    /// way, so the harness that measures them has to be able to vary them.
    /// Callers that are not measuring want
    /// [`search_reranked`](Self::search_reranked), which is what `pamin
    /// search` calls. This used to name a third entry point, one stage
    /// shorter than the shipped one, and both evaluation harnesses took it:
    /// every retrieval figure this project published described a pipeline
    /// that does not ship, and the reranker gains were about half again too
    /// high. That method had no other caller and is gone.
    pub async fn search_fused(
        &self,
        query: &str,
        limit: u32,
        depths: Depths,
        fusion: Fusion,
    ) -> Result<Vec<SearchHit>> {
        let lists = off_the_runtime(|| {
            // Embedded before the index is read, and the model lock released
            // before the read lock is taken: holding both is what would turn
            // one slow inference into a queue for every reader.
            let embedding = self.embedding()?.embed_query(query)?;
            let index = self.index();
            Ok::<_, pamin_index::IndexError>(vec![
                ChannelResults::new(
                    Channel::LexicalSegmented,
                    index.recall_segmented(query, depths.channel)?,
                ),
                ChannelResults::new(
                    Channel::LexicalNgram,
                    index.recall_ngram(query, depths.channel)?,
                ),
                ChannelResults::new(
                    Channel::Vector,
                    index.recall_vector(&embedding, depths.channel)?,
                ),
            ])
        })?;

        // Only the ledger knows what a topic stands for now, what it is worth,
        // and whether it still stands for anything -- so what the index
        // returned is resolved rather than trusted. A topic whose every state
        // has been soft deleted resolves to nothing and drops out here, before
        // fusion, so a deleted memory stops occupying a place in a channel's
        // candidate budget.
        // Interleaved by rank rather than concatenated, because this order is
        // what the graph channel seeds from and concatenating would offer it
        // one channel's whole list before another's first result. Fusion
        // cannot do the ordering: the graph is one of the lists it fuses.
        let candidates = best_first(&lists);
        let mut working = WorkingSet::default();
        working.add(
            repository::current_states_named(self.database.pool(), self.project, &candidates)
                .await?,
        );

        // The graph is the one channel the index cannot see, which is the
        // entire reason fusion happens here rather than inside the engine.
        let (graph_list, paths) = self
            .recall_graph(query, &candidates, &lists, &mut working, depths)
            .await?;

        // A path explains itself by the topics at both ends of its last edge,
        // and at two hops the near end is the topic in the middle -- which the
        // search need not have resolved: it can have been cut from the graph's
        // list, or stand for nothing now. Named here, and only when a path
        // needs it, so a one-hop walk still pays nothing.
        let unnamed: Vec<TopicId> = paths
            .values()
            .map(|reached| reached.via)
            .filter(|via| !working.names.contains_key(via))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        if !unnamed.is_empty() {
            working.name(
                repository::topics_by_id(self.database.pool(), self.project, &unnamed).await?,
            );
        }
        let mut lists = lists;
        lists.push(graph_list);
        let live = working;

        let mut fused = fusion.fuse(&lists);

        // A topic the index still knows about but the ledger no longer
        // resolves never reached the working set, so it is not ranked.
        fused.retain(|result| live.state(result.topic).is_some());

        for result in &mut fused {
            if let Some(reached) = paths.get(&result.topic) {
                let (asserted_from, asserted_to) = if reached.outbound {
                    (reached.via, reached.topic)
                } else {
                    (reached.topic, reached.via)
                };
                result.why.push(Why::Path {
                    from: live.topic_name(reached.origin),
                    via: live.topic_name(reached.via),
                    hops: reached.hops,
                    asserted_from: live.topic_name(asserted_from),
                    asserted_to: live.topic_name(asserted_to),
                    edge: reached.kind,
                    derivation: reached.derivation,
                });
            }
        }

        // No re-sort. `fuse` ordered these and nothing above changes a score --
        // the path entry is an explanation of a position, not a reason to move
        // one. Anything added here that does touch `score` has to sort again.
        Ok(fused
            .into_iter()
            .take(limit as usize)
            .map(|result| {
                let state = live.state(result.topic).expect("retained above");
                let seed = paths
                    .get(&result.topic)
                    .and_then(|reached| live.state(reached.origin))
                    .map(|origin| origin.content.clone());
                SearchHit {
                    topic: live.topic_name(result.topic),
                    state: state.clone(),
                    result,
                    seed,
                }
            })
            .collect())
    }

    /// Expands the graph around what the other channels found.
    ///
    /// Seeds come from the lexical and vector lists, so a seed handed back as a
    /// graph result would be one piece of evidence counted twice under two
    /// names — the exact double weighting that owning fusion is supposed to
    /// prevent. The expansion therefore returns only topics reached across at
    /// least one edge, and a seed appears among them only when something else
    /// in the graph reaches it.
    ///
    /// Returns the ranked list and the path evidence for each result, keyed by
    /// state, so the trace can say why the graph could see it.
    async fn recall_graph(
        &self,
        query: &str,
        ranked: &[TopicId],
        lists: &[ChannelResults],
        working: &mut WorkingSet,
        depths: Depths,
    ) -> Result<(ChannelResults, std::collections::HashMap<TopicId, Neighbor>)> {
        // Topics the query names directly. Without these, a question about a
        // topic whose own content happens not to match lexically never walks
        // out from it, and "what depends on X" cannot be answered by naming X.
        // Resolving query entities against known topics is the retrieval half
        // of entity linking; the write path does the other, and both ask the
        // same index the same way.
        let widest = self.widest_name().await?;
        let runs = off_the_runtime(|| runs_of_tokens(&self.segmenter.name_sequence(query), widest));
        let named = repository::topics_named_by(self.database.pool(), self.project, &runs).await?;
        let relevance = seed_relevance(&named, lists);

        // A named topic no channel returned is not in the working set, and the
        // filter below keeps only topics that are -- which is what it is for
        // with the ranked candidates, where it drops one that no longer stands
        // for anything, and exactly wrong for a name, whose point is to seed
        // the walk when the content did not match. So resolve those here, in
        // one lookup, and let the same filter judge them on the same terms.
        let unresolved: Vec<TopicId> = named
            .iter()
            .copied()
            .filter(|topic| working.state(*topic).is_none())
            .collect();
        if !unresolved.is_empty() {
            working.add(
                repository::current_states_named(self.database.pool(), self.project, &unresolved)
                    .await?,
            );
        }

        let seeds: Vec<TopicId> = {
            let mut seen = std::collections::HashSet::new();

            // Topics the query named come first, so a walk that has to give
            // something up gives up the weakest lexical and vector candidates
            // rather than the seed the caller asked about. The rest follow in
            // the order the channels ranked them, which is what makes "the
            // weakest" mean anything: taken from the working set instead, they
            // arrive in whatever order a hash map yields, and the bound below
            // keeps an arbitrary sixty-four rather than the best sixty-four.
            named
                .into_iter()
                .chain(ranked.iter().copied())
                .filter(|topic| working.state(*topic).is_some())
                .filter(|topic| seen.insert(*topic))
                .take(MAX_SEEDS)
                .collect()
        };

        let mut neighbors = graph::expand(
            self.database.pool(),
            self.project,
            &seeds,
            // Bounded by what this channel keeps, so a hub-shaped project
            // does not make the walk the whole cost of a search.
            &Expansion::to_depth(depths.graph).keeping(depths.channel as usize),
        )
        .await?;

        // Ordered by what each arrival is worth to *this query* before the cut,
        // not by the order the walk returned them in. The walk's order is
        // fewest hops, then most confident edge, then topic identifier -- and
        // with every derived edge at the same confidence and the walk stopping
        // at one hop, that is identifier order. Cutting it to the channel's
        // depth kept an arbitrary fifty and could drop every neighbour of the
        // seed that matched the query best.
        let strength = |neighbor: &Neighbor| {
            path_strength(neighbor) * relevance.get(&neighbor.origin).copied().unwrap_or(0.0)
        };
        neighbors.sort_by(|left, right| {
            strength(right)
                .total_cmp(&strength(left))
                .then_with(|| left.topic.0.cmp(&right.topic.0))
        });

        // Cut to the channel's depth before anything is resolved. Cutting after
        // meant every neighbour the walk found was looked up and given a path,
        // and the paths were not cut with the results -- so the list was bounded
        // and the work behind it was not.
        neighbors.truncate(depths.channel as usize);

        // Resolved here rather than at the end, because a topic that stands
        // for nothing is not a result and should not take a place in this
        // channel's budget. One lookup for all of them, through the pointer on
        // `topics`.
        let reached: Vec<TopicId> = neighbors.iter().map(|neighbor| neighbor.topic).collect();
        let states =
            repository::current_states_named(self.database.pool(), self.project, &reached).await?;
        let resolves: std::collections::HashSet<TopicId> =
            states.iter().map(|(_, state)| state.topic_id).collect();
        working.add(states);

        let mut candidates = Vec::new();
        let mut paths = std::collections::HashMap::new();
        for neighbor in neighbors {
            // A topic whose every state has been soft deleted resolves to
            // nothing and drops out here.
            if !resolves.contains(&neighbor.topic) {
                continue;
            }
            candidates.push(Scored::new(neighbor.topic, strength(&neighbor)));
            paths.insert(neighbor.topic, neighbor);
        }

        Ok((ChannelResults::new(Channel::Graph, candidates), paths))
    }

    /// Rebuilds the projection index from the authority store.
    ///
    /// Returns how many topics were indexed. The caller discards the index
    /// directory first, which is what makes this a genuine rebuild rather than
    /// an overwrite that could leave orphans behind.
    pub async fn reindex(&self) -> Result<Rebuilt> {
        // Before the states are read: the pointer decides which state a topic
        // resolves to, so a rebuild that trusted a stale one would index the
        // wrong content and look like it had worked.
        let repaired_pointers =
            repository::repair_current_state_pointers(self.database.pool(), self.project).await?;

        // Rebuilt with the projection because it is the same kind of thing: a
        // derived index of what the ledger already says, which the ledger can
        // restate at any time. It is also how a project that predates the
        // table gets one -- and every project does, because nothing else
        // backfills it and a topic missing from it is a topic no memory will
        // derive an edge to.
        let names = self.rebuild_name_index().await?;

        // Current states, one per topic: the projection holds one document per
        // topic, so every live state would write a topic's whole history onto
        // one key and leave whichever row the scan reached last.
        let states =
            repository::all_current_topic_states(self.database.pool(), self.project).await?;
        let passages = self.passages(&states).await?;

        // Held to the end of this function, so a reshape cannot start on the
        // index while it is being rebuilt.
        let (previous, _exclusive) = match self
            .rebuilding
            .lock()
            .expect("the set-aside index lock is poisoned")
            .take()
        {
            Some(Rebuilding {
                previous,
                _exclusive,
            }) => (previous, Some(_exclusive)),
            None => (None, None),
        };

        let reused = off_the_runtime(|| {
            fn pairs(batch: &[TopicState]) -> Vec<(TopicId, &str)> {
                batch
                    .iter()
                    .map(|state| (state.topic_id, state.content.as_str()))
                    .collect()
            }
            // Counted before any lock is taken, so a rebuild that can reuse
            // every vector never loads the model at all.
            let mut lent = 0;
            if let Some(previous) = &previous {
                for batch in states.chunks(REINDEX_BATCH) {
                    lent += previous.lends(&pairs(batch))?;
                }
            }

            // Both locks, in the order every other caller takes them, and held
            // for the whole rebuild. Taking the index first here would invert
            // the order against `search` and deadlock: a rebuild holding the
            // index and wanting the model, against a search holding the model
            // and wanting the index. Holding both throughout also matches what
            // a rebuild has always done -- it opens the collection for writing,
            // which excluded every reader in every other process already.
            let mut embedder = if lent < states.len() {
                Some(self.embedding()?)
            } else {
                None
            };
            let index = self.index();

            for (batch, batch_passages) in states
                .chunks(REINDEX_BATCH)
                .zip(passages.chunks(REINDEX_BATCH))
            {
                let mut vectors = match &previous {
                    Some(previous) => previous.vectors(&pairs(batch))?,
                    None => vec![None; batch.len()],
                };

                // One forward pass over what the old index could not supply,
                // rather than one per state. Measured on the smallest profile,
                // thirty-two texts together take 190 ms against 409 ms one at
                // a time -- the model is the same work either way, and what
                // the batch saves is everything around it.
                let missing: Vec<usize> = (0..batch.len())
                    .filter(|at| vectors[*at].is_none())
                    .collect();
                if !missing.is_empty() {
                    let texts: Vec<&str> = missing
                        .iter()
                        .map(|at| batch_passages[*at].as_str())
                        .collect();
                    let embedder = embedder
                        .as_mut()
                        .expect("the model is loaded whenever a vector is missing");
                    for (at, embedding) in missing.iter().zip(embedder.embed_passages(&texts)?) {
                        vectors[*at] = Some(embedding);
                    }
                }

                let documents: Vec<_> = batch
                    .iter()
                    .zip(&vectors)
                    .map(|(state, embedding)| {
                        (
                            state.topic_id,
                            state.content.as_str(),
                            embedding.as_deref().expect("every vector is filled above"),
                        )
                    })
                    .collect();

                index.upsert_batch(&documents)?;
            }
            index.flush()?;
            // A rebuild is the one point where building the vector graph is
            // clearly worth its cost: everything has just been written, and
            // without this the graph the index was configured for does not
            // exist and every vector query scans the buffer instead.
            index.optimize()?;
            if let Some(previous) = previous {
                previous.discard()?;
            }
            Ok::<_, pamin_index::IndexError>(lent)
        })?;

        Ok(Rebuilt {
            indexed: states.len(),
            reused,
            repaired_pointers,
            names,
        })
    }

    /// Restates every topic's name in the name index.
    ///
    /// Reads the topics rather than the table, so a name that is missing is
    /// added and one that is wrong is corrected. This is the one path that
    /// still walks every topic in a project, and it is the right one to: a
    /// rebuild is by definition proportional to what it rebuilds.
    async fn rebuild_name_index(&self) -> Result<usize> {
        let topics = repository::all_topics(self.database.pool(), self.project).await?;

        let keys: Vec<(TopicId, String, usize)> = off_the_runtime(|| {
            let segmenter = &self.segmenter;
            topics
                .iter()
                .map(|topic| {
                    let tokens = segmenter.name_sequence(&topic.name);
                    (topic.id, tokens.join(" "), tokens.len())
                })
                .collect()
        });

        // One statement rather than one per topic. A rebuild records every
        // topic in the project, so the row-at-a-time form was a round trip per
        // topic -- thirteen thousand of them on the corpora this project
        // measures, for a table with no more rows than that.
        repository::record_topic_names(self.database.pool(), self.project, &keys).await?;

        Ok(keys.len())
    }
}

/// This machine's name, for identifying who holds a cascade claim.
///
/// Best effort: the name only has to distinguish one worker from another, and
/// the process id already does that within a machine.
fn hostname() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| "local".to_string())
}

/// One memory, with the filter's verdict already reached.
///
/// Grouped rather than passed positionally: most of it is text and optional
/// text, and a caller swapping two of those would compile.
pub struct Write<'a> {
    pub topic: &'a str,
    pub content: &'a str,
    pub content_hash: &'a str,
    /// What the sensory filter decided, and why.
    pub verdict: FilterDecision,
    pub reason: &'a str,
    pub promoted: bool,
    /// Detected per span rather than per deployment: one workspace holds many
    /// languages, and this is what the note-language rule reads later.
    pub language: Option<&'a str>,
    pub language_confidence: Option<f32>,
    pub observed_at: OffsetDateTime,
    pub validity: Validity,
}

/// What a write left in the ledger.
pub struct Recorded {
    /// Always set: evidence is recorded whatever the filter decides.
    pub source_version: u32,
    /// Absent when the filter held the content in the evidence layer.
    pub state: Option<TopicState>,
}

/// What a rebuild did.
#[derive(Clone, Copy, Debug)]
pub struct Rebuilt {
    /// Topics written to the projection, which is one document each.
    pub indexed: usize,
    /// Of those, how many kept the vector the replaced index already held
    /// rather than being embedded again. See [`Previous`].
    pub reused: usize,
    /// Topics whose current-state pointer disagreed with the ledger.
    ///
    /// Expected to be zero: both writers move it under the topic's lock in the
    /// transaction that changed the ledger. Reported rather than swallowed
    /// because a number that is not zero is the only outward sign that some
    /// write path stopped maintaining it.
    pub repaired_pointers: u64,
    /// Topic names restated in the name index.
    ///
    /// The index that answers "which topics does this text name". A rebuild is
    /// the only thing that restates all of them, and for a project that
    /// predates the index it is the only thing that fills it at all.
    pub names: usize,
}

/// The states one search actually touched, and what the ledger says about them.
///
/// This used to be every live state in the project, loaded on every query into
/// four maps. That was workable while a workspace held thousands of states and
/// is the single thing that stopped being workable first: the cost of a search
/// was set by how much had ever been written rather than by how much the
/// channels returned.
///
/// It is filled in two steps because the search path finds its results in two
/// steps: the index and then the graph, each naming topics. Both go in here,
/// each state with its topic's name, which the lookup that resolves the state
/// returns beside it.
#[derive(Default)]
struct WorkingSet {
    /// What each topic stands for now. Only current states are ranked, so
    /// there is one per topic and no question of which.
    current: std::collections::HashMap<TopicId, TopicState>,
    /// Topic names, so a path can explain itself in the terms a caller uses.
    names: std::collections::HashMap<TopicId, String>,
}

impl WorkingSet {
    /// Records states and what the ledger calls their topics, which arrive
    /// together.
    fn add(&mut self, states: Vec<(String, TopicState)>) {
        for (name, state) in states {
            self.names.insert(state.topic_id, name);
            self.current.insert(state.topic_id, state);
        }
    }

    /// Records names for topics the search reached without resolving.
    fn name(&mut self, topics: Vec<(TopicId, String, Option<TopicStateId>)>) {
        for (topic, name, _) in topics {
            self.names.insert(topic, name);
        }
    }

    fn state(&self, topic: TopicId) -> Option<&TopicState> {
        self.current.get(&topic)
    }

    fn topic_name(&self, topic: TopicId) -> String {
        self.names
            .get(&topic)
            .cloned()
            .unwrap_or_else(|| topic.to_string())
    }
}

/// One search result: the topic's current state, its position, and why.
pub struct SearchHit {
    /// The topic, by the name a caller addresses it with.
    pub topic: String,
    /// What that topic stands for now. Search ranks topics and never their
    /// history; `pamin read --version-offset` is what reaches an earlier one.
    pub state: TopicState,
    pub result: FusedResult,
    /// For a hit the graph reached, the content of the memory the walk started
    /// from -- the sentence that makes this one relevant, which is why the
    /// reranker is shown it. Taken from every state the search resolved rather
    /// than from the results, so it does not depend on how many were asked for.
    pub seed: Option<String>,
}

/// The channels' candidates in one order, best first, without duplicates.
///
/// Round-robin by rank rather than one list after another: the lists are three
/// independent rankings of the same corpus and nothing has fused them yet, so
/// the only defensible reading of "best" across them is that each channel's
/// first pick outranks every channel's second. Concatenating would give one
/// channel's fiftieth candidate a better place than another channel's first.
fn best_first(lists: &[ChannelResults]) -> Vec<TopicId> {
    let deepest = lists
        .iter()
        .map(|list| list.candidates.len())
        .max()
        .unwrap_or(0);
    let mut seen = std::collections::HashSet::new();
    let mut ranked = Vec::new();

    for rank in 0..deepest {
        for list in lists {
            if let Some(candidate) = list.candidates.get(rank)
                && seen.insert(candidate.topic)
            {
                ranked.push(candidate.topic);
            }
        }
    }
    ranked
}

/// How relevant each seed is to the query, on `(0, 1]`.
///
/// **What a graph arrival is worth depends on where it was reached from, and
/// the channel's score used to ignore that.** A neighbour of the memory that
/// matched the query best and a neighbour of the sixty-fourth seed both scored
/// one derived mention's 0.5, so the channel could not tell an answer reached
/// from the right place from noise reached from a weak one -- and its weight
/// had to be cut for every query to protect the queries it was hurting. That
/// is the whole reason it ships at three tenths.
///
/// Personalised PageRank, as HippoRAG and its successors use it for graph
/// retrieval, starts the walk from a personalisation vector weighted by each
/// seed's relevance to the query, so what an expanded node scores is
/// proportional to how relevant its seed was. This is the one-step version of
/// that: an arrival's strength is its seed's relevance times the path's.
///
/// A seed the query names outright is fully relevant. Any other seed is worth
/// its best rank in any of the three index channels, discounted the way rank
/// fusion discounts it -- `(k + 1) / (k + rank)` with the same `k` fusion uses,
/// so this adds no constant of its own to tune.
fn seed_relevance(
    named: &[TopicId],
    lists: &[ChannelResults],
) -> std::collections::HashMap<TopicId, f32> {
    let mut relevance = std::collections::HashMap::new();
    for list in lists {
        for (index, candidate) in list.candidates.iter().enumerate() {
            let rank = index as f32 + 1.0;
            let worth = (pamin_core::DEFAULT_K + 1.0) / (pamin_core::DEFAULT_K + rank);
            relevance
                .entry(candidate.topic)
                .and_modify(|held: &mut f32| *held = held.max(worth))
                .or_insert(worth);
        }
    }
    for topic in named {
        relevance.insert(*topic, 1.0);
    }
    relevance
}

/// How strongly the graph vouches for one arrival: edge confidence, per hop.
///
/// **This is the one channel whose score is not its sort key, and the
/// disagreement is real rather than an oversight.** `graph::expand` orders its
/// neighbours lexicographically -- fewest hops first, then most confident, then
/// by identifier -- and no single number reproduces a lexicographic order:
/// a two-hop arrival at confidence 0.9 scores above a one-hop arrival at
/// confidence 0.3 here, while the walk ranks the one-hop first. Reordering the
/// channel by this number instead would be a ranking change nothing can measure
/// -- the graph channel contributes exactly 0.0000 to every group of all three
/// evaluation corpora -- so the order stays as the walk made it, and this
/// number answers the separate question fusion needs: how far this channel's
/// best arrival stands above its own field.
fn path_strength(neighbor: &Neighbor) -> f32 {
    neighbor.confidence * HOP_DECAY.powi(i32::from(neighbor.hops.saturating_sub(1)))
}

/// Every contiguous run of up to `widest` tokens, as the name index stores them.
///
/// The bound is what keeps this proportional to the text: without it the runs
/// are quadratic in the length of a memory, and a run longer than the longest
/// name in the project cannot be a name.
fn runs_of_tokens(tokens: &[String], widest: usize) -> Vec<String> {
    let mut runs = Vec::new();
    for width in 1..=widest.min(tokens.len()) {
        for window in tokens.windows(width) {
            runs.push(window.join(" "));
        }
    }
    runs.sort_unstable();
    runs.dedup();
    runs
}

/// What the reranker is shown for one candidate: its topic's name, its
/// content, and -- for a candidate the graph reached -- the memory the walk
/// started from ([`SearchHit::seed`]).
///
/// **Content alone is not enough to judge a memory by.** A memory's text
/// leaves implicit what its topic's name says -- `platform rota` is "it pages
/// ines on weekends" -- and a candidate the graph reached is relevant because
/// of an edge whose sentence is in *another* memory: "a sev one escalates to
/// the platform rota". Shown only its own text, a cross-encoder has nothing to
/// connect "who gets paged when a sev one escalates" to it and demotes it.
/// Titles before bodies is how the standard retrieval benchmarks present a
/// document to a reranker; the seed is the same argument for the one kind of
/// evidence this engine has that a benchmark does not.
///
/// Measured on the own corpus through the `CONTEXT` arm, against content
/// alone at the `accurate` tier: relational 0.6230 to 0.6739, 4 queries better
/// and none worse, and cross-lingual +0.0235, 16 better and 1 worse (family
/// p = 0.035). Chosen in every fold of a five-fold cross-validation, +0.0129
/// on the queries it did not choose on (p = 0.0005).
fn shown(topic: &str, content: &str, seed: Option<&str>) -> String {
    let mut text = format!("{topic}: {content}");
    if let Some(seed) = seed {
        text.push_str(". ");
        text.push_str(seed);
    }
    text
}

/// How many candidates only the graph found are shown to the reranker beside
/// the head.
///
/// The graph finds what the other channels cannot -- the memory a question
/// needs because another memory names it -- and fusion, weighing it at 0.30,
/// ranks those finds far below the head: on MuSiQue's 1,000 two-hop questions,
/// 153 supporting titles were found by the graph alone and not one reached the
/// reranker's twenty, at a median fused rank of 99. That rank was taken while
/// fusion also floored every candidate only the graph found; removing the
/// floor can only raise them, and it left all 1,000 questions' nDCG@10 where
/// it was. The reranker is the one stage that is shown the memory that
/// reached them, so it is the one that can judge them. Handing it the
/// strongest ten lifts nDCG@10 from 0.6573 to 0.6834 (108 questions
/// better, 47 worse, p = 0.0001) and recall@50 from 0.7940 to 0.8435; five
/// was worth +0.0234. Chosen by five-fold cross-validation, every fold picking
/// ten. The cost is ten more pairs a search, only where there are edges: a
/// project with none has no graph candidates and pays nothing.
const GRAPH_CANDIDATES: usize = 10;

/// The positions of a fused list the reranker is shown: the head's candidates
/// no lexical channel found, and the [`GRAPH_CANDIDATES`] strongest below the
/// head that only the graph found. Ascending.
///
/// One rule, public so the harnesses that replay a search from its trace use
/// this rather than a copy that could drift from it. `traces` is each fused
/// result's `why`, in fused order.
pub fn rerankable(traces: &[&[Why]], rerank: Rerank) -> Vec<usize> {
    let head = rerank.depth().min(traces.len());
    let lexical = |why: &[Why]| {
        why.iter().any(|entry| {
            matches!(
                entry,
                Why::Channel { channel, .. }
                    if *channel == Channel::LexicalSegmented || *channel == Channel::LexicalNgram
            )
        })
    };
    let graph_only = |why: &[Why]| -> Option<f32> {
        let mut graph = None;
        for entry in why {
            if let Why::Channel { channel, score, .. } = entry {
                if *channel != Channel::Graph {
                    return None;
                }
                graph = Some(score.unwrap_or(0.0));
            }
        }
        graph
    };

    let mut positions: Vec<usize> = (0..head).filter(|at| !lexical(traces[*at])).collect();
    let mut graph: Vec<(usize, f32)> = (head..traces.len())
        .filter_map(|at| graph_only(traces[at]).map(|score| (at, score)))
        .collect();
    graph.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    positions.extend(graph.into_iter().take(GRAPH_CANDIDATES).map(|(at, _)| at));
    positions.sort_unstable();
    positions
}

/// Puts the candidates the reranker scored back into the list, best first.
///
/// `shown` is what [`rerankable`] chose, ascending, and `best_first` indexes
/// into it in the model's order. A head candidate's slot is refilled in place,
/// so nothing else in the head moves. A graph candidate from below the head
/// has no slot there, so it is *inserted*: the list gains one slot at the end
/// of the head for each, the graph candidates leave their old positions, and
/// the whole shown set fills the head's slots and the new ones in the model's
/// order. So a graph find the model rates rises into the head, and a head
/// candidate it rates below one falls to just after the head -- not to rank
/// ninety-nine, where the find came from and where it would drop out of every
/// list a caller reads. With no graph candidates this is the in-place refill
/// it always was.
///
/// Generic because the harnesses replay a search by names through the same
/// function the engine places hits with.
pub fn place<T>(list: Vec<T>, shown: &[usize], head: usize, best_first: &[usize]) -> Vec<T> {
    let head = head.min(list.len());
    let mut slots: Vec<Option<T>> = list.into_iter().map(Some).collect();
    let mut taken: Vec<Option<T>> = shown.iter().map(|at| slots[*at].take()).collect();

    let below = shown.iter().filter(|at| **at >= head).count();
    let mut rest = slots.split_off(head);
    rest.retain(Option::is_some);
    slots.extend(std::iter::repeat_with(|| None).take(below));
    slots.extend(rest);

    let targets = shown
        .iter()
        .copied()
        .filter(|at| *at < head)
        .chain(head..head + below);
    for (slot, pick) in targets.zip(best_first) {
        slots[slot] = taken[*pick].take();
    }
    slots
        .into_iter()
        .map(|slot| slot.expect("every slot refilled"))
        .collect()
}

/// How deep to fuse when a reranker is going to reorder the head.
///
/// A cross-encoder can only reorder what it is shown, so the list it works on
/// has to be at least as long as the tier's depth even when the caller wants
/// five results. Cutting to the caller's limit first is what made the tuned
/// depth unreachable: `--limit` defaults to five, the tier's depth is twenty,
/// and the sweep that chose twenty was run over a list of fifty. What arrived
/// at the reranker was five candidates, of which the two it needs to find
/// unlexical are usually not among them -- so the shipped default reordered
/// nothing and returned the ranking a search with reranking off would have.
///
/// Deeper costs the fusion nothing extra: every channel already contributes
/// [`Depths::channel`] candidates and all of them are already resolved against
/// the ledger. What grows is the hydration, by the difference between the two
/// numbers.
///
/// And then the whole list rather than the head: the reranker is also shown the
/// strongest candidates only the graph found, wherever fusion put them (see
/// [`GRAPH_CANDIDATES`]), so a list cut at the head would have cut them off.
/// Every fused result is already resolved by then; what the depth costs is one
/// `SearchHit` a result, and the caller's limit is applied after the pass.
fn fused_for(limit: u32, rerank: Rerank) -> u32 {
    match rerank {
        Rerank::Off => limit,
        _ => u32::MAX,
    }
}

/// The first `limit` of a list that was fused deeper than the caller asked for.
/// Whether reranking these positions can change what the caller is given.
///
/// The pass reorders the candidates at `unlexical` *into the positions they
/// already hold* and the caller is then given the first `limit` of the list.
/// Two things follow, and one of them was being paid for.
///
/// Fewer than two positions cannot be reordered at all, which this always
/// checked.
///
/// And if every one of those positions sits at or past `limit`, the pass can
/// only permute candidates the caller never sees. That is not a heuristic
/// about when reranking is unlikely to help: the returned results are
/// identical either way, because the only thing the pass produces is a
/// permutation of those positions -- it attaches no score to a hit and writes
/// nothing into the trace. So the work is not merely unlikely to pay, it is
/// provably invisible.
///
/// This is worth nothing to the evaluation harnesses and something to every
/// user. The harnesses ask for 51 results so they can measure recall@50, and
/// the reranker's head is 20, so every position it touches is inside what they
/// read and this returns `true` on every query they run -- the accuracy
/// figures cannot move, by construction rather than by measurement. `pamin
/// search` defaults to five. On a query whose first five results all carry a
/// lexical hit, which is the ordinary shape of a same-language query, every
/// unlexical candidate is at rank five or beyond and the 226 ms was buying a
/// reordering of results nobody was going to be shown.
///
/// **What it is worth is 7%, and the first figure recorded for it was a
/// third.** Both arms below are `accuracy` / `fast` over XQuAD-R's 1,190
/// queries, taken in the same session on an idle machine, one thing apart:
///
/// | | reranking | candidates a query |
/// | --- | --- | --- |
/// | `--limit 51`, the harness | 252 ms | 15.3 |
/// | `--limit 5`, the default | 235 ms | 14.5 |
///
/// The third came from subtracting two arms taken in *different* runs, which
/// is the mistake this project has already withdrawn one figure for: the
/// module notes on `reranking` warn that this harness's latency column moves
/// by as much as a fifth between runs, and a fifth is three times the effect
/// being measured.
///
/// Seven per cent for nothing is still worth keeping, and the reason it is
/// small here is the reason it should be larger elsewhere: half of XQuAD-R's
/// queries are answered in another language, so an unlexical candidate is
/// almost always inside the first five and there is nothing to skip. A
/// workspace in one language is the case this gate is for, and that is
/// MIRACL's shape rather than this corpus's. Not measured there yet.
fn can_be_seen(unlexical: &[usize], limit: u32) -> bool {
    if unlexical.len() < 2 {
        return false;
    }
    // Ascending, because it was built by filtering a range. So the first entry
    // is the highest-ranked candidate the pass could move.
    unlexical
        .first()
        .is_some_and(|highest| *highest < limit as usize)
}

fn only(mut hits: Vec<SearchHit>, limit: u32) -> Vec<SearchHit> {
    hits.truncate(limit as usize);
    hits
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        GRAPH_CANDIDATES, MODEL_IDLE, Neighbor, best_first, can_be_seen, fused_for, is_idle,
        path_strength, place, rerankable, runs_of_tokens, seed_relevance, shown,
    };
    use pamin_core::{Channel, ChannelResults, TopicId};
    use pamin_index::Rerank;

    /// The reranker reads a candidate under its name, and a graph arrival with
    /// the memory it was reached from -- the sentence that makes it relevant.
    #[test]
    fn a_candidate_is_shown_with_its_name_and_the_memory_that_reached_it() {
        assert_eq!(
            shown(
                "platform rota",
                "it pages ines on weekends",
                Some("a sev one escalates to the platform rota")
            ),
            "platform rota: it pages ines on weekends. a sev one escalates to the platform rota"
        );
        // Not reached by the graph: the name and nothing else added.
        assert_eq!(
            shown("platform rota", "it pages ines on weekends", None),
            "platform rota: it pages ines on weekends"
        );
    }

    /// A seed is as relevant as its best rank anywhere, and a named seed is
    /// fully relevant.
    ///
    /// What makes a graph arrival from the top of the lexical list worth more
    /// than one from the bottom of it. The first place in any channel is
    /// `(k + 1) / (k + 1) = 1`, the eleventh is `11 / 21`, a topic ranked well
    /// by one channel and badly by another keeps the better, and a topic the
    /// query names outright is worth one whatever the channels thought.
    #[test]
    fn a_seed_is_as_relevant_as_its_best_rank() {
        let id = |byte: u8| TopicId::from(uuid::Uuid::from_bytes([byte; 16]));
        let list = |channel, topics: &[u8]| {
            ChannelResults::unscored(channel, topics.iter().map(|byte| id(*byte)).collect())
        };
        let lists = [
            list(
                Channel::LexicalSegmented,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            ),
            list(Channel::Vector, &[11, 1]),
        ];

        let relevance = seed_relevance(&[id(99)], &lists);
        assert_eq!(relevance[&id(1)], 1.0, "first in a channel");
        let eleventh = (pamin_core::DEFAULT_K + 1.0) / (pamin_core::DEFAULT_K + 11.0);
        assert!((relevance[&id(2)] - 11.0 / 12.0).abs() < 1e-6, "second");
        assert_eq!(
            relevance[&id(11)],
            1.0,
            "eleventh lexically but first by vector"
        );
        assert!(
            relevance[&id(10)] > eleventh,
            "tenth is worth more than eleventh would be"
        );
        assert_eq!(relevance[&id(99)], 1.0, "named by the query");
        assert!(!relevance.contains_key(&id(50)), "not a seed at all");
    }

    /// A model in use is not idle, however long ago it was handed out.
    ///
    /// The pair that matters: the window has to be reached, and reaching it is
    /// not enough on its own -- the caller checks that nothing holds the model
    /// as well, because dropping the registry's handle while a search holds
    /// its own frees nothing and makes the next search load a second copy.
    #[test]
    fn a_pass_over_candidates_below_the_limit_cannot_be_seen() {
        // Five results asked for, and the only candidates the pass may move
        // are at ranks 5 through 8. Whatever order it puts them in, the caller
        // is given ranks 0 through 4.
        assert!(!can_be_seen(&[5, 6, 7, 8], 5));
        // The same list when the caller asks for more of it.
        assert!(can_be_seen(&[5, 6, 7, 8], 6));
    }

    #[test]
    fn a_pass_reaching_the_limit_can_be_seen() {
        // One candidate inside what the caller reads is enough: reordering
        // can carry any of the others into that slot.
        assert!(can_be_seen(&[4, 11, 19], 5));
        assert!(can_be_seen(&[0, 1], 5));
    }

    #[test]
    fn one_candidate_is_not_a_reordering() {
        // Nothing to permute, wherever it sits.
        assert!(!can_be_seen(&[0], 5));
        assert!(!can_be_seen(&[], 5));
    }

    #[test]
    fn the_evaluation_harnesses_never_skip_the_pass() {
        // Both harnesses ask for `RECALL_AT + 1` results so they can measure
        // recall@50, and the reranker's head is 20. So every position it could
        // touch is inside what they read, and the published accuracy figures
        // cannot move because of this gate. Asserted rather than argued,
        // because the argument is the whole reason the gate is allowed to be
        // exact rather than swept.
        let whole_head: Vec<usize> = (0..20).collect();
        assert!(can_be_seen(&whole_head, 51));
        // Even the worst case for the harness -- only the last two positions
        // of the head are unlexical -- is still inside 51.
        assert!(can_be_seen(&[18, 19], 51));
    }

    #[test]
    fn a_model_is_idle_only_once_the_window_has_passed() {
        let window = Duration::from_secs(300);
        let now = Instant::now();
        let handed_out = now - Duration::from_secs(299);

        assert!(
            !is_idle(handed_out, now, window),
            "a reranker wanted a second ago was called idle"
        );
        assert!(
            is_idle(handed_out - Duration::from_secs(2), now, window),
            "a reranker nobody has wanted for five minutes was not called idle"
        );
    }

    /// The clock going backwards releases nothing rather than everything.
    ///
    /// `Instant` is monotonic, so this is not a real clock but a real
    /// arithmetic case: subtracting a later instant from an earlier one panics
    /// on a plain subtraction, and saturating it to zero is what keeps a
    /// reordering inside one tick from looking like five idle minutes.
    #[test]
    fn a_hand_out_in_the_future_is_not_idle() {
        let now = Instant::now();
        assert!(!is_idle(now + Duration::from_secs(600), now, MODEL_IDLE));
    }

    fn topic(byte: u8) -> TopicId {
        TopicId(uuid::Uuid::from_bytes([byte; 16]))
    }

    /// Every channel's first pick outranks every channel's second.
    ///
    /// The graph channel seeds from this order and keeps only the first
    /// sixty-four, so what the order means decides which topics get walked.
    /// Concatenating the lists would hand one channel's fiftieth candidate a
    /// better place than another channel's first.
    #[test]
    fn the_channels_merge_by_rank_and_not_by_channel() {
        let lists = vec![
            ChannelResults::unscored(
                Channel::LexicalSegmented,
                vec![topic(1), topic(2), topic(3)],
            ),
            ChannelResults::unscored(Channel::Vector, vec![topic(9), topic(8)]),
        ];

        assert_eq!(
            best_first(&lists),
            vec![topic(1), topic(9), topic(2), topic(8), topic(3)],
            "the lists were concatenated rather than interleaved by rank"
        );
    }

    /// A topic several channels agree on takes its best place, once.
    #[test]
    fn a_topic_two_channels_found_appears_at_its_best_rank() {
        let shared = topic(5);
        let lists = vec![
            ChannelResults::unscored(Channel::LexicalSegmented, vec![topic(1), shared]),
            ChannelResults::unscored(Channel::Vector, vec![shared, topic(2)]),
        ];

        assert_eq!(
            best_first(&lists),
            vec![topic(1), shared, topic(2)],
            "agreement should promote a candidate, not duplicate it"
        );
    }

    /// Channels of different depths do not lose their tail.
    #[test]
    fn a_deeper_channel_keeps_the_rest_of_its_list() {
        let lists = vec![
            ChannelResults::unscored(Channel::LexicalSegmented, vec![topic(1)]),
            ChannelResults::unscored(Channel::Vector, vec![topic(7), topic(8), topic(9)]),
        ];

        assert_eq!(
            best_first(&lists),
            vec![topic(1), topic(7), topic(8), topic(9)]
        );
    }

    use pamin_index::Segmenter;
    use pamin_index::segmentation::names;

    /// A reranker is shown at least as many candidates as it was tuned for.
    ///
    /// The constant saying how many a tier looks at is measured, and before
    /// this it was unreachable: the fused list was cut to the caller's limit
    /// first, so at the default `--limit 5` the tier saw five candidates rather
    /// than the twenty it then read, and the shipped default reordered nothing. This is the
    /// arithmetic that was wrong, on its own, because the alternative is a test
    /// that needs half a gigabyte of weights to observe a reordering that
    /// silently did not happen.
    #[test]
    fn a_reranker_is_fused_at_least_as_deep_as_it_reads() {
        for tier in [Rerank::Fast, Rerank::Accurate] {
            assert!(
                fused_for(5, tier) >= tier.depth() as u32,
                "{tier:?} reads {} candidates and was handed {}",
                tier.depth(),
                fused_for(5, tier)
            );
        }
    }

    /// The reranker is shown the head's unlexical candidates and the strongest
    /// graph-only ones below it -- not a lexical one, not a corroborated one
    /// from below the head, and no more graph ones than the cap.
    #[test]
    fn the_reranker_is_shown_the_unlexical_head_and_the_strongest_graph_finds() {
        use pamin_core::Why;

        let channel = |channel, score| Why::Channel {
            channel,
            rank: 1,
            score: Some(score),
            weight: 1.0,
            contribution: 0.0,
        };
        let head = Rerank::Accurate.depth();
        let mut traces: Vec<Vec<Why>> = Vec::new();
        for at in 0..head {
            // Every other head position has lexical evidence.
            traces.push(if at % 2 == 0 {
                vec![channel(Channel::LexicalSegmented, 1.0)]
            } else {
                vec![channel(Channel::Vector, 0.5)]
            });
        }
        // Below the head: a vector candidate, a corroborated graph one, and
        // more graph-only ones than the cap, with increasing strength.
        traces.push(vec![channel(Channel::Vector, 0.4)]);
        traces.push(vec![
            channel(Channel::Vector, 0.4),
            channel(Channel::Graph, 0.9),
        ]);
        let first_graph = traces.len();
        for strength in 0..GRAPH_CANDIDATES + 3 {
            traces.push(vec![channel(Channel::Graph, strength as f32 / 100.0)]);
        }
        let borrowed: Vec<&[Why]> = traces.iter().map(Vec::as_slice).collect();

        let shown = rerankable(&borrowed, Rerank::Accurate);
        let head_shown: Vec<usize> = shown.iter().copied().filter(|at| *at < head).collect();
        assert_eq!(
            head_shown,
            (0..head).filter(|at| at % 2 == 1).collect::<Vec<_>>()
        );
        let below: Vec<usize> = shown.iter().copied().filter(|at| *at >= head).collect();
        // The strongest GRAPH_CANDIDATES, which are the last ones pushed.
        let strongest = (first_graph + 3..first_graph + GRAPH_CANDIDATES + 3).collect::<Vec<_>>();
        assert_eq!(below, strongest);
    }

    /// A head candidate is refilled in place; a graph find from below the head
    /// is inserted at the end of it, and what it beats falls to just after the
    /// head rather than to where the find came from.
    #[test]
    fn a_graph_find_is_inserted_and_what_it_beats_falls_only_past_the_head() {
        // Head of three: a, b (shown), c (lexical, not shown). Below: d, e,
        // and the graph find g at the bottom.
        let list = vec!["a", "b", "c", "d", "e", "g"];
        let shown = [0, 1, 5];
        // The model: g best, then a, then b.
        let placed = place(list.clone(), &shown, 3, &[2, 0, 1]);
        assert_eq!(placed, vec!["g", "a", "c", "b", "d", "e"]);

        // Without a graph find it is the in-place refill it always was.
        assert_eq!(
            place(list, &[0, 1], 3, &[1, 0]),
            vec!["b", "a", "c", "d", "e", "g"]
        );
    }

    /// What this process remembers about the widest name only ever grows.
    ///
    /// The direction matters more than the caching does. Remembering a value
    /// that is too high costs a search some query windows that match nothing;
    /// too low means a whole class of name is never looked for. So a second
    /// writer reporting a narrower name must not be able to lower it.
    #[test]
    fn the_widest_name_is_raised_and_never_lowered() {
        let widest = std::sync::atomic::AtomicUsize::new(0);
        let raise = |tokens: usize| {
            widest.fetch_max(tokens, std::sync::atomic::Ordering::Relaxed);
            widest.load(std::sync::atomic::Ordering::Relaxed)
        };

        assert_eq!(raise(3), 3);
        assert_eq!(raise(5), 5, "a wider name did not raise it");
        assert_eq!(raise(2), 5, "a narrower name lowered it");
    }

    /// A caller wanting more than the reranker reads still gets what it asked.
    #[test]
    fn fusing_for_a_reranker_never_shortens_what_was_asked_for() {
        assert!(fused_for(500, Rerank::Fast) >= 500);
        // Nothing is going to reorder it, so nothing needs to be fused deep.
        assert_eq!(fused_for(5, Rerank::Off), 5);
    }

    /// Every case the segmenter's own naming tests pin, and one that is not a
    /// name in either scheme.
    const CASES: &[(&str, &str)] = &[
        ("the deployment pipeline runs on ci", "deployment_pipeline"),
        ("we moved off Argo CD last week", "argo_cd"),
        ("the technical debt is mounting", "db"),
        ("the db is mounting", "db"),
        ("the pipeline handles deployment", "deployment_pipeline"),
        ("部署流水线运行在持续集成上面", "流水线"),
        ("デプロイパイプラインは東京で動いています", "東京"),
        ("call deploy_service now", "deploy_service"),
        ("call the deploy service now", "deploy_service"),
        (
            "see crates/pamin-store/src/database.rs for it",
            "database.rs",
        ),
        ("any content at all", "   "),
        ("any content at all", "!!!"),
    ];

    /// The lookup finds a name exactly when comparing the sequences would.
    ///
    /// Deriving edges used to load every topic in a project and run the
    /// sequence comparison against each one. It now asks a table keyed by the
    /// name, which is only the same question if the runs offered to that table
    /// are exactly the sequences that would have matched. Nothing else checks
    /// that: a run scheme that missed a case would derive fewer edges, and
    /// fewer edges is not an error anything reports -- the graph channel would
    /// simply stop reaching things, on the queries nobody thought to try.
    #[test]
    fn a_run_lookup_finds_what_a_sequence_comparison_would() {
        let segmenter = Segmenter::new();

        for (text, name) in CASES {
            let content = segmenter.name_sequence(text);
            let needle = segmenter.name_sequence(name);

            let by_comparison = names(&content, &needle);
            let by_lookup = !needle.is_empty()
                && runs_of_tokens(&content, needle.len()).contains(&needle.join(" "));

            assert_eq!(
                by_comparison, by_lookup,
                "{text:?} naming {name:?}: comparison said {by_comparison}, lookup said {by_lookup}"
            );
        }
    }

    /// The bound is what keeps the runs proportional to the text.
    #[test]
    fn runs_stop_at_the_widest_name_there_is() {
        let tokens: Vec<String> = ["a", "b", "c", "d"].iter().map(|t| t.to_string()).collect();

        // Widths one and two only: 4 + 3 runs.
        assert_eq!(runs_of_tokens(&tokens, 2).len(), 7);
        // A project with no topics asks about nothing at all.
        assert!(runs_of_tokens(&tokens, 0).is_empty());
        // Asking for more width than there is text is not an error.
        assert_eq!(runs_of_tokens(&tokens, 99).len(), 4 + 3 + 2 + 1);
    }

    /// A hop costs half, and an uncertain edge costs whatever it is uncertain by.
    ///
    /// The point is that these are now different numbers at all. Before, every
    /// graph arrival reached fusion as a position in a list and nothing else,
    /// so a topic one asserted edge from the query's own answer and a topic two
    /// derived edges away were scored identically apart from which came first.
    #[test]
    fn the_graph_vouches_less_for_each_hop_it_took() {
        let reached = |hops: u8, confidence: f32| Neighbor {
            topic: TopicId(uuid::Uuid::from_bytes([1; 16])),
            origin: TopicId(uuid::Uuid::from_bytes([2; 16])),
            hops,
            via: TopicId(uuid::Uuid::from_bytes([3; 16])),
            kind: pamin_core::EdgeKind::RelatedTo,
            derivation: pamin_core::Derivation::Deterministic,
            confidence,
            outbound: true,
        };

        assert_eq!(path_strength(&reached(1, 1.0)), 1.0, "a certain first hop");
        assert_eq!(path_strength(&reached(2, 1.0)), 0.5, "one hop further");
        assert_eq!(path_strength(&reached(3, 1.0)), 0.25, "and one further");
        assert_eq!(
            path_strength(&reached(1, 0.5)),
            path_strength(&reached(2, 1.0)),
            "half the confidence at one hop is one certain hop further away"
        );
    }
}
