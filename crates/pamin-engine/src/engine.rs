//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection are kept apart everywhere else in the
//! codebase; this is the one place that holds both, so it is also the only
//! place where the two can drift out of step.

use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::Result;
use pamin_core::{
    Channel, ChannelResults, EdgeKind, FilterDecision, FusedResult, Fusion, JobKind, Modifiers,
    ProjectId, SourceKind, Topic, TopicId, TopicState, TopicStateId, Validity, Why,
};
use pamin_index::{Access, Embedder, Profile, Projection, ProjectionIndex, Rerank, Reranker};
use pamin_store::graph::{EdgeClaim, Expansion, Neighbor};
use pamin_store::{Connections, Database, PgExecutor, Workspace, graph, jobs, repository};
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
/// Every seed is a separate expansion, and each one costs a neighbourhood that
/// grows with the depth. Without a bound the cost of the graph channel is set
/// by how many topics the other channels happened to surface, which is not a
/// quantity anything holds down.
const MAX_SEEDS: usize = 64;

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
    /// changing, reported upstream as alibaba/zvec#714 and still open. Readers
    /// share; a write excludes them.
    ///
    /// Not a precaution. Taking this lock out makes searches fail inside a
    /// minute under the concurrency `readers_and_writers_share_one_index_
    /// without_bringing_it_down` puts through it, and that is the mild form --
    /// upstream reports the same race faulting. On macOS it is latent, so it
    /// looks like a precaution there.
    index: Arc<Mutex<Box<dyn Projection + Send + Sync>>>,
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
    /// One model, and one caller into it at a time.
    ///
    /// Inference wants `&mut`, which is the only reason anything here ever
    /// needed `&mut self`. Putting it behind its own lock rather than the
    /// index's is what lets several searches read the index at once while one
    /// of them is embedding.
    ///
    /// Shared with every other engine on the same profile, which is why it
    /// arrives rather than being loaded here. Two projects are two indexes and
    /// one model.
    embedder: Arc<Mutex<Embedder>>,
    /// Where a reranker comes from, if a search asks for one.
    ///
    /// The registry rather than a loaded model: most searches do not rerank,
    /// most workspaces never will, and half a gigabyte should not be read off
    /// disk by opening a project.
    models: Models,
    pub project: ProjectId,
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
    loaded: Arc<Mutex<std::collections::HashMap<Profile, Arc<Mutex<Embedder>>>>>,
    /// The same arrangement for rerankers, keyed by tier for the same reason:
    /// the tier is what decides which weights these are.
    rerankers: Arc<Mutex<std::collections::HashMap<Rerank, Arc<Mutex<Reranker>>>>>,
}

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

        if let Some(embedder) = loaded.get(&profile) {
            return Ok(Arc::clone(embedder));
        }

        let embedder = Arc::new(Mutex::new(Embedder::load(profile, &self.dir)?));
        loaded.insert(profile, Arc::clone(&embedder));
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

        if let Some(reranker) = rerankers.get(&tier) {
            return Ok(Arc::clone(reranker));
        }

        let reranker = Arc::new(Mutex::new(Reranker::load(tier, &self.dir)?));
        rerankers.insert(tier, Arc::clone(&reranker));
        Ok(reranker)
    }
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

        let (index, embedder) = off_the_runtime(|| {
            if discard {
                ProjectionIndex::discard(&dir)?;
                // A rebuild is also the migration off the shared layout, which
                // is what the error about it tells the caller to run.
                ProjectionIndex::discard(&legacy)?;
            }

            let index = ProjectionIndex::open(&dir, &legacy, profile, access, documents)?;
            let embedder = models.get(profile)?;
            Ok::<_, pamin_index::IndexError>((
                Box::new(index) as Box<dyn Projection + Send + Sync>,
                embedder,
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
            index: Arc::new(Mutex::new(index)),
            embedder,
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
    pub(crate) fn index(&self) -> MutexGuard<'_, Box<dyn Projection + Send + Sync>> {
        self.index.lock().expect("the index lock is poisoned")
    }

    /// The model. One caller at a time, because inference wants `&mut`.
    pub(crate) fn embedding(&self) -> std::sync::MutexGuard<'_, Embedder> {
        self.embedder.lock().expect("the embedder lock is poisoned")
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
        off_the_runtime(|| {
            let embedding = self.embedding().embed_passage(&state.content)?;
            self.index()
                .upsert(state.topic_id, &state.content, &embedding)
        })?;
        Ok(())
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
        let evidence = repository::append_source_version(
            &mut transaction,
            self.project,
            source,
            request.content,
            request.content_hash,
            request.verdict,
            request.reason,
        )
        .await?;

        let span = repository::append_source_span(
            &mut *transaction,
            self.project,
            evidence.id,
            0,
            request.content.len() as u32,
            request.language,
            request.language_confidence,
        )
        .await?;

        let state = if request.promoted {
            let existed =
                repository::find_topic(&mut *transaction, self.project, request.topic).await?;
            let topic = match existed.clone() {
                Some(topic) => topic,
                None => {
                    let topic =
                        repository::ensure_topic(&mut transaction, self.project, request.topic)
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
                request.content,
                span.id,
                request.observed_at,
                request.validity,
            )
            .await?;

            // Committed with the state rather than after it. A crash between
            // the two would otherwise leave a memory the ledger knows about and
            // the projection never hears of -- which is the failure an outbox
            // exists to make impossible, and the one a `tokio::spawn` here
            // would leave wide open.
            jobs::enqueue(
                &mut *transaction,
                self.project,
                JobKind::SyncTopicIndex,
                Some(topic.id.0),
            )
            .await?;
            jobs::enqueue(
                &mut *transaction,
                self.project,
                JobKind::DeriveMentions,
                Some(topic.id.0),
            )
            .await?;

            // A topic that did not exist a moment ago may already be named by
            // memories written before it. Finding them is a scan, so it is
            // scheduled rather than paid for by whoever created the topic.
            if existed.is_none() {
                jobs::enqueue(
                    &mut *transaction,
                    self.project,
                    JobKind::BackfillMentions,
                    Some(topic.id.0),
                )
                .await?;
            }

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

    /// Returns the topic with this name, creating it if it does not exist.
    pub async fn ensure_topic(&self, name: &str) -> Result<Topic> {
        let mut connection = self.database.pool().acquire().await?;
        let topic = repository::ensure_topic(&mut connection, self.project, name).await?;
        self.record_name(&mut *connection, &topic).await?;
        Ok(topic)
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
        // Every run of tokens this memory contains that is short enough to be
        // somebody's name. A name matches only as a contiguous run, so this is
        // the complete set of things it could be naming -- and asking the index
        // for these is the same question the old loop asked of every topic in
        // the project one at a time, with the cost following the length of the
        // memory rather than the size of the project.
        let widest = repository::widest_topic_name(self.database.pool(), self.project).await?;
        let runs = off_the_runtime(|| {
            runs_of_tokens(&self.segmenter.name_sequence(&state.content), widest)
        });

        let mut named = repository::topics_named_by(self.database.pool(), self.project, &runs)
            .await?
            .into_iter()
            // A topic naming itself is not a relationship, and the schema
            // rejects the edge anyway.
            .filter(|topic| *topic != state.topic_id)
            .collect::<Vec<TopicId>>();
        named.sort_unstable();
        named.dedup();

        let edges: Vec<_> = named
            .iter()
            .map(|target| {
                (
                    state.topic_id,
                    *target,
                    EdgeClaim::derived(EdgeKind::Mentions, state.id, MENTION_CONFIDENCE),
                )
            })
            .collect();

        // One transaction: the edges a memory derives are one statement about
        // what it says, and asserting them separately both cost a commit each
        // and let a crash tell half of it.
        let asserted = graph::assert_edges(self.database.pool(), self.project, &edges).await?;

        graph::retract_derived(
            self.database.pool(),
            self.project,
            state.topic_id,
            EdgeKind::Mentions,
            &named,
        )
        .await?;

        Ok(asserted
            .iter()
            .filter(|assertion| assertion.is_new())
            .count())
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

    /// Recalls candidates from every channel and fuses them here.
    ///
    /// The retrieval engine can fuse its own two channels in one call, and that
    /// path is deliberately not taken: fusing there would produce a list that
    /// then had to be fused again with anything PostgreSQL contributes, and the
    /// per-channel ranks each result reports would already be lost.
    pub async fn search(&self, query: &str, limit: u32, depths: Depths) -> Result<Vec<SearchHit>> {
        self.search_fused(query, limit, depths, Fusion::default())
            .await
    }

    /// Search, then reorder the head of the result with a cross-encoder.
    ///
    /// Only the candidates no lexical channel found, and only into the
    /// positions those candidates already hold.
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
    pub async fn search_reranked(
        &self,
        query: &str,
        limit: u32,
        depths: Depths,
        rerank: Rerank,
    ) -> Result<Vec<SearchHit>> {
        let hits = self
            .search_fused(query, limit, depths, Fusion::default())
            .await?;
        if rerank == Rerank::Off || hits.is_empty() {
            return Ok(hits);
        }

        let head = rerank.depth().min(hits.len());
        let unlexical: Vec<usize> = (0..head)
            .filter(|position| {
                !hits[*position].result.why.iter().any(|why| {
                    matches!(
                        why,
                        Why::Channel { channel, .. }
                            if *channel == Channel::LexicalSegmented
                                || *channel == Channel::LexicalNgram
                    )
                })
            })
            .collect();
        if unlexical.len() < 2 {
            return Ok(hits);
        }

        let documents: Vec<&str> = unlexical
            .iter()
            .map(|position| hits[*position].state.content.as_str())
            .collect();
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

        // Back into the positions those candidates already held, so nothing
        // else in the list moves.
        let mut slots: Vec<Option<SearchHit>> = hits.into_iter().map(Some).collect();
        let mut taken: Vec<Option<SearchHit>> = unlexical
            .iter()
            .map(|position| slots[*position].take())
            .collect();
        for (slot, from) in unlexical.iter().zip(&ordered) {
            slots[*slot] = taken[*from].take();
        }

        Ok(slots
            .into_iter()
            .map(|hit| hit.expect("every position refilled"))
            .collect())
    }

    /// The same search, with the fusion settings supplied.
    ///
    /// Exists for the same reason [`Depths`] is a parameter: the constants
    /// fusion runs on were settled by measurement and are re-settled the same
    /// way, so the harness that measures them has to be able to vary them.
    /// Callers that are not measuring want [`search`](Self::search), which is
    /// this with what ships.
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
            let embedding = self.embedding().embed_query(query)?;
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
        let candidates: Vec<TopicId> = lists
            .iter()
            .flat_map(|list| list.candidates.iter().copied())
            .collect();
        let mut working = WorkingSet::default();
        working.add(
            repository::current_states_of(self.database.pool(), self.project, &candidates).await?,
        );

        // The graph is the one channel the index cannot see, which is the
        // entire reason fusion happens here rather than inside the engine.
        let (graph_list, paths) = self.recall_graph(query, &mut working, depths).await?;
        let mut lists = lists;
        lists.push(graph_list);

        // Names, and which state each topic stands for now. Asked once, for the
        // topics that actually produced a result, rather than for the project.
        working.describe(
            repository::topics_by_id(self.database.pool(), self.project, &working.topics()).await?,
        );
        let live = working;

        let mut fused = fusion.fuse(&lists);

        // A topic the index still knows about but the ledger no longer
        // resolves never reached the working set, so it is not ranked.
        fused.retain(|result| live.state(result.topic).is_some());

        let modifiers = Modifiers::default();
        for result in &mut fused {
            let state = live.state(result.topic).expect("retained above");
            if let Some(reached) = paths.get(&result.topic) {
                result.why.push(Why::Path {
                    from: live.topic_name(reached.origin),
                    via: live.topic_name(reached.via),
                    hops: reached.hops,
                    edge: reached.kind,
                    derivation: reached.derivation,
                });
            }
            modifiers.apply(result, &state.signals);
        }
        pamin_core::sort_results(&mut fused);

        Ok(fused
            .into_iter()
            .take(limit as usize)
            .map(|result| {
                let state = live.state(result.topic).expect("retained above");
                SearchHit {
                    topic: live.topic_name(result.topic),
                    state: state.clone(),
                    result,
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
        working: &mut WorkingSet,
        depths: Depths,
    ) -> Result<(ChannelResults, std::collections::HashMap<TopicId, Neighbor>)> {
        // Topics the query names directly. Without these, a question about a
        // topic whose own content happens not to match lexically never walks
        // out from it, and "what depends on X" cannot be answered by naming X.
        // Resolving query entities against known topics is the retrieval half
        // of entity linking; the write path does the other, and both ask the
        // same index the same way.
        let widest = repository::widest_topic_name(self.database.pool(), self.project).await?;
        let runs = off_the_runtime(|| runs_of_tokens(&self.segmenter.name_sequence(query), widest));
        let named = repository::topics_named_by(self.database.pool(), self.project, &runs).await?;

        let seeds: Vec<TopicId> = {
            let mut seen = std::collections::HashSet::new();

            // Topics the query named come first, so a walk that has to give
            // something up gives up the weakest lexical and vector candidates
            // rather than the seed the caller asked about.
            named
                .into_iter()
                .chain(working.topics())
                .filter(|topic| seen.insert(*topic))
                .take(MAX_SEEDS)
                .collect()
        };

        let mut neighbors = graph::expand(
            self.database.pool(),
            self.project,
            &seeds,
            &Expansion::to_depth(depths.graph),
        )
        .await?;

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
            repository::current_states_of(self.database.pool(), self.project, &reached).await?;
        let resolves: std::collections::HashSet<TopicId> =
            states.iter().map(|state| state.topic_id).collect();
        working.add(states);

        let mut candidates = Vec::new();
        let mut paths = std::collections::HashMap::new();
        for neighbor in neighbors {
            // A topic whose every state has been soft deleted resolves to
            // nothing and drops out here.
            if !resolves.contains(&neighbor.topic) {
                continue;
            }
            candidates.push(neighbor.topic);
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

        off_the_runtime(|| {
            // Both locks, in the order every other caller takes them, and held
            // for the whole rebuild. Taking the index first here would invert
            // the order against `search` and deadlock: a rebuild holding the
            // index and wanting the model, against a search holding the model
            // and wanting the index. Holding both throughout also matches what
            // a rebuild has always done -- it opens the collection for writing,
            // which excluded every reader in every other process already.
            let mut embedder = self.embedding();
            let index = self.index();

            for batch in states.chunks(REINDEX_BATCH) {
                // One forward pass over the batch rather than one per state.
                // Measured on the smallest profile, thirty-two texts together
                // take 190 ms against 409 ms one at a time -- the model is the
                // same work either way, and what the batch saves is everything
                // around it. A rebuild is the one path that always has a batch
                // in hand.
                let texts: Vec<&str> = batch.iter().map(|state| state.content.as_str()).collect();
                let embeddings = embedder.embed_passages(&texts)?;

                let documents: Vec<_> = batch
                    .iter()
                    .zip(&embeddings)
                    .map(|(state, embedding)| {
                        (state.topic_id, state.content.as_str(), embedding.as_slice())
                    })
                    .collect();

                index.upsert_batch(&documents)?;
            }
            index.flush()?;
            // A rebuild is the one point where building the vector graph is
            // clearly worth its cost: everything has just been written, and
            // without this the graph the index was configured for does not
            // exist and every vector query scans the buffer instead.
            index.optimize()
        })?;

        Ok(Rebuilt {
            indexed: states.len(),
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

        let mut connection = self.database.pool().acquire().await?;
        for (topic, key, tokens) in &keys {
            repository::record_topic_name(&mut *connection, self.project, *topic, key, *tokens)
                .await?;
        }

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
/// and the names are attached once at the end -- when the set of topics that
/// produced a result is finally known.
#[derive(Default)]
struct WorkingSet {
    /// What each topic stands for now. Only current states are ranked, so
    /// there is one per topic and no question of which.
    current: std::collections::HashMap<TopicId, TopicState>,
    /// Topic names, so a path can explain itself in the terms a caller uses.
    names: std::collections::HashMap<TopicId, String>,
}

impl WorkingSet {
    fn add(&mut self, states: Vec<TopicState>) {
        for state in states {
            self.current.insert(state.topic_id, state);
        }
    }

    /// Records what the ledger calls these topics.
    fn describe(&mut self, topics: Vec<(TopicId, String, Option<TopicStateId>)>) {
        for (topic, name, _) in topics {
            self.names.insert(topic, name);
        }
    }

    /// The topics found so far.
    fn topics(&self) -> Vec<TopicId> {
        let mut topics: Vec<TopicId> = self.current.keys().copied().collect();
        topics.sort_unstable_by_key(|topic| topic.0);
        topics
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

#[cfg(test)]
mod tests {
    use super::runs_of_tokens;
    use pamin_index::Segmenter;
    use pamin_index::segmentation::names;

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
}
