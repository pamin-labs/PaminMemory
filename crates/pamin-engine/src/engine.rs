//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection are kept apart everywhere else in the
//! codebase; this is the one place that holds both, so it is also the only
//! place where the two can drift out of step.

use anyhow::Result;
use pamin_core::{
    Channel, ChannelResults, EdgeKind, FilterDecision, FusedResult, Fusion, JobKind, Modifiers,
    ProjectId, SourceKind, Topic, TopicId, TopicState, TopicStateId, Validity, Why,
};
use pamin_index::{Access, Embedder, Profile, Projection, ProjectionIndex};
use pamin_store::graph::{EdgeClaim, Expansion, Neighbor};
use pamin_store::{Database, Workspace, graph, jobs, repository};
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
pub struct Engine {
    pub database: Database,
    /// Who this process is when it claims cascade work. Distinct per process so
    /// a claim identifies its holder, which is what completion checks against.
    pub(crate) worker: String,
    /// Behind the trait rather than the concrete type, so the composition layer
    /// names what it needs from a projection and not which engine provides it.
    pub index: Box<dyn Projection>,
    pub embedder: Embedder,
    pub project: ProjectId,
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
        Self::open_index(workspace, project, profile, access, false).await
    }

    /// Opens with the projection discarded first, for a rebuild.
    ///
    /// Discarding before opening rather than overwriting in place is what
    /// makes a rebuild a rebuild: an overwrite leaves behind anything the
    /// ledger no longer has, which is the drift the rebuild exists to remove.
    pub async fn rebuilding(
        workspace: &Workspace,
        project: &str,
        profile: Profile,
    ) -> Result<Self> {
        Self::open_index(workspace, project, profile, Access::ReadWrite, true).await
    }

    async fn open_index(
        workspace: &Workspace,
        project: &str,
        profile: Profile,
        access: Access,
        discard: bool,
    ) -> Result<Self> {
        let database = Database::open(workspace).await?;
        let project = repository::ensure_project(database.pool(), project).await?;

        // The index is per project, so the identity has to be resolved before
        // the index can be located at all.
        let dir = workspace.index_dir(project.id);
        let legacy = workspace.legacy_index_dir();
        let models = workspace.root().join("models");

        let (index, embedder) = off_the_runtime(|| {
            if discard {
                ProjectionIndex::discard(&dir)?;
                // A rebuild is also the migration off the shared layout, which
                // is what the error about it tells the caller to run.
                ProjectionIndex::discard(&legacy)?;
            }

            let index = ProjectionIndex::open(&dir, &legacy, profile, access)?;
            let embedder = Embedder::load(profile, &models)?;
            Ok::<_, pamin_index::IndexError>((Box::new(index) as Box<dyn Projection>, embedder))
        })?;

        Ok(Self {
            database,
            worker: format!("{}:{}", hostname(), std::process::id()),
            index,
            embedder,
            project: project.id,
        })
    }

    /// Adds one topic state to the projection index.
    pub async fn index_state(&mut self, state: &TopicState) -> Result<()> {
        let (index, embedder) = (&self.index, &mut self.embedder);
        off_the_runtime(|| {
            let embedding = embedder.embed_passage(&state.content)?;
            index.upsert(state.id, &state.content, &embedding)?;
            index.flush()
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
    pub async fn write(&mut self, request: &Write<'_>) -> Result<Recorded> {
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
                    repository::ensure_topic(&mut transaction, self.project, request.topic).await?
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
    pub async fn ensure_topic(&mut self, name: &str) -> Result<Topic> {
        let mut connection = self.database.pool().acquire().await?;
        let topic = repository::ensure_topic(&mut connection, self.project, name).await?;
        Ok(topic)
    }

    /// Derives edges from the topics this state's content names.
    ///
    /// In a topic-centred graph the topics are the entities, so a memory naming
    /// another topic is entity linking with no model in the path. Returns how
    /// many edges this call actually added, which is zero when the content is
    /// unchanged: asserting an edge is idempotent, so rewriting a memory does
    /// not grow the ledger.
    pub async fn derive_mentions(&mut self, state: &TopicState) -> Result<usize> {
        let topics = repository::all_topics(self.database.pool(), self.project).await?;

        let named: Vec<TopicId> = {
            let segmenter = self.index.segmenter();
            // Segmented once rather than once per topic: this is the same
            // question asked of every topic in the project, and only the name
            // changes between askings.
            let content = segmenter.name_sequence(&state.content);
            topics
                .iter()
                // A topic naming itself is not a relationship, and the schema
                // rejects the edge anyway.
                .filter(|topic| topic.id != state.topic_id)
                .filter(|topic| {
                    pamin_index::segmentation::names(
                        &content,
                        &segmenter.name_sequence(&topic.name),
                    )
                })
                .map(|topic| topic.id)
                .collect()
        };

        let edges: Vec<_> = named
            .into_iter()
            .map(|target| {
                (
                    state.topic_id,
                    target,
                    EdgeClaim::derived(EdgeKind::Mentions, state.id, MENTION_CONFIDENCE),
                )
            })
            .collect();

        // One transaction: the edges a memory derives are one statement about
        // what it says, and asserting them separately both cost a commit each
        // and let a crash tell half of it.
        let asserted = graph::assert_edges(self.database.pool(), self.project, &edges).await?;

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
    /// Segments every live state in the project, which is why it is a job
    /// rather than something the write that created the topic pays for. It
    /// still runs once per topic ever created; 6.2 replaces the scan with a
    /// lexical probe.
    pub(crate) async fn backfill_mentions(&mut self, topic: TopicId, name: &str) -> Result<usize> {
        let states = repository::all_live_topic_states(self.database.pool(), self.project).await?;

        let naming: Vec<(TopicId, pamin_core::TopicStateId)> = {
            let segmenter = self.index.segmenter();
            // The fixed side here is the name, so that is the side prepared.
            let name = segmenter.name_sequence(name);
            states
                .iter()
                .filter(|state| state.topic_id != topic)
                .filter(|state| {
                    pamin_index::segmentation::names(
                        &segmenter.name_sequence(&state.content),
                        &name,
                    )
                })
                .map(|state| (state.topic_id, state.id))
                .collect()
        };

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
    pub async fn search(
        &mut self,
        query: &str,
        limit: u32,
        depths: Depths,
    ) -> Result<Vec<SearchHit>> {
        let (index, embedder) = (&self.index, &mut self.embedder);
        let lists = off_the_runtime(|| {
            let embedding = embedder.embed_query(query)?;
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

        // Only the ledger knows whether a state is current, what it is worth,
        // and which states still exist at all -- so what the index returned is
        // looked up rather than trusted. Soft-deleted states drop out here,
        // before fusion, so a deleted memory stops occupying a place in a
        // channel's candidate budget.
        let candidates: Vec<TopicStateId> = lists
            .iter()
            .flat_map(|list| list.candidates.iter().copied())
            .collect();
        let mut working = WorkingSet::default();
        working.add(
            repository::topic_states_by_id(self.database.pool(), self.project, &candidates).await?,
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

        let mut fused = Fusion::default().fuse(&lists);

        // A state the index still knows about but the ledger has soft deleted
        // never reached the working set, so it is not ranked.
        fused.retain(|result| live.state(result.topic_state).is_some());

        let modifiers = Modifiers::default();
        for result in &mut fused {
            let state = live.state(result.topic_state).expect("retained above");
            if let Some(reached) = paths.get(&result.topic_state) {
                result.why.push(Why::Path {
                    from: live.topic_name(reached.origin),
                    via: live.topic_name(reached.via),
                    hops: reached.hops,
                    edge: reached.kind,
                    derivation: reached.derivation,
                });
            }
            modifiers.apply(result, &state.signals, live.is_current(state));
        }
        pamin_core::sort_results(&mut fused);

        Ok(fused
            .into_iter()
            .take(limit as usize)
            .map(|result| {
                let state = live.state(result.topic_state).expect("retained above");
                SearchHit {
                    topic: live.topic_name(state.topic_id),
                    is_current: live.is_current(state),
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
    ) -> Result<(
        ChannelResults,
        std::collections::HashMap<TopicStateId, Neighbor>,
    )> {
        // ponytail: reads every topic in the project to match the query against
        // their names. The rest of this path no longer scans, and this is what
        // is left; the inverted table of name tokens replaces it, and until
        // then a project's topic count still sets the cost of a search.
        let topics = repository::all_topics(self.database.pool(), self.project).await?;

        let seeds: Vec<TopicId> = {
            let segmenter = self.index.segmenter();
            let mut seen = std::collections::HashSet::new();

            // Topics the query names directly. Without these, a question about
            // a topic whose own content happens not to match lexically never
            // walks out from it, and "what depends on X" cannot be answered by
            // naming X. Resolving query entities against known topics is the
            // retrieval half of entity linking; the write path does the other.
            let prepared = segmenter.name_sequence(query);
            let named: Vec<TopicId> = topics
                .iter()
                .filter(|topic| {
                    pamin_index::segmentation::names(
                        &prepared,
                        &segmenter.name_sequence(&topic.name),
                    )
                })
                .map(|topic| topic.id)
                .filter(|topic| seen.insert(*topic))
                .collect();

            named
                .into_iter()
                .chain(
                    working
                        .topics()
                        .into_iter()
                        .filter(|topic| seen.insert(*topic)),
                )
                // Topics the query named come first, so a walk that has to
                // give something up gives up the weakest lexical and vector
                // candidates rather than the seed the caller asked about.
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

        // A topic identity is not a retrieval result; its current state is. One
        // lookup for all of them, through the pointer on `topics`.
        let reached: Vec<TopicId> = neighbors.iter().map(|neighbor| neighbor.topic).collect();
        let states =
            repository::current_states_of(self.database.pool(), self.project, &reached).await?;
        let resolves_to: std::collections::HashMap<TopicId, TopicStateId> = states
            .iter()
            .map(|state| (state.topic_id, state.id))
            .collect();
        working.add(states);

        let mut candidates = Vec::new();
        let mut paths = std::collections::HashMap::new();
        for neighbor in neighbors {
            // A topic whose every state has been soft deleted resolves to
            // nothing and drops out here.
            let Some(state) = resolves_to.get(&neighbor.topic).copied() else {
                continue;
            };
            candidates.push(state);
            paths.insert(state, neighbor);
        }

        Ok((ChannelResults::new(Channel::Graph, candidates), paths))
    }

    /// Rebuilds the projection index from the authority store.
    ///
    /// Returns how many states were indexed. The caller discards the index
    /// directory first, which is what makes this a genuine rebuild rather than
    /// an overwrite that could leave orphans behind.
    pub async fn reindex(&mut self) -> Result<Rebuilt> {
        // Before the states are read: the pointer decides which state a topic
        // resolves to, so a rebuild that trusted a stale one would index the
        // wrong content and look like it had worked.
        let repaired_pointers =
            repository::repair_current_state_pointers(self.database.pool(), self.project).await?;

        let states = repository::all_live_topic_states(self.database.pool(), self.project).await?;

        let (index, embedder) = (&self.index, &mut self.embedder);
        off_the_runtime(|| {
            for batch in states.chunks(REINDEX_BATCH) {
                let embeddings = batch
                    .iter()
                    .map(|state| embedder.embed_passage(&state.content))
                    .collect::<pamin_index::Result<Vec<_>>>()?;

                let documents: Vec<_> = batch
                    .iter()
                    .zip(&embeddings)
                    .map(|(state, embedding)| {
                        (state.id, state.content.as_str(), embedding.as_slice())
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
        })
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
    /// States written to the projection.
    pub indexed: usize,
    /// Topics whose current-state pointer disagreed with the ledger.
    ///
    /// Expected to be zero: both writers move it under the topic's lock in the
    /// transaction that changed the ledger. Reported rather than swallowed
    /// because a number that is not zero is the only outward sign that some
    /// write path stopped maintaining it.
    pub repaired_pointers: u64,
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
/// steps: the index names states, and the graph names topics that then resolve
/// to states. Both go in here, and the topics behind them are described once at
/// the end -- when the set of topics that produced a result is finally known.
#[derive(Default)]
struct WorkingSet {
    by_id: std::collections::HashMap<TopicStateId, TopicState>,
    /// The state each topic stands for now, from the pointer on `topics`.
    current_state: std::collections::HashMap<TopicId, TopicStateId>,
    /// Topic names, so a path can explain itself in the terms a caller uses.
    names: std::collections::HashMap<TopicId, String>,
}

impl WorkingSet {
    fn add(&mut self, states: Vec<TopicState>) {
        for state in states {
            self.by_id.insert(state.id, state);
        }
    }

    /// Records what the ledger says about the topics behind these states.
    fn describe(&mut self, topics: Vec<(TopicId, String, Option<TopicStateId>)>) {
        for (topic, name, current) in topics {
            self.names.insert(topic, name);
            if let Some(current) = current {
                self.current_state.insert(topic, current);
            }
        }
    }

    /// The topics these states belong to.
    fn topics(&self) -> Vec<TopicId> {
        let mut topics: Vec<TopicId> = self
            .by_id
            .values()
            .map(|state| state.topic_id)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        topics.sort_unstable_by_key(|topic| topic.0);
        topics
    }

    fn state(&self, id: TopicStateId) -> Option<&TopicState> {
        self.by_id.get(&id)
    }

    fn is_current(&self, state: &TopicState) -> bool {
        self.current_state.get(&state.topic_id) == Some(&state.id)
    }

    fn topic_name(&self, topic: TopicId) -> String {
        self.names
            .get(&topic)
            .cloned()
            .unwrap_or_else(|| topic.to_string())
    }
}

/// One search result: the state, its position, and why it is there.
pub struct SearchHit {
    /// The topic this state belongs to, by the name a caller addresses it with.
    pub topic: String,
    pub state: TopicState,
    pub is_current: bool,
    pub result: FusedResult,
}
