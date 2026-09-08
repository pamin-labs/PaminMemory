//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection are kept apart everywhere else in the
//! codebase; this is the one place that holds both, so it is also the only
//! place where the two can drift out of step.

use anyhow::Result;
use pamin_core::{
    Channel, ChannelResults, EdgeKind, FusedResult, Fusion, Modifiers, ProjectId, Topic, TopicId,
    TopicState, TopicStateId, Why,
};
use pamin_index::{Embedder, Profile, ProjectionIndex};
use pamin_store::graph::{EdgeClaim, Expansion};
use pamin_store::{Database, Workspace, graph, repository};

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

/// The store, the index, and the embedder, wired together.
pub struct Engine {
    pub database: Database,
    pub index: ProjectionIndex,
    pub embedder: Embedder,
    pub project: ProjectId,
}

impl Engine {
    /// Opens everything a search or a write needs.
    pub async fn open(workspace: &Workspace, project: &str, profile: Profile) -> Result<Self> {
        Self::open_index(workspace, project, profile, false).await
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
        Self::open_index(workspace, project, profile, true).await
    }

    async fn open_index(
        workspace: &Workspace,
        project: &str,
        profile: Profile,
        discard: bool,
    ) -> Result<Self> {
        let database = Database::open(workspace).await?;
        let project = repository::ensure_project(database.client(), project).await?;

        // The index is per project, so the identity has to be resolved before
        // the index can be located at all.
        let dir = workspace.index_dir(project.id);
        let legacy = workspace.legacy_index_dir();
        if discard {
            ProjectionIndex::discard(&dir)?;
            // A rebuild is also the migration off the shared layout, which is
            // what the error about it tells the caller to run.
            ProjectionIndex::discard(&legacy)?;
        }

        let index = ProjectionIndex::open(&dir, &legacy, profile)?;
        let embedder = Embedder::load(profile, &workspace.root().join("models"))?;

        Ok(Self {
            database,
            index,
            embedder,
            project: project.id,
        })
    }

    /// Adds one topic state to the projection index.
    pub fn index_state(&mut self, state: &TopicState) -> Result<()> {
        let embedding = self.embedder.embed_passage(&state.content)?;
        self.index.upsert(state.id, &state.content, &embedding)?;
        self.index.flush()?;
        Ok(())
    }

    /// Returns the topic with this name, creating it if it does not exist.
    ///
    /// A topic created here is also linked backwards: memories written before
    /// it existed may already name it, and without this pass an edge would
    /// appear only when one of those memories happened to be rewritten. The
    /// scan runs once in a topic's life, when it is first created.
    pub async fn ensure_topic(&mut self, name: &str) -> Result<Topic> {
        let existed = repository::find_topic(self.database.client(), self.project, name)
            .await?
            .is_some();
        let topic = repository::ensure_topic(self.database.client(), self.project, name).await?;

        if !existed {
            self.backfill_mentions(&topic).await?;
        }
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
        let topics = repository::all_topics(self.database.client(), self.project).await?;

        let named: Vec<TopicId> = {
            let segmenter = self.index.segmenter();
            topics
                .iter()
                // A topic naming itself is not a relationship, and the schema
                // rejects the edge anyway.
                .filter(|topic| topic.id != state.topic_id)
                .filter(|topic| segmenter.names(&state.content, &topic.name))
                .map(|topic| topic.id)
                .collect()
        };

        let mut added = 0;
        for target in named {
            let claim = EdgeClaim::derived(EdgeKind::Mentions, state.id, MENTION_CONFIDENCE);
            let assertion = graph::assert_edge(
                self.database.client_mut(),
                self.project,
                state.topic_id,
                target,
                &claim,
            )
            .await?;
            if assertion.is_new() {
                added += 1;
            }
        }
        Ok(added)
    }

    /// Links a newly created topic to memories that already named it.
    ///
    /// ponytail: segments every live state in the project. It runs once per
    /// topic ever created, which is rare enough to pay for; when the cascade
    /// worker exists this moves there and becomes a queued job.
    async fn backfill_mentions(&mut self, topic: &Topic) -> Result<usize> {
        let states =
            repository::all_live_topic_states(self.database.client(), self.project).await?;

        let naming: Vec<(TopicId, pamin_core::TopicStateId)> = {
            let segmenter = self.index.segmenter();
            states
                .iter()
                .filter(|state| state.topic_id != topic.id)
                .filter(|state| segmenter.names(&state.content, &topic.name))
                .map(|state| (state.topic_id, state.id))
                .collect()
        };

        let mut added = 0;
        for (from, caused_by) in naming {
            let claim = EdgeClaim::derived(EdgeKind::Mentions, caused_by, MENTION_CONFIDENCE);
            let assertion = graph::assert_edge(
                self.database.client_mut(),
                self.project,
                from,
                topic.id,
                &claim,
            )
            .await?;
            if assertion.is_new() {
                added += 1;
            }
        }
        Ok(added)
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
        let embedding = self.embedder.embed_query(query)?;

        let lists = vec![
            ChannelResults::new(
                Channel::LexicalSegmented,
                self.index.recall_segmented(query, depths.channel)?,
            ),
            ChannelResults::new(
                Channel::LexicalNgram,
                self.index.recall_ngram(query, depths.channel)?,
            ),
            ChannelResults::new(
                Channel::Vector,
                self.index.recall_vector(&embedding, depths.channel)?,
            ),
        ];

        // The ledger is read once and shared. The index knows ranks; only the
        // ledger knows whether a state is current, what it is worth, and which
        // states still exist at all.
        let live = LiveStates::load(&self.database, self.project).await?;

        // The graph is the one channel the index cannot see, which is the
        // entire reason fusion happens here rather than inside the engine.
        let (graph_list, paths) = self.recall_graph(query, &lists, &live, depths).await?;
        let mut lists = lists;
        lists.push(graph_list);

        let mut fused = Fusion::default().fuse(&lists);

        // A state the index still knows about but the ledger has soft deleted
        // is dropped rather than ranked.
        fused.retain(|result| live.contains(result.topic_state));

        let modifiers = Modifiers::default();
        for result in &mut fused {
            let state = live.state(result.topic_state).expect("retained above");
            if let Some(path) = paths.get(&result.topic_state) {
                result.why.push(path.clone());
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
        lists: &[ChannelResults],
        live: &LiveStates,
        depths: Depths,
    ) -> Result<(ChannelResults, std::collections::HashMap<TopicStateId, Why>)> {
        let seeds: Vec<TopicId> = {
            let segmenter = self.index.segmenter();
            let mut seen = std::collections::HashSet::new();

            // Topics the query names directly. Without these, a question about
            // a topic whose own content happens not to match lexically never
            // walks out from it, and "what depends on X" cannot be answered by
            // naming X. Resolving query entities against known topics is the
            // retrieval half of entity linking; the write path does the other.
            let named: Vec<TopicId> = live
                .topic_names()
                .filter(|(_, name)| segmenter.names(query, name))
                .map(|(topic, _)| topic)
                .filter(|topic| seen.insert(*topic))
                .collect();

            named
                .into_iter()
                .chain(
                    lists
                        .iter()
                        .flat_map(|list| list.candidates.iter())
                        .filter_map(|candidate| live.state(*candidate))
                        .map(|state| state.topic_id)
                        .filter(|topic| seen.insert(*topic)),
                )
                .collect()
        };

        let neighbors = graph::expand(
            self.database.client(),
            self.project,
            &seeds,
            &Expansion::to_depth(depths.graph),
        )
        .await?;

        let mut candidates = Vec::new();
        let mut paths = std::collections::HashMap::new();
        for neighbor in neighbors {
            // A topic identity is not a retrieval result; its current state is.
            // A topic whose every state has been soft deleted resolves to
            // nothing and drops out here.
            let Some(state) = live.current_state_of(neighbor.topic) else {
                continue;
            };
            candidates.push(state);
            paths.insert(
                state,
                Why::Path {
                    from: live.topic_name(neighbor.origin),
                    via: live.topic_name(neighbor.via),
                    hops: neighbor.hops,
                    edge: neighbor.kind,
                    derivation: neighbor.derivation,
                },
            );
        }

        candidates.truncate(depths.channel as usize);
        Ok((ChannelResults::new(Channel::Graph, candidates), paths))
    }

    /// Rebuilds the projection index from the authority store.
    ///
    /// Returns how many states were indexed. The caller discards the index
    /// directory first, which is what makes this a genuine rebuild rather than
    /// an overwrite that could leave orphans behind.
    pub async fn reindex(&mut self) -> Result<usize> {
        let states =
            repository::all_live_topic_states(self.database.client(), self.project).await?;

        for state in &states {
            let embedding = self.embedder.embed_passage(&state.content)?;
            self.index.upsert(state.id, &state.content, &embedding)?;
        }
        self.index.flush()?;

        Ok(states.len())
    }
}

/// Every live topic state in the project, indexed the ways a search needs it.
///
/// Loaded once per search and shared. Two parts of the search path need the
/// same rows for different reasons — the modifier pass needs each result's
/// signals and whether it is current, and the graph channel needs to resolve a
/// topic to its current state — and loading them twice would mean scanning the
/// project twice for one query.
///
/// ponytail: whole-project scan per query. Fine while a workspace holds
/// thousands of states; when it does not, this becomes a lookup restricted to
/// the candidate set plus a per-topic current-state index.
struct LiveStates {
    by_id: std::collections::HashMap<TopicStateId, TopicState>,
    /// The newest surviving version of each topic. Current is computed rather
    /// than stored, so it is derived here rather than read from a column.
    current: std::collections::HashMap<TopicId, u32>,
    /// The state each topic currently resolves to, which is what the graph
    /// channel needs: it walks topic identities and has to return states.
    current_state: std::collections::HashMap<TopicId, TopicStateId>,
    /// Topic names, so a path can explain itself in the terms a caller uses.
    names: std::collections::HashMap<TopicId, String>,
}

impl LiveStates {
    async fn load(database: &Database, project: ProjectId) -> Result<Self> {
        let states = repository::all_live_topic_states(database.client(), project).await?;

        let mut current: std::collections::HashMap<TopicId, u32> = std::collections::HashMap::new();
        for state in &states {
            let entry = current.entry(state.topic_id).or_insert(state.version);
            *entry = (*entry).max(state.version);
        }

        let current_state = states
            .iter()
            .filter(|state| current.get(&state.topic_id) == Some(&state.version))
            .map(|state| (state.topic_id, state.id))
            .collect();

        let names = repository::all_topics(database.client(), project)
            .await?
            .into_iter()
            .map(|topic| (topic.id, topic.name))
            .collect();

        Ok(Self {
            by_id: states.into_iter().map(|state| (state.id, state)).collect(),
            current,
            current_state,
            names,
        })
    }

    fn contains(&self, id: TopicStateId) -> bool {
        self.by_id.contains_key(&id)
    }

    fn state(&self, id: TopicStateId) -> Option<&TopicState> {
        self.by_id.get(&id)
    }

    fn is_current(&self, state: &TopicState) -> bool {
        self.current.get(&state.topic_id) == Some(&state.version)
    }

    /// The state that represents a topic right now.
    ///
    /// The graph connects topic identities, but only a state can be returned
    /// from a search, so every neighbour resolves through here. A topic whose
    /// versions have all been soft deleted resolves to nothing.
    fn current_state_of(&self, topic: TopicId) -> Option<TopicStateId> {
        self.current_state.get(&topic).copied()
    }

    /// Every topic and its name, for matching query text against them.
    fn topic_names(&self) -> impl Iterator<Item = (TopicId, &str)> {
        self.names
            .iter()
            .map(|(topic, name)| (*topic, name.as_str()))
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
