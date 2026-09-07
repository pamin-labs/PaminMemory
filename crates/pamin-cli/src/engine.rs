//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection are kept apart everywhere else in the
//! codebase; this is the one place that holds both, so it is also the only
//! place where the two can drift out of step.

use anyhow::Result;
use pamin_core::{
    Channel, ChannelResults, FusedResult, Fusion, Modifiers, ProjectId, TopicState, TopicStateId,
};
use pamin_index::{Embedder, Profile, ProjectionIndex};
use pamin_store::{Database, Workspace, repository};

/// How many candidates each channel contributes before fusion.
///
/// Deep enough for rank fusion to find agreement between channels, shallow
/// enough that reranking stays cheap. A default to be settled by measurement,
/// not by argument.
const CHANNEL_DEPTH: u32 = 50;

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
        let database = Database::open(workspace).await?;
        let project = repository::ensure_project(database.client(), project).await?;
        let index = ProjectionIndex::open(&workspace.index_dir(), profile)?;
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

    /// Recalls candidates from every channel and fuses them here.
    ///
    /// The retrieval engine can fuse its own two channels in one call, and that
    /// path is deliberately not taken: fusing there would produce a list that
    /// then had to be fused again with anything PostgreSQL contributes, and the
    /// per-channel ranks each result reports would already be lost.
    pub async fn search(&mut self, query: &str, limit: u32) -> Result<Vec<SearchHit>> {
        let embedding = self.embedder.embed_query(query)?;

        let lists = vec![
            ChannelResults::new(
                Channel::LexicalSegmented,
                self.index.recall_segmented(query, CHANNEL_DEPTH)?,
            ),
            ChannelResults::new(
                Channel::LexicalNgram,
                self.index.recall_ngram(query, CHANNEL_DEPTH)?,
            ),
            ChannelResults::new(
                Channel::Vector,
                self.index.recall_vector(&embedding, CHANNEL_DEPTH)?,
            ),
        ];

        let mut fused = Fusion::default().fuse(&lists);

        // The index knows ranks; only the ledger knows whether a state is
        // current and what it is worth, so modifiers are applied after loading.
        let states = self.load_states(&fused).await?;
        let modifiers = Modifiers::default();
        fused.retain(|result| states.contains_key(&result.topic_state));
        for result in &mut fused {
            let (state, is_current) = &states[&result.topic_state];
            modifiers.apply(result, &state.signals, *is_current);
        }
        pamin_core::sort_results(&mut fused);

        Ok(fused
            .into_iter()
            .take(limit as usize)
            .map(|result| {
                let (state, is_current) = states[&result.topic_state].clone();
                SearchHit {
                    state,
                    is_current,
                    result,
                }
            })
            .collect())
    }

    /// Loads the states behind a result set, dropping any the index still knows
    /// about but the ledger has since soft deleted.
    async fn load_states(
        &self,
        results: &[FusedResult],
    ) -> Result<std::collections::HashMap<TopicStateId, (TopicState, bool)>> {
        let live = repository::all_live_topic_states(self.database.client(), self.project).await?;

        let mut current: std::collections::HashMap<_, u32> = std::collections::HashMap::new();
        for state in &live {
            let entry = current.entry(state.topic_id).or_insert(state.version);
            *entry = (*entry).max(state.version);
        }

        let wanted: std::collections::HashSet<_> =
            results.iter().map(|result| result.topic_state).collect();

        Ok(live
            .into_iter()
            .filter(|state| wanted.contains(&state.id))
            .map(|state| {
                let is_current = current.get(&state.topic_id) == Some(&state.version);
                (state.id, (state, is_current))
            })
            .collect())
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

/// One search result: the state, its position, and why it is there.
pub struct SearchHit {
    pub state: TopicState,
    pub is_current: bool,
    pub result: FusedResult,
}
