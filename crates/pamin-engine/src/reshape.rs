//! Reshaping a project's index while it is served.
//!
//! The index side -- the copy, the recording, the swap -- is
//! [`pamin_index::Reshape`]. What is here is what only this layer knows: which
//! topics the index should hold, which the ledger answers, and which other
//! operation must not run on the same directory at once.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use pamin_index::{Access, Reshape, Reshaped};
use pamin_store::repository;

use crate::engine::{Engine, off_the_runtime};

/// One exclusion per index directory, shared by every engine in the process.
///
/// A reshape and a rebuild both move an index's directory, and neither can
/// see the other through the engine: a rebuild opens a new engine over the
/// same directory while the old one -- the one a reshape is running on -- is
/// still alive. Two engines over one directory in one process is exactly that
/// case, so the exclusion is keyed by the directory and not held by an
/// engine. Another process cannot reach the directory while this one holds
/// the index open for writing, which a reshape requires.
static EXCLUSIVE: Mutex<BTreeMap<PathBuf, Arc<tokio::sync::Mutex<()>>>> =
    Mutex::new(BTreeMap::new());

/// The exclusion for the index at `dir`.
///
/// A reshape tries it and gives up if it is held; a rebuild waits for it.
pub(crate) fn exclusive(dir: &Path) -> Arc<tokio::sync::Mutex<()>> {
    Arc::clone(
        EXCLUSIVE
            .lock()
            .expect("the restructuring registry is poisoned")
            .entry(dir.to_path_buf())
            .or_default(),
    )
}

impl Engine {
    /// Copies this project's index into the shape the segment policy wants,
    /// and serves the copy, if the shape it is in is worth the work.
    ///
    /// `None` when there was nothing to do: the index is in shape, it was
    /// opened read-only, or a reshape or a rebuild of it is already running.
    ///
    /// Never embeds -- every document is copied, vector and all, from the
    /// index already there -- so it takes no model and never the model lock,
    /// and the lock order has nothing to say about it. The index lock is taken
    /// a batch at a time and for the swap; the graph over the copy, which is
    /// the slow part, is built with no lock held. For its length the process
    /// holds both indexes and the disk holds both directories.
    ///
    /// The topics to copy are the ledger's, listed after the reshape has
    /// started recording writes so that a topic created in between is either
    /// listed or recorded. A topic the ledger no longer has is therefore not
    /// copied, which is the same drift a rebuild removes.
    ///
    /// An index whose vectors a replaced model embedded is not copied but
    /// embedded again, the same way -- beside the served index, which answers
    /// without its vector channel until the copy replaces it. That does take
    /// the model, a text at a time, so a search waits behind one forward pass
    /// rather than behind the re-embedding; and it resumes what an attempt
    /// that did not finish left (see [`Reshape::reembed`]).
    pub async fn reshape(&self) -> Result<Option<Reshaped>> {
        if self.access != Access::ReadWrite {
            return Ok(None);
        }
        let Ok(_exclusive) = exclusive(&self.dir).try_lock_owned() else {
            return Ok(None);
        };
        if let Some(reembed) =
            off_the_runtime(|| Reshape::reembed(&self.index, &self.dir, self.profile))?
        {
            return self.reembed(reembed).await.map(Some);
        }
        let Some(mut reshape) =
            off_the_runtime(|| Reshape::begin(&self.index, &self.dir, self.profile))?
        else {
            return Ok(None);
        };

        let listed = repository::current_topic_ids(self.database.pool(), self.project).await;
        // Everything from here to the swap, including dropping the reshape if
        // a step fails -- which deletes the copy -- is blocking work.
        let reshaped = off_the_runtime(move || -> Result<Reshaped> {
            reshape.copy(&listed?)?;
            reshape.build()?;
            Ok(reshape.swap()?)
        })?;
        Ok(Some(reshaped))
    }

    /// Embeds every topic the ledger lists into a re-embedding's copy, and
    /// serves the copy.
    async fn reembed(&self, mut reembed: Reshape) -> Result<Reshaped> {
        let listed = repository::current_topic_ids(self.database.pool(), self.project).await?;
        let named = repository::topics_by_id(self.database.pool(), self.project, &listed).await?;
        off_the_runtime(move || -> Result<Reshaped> {
            let topics: Vec<(pamin_core::TopicId, &str)> = named
                .iter()
                .map(|(topic, name, _)| (*topic, name.as_str()))
                .collect();
            reembed.embed(&topics, |passage| self.embedding()?.embed_passage(passage))?;
            reembed.build()?;
            Ok(reembed.swap()?)
        })
    }
}
