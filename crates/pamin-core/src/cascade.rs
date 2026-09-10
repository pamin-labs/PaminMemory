//! The work a write leaves behind for the projection to catch up on.
//!
//! A write commits evidence, a span, a topic state, and the topic's current
//! pointer in one transaction, and that transaction touches nothing outside
//! PostgreSQL. Everything derived from it -- the embedding, the projection
//! entry, the edges the content implies -- happens afterwards, from a row in
//! the outbox committed alongside the rest.
//!
//! That is what keeps a write from depending on the index being reachable,
//! and what makes the derived work survive the process that scheduled it.

use serde::{Deserialize, Serialize};

/// A kind of deferred work.
///
/// A closed set rather than free strings, for the same reason [`EdgeKind`] is:
/// a typo would otherwise become a job type nothing claims and nothing runs,
/// and the row would sit in the outbox looking scheduled.
///
/// Every kind names a **topic** rather than a state, except the two that are
/// about the whole projection. "Bring this topic up to date" is idempotent by
/// construction and coalesces: fourteen edits to one topic leave one pending
/// job and thirteen embeddings never computed.
///
/// [`EdgeKind`]: crate::EdgeKind
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    /// Bring the projection's entry for this topic to its current state,
    /// which includes removing it when the topic stands for nothing.
    SyncTopicIndex,
    /// Recompute the edges this topic's current content implies.
    DeriveMentions,
    /// Link a newly created topic to memories that already named it.
    BackfillMentions,
    /// Build the vector index over everything written since the last build.
    OptimizeIndex,
}

impl JobKind {
    /// Parses a job kind from its wire name.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::SyncTopicIndex => "sync_topic_index",
            Self::DeriveMentions => "derive_mentions",
            Self::BackfillMentions => "backfill_mentions",
            Self::OptimizeIndex => "optimize_index",
        }
    }

    /// Every kind, so the schema and the CLI can enumerate them.
    pub const ALL: [Self; 4] = [
        Self::SyncTopicIndex,
        Self::DeriveMentions,
        Self::BackfillMentions,
        Self::OptimizeIndex,
    ];
}

impl std::fmt::Display for JobKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// How many times a job is retried before it is left alone.
///
/// Retrying for ever turns one poisoned job into a worker that never does
/// anything else. Eight attempts with the backoff below spans about eight
/// hours, which is long enough to outlast anything transient and short enough
/// that a genuine failure is still in front of whoever looks next.
pub const MAX_ATTEMPTS: i32 = 8;

/// How much owed work means the projection has fallen behind.
///
/// The queue is unbounded on purpose: a write must not fail because the index
/// is slow, and coalescing means a backlog of pending rows is much smaller
/// than the writes that produced it. But unbounded and unreported are
/// different things, and the second is what a bulk import that outran its
/// cascade looked like from outside -- searches quietly missing the newest
/// memories, with every write reporting success.
///
/// Ten thousand is far above what a normal session accumulates and far below
/// where the backlog is a problem, so crossing it says the writer is producing
/// faster than the cascade drains rather than that anything is wrong yet.
pub const LAGGING_AT: i64 = 10_000;

#[cfg(test)]
mod tests {
    use super::JobKind;

    /// Every kind round-trips through the name the database stores.
    ///
    /// The wire name is what the column holds and what its CHECK constraint
    /// lists, so a variant whose name does not parse back is a row that can be
    /// written and never claimed.
    #[test]
    fn every_kind_parses_back_from_its_name() {
        for kind in JobKind::ALL {
            assert_eq!(JobKind::parse(kind.as_str()), Some(kind));
        }

        assert_eq!(JobKind::parse("sync_topic"), None);
    }
}
