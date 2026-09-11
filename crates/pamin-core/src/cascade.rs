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

/// Whether a writer that asked to defer its index work may still have it.
///
/// The second and last thing that acts on a backlog, the first being the signal
/// above. Deferring is what makes a bulk import cheap -- write everything, pay
/// the index once at the end, rather than an embedding and a flush per memory
/// -- and it is the only way the queue grows without bound, because every other
/// write drains before it returns. So the bound goes here: a deferred write is
/// deferred while the queue is within the depth that reports it behind, and
/// pays it down once it is not. The writer that outran the cascade becomes the
/// cascade.
///
/// **It is deliberately not a pause.** The plan for this layer was that a
/// writer past the ceiling should sleep, which is what backpressure usually
/// means and what it cannot mean here: the writer is ordinarily the only thing
/// that drains, so a writer that sleeps slows the import and leaves the backlog
/// exactly where it was. Paying it down costs the same writer the same time and
/// bounds the queue, which sleeping does not do at all.
///
/// **One number, not two.** The depth that means "behind" and the depth at
/// which something is done about it are the same on purpose. A ceiling ten
/// times the signal would mean watching a backlog run away for an order of
/// magnitude after already knowing it was running away.
pub fn may_defer(pending: i64) -> bool {
    pending < LAGGING_AT
}

#[cfg(test)]
mod tests {
    use super::{JobKind, LAGGING_AT, may_defer};

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

    /// A queue nobody is behind on lets a writer defer.
    ///
    /// Which is the case a bulk import is in for all but its last few rounds,
    /// and the reason deferring exists at all.
    #[test]
    fn a_writer_may_defer_while_the_queue_is_within_the_ceiling() {
        assert!(may_defer(0));
        assert!(may_defer(LAGGING_AT - 1));
    }

    /// The depth that reports a backlog is the depth that bounds it.
    ///
    /// Stated as one assertion rather than two constants because the whole
    /// argument for the pair is that they are the same number: a ceiling above
    /// the signal is an interval during which the queue is known to be running
    /// away and nothing acts on it.
    #[test]
    fn the_ceiling_is_the_depth_that_reports_the_projection_behind() {
        assert!(!may_defer(LAGGING_AT));
        assert!(!may_defer(LAGGING_AT * 10));
    }
}
