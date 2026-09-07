//! The evidence and version ledger.
//!
//! Two rules shape every type here. Evidence is immutable and is written before
//! anything decides whether it is worth remembering, so a filtering mistake is
//! recoverable. And nothing is updated in place: a change appends a new version
//! and links back to the one it supersedes, so the history of a fact stays
//! readable rather than being overwritten by its latest value.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::id::{ProjectId, SourceId, SourceSpanId, SourceVersionId, TopicId, TopicStateId};

/// A namespace for memories and sources, and the shard key for every table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub id: ProjectId,
    pub name: String,
    pub created_at: OffsetDateTime,
}

/// Where evidence came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// Written directly through the CLI or API.
    Manual,
    File,
    Directory,
    ChatLog,
}

/// A stable identity for something that produces evidence over time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub id: SourceId,
    pub project_id: ProjectId,
    pub kind: SourceKind,
    /// Path, URI, or label. Unique per project, so re-ingesting the same file
    /// appends a version instead of creating a second source.
    pub locator: String,
    pub created_at: OffsetDateTime,
}

/// What the sensory filter decided about a piece of evidence.
///
/// The filter gates promotion to the retrieval surface, never persistence. A
/// rejected version is still stored, still queryable, and still promotable
/// later, which is what makes a mandatory filter safe to run at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterDecision {
    /// Promoted to a topic state.
    Promoted,
    /// Held in the evidence layer only.
    Filtered,
}

/// An immutable snapshot of a source's content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceVersion {
    pub id: SourceVersionId,
    pub project_id: ProjectId,
    pub source_id: SourceId,
    /// Monotonic per source.
    pub version: u32,
    /// Stored verbatim, in whatever language it arrived in. Never translated,
    /// never rewritten.
    pub content: String,
    /// Content hash, used to skip re-ingesting unchanged sources.
    pub content_hash: String,
    pub filter_decision: FilterDecision,
    /// Why the filter decided as it did, in the same explainability spirit as a
    /// retrieval `why` trace.
    pub filter_reason: String,
    pub recorded_at: OffsetDateTime,
}

/// A byte range into a source version, and the unit a retrieval hit cites.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceSpan {
    pub id: SourceSpanId,
    pub project_id: ProjectId,
    pub source_version_id: SourceVersionId,
    /// Byte offsets into the source version's content. Half-open.
    pub byte_start: u32,
    pub byte_end: u32,
    /// BCP-47 tag from per-span detection, absent when detection was not
    /// confident. Drives tokenizer choice and the note-language rule.
    pub detected_language: Option<String>,
    pub language_confidence: Option<f32>,
}

/// A stable topic identity. Content lives in its states, not here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Topic {
    pub id: TopicId,
    pub project_id: ProjectId,
    /// The handle agents and humans use.
    pub name: String,
    /// Optional hierarchical namespace, for when names would collide.
    pub path: Option<String>,
    pub created_at: OffsetDateTime,
}

/// One immutable version of a topic's content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TopicState {
    pub id: TopicStateId,
    pub project_id: ProjectId,
    pub topic_id: TopicId,
    /// Monotonic per topic. Gaps are expected, because soft deletes do not
    /// renumber the versions that survive.
    pub version: u32,
    pub content: String,
    /// The span this state was derived from, so every claim can be traced back
    /// to bytes in a source.
    pub source_span_id: SourceSpanId,
    /// When the source claims the fact was true or happened.
    pub observed_at: OffsetDateTime,
    /// When PaminMemory recorded it.
    pub recorded_at: OffsetDateTime,
    /// Optional truth interval, independent of when we learned about it.
    pub valid_from: Option<OffsetDateTime>,
    pub valid_to: Option<OffsetDateTime>,
    /// The state this one replaced, forming the update chain.
    pub supersedes: Option<TopicStateId>,
    /// Set when soft deleted. Deleted states leave the default retrieval
    /// surface but stay available for audit and historical traversal.
    pub deleted_at: Option<OffsetDateTime>,
    pub signals: RetrievalSignals,
}

/// Per-state signals that feed post-fusion modifiers.
///
/// These are modifiers rather than recall channels. Recency and importance used
/// to appear as candidate channels as well, which counted the same evidence
/// twice: once when it was recalled and again when it was reranked.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct RetrievalSignals {
    pub importance: f32,
    /// Times this state co-occurred with a successful outcome.
    pub worth_positive: u32,
    /// Times it co-occurred with a failed one.
    pub worth_negative: u32,
    pub access_count: u32,
    pub last_accessed_at: Option<OffsetDateTime>,
}
