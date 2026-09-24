//! The relationship graph.
//!
//! The graph is topic-centred: edges connect stable topic identities, and each
//! endpoint is resolved to a version at retrieval time. An edge asserted between
//! two topics therefore survives both of them changing, which is the point —
//! "these two things are related" is a longer-lived claim than any one version
//! of either.
//!
//! Edges are versioned on the same terms as topic states. Changing one closes
//! the current version and appends a new one, so a query about what we believed
//! before a relationship changed has something to read.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::id::{ProjectId, RelationshipId, RelationshipVersionId, TopicId, TopicStateId};
use crate::ledger::Validity;

/// What one topic asserts about another.
///
/// A closed set rather than free strings, for the same reason the post-fusion
/// modifiers are: a typo in an edge type would otherwise become a silent
/// second kind that nothing traverses and nothing reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// One topic's content names another.
    Mentions,
    Supports,
    Contradicts,
    Supersedes,
    RelatedTo,
    PartOf,
    DerivedFrom,
    SameAs,
    DependsOn,
}

impl EdgeKind {
    /// Parses an edge kind from its wire name.
    pub fn parse(name: &str) -> Option<Self> {
        [
            Self::Mentions,
            Self::Supports,
            Self::Contradicts,
            Self::Supersedes,
            Self::RelatedTo,
            Self::PartOf,
            Self::DerivedFrom,
            Self::SameAs,
            Self::DependsOn,
        ]
        .into_iter()
        .find(|kind| kind.as_str() == name)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mentions => "mentions",
            Self::Supports => "supports",
            Self::Contradicts => "contradicts",
            Self::Supersedes => "supersedes",
            Self::RelatedTo => "related_to",
            Self::PartOf => "part_of",
            Self::DerivedFrom => "derived_from",
            Self::SameAs => "same_as",
            Self::DependsOn => "depends_on",
        }
    }
}

/// How an edge came to exist.
///
/// Recorded rather than inferred, because it is the difference between a claim
/// a caller made and one the engine guessed at, and a result explaining itself
/// should be able to say which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Derivation {
    /// Asserted directly through the CLI or API.
    Explicit,
    /// Derived by a rule with no model in the path.
    Deterministic,
    /// Extracted by a local model.
    Model,
    /// Carried in from another system.
    Imported,
}

/// Why an edge version stopped being believed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TombstoneReason {
    /// The relationship ended; nothing replaced it.
    Closed,
    /// A newer version took over.
    Superseded,
    /// Retracted by a caller.
    Deleted,
}

/// The stable identity of one edge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub id: RelationshipId,
    pub project_id: ProjectId,
    pub from_topic: TopicId,
    pub to_topic: TopicId,
    pub kind: EdgeKind,
    pub created_at: OffsetDateTime,
}

/// One immutable fact about an edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationshipVersion {
    pub id: RelationshipVersionId,
    pub relationship_id: RelationshipId,
    pub version: u32,
    /// When the relationship is asserted to hold. Independent of when we
    /// learned about it, and independent of the endpoints' own validity: a
    /// relationship can predate or outlive the topic versions it connects.
    pub validity: Validity,
    /// When we recorded the claim.
    pub created_at: OffsetDateTime,
    /// When we stopped standing behind it. `None` means this is the live one.
    pub invalidated_at: Option<OffsetDateTime>,
    pub supersedes: Option<RelationshipVersionId>,
    /// The topic state that caused the claim, absent for an explicit edge.
    pub caused_by_topic_state: Option<TopicStateId>,
    /// Orders neighbours at equal graph distance.
    pub confidence: f32,
    pub derivation: Derivation,
    pub tombstone_reason: Option<TombstoneReason>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_kind_names_round_trip() {
        for kind in [EdgeKind::Mentions, EdgeKind::DependsOn, EdgeKind::SameAs] {
            assert_eq!(EdgeKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(EdgeKind::parse("entangled_with"), None);
    }
}
