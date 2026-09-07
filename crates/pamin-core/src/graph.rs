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
    /// learned about it, and independent of the endpoints' own validity.
    pub valid_from: Option<OffsetDateTime>,
    pub valid_to: Option<OffsetDateTime>,
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

impl RelationshipVersion {
    /// Whether this version is the one currently believed.
    pub fn is_live(&self) -> bool {
        self.invalidated_at.is_none()
    }

    /// Whether the relationship is asserted to hold at `at`.
    ///
    /// An open bound means the assertion extends indefinitely in that
    /// direction, which is the common case: most relationships are stated
    /// without an end date rather than as a closed interval.
    pub fn holds_at(&self, at: OffsetDateTime) -> bool {
        self.valid_from.is_none_or(|from| from <= at) && self.valid_to.is_none_or(|to| at < to)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn version(
        valid_from: Option<OffsetDateTime>,
        valid_to: Option<OffsetDateTime>,
    ) -> RelationshipVersion {
        RelationshipVersion {
            id: RelationshipVersionId::new(),
            relationship_id: RelationshipId::new(),
            version: 1,
            valid_from,
            valid_to,
            created_at: OffsetDateTime::UNIX_EPOCH,
            invalidated_at: None,
            supersedes: None,
            caused_by_topic_state: None,
            confidence: 1.0,
            derivation: Derivation::Explicit,
            tombstone_reason: None,
        }
    }

    fn at(seconds: i64) -> OffsetDateTime {
        OffsetDateTime::UNIX_EPOCH + time::Duration::seconds(seconds)
    }

    #[test]
    fn edge_kind_names_round_trip() {
        for kind in [EdgeKind::Mentions, EdgeKind::DependsOn, EdgeKind::SameAs] {
            assert_eq!(EdgeKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(EdgeKind::parse("entangled_with"), None);
    }

    #[test]
    fn an_unbounded_assertion_holds_at_every_instant() {
        // Most relationships are stated without an end date. Treating an open
        // bound as a closed one would make them invisible to every temporal
        // query, which is the majority of the graph.
        let unbounded = version(None, None);
        assert!(unbounded.holds_at(at(0)));
        assert!(unbounded.holds_at(at(1_000_000)));
    }

    #[test]
    fn a_bounded_assertion_is_half_open() {
        let bounded = version(Some(at(10)), Some(at(20)));
        assert!(!bounded.holds_at(at(9)));
        assert!(bounded.holds_at(at(10)), "the start is included");
        assert!(bounded.holds_at(at(19)));
        assert!(
            !bounded.holds_at(at(20)),
            "the end is excluded, so consecutive intervals do not overlap"
        );
    }

    #[test]
    fn closing_a_version_is_separate_from_the_truth_interval() {
        // System validity and truth validity are different axes: retracting a
        // claim says nothing about when the relationship held.
        let mut closed = version(Some(at(10)), None);
        closed.invalidated_at = Some(at(5));
        closed.tombstone_reason = Some(TombstoneReason::Deleted);

        assert!(!closed.is_live());
        assert!(closed.holds_at(at(15)), "retraction is not an end date");
    }
}
