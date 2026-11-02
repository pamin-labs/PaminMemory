//! The sensory filter.
//!
//! Most of what an agent writes is restatement or transient chatter, and
//! admitting all of it inflates embeddings, index size, and above all retrieval
//! precision. The filter is mandatory and has no configuration switch.
//!
//! It is safe to make mandatory because of where it sits: evidence is persisted
//! before it runs and is never subject to it. The filter decides only whether
//! content is promoted to the retrieval surface, so a false negative stays
//! queryable in the evidence layer and can be promoted later.
//!
//! Every rejection records why, joining the same explainability contract as
//! retrieval scoring: a developer can always ask why something never became a
//! memory.

use serde::{Deserialize, Serialize};

use crate::ledger::FilterDecision;

/// Why content was not promoted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejection {
    /// Nothing but whitespace.
    Empty,
    /// Too short to carry a durable claim.
    TooShort,
    /// Identical to the topic's current content once whitespace is normalized.
    Restatement,
}

impl Rejection {
    /// The reason recorded alongside the evidence.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Empty => "content was empty",
            Self::TooShort => "content was too short to carry a durable claim",
            Self::Restatement => "content restates the topic's current state",
        }
    }
}

/// What the filter decided, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Verdict {
    pub decision: FilterDecision,
    pub rejection: Option<Rejection>,
}

impl Verdict {
    fn promote() -> Self {
        Self {
            decision: FilterDecision::Promoted,
            rejection: None,
        }
    }

    fn reject(rejection: Rejection) -> Self {
        Self {
            decision: FilterDecision::Filtered,
            rejection: Some(rejection),
        }
    }

    /// The reason string stored with the evidence.
    pub fn reason(&self) -> &'static str {
        match self.rejection {
            Some(rejection) => rejection.reason(),
            None => "promoted to the retrieval surface",
        }
    }

    pub fn is_promoted(&self) -> bool {
        self.decision == FilterDecision::Promoted
    }
}

/// The deterministic sensory filter.
///
/// Every rule here is language-neutral by construction. A stop-phrase list
/// would be the obvious way to catch acknowledgements, and it is deliberately
/// absent: any such list is written in one language and would silently filter
/// far less for users writing in another, which is the wrong failure to build
/// in for a product whose evidence may arrive in any language. Model-assisted
/// filtering can be added later as an adapter, but it must not become the only
/// path, since the default install makes no model call on the write path.
#[derive(Clone, Copy, Debug)]
pub struct SensoryFilter {
    /// Minimum characters, counted after whitespace normalization.
    min_chars: usize,
}

impl Default for SensoryFilter {
    fn default() -> Self {
        // Deliberately low. The filter's job is to drop the obviously worthless,
        // and every rejection is recoverable, but a high threshold would still
        // push recoverable content out of reach of anyone not looking for it.
        Self { min_chars: 8 }
    }
}

impl SensoryFilter {
    pub fn new(min_chars: usize) -> Self {
        Self { min_chars }
    }

    /// Decides whether content should be promoted to the retrieval surface.
    ///
    /// `current` is the topic's current content when one exists.
    pub fn judge(&self, content: &str, current: Option<&str>) -> Verdict {
        let normalized = normalize(content);

        if normalized.is_empty() {
            return Verdict::reject(Rejection::Empty);
        }

        // Counted in characters rather than bytes: a byte threshold would demand
        // several times more text in scripts that encode wider.
        if normalized.chars().count() < self.min_chars {
            return Verdict::reject(Rejection::TooShort);
        }

        if current.map(normalize).as_deref() == Some(normalized.as_str()) {
            return Verdict::reject(Rejection::Restatement);
        }

        Verdict::promote()
    }
}

/// Collapses whitespace runs and trims, so a reformatted restatement is still
/// recognized as one.
fn normalize(content: &str) -> String {
    content.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substantive_content_is_promoted() {
        let verdict = SensoryFilter::default().judge("deploys through the ci pipeline", None);
        assert!(verdict.is_promoted());
    }

    #[test]
    fn whitespace_is_rejected_as_empty() {
        let verdict = SensoryFilter::default().judge("   \n\t ", None);
        assert_eq!(verdict.rejection, Some(Rejection::Empty));
    }

    #[test]
    fn very_short_content_is_rejected() {
        let verdict = SensoryFilter::default().judge("ok", None);
        assert_eq!(verdict.rejection, Some(Rejection::TooShort));
    }

    #[test]
    fn a_reformatted_restatement_is_still_a_restatement() {
        let verdict =
            SensoryFilter::default().judge("deploys   through\nci", Some("deploys through ci"));
        assert_eq!(verdict.rejection, Some(Rejection::Restatement));
    }

    #[test]
    fn a_genuine_update_is_promoted_over_the_current_state() {
        let verdict =
            SensoryFilter::default().judge("deploys through argo", Some("deploys through ci"));
        assert!(verdict.is_promoted());
    }

    #[test]
    fn the_length_rule_counts_characters_not_bytes() {
        // Eight characters that occupy far more than eight bytes. A byte
        // threshold would let this through while rejecting eight ASCII
        // characters, which is the wrong way round.
        let verdict = SensoryFilter::default().judge("部署走持续集成流水", None);
        assert!(
            verdict.is_promoted(),
            "scripts that encode wider must not face a higher bar"
        );
    }

    #[test]
    fn every_verdict_carries_a_reason() {
        let filter = SensoryFilter::default();
        assert!(!filter.judge("", None).reason().is_empty());
        assert!(
            !filter
                .judge("a durable claim about deploys", None)
                .reason()
                .is_empty()
        );
    }
}
