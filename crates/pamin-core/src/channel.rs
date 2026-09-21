//! Recall channels and the ranked lists they return.

use serde::{Deserialize, Serialize};

use crate::id::TopicId;

/// A source of candidates.
///
/// There are three, and the list is short on purpose. Earlier designs also
/// treated recency and explicit importance as channels while applying them
/// again as post-fusion modifiers, which counted the same signal twice. Notes
/// and page nodes were channels too, although they live in the same projection
/// index as everything else, so querying them separately split one population
/// into several and left the redundancy penalty reasoning across all of them.
/// Both are now expressed as modifiers and document-type filters instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    /// BM25 over segmented text. Word-level recall in any language.
    LexicalSegmented,
    /// BM25 over character n-grams. Substrings that segmentation destroys:
    /// paths, error codes, identifiers.
    LexicalNgram,
    /// Approximate nearest neighbours over dense embeddings.
    Vector,
    /// Expansion through the relationship graph, which lives in PostgreSQL and
    /// is therefore invisible to the projection index.
    Graph,
}

impl Channel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LexicalSegmented => "lexical_segmented",
            Self::LexicalNgram => "lexical_ngram",
            Self::Vector => "vector",
            Self::Graph => "graph",
        }
    }
}

/// One channel's ranked candidates, best first.
///
/// Ranks travel; scores do not. BM25 scores and vector distances are not
/// comparable quantities, and rank fusion is what lets them be combined without
/// pretending they are.
#[derive(Clone, Debug)]
pub struct ChannelResults {
    pub channel: Channel,
    pub candidates: Vec<TopicId>,
}

impl Channel {
    /// Whether this channel scores wording rather than meaning or structure.
    ///
    /// The pair is asked together wherever it is asked at all: they run BM25
    /// over the same text, one over segmented words and one over character
    /// n-grams, so they agree with each other far more than either agrees with
    /// the vector or the graph. Naming the pair in one place keeps a third
    /// lexical channel from being added to the fusion weights and forgotten
    /// here.
    pub fn is_lexical(self) -> bool {
        matches!(self, Self::LexicalSegmented | Self::LexicalNgram)
    }
}

impl ChannelResults {
    pub fn new(channel: Channel, candidates: Vec<TopicId>) -> Self {
        Self {
            channel,
            candidates,
        }
    }
}
