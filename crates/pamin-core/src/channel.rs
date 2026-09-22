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

/// One candidate as a channel ranked it, with whatever the channel scored it.
///
/// The score is the channel's own quantity in the channel's own units -- a BM25
/// score from one of the lexical channels, a similarity from the vector one --
/// and it is never comparable across channels. It travels anyway, for the
/// reason [`ChannelResults`] gives.
///
/// `None` means this channel has no score to give, which is a different claim
/// from a score of zero. The graph channel is the case: it reaches a topic
/// across edges rather than scoring it against a query, so a zero there would
/// assert a measurement nobody took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scored {
    pub topic: TopicId,
    pub score: Option<f32>,
}

impl Scored {
    pub fn new(topic: TopicId, score: f32) -> Self {
        Self {
            topic,
            score: Some(score),
        }
    }

    /// A candidate from a channel that does not score.
    pub fn unscored(topic: TopicId) -> Self {
        Self { topic, score: None }
    }
}

/// One channel's ranked candidates, best first.
///
/// Ranks are what fusion combines, and that has not changed: a BM25 score and a
/// vector similarity are not comparable quantities, and rank fusion is what
/// lets them be combined without pretending they are.
///
/// Scores travel alongside them all the same, because the thing rank fusion
/// cannot express is *how sure a channel is*. A channel's first candidate
/// contributes `weight / (k + 1)` whether it found the answer or merely found
/// the least bad of fifty wrong ones, so fusion has no way to tell a confident
/// channel from one that is guessing -- measured on two corpora as fusing all
/// four channels ranking below the vector channel by itself on exactly the
/// queries where the lexical pair has nothing to say. A channel's confidence is
/// legible only in the spread of its own scores, which is why they are carried
/// even though they are not summed.
#[derive(Clone, Debug)]
pub struct ChannelResults {
    pub channel: Channel,
    pub candidates: Vec<Scored>,
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
    pub fn new(channel: Channel, candidates: Vec<Scored>) -> Self {
        Self {
            channel,
            candidates,
        }
    }

    /// The same candidates from a channel that does not score, in the same
    /// order.
    pub fn unscored(channel: Channel, candidates: Vec<TopicId>) -> Self {
        Self::new(
            channel,
            candidates.into_iter().map(Scored::unscored).collect(),
        )
    }

    /// The candidates as topics, best first.
    ///
    /// For the callers that only need the ordering -- seeding the graph walk,
    /// and anything asking which topics a channel proposed at all.
    pub fn topics(&self) -> impl Iterator<Item = TopicId> + '_ {
        self.candidates.iter().map(|candidate| candidate.topic)
    }
}
