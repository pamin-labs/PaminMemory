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
    /// The range this channel's score means the same thing over, if any.
    ///
    /// **Bounded is not the same as calibrated, and only one of the two is
    /// useful here.** A cosine similarity is bounded on `[-1, 1]` and a BM25
    /// score is not bounded at all, but neither is *calibrated*: a cosine of
    /// 0.7 does not mean a fixed thing from one query to the next, and the
    /// fifty candidates a query returns cluster into a narrow part of the range
    /// that moves with the query. Normalising those against their theoretical
    /// bounds would compress a channel's whole list into one corner of the
    /// band and destroy its internal ordering, which is the opposite of what
    /// is wanted.
    ///
    /// The graph channel is different, and it is the only one. Its score is
    /// the seed's relevance times `confidence * decay^(hops - 1)`: `confidence`
    /// is constrained by the schema to `(0, 1]`, the decay is a constant, and a
    /// seed's relevance is a fixed function of its best rank, `(k + 1) /
    /// (k + rank)`, or one for a topic the query names. Every factor is on
    /// `(0, 1]` and none is relative to the other candidates, so the product
    /// means the same thing on every query -- one derived mention from the
    /// best-matching memory is 0.5 wherever it occurs, and 1.0 is an edge
    /// somebody asserted outright from a topic the query named. That is a
    /// quantity a normaliser must not touch: min-maxing it inside one query
    /// maps whatever the best path happened to be onto the top of the band, so
    /// a channel whose only path is a single weak guess votes exactly as
    /// loudly as one that found an explicit assertion, and a channel with
    /// nothing good to say cannot say so.
    ///
    /// Worse, in production it is degenerate. Every edge the write path
    /// derives is a `Mentions` at the same 0.5, and the walk stops at one hop
    /// whenever one hop fills the channel's depth -- so every candidate scores
    /// 0.5, the empirical minimum equals the maximum, and fusion falls back to
    /// ranking by the order the walk returned. That order's live tie-break,
    /// once distance and confidence are constant, is the topic's identifier.
    /// **A channel was ordering the head of a search by UUID.**
    pub fn calibrated(self) -> Option<(f32, f32)> {
        match self {
            // BM25 is unbounded above and scale-free; a cosine similarity is
            // bounded and query-relative. See above for why neither qualifies.
            Self::LexicalSegmented | Self::LexicalNgram | Self::Vector => None,
            // `confidence` is `(0, 1]` by a schema constraint and the hop
            // decay only ever reduces it.
            Self::Graph => Some((0.0, 1.0)),
        }
    }

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
/// **Larger is always better, and a channel whose engine reports a distance
/// converts before it gets here.** That is the one thing about this field that
/// is not the channel's own business, because everything reading it compares
/// magnitudes: a channel handing over a distance would be summed backwards and
/// judged on its worst candidate. The vector channel is that case -- zvec
/// reports cosine *distance*, so [`pamin_index`] subtracts it from one -- and
/// the test that would have caught it did not, because every document in it was
/// written with the same stub vector and the whole channel reported one
/// constant.
///
/// [`pamin_index`]: https://docs.rs/pamin-index
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
/// legible only in the spread of its own scores, which is why they are carried:
/// [`Combine::Banded`] orders a channel's candidates inside its band by them,
/// and the trace reports them.
///
/// [`Combine::Banded`]: crate::Combine::Banded
#[derive(Clone, Debug)]
pub struct ChannelResults {
    pub channel: Channel,
    pub candidates: Vec<Scored>,
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
