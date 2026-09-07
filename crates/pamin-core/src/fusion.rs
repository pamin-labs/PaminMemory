//! Reciprocal rank fusion and the modifiers applied after it.
//!
//! Fusion happens here rather than inside a retrieval engine, and that is a
//! correctness requirement rather than a preference. The graph channel lives in
//! PostgreSQL, where the projection index cannot see it. An engine that pre-fused
//! its own channels would hand back a list that then had to be fused again with
//! the graph list, weighting the pre-fused members twice, and it would erase the
//! per-channel ranks that every result is required to be able to report.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::channel::{Channel, ChannelResults};
use crate::graph::{Derivation, EdgeKind};
use crate::id::TopicStateId;
use crate::ledger::RetrievalSignals;

/// The standard reciprocal rank fusion constant.
///
/// It needs no tuning and is used unchanged across systems and datasets, which
/// is most of why rank fusion is the default here.
pub const DEFAULT_K: f32 = 60.0;

/// One line of the explanation attached to a result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Why {
    /// The result appeared in this channel at this rank.
    Channel {
        channel: Channel,
        rank: u32,
        weight: f32,
        contribution: f32,
    },
    /// A post-fusion modifier adjusted the score.
    Modifier { modifier: Modifier, factor: f32 },
    /// The graph reached this result from somewhere else, along this edge.
    ///
    /// Carried separately from the channel entry because it answers a
    /// different question. The channel entry says how highly the graph ranked
    /// this result; this says why the graph could see it at all, which is the
    /// only part a reader can check against their own understanding of how two
    /// topics relate. It also distinguishes an edge somebody asserted from one
    /// the engine derived, so a surprising connection can be traced to whoever
    /// or whatever claimed it.
    Path {
        /// The name of the topic the walk started from, which is one of the
        /// results the other channels found.
        from: String,
        /// The name of the topic on the other end of the final edge.
        ///
        /// A name rather than an identifier because this entry exists to be
        /// checked by whoever reads it, and topics are addressed by name
        /// everywhere else a caller touches them.
        via: String,
        /// Edges traversed from the seed. Never zero.
        hops: u8,
        /// Named `edge` rather than `kind`, which serde already uses to tag
        /// the variant itself.
        edge: EdgeKind,
        derivation: Derivation,
    },
}

/// A post-fusion adjustment.
///
/// Naming these as a closed set rather than free strings is what makes
/// "applied exactly once" checkable: a typo cannot quietly become a second,
/// separately-counted modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modifier {
    /// Explicit importance assigned to the state.
    Importance,
    /// The balance of successful against failed outcomes it took part in.
    Worth,
    /// The state has been replaced by a newer one.
    Superseded,
}

/// A fused result and the reasoning behind its position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedResult {
    pub topic_state: TopicStateId,
    pub score: f32,
    pub why: Vec<Why>,
}

/// Fuses ranked lists and applies post-fusion modifiers.
#[derive(Clone, Debug)]
pub struct Fusion {
    k: f32,
    weights: BTreeMap<Channel, f32>,
}

impl Default for Fusion {
    fn default() -> Self {
        // Equal weights until the evaluation harness has something to say.
        // Guessing weights before measuring is how a retrieval stack acquires
        // constants nobody can later justify.
        Self {
            k: DEFAULT_K,
            weights: BTreeMap::new(),
        }
    }
}

impl Fusion {
    /// Overrides the weight of one channel.
    pub fn with_weight(mut self, channel: Channel, weight: f32) -> Self {
        self.weights.insert(channel, weight);
        self
    }

    fn weight(&self, channel: Channel) -> f32 {
        self.weights.get(&channel).copied().unwrap_or(1.0)
    }

    /// Fuses per-channel ranked lists into one ordered result set.
    pub fn fuse(&self, lists: &[ChannelResults]) -> Vec<FusedResult> {
        let mut accumulated: BTreeMap<TopicStateId, (f32, Vec<Why>)> = BTreeMap::new();

        for list in lists {
            let weight = self.weight(list.channel);
            for (index, candidate) in list.candidates.iter().enumerate() {
                let rank = index as u32 + 1;
                let contribution = weight / (self.k + rank as f32);
                let entry = accumulated.entry(*candidate).or_insert((0.0, Vec::new()));
                entry.0 += contribution;
                entry.1.push(Why::Channel {
                    channel: list.channel,
                    rank,
                    weight,
                    contribution,
                });
            }
        }

        let mut results: Vec<FusedResult> = accumulated
            .into_iter()
            .map(|(topic_state, (score, why))| FusedResult {
                topic_state,
                score,
                why,
            })
            .collect();

        sort_results(&mut results);
        results
    }
}

/// Post-fusion adjustments.
///
/// Each modifier is applied exactly once per result. Applying one twice, or
/// applying it here after a channel already expressed the same signal, inflates
/// whatever it measures without anything in the trace revealing that it
/// happened.
#[derive(Clone, Copy, Debug)]
pub struct Modifiers {
    /// How strongly explicit importance lifts a result.
    pub importance_weight: f32,
    /// How strongly the balance of successful against failed outcomes lifts it.
    pub worth_weight: f32,
    /// What a superseded state keeps of its score.
    pub superseded_factor: f32,
}

impl Default for Modifiers {
    fn default() -> Self {
        Self {
            importance_weight: 0.2,
            worth_weight: 0.2,
            superseded_factor: 0.5,
        }
    }
}

impl Modifiers {
    /// Applies every modifier to one result, appending a trace line for each.
    ///
    /// `is_current` says whether this state is the topic's current one; a
    /// superseded state is down-weighted rather than removed, because a query
    /// about how something changed needs it.
    pub fn apply(&self, result: &mut FusedResult, signals: &RetrievalSignals, is_current: bool) {
        let importance = 1.0 + self.importance_weight * signals.importance.clamp(0.0, 1.0);
        self.record(result, Modifier::Importance, importance);

        let worth = 1.0 + self.worth_weight * worth_ratio(signals);
        self.record(result, Modifier::Worth, worth);

        if !is_current {
            self.record(result, Modifier::Superseded, self.superseded_factor);
        }
    }

    fn record(&self, result: &mut FusedResult, modifier: Modifier, factor: f32) {
        debug_assert!(
            !result.why.iter().any(|why| matches!(
                why,
                Why::Modifier { modifier: existing, .. } if *existing == modifier
            )),
            "modifier {modifier:?} applied twice to one result"
        );
        result.score *= factor;
        result.why.push(Why::Modifier { modifier, factor });
    }
}

/// Where a state sits between failure and success, mapped onto -1.0 to 1.0.
///
/// A state nothing has been learned about scores zero, so it is neither
/// promoted nor punished for being new.
fn worth_ratio(signals: &RetrievalSignals) -> f32 {
    let total = signals.worth_positive + signals.worth_negative;
    if total == 0 {
        return 0.0;
    }
    (signals.worth_positive as f32 - signals.worth_negative as f32) / total as f32
}

/// Orders by score, breaking ties by identifier.
///
/// The tie-break is not cosmetic. Context assembly must produce the same
/// ordering for the same inputs, because an unstable order turns a reusable
/// prompt prefix into a fresh one and silently discards the cache hit.
pub fn sort_results(results: &mut [FusedResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.topic_state.0.cmp(&right.topic_state.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> TopicStateId {
        TopicStateId(uuid::Uuid::from_bytes([byte; 16]))
    }

    #[test]
    fn appearing_in_two_channels_beats_appearing_in_one() {
        let both = id(1);
        let single = id(2);

        let fused = Fusion::default().fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, vec![single, both]),
            ChannelResults::new(Channel::Vector, vec![both]),
        ]);

        assert_eq!(
            fused[0].topic_state, both,
            "agreement across channels should outrank a single strong hit"
        );
    }

    #[test]
    fn the_trace_reports_the_rank_in_every_channel_it_appeared_in() {
        let target = id(1);
        let fused = Fusion::default().fuse(&[
            ChannelResults::new(Channel::LexicalNgram, vec![id(9), target]),
            ChannelResults::new(Channel::Vector, vec![target]),
        ]);

        let entry = fused.iter().find(|r| r.topic_state == target).unwrap();
        let ranks: Vec<_> = entry
            .why
            .iter()
            .filter_map(|why| match why {
                Why::Channel { channel, rank, .. } => Some((*channel, *rank)),
                Why::Modifier { .. } | Why::Path { .. } => None,
            })
            .collect();

        assert!(ranks.contains(&(Channel::LexicalNgram, 2)));
        assert!(ranks.contains(&(Channel::Vector, 1)));
    }

    #[test]
    fn fusion_uses_ranks_so_channel_score_scales_never_meet() {
        // Both channels contribute the same amount at the same rank, whatever
        // the underlying scores were. That is the property that lets a BM25
        // score and a vector distance be combined at all.
        let fused = Fusion::default().fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, vec![id(1)]),
            ChannelResults::new(Channel::Vector, vec![id(2)]),
        ]);
        assert!((fused[0].score - fused[1].score).abs() < f32::EPSILON);
    }

    #[test]
    fn channel_weights_shift_the_balance() {
        let fused = Fusion::default().with_weight(Channel::Vector, 2.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, vec![id(1)]),
            ChannelResults::new(Channel::Vector, vec![id(2)]),
        ]);
        assert_eq!(fused[0].topic_state, id(2));
    }

    #[test]
    fn each_modifier_appears_at_most_once_in_the_trace() {
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);

        Modifiers::default().apply(
            &mut fused,
            &RetrievalSignals {
                importance: 0.8,
                worth_positive: 3,
                worth_negative: 1,
                ..RetrievalSignals::default()
            },
            true,
        );

        let mut applied: Vec<_> = fused
            .why
            .iter()
            .filter_map(|why| match why {
                Why::Modifier { modifier, .. } => Some(*modifier),
                Why::Channel { .. } | Why::Path { .. } => None,
            })
            .collect();
        let before = applied.len();
        applied.sort_unstable_by_key(|modifier| format!("{modifier:?}"));
        applied.dedup();
        assert_eq!(
            before,
            applied.len(),
            "a modifier was applied more than once"
        );
    }

    #[test]
    fn a_superseded_state_is_down_weighted_rather_than_dropped() {
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);
        let original = fused.score;

        Modifiers::default().apply(&mut fused, &RetrievalSignals::default(), false);

        assert!(fused.score < original);
        assert!(
            fused.score > 0.0,
            "history a query might ask for must stay reachable"
        );
    }

    #[test]
    fn a_state_with_no_recorded_outcomes_is_neither_promoted_nor_punished() {
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);
        let original = fused.score;

        Modifiers::default().apply(&mut fused, &RetrievalSignals::default(), true);

        assert!((fused.score - original).abs() < f32::EPSILON);
    }

    #[test]
    fn path_evidence_is_neither_a_channel_nor_a_modifier() {
        // The channel entry says how highly the graph ranked this result; the
        // path says why the graph could see it at all. Counting a path as
        // either of the others would inflate a rank tally or trip the
        // applied-once check on modifiers.
        let target = id(1);
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Graph, vec![target])])
            .remove(0);

        fused.why.push(Why::Path {
            from: "oncall_rota".to_string(),
            via: "release_process".to_string(),
            hops: 2,
            edge: EdgeKind::DependsOn,
            derivation: Derivation::Deterministic,
        });

        Modifiers::default().apply(&mut fused, &RetrievalSignals::default(), true);

        let channels = fused
            .why
            .iter()
            .filter(|why| matches!(why, Why::Channel { .. }))
            .count();
        let paths = fused
            .why
            .iter()
            .filter(|why| matches!(why, Why::Path { .. }))
            .count();
        assert_eq!(channels, 1, "one graph rank, not two");
        assert_eq!(paths, 1);
    }

    #[test]
    fn equal_scores_order_the_same_way_every_time() {
        let lists = [ChannelResults::new(
            Channel::Vector,
            vec![id(3), id(1), id(2)],
        )];
        let first = Fusion::default().fuse(&lists);

        let mut tied: Vec<FusedResult> = first
            .iter()
            .map(|result| FusedResult {
                score: 1.0,
                ..result.clone()
            })
            .collect();
        sort_results(&mut tied);
        let mut reversed: Vec<FusedResult> = tied.iter().rev().cloned().collect();
        sort_results(&mut reversed);

        let left: Vec<_> = tied.iter().map(|r| r.topic_state).collect();
        let right: Vec<_> = reversed.iter().map(|r| r.topic_state).collect();
        assert_eq!(left, right, "ordering must not depend on input order");
    }
}
