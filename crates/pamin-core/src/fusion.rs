//! Reciprocal rank fusion and the modifiers applied after it.
//!
//! Fusion happens here rather than inside a retrieval engine, and that is a
//! correctness requirement rather than a preference. The graph channel lives in
//! PostgreSQL, where the projection index cannot see it. An engine that pre-fused
//! its own channels would hand back a list that then had to be fused again with
//! the graph list, weighting the pre-fused members twice, and it would erase the
//! per-channel ranks that every result is required to be able to report.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::channel::{Channel, ChannelResults};
use crate::graph::{Derivation, EdgeKind};
use crate::id::TopicId;
use crate::ledger::RetrievalSignals;

/// How sharply a result's rank in one channel counts toward its fused score.
///
/// The rank fusion literature uses 60, which came from runs over lists
/// thousands of results deep. Each channel here proposes fifty, and at 60 the
/// curve across fifty candidates is almost flat: rank 1 contributes 0.0164 and
/// rank 10 contributes 0.0143, so the whole top ten spans 14% and a channel
/// that put the right memory first says barely more than one that put it
/// tenth.
///
/// Ten, measured. On this project's evaluation corpus, moving 60 to 10 with
/// the weights below takes cross-lingual nDCG@10 from 0.2245 to 0.3383, and
/// monolingual from 0.9892 to 0.9940 rather than paying for it.
///
/// Ten rather than the best number measured. The curve is monotonic all the
/// way down -- 5 scores 0.3599 and 1 scores 0.3874 -- which means the corpus
/// cannot locate an optimum, only say that 60 is too flat for lists this
/// short. Taking the boundary would be fitting a constant to 137 queries
/// somebody here wrote. Ten is a fifth of the channel depth, well inside the
/// improving region, and far from the point where rank 1 counts double rank 2
/// and one channel's mistaken top hit decides the answer.
pub const DEFAULT_K: f32 = 10.0;

/// What a lexical channel is worth when it agrees fully with the vector one.
///
/// The ceiling of the per-query weight rather than the weight itself; see
/// [`Fusion::lexical_agreement`].
///
/// An eighth, and it was a quarter until a third corpus was measured. The
/// sweep that settled the quarter ran `1.00 / 0.50 / 0.25 / 0.00` -- so **a
/// quarter was the smallest non-zero value ever tried**, and the argument for
/// it was against zero rather than against half of itself. Half of itself is
/// better on every group of every corpus but one; see [`Fusion::default`].
const LEXICAL_WEIGHT: f32 = 0.125;

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
        /// The end the final edge was asserted from, and the end it points at.
        ///
        /// The graph walk is undirected, so `via` says how this result was
        /// reached rather than what was claimed, and the same edge reads in
        /// opposite directions depending on which end the walk started from.
        /// For `depends_on` and its like the direction is the claim, so it is
        /// named outright rather than left to be inferred from the traversal.
        asserted_from: String,
        asserted_to: String,
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
}

/// A fused result and the reasoning behind its position.
///
/// A topic rather than one of its states. The channels rank topics because the
/// projection holds one document per topic: a topic's history lives in the
/// ledger and is read by version, never ranked against itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedResult {
    pub topic: TopicId,
    pub score: f32,
    pub why: Vec<Why>,
}

/// Fuses ranked lists and applies post-fusion modifiers.
#[derive(Clone, Debug)]
pub struct Fusion {
    k: f32,
    weights: BTreeMap<Channel, f32>,
    /// Whether the lexical weight follows the query rather than the config.
    ///
    /// Off whenever a caller sets a lexical weight by hand, so a sweep
    /// measures the constant it asked for.
    adaptive: bool,
    /// What the lexical pair keeps at zero agreement, as a share of its weight.
    agreement_floor: f32,
    /// What it reaches at full agreement, as a share of its weight.
    agreement_ceiling: f32,
}

impl Default for Fusion {
    fn default() -> Self {
        // The two lexical channels count as half of one, and both halves of
        // that are measured rather than reasoned.
        //
        // They are nearly one channel: both run BM25 over the same text, one
        // over segmented words and one over character n-grams, so they agree
        // with each other far more often than either agrees with the vector or
        // the graph. Counted at full weight the pair outvotes the other two on
        // every query where the wording matches and the meaning does not.
        //
        // What is not obvious, and needed a corpus where a query and its answer
        // are in different languages, is that half of one channel is still too
        // much -- and a third corpus, judged by people rather than aligned by
        // translation, said a quarter is still too much. Fusion alone at
        // `k=10`, nDCG@10, cross-lingual where a corpus has two groups:
        //
        //   weight      ours   XQuAD-R   XQuAD-R same   MIRACL sw
        //     1.00    0.4329    0.1569         0.8332      0.5676
        //     0.50    0.6619    0.4572         0.8379      0.6606
        //     0.25    0.7322    0.5700         0.8057      0.6826
        //     0.125   0.7910    0.6077         0.7556      0.6882
        //     0.00    0.8268    0.6335         0.6787      0.6848
        //
        // Equal weighting is not a trade at any `k`: it is worse than half
        // everywhere it can be, which is every group of every corpus except
        // our lexical group, where both sit on the ceiling. Zero is a trade
        // and a bad one -- it takes our monolingual group from 0.9940 to
        // 0.9860 and the lexical group off that ceiling, which is the one
        // thing the n-gram channel exists for, and it would make both channels
        // dead code.
        //
        // An eighth is the one value that is not a trade against zero at all.
        // It holds the monolingual and lexical groups at the same 0.9940 and
        // 1.0000 the quarter held.
        //
        // Two sentences that used to stand here have been withdrawn: that an
        // eighth is the best of the five on MIRACL, and that on MIRACL the
        // quarter ranks below the vector channel on its own. Paired against
        // the quarter, MIRACL gives 77 wins to 72 losses at p = 0.2300, and
        // against zero weight, 30 to 59 at p = 0.2972. Neither is a result.
        // On 482 real single-language queries nothing in the range from zero
        // to a quarter is distinguishable at all, while half weight and full
        // weight are clearly worse -- so the corpus can separate what matters
        // and cannot separate these.
        //
        // What holds is the pair of cross-lingual groups: 698 wins to 26 on
        // XQuAD-R and 21 to 2 on this project's own corpus, both at p <= 0.0004.
        // And what it costs holds too -- XQuAD-R's same-language group loses on
        // 252 queries and wins on 16. That is a two-sided trade with both sides
        // measured, taken because the corpus on the losing side writes its
        // questions out of its answers' own words.
        // Its single cost is XQuAD-R's same-language group, -0.0501, and that
        // is the one quantity the two same-language corpora disagree about by
        // twenty times, because SQuAD's questions are written out of their
        // answers' own words. See `docs/adr/0001-tech-selection.md`.
        //
        // Two things about that table are worth distrusting, and both point the
        // same way -- that a single global constant is the wrong shape.
        //
        // The MIRACL column is +0.0056 over the quarter. Nothing here has ever
        // checked whether 482 queries support a difference that size; the
        // harnesses only learned to ask in `statistics`, and until that
        // comparison is re-run this row is a mean with no evidence under it.
        //
        // And the published work predicts the opposite sign for that corpus.
        // MIRACL's own authors report a BM25-dense hybrid as the strongest
        // zero-shot baseline and name Swahili among the languages where the
        // dense side is the weak one -- which argues the lexical pair should be
        // worth *more* there, not less. Meanwhile `arXiv:2510.00671` (2025)
        // establishes that lexical matching cannot cross a language boundary
        // without a shared lexical space, which argues it should be worth
        // nothing at all on XQuAD-R's cross-lingual group. Those two together
        // do not describe a constant; they describe a weight that belongs to a
        // corpus. One number is serving both, and the number it settles on is
        // whichever corpus was measured loudest.
        Self {
            k: DEFAULT_K,
            weights: BTreeMap::from([
                (Channel::LexicalSegmented, LEXICAL_WEIGHT),
                (Channel::LexicalNgram, LEXICAL_WEIGHT),
            ]),
            agreement_floor: 0.0,
            agreement_ceiling: 1.0,
            // Off by default. It is worth a great deal on one group of one
            // corpus and it takes the other group apart -- see
            // `lexical_agreement`. A default has to be good for a user and
            // good on every benchmark, and this is not that yet.
            adaptive: false,
        }
    }
}

impl Fusion {
    /// Overrides the weight of one channel, and pins it.
    ///
    /// Setting a lexical weight by hand turns the per-query adjustment off:
    /// a sweep asked for a constant and has to be measured on one, or the
    /// column it prints is not the column it named.
    pub fn with_weight(mut self, channel: Channel, weight: f32) -> Self {
        self.weights.insert(channel, weight);
        if channel.is_lexical() {
            self.adaptive = false;
        }
        self
    }

    /// Scales the lexical weight by per-query agreement, between `floor` and
    /// `ceiling` of the configured weight.
    ///
    /// Opt-in, and swept rather than argued. `floor` is what the lexical pair
    /// keeps when it agrees with the vector channel about nothing, `ceiling`
    /// what it reaches at full agreement; `with_adaptive(1.0, 1.0)` is the
    /// constant, and `with_adaptive(0.0, 1.0)` is the unbounded form measured
    /// below.
    pub fn with_adaptive(mut self, floor: f32, ceiling: f32) -> Self {
        self.adaptive = true;
        self.agreement_floor = floor.clamp(0.0, 1.0);
        self.agreement_ceiling = ceiling.clamp(0.0, 1.0).max(floor);
        self
    }

    /// Overrides the rank constant.
    ///
    /// Alongside [`with_weight`](Self::with_weight) because the two are the
    /// whole of what fusion can be tuned to, and the evaluation harnesses
    /// sweep them together -- neither number was arrived at by argument and
    /// neither should be changed by one.
    pub fn with_k(mut self, k: f32) -> Self {
        self.k = k;
        self
    }

    fn weight(&self, channel: Channel) -> f32 {
        self.weights.get(&channel).copied().unwrap_or(1.0)
    }

    /// What the lexical channels are worth on this query, and why.
    ///
    /// The sweep that settled the constant also showed the constant cannot be
    /// right. On XQuAD-R at `k = 10`, fusing the four channels against running
    /// the embedding model alone:
    ///
    /// ```text
    ///   lexical   cross-lingual nDCG@10   same-language
    ///      0.00                  0.6335          0.6787
    ///      0.125                 0.6077          0.7556
    ///      0.25                  0.5700          0.8057
    ///      0.50                  0.4572          0.8379
    ///      1.00                  0.1569          0.8332
    /// ```
    ///
    /// Zero *is* the model alone -- this corpus gives the graph channel
    /// nothing to return, so zeroing the lexical pair leaves the vector
    /// channel by itself. So the eighth that ships costs 0.0258 of
    /// cross-lingual ranking to buy 0.0769 of same-language, where the quarter
    /// before it paid 0.0635 for 0.1270, and a constant has to pay that on
    /// every query including the ones where the lexical channels found nothing
    /// to contribute.
    ///
    /// The signal is how much the lexical channels agree with the vector
    /// channel about what is relevant: the share of the vector channel's
    /// candidates that either lexical channel also returned. Low agreement
    /// means the wording did not carry -- which is what a cross-language query
    /// looks like from the inside -- and the lexical pair is then voting on
    /// rank positions it has no evidence for.
    ///
    /// It is deliberately not a language test. Inferring the query's language
    /// from which language its lexical hits sit in was measured and cleared
    /// nothing, and the reason it failed is that a query's language cannot be
    /// read off the channel being corrected. Agreement is read off the
    /// *vector* channel, so it does not have that circularity, and it is the
    /// same judgement the reranker already makes when it confines itself to
    /// candidates no lexical channel found.
    fn lexical_agreement(lists: &[ChannelResults]) -> Option<f32> {
        let vector = lists.iter().find(|list| list.channel == Channel::Vector)?;
        if vector.candidates.is_empty() {
            return None;
        }
        let lexical: BTreeSet<TopicId> = lists
            .iter()
            .filter(|list| list.channel.is_lexical())
            .flat_map(|list| list.candidates.iter().copied())
            .collect();
        if lexical.is_empty() {
            return None;
        }
        let shared = vector
            .candidates
            .iter()
            .filter(|candidate| lexical.contains(candidate))
            .count();
        Some(shared as f32 / vector.candidates.len() as f32)
    }

    /// Fuses per-channel ranked lists into one ordered result set.
    pub fn fuse(&self, lists: &[ChannelResults]) -> Vec<FusedResult> {
        let mut accumulated: BTreeMap<TopicId, (f32, Vec<Why>)> = BTreeMap::new();

        // Scaled by how far the lexical channels agree with the vector channel
        // on this query, when nothing pinned them. Absent agreement -- no
        // vector candidates, or no lexical ones -- there is nothing to read,
        // so the configured weight stands.
        let agreement = self
            .adaptive
            .then(|| Self::lexical_agreement(lists))
            .flatten()
            .map(|agreement| {
                let span = self.agreement_ceiling - self.agreement_floor;
                self.agreement_floor + span * agreement
            });

        for list in lists {
            let weight = match agreement {
                Some(agreement) if list.channel.is_lexical() => {
                    self.weight(list.channel) * agreement
                }
                _ => self.weight(list.channel),
            };
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
            .map(|(topic, (score, why))| FusedResult { topic, score, why })
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
}

impl Default for Modifiers {
    fn default() -> Self {
        Self {
            importance_weight: 0.2,
            worth_weight: 0.2,
        }
    }
}

impl Modifiers {
    /// Applies every modifier to one result, appending a trace line for each.
    pub fn apply(&self, result: &mut FusedResult, signals: &RetrievalSignals) {
        let importance = 1.0 + self.importance_weight * signals.importance.clamp(0.0, 1.0);
        self.record(result, Modifier::Importance, importance);

        let worth = 1.0 + self.worth_weight * worth_ratio(signals);
        self.record(result, Modifier::Worth, worth);
    }

    /// Applies one modifier, and records it only if it changed anything.
    ///
    /// A factor of one moved no result past any other, so a trace line for it
    /// says only that the modifier exists. Two of the three are in that state
    /// permanently: `importance` and `worth_*` are read here and written by
    /// nothing, so every result carried `Importancex1.00 Worthx1.00` — and the
    /// trace is the product. Noise in it costs more than a missing line,
    /// because a reader who learns to skip `why[]` stops reading the part that
    /// does carry a reason.
    fn record(&self, result: &mut FusedResult, modifier: Modifier, factor: f32) {
        debug_assert!(
            !result.why.iter().any(|why| matches!(
                why,
                Why::Modifier { modifier: existing, .. } if *existing == modifier
            )),
            "modifier {modifier:?} applied twice to one result"
        );
        result.score *= factor;

        if (factor - 1.0).abs() > f32::EPSILON {
            result.why.push(Why::Modifier { modifier, factor });
        }
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
            .then_with(|| left.topic.0.cmp(&right.topic.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(byte: u8) -> TopicId {
        TopicId(uuid::Uuid::from_bytes([byte; 16]))
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
            fused[0].topic, both,
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

        let entry = fused.iter().find(|r| r.topic == target).unwrap();
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
        //
        // The vector and graph channels, because the two lexical ones
        // deliberately carry half weight. That says how much they duplicate
        // each other, not that a rank means something different in each.
        let fused = Fusion::default().fuse(&[
            ChannelResults::new(Channel::Graph, vec![id(1)]),
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
        assert_eq!(fused[0].topic, id(2));
    }

    /// A modifier that changed nothing is not worth a line in the trace.
    ///
    /// `importance` and `worth_*` are read by the ranker and written by no
    /// code at all, so with the signals every result actually carries today
    /// both come out at exactly one — and every explanation was two lines of
    /// `x1.00` before anything that moved the result. The trace is the product
    /// here, so padding it is not harmless: it teaches the reader to skip the
    /// part that does carry a reason.
    #[test]
    fn a_modifier_that_changed_nothing_leaves_no_trace() {
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);
        let ranked = fused.score;

        // What a state the ledger has never learned anything about looks like,
        // which today is every state.
        Modifiers::default().apply(&mut fused, &RetrievalSignals::default());

        let recorded: Vec<_> = fused
            .why
            .iter()
            .filter_map(|why| match why {
                Why::Modifier { modifier, factor } => Some((*modifier, *factor)),
                Why::Channel { .. } | Why::Path { .. } => None,
            })
            .collect();
        assert!(
            recorded.is_empty(),
            "nothing moved the result, so nothing should claim to have: {recorded:?}"
        );
        assert!(
            (fused.score - ranked).abs() < f32::EPSILON,
            "and the score is what the channels made it"
        );

        // A modifier that does move the result still says so.
        let mut moved = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);
        Modifiers::default().apply(
            &mut moved,
            &RetrievalSignals {
                importance: 1.0,
                ..RetrievalSignals::default()
            },
        );
        assert!(
            moved.why.iter().any(|why| matches!(
                why,
                Why::Modifier {
                    modifier: Modifier::Importance,
                    ..
                }
            )),
            "a result was lifted with nothing to show for it: {:?}",
            moved.why
        );
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
    fn a_state_with_no_recorded_outcomes_is_neither_promoted_nor_punished() {
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::new(Channel::Vector, vec![id(1)])])
            .remove(0);
        let original = fused.score;

        Modifiers::default().apply(&mut fused, &RetrievalSignals::default());

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
            asserted_from: "release_process".to_string(),
            asserted_to: "deployment_pipeline".to_string(),
            edge: EdgeKind::DependsOn,
            derivation: Derivation::Deterministic,
        });

        Modifiers::default().apply(&mut fused, &RetrievalSignals::default());

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

        let left: Vec<_> = tied.iter().map(|r| r.topic).collect();
        let right: Vec<_> = reversed.iter().map(|r| r.topic).collect();
        assert_eq!(left, right, "ordering must not depend on input order");
    }

    /// The weight the trace reports is the weight that was applied.
    fn lexical_weight_in(fused: &FusedResult) -> f32 {
        fused
            .why
            .iter()
            .find_map(|why| match why {
                Why::Channel {
                    channel, weight, ..
                } if channel.is_lexical() => Some(*weight),
                _ => None,
            })
            .expect("a lexical channel in the trace")
    }

    /// Two channels that name the same candidates are a channel worth
    /// listening to.
    #[test]
    fn full_agreement_leaves_the_lexical_pair_at_its_ceiling() {
        let shared = [id(1), id(2), id(3), id(4)];
        let fused = Fusion::default().with_adaptive(0.0, 1.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, shared.to_vec()),
            ChannelResults::new(Channel::Vector, shared.to_vec()),
        ]);

        let applied = lexical_weight_in(&fused[0]);
        assert!(
            (applied - LEXICAL_WEIGHT).abs() < f32::EPSILON,
            "full agreement should leave the ceiling alone, got {applied}"
        );
    }

    /// And a lexical pair that found none of what the vector channel found has
    /// no evidence to vote with. This is the cross-language case: the wording
    /// did not carry, so the words should not decide the ranking.
    #[test]
    fn no_agreement_takes_the_lexical_pair_out_of_the_vote() {
        let vector: Vec<_> = (1..=4).map(id).collect();
        let lexical: Vec<_> = (10..=13).map(id).collect();
        let fused = Fusion::default().with_adaptive(0.0, 1.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, lexical),
            ChannelResults::new(Channel::Vector, vector.clone()),
        ]);

        // The vector channel's own order survives intact, which is the point:
        // at zero agreement this is the embedding model's ranking.
        let kept: Vec<_> = fused
            .iter()
            .filter(|result| vector.contains(&result.topic))
            .map(|result| result.topic)
            .collect();
        assert_eq!(kept, vector, "the vector ranking should be left as it is");

        let lexical_first = fused
            .iter()
            .find(|result| !vector.contains(&result.topic))
            .expect("the lexical candidates are still returned");
        assert_eq!(
            lexical_weight_in(lexical_first),
            0.0,
            "a lexical channel that agrees with nothing votes with nothing"
        );
    }

    /// Half the vector channel's candidates found lexically is half the
    /// ceiling. Asserted because the shape of the mapping is a choice, and a
    /// change to it should be a change somebody made on purpose.
    #[test]
    fn partial_agreement_scales_the_lexical_pair() {
        let vector: Vec<_> = (1..=4).map(id).collect();
        let lexical = vec![id(1), id(2), id(20), id(21)];
        let fused = Fusion::default().with_adaptive(0.0, 1.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, lexical),
            ChannelResults::new(Channel::Vector, vector),
        ]);

        let applied = lexical_weight_in(&fused[0]);
        assert!(
            (applied - LEXICAL_WEIGHT * 0.5).abs() < 1e-6,
            "two of four shared should halve the ceiling, got {applied}"
        );
    }

    /// A pinned weight is a pinned weight. The sweeps name a constant and have
    /// to be measured on it, or the column they print is not the column they
    /// named.
    #[test]
    fn setting_a_lexical_weight_by_hand_turns_the_adjustment_off() {
        let vector: Vec<_> = (1..=4).map(id).collect();
        let lexical: Vec<_> = (10..=13).map(id).collect();
        let fused = Fusion::default()
            .with_weight(Channel::LexicalSegmented, 0.25)
            .with_weight(Channel::LexicalNgram, 0.25)
            .fuse(&[
                ChannelResults::new(Channel::LexicalSegmented, lexical),
                ChannelResults::new(Channel::Vector, vector),
            ]);

        let lexical_hit = fused
            .iter()
            .find(|result| result.topic == id(10))
            .expect("the lexical candidates are still returned");
        assert_eq!(
            lexical_weight_in(lexical_hit),
            0.25,
            "a pinned lexical weight still applies at zero agreement"
        );
    }

    /// Nothing to read the agreement from means the configured weight stands,
    /// rather than silently becoming zero.
    #[test]
    fn without_a_vector_channel_the_configured_weight_stands() {
        let fused = Fusion::default()
            .with_adaptive(0.0, 1.0)
            .fuse(&[ChannelResults::new(
                Channel::LexicalSegmented,
                vec![id(1), id(2)],
            )]);

        assert_eq!(lexical_weight_in(&fused[0]), LEXICAL_WEIGHT);
    }

    /// A floor keeps the lexical pair in the vote when agreement is low.
    ///
    /// The unbounded form took same-language ranking apart on XQuAD-R, so the
    /// mapping has a floor and a ceiling and both are swept. This pins the
    /// arithmetic, not the values.
    #[test]
    fn a_floor_keeps_the_lexical_pair_in_the_vote() {
        let vector: Vec<_> = (1..=4).map(id).collect();
        let lexical: Vec<_> = (10..=13).map(id).collect();
        let fused = Fusion::default().with_adaptive(0.5, 1.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, lexical),
            ChannelResults::new(Channel::Vector, vector),
        ]);

        let lexical_hit = fused
            .iter()
            .find(|result| result.topic == id(10))
            .expect("the lexical candidates are still returned");
        assert!(
            (lexical_weight_in(lexical_hit) - LEXICAL_WEIGHT * 0.5).abs() < 1e-6,
            "zero agreement should land on the floor, not on zero"
        );
    }

    /// And the constant is reachable through the same knob, so the sweep can
    /// include it without a second code path.
    #[test]
    fn a_floor_equal_to_the_ceiling_is_the_constant() {
        let vector: Vec<_> = (1..=4).map(id).collect();
        let lexical: Vec<_> = (10..=13).map(id).collect();
        let fused = Fusion::default().with_adaptive(1.0, 1.0).fuse(&[
            ChannelResults::new(Channel::LexicalSegmented, lexical),
            ChannelResults::new(Channel::Vector, vector),
        ]);

        let lexical_hit = fused
            .iter()
            .find(|result| result.topic == id(10))
            .expect("the lexical candidates are still returned");
        assert_eq!(lexical_weight_in(lexical_hit), LEXICAL_WEIGHT);
    }
}
