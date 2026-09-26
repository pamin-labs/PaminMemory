//! Rank fusion and the modifiers applied after it.
//!
//! Fusion happens here rather than inside a retrieval engine, and that is a
//! correctness requirement rather than a preference. The graph channel lives in
//! PostgreSQL, where the projection index cannot see it. An engine that pre-fused
//! its own channels would hand back a list that then had to be fused again with
//! the graph list, weighting the pre-fused members twice, and it would erase the
//! per-channel ranks that every result is required to be able to report.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::channel::{Channel, ChannelResults, Scored};
use crate::graph::{Derivation, EdgeKind};
use crate::id::TopicId;

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

/// What word-level BM25 is worth.
///
/// An eighth, and it was a quarter until a third corpus was measured. The
/// sweep that settled the quarter ran `1.00 / 0.50 / 0.25 / 0.00` -- so **a
/// quarter was the smallest non-zero value ever tried**, and the argument for
/// it was against zero rather than against half of itself. Half of itself is
/// better on every group of every corpus but one; see [`Fusion::default`].
const SEGMENTED_WEIGHT: f32 = 0.125;

/// What character-n-gram BM25 is worth.
///
/// The same eighth, and that is the part with no evidence under it. Every
/// sweep this project has run moved both lexical channels together, so no
/// measurement anywhere distinguishes these two numbers -- the grid was
/// one-dimensional and the conclusion is being read as though it were two.
///
/// Two constants rather than one because the premise that justified sharing is
/// refuted; see [`Fusion::default`]. Splitting them changes nothing on its own
/// and is not meant to: it makes the second number nameable, and therefore
/// sweepable, which it was not.
const NGRAM_WEIGHT: f32 = 0.125;

/// What the graph channel's rank is worth against the vector channel's.
///
/// **Measured, and the previous value was not.** Every unnamed channel
/// defaults to 1.0, and the graph channel was simply never named -- so it
/// voted as loudly as the dense channel on the strength of a hop. Nothing
/// could see that, because all three evaluation corpora derived zero edges and
/// a channel with no candidates has no weight worth arguing about.
///
/// The `relational` group in `pamin-engine/tests/corpus` ends that: ten pairs
/// of memories whose answering half is named by a phrase the other half's
/// prose contains, so mention derivation fires and the channel has something
/// to walk. With eleven live edges in that project, fusion alone, nDCG@10:
///
///   weight   cross-lingual   lexical   monolingual   relational
///     0.00          0.7903    1.0000        0.9940       0.5237
///     0.15          0.7853    1.0000        0.9940       0.5517
///     0.30          0.7746    1.0000        0.9940       0.6295
///     0.50          0.7415    1.0000        0.9821       0.6583
///     1.00          0.5109    0.9885        0.9246       0.6910
///
/// At 1.0 the channel costs **0.2794** on the cross-lingual group -- forty
/// wins to nothing for removing it, `p = 0.0001` -- and 0.0694 on the
/// monolingual group, against 0.1673 earned on the twenty queries written to
/// favour it. It is net negative even counting the group built for it, and the
/// whole search path at 1.0 fails this repository's own `monolingual` floor:
/// 0.9246 against 0.9400.
///
/// **Why three tenths, read the way the evidence can support.** An earlier
/// version of this note called 0.30 "the knee" of that table, and the knee was
/// read off increments of 0.03 to 0.08 on a twenty-query group whose smallest
/// detectable difference, from its own paired differences, is about 0.08. It
/// was steering by noise. Priced again as one family -- Westfall-Young across
/// every row of the sweep, against 0.30 -- the table says three things and no
/// more:
///
/// - **1.0 is wrong.** Cross-lingual -0.2140 and monolingual -0.0694 against
///   0.30, family-adjusted `p = 0.0001` and `0.0003`; its relational advantage
///   is not detectable, `p = 0.18`.
/// - **0.5 is worse on cross-lingual**, -0.0305 at family `p = 0.0027`, and its
///   relational gain is below what twenty queries can see.
/// - **0.15 and 0.30 cannot be told apart**, and zero trades a cross-lingual
///   gain of 0.0157 (`p = 0.043`) for a relational loss of 0.1058 (`p = 0.007`).
///
/// So the choice is anywhere in `[0.15, 0.30]`, and 0.30 is kept because the
/// sweep has no evidence to move it -- which is the honest description of most
/// constants in this file. Cross-validated over five folds, choosing from the
/// whole sweep by the mean over groups picked no other graph weight in any
/// fold.
///
/// The number is not read off the four-group mean, and the reason is a bias
/// this weight would otherwise be chosen by: the relational group is twenty
/// queries written in this repository *to make the graph channel look useful*,
/// and the other 137 were written before that intent existed.
///
/// The only comparable published system (`arXiv:2609.01617`) weights its graph
/// channel at 0.15 against a dense 0.50, which is the same order and was the
/// prediction written before this sweep ran.
const GRAPH_WEIGHT: f32 = 0.30;

/// One line of the explanation attached to a result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Why {
    /// The result appeared in this channel at this rank.
    Channel {
        channel: Channel,
        rank: u32,
        /// What this channel scored this candidate, in this channel's own
        /// units.
        ///
        /// Never comparable across channels -- a BM25 score and a vector
        /// similarity are different quantities, which is why the entry above it
        /// is a rank and why fusion combines ranks. It is here because the one
        /// question a rank cannot answer is how sure the channel was: a
        /// candidate at rank 1 looks identical whether its channel put it a
        /// long way clear of the field or could barely separate it from rank
        /// 50. Reading that off requires the channel's own scores, so they are
        /// written down.
        ///
        /// `None` from a channel that does not score against the query. The
        /// graph channel reaches a topic across edges instead.
        score: Option<f32>,
        weight: f32,
        contribution: f32,
    },
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
    /// A reranker scored this result.
    ///
    /// Present only on the candidates that reached the model, which is the
    /// part of a search's explanation that was missing. The accurate tier
    /// blends this score with fusion, while fast orders only its selected
    /// candidates. No entry means the model did not score the result.
    ///
    /// **Never comparable across queries**, for the same reason the channel
    /// score above is not: a cross-encoder's logit is calibrated against
    /// nothing. It separates the candidates of one shortlist. A cut expressed
    /// relative to the other candidates of the same query can be read off
    /// this; a fixed absolute threshold cannot, and would behave differently
    /// on every query.
    Reranked {
        /// The model's own score, larger being more relevant.
        score: f32,
    },
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

/// How a candidate's places across the channels become one number.
///
/// This project argued at length about `k` and about the channel weights --
/// both of them parameters *of* reciprocal rank fusion -- and never recorded
/// that fusing ranks rather than normalised scores was a choice at all. It is
/// the load-bearing one, and it is the one that was never tested.
///
/// What the 2025--2026 work says about it, all of it against rank fusion:
///
/// - Training-free lexical-dense fusion (`arXiv:2606.04194`, 2026) is the one
///   result measured on this project's own benchmarks, on CPU, without
///   training: **z-score weighted fusion at Hit@1 0.752 against RRF's 0.718**
///   on LoCoMo, with a wide plateau in the mixing weight.
/// - The systematic comparison (`arXiv:2507.03761`, 2025), ten fusion
///   algorithms against six normalisers over four corpora, puts **standardised
///   scores with CombMNZ highest on all four**, and every rank-based and
///   vote-based method below every score-based one.
/// - The calibrated graph-vector fusion paper (`arXiv:2603.28886`, 2026)
///   reports RRF's mean gain as statistically insignificant where its
///   calibrated fusion's smaller mean gain is significant, and its ablation
///   names normalisation as the dominant factor.
///
/// So the alternatives were implemented and measured here rather than argued
/// about, and the sweep has run: [`Banded`](Self::Banded) is what ships, and
/// [`Reciprocal`](Self::Reciprocal) stays, reachable through [`Fusion::with`],
/// as the baseline every figure is quoted against. The score combiners the
/// literature favours -- a weighted sum of standardised scores, CombMNZ,
/// TM2C2 -- and the band read on theoretical min-max all measured worse and
/// were removed. What each was worth, and why it lost, is recorded under
/// "Fusion designs measured and removed" in `docs/adr/0001-tech-selection.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    /// `sum over channels of weight / (k + rank)`.
    ///
    /// The rank is all it reads, which is the property that makes it robust to
    /// incomparable channels and the property that makes it unable to tell a
    /// channel that is certain from one that is guessing.
    Reciprocal,
    /// Each channel's scores, rescaled into the band reciprocal rank fusion
    /// would have spanned over the same candidates.
    ///
    /// **This isolates the one variable the score combiners confounded.** A
    /// weighted sum of standardised scores lost recall -- XQuAD-R cross-lingual
    /// `recall@50` 0.8960 to 0.7765, recorded with the other removed designs in
    /// `docs/adr/0001-tech-selection.md` -- and the reason is not centring as
    /// such: it is the *width* of the range a channel's contributions span. Over
    /// fifty candidates at `k = 10`, reciprocal rank fusion spans `1/11` down
    /// to `1/60`: a factor of 5.45, narrow enough that a channel's **weight**
    /// decides against another channel's position. A lexical channel's top hit
    /// at an eighth weight scores 0.0114 and the vector channel's fiftieth
    /// scores 0.0167, so the good channel's marginal candidate still wins, and
    /// that is what keeps the tail of the list full of things worth reranking.
    ///
    /// Any normaliser onto `[0, 1]` spans a factor of infinity inside one
    /// channel, so position beats weight and a worthless channel's confident
    /// hit displaces a good channel's deep one. Centring makes it worse by
    /// adding a sign, but min-max would do the same thing, which is why it is
    /// not a variant here.
    ///
    /// So: min-max the scores, then map them onto `[(k + 1) / (k + n), 1]`,
    /// where `n` is how many candidates the channel returned. The band is
    /// derived rather than tuned -- it is exactly the range reciprocal rank
    /// fusion would have used -- and what changes is only whether a channel's
    /// candidates are ordered inside it by their score or by their rank. That
    /// is the question this whole line of work has been trying to ask, and
    /// nothing before this asked it without also changing the band.
    ///
    /// **Measured, and it is what ships.** Against reciprocal rank fusion,
    /// nDCG@10 and `recall@50`, over four groups of three corpora:
    ///
    /// ```text
    ///   group                     nDCG        p        recall
    ///   XQuAD-R cross-lingual   +0.0037   0.0003   0.8960 -> 0.8962
    ///   XQuAD-R same-language   +0.0273   0.0001   0.9580 -> 0.9571
    ///   MIRACL Swahili          -0.0003   0.9210   0.9314 -> 0.9309
    ///   this project, cross     +0.0075   0.3731   0.9605 -> 0.9605
    /// ```
    ///
    /// Two groups significantly better, none significantly worse, and recall
    /// moves by at most 0.0009 anywhere -- where the standardised sum cost
    /// 0.1195 of it on the first row. On the shipped path, reranker included,
    /// XQuAD-R goes from 0.6480 to 0.6511 cross-lingual and from 0.7495 to
    /// **0.7769** same-language.
    ///
    /// The same-language gain is the part worth noticing. Every weight this
    /// project ever changed took something from that group to pay for the
    /// cross-lingual one; this is the first change that improves it, and it
    /// does so by letting the channel that is genuinely good there --
    /// segmented BM25, 0.7299 alone against the vector channel's 0.6787 -- say
    /// *how much* better its top candidates are rather than only that they
    /// came first.
    Banded,
}

/// Fuses ranked lists into one ordered result set.
#[derive(Clone, Debug)]
pub struct Fusion {
    combine: Combine,
    k: f32,
    weights: BTreeMap<Channel, f32>,
}

impl Default for Fusion {
    fn default() -> Self {
        // The two lexical channels count as half of one, and the size of that
        // half is measured. What is *not* measured is the decision to give them
        // the same number, and the argument that used to justify it has been
        // refuted by this project's own diagnostic.
        //
        // The claim was that they are nearly one channel: both run BM25 over
        // the same text, one over segmented words and one over character
        // n-grams, so they were said to agree with each other far more often
        // than either agrees with the vector or the graph. Kendall tau-b over
        // the candidates they share says 0.2816 on this project's own corpus,
        // 0.3188 on XQuAD-R and 0.2973 on MIRACL. Three corpora, one answer:
        // they are two channels that agree about a third of the time. The thing
        // that is true is the consequence of their sharing a *field*, not a
        // ranking -- at full weight the pair still outvotes the other two on
        // every query where the wording matches and the meaning does not, which
        // is what the sweep below measures.
        //
        // Every row of that sweep moved both channels together, so it cannot
        // separate them, and the n-gram channel is the weaker of the two in
        // every group of every corpus measured alone. One number is doing the
        // work of two and nothing has ever been asked which.
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
            combine: Combine::Banded,
            k: DEFAULT_K,
            weights: BTreeMap::from([
                (Channel::LexicalSegmented, SEGMENTED_WEIGHT),
                (Channel::LexicalNgram, NGRAM_WEIGHT),
                (Channel::Graph, GRAPH_WEIGHT),
            ]),
        }
    }
}

impl Fusion {
    /// Overrides the weight of one channel.
    pub fn with_weight(mut self, channel: Channel, weight: f32) -> Self {
        self.weights.insert(channel, weight);
        self
    }

    /// Stops asking this channel entirely.
    ///
    /// The same thing as a weight of zero, and that is the point: a channel
    /// worth nothing should not be able to put a candidate into the result set.
    /// It used to. `fuse` created its entry for every candidate of every
    /// channel before applying any weight, so the fused set was always the
    /// union of all four, and a candidate only a zero-weighted channel proposed
    /// arrived at score 0.0 ordered against the other zeroes by topic
    /// identifier. The head of the list stayed clean, because any positive
    /// score beats zero -- but anything reading past the head, `recall@50`
    /// above all, counted candidates no surviving channel had proposed, ranked
    /// by nothing but the accident of a UUID.
    pub fn without(self, channel: Channel) -> Self {
        self.with_weight(channel, 0.0)
    }

    /// Combines the channels this way instead of the default.
    ///
    /// See [`Combine`] for what the choices are and what the literature says
    /// about them. [`Combine::Banded`] ships; [`Combine::Reciprocal`] is kept
    /// so a change to the default is a comparison against the baseline rather
    /// than a claim.
    pub fn with(mut self, combine: Combine) -> Self {
        self.combine = combine;
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

    /// What one candidate is worth to the sum, before its channel's weight.
    ///
    /// `Reciprocal` reads the rank and nothing else. `Banded` reads `within`,
    /// where this candidate's score falls on its channel's scale.
    ///
    /// A channel with no place to read -- one that returned no scores at all,
    /// or an uncalibrated one whose candidates all scored the same -- falls
    /// back to `1 / (k + rank)`, which is the band's own shape read by rank
    /// rather than a third shape invented for the case. Nothing the engine
    /// assembles returns no scores -- all four channels score now -- but a
    /// caller can build one and it should not silently mean zero.
    fn share(&self, scaled: Scaled, rank: u32) -> f32 {
        let reciprocal = 1.0 / (self.k + rank as f32);
        match self.combine {
            Combine::Reciprocal => reciprocal,
            Combine::Banded => match scaled.within {
                // The band reciprocal rank fusion would have spanned over the
                // same candidates, so the only thing that changed is whether
                // position inside it comes from the score or from the rank.
                Some(within) => {
                    let floor = (self.k + 1.0) / (self.k + scaled.of as f32);
                    (floor + (1.0 - floor) * within) / (self.k + 1.0)
                }
                None => reciprocal,
            },
        }
    }

    /// Fuses per-channel ranked lists into one ordered result set.
    pub fn fuse(&self, lists: &[ChannelResults]) -> Vec<FusedResult> {
        let mut accumulated: BTreeMap<TopicId, (f32, Vec<Why>)> = BTreeMap::new();

        for list in lists {
            let weight = self.weight(list.channel);

            // A channel worth nothing does not get to name candidates. See
            // `without`, which is the same statement made deliberately.
            if weight == 0.0 {
                continue;
            }

            // Once per channel, not once per candidate: these are the
            // channel's own minimum and maximum, so recomputing them inside
            // the loop would be the same numbers at fifty times the cost.
            let scaled = rescale(&list.candidates, list.channel.calibrated());

            for (index, candidate) in list.candidates.iter().enumerate() {
                let rank = index as u32 + 1;
                let contribution = weight * self.share(scaled[index], rank);
                let entry = accumulated
                    .entry(candidate.topic)
                    .or_insert((0.0, Vec::new()));
                entry.0 += contribution;
                entry.1.push(Why::Channel {
                    channel: list.channel,
                    rank,
                    score: candidate.score,
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

/// One candidate's score, placed on its own channel's scale.
///
/// `within` is `None` when there is nothing to compute: a channel that does
/// not score, or an uncalibrated one whose candidates all scored the same,
/// where the divisor is zero. `Fusion::share` decides what to do with that;
/// this only reports that there is nothing to report.
#[derive(Clone, Copy, Debug, Default)]
struct Scaled {
    /// Where it falls between the channel's worst and best candidate, on
    /// `[0, 1]` -- or on the channel's declared range, if it has one.
    within: Option<f32>,
    /// How many candidates the channel returned, which is what sets the width
    /// of [`Combine::Banded`]'s band.
    of: usize,
}

/// Every candidate's score expressed against its own channel's distribution.
///
/// Positionally aligned with the candidates. Computed once per channel because
/// every value in it is a property of the channel rather than of a candidate.
fn rescale(candidates: &[Scored], calibrated: Option<(f32, f32)>) -> Vec<Scaled> {
    let blank = Scaled {
        of: candidates.len(),
        ..Scaled::default()
    };

    let scores: Vec<f32> = candidates
        .iter()
        .filter_map(|candidate| candidate.score)
        .collect();
    if scores.len() != candidates.len() || scores.is_empty() {
        return vec![blank; candidates.len()];
    }

    let least = scores.iter().copied().fold(f32::MAX, f32::min);
    let most = scores.iter().copied().fold(f32::MIN, f32::max);

    // `within` asks where a candidate falls on the scale its score is on, and
    // for a channel that declares a calibrated range there is an answer even
    // when every candidate scored the same: fifty candidates all at 0.5 are
    // all at the middle of `[0, 1]`, which is the whole point of a score that
    // means the same thing on every query. Only an empirical min-max has
    // nothing to divide by.
    let scale = match calibrated {
        Some((low, high)) if high - low > f32::EPSILON => Some((low, high - low)),
        Some(_) => None,
        None => (most - least > f32::EPSILON).then_some((least, most - least)),
    };

    scores
        .into_iter()
        .map(|score| Scaled {
            within: scale.map(|(low, width)| ((score - low) / width).clamp(0.0, 1.0)),
            of: candidates.len(),
        })
        .collect()
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
            ChannelResults::unscored(Channel::LexicalSegmented, vec![single, both]),
            ChannelResults::unscored(Channel::Vector, vec![both]),
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
            ChannelResults::unscored(Channel::LexicalNgram, vec![id(9), target]),
            ChannelResults::unscored(Channel::Vector, vec![target]),
        ]);

        let entry = fused.iter().find(|r| r.topic == target).unwrap();
        let ranks: Vec<_> = entry
            .why
            .iter()
            .filter_map(|why| match why {
                Why::Channel { channel, rank, .. } => Some((*channel, *rank)),
                // `fuse` writes neither: a path comes from the graph channel's
                // own evidence and a rerank entry is added a stage later. Both
                // are matched rather than caught by a wildcard, so adding a
                // variant makes the compiler ask about this test.
                Why::Path { .. } | Why::Reranked { .. } => None,
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
        // Both weights set here rather than inherited, because every channel
        // but the vector one now carries a measured weight -- the two lexical
        // ones an eighth each and the graph channel three tenths. Those say
        // what a channel is worth, not that a rank means something different
        // inside it, and this test is about the second.
        let fused = Fusion::default()
            .with_weight(Channel::Graph, 1.0)
            .with_weight(Channel::Vector, 1.0)
            .fuse(&[
                ChannelResults::unscored(Channel::Graph, vec![id(1)]),
                ChannelResults::unscored(Channel::Vector, vec![id(2)]),
            ]);
        assert!((fused[0].score - fused[1].score).abs() < f32::EPSILON);
    }

    /// A calibrated channel votes as loudly as its evidence, not as loudly as
    /// its best candidate.
    ///
    /// The defect this fixes, stated in the numbers it changes. The graph
    /// channel's score is `confidence * decay^(hops - 1)` over a schema-bounded
    /// `(0, 1]`, so 0.5 means "one derived mention" on every query. Min-maxed
    /// inside a query, whatever the best path happened to be became the top of
    /// the band -- so a channel whose only path was a single weak guess
    /// contributed exactly what a channel that found an explicit assertion
    /// contributed, and a channel with nothing good to say could not say so.
    #[test]
    fn a_calibrated_channel_cannot_promote_a_weak_path_to_the_top_of_the_band() {
        let contribution = |scores: &[f32]| -> f32 {
            let lists = [ChannelResults::new(
                Channel::Graph,
                scores
                    .iter()
                    .enumerate()
                    .map(|(n, score)| Scored::new(id(n as u8 + 1), *score))
                    .collect(),
            )];
            // Full weight, so what this reads is the declared scale alone
            // and not the three tenths the channel is worth against the
            // others.
            Fusion::default()
                .with_weight(Channel::Graph, 1.0)
                .fuse(&lists)[0]
                .score
        };

        // Three explicit one-hop assertions against three weak derived ones.
        let strong = contribution(&[1.0, 0.9, 0.8]);
        let weak = contribution(&[0.1, 0.09, 0.08]);
        assert!(
            strong > weak,
            "a channel holding explicit assertions should outvote one holding \
             guesses: {strong} against {weak}"
        );

        // And the arithmetic is the declared scale rather than the query's own
        // extremes. Over three candidates at `k = 10` the band runs from
        // `(11/13) / 11` to `1 / 11`; the weak list's best candidate is a tenth
        // of the way up `[0, 1]`, so it lands at `(11/13 + 0.154 * 0.1) / 11`
        // and not at the top. Min-maxed it would have been the top exactly,
        // because min-maxing maps a list's best candidate to one whatever it
        // scored -- which is the whole defect.
        let top = 1.0 / (DEFAULT_K + 1.0);
        assert!(
            weak < top,
            "the weak list's best path still took the top of the band: \
             {weak} against {top}"
        );
        assert!(
            (strong - top).abs() < 1e-6,
            "and the strong list's best path, at 1.0 of its declared range, \
             should be the top: {strong} against {top}"
        );
    }

    /// A calibrated channel whose candidates all score the same still says
    /// where that is, rather than falling back to the order it was handed.
    ///
    /// The second half of the same defect, and the one that was live in
    /// production. Every edge the write path derives is a `Mentions` at the
    /// same confidence, and the walk stops at one hop whenever one hop fills
    /// the channel's depth -- so every candidate scored identically, the
    /// empirical minimum equalled the maximum, and fusion fell back to rank.
    /// The rank came from a sort whose only live tie-break, once distance and
    /// confidence were constant, was the topic's identifier: the head of a
    /// search was being ordered by UUID, at full weight.
    ///
    /// Now they tie, which is what identical evidence should produce, and the
    /// identifier decides only the order among equals -- which is what
    /// `sort_results`' tie-break is for and is documented as.
    #[test]
    fn a_calibrated_channel_with_nothing_to_separate_does_not_fall_back_to_rank() {
        let identical: Vec<Scored> = (1..=6).map(|n| Scored::new(id(n), 0.5)).collect();
        // Full weight throughout, so this reads the declared scale rather
        // than what the channel is worth against the others.
        let graph = || Fusion::default().with_weight(Channel::Graph, 1.0);
        let fused = graph().fuse(&[ChannelResults::new(Channel::Graph, identical.clone())]);

        let first = fused[0].score;
        assert!(
            fused.iter().all(|result| result.score == first),
            "identical evidence produced different contributions: {fused:?}"
        );

        // Where it ties is the middle of the band, because 0.5 is the middle
        // of the declared range -- not its top and not its bottom.
        let strong = graph().fuse(&[ChannelResults::new(
            Channel::Graph,
            (1..=6).map(|n| Scored::new(id(n), 1.0)).collect(),
        )])[0]
            .score;
        let faint = graph().fuse(&[ChannelResults::new(
            Channel::Graph,
            (1..=6).map(|n| Scored::new(id(n), 0.01)).collect(),
        )])[0]
            .score;
        assert!(
            faint < first && first < strong,
            "the middle of the range should sit between its ends: \
             {faint} < {first} < {strong}"
        );

        // An uncalibrated channel keeps the old behaviour, so this is a
        // property of the declaration and not a change to fusion itself.
        let lexical = Fusion::default()
            .with_weight(Channel::LexicalSegmented, 1.0)
            .fuse(&[ChannelResults::new(Channel::LexicalSegmented, identical)]);
        assert!(
            lexical[0].score > lexical[5].score,
            "an uncalibrated channel with nothing to separate still falls back \
             to rank, which is what it has: {lexical:?}"
        );
    }

    /// The banded combiner spans exactly what reciprocal rank fusion spans.
    ///
    /// The premise the variant exists to test. If the band were wider than
    /// rank fusion's, the comparison between them would be measuring the width
    /// as well as the ordering, which is the confound that made the
    /// standardised sum's recall loss hard to attribute.
    #[test]
    fn the_banded_combiner_spans_the_same_range_rank_fusion_does() {
        let deep: Vec<Scored> = (1..=50)
            .map(|n| Scored::new(id(n), (51 - n) as f32))
            .collect();
        let lists = [ChannelResults::new(Channel::Vector, deep)];

        let extremes = |fusion: &Fusion| -> (f32, f32) {
            let fused = fusion.fuse(&lists);
            let scores: Vec<f32> = fused.iter().map(|result| result.score).collect();
            (
                scores.iter().copied().fold(f32::MAX, f32::min),
                scores.iter().copied().fold(f32::MIN, f32::max),
            )
        };

        let (rank_low, rank_high) = extremes(&Fusion::default().with(Combine::Reciprocal));
        let (band_low, band_high) = extremes(&Fusion::default().with(Combine::Banded));

        assert!(
            (rank_low - band_low).abs() < 1e-6 && (rank_high - band_high).abs() < 1e-6,
            "rank fusion spans {rank_low}..{rank_high} and the band spans \
             {band_low}..{band_high}; the two have to span the same range or a \
             comparison between them is measuring the range"
        );
    }

    /// The banded combiner ships: two groups better, none worse, recall held.
    #[test]
    fn the_banded_combiner_is_what_ships() {
        assert_eq!(Fusion::default().combine, Combine::Banded);
    }

    /// Four channels as the engine hands them over: two unbounded BM25 lists,
    /// a cosine list, and a calibrated graph list with a tie and a candidate
    /// nothing else found.
    fn every_kind_of_channel() -> [ChannelResults; 4] {
        let scored = |channel, pairs: &[(u8, f32)]| {
            ChannelResults::new(
                channel,
                pairs
                    .iter()
                    .map(|(n, score)| Scored::new(id(*n), *score))
                    .collect(),
            )
        };
        [
            scored(
                Channel::Vector,
                &[
                    (1, 0.863),
                    (2, 0.826),
                    (3, 0.789),
                    (4, 0.752),
                    (5, 0.715),
                    (6, 0.678),
                    (7, 0.641),
                    (8, 0.604),
                ],
            ),
            scored(
                Channel::LexicalSegmented,
                &[(3, 14.2), (9, 11.7), (1, 6.1), (10, 5.9), (5, 2.3)],
            ),
            scored(
                Channel::LexicalNgram,
                &[(9, 31.0), (11, 30.5), (2, 12.25), (6, 4.0)],
            ),
            scored(Channel::Graph, &[(4, 0.5), (12, 0.5), (7, 0.25)]),
        ]
    }

    /// The same shape with no scores, where both combiners read rank alone.
    fn every_channel_unscored() -> [ChannelResults; 4] {
        [
            ChannelResults::unscored(Channel::Vector, (1..=6).map(id).collect()),
            ChannelResults::unscored(Channel::LexicalSegmented, vec![id(3), id(9), id(1)]),
            ChannelResults::unscored(Channel::LexicalNgram, vec![id(9), id(11), id(2)]),
            ChannelResults::unscored(Channel::Graph, vec![id(4), id(12)]),
        ]
    }

    fn fingerprint(fused: &[FusedResult]) -> Vec<(u8, u32)> {
        fused
            .iter()
            .map(|result| (result.topic.0.as_bytes()[0], result.score.to_bits()))
            .collect()
    }

    /// What ships and the baseline every figure is quoted against, to the bit.
    ///
    /// Every accuracy floor in this repository was taken under these two, so a
    /// change that moves either one by a single ulp -- on the order or the
    /// score -- is a change to what those floors describe and has to be made
    /// on purpose. Recorded from the code before the
    /// combiners that measured worse were removed around it.
    #[test]
    fn the_shipped_and_baseline_combiners_are_pinned_to_the_bit() {
        let banded = Fusion::default();
        let reciprocal = Fusion::default().with(Combine::Reciprocal);

        let scored = every_kind_of_channel();
        assert_eq!(
            fingerprint(&banded.fuse(&scored)),
            [
                (4, 0x3dceb5a6),
                (1, 0x3dcd3af2),
                (2, 0x3dc3a5dd),
                (3, 0x3dbcc486),
                (7, 0x3dad87f0),
                (5, 0x3da1dff0),
                (6, 0x3d98c018),
                (8, 0x3d638e39),
                (12, 0x3cce3b70),
                (9, 0x3cb4f776),
                (11, 0x3c397169),
                (10, 0x3c178d94),
            ]
        );
        assert_eq!(
            fingerprint(&reciprocal.fuse(&scored)),
            [
                (1, 0x3dcddfc7),
                (4, 0x3dca23e9),
                (2, 0x3dbe5be6),
                (3, 0x3db4cfaa),
                (7, 0x3da7bb6d),
                (5, 0x3d99999a),
                (6, 0x3d924925),
                (8, 0x3d638e39),
                (12, 0x3cccccce),
                (9, 0x3cb26c9c),
                (11, 0x3c2aaaab),
                (10, 0x3c124925),
            ]
        );

        // With nothing to read but rank, the band has nothing to order by and
        // the two are the same function.
        let unscored = every_channel_unscored();
        let rank_only = [
            (1, 0x3dcddfc7),
            (4, 0x3dca23e9),
            (2, 0x3dbe5be6),
            (3, 0x3db4cfaa),
            (5, 0x3d888889),
            (6, 0x3d800000),
            (12, 0x3cccccce),
            (9, 0x3cb26c9c),
            (11, 0x3c2aaaab),
        ];
        assert_eq!(fingerprint(&banded.fuse(&unscored)), rank_only);
        assert_eq!(fingerprint(&reciprocal.fuse(&unscored)), rank_only);
    }

    /// A channel worth nothing cannot put anything into the result set.
    ///
    /// The behaviour this replaced was invisible at the head and wrong
    /// everywhere else. `fuse` built its entry before applying any weight, so
    /// the fused set was the union of every channel asked, and a candidate only
    /// the zero-weighted channel proposed scored 0.0 and sorted among the other
    /// zeroes by topic identifier. Nothing reading the top ten would notice,
    /// because a positive score always beats zero. `recall@50` counted it.
    #[test]
    fn a_channel_worth_nothing_contributes_nothing_to_the_result_set() {
        let theirs = id(1);
        let ours = id(2);

        let fused = Fusion::default().without(Channel::LexicalSegmented).fuse(&[
            ChannelResults::unscored(Channel::LexicalSegmented, vec![theirs]),
            ChannelResults::unscored(Channel::Vector, vec![ours]),
        ]);

        assert_eq!(
            fused.iter().map(|result| result.topic).collect::<Vec<_>>(),
            vec![ours],
            "a candidate only the silenced channel proposed is still in the results"
        );
        assert!(
            fused[0].why.iter().all(
                |why| !matches!(why, Why::Channel { channel, .. } if *channel
                    == Channel::LexicalSegmented)
            ),
            "a silenced channel still wrote itself into the trace: {:?}",
            fused[0].why
        );
    }

    #[test]
    fn channel_weights_shift_the_balance() {
        let fused = Fusion::default().with_weight(Channel::Vector, 2.0).fuse(&[
            ChannelResults::unscored(Channel::LexicalSegmented, vec![id(1)]),
            ChannelResults::unscored(Channel::Vector, vec![id(2)]),
        ]);
        assert_eq!(fused[0].topic, id(2));
    }

    #[test]
    fn path_evidence_is_not_a_channel_entry() {
        // The channel entry says how highly the graph ranked this result; the
        // path says why the graph could see it at all. Counting a path as a
        // channel would inflate the rank tally the trace is read for.
        let target = id(1);
        let mut fused = Fusion::default()
            .fuse(&[ChannelResults::unscored(Channel::Graph, vec![target])])
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
        let lists = [ChannelResults::unscored(
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
}
