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
    /// A reranker scored this result and put it where it is.
    ///
    /// Present only on the candidates that reached the model, which is the
    /// part of a search's explanation that was missing: every channel wrote
    /// down what it scored a candidate, and the one score that decided the
    /// final order was computed and thrown away. A result with no entry of
    /// this kind was not reranked -- either a lexical channel found it, or it
    /// sat below the tier's depth -- and that distinction is readable from the
    /// absence.
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
/// So the alternatives are implemented and measured here rather than argued
/// about. [`Fusion::with`] selects one, and the sweep has since run:
/// [`Banded`](Self::Banded) is what ships, and what each of the others is
/// worth against it is recorded on its own variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Combine {
    /// `sum over channels of weight / (k + rank)`.
    ///
    /// The rank is all it reads, which is the property that makes it robust to
    /// incomparable channels and the property that makes it unable to tell a
    /// channel that is certain from one that is guessing.
    #[default]
    Reciprocal,
    /// `sum over channels of weight * z(score)`, standardised per channel.
    ///
    /// Each channel's scores are centred and divided by their own deviation, so
    /// a BM25 score and a cosine similarity become the same kind of quantity
    /// before they are added. A candidate a channel never returned contributes
    /// nothing, which under standardisation is that channel's own mean -- the
    /// neutral value, not a penalty.
    ///
    /// What this fixes relative to `Reciprocal` is the *magnitude*: a candidate
    /// that a channel scored far above its field now contributes more than one
    /// that merely came first in a flat list. What it does **not** fix is
    /// per-channel quality, for the reason
    /// [`Fusion::with_confidence`] gives -- standardising removes the units and
    /// not the quality, so a worthless channel's leader is still about `+2`.
    /// The two are separate mechanisms and compose.
    ///
    /// **Measured, and it does not ship, for a reason nDCG cannot show.** On
    /// nDCG@10 it is better in three of four groups and significantly so --
    /// +0.0281 on this project's own cross-lingual group (15 wins, 1 loss,
    /// p = 0.0037), +0.0134 on XQuAD-R's (448 / 120, p = 0.0001), +0.0177 on
    /// XQuAD-R's same-language group (142 / 106, p = 0.0001) -- and not
    /// significantly worse anywhere. It was made the default on that reading
    /// and the accuracy gates rejected it: XQuAD-R's cross-lingual `recall@50`
    /// fell from 0.8960 to **0.7765**, through a floor of 0.8000.
    ///
    /// The mechanism is the sign. Every reciprocal-rank contribution is
    /// positive, so a candidate one channel ranked fiftieth still helps it
    /// stay in the list. A standardised score is centred, so a candidate below
    /// its channel's own mean contributes a *negative* number -- and on a
    /// cross-lingual query, where a lexical channel scores 0.0366 alone, that
    /// channel's confident top hit at `+2` outranks a genuine deep hit from
    /// the vector channel at `-1`. The head improves because strong vector
    /// hits dominate the top ten; the tail fills with lexical noise and the
    /// relevant sentences that used to sit at ranks ten to fifty fall past
    /// fifty.
    ///
    /// So this is a precision-for-recall trade rather than an improvement, and
    /// a search that hands its results to a reranker cannot afford it: nothing
    /// recovers a memory that was never returned. What would make it shippable
    /// is a floor under each contribution, or normalising to `[0, 1]` rather
    /// than centring -- neither of which is what the published work measured,
    /// so neither is in here yet.
    Standardised,
    /// Each channel's scores, rescaled into the band reciprocal rank fusion
    /// would have spanned over the same candidates.
    ///
    /// **This isolates the one variable the measurements above confound.**
    /// `Standardised` loses recall, and the reason is not centring as such --
    /// it is the *width* of the range a channel's contributions span. Over
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
    /// `Standardised`, multiplied by how many channels returned the candidate.
    ///
    /// CombMNZ. In the systematic comparison above it is the best of ten
    /// combiners on all four corpora, and on this project's corpora it should
    /// be the worst of the three here -- it multiplies by agreement, and
    /// agreement between a confident channel and a worthless one is exactly
    /// the weakest-link failure measured on both cross-lingual groups. It is
    /// implemented so that prediction can be checked rather than asserted.
    StandardisedTimesVotes,
}

/// Fuses ranked lists into one ordered result set.
#[derive(Clone, Debug)]
pub struct Fusion {
    combine: Combine,
    k: f32,
    weights: BTreeMap<Channel, f32>,
    /// How a channel's own scores scale its weight on this query, if at all.
    confidence: Option<Confidence>,
}

/// How far a channel's best candidate has to stand above its own field to be
/// worth its full weight.
///
/// See [`Fusion::with_confidence`] for what this is for and
/// [`Fusion::how_sure`] for the arithmetic.
#[derive(Clone, Copy, Debug)]
struct Confidence {
    /// The standardised top score that earns the full weight.
    spread: f32,
    /// What a channel with no opinion keeps.
    floor: f32,
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
            ]),
            // Off. The three corpora's accuracy floors are all standing on
            // constant weights, and a mechanism turned on before it is measured
            // is a mechanism nobody can price. See `with_confidence`.
            confidence: None,
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

    /// Scales every channel's weight by how sure that channel is on this query.
    ///
    /// **What this is for.** Rank fusion cannot tell a channel that found the
    /// answer from a channel that returned the least bad of fifty wrong
    /// documents. A first-placed candidate contributes `weight / (k + 1)`
    /// either way, so a channel with nothing to say votes exactly as loudly as
    /// one that is certain. Measured, that is not a hypothetical: fusing all
    /// four channels ranks *below* the vector channel by itself on the
    /// cross-lingual group of two separate corpora -- 0.7910 against 0.8268 on
    /// this project's own and 0.6077 against 0.6335 on XQuAD-R -- while the
    /// same lexical channels are worth having on the same-language queries of
    /// the same corpus, where segmented BM25 alone scores 0.7299 against the
    /// vector channel's 0.6787. The channels are not weak. A single constant
    /// cannot tell the two cases apart, and rank fusion gives it nothing to
    /// tell them apart with.
    ///
    /// **Why the channel's own scores and not agreement between channels.**
    /// Cross-channel agreement is the cheap answer, it needs nothing plumbed,
    /// and this project measured it and removed it: scaling the lexical pair by
    /// how much of the vector channel's list it also returned interpolated
    /// monotonically between the constants bounding it and was worth between
    /// +0.0003 and +0.0085. That route is closed. What is left is the shape of
    /// each channel's own score distribution, which no rank can show.
    ///
    /// **Why a standardised top score and not a z-score of every candidate.**
    /// Standardising the candidates does not answer this question. A channel
    /// whose fifty candidates are all worthless still maps its best one to
    /// about `z = +2`, because standardising removes the units and not the
    /// quality. What separates a sure channel from an unsure one is how far its
    /// leader stands above its own field, which is a single number:
    /// `(best - mean) / deviation`, over that channel's candidates, in that
    /// channel's units. It is dimensionless, so BM25 and cosine similarity are
    /// comparable after it where they are not before it, and it needs nothing
    /// but the candidates already in hand -- no corpus-wide distribution to
    /// build, and none to maintain as memories are written. That last part is
    /// the constraint: a normaliser that needs a corpus histogram would be
    /// wrong for a user the moment they wrote something, whatever it did to a
    /// static benchmark.
    ///
    /// `spread` is the standardised top score that earns the full weight and
    /// `floor` is what a channel with no opinion keeps. Both are swept, not
    /// argued.
    pub fn with_confidence(mut self, spread: f32, floor: f32) -> Self {
        self.confidence = Some(Confidence {
            spread: spread.max(f32::EPSILON),
            floor: floor.clamp(0.0, 1.0),
        });
        self
    }

    /// How far this channel's best candidate stands above its own field, as a
    /// share of its weight.
    ///
    /// One when nothing configured this, so the arithmetic below is the same
    /// arithmetic either way.
    ///
    /// Three cases have no distribution to read and all three return one rather
    /// than the floor, because *unable to judge* is not the same finding as
    /// *judged and found wanting*: a channel that does not score at all, a
    /// channel that returned a single candidate, and a channel that returned
    /// none. A channel whose candidates all scored the same does have a
    /// distribution, and what it says is that the channel cannot separate its
    /// own candidates -- that one gets the floor, and it is the case this
    /// mechanism exists for.
    /// The value is bounded above by `sqrt(n - 1) / spread`, which is why
    /// [`with_confidence`](Self::with_confidence) says `spread` is coupled to
    /// how many candidates a channel proposes.
    fn how_sure(&self, candidates: &[Scored]) -> f32 {
        let Some(confidence) = self.confidence else {
            return 1.0;
        };
        if candidates.len() < 2 || candidates.iter().any(|it| it.score.is_none()) {
            return 1.0;
        }

        // The same standardisation the score combiners use, so there is one
        // mean and one deviation per channel in this file rather than two.
        // `None` from all of them means the channel could not separate its own
        // candidates, which is the case this mechanism exists for.
        //
        // Read as a maximum rather than taken from the head of the list,
        // because the graph channel's score is not its sort key and this should
        // not quietly depend on which channel it is looking at.
        // `None` rather than the channel's declared scale, and it costs
        // nothing: this reads `standardised` only, which is centred on the
        // channel's own mean and divided by its own deviation whatever scale
        // the score is on.
        let Some(best) = rescale(candidates, None)
            .into_iter()
            .filter_map(|scaled| scaled.standardised)
            .reduce(f32::max)
        else {
            return confidence.floor;
        };

        (best / confidence.spread).clamp(confidence.floor, 1.0)
    }

    /// Combines the channels this way instead of by reciprocal rank.
    ///
    /// See [`Combine`] for what the choices are, what the literature says
    /// about them, and what each is worth measured. [`Combine::Banded`] ships;
    /// the others are kept so a change to the default is a comparison against
    /// something rather than a claim.
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
    /// `Reciprocal` reads the rank and nothing else. The standardised
    /// combiners read `standardised`, which is this channel's score centred on
    /// its own mean and divided by its own deviation.
    ///
    /// A channel that returned no scores at all standardises `1 / (k + rank)`
    /// instead, which keeps reciprocal rank fusion's shape and removes its
    /// scale rather than inventing a third shape for the case. Nothing the
    /// engine assembles is in that case -- all four channels score now -- but a
    /// caller can build one and it should not silently mean zero.
    fn share(&self, scaled: Scaled, rank: u32) -> f32 {
        let reciprocal = 1.0 / (self.k + rank as f32);
        match self.combine {
            Combine::Reciprocal => reciprocal,
            Combine::Standardised | Combine::StandardisedTimesVotes => {
                scaled.standardised.unwrap_or(reciprocal)
            }
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
        let mut votes: BTreeMap<TopicId, u32> = BTreeMap::new();

        for list in lists {
            let weight = self.weight(list.channel) * self.how_sure(&list.candidates);

            // A channel worth nothing does not get to name candidates. See
            // `without`, which is the same statement made deliberately -- and
            // `with_confidence`, under which a channel that cannot separate its
            // own candidates arrives here at exactly zero if the floor allows
            // it, which is the filtering this is for.
            if weight == 0.0 {
                continue;
            }

            // Once per channel, not once per candidate: these are the
            // channel's own mean, deviation, minimum and maximum, so
            // recomputing them inside the loop would be the same numbers at
            // fifty times the cost.
            let scaled = rescale(&list.candidates, list.channel.calibrated());

            for (index, candidate) in list.candidates.iter().enumerate() {
                let rank = index as u32 + 1;
                let contribution = weight * self.share(scaled[index], rank);
                *votes.entry(candidate.topic).or_insert(0) += 1;
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
            .map(|(topic, (score, why))| {
                // CombMNZ's one difference from a plain weighted sum: multiply
                // by how many channels named this candidate at all.
                let score = match self.combine {
                    Combine::StandardisedTimesVotes => {
                        score * votes.get(&topic).copied().unwrap_or(0) as f32
                    }
                    Combine::Reciprocal | Combine::Standardised | Combine::Banded => score,
                };
                FusedResult { topic, score, why }
            })
            .collect();

        sort_results(&mut results);
        results
    }
}

/// One candidate's score, said two ways its own channel can compare.
///
/// Both are `None` when there is nothing to compute: a channel that does not
/// score, or one whose candidates all scored the same, where both divisors are
/// zero. `Fusion::share` decides what to do with that; this only reports that
/// there is nothing to report.
#[derive(Clone, Copy, Debug, Default)]
struct Scaled {
    /// Centred on the channel's mean, divided by its deviation.
    standardised: Option<f32>,
    /// Where it falls between the channel's worst and best candidate, on
    /// `[0, 1]`.
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

    let count = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / count;
    let deviation = (scores
        .iter()
        .map(|score| (score - mean) * (score - mean))
        .sum::<f32>()
        / count)
        .sqrt();
    let least = scores.iter().copied().fold(f32::MAX, f32::min);
    let most = scores.iter().copied().fold(f32::MIN, f32::max);

    // Two different questions, and they fail separately.
    //
    // `standardised` asks how far a candidate stands from its channel's own
    // field, which a channel whose candidates all scored the same cannot
    // answer at any scale. That one is `None` when the deviation vanishes,
    // whatever the channel is.
    //
    // `within` asks where a candidate falls on the scale its score is on, and
    // for a channel that declares a calibrated range there *is* an answer in
    // that case: fifty candidates all at 0.5 are all at the middle of `[0, 1]`,
    // which is the whole point of a score that means the same thing on every
    // query. Only an empirical min-max has nothing to divide by.
    let spread = (deviation > f32::EPSILON).then_some(deviation);
    let scale = match calibrated {
        Some((low, high)) if high - low > f32::EPSILON => Some((low, high - low)),
        Some(_) => None,
        None => (most - least > f32::EPSILON).then_some((least, most - least)),
    };

    scores
        .into_iter()
        .map(|score| Scaled {
            standardised: spread.map(|deviation| (score - mean) / deviation),
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
        // The vector and graph channels, because the two lexical ones
        // deliberately carry half weight. That says how much they duplicate
        // each other, not that a rank means something different in each.
        let fused = Fusion::default().fuse(&[
            ChannelResults::unscored(Channel::Graph, vec![id(1)]),
            ChannelResults::unscored(Channel::Vector, vec![id(2)]),
        ]);
        assert!((fused[0].score - fused[1].score).abs() < f32::EPSILON);
    }

    /// Rank fusion cannot see a margin; standardised fusion can.
    ///
    /// The difference between the two combiners, in one case. Both channels put
    /// their own candidate first, so both candidates get `weight / (k + 1)` and
    /// reciprocal rank fusion has to break the tie on the identifier -- even
    /// though one channel scored its leader far above its own field and the
    /// other could barely separate its top two. Standardising reads exactly
    /// that.
    #[test]
    fn a_margin_decides_under_standardised_fusion_and_is_invisible_to_ranks() {
        let convinced = id(1);
        let unconvinced = id(2);
        let lists = [
            ChannelResults::new(
                Channel::Vector,
                vec![
                    Scored::new(convinced, 0.99),
                    Scored::new(id(3), 0.10),
                    Scored::new(id(4), 0.09),
                ],
            ),
            ChannelResults::new(
                Channel::LexicalSegmented,
                vec![
                    Scored::new(unconvinced, 4.01),
                    Scored::new(id(5), 4.00),
                    Scored::new(id(6), 3.99),
                ],
            ),
        ];

        // Equal weights, so the combiner is the only thing being compared.
        let level = Fusion::default()
            .with_weight(Channel::Vector, 1.0)
            .with_weight(Channel::LexicalSegmented, 1.0);

        let ranked = level.clone().with(Combine::Reciprocal).fuse(&lists);
        let top = ranked.iter().find(|r| r.topic == convinced).unwrap();
        let other = ranked.iter().find(|r| r.topic == unconvinced).unwrap();
        assert!(
            (top.score - other.score).abs() < f32::EPSILON,
            "reciprocal rank fusion should see these two as identical: {} against {}",
            top.score,
            other.score
        );

        let scored = level.clone().with(Combine::Standardised).fuse(&lists);
        assert_eq!(
            scored[0].topic, convinced,
            "the channel with a real margin should win once magnitudes are read"
        );
    }

    /// A channel's units do not survive standardisation, which is the point.
    #[test]
    fn standardised_fusion_does_not_care_what_units_a_channel_uses() {
        let target = id(1);
        let shape = [9.0f32, 1.0, 0.5];
        let build = |scale: f32| {
            [
                ChannelResults::new(
                    Channel::Vector,
                    shape
                        .iter()
                        .enumerate()
                        .map(|(n, score)| Scored::new(id(n as u8 + 1), score * scale))
                        .collect(),
                ),
                ChannelResults::new(
                    Channel::LexicalSegmented,
                    vec![Scored::new(id(7), 2.0), Scored::new(target, 1.0)],
                ),
            ]
        };

        let fusion = Fusion::default().with(Combine::Standardised);
        let small = fusion.fuse(&build(1.0));
        let large = fusion.fuse(&build(1000.0));

        let order = |results: &[FusedResult]| -> Vec<TopicId> {
            results.iter().map(|result| result.topic).collect()
        };
        assert_eq!(
            order(&small),
            order(&large),
            "multiplying one channel's scores by a thousand reordered the results"
        );
    }

    /// CombMNZ multiplies by agreement, and agreement is what goes wrong here.
    ///
    /// Recorded as a test rather than a claim because the systematic comparison
    /// that favours CombMNZ (`arXiv:2507.03761`) and the four-channel analysis
    /// that names agreement as the failure mode (`arXiv:2508.01405`) predict
    /// opposite things for this project. The mechanism is small enough to state
    /// exactly, and it is stated in numbers: the vector channel is certain
    /// about one candidate and the lexical channel is certain about another,
    /// which the lexical channel's own weaker margin does not overcome -- until
    /// being named by two channels doubles it.
    #[test]
    fn combmnz_rewards_being_named_twice_over_being_scored_well() {
        let certain = id(1);
        let agreed = id(2);
        let lists = [
            ChannelResults::new(
                Channel::Vector,
                vec![
                    Scored::new(certain, 1.00),
                    Scored::new(agreed, 0.10),
                    Scored::new(id(3), 0.09),
                    Scored::new(id(4), 0.08),
                    Scored::new(id(5), 0.07),
                    Scored::new(id(6), 0.06),
                ],
            ),
            ChannelResults::new(
                Channel::LexicalSegmented,
                vec![
                    Scored::new(agreed, 1.00),
                    Scored::new(id(7), 0.35),
                    Scored::new(id(8), 0.30),
                    Scored::new(id(9), 0.25),
                    Scored::new(id(10), 0.20),
                    Scored::new(id(11), 0.15),
                ],
            ),
        ];
        let level = Fusion::default()
            .with_weight(Channel::Vector, 1.0)
            .with_weight(Channel::LexicalSegmented, 1.0);

        // Standardised: the vector channel's +2.23 beats the lexical channel's
        // +2.18 once the vector channel's own -0.39 for the same candidate is
        // taken off it.
        assert_eq!(
            level.clone().with(Combine::Standardised).fuse(&lists)[0].topic,
            certain,
            "a weighted sum should follow the margin"
        );
        // CombMNZ: the same 1.79 doubled for being named twice is 3.58, which
        // is more than 2.23, so agreement wins on nothing but its own count.
        assert_eq!(
            level.with(Combine::StandardisedTimesVotes).fuse(&lists)[0].topic,
            agreed,
            "and CombMNZ should follow the agreement, which is the whole disagreement"
        );
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
            Fusion::default().fuse(&lists)[0].score
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
        let fused =
            Fusion::default().fuse(&[ChannelResults::new(Channel::Graph, identical.clone())]);

        let first = fused[0].score;
        assert!(
            fused.iter().all(|result| result.score == first),
            "identical evidence produced different contributions: {fused:?}"
        );

        // Where it ties is the middle of the band, because 0.5 is the middle
        // of the declared range -- not its top and not its bottom.
        let strong = Fusion::default().fuse(&[ChannelResults::new(
            Channel::Graph,
            (1..=6).map(|n| Scored::new(id(n), 1.0)).collect(),
        )])[0]
            .score;
        let faint = Fusion::default().fuse(&[ChannelResults::new(
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

    /// A channel that cannot separate its own candidates loses its vote.
    ///
    /// The case the mechanism exists for. Flat scores mean the channel returned
    /// fifty things and is equally unimpressed by all of them, which is what a
    /// lexical channel looks like from the inside on a query whose answer is in
    /// another language. At a floor of zero it says nothing at all.
    #[test]
    fn a_channel_that_cannot_separate_its_candidates_says_nothing() {
        let flat: Vec<Scored> = (1..=4).map(|n| Scored::new(id(n), 3.0)).collect();
        let sharp = vec![
            Scored::new(id(10), 9.0),
            Scored::new(id(11), 1.0),
            Scored::new(id(12), 0.9),
            Scored::new(id(13), 0.8),
        ];

        let fused = Fusion::default()
            .with_confidence(2.0, 0.0)
            .with_weight(Channel::LexicalSegmented, 1.0)
            .fuse(&[
                ChannelResults::new(Channel::LexicalSegmented, flat),
                ChannelResults::new(Channel::Vector, sharp),
            ]);

        assert_eq!(
            fused.iter().map(|result| result.topic).collect::<Vec<_>>(),
            vec![id(10), id(11), id(12), id(13)],
            "the flat channel still put candidates in the results"
        );
    }

    /// And a channel with a standout keeps its vote.
    ///
    /// The other half, asserted separately, because a mechanism that silences
    /// everything would pass the test above and be worthless.
    #[test]
    fn a_channel_with_a_standout_keeps_its_weight() {
        let confident = vec![
            Scored::new(id(1), 40.0),
            Scored::new(id(2), 1.0),
            Scored::new(id(3), 1.0),
            Scored::new(id(4), 1.0),
        ];

        // A spread of one, because the standardised top score over four
        // candidates cannot exceed `sqrt(3)` = 1.73 and this channel is at that
        // ceiling already -- at a spread of two it could not reach full weight
        // however cleanly it separated them. See `with_confidence`.
        let fused = Fusion::default()
            .with_confidence(1.0, 0.0)
            .with_weight(Channel::LexicalSegmented, 1.0)
            .fuse(&[ChannelResults::new(Channel::LexicalSegmented, confident)]);

        let applied = fused[0]
            .why
            .iter()
            .find_map(|why| match why {
                Why::Channel { weight, .. } => Some(*weight),
                Why::Path { .. } | Why::Reranked { .. } => None,
            })
            .expect("a channel entry");
        assert_eq!(
            applied, 1.0,
            "a channel at the ceiling of what four candidates can separate was discounted"
        );
    }

    /// Standardising the candidates is not what does the work here.
    ///
    /// Worth asserting because it is the mistake this design was nearly built
    /// on. A z-score over a channel's candidates removes the units, not the
    /// quality: scale every score by a hundred and nothing about the channel
    /// has changed, and nothing about its confidence should either. What the
    /// floor catches is a channel with no shape to its scores, at any scale.
    #[test]
    fn multiplying_a_channels_scores_changes_nothing_about_its_confidence() {
        let fusion = Fusion::default().with_confidence(2.0, 0.0);
        let shape = [9.0f32, 1.0, 0.9, 0.8];

        let small: Vec<Scored> = shape
            .iter()
            .enumerate()
            .map(|(n, score)| Scored::new(id(n as u8 + 1), *score))
            .collect();
        let large: Vec<Scored> = shape
            .iter()
            .enumerate()
            .map(|(n, score)| Scored::new(id(n as u8 + 1), score * 100.0))
            .collect();

        assert!(
            (fusion.how_sure(&small) - fusion.how_sure(&large)).abs() < 1e-6,
            "{} against {}",
            fusion.how_sure(&small),
            fusion.how_sure(&large)
        );
    }

    /// Unable to judge is not the same finding as judged and found wanting.
    #[test]
    fn a_channel_with_no_distribution_to_read_is_not_penalised() {
        let fusion = Fusion::default().with_confidence(2.0, 0.0);

        assert_eq!(fusion.how_sure(&[]), 1.0, "no candidates");
        assert_eq!(
            fusion.how_sure(&[Scored::new(id(1), 5.0)]),
            1.0,
            "one candidate is not a distribution"
        );
        assert_eq!(
            fusion.how_sure(&[Scored::unscored(id(1)), Scored::unscored(id(2))]),
            1.0,
            "a channel that does not score cannot be judged on its scores"
        );
    }

    /// Nothing changes until somebody asks for it.
    #[test]
    fn confidence_is_off_unless_it_is_configured() {
        let flat: Vec<Scored> = (1..=4).map(|n| Scored::new(id(n), 3.0)).collect();
        assert_eq!(Fusion::default().how_sure(&flat), 1.0);
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
