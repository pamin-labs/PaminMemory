//! Whether a difference between two runs is a result or a rounding error.
//!
//! Every accuracy conclusion this project has drawn was a difference of two
//! means. That is the weakest of the standard readings and it has been wrong
//! here before at exactly the sizes being argued about: the fusion weight moved
//! on +0.0056, and the reranker was priced at -0.0152. Both are small enough
//! that a handful of queries decides them, and neither was ever checked against
//! the possibility that the handful was luck.
//!
//! The published work that made this hard to ignore is the one paper that fuses
//! a graph channel with a vector channel (`arXiv:2603.28886`, 2026). It reports
//! rank fusion beating vector-only by +1.7 points -- and then reports that the
//! same comparison is 15 wins against 6 losses at p = 0.078, while its own
//! method's *smaller* mean gain is 8 wins against 1 loss at p = 0.039. The
//! means say the wrong thing; the per-query counts say the right one.
//!
//! So a comparison here reports three things: how many queries moved each way,
//! the mean difference, and how often chance alone produces a difference that
//! large.
//!
//! ## Why a bootstrap rather than a t-test or Wilcoxon
//!
//! Per-query nDCG is bounded in `[0, 1]`, is not normal, and piles up at both
//! ends -- on this project's own corpus two of three groups sit at 1.000, so
//! most differences are exactly zero. A t-test assumes a shape this data does
//! not have, and Wilcoxon's signed rank discards the size of each difference
//! just when the sizes are the whole question. The paired bootstrap assumes
//! nothing beyond the queries being a sample, which is the assumption the
//! evaluation already rests on.
//!
//! It is also deterministic here: the resampling uses a fixed seed, so two runs
//! of the same comparison report the same p to the last digit. A significance
//! figure that moved between runs would be one more number nobody could check.
// A fourth test binary includes these modules and does not use every helper --
// the same reason `scoring` carries this. One helper per harness is what these
// shared modules exist to undo.
#![allow(dead_code)]

use std::fmt;

/// How many times the queries are resampled.
///
/// Ten thousand puts the granularity of the reported p at 1e-4, which is finer
/// than any threshold anyone reads it against, and costs milliseconds on the
/// query counts here -- 1,190 for the cross-lingual corpus, 482 for MIRACL.
const RESAMPLES: usize = 10_000;

/// Differences smaller than this count as neither a win nor a loss.
///
/// Per-query nDCG is a ratio of sums of `1/log2(rank + 2)`, so two orderings
/// that differ nowhere inside the cutoff produce bit-identical values and a
/// difference of exactly zero. This is only here so that a difference produced
/// by floating-point summation order is not read as a query changing its mind.
const NOTHING_HAPPENED: f64 = 1e-9;

/// What one run did against another, query by query.
pub struct Paired {
    /// Queries the second run scored higher on.
    pub wins: usize,
    /// Queries the second run scored lower on.
    pub losses: usize,
    /// Queries neither run separated.
    pub ties: usize,
    /// The difference of the means, second run less the first.
    pub mean: f64,
    /// How often resampling the same queries produces a mean difference at
    /// least this large in absolute value, when the true difference is zero.
    pub p: f64,
}

impl Paired {
    /// Whether the difference clears the conventional bar.
    ///
    /// Five per cent, and named rather than written at each call site so that
    /// nobody has to wonder which convention a given table used. A comparison
    /// that does not clear it is not a small result; it is an absent one, and
    /// the wins and losses are what to report instead.
    pub fn is_significant(&self) -> bool {
        self.p < 0.05
    }
}

impl fmt::Display for Paired {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "{:+.4}  {}W/{}L/{}T  p={:.4}{}",
            self.mean,
            self.wins,
            self.losses,
            self.ties,
            self.p,
            if self.is_significant() {
                ""
            } else {
                "  (not significant)"
            }
        )
    }
}

/// Compares two runs over the same queries, in the same order.
///
/// Panics on differing lengths rather than truncating, because a comparison of
/// two runs that scored different query sets is not a comparison of anything,
/// and silently taking the shorter one is how that mistake survives.
pub fn compare(before: &[f64], after: &[f64]) -> Paired {
    assert_eq!(
        before.len(),
        after.len(),
        "a paired comparison needs the same queries on both sides, in the same order"
    );

    let differences: Vec<f64> = before
        .iter()
        .zip(after)
        .map(|(first, second)| second - first)
        .collect();

    let wins = differences
        .iter()
        .filter(|difference| **difference > NOTHING_HAPPENED)
        .count();
    let losses = differences
        .iter()
        .filter(|difference| **difference < -NOTHING_HAPPENED)
        .count();
    let ties = differences.len() - wins - losses;
    let mean = average(&differences);

    Paired {
        wins,
        losses,
        ties,
        mean,
        p: how_often_chance_does_this(&differences, mean),
    }
}

/// The share of sign-flips whose mean difference is at least as extreme as the
/// one observed, under the hypothesis that the true difference is zero.
///
/// **A paired randomisation test, not a bootstrap, and the difference is
/// measured rather than stylistic.** This used to centre the differences and
/// resample them with replacement -- the bootstrap-shift test. Urbano, Lima and
/// Hanjalic measured that test over five hundred million simulated p values
/// (`arXiv:1905.11096`) and found it anti-conservative: 0.059 actual against
/// 0.050 nominal, 0.014 against 0.010, with "a systematic bias towards small p
/// values". They recommend discontinuing it in favour of the permutation or
/// `t` test, finding "virtually no gain" from the bootstrap. Every p value this
/// repository had published was from the test they name.
///
/// The randomisation is the exact one for paired data: under the null, which
/// of the two settings produced the higher score on a given query is a coin
/// flip, so each difference's *sign* is exchangeable and flipping signs
/// enumerates the null distribution. Nothing is assumed about the shape of the
/// differences, which matters here because they are sparse and heavy-tailed --
/// two near-identical fusion settings agree exactly on most queries and differ
/// a lot on a few.
///
/// The count is offset by one on both sides, which keeps a p of exactly zero
/// from being reported for an effect that merely exceeded every draw -- ten
/// thousand draws cannot distinguish 1e-5 from impossible.
fn how_often_chance_does_this(differences: &[f64], observed: f64) -> f64 {
    if differences.is_empty() {
        return 1.0;
    }

    let mut draws = Draws(0x5eed_600d_15c0);
    let extreme = (0..RESAMPLES)
        .filter(|_| {
            let flipped: f64 = differences
                .iter()
                .map(|difference| {
                    if draws.below(2) == 0 {
                        *difference
                    } else {
                        -difference
                    }
                })
                .sum::<f64>()
                / differences.len() as f64;
            flipped.abs() >= observed.abs()
        })
        .count();

    (extreme + 1) as f64 / (RESAMPLES + 1) as f64
}

/// Which row of a sweep the others are priced against.
///
/// Here rather than in each harness because all three sweep the same grid and
/// all three had the same hole: they printed means and left the reader to
/// subtract. Subtracting is what this module exists to replace.
///
/// The best-scoring row by default, which answers "is the winner
/// distinguishable from the field at all". `SWEEP_AGAINST` names a row by
/// label substring instead, for the other question a sweep gets asked: two
/// particular settings, one of which is what ships. The two are not
/// interchangeable. Two rows each compared against a third are *not* compared
/// against each other -- a paired test needs the pair -- and the comparison a
/// default rests on is almost always a pair.
pub fn baseline<T>(measured: &[(String, T)], score: impl Fn(&T) -> f64) -> Option<&(String, T)> {
    match std::env::var("SWEEP_AGAINST") {
        Ok(wanted) => measured.iter().find(|(label, _)| label.contains(&wanted)),
        Err(_) => measured
            .iter()
            .max_by(|left, right| score(&left.1).total_cmp(&score(&right.1))),
    }
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// A fixed sequence, so a p value is the same on every run.
struct Draws(u64);

impl Draws {
    fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // The high bits, because the low bits of a linear congruential
        // generator cycle far too visibly to resample with.
        ((self.0 >> 33) as usize) % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handful of queries all improving is not evidence, and the test this
    /// replaced said it was.
    ///
    /// Three differences of one. Under the sign-flip randomisation there are
    /// eight assignments and two of them -- all positive and all negative --
    /// have a mean at least as extreme as the observed one, so `p = 0.25`:
    /// three queries cannot distinguish a real effect from a coin landing the
    /// same way three times.
    ///
    /// The bootstrap-shift test this replaced centred the differences first,
    /// which turns `[1, 1, 1]` into `[0, 0, 0]`. Every resample then has a
    /// mean of zero, none is as extreme as the observed one, and it reports
    /// `p = 0.0001`. That is not a rounding difference; it is the difference
    /// between "not evidence" and "overwhelming", and this repository
    /// published sweep rows at `p = 0.0618` on four non-tied queries where the
    /// smallest attainable p is `2 / 2^4 = 0.125`.
    #[test]
    fn a_few_queries_all_improving_is_a_coin_landing_the_same_way() {
        let before = [0.0, 0.0, 0.0];
        let after = [1.0, 1.0, 1.0];
        let paired = compare(&before, &after);
        assert_eq!((paired.wins, paired.losses, paired.ties), (3, 0, 0));
        assert!(
            (paired.p - 0.25).abs() < 0.02,
            "the sign-flip null over three differences has two extreme \
             assignments of eight, so p is a quarter; got {}",
            paired.p
        );
        assert!(!paired.is_significant(), "p was {}", paired.p);
    }

    /// The smallest p a given number of untied queries can reach.
    ///
    /// `2 / 2^n`, because only the all-positive and all-negative assignments
    /// are at least as extreme when every difference has the same sign. Ties
    /// contribute nothing: flipping a zero leaves it zero. This is the bound
    /// that makes the row above impossible, and it is asserted so that a
    /// future change to the test cannot quietly go below it.
    #[test]
    fn ties_do_not_buy_significance() {
        // Four untied queries among twenty: the effective sample is four.
        let before = vec![0.0; 20];
        let mut after = vec![0.0; 20];
        for difference in after.iter_mut().take(4) {
            *difference = 1.0;
        }

        let paired = compare(&before, &after);
        assert_eq!((paired.wins, paired.losses, paired.ties), (4, 0, 16));
        assert!(
            paired.p >= 2.0 / 16.0 - 0.02,
            "four untied queries cannot reach below 2/2^4 = 0.125; got {}",
            paired.p
        );
    }

    #[test]
    fn two_identical_runs_have_nothing_to_report() {
        let run = [0.5, 0.25, 1.0, 0.0];
        let paired = compare(&run, &run);
        assert_eq!((paired.wins, paired.losses, paired.ties), (0, 0, 4));
        assert_eq!(paired.mean, 0.0);
        assert!(!paired.is_significant(), "p was {}", paired.p);
    }

    #[test]
    fn one_query_in_a_hundred_is_not_a_result() {
        // The shape this whole module exists to catch: a positive mean carried
        // by a single query, which is what a mean alone cannot tell you.
        let before = vec![0.5; 100];
        let mut after = before.clone();
        after[0] = 1.0;

        let paired = compare(&before, &after);
        assert_eq!((paired.wins, paired.losses), (1, 0));
        assert!(
            !paired.is_significant(),
            "a single query moved and p came back at {:.4}",
            paired.p
        );
    }

    #[test]
    fn a_consistent_gain_is_a_result() {
        // Every query improves a little. Same mean as the test above -- 0.005
        // -- and it should read completely differently.
        let before = vec![0.5; 100];
        let after = vec![0.505; 100];

        let paired = compare(&before, &after);
        assert_eq!((paired.wins, paired.losses), (100, 0));
        assert!(
            paired.is_significant(),
            "every query improved and p came back at {:.4}",
            paired.p
        );
    }

    #[test]
    fn wins_and_losses_that_cancel_are_not_a_result() {
        // Half up, half down, by the same amount. The mean is zero and so is
        // the finding, but a table of two means would show two numbers and
        // invite a reader to pick one.
        let before = vec![0.5; 100];
        let after: Vec<f64> = (0..100)
            .map(|query| if query % 2 == 0 { 0.6 } else { 0.4 })
            .collect();

        let paired = compare(&before, &after);
        assert_eq!((paired.wins, paired.losses), (50, 50));
        assert!(!paired.is_significant(), "p was {}", paired.p);
    }

    #[test]
    fn the_same_comparison_reports_the_same_p_twice() {
        // A significance figure that moved between runs would be one more
        // number nobody could check.
        let before: Vec<f64> = (0..200).map(|query| (query % 7) as f64 / 7.0).collect();
        let after: Vec<f64> = before.iter().map(|score| score + 0.01).collect();

        assert_eq!(compare(&before, &after).p, compare(&before, &after).p);
    }
}
