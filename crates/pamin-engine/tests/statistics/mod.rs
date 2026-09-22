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

/// The share of resamples whose mean difference is at least as extreme as the
/// one observed, under the hypothesis that the true difference is zero.
///
/// The differences are centred before resampling, which is what makes this a
/// test of that hypothesis rather than a confidence interval around the
/// observed value. Resampling the uncentred differences would answer "how
/// precisely do we know this number", and the question here is the other one:
/// "would we have seen a number like this if there were nothing there".
///
/// The count is offset by one on both sides, which keeps a p of exactly zero
/// from being reported for an effect that merely exceeded every resample --
/// ten thousand resamples cannot distinguish 1e-5 from impossible.
fn how_often_chance_does_this(differences: &[f64], observed: f64) -> f64 {
    if differences.is_empty() {
        return 1.0;
    }

    let centred: Vec<f64> = differences
        .iter()
        .map(|difference| difference - observed)
        .collect();

    let mut draws = Draws(0x5eed_600d_15c0);
    let extreme = (0..RESAMPLES)
        .filter(|_| {
            let resampled: f64 = (0..centred.len())
                .map(|_| centred[draws.below(centred.len())])
                .sum::<f64>()
                / centred.len() as f64;
            resampled.abs() >= observed.abs()
        })
        .count();

    (extreme + 1) as f64 / (RESAMPLES + 1) as f64
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
