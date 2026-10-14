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
//! ## Why a paired randomisation test
//!
//! Per-query nDCG is bounded in `[0, 1]`, is not normal, and piles up at both
//! ends -- on this project's own corpus two of three groups sit at 1.000, so
//! most differences are exactly zero. A t-test assumes a shape this data does
//! not have, and Wilcoxon's signed rank discards the size of each difference
//! just when the sizes are the whole question.
//!
//! This section used to argue for the paired bootstrap on those grounds, and
//! the grounds were right and the conclusion was not. The bootstrap-shift test
//! is measured to be anti-conservative in exactly this setting -- see
//! `how_often_chance_does_this` -- and the sign-flip randomisation keeps both
//! properties the argument wanted: it assumes nothing about shape, and it
//! reads the size of every difference.
//!
//! It is also deterministic here: the draws use a fixed seed, so two runs of
//! the same comparison report the same p to the last digit. A significance
//! figure that moved between runs would be one more number nobody could check.
//!
//! ## Choosing is a fit, and a fit is scored on what it did not see
//!
//! A sweep that scores ninety settings on every query and ships the best one
//! reports the best of ninety draws as though it were one, and the same
//! queries both choose the setting and grade it. Fuhr (SIGIR Forum 51(3),
//! 2017) is blunt that the tuning set must be disjoint from the test set, and
//! Cawley and Talbot (JMLR 11, 2010) measure what skipping it costs: an
//! optimism the size of the difference between competing methods, large
//! enough to reorder them. [`cross_validate`] is the procedure instead -- the
//! choice is made on four fifths of the queries and scored on the fifth it
//! never saw, five times -- and [`family`] prices a whole sweep as one family
//! of comparisons rather than ninety separate ones.
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
        self.p < SIGNIFICANCE
    }
}

/// The bar [`Paired::is_significant`] holds a p to.
pub const SIGNIFICANCE: f64 = 0.05;

/// The smallest p [`compare`] can report over `n` paired differences.
///
/// Only the two sign patterns that agree with every observed sign are as
/// extreme as a run where every difference points one way, so the exact
/// sign-flip p cannot fall below `2 / 2^n`, and the sampled estimate cannot
/// fall below `1 / (RESAMPLES + 1)`. Ties make it larger still, so this is
/// the best case.
pub fn smallest_p(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let exact = 2.0 * 0.5f64.powi(n.min(1_100) as i32);
    exact.max(1.0 / (RESAMPLES + 1) as f64)
}

/// The fewest paired observations whose [`smallest_p`] clears
/// [`SIGNIFICANCE`].
pub fn fewest_for_significance() -> usize {
    (1..)
        .find(|n| smallest_p(*n) < SIGNIFICANCE)
        .expect("some count clears the bar")
}

/// Whether a harness applies its `is_significant` gate to `what`, measured
/// over `n` paired units, failing up front when `n` can never clear it.
///
/// Called before the corpus is indexed: a gate certain to fail is otherwise
/// found only after all of that work. `SIGNIFICANCE_GATE=off` opts an
/// exploratory run out of the gate, and the harness then skips it.
pub fn gate(what: &str, n: usize) -> bool {
    if std::env::var("SIGNIFICANCE_GATE").as_deref() == Ok("off") {
        return false;
    }
    assert!(
        smallest_p(n) < SIGNIFICANCE,
        "{what}: {n} paired units can reach p = {:.4} at best, never p < {SIGNIFICANCE}, so \
         its significance gate would fail after all the indexing. Ask for at least {} \
         units, or set SIGNIFICANCE_GATE=off for an exploratory run without the gate",
        smallest_p(n),
        fewest_for_significance()
    );
    true
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

/// Which fold each query falls in: `k` folds, stratified by group.
///
/// Within each group the queries are dealt round-robin in a fixed shuffled
/// order, so every fold holds about a `k`th of every group. Stratified because
/// the groups here differ in size by a factor of three and in kind entirely --
/// a fold that happened to hold none of a twenty-query group would choose a
/// setting without ever seeing what that group wanted.
///
/// Deterministic, for the same reason the tests above are: two runs of a
/// cross-validation that disagreed about the folds would disagree about the
/// answer, and neither would be checkable.
pub fn folds(groups: &[usize], k: usize) -> Vec<usize> {
    assert!(k >= 2, "one fold has nothing held out");
    let mut assigned = vec![0; groups.len()];
    let mut draws = Draws(0xf01d_5eed_c0de);

    let mut by_group: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (query, group) in groups.iter().enumerate() {
        by_group.entry(*group).or_default().push(query);
    }
    for queries in by_group.values_mut() {
        // Fisher-Yates, from the fixed sequence.
        for at in (1..queries.len()).rev() {
            let other = draws.below(at + 1);
            queries.swap(at, other);
        }
        for (dealt, query) in queries.iter().enumerate() {
            assigned[*query] = dealt % k;
        }
    }
    assigned
}

/// Which fold each query falls in when queries come in clusters: queries that
/// share a key share a fold.
///
/// A cluster is one unit of evidence: questions asked of one LOCOMO
/// conversation share its corpus, index and graph, and XQuAD-R scores every
/// asking once under each of two answer keys. Dealing their questions to
/// different folds lets the selection train on part of a cluster and be
/// scored on the rest of it.
///
/// Clusters are dealt by [`folds`], in the order they first appear. A cluster
/// whose queries all sit in one group is stratified by that group; one that
/// spans groups joins a single shared stratum. So a key per query gives
/// exactly `folds(groups, k)`, and one key per position across groups that
/// list the same questions gives every group the folds of its questions.
pub fn cluster_folds(groups: &[usize], keys: &[String], k: usize) -> Vec<usize> {
    assert_eq!(groups.len(), keys.len(), "one key a query");
    let mut first: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut strata: Vec<Option<usize>> = Vec::new();
    let mut cluster = Vec::with_capacity(keys.len());
    for (key, group) in keys.iter().zip(groups) {
        let next = strata.len();
        let at = *first.entry(key.as_str()).or_insert(next);
        if at == next {
            strata.push(Some(*group));
        } else if strata[at] != Some(*group) {
            strata[at] = None;
        }
        cluster.push(at);
    }
    let spanning = groups.iter().max().map_or(0, |max| max + 1);
    let strata: Vec<usize> = strata
        .iter()
        .map(|stratum| stratum.unwrap_or(spanning))
        .collect();
    let dealt = folds(&strata, k);
    cluster.iter().map(|at| dealt[*at]).collect()
}

/// One score per cluster, its queries' mean, in the order clusters first
/// appear: the unit a paired test over clustered queries counts.
pub fn cluster_means(scores: &[f64], keys: &[String]) -> Vec<f64> {
    assert_eq!(scores.len(), keys.len(), "one key a score");
    let mut first: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut sums: Vec<(f64, usize)> = Vec::new();
    for (score, key) in scores.iter().zip(keys) {
        let next = sums.len();
        let at = *first.entry(key.as_str()).or_insert(next);
        if at == next {
            sums.push((0.0, 0));
        }
        sums[at].0 += score;
        sums[at].1 += 1;
    }
    sums.into_iter()
        .map(|(sum, count)| sum / count as f64)
        .collect()
}

/// One score per cluster, in the order clusters first appear, under the same
/// rule as a macro mean over groups: each of the cluster's groups is the mean
/// of its questions there, and the cluster is the mean of its groups, each
/// group counting once whatever its size.
///
/// A conversation that asks three questions in one category and one in
/// another is then weighed as the reported macro mean weighs it, rather than
/// as three to one. A cluster of one question is that question's score, and
/// one question under each of several answer keys is the mean of its keys.
pub fn cluster_macro_means(scores: &[f64], groups: &[usize], keys: &[String]) -> Vec<f64> {
    assert_eq!(scores.len(), keys.len(), "one key a score");
    assert_eq!(groups.len(), keys.len(), "one group a score");
    let mut first: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut sums: Vec<std::collections::BTreeMap<usize, (f64, usize)>> = Vec::new();
    for ((score, group), key) in scores.iter().zip(groups).zip(keys) {
        let next = sums.len();
        let at = *first.entry(key.as_str()).or_insert(next);
        if at == next {
            sums.push(Default::default());
        }
        let entry = sums[at].entry(*group).or_insert((0.0, 0));
        entry.0 += score;
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|by_group| {
            by_group
                .values()
                .map(|(sum, count)| sum / *count as f64)
                .sum::<f64>()
                / by_group.len() as f64
        })
        .collect()
}

/// What a selection procedure is worth on queries it did not choose on.
pub struct Selected {
    /// Per query, in the input's order, the score of the setting that was
    /// chosen *without* that query. This is the procedure's score, not any one
    /// setting's, and it is what a paired test against the shipped setting
    /// should be run on.
    pub held_out: Vec<f64>,
    /// The setting each fold chose, by index. If these disagree, the choice
    /// itself is not stable at this sample size, whatever its mean says.
    pub chosen: Vec<usize>,
}

/// Cross-validates a selection rule over queries.
///
/// `scores[setting][query]` is every setting's per-query score, `groups` and
/// `folds` label each query, and `select` is the rule: given the
/// `[setting][group]` table of mean scores on some queries, it names a
/// setting. For each fold, the rule sees only the other folds' means and its
/// choice is scored on this fold.
///
/// **The rule is an argument rather than being built in, because choosing it
/// is the decision that has to be made before the table is looked at.**
/// Maximising the mean over groups, minimising the worst group's regret, and
/// maximising one group subject to another's floor are all rules over this
/// same table, and they disagree. Whichever is used has to be written down
/// first -- Fuhr's "thou shalt not formulate hypotheses after the experiment".
pub fn cross_validate(
    scores: &[Vec<f64>],
    groups: &[usize],
    folds: &[usize],
    select: impl Fn(&[Vec<f64>]) -> usize,
) -> Selected {
    let queries = groups.len();
    assert!(
        scores.iter().all(|setting| setting.len() == queries),
        "every setting must be scored on every query"
    );
    assert_eq!(folds.len(), queries, "every query needs a fold");

    let k = folds.iter().copied().max().map_or(0, |last| last + 1);
    let group_count = groups.iter().copied().max().map_or(0, |last| last + 1);
    let mut held_out = vec![0.0; queries];
    let mut chosen = Vec::with_capacity(k);

    for fold in 0..k {
        let table: Vec<Vec<f64>> = scores
            .iter()
            .map(|setting| {
                let mut sums = vec![0.0; group_count];
                let mut counts = vec![0usize; group_count];
                for (query, score) in setting.iter().enumerate() {
                    if folds[query] != fold {
                        sums[groups[query]] += score;
                        counts[groups[query]] += 1;
                    }
                }
                sums.iter()
                    .zip(&counts)
                    .map(|(sum, count)| {
                        if *count == 0 {
                            0.0
                        } else {
                            sum / *count as f64
                        }
                    })
                    .collect()
            })
            .collect();

        let pick = select(&table);
        chosen.push(pick);
        for query in 0..queries {
            if folds[query] == fold {
                held_out[query] = scores[pick][query];
            }
        }
    }

    Selected { held_out, chosen }
}

/// Adjusted p values for a whole sweep against one baseline, as one family.
///
/// **Ninety comparisons are ninety chances to be lucky**, and a table that
/// reports each one's own p lets the reader find the lucky ones by eye.
/// Bonferroni would fix that by dividing by ninety, and here it would be
/// uselessly conservative, because the settings are near-duplicates -- four
/// weights on a grid and one shared constant -- whose differences from the
/// baseline are enormously correlated. Ninety correlated tests are not ninety
/// independent chances.
///
/// So this is Westfall and Young's single-step max-T, which is the
/// correlation-aware procedure Boytsov, Belova and Westfall evaluated for
/// exactly this design -- many comparisons against one baseline on a small IR
/// test set (SIGIR 2013). Each draw flips *one* vector of signs across the
/// queries and applies it to every setting at once, which is what preserves
/// the correlation; the adjusted p for a setting is how often the largest
/// statistic in the whole family, under that draw, reaches that setting's
/// observed one. It controls the chance of *any* false positive in the table.
///
/// The statistic is `|sum d| / sqrt(sum d^2)`, which is monotone in the
/// paired t for a fixed sum of squares and whose denominator does not change
/// under a sign flip, so settings with differences of different sizes are
/// compared on one scale. A setting identical to the baseline scores zero and
/// gets an adjusted p of one.
pub fn family(baseline: &[f64], settings: &[Vec<f64>]) -> Vec<f64> {
    // Only the queries where a setting differs from the baseline carry
    // anything, and in a sweep of near-identical settings that is a small
    // share of them. Kept sparse, the family costs what those queries cost.
    let sparse: Vec<(Vec<(usize, f64)>, f64)> = settings
        .iter()
        .map(|setting| {
            assert_eq!(
                setting.len(),
                baseline.len(),
                "paired needs the same queries"
            );
            let differences: Vec<(usize, f64)> = setting
                .iter()
                .zip(baseline)
                .enumerate()
                .map(|(query, (after, before))| (query, after - before))
                .filter(|(_, difference)| difference.abs() > NOTHING_HAPPENED)
                .collect();
            let norm = differences
                .iter()
                .map(|(_, difference)| difference * difference)
                .sum::<f64>()
                .sqrt();
            (differences, norm)
        })
        .collect();

    let statistic = |differences: &[(usize, f64)], norm: f64, signs: Option<&[bool]>| -> f64 {
        if norm == 0.0 {
            return 0.0;
        }
        differences
            .iter()
            .map(|(query, difference)| match signs {
                Some(signs) if signs[*query] => -difference,
                _ => *difference,
            })
            .sum::<f64>()
            .abs()
            / norm
    };

    let observed: Vec<f64> = sparse
        .iter()
        .map(|(differences, norm)| statistic(differences, *norm, None))
        .collect();

    let mut draws = Draws(0xfa31_1e5d_7a7e);
    let mut signs = vec![false; baseline.len()];
    let mut reached = vec![0usize; settings.len()];
    for _ in 0..RESAMPLES {
        for sign in signs.iter_mut() {
            *sign = draws.below(2) == 1;
        }
        let largest = sparse
            .iter()
            .map(|(differences, norm)| statistic(differences, *norm, Some(&signs)))
            .fold(0.0, f64::max);
        for (setting, value) in observed.iter().enumerate() {
            if largest >= *value {
                reached[setting] += 1;
            }
        }
    }

    reached
        .iter()
        .zip(&observed)
        .map(|(count, value)| {
            if *value == 0.0 {
                1.0
            } else {
                (*count + 1) as f64 / (RESAMPLES + 1) as f64
            }
        })
        .collect()
}

/// The smallest mean difference these queries could detect.
///
/// At a two-sided 0.05 and 80% power, `(z_0.975 + z_0.8) * sd / sqrt(n)`,
/// with `sd` the standard deviation of the observed paired differences -- the
/// sample-size design Sakai sets out for IR test collections (Information
/// Retrieval Journal 19, 2016). Printed beside a group's results so that a
/// twenty-query group announces, before anyone reads its table, that it cannot
/// resolve the differences a fusion weight makes.
///
/// From the paired differences rather than from a published variance,
/// because two fusion settings that share three of four channels disagree on
/// far fewer queries than two unrelated systems do, and a borrowed variance
/// would overstate how little this can see.
pub fn minimum_detectable(differences: &[f64]) -> f64 {
    let n = differences.len();
    if n < 2 {
        return f64::INFINITY;
    }
    let mean = average(differences);
    let variance = differences
        .iter()
        .map(|difference| (difference - mean) * (difference - mean))
        .sum::<f64>()
        / (n - 1) as f64;
    (1.959_964 + 0.841_621) * variance.sqrt() / (n as f64).sqrt()
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

    /// Every group is spread across every fold, and the assignment does not
    /// move between runs.
    #[test]
    fn folds_are_stratified_and_fixed() {
        let groups: Vec<usize> = (0..60).map(|query| usize::from(query >= 50)).collect();
        let assigned = folds(&groups, 5);
        assert_eq!(assigned, folds(&groups, 5), "the same folds every time");

        // Ten queries in the small group, five folds: two in each.
        for fold in 0..5 {
            let small = (50..60).filter(|query| assigned[*query] == fold).count();
            assert_eq!(small, 2, "fold {fold} held {small} of the small group");
        }
    }

    #[test]
    fn two_answer_keys_for_one_question_never_cross_folds() {
        // Eleven questions, each listed once in each of two groups.
        let groups: Vec<usize> = [0; 11].into_iter().chain([1; 11]).collect();
        let keys: Vec<String> = (0..22).map(|at| (at % 11).to_string()).collect();
        let assigned = cluster_folds(&groups, &keys, 5);
        assert_eq!(&assigned[..11], &assigned[11..]);
        assert_eq!(&assigned[..11], folds(&[0; 11], 5).as_slice());
        let means = cluster_means(
            &[1.0, 3.0, 5.0, 7.0],
            &["a", "b", "a", "b"].map(String::from),
        );
        assert_eq!(means, vec![3.0, 5.0]);
    }

    /// A key per query is the plain stratified split, and a paired test over
    /// one-query clusters is a test over queries.
    #[test]
    fn a_query_its_own_cluster_changes_nothing() {
        let groups: Vec<usize> = (0..60).map(|query| usize::from(query >= 50)).collect();
        let keys: Vec<String> = (0..60).map(|query| query.to_string()).collect();
        assert_eq!(cluster_folds(&groups, &keys, 5), folds(&groups, 5));
        let scores: Vec<f64> = (0..60).map(|query| query as f64 / 7.0).collect();
        assert_eq!(cluster_means(&scores, &keys), scores);
    }

    /// The questions of one conversation, spread over several categories,
    /// stay in one fold and count once.
    #[test]
    fn a_conversation_never_crosses_folds() {
        let groups: Vec<usize> = (0..40).map(|query| query % 4).collect();
        let keys: Vec<String> = (0..40).map(|query| format!("conv-{}", query / 4)).collect();
        let assigned = cluster_folds(&groups, &keys, 5);
        for conversation in 0..10 {
            let folds: std::collections::BTreeSet<usize> = assigned
                [conversation * 4..conversation * 4 + 4]
                .iter()
                .copied()
                .collect();
            assert_eq!(folds.len(), 1, "conversation {conversation} crossed folds");
        }
        for fold in 0..5 {
            let held = (0..40).filter(|query| assigned[*query] == fold).count();
            assert_eq!(held, 8, "two conversations a fold");
        }
        let scores: Vec<f64> = (0..40).map(|query| (query / 4) as f64).collect();
        assert_eq!(
            cluster_means(&scores, &keys),
            (0..10).map(f64::from).collect::<Vec<_>>()
        );
    }

    /// Six units are the fewest a one-sided run of signs can take below
    /// five per cent; five cannot, whatever they say.
    #[test]
    fn too_few_units_cannot_be_significant() {
        assert_eq!(fewest_for_significance(), 6);
        assert!(smallest_p(5) >= SIGNIFICANCE);
        let five = compare(&[0.0; 5], &[1.0; 5]);
        assert!(!five.is_significant(), "p was {}", five.p);
        let six = compare(&[0.0; 6], &[1.0; 6]);
        assert!(six.is_significant(), "p was {}", six.p);
        assert!(six.p >= smallest_p(6) * 0.5, "p was {}", six.p);
        assert_eq!(smallest_p(0), 1.0);
        assert_eq!(smallest_p(10_000), 1.0 / (RESAMPLES + 1) as f64);
    }

    /// A conversation with three questions up a little in one category and
    /// one down more in another is a loss by the macro rule the report uses,
    /// and a win when its questions are pooled.
    #[test]
    fn a_cluster_is_weighed_as_the_macro_mean_weighs_it() {
        let keys: Vec<String> = ["c"; 4].map(String::from).to_vec();
        let groups = [0, 0, 0, 1];
        let differences = [1.0, 1.0, 1.0, -2.0];
        assert_eq!(cluster_means(&differences, &keys), vec![0.25]);
        assert_eq!(
            cluster_macro_means(&differences, &groups, &keys),
            vec![-0.5]
        );
        // One question a cluster, and one question under two answer keys.
        let single: Vec<String> = (0..3).map(|at| at.to_string()).collect();
        assert_eq!(
            cluster_macro_means(&[0.1, 0.2, 0.3], &[0, 1, 1], &single),
            vec![0.1, 0.2, 0.3]
        );
        let linked: Vec<String> = ["a", "b", "a", "b"].map(String::from).to_vec();
        assert_eq!(
            cluster_macro_means(&[1.0, 3.0, 5.0, 7.0], &[0, 0, 1, 1], &linked),
            vec![3.0, 5.0]
        );
    }

    /// Choosing the best of many settings on the same queries that grade it
    /// reports an effect that is not there.
    ///
    /// The whole reason for [`cross_validate`], shown on data where the truth
    /// is known: twenty settings whose per-query scores are the same noise,
    /// shuffled. None of them is better than another. The best one's
    /// in-sample mean is above the grand mean anyway -- that is what a maximum
    /// of twenty draws does -- and a sweep that reports it has invented a
    /// gain. The cross-validated score of the *procedure* does not, because
    /// each fold's choice is graded on queries that played no part in it.
    #[test]
    fn choosing_on_what_you_report_invents_a_gain() {
        let queries = 200;
        let mut draws = Draws(0x0000_c0ff_ee00);
        let scores: Vec<Vec<f64>> = (0..20)
            .map(|_| {
                (0..queries)
                    .map(|_| draws.below(1_000) as f64 / 1_000.0)
                    .collect()
            })
            .collect();
        let groups = vec![0; queries];

        let grand = scores.iter().map(|setting| average(setting)).sum::<f64>() / 20.0;
        let best_in_sample = scores
            .iter()
            .map(|setting| average(setting))
            .fold(f64::MIN, f64::max);

        let selected = cross_validate(&scores, &groups, &folds(&groups, 5), |table| {
            (0..table.len())
                .max_by(|left, right| table[*left][0].total_cmp(&table[*right][0]))
                .expect("settings")
        });
        let held_out = average(&selected.held_out);

        assert!(
            best_in_sample - grand > 0.02,
            "the premise: the best of twenty noisy settings looks better than \
             the grand mean in-sample ({best_in_sample} against {grand})"
        );
        assert!(
            (held_out - grand).abs() < best_in_sample - grand,
            "held out, the procedure should be much nearer the grand mean than \
             the in-sample winner is: {held_out} against {grand}, in-sample {best_in_sample}"
        );
    }

    /// A setting that really is better is chosen in every fold.
    #[test]
    fn a_real_difference_survives_cross_validation() {
        let queries = 100;
        let mut draws = Draws(0x0000_beef_0001);
        let noise: Vec<f64> = (0..queries)
            .map(|_| draws.below(1_000) as f64 / 2_000.0)
            .collect();
        let scores = vec![
            noise.clone(),
            noise.iter().map(|value| value + 0.3).collect(),
            noise.iter().map(|value| value + 0.1).collect(),
        ];
        let groups = vec![0; queries];
        let selected = cross_validate(&scores, &groups, &folds(&groups, 5), |table| {
            (0..table.len())
                .max_by(|left, right| table[*left][0].total_cmp(&table[*right][0]))
                .expect("settings")
        });
        assert_eq!(selected.chosen, vec![1; 5]);
    }

    /// The family's adjusted p is never smaller than a setting's own, and a
    /// setting identical to the baseline is not a result.
    #[test]
    fn a_family_is_harder_to_pass_than_one_comparison() {
        let baseline: Vec<f64> = (0..40).map(|query| f64::from(query % 3) / 3.0).collect();
        let identical = baseline.clone();
        let better: Vec<f64> = baseline.iter().map(|value| value + 0.05).collect();
        let mut mixed = baseline.clone();
        for value in mixed.iter_mut().take(6) {
            *value += 0.2;
        }

        let settings = vec![identical, better.clone(), mixed.clone()];
        let adjusted = family(&baseline, &settings);

        assert_eq!(
            adjusted[0], 1.0,
            "identical to the baseline is not a result"
        );
        assert!(
            adjusted[1] < 0.05,
            "a gain on every query survives: {}",
            adjusted[1]
        );
        let own = compare(&baseline, &mixed).p;
        assert!(
            adjusted[2] >= own - 0.02,
            "the family can only raise a p: {} against its own {own}",
            adjusted[2]
        );
    }

    /// Fewer queries, or noisier differences, see less.
    #[test]
    fn the_detectable_difference_shrinks_with_the_square_root_of_the_queries() {
        let twenty: Vec<f64> = (0..20)
            .map(|query| if query % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let eighty: Vec<f64> = (0..80)
            .map(|query| if query % 2 == 0 { 0.1 } else { -0.1 })
            .collect();
        let small = minimum_detectable(&twenty);
        let large = minimum_detectable(&eighty);
        assert!(
            (small / large - 2.0).abs() < 0.05,
            "four times the queries should halve it: {small} against {large}"
        );
        assert!(minimum_detectable(&[0.1]).is_infinite());
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
