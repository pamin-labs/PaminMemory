//! A learned decision whether to rerank, held out the way it would ship.
//!
//! The hand-written gates in [`super::Routes`] are rules someone guessed. The
//! literature's verdict on guessing is that no single signal is strong enough
//! (Adaptive Re-Ranking, arXiv 2606.25249: every simple signal |rho| < 0.15),
//! and its verdict on predicting is that a router recovers 60-80% of an
//! oracle's saving at best, and only when it is trained on its own gain label
//! and tested on queries it never saw. So this is that: a ridge regression of
//! the pass's gain -- the shipped order's nDCG minus fusion's -- on features
//! read off the list *before* the pass, cross-validated so every query's
//! decision comes from a model that did not see it, with the threshold chosen
//! inside the training folds too.
//!
//! **Held out by question, not by row.** XQuAD-R asks each question in eleven
//! languages; a fold that trained on the German asking and tested on the
//! Spanish one would be grading memory, not prediction. The caller names the
//! question, and a question's rows share a fold.
//!
//! Two feature sets: what the list already says, which costs nothing, and that
//! plus the small reranker's opinion, which costs its pass on every query.

use std::collections::BTreeMap;

use pamin_core::{Channel, Why};
use pamin_engine::SearchHit;

use super::Replayed;
use crate::statistics;

const FOLDS: u64 = 5;

/// Features read off the list before the pass, which cost nothing.
const FREE: [&str; 10] = [
    "channels proposing the top",
    "top is first in both lexical channels",
    "top has a lexical channel",
    "graph found something",
    "candidates the pass would show",
    "vector's best similarity",
    "vector's first-to-second margin",
    "lexical and vector top ten overlap",
    "query and top differ in language",
    "a language could not be told",
];

/// What the small model adds, at the price of its pass.
const SMALL: [&str; 2] = [
    "small model's first is fusion's first movable",
    "small model's first-to-second margin",
];

/// The share of queries the router would rerank, as targets. The rate
/// actually reached is reported, because it comes from a threshold chosen on
/// other folds.
const RATES: [f64; 7] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];

/// Which of a query's features a variant of the router reads.
type Features = fn(&Seen) -> Vec<f64>;

/// One query as the router sees it.
struct Seen {
    group: String,
    fold: u64,
    free: [f64; FREE.len()],
    small: [f64; SMALL.len()],
}

#[derive(Default)]
pub struct Router {
    seen: Vec<Seen>,
}

impl Router {
    /// Records one query. `question` names what the query asks, so every
    /// asking of one question is held out together; `cheap` is the small
    /// model's score for each of `replayed.movable`.
    pub fn observe(
        &mut self,
        group: &str,
        question: &str,
        query: &str,
        hits: &[SearchHit],
        replayed: &Replayed,
        cheap: &[f32],
    ) {
        let by_name: BTreeMap<&str, &SearchHit> =
            hits.iter().map(|hit| (hit.topic.as_str(), hit)).collect();
        let channels_of = |name: &str| -> Vec<(Channel, u32, Option<f32>)> {
            by_name[name]
                .result
                .why
                .iter()
                .filter_map(|why| match why {
                    Why::Channel {
                        channel,
                        rank,
                        score,
                        ..
                    } => Some((*channel, *rank, *score)),
                    _ => None,
                })
                .collect()
        };
        let lexical = |channel: &Channel| {
            matches!(channel, Channel::LexicalSegmented | Channel::LexicalNgram)
        };

        let top = replayed
            .fused
            .first()
            .map(|name| channels_of(name))
            .unwrap_or_default();
        let exact = top
            .iter()
            .filter(|(channel, rank, _)| lexical(channel) && *rank == 1)
            .count()
            == 2;

        // Each channel's own first ten, from the traces.
        let mut vector: Vec<(u32, f32)> = Vec::new();
        let mut vector_ten = Vec::new();
        let mut lexical_ten = Vec::new();
        for hit in hits {
            for (channel, rank, score) in channels_of(&hit.topic) {
                if channel == Channel::Vector {
                    vector.push((rank, score.unwrap_or(0.0)));
                    if rank <= 10 {
                        vector_ten.push(hit.topic.as_str());
                    }
                }
                if lexical(&channel) && rank <= 10 {
                    lexical_ten.push(hit.topic.as_str());
                }
            }
        }
        vector.sort_by_key(|(rank, _)| *rank);
        lexical_ten.sort_unstable();
        lexical_ten.dedup();
        let union = {
            let mut all = [vector_ten.clone(), lexical_ten.clone()].concat();
            all.sort_unstable();
            all.dedup();
            all.len()
        };
        let shared = vector_ten
            .iter()
            .filter(|name| lexical_ten.binary_search(name).is_ok())
            .count();

        let asked = pamin_index::detect_language(query).map(|(code, _)| code);
        let answered = replayed
            .fused
            .first()
            .and_then(|name| pamin_index::detect_language(&by_name[name.as_str()].state.content))
            .map(|(code, _)| code);
        let (differ, unknown) = match (asked, answered) {
            (Some(asked), Some(answered)) => (f64::from(asked != answered), 0.0),
            _ => (0.0, 1.0),
        };

        let free = [
            top.len() as f64,
            f64::from(exact),
            f64::from(top.iter().any(|(channel, ..)| lexical(channel))),
            f64::from(replayed.movable.iter().any(|at| *at >= replayed.head)),
            replayed.movable.len() as f64,
            vector.first().map_or(0.0, |(_, score)| f64::from(*score)),
            match vector.as_slice() {
                [first, second, ..] => f64::from(first.1 - second.1),
                _ => 0.0,
            },
            if union == 0 {
                0.0
            } else {
                shared as f64 / union as f64
            },
            differ,
            unknown,
        ];

        // `movable` is in fused order, so its first is fusion's first movable.
        let mut order: Vec<usize> = (0..cheap.len()).collect();
        order.sort_by(|left, right| cheap[*right].total_cmp(&cheap[*left]).then(left.cmp(right)));
        let small = match order.as_slice() {
            [] => [1.0, 0.0],
            [only] => [f64::from(*only == 0), 0.0],
            [first, second, ..] => [
                f64::from(*first == 0),
                f64::from(cheap[*first] - cheap[*second]),
            ],
        };

        self.seen.push(Seen {
            group: group.to_string(),
            fold: fold_of(question),
            free,
            small,
        });
    }

    /// The trade-off curve: at each target rate, the held-out router against
    /// a random choice at the rate it reached and against the oracle.
    /// `fused` and `shipped` are the two rows' per-query scores by group, in
    /// the order [`Router::observe`] saw each group's queries.
    pub fn report(
        &self,
        fused: &BTreeMap<String, crate::scoring::Scores>,
        shipped: &BTreeMap<String, crate::scoring::Scores>,
    ) {
        // Line each query up with its two scores.
        let mut next: BTreeMap<&str, usize> = BTreeMap::new();
        let mut off = Vec::with_capacity(self.seen.len());
        let mut on = Vec::with_capacity(self.seen.len());
        for seen in &self.seen {
            let at = next.entry(seen.group.as_str()).or_default();
            off.push(fused[&seen.group].per_query[*at]);
            on.push(shipped[&seen.group].per_query[*at]);
            *at += 1;
        }
        for (group, count) in &next {
            assert_eq!(
                *count,
                fused[*group].per_query.len(),
                "every scored query of {group} was seen by the router"
            );
        }
        let gain: Vec<f64> = on.iter().zip(&off).map(|(on, off)| on - off).collect();
        let n = gain.len();
        let groups: Vec<&str> = next.keys().copied().collect();

        println!(
            "\n  learned router: ridge on the gain, held out by question over {FOLDS} folds, \
             threshold chosen on the training folds"
        );
        let mean = |values: &[f64], of: &dyn Fn(usize) -> bool| {
            let picked: Vec<f64> = (0..n).filter(|at| of(*at)).map(|at| values[at]).collect();
            picked.iter().sum::<f64>() / picked.len().max(1) as f64
        };
        for group in &groups {
            let of = |at: usize| self.seen[at].group == *group;
            println!(
                "    {group}: fusion alone {:.4}, reranked {:.4}",
                mean(&off, &of),
                mean(&on, &of)
            );
        }

        let variants: [(&str, Features); 2] = [
            ("free", |seen| seen.free.to_vec()),
            ("+ small model", |seen| {
                [seen.free.as_slice(), seen.small.as_slice()].concat()
            }),
        ];
        let mut oracle: Vec<usize> = (0..n).collect();
        oracle.sort_by(|left, right| gain[*right].total_cmp(&gain[*left]));

        for (name, features) in &variants {
            let rows: Vec<Vec<f64>> = self.seen.iter().map(features).collect();
            println!("\n    {name}");
            let header: String = groups
                .iter()
                .map(|group| format!("  {group:>16}"))
                .collect();
            println!(
                "    {:>6} {:>7}{header}  {:>9} {:>9}  against reranking every query",
                "target", "reached", "recovered", "oracle"
            );
            for rate in RATES {
                let chosen = held_out_choice(&rows, &gain, &self.seen, rate);
                let routed: Vec<f64> = (0..n)
                    .map(|at| if chosen[at] { on[at] } else { off[at] })
                    .collect();
                let reached = chosen.iter().filter(|chosen| **chosen).count() as f64 / n as f64;
                let cells: String = groups
                    .iter()
                    .map(|group| {
                        let of = |at: usize| self.seen[at].group == *group;
                        format!("  {:>16.4}", mean(&routed, &of))
                    })
                    .collect();
                // The share of the whole pass's gain kept, against the
                // share a random choice at the same rate keeps: `reached`.
                let total: f64 = gain.iter().sum();
                let kept: f64 = (0..n).filter(|at| chosen[*at]).map(|at| gain[at]).sum();
                let best: f64 = oracle
                    .iter()
                    .take((reached * n as f64).round() as usize)
                    .map(|at| gain[*at])
                    .sum();
                println!(
                    "    {rate:>6.1} {reached:>7.3}{cells}  {:>9.3} {:>9.3}  {}",
                    kept / total,
                    best / total,
                    statistics::compare(&on, &routed)
                );
            }
        }
        println!("    (recovered: share of the pass's gain kept; random keeps `reached`)");
    }
}

/// Which queries the router would rerank at `rate`, each decided by a model
/// fitted without its question's fold.
fn held_out_choice(rows: &[Vec<f64>], gain: &[f64], seen: &[Seen], rate: f64) -> Vec<bool> {
    let mut chosen = vec![false; rows.len()];
    for fold in 0..FOLDS {
        let train: Vec<usize> = (0..rows.len())
            .filter(|at| seen[*at].fold != fold)
            .collect();
        let model = Ridge::fit(
            &train
                .iter()
                .map(|at| rows[*at].as_slice())
                .collect::<Vec<_>>(),
            &train.iter().map(|at| gain[*at]).collect::<Vec<_>>(),
            1.0,
        );
        let mut predicted: Vec<f64> = train.iter().map(|at| model.predict(&rows[*at])).collect();
        predicted.sort_by(f64::total_cmp);
        // Rerank what the training folds put in their top `rate`.
        let cut = ((1.0 - rate) * predicted.len() as f64).floor() as usize;
        let threshold = predicted[cut.min(predicted.len() - 1)];
        for at in (0..rows.len()).filter(|at| seen[*at].fold == fold) {
            chosen[at] = model.predict(&rows[at]) >= threshold;
        }
    }
    chosen
}

/// Stable across runs and machines, unlike the standard library's hasher.
fn fold_of(question: &str) -> u64 {
    let hash = question
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    hash % FOLDS
}

/// Ridge regression on standardised features, the intercept unpenalised.
struct Ridge {
    means: Vec<f64>,
    scales: Vec<f64>,
    weights: Vec<f64>,
    intercept: f64,
}

impl Ridge {
    fn fit(rows: &[&[f64]], targets: &[f64], lambda: f64) -> Self {
        let n = rows.len() as f64;
        let d = rows[0].len();
        let means: Vec<f64> = (0..d)
            .map(|j| rows.iter().map(|row| row[j]).sum::<f64>() / n)
            .collect();
        let scales: Vec<f64> = (0..d)
            .map(|j| {
                let variance = rows
                    .iter()
                    .map(|row| (row[j] - means[j]).powi(2))
                    .sum::<f64>()
                    / n;
                // A constant feature carries nothing; leave it at zero.
                if variance > 0.0 {
                    variance.sqrt()
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        let intercept = targets.iter().sum::<f64>() / n;
        let z = |row: &[f64], j: usize| (row[j] - means[j]) / scales[j];

        // (Z'Z + lambda I) w = Z'(y - mean y), solved by elimination.
        let mut system = vec![vec![0.0; d + 1]; d];
        for (row, target) in rows.iter().zip(targets) {
            let zs: Vec<f64> = (0..d).map(|j| z(row, j)).collect();
            for (line, zi) in system.iter_mut().zip(&zs) {
                for (cell, zj) in line.iter_mut().zip(&zs) {
                    *cell += zi * zj;
                }
                line[d] += zi * (target - intercept);
            }
        }
        for (i, line) in system.iter_mut().enumerate() {
            line[i] += lambda;
        }
        for col in 0..d {
            let pivot = (col..d)
                .max_by(|a, b| system[*a][col].abs().total_cmp(&system[*b][col].abs()))
                .expect("a column to pivot on");
            system.swap(col, pivot);
            let lead = system[col].clone();
            for (row, line) in system.iter_mut().enumerate() {
                if row != col {
                    let factor = line[col] / lead[col];
                    for (cell, from) in line.iter_mut().zip(&lead).skip(col) {
                        *cell -= factor * from;
                    }
                }
            }
        }
        let weights = (0..d).map(|i| system[i][d] / system[i][i]).collect();
        Self {
            means,
            scales,
            weights,
            intercept,
        }
    }

    fn predict(&self, row: &[f64]) -> f64 {
        self.intercept
            + (0..row.len())
                .map(|j| self.weights[j] * (row[j] - self.means[j]) / self.scales[j])
                .sum::<f64>()
    }
}

#[test]
fn ridge_recovers_a_linear_rule() {
    let rows: Vec<Vec<f64>> = (0..200)
        .map(|at| vec![f64::from(at % 7), f64::from(at % 5), 1.0])
        .collect();
    let targets: Vec<f64> = rows.iter().map(|row| 2.0 * row[0] - row[1] + 0.5).collect();
    let borrowed: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let model = Ridge::fit(&borrowed, &targets, 1e-9);
    for (row, target) in rows.iter().zip(&targets) {
        assert!((model.predict(row) - target).abs() < 1e-6);
    }
}
