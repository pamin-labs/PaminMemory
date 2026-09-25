//! Whether a search found anything worth answering from.
//!
//! `pamin search` returns its best candidates whatever they are, which is
//! right for a ranking and wrong for a question nothing in memory answers: the
//! caller is handed five memories and no way to tell a strong top hit from the
//! least bad of five irrelevant ones. The channels cannot say, because none of
//! their scores means the same thing on two queries; the cross-encoder's logit
//! does not either, until it is mapped onto a probability fitted against
//! judgements.
//!
//! This module is that map and the one decision made from it. The results are
//! still returned in full -- a verdict is advice to the caller, not a filter.
//!
//! **Calibrated for one model.** The fit is of the `accurate` tier's logit;
//! `fast` is a different model with a different scale, so it gets no verdict
//! rather than a wrong one, and `off` loads no model to ask.

use pamin_index::Rerank;
use serde::{Deserialize, Serialize};

/// The calibrated probability that the top hit is relevant, and what that
/// means for a caller deciding whether to answer from it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// `P(the top hit is relevant | the query)`, on `[0, 1]`, and unlike every
    /// other score a search produces, comparable across queries.
    pub probability: f32,
    pub verdict: Verdict,
}

/// Whether the top hit is more likely relevant than not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Sufficient,
    Weak,
}

/// The tier the map below was fitted for.
pub const CALIBRATED: Rerank = Rerank::Accurate;

/// Where the verdict turns.
///
/// One half, and not a fitted value: on a calibrated probability it is the
/// point past which the top hit is more likely irrelevant than relevant, which
/// is the decision under equal costs. A cut tuned on a benchmark's own
/// questions would be a fit to that benchmark.
pub const THRESHOLD: f32 = 0.5;

/// Isotonic regression of relevance on the `accurate` tier's logit: each entry
/// is the lowest logit of a block and the share of top hits in that block that
/// were relevant.
///
/// Fitted on the top hits `pamin search --limit 10` returned for 484 MuSiQue
/// answerable dev questions (every fifth, 5,964 paragraphs pooled into one
/// project), labelled relevant when the top hit is a supporting paragraph;
/// 74.4% were. Held-out ECE, fitted on one half by question and scored on the
/// other: 0.0590, against 0.1804 for the raw sigmoid.
const CALIBRATION: &[(f32, f32)] = &[
    (-9.2595, 0.0000),
    (-6.8309, 0.3723),
    (-2.3074, 0.5000),
    (-2.0781, 0.5641),
    (-0.9588, 0.6000),
    (-0.8725, 0.6250),
    (-0.7251, 0.6667),
    (-0.6442, 0.7619),
    (0.9408, 0.8947),
    (2.3952, 0.9630),
    (3.6862, 0.9890),
    (5.9135, 1.0000),
];

impl Evidence {
    /// What a top hit the calibrated tier scored at `logit` is worth.
    pub fn of(logit: f32) -> Self {
        let probability = calibrated(logit);
        Self {
            probability,
            verdict: if probability < THRESHOLD {
                Verdict::Weak
            } else {
                Verdict::Sufficient
            },
        }
    }
}

/// The last block starting at or below `logit`; below every block, the first.
fn calibrated(logit: f32) -> f32 {
    if CALIBRATION.is_empty() {
        return 1.0 / (1.0 + (-logit).exp());
    }
    let at = CALIBRATION.partition_point(|(lowest, _)| *lowest <= logit);
    CALIBRATION[at.saturating_sub(1)].1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An isotonic fit is a non-decreasing step function over strictly
    /// increasing boundaries. A table pasted out of order, or edited by hand,
    /// breaks the lookup's binary search silently -- this is what notices.
    #[test]
    fn the_table_is_a_monotone_step_function() {
        assert!(!CALIBRATION.is_empty());
        for pair in CALIBRATION.windows(2) {
            assert!(pair[0].0 < pair[1].0, "boundaries out of order: {pair:?}");
            assert!(pair[0].1 <= pair[1].1, "probability falls: {pair:?}");
        }
        for (_, probability) in CALIBRATION {
            assert!((0.0..=1.0).contains(probability));
        }
    }

    #[test]
    fn each_logit_reads_the_block_it_falls_in() {
        let (first, lowest) = CALIBRATION[0];
        assert_eq!(calibrated(first - 100.0), lowest);
        for (at, (boundary, probability)) in CALIBRATION.iter().enumerate() {
            assert_eq!(calibrated(*boundary), *probability);
            if let Some((next, _)) = CALIBRATION.get(at + 1) {
                assert_eq!(calibrated((boundary + next) / 2.0), *probability);
            }
        }
        let (last, highest) = CALIBRATION[CALIBRATION.len() - 1];
        assert_eq!(calibrated(last + 100.0), highest);
    }

    #[test]
    fn the_verdict_turns_at_the_threshold() {
        for (boundary, probability) in CALIBRATION {
            let evidence = Evidence::of(*boundary);
            assert_eq!(evidence.probability, *probability);
            let expected = if *probability < THRESHOLD {
                Verdict::Weak
            } else {
                Verdict::Sufficient
            };
            assert_eq!(evidence.verdict, expected);
        }
    }
}
