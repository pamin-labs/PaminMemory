//! What each channel was worth on its own, out of one fused run.
//!
//! The published prescription for weighting a fused channel is to weight it by
//! its own standalone quality (`arXiv:2508.01405`, 2025 -- the only
//! four-channel analysis there is, and the one that names the "weakest link"
//! failure where rank fusion reads coincidental agreement as confirmation and
//! loses eighteen nDCG points to it). This project has never had that figure
//! for any channel. Every number it reports is of the four fused.
//!
//! It turns out not to need new runs. [`pamin_core::Fusion::fuse`] writes a
//! `Why::Channel { channel, rank, .. }` for every candidate of every channel,
//! unconditionally -- the weight changes the recorded contribution and not
//! whether the line is written. So one fused run carries the complete matrix of
//! "where did each channel rank each candidate", and every channel's own
//! ordering can be read back out of it.
//!
//! ## The two things that make this exact rather than approximate
//!
//! **The limit has to be wide enough that nothing was truncated.** The engine
//! ends with `take(limit)` over a list that can hold the union of four fifty-deep
//! channels. At the harnesses' usual limit of 51 the truncation is not
//! hypothetical: at the shipped eighth weight a lexical candidate ranked *first*
//! scores `0.125 / (10 + 1)` = 0.0114 while a vector candidate ranked *fiftieth*
//! scores `1.0 / (10 + 50)` = 0.0167, so a lexical channel's entire head can sit
//! below fifty vector results and fall off the end. [`enough_room`] asserts the
//! returned list was shorter than the limit, which is the only way to know the
//! matrix is whole.
//!
//! **Leave-one-out has to mean absence, and it now does on both sides.** It
//! used not to: `fuse` created an entry for every candidate of every channel
//! before applying any weight, so the fused set was always the union and a
//! candidate only a zero-weighted channel found landed at score 0.0, ordered
//! against its peers by topic UUID. The head stayed clean -- any positive score
//! beats zero -- but recall past the head counted candidates the surviving
//! channels never proposed. `Fusion::without` says the thing outright now and
//! `fuse` skips a channel weighted at zero, so the engine and [`refuse`] agree
//! about what leaving a channel out means. [`refuse`] still takes a weight
//! function returning `None` rather than `Some(0.0)`, because the two readings
//! are equal in effect and only one of them says which was meant.
//!
//! ## What "the graph channel alone" cannot mean
//!
//! The graph channel's candidates are an expansion from seeds taken out of the
//! other three channels' lists, and the seeding does not read weights. So it has
//! no standalone existence to measure: its ranking here is "what the walk
//! reached from what the others found", which is the only thing it is ever. The
//! weight-by-standalone-quality prescription does not apply to it in the sense
//! its authors meant, and saying so is more useful than printing a number that
//! looks comparable and is not.

use std::collections::{BTreeMap, BTreeSet};

use pamin_core::{Channel, Why};
use pamin_engine::SearchHit;

/// Every channel that returned anything, and the order it returned it in.
///
/// Keyed by channel; each value is topic names in that channel's own rank
/// order. Ranks are unique within a channel -- they are positions in a list --
/// so there is no tie to break and the reconstruction is exact rather than
/// stable-by-convention.
pub fn each_alone(hits: &[SearchHit]) -> BTreeMap<Channel, Vec<String>> {
    let mut ranked: BTreeMap<Channel, Vec<(u32, String)>> = BTreeMap::new();
    for hit in hits {
        for why in &hit.result.why {
            if let Why::Channel { channel, rank, .. } = why {
                ranked
                    .entry(*channel)
                    .or_default()
                    .push((*rank, hit.topic.clone()));
            }
        }
    }

    ranked
        .into_iter()
        .map(|(channel, mut candidates)| {
            candidates.sort_by_key(|(rank, _)| *rank);
            (
                channel,
                candidates.into_iter().map(|(_, topic)| topic).collect(),
            )
        })
        .collect()
}

/// Re-fuses the recorded ranks, leaving out whatever the weights call absent.
///
/// `weight` returns `None` for a channel that was never asked and `Some(w)` for
/// one that was. The difference matters and the engine cannot express it -- see
/// the module notes.
///
/// This duplicates four lines of `Fusion::fuse`'s arithmetic, which is a thing
/// that drifts. [`same_as_the_engine`] is the guard: it re-fuses with every
/// channel at the weights the run used and requires the result to reproduce the
/// engine's own ordering exactly. Nothing derived from `refuse` is worth
/// reading unless that assertion holds.
pub fn refuse(hits: &[SearchHit], k: f32, weight: impl Fn(Channel) -> Option<f32>) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = hits
        .iter()
        .map(|hit| {
            let score: f32 = hit
                .result
                .why
                .iter()
                .filter_map(|why| match why {
                    Why::Channel { channel, rank, .. } => {
                        weight(*channel).map(|weight| weight / (k + *rank as f32))
                    }
                    _ => None,
                })
                .sum();
            (score, hit.topic.as_str())
        })
        // A candidate no surviving channel proposed is not in the list at all,
        // which is the whole point of distinguishing absent from zero.
        .filter(|(score, _)| *score > 0.0)
        .collect();

    // Score descending, then by name, mirroring `pamin_core::sort_results`'
    // descending-score-then-identity. The engine breaks ties on `TopicId` and
    // this breaks them on the name, which is why `same_as_the_engine` compares
    // orderings on a run whose scores separate everything it ranks.
    scored.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(right.1))
    });
    scored
        .into_iter()
        .map(|(_, topic)| topic.to_string())
        .collect()
}

/// Asserts the returned list was not truncated, so the rank matrix is whole.
///
/// Panics with the numbers rather than returning a bool, because every figure
/// downstream of a truncated matrix is wrong in a way that looks plausible: a
/// channel whose head was cut simply appears to have a worse head.
pub fn enough_room(hits: &[SearchHit], limit: u32) {
    assert!(
        (hits.len() as u32) < limit,
        "the fused list came back at the limit ({} of {limit}), so the engine truncated it and \
         each channel's ranking is missing however much fell off the end. Raise the limit past \
         four times the channel depth",
        hits.len()
    );
}

/// Asserts that re-fusing the recorded ranks reproduces what the engine ranked.
///
/// The premise every leave-one-out number rests on. `weights` must be the ones
/// the run was made with, and every channel present in the trace must appear in
/// it -- a channel left out here is being called absent, which is the one thing
/// this assertion cannot be asked to check.
pub fn same_as_the_engine(hits: &[SearchHit], k: f32, weights: &BTreeMap<Channel, f32>) {
    let present: BTreeSet<Channel> = hits
        .iter()
        .flat_map(|hit| &hit.result.why)
        .filter_map(|why| match why {
            Why::Channel { channel, .. } => Some(*channel),
            _ => None,
        })
        .collect();
    for channel in &present {
        assert!(
            weights.contains_key(channel),
            "{channel:?} appears in the trace but not in the weights this check was given, so \
             the check would be comparing a leave-one-out against the engine's full fusion"
        );
    }

    let engine: Vec<&str> = hits.iter().map(|hit| hit.topic.as_str()).collect();
    let ours = refuse(hits, k, |channel| weights.get(&channel).copied());

    assert_eq!(
        ours.len(),
        engine.len(),
        "re-fusing the trace produced {} results against the engine's {}",
        ours.len(),
        engine.len()
    );
    // Compared as a set of scores rather than position for position, because
    // the two break ties differently -- the engine on `TopicId`, this on the
    // name. Where scores separate the candidates the orders agree exactly;
    // where they do not, neither order means anything.
    for (position, (ours, theirs)) in ours.iter().zip(&engine).enumerate() {
        if ours != theirs {
            let tied = tie_at(hits, k, weights, position);
            assert!(
                tied,
                "re-fusing the trace put {ours:?} at position {position} where the engine put \
                 {theirs:?}, and their scores differ -- the arithmetic here has drifted from \
                 `Fusion::fuse`"
            );
        }
    }
}

/// Whether the two candidates a position disagrees about scored the same.
fn tie_at(hits: &[SearchHit], k: f32, weights: &BTreeMap<Channel, f32>, position: usize) -> bool {
    let score = |topic: &str| -> f32 {
        hits.iter()
            .find(|hit| hit.topic == topic)
            .map(|hit| {
                hit.result
                    .why
                    .iter()
                    .filter_map(|why| match why {
                        Why::Channel { channel, rank, .. } => weights
                            .get(channel)
                            .map(|weight| weight / (k + *rank as f32)),
                        _ => None,
                    })
                    .sum()
            })
            .unwrap_or(0.0)
    };

    let ours = refuse(hits, k, |channel| weights.get(&channel).copied());
    match (ours.get(position), hits.get(position)) {
        (Some(ours), Some(theirs)) => (score(ours) - score(&theirs.topic)).abs() < 1e-9,
        _ => false,
    }
}

/// Kendall's tau-b between two channels' orderings.
///
/// Computed over the candidates both channels returned, because a channel
/// cannot be said to disagree about something it never saw. The question it
/// answers is whether the two lexical channels are one signal counted twice:
/// they run BM25 over the same text, once over segmented words and once over
/// character n-grams, so they agree by construction rather than by independent
/// confirmation -- and rank fusion rewards agreement.
///
/// Tau-b rather than tau-a so that the shared-candidate count, which differs by
/// query, does not make the figures incomparable. There are no ties to correct
/// for within a single channel's ranking, so the correction only ever divides
/// by the number of comparable pairs.
pub fn agreement(left: &[String], right: &[String]) -> Option<f64> {
    fn place(list: &[String]) -> BTreeMap<&str, usize> {
        list.iter()
            .enumerate()
            .map(|(position, topic)| (topic.as_str(), position))
            .collect()
    }
    let (first, second) = (place(left), place(right));

    let shared: Vec<&str> = first
        .keys()
        .filter(|topic| second.contains_key(*topic))
        .copied()
        .collect();
    if shared.len() < 2 {
        return None;
    }

    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for (index, earlier) in shared.iter().enumerate() {
        for later in &shared[index + 1..] {
            let ours = first[*earlier].cmp(&first[*later]);
            let theirs = second[*earlier].cmp(&second[*later]);
            if ours == theirs {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }

    let pairs = concordant + discordant;
    (pairs > 0).then(|| (concordant - discordant) as f64 / pairs as f64)
}

/// The mean of whatever was collected, or zero when nothing was.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_orderings_agree_completely() {
        let list: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        assert_eq!(agreement(&list, &list), Some(1.0));
    }

    #[test]
    fn reversed_orderings_disagree_completely() {
        let forward: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let backward: Vec<String> = forward.iter().rev().cloned().collect();
        assert_eq!(agreement(&forward, &backward), Some(-1.0));
    }

    #[test]
    fn only_the_candidates_both_returned_are_compared() {
        // The second channel never saw `c` or `d`. Over what they share, `a`
        // before `b` in both, so they agree -- a channel cannot be said to
        // disagree about something it never returned.
        let left: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let right: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(agreement(&left, &right), Some(1.0));
    }

    #[test]
    fn one_shared_candidate_is_not_a_correlation() {
        let left: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let right: Vec<String> = ["a", "z"].iter().map(|s| s.to_string()).collect();
        assert_eq!(agreement(&left, &right), None);
    }

    #[test]
    fn one_swap_in_four_is_most_of_the_way_to_agreement() {
        // Six pairs, one discordant: (6 - 2*1 - ... ) -- concordant 5,
        // discordant 1, so (5 - 1) / 6.
        let left: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let right: Vec<String> = ["b", "a", "c", "d"].iter().map(|s| s.to_string()).collect();
        let tau = agreement(&left, &right).expect("four shared candidates");
        assert!((tau - 4.0 / 6.0).abs() < 1e-12, "tau was {tau}");
    }
}
