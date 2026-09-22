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
//! `Why::Channel { channel, rank, score, .. }` for every candidate of every
//! channel, unconditionally -- the weight changes the recorded contribution and
//! not whether the line is written. So one fused run carries the complete matrix
//! of "where did each channel rank each candidate, and what did it score it",
//! and every channel's own list can be rebuilt exactly as the channel returned
//! it.
//!
//! Which means the engine's own [`pamin_core::Fusion::fuse`] can be run again
//! over the rebuilt lists, at any settings, without touching the corpus.
//! [`replay`] rebuilds them and [`as_if`] fuses them, so **any fusion setting
//! this project can express is measurable offline from one pass**: a
//! leave-one-out, a weight grid, a confidence spread. A sweep row on XQuAD-R is
//! thirteen minutes; an offline row is microseconds. And because it calls the
//! shipped `fuse` rather than restating its arithmetic, there is nothing here
//! that can drift away from what the product does.
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
//! **Leave-one-out has to mean absence, and it now does.** It used not to:
//! `fuse` created an entry for every candidate of every channel before applying
//! any weight, so the fused set was always the union and a candidate only a
//! zero-weighted channel found landed at score 0.0, ordered against its peers by
//! topic UUID. The head stayed clean -- any positive score beats zero -- but
//! recall past the head counted candidates the surviving channels never
//! proposed, placed by the accident of a UUID. `Fusion::without` says the thing
//! outright now and `fuse` skips a channel weighted at zero, so leaving a
//! channel out means the same thing here and in the product.
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

use std::collections::BTreeMap;

use pamin_core::{Channel, ChannelResults, Combine, Fusion, Scored, TopicId, Why};
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

/// Rebuilds each channel's ranked list, exactly as the channel returned it.
///
/// Read out of the trace rather than re-queried, which is what makes every
/// offline variant free. Each `Why::Channel` line carries the channel, the rank
/// that channel gave this candidate, and what it scored it, so a channel's list
/// is those lines sorted by rank.
///
/// **Panics unless each channel's ranks are exactly `1..=n` with nothing
/// missing.** `Fusion::fuse` takes position for rank, so a gap would compact and
/// every rank after it would shift by one -- silently, and in a direction that
/// looks like a slightly better ranking. A gap can only come from the engine
/// dropping a candidate after fusion, which on these corpora it never does,
/// because every indexed topic resolves to a live state. If that ever changes,
/// this says so rather than returning a matrix that is off by one.
pub fn replay(hits: &[SearchHit]) -> Vec<ChannelResults> {
    let mut ranked: BTreeMap<Channel, Vec<(u32, Scored)>> = BTreeMap::new();
    for hit in hits {
        for why in &hit.result.why {
            if let Why::Channel {
                channel,
                rank,
                score,
                ..
            } = why
            {
                ranked.entry(*channel).or_default().push((
                    *rank,
                    Scored {
                        topic: hit.result.topic,
                        score: *score,
                    },
                ));
            }
        }
    }

    ranked
        .into_iter()
        .map(|(channel, mut candidates)| {
            candidates.sort_by_key(|(rank, _)| *rank);
            for (position, (rank, _)) in candidates.iter().enumerate() {
                assert_eq!(
                    *rank as usize,
                    position + 1,
                    "{channel:?} is missing rank {}: the trace has a gap, so replaying it would \
                     compact every rank after the gap and quietly improve the ranking",
                    position + 1
                );
            }
            ChannelResults::new(
                channel,
                candidates
                    .into_iter()
                    .map(|(_, candidate)| candidate)
                    .collect(),
            )
        })
        .collect()
}

/// The ranking these settings would have produced over the same candidates.
///
/// Calls the shipped [`Fusion::fuse`] on the rebuilt lists, so it is the
/// engine's arithmetic and not a copy of it -- there is nothing here to drift.
/// Names rather than identifiers, because that is what the scorers compare.
pub fn as_if(hits: &[SearchHit], fusion: &Fusion) -> Vec<String> {
    let named: BTreeMap<TopicId, &str> = hits
        .iter()
        .map(|hit| (hit.result.topic, hit.topic.as_str()))
        .collect();

    fusion
        .fuse(&replay(hits))
        .into_iter()
        .map(|result| {
            named
                .get(&result.topic)
                .unwrap_or_else(|| panic!("{:?} was fused from a trace it is not in", result.topic))
                .to_string()
        })
        .collect()
}

/// Asserts the trace rebuilds into what the engine actually ranked.
///
/// The premise every offline number rests on, and now it checks the one thing
/// that can go wrong. `as_if` runs the product's own `fuse`, so the arithmetic
/// cannot disagree; what can disagree is the reconstruction -- a rank read
/// wrong, a candidate dropped, a channel missed. Asserted position for position
/// including ties, because both sides break ties the same way now: they are the
/// same function.
///
/// `fusion` must be the settings the run was made with, or this is comparing a
/// variant against the engine's default and will fail for the right reason
/// stated wrongly.
pub fn same_as_the_engine(hits: &[SearchHit], fusion: &Fusion) {
    let engine: Vec<&str> = hits.iter().map(|hit| hit.topic.as_str()).collect();
    let ours = as_if(hits, fusion);

    assert_eq!(
        ours.len(),
        engine.len(),
        "replaying the trace produced {} results against the engine's {}",
        ours.len(),
        engine.len()
    );
    for (position, (ours, theirs)) in ours.iter().zip(&engine).enumerate() {
        assert_eq!(
            ours, theirs,
            "replaying the trace put {ours:?} at position {position} where the engine put \
             {theirs:?}, so the reconstruction is not the engine's own candidates"
        );
    }
}

/// Every fusion setting worth pricing against the one that ships, labelled.
///
/// **Read against recall as well as nDCG, and that is not a formatting
/// preference.** The accuracy gates assert both, and the first version of this
/// table printed only nDCG@10 -- on which the standardised sum is better in
/// three of four groups and was very nearly made the default. It takes
/// XQuAD-R's cross-lingual `recall@50` from 0.8960 to 0.7765, straight through
/// a floor, because a sum of standardised scores gives a candidate that sits
/// below its channel's mean a *negative* contribution where reciprocal rank
/// fusion gives every candidate a positive one. The head gains and the tail
/// collapses. A grid judged on the head alone cannot see that.
///
/// Offline, so the whole grid costs one pass over the corpus rather than one
/// pass per row. That changes what is affordable: a row on XQuAD-R used to be
/// thirteen minutes, which is why every sweep this project ever ran moved both
/// lexical channels together and left the rank constant to a coarse handful.
///
/// Four grids, because there are four open questions and they are separate.
///
/// **The lexical weights, now separable.** Kendall tau-b between the two
/// lexical channels is around 0.30 on all three corpora, so the premise that
/// justified one shared constant is refuted, and no measurement anywhere
/// distinguishes the two numbers. The grid crosses them, including an eighth
/// against an eighth, which is what ships and must come back identical.
///
/// **The confidence spread and floor.** `spread` is coupled to how many
/// candidates a channel proposes -- a standardised top score cannot exceed
/// `sqrt(n - 1)`, so 7.00 is the ceiling at the fifty each channel returns
/// here -- which is why the values run up to seven rather than stopping at the
/// two a reader might expect from a z-score. `floor` is what a channel with no
/// opinion keeps: zero silences it outright, and the rows above zero are there
/// to say whether silencing is the part that works or whether merely
/// discounting is enough.
///
/// **The combiner, which is the choice nobody here recorded making.** This
/// project argued about `k` and about the channel weights, both of them
/// parameters *of* reciprocal rank fusion, and never wrote down that fusing
/// ranks rather than normalised scores was a choice. Every 2025--2026 result
/// found goes the other way -- see [`Combine`] -- including one measured on
/// this project's own benchmarks, on CPU, without training. The rows here put
/// all three combiners on the same queries, each crossed with the confidence
/// rule, because the two mechanisms are independent: standardising fixes the
/// magnitude a rank cannot express, and confidence fixes the channel quality
/// standardising cannot express.
///
/// **The rank constant, once, to close the question.** The only real sweep of
/// it in the recent literature (`arXiv:2604.01733`, 2026, 23,088 queries) puts
/// `k = 10` ahead of the customary 60 at Recall@5 0.716 against 0.695, and
/// `MMMORRF` (SIGIR 2025) uses zero. Nothing published derives it. This
/// project's own sweep already found the curve monotonic all the way down and
/// took ten as a fifth of the channel depth rather than the boundary. These
/// rows are here to confirm the curve is flat near ten and then stop asking:
/// the literature makes `k` worth one to three points and normalisation worth
/// three to eight, so it is the low-leverage knob and it has had more attention
/// than the high-leverage one.
pub fn variants() -> Vec<(String, Fusion)> {
    let mut variants = Vec::new();

    for segmented in [0.0, 0.0625, 0.125, 0.25, 0.5] {
        for ngram in [0.0, 0.0625, 0.125, 0.25, 0.5] {
            variants.push((
                format!("lex seg {segmented:.4} ngram {ngram:.4}"),
                Fusion::default()
                    .with_weight(Channel::LexicalSegmented, segmented)
                    .with_weight(Channel::LexicalNgram, ngram),
            ));
        }
    }

    // On the shipped weights, so a gain here is the confidence rule and not a
    // smaller lexical weight wearing its name.
    for spread in [1.0, 2.0, 3.0, 5.0, 7.0] {
        for floor in [0.0, 0.25, 0.5] {
            variants.push((
                format!("conf spread {spread:.1} floor {floor:.2}"),
                Fusion::default().with_confidence(spread, floor),
            ));
        }
    }

    // Score fusion against rank fusion, and each combiner with and without the
    // confidence rule, because the two mechanisms answer different halves and
    // a row that moved both cannot say which half moved it.
    for (name, combine) in [
        ("rrf", Combine::Reciprocal),
        ("band", Combine::Banded),
        ("zsum", Combine::Standardised),
        ("zmnz", Combine::StandardisedTimesVotes),
    ] {
        variants.push((format!("combine {name}"), Fusion::default().with(combine)));
        for spread in [2.0, 5.0] {
            variants.push((
                format!("combine {name} conf {spread:.1}/0.00"),
                Fusion::default().with(combine).with_confidence(spread, 0.0),
            ));
        }
    }

    // The rank constant, to confirm the curve is flat near ten. Zero because
    // `MMMORRF` ships it and nothing here has ever tried it.
    for k in [0.0, 5.0, 10.0, 20.0, 60.0] {
        variants.push((format!("k {k:.0}"), Fusion::default().with_k(k)));
    }

    variants
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
