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
// A fourth test binary includes these modules and does not use every helper --
// the same reason `scoring` carries this. One helper per harness is what these
// shared modules exist to undo.
#![allow(dead_code)]

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

    // The graph channel's weight, which has never been swept. It is the least
    // justified constant in the default: 1.0, equal to the vector channel's,
    // arrived at by nothing, while the only comparable published system
    // (arXiv:2609.01617) weights its graph channel at 0.15 against a dense
    // 0.50. The channel is also the only one seeded from the other three, so
    // it is the one whose candidates are least independent of theirs.
    for graph in [0.0, 0.15, 0.3, 0.5, 1.0] {
        variants.push((
            format!("graph {graph:.2}"),
            Fusion::default().with_weight(Channel::Graph, graph),
        ));
    }

    // Requiring corroboration instead of cutting the weight. The weight rows
    // above are a global constant that has to serve every group of a corpus;
    // these condition on the candidate, so they can in principle take one
    // group's gain without another's cost.
    //
    // **Two of these rows were measured once and the measurement was empty.**
    // `support graph` and `support lex+graph` came back as bit-identical
    // no-ops, and the rule was deleted partly on that reading -- when the
    // reason was that the graph channel returned no candidates at all, on
    // every corpus, because none of them had edges. That is the same premise
    // failure that hid the graph channel's weight being wrong by a factor of
    // three. The own corpus has a `relational` group now, so these two rows
    // finally ask something.
    //
    // The graph channel is also the one this rule should bite hardest on, and
    // for a structural reason rather than an empirical one: `recall_graph`
    // returns topics reached across an edge, so a graph candidate is
    // uncorroborated unless some other channel independently found it. It is
    // the only channel whose candidates are *by construction* the case the
    // rule exists for.
    for (name, needy) in [
        (
            "lex",
            vec![Channel::LexicalSegmented, Channel::LexicalNgram],
        ),
        ("seg", vec![Channel::LexicalSegmented]),
        ("ngram", vec![Channel::LexicalNgram]),
        ("graph", vec![Channel::Graph]),
        (
            "lex+graph",
            vec![
                Channel::LexicalSegmented,
                Channel::LexicalNgram,
                Channel::Graph,
            ],
        ),
    ] {
        variants.push((
            format!("support {name}"),
            Fusion::default().needing_support(needy.clone()),
        ));

        // Across the lexical weight, because at the shipped eighth the rule is
        // a measured no-op on XQuAD-R and the arithmetic says why: an
        // uncorroborated candidate is worth at most 0.0114 there and this
        // floors it at 0.0069, a difference of 0.0045 that does not reorder a
        // top ten. Where it does bite, it dominates the plain weight -- at a
        // half, +0.0206 cross-lingual nDCG and +0.0335 recall; at one, +0.1986
        // and +0.0836. The open question is whether any pair of these beats
        // the shipped point, which needs both dials moved together.
        if needy.iter().any(|channel| *channel != Channel::Graph) {
            for lexical in [0.25, 0.5, 1.0] {
                variants.push((
                    format!("support {name} lex {lexical:.2}"),
                    Fusion::default()
                        .needing_support(needy.clone())
                        .with_weight(Channel::LexicalSegmented, lexical)
                        .with_weight(Channel::LexicalNgram, lexical),
                ));
            }
        }

        // And across the graph weight wherever the graph channel is named,
        // because that weight has just moved from 1.0 to 0.30 and the rule and
        // the weight are two ways of quieting the same channel. If
        // corroboration is what the weight cut was standing in for, the graph
        // channel should be worth more than three tenths with the rule on.
        if needy.contains(&Channel::Graph) {
            for graph in [0.30, 0.50, 1.0] {
                variants.push((
                    format!("support {name} graph {graph:.2}"),
                    Fusion::default()
                        .needing_support(needy.clone())
                        .with_weight(Channel::Graph, graph),
                ));
            }
        }
    }

    variants
}

/// Which row of a sweep is the setting that ships.
///
/// Found by value rather than by label, so relabelling a row cannot silently
/// make a different setting the baseline everything is priced against.
pub fn shipped_row(variants: &[(String, Fusion)]) -> Option<usize> {
    let shipped = format!("{:?}", Fusion::default());
    variants
        .iter()
        .position(|(_, fusion)| format!("{fusion:?}") == shipped)
}

/// One group's sweep, priced as a family.
///
/// Every row against the shipped setting, with three columns the table used to
/// lack. `family p` is the Westfall-Young adjusted p across *all* the rows at
/// once -- the only p here that means what a reader will take it to mean,
/// because a table of ninety rows each with its own p is a table in which
/// about four of them clear 0.05 by chance. `can see` is the smallest mean
/// difference that row's paired differences could have detected at 80% power,
/// so a difference smaller than it is readable at a glance as noise. The
/// row's own p is still printed, and it is the less important of the two.
pub fn sweep_table<T>(
    title: &str,
    shipped: &crate::scoring::Scores,
    variants: &[(String, T)],
    rows: &[Option<&crate::scoring::Scores>],
) {
    use crate::scoring::{NDCG_AT, RECALL_AT};
    use crate::statistics;

    let present: Vec<(usize, &crate::scoring::Scores)> = rows
        .iter()
        .enumerate()
        .filter_map(|(at, row)| row.map(|row| (at, row)))
        .collect();
    let adjusted = statistics::family(
        &shipped.per_query,
        &present
            .iter()
            .map(|(_, row)| row.per_query.clone())
            .collect::<Vec<_>>(),
    );

    println!("\n  every setting against the one that ships, {title}");
    println!(
        "  setting                        nDCG@{NDCG_AT}   recall@{RECALL_AT}     diff   can see   family p   own comparison"
    );
    println!("  {}", "-".repeat(118));
    for ((at, row), family_p) in present.iter().zip(&adjusted) {
        let differences: Vec<f64> = row
            .per_query
            .iter()
            .zip(&shipped.per_query)
            .map(|(after, before)| after - before)
            .collect();
        println!(
            "  {:<28}   {:>7.4}   {:>9.4}   {:>+7.4}   {:>7.4}   {:>8.4}   {}",
            variants[*at].0,
            row.mean_ndcg(),
            row.mean_recall(),
            row.mean_ndcg() - shipped.mean_ndcg(),
            statistics::minimum_detectable(&differences),
            family_p,
            statistics::compare(&shipped.per_query, &row.per_query)
        );
    }
}

/// What choosing a setting from this sweep is worth, measured on queries the
/// choice never saw.
///
/// **The rule is fixed here, before any table is read, and it is stated so it
/// can be argued with:** maximise the mean over groups of each group's mean
/// nDCG@10, every group counting equally whatever its size. Equal weight is a
/// prior -- that no kind of query here matters more than another -- and it is
/// written down because the old procedure had one too and never said so. Ties
/// go to the shipped setting, so a sweep that finds nothing changes nothing.
/// `recall@50` is not in the rule; it is reported beside it.
///
/// Five folds stratified by group. For each, the rule sees the other four
/// folds' means and its choice is scored on the fifth. Three numbers come out,
/// and the gap between the first two is the one this repository never had:
///
/// - the best macro nDCG@10 of any row, scored on the queries that chose it --
///   what the old procedure would have reported;
/// - the procedure's macro nDCG@10 on queries it did not choose on;
/// - the shipped setting's, on the same queries, with a paired test against
///   the procedure.
pub fn cross_validated<T>(
    title: &str,
    groups: &[&str],
    shipped: &BTreeMap<String, crate::scoring::Scores>,
    variants: &[(String, T)],
    ship: Option<usize>,
    offline: &[BTreeMap<String, crate::scoring::Scores>],
) {
    use crate::statistics;

    let Some(ship) = ship else {
        println!("\n  no row of the sweep is the shipped setting, so nothing is cross-validated");
        return;
    };
    // Only groups every row scored, so the matrix is rectangular.
    let groups: Vec<&str> = groups
        .iter()
        .copied()
        .filter(|group| {
            shipped.contains_key(*group) && offline.iter().all(|row| row.contains_key(*group))
        })
        .collect();
    if groups.is_empty() {
        return;
    }

    let mut labels = Vec::new();
    for (index, group) in groups.iter().enumerate() {
        labels.extend(std::iter::repeat_n(index, shipped[*group].per_query.len()));
    }
    let matrix: Vec<Vec<f64>> = offline
        .iter()
        .map(|row| {
            groups
                .iter()
                .flat_map(|group| row[*group].per_query.iter().copied())
                .collect()
        })
        .collect();
    let baseline: Vec<f64> = groups
        .iter()
        .flat_map(|group| shipped[*group].per_query.iter().copied())
        .collect();

    let macro_mean = |per_group: &[f64]| per_group.iter().sum::<f64>() / per_group.len() as f64;
    let rule = |table: &[Vec<f64>]| -> usize {
        let mut best = ship;
        for (at, row) in table.iter().enumerate() {
            if macro_mean(row) > macro_mean(&table[best]) + 1e-12 {
                best = at;
            }
        }
        best
    };

    let by_group = |scores: &[f64]| -> Vec<f64> {
        (0..groups.len())
            .map(|group| {
                let (sum, count) = scores
                    .iter()
                    .zip(&labels)
                    .filter(|(_, label)| **label == group)
                    .fold((0.0, 0usize), |(sum, count), (score, _)| {
                        (sum + score, count + 1)
                    });
                sum / count.max(1) as f64
            })
            .collect()
    };

    let in_sample = matrix
        .iter()
        .map(|row| macro_mean(&by_group(row)))
        .fold(f64::MIN, f64::max);
    let selected =
        statistics::cross_validate(&matrix, &labels, &statistics::folds(&labels, 5), rule);
    let procedure = macro_mean(&by_group(&selected.held_out));
    let current = macro_mean(&by_group(&baseline));

    println!("\n  choosing from this sweep, cross-validated over five folds, {title}");
    println!(
        "  rule: maximise the mean over {} groups of nDCG@10, ties to what ships",
        groups.len()
    );
    println!(
        "  best row, scored on the queries that chose it   {in_sample:.4}   <- what a sweep used to report"
    );
    println!("  the procedure, on queries it did not choose on  {procedure:.4}");
    println!("  what ships, on the same queries                 {current:.4}");
    println!(
        "  optimism of choosing in-sample                  {:+.4}",
        in_sample - procedure
    );
    println!(
        "  the procedure against what ships: {}",
        statistics::compare(&baseline, &selected.held_out)
    );
    let chosen: Vec<&str> = selected
        .chosen
        .iter()
        .map(|at| variants[*at].0.as_str())
        .collect();
    println!("  chosen per fold: {chosen:?}");

    // The same rule on every query, which is what would actually ship if this
    // procedure were trusted -- and which is the row to look up in *another*
    // corpus's table to ask whether the choice transfers. A choice that
    // improves the corpus it was made on and not the next one is a property of
    // that corpus, which is what Bruch, Gai and Ingber found for per-channel
    // rank constants (`arXiv:2210.11934`).
    let everything: Vec<Vec<f64>> = matrix.iter().map(|row| by_group(row)).collect();
    println!(
        "  chosen on all of this corpus's queries: {:?}  <- look this row up in another corpus's table",
        variants[rule(&everything)].0
    );
    if selected.chosen.iter().any(|at| *at != selected.chosen[0]) {
        println!("  the folds disagree, so no single setting is a stable choice at this size");
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
