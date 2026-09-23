//! What the reranker's scores are worth under another rule for using them,
//! priced from the run that used them the shipped way.
//!
//! `search_reranked` *substitutes*: the candidates no lexical channel found
//! keep their slots and are refilled in the model's order, so fusion's opinion
//! of them is thrown away once the model has one. The standard alternative in
//! the literature is to *interpolate* -- a weighted sum of the first stage's
//! score and the model's, each normalised per query -- which keeps what the
//! first stage knew and the model could not see. Here that is the graph path
//! and the channels' agreement: a cross-encoder reads the query and one memory,
//! and a memory relevant because another one mentions it looks, to the model,
//! like a memory that is not relevant.
//!
//! **Every rule from one run.** The shipped path records every score the model
//! gave (`Why::Reranked`) and every channel's rank (`Why::Channel`), so the
//! fused order is rebuilt with the product's own `fuse` and each rule is
//! replayed over the same movable candidates with the same model scores.
//! [`replay`] first asserts that the shipped rule, replayed, reproduces the
//! engine's order position for position; only then is another rule's figure a
//! comparison rather than a reconstruction error.

// Included by harnesses that use only some of it, like the other shared
// modules here.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use pamin_core::{Channel, Fusion, Why};
use pamin_engine::SearchHit;
use pamin_index::Rerank;

/// How the two scores are put on one scale before they are summed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scale {
    /// Min-max over the movable candidates, per query. Keeps how far apart
    /// two candidates are, which is also what lets one outlier flatten the
    /// rest.
    Score,
    /// Place among the movable candidates, first 1 and last 0. Throws the
    /// distances away, and with them every question of whether the two
    /// scores' shapes are comparable.
    Rank,
}

/// A rule for turning the fused order and the model's scores into one order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// What ships: the movable candidates in the model's order.
    Substitute,
    /// `fusion * fused + (1 - fusion) * model`, both on `scale`. `fusion` of
    /// one is fusion alone, which is the `off` tier.
    Blend { fusion: f64, scale: Scale },
}

/// Every rule measured, labelled. The shipped rule is found by value, not by
/// position, by [`shipped`].
pub fn rules() -> Vec<(String, Rule)> {
    let mut rules = vec![("substitute (ships)".to_string(), Rule::Substitute)];
    for scale in [Scale::Score, Scale::Rank] {
        for fusion in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] {
            rules.push((
                format!("blend {scale:?}, fusion {fusion:.1}").to_lowercase(),
                Rule::Blend { fusion, scale },
            ));
        }
    }
    rules
}

/// Which row of [`rules`] ships.
pub fn shipped(rules: &[(String, Rule)]) -> Option<usize> {
    rules.iter().position(|(_, rule)| *rule == Rule::Substitute)
}

/// Ways of spending less on the reranker, each priced against spending all of
/// it: a cascade, where the small model scores what the pass would show and
/// the large one orders only its best `n`; and gates, where a signal read
/// before the pass decides the query does not need it at all.
///
/// **Why both.** The large model is 1.4 s of a 1.5 s search. Replacing it with
/// the small one is measured to cost accuracy -- `fast` is below no reranking
/// on same-language queries -- so the small model is only allowed to *choose*
/// here, never to order what is returned. And on XQuAD-R no reranking at all
/// was the cheapest sufficient choice for 43.4% of cross-lingual queries and
/// 98.4% of same-language ones, so a gate has room -- if anything visible
/// before the pass can find those queries, which a relative-score gate could
/// not.
///
/// One approximation, the one the pairs sweep states: the large model's scores
/// are the ones it gave in the shipped batch, and a smaller batch would score
/// a pair slightly differently. A chosen row is confirmed by a timed run.
pub struct Routes {
    labels: Vec<(String, ())>,
    measured: Vec<BTreeMap<String, crate::scoring::Scores>>,
    /// Per row, total pairs through the small model and through the large one.
    pairs: Vec<(u64, u64)>,
    queries: u64,
}

/// How many the cascade lets the large model order.
const CASCADE: [usize; 5] = [4, 6, 8, 10, 12];

/// The gates, each a rule on the fused list before the pass.
const GATES: [&str; 5] = [
    "skip: top has lexical and another channel",
    "skip: top found by three channels",
    "skip: top is first in both lexical channels",
    "skip: three channels agree, no graph find",
    "skip: first in both lexical, no graph find",
];

impl Default for Routes {
    fn default() -> Self {
        let mut labels = vec![
            ("fusion alone".to_string(), ()),
            ("accurate on all (ships)".to_string(), ()),
        ];
        labels.extend(
            CASCADE
                .iter()
                .map(|n| (format!("cascade: fast picks {n}"), ())),
        );
        labels.extend(GATES.iter().map(|gate| (gate.to_string(), ())));
        let rows = labels.len();
        Self {
            labels,
            measured: vec![BTreeMap::new(); rows],
            pairs: vec![(0, 0); rows],
            queries: 0,
        }
    }
}

impl Routes {
    /// One shipped search, its replay, and a small model to score what the
    /// pass showed. `score` scores one ranking into `group`.
    pub fn observe(
        &mut self,
        group: &str,
        hits: &[SearchHit],
        replayed: &Replayed,
        small: &mut pamin_index::Reranker,
        query: &str,
        score: impl Fn(&mut crate::scoring::Scores, &[String]),
    ) {
        self.queries += 1;
        let by_name: HashMap<&str, &SearchHit> =
            hits.iter().map(|hit| (hit.topic.as_str(), hit)).collect();
        let shown = replayed.movable.len() as u64;

        // What the list looked like before the pass, for the gates.
        let channels_of = |name: &str| -> Vec<(Channel, u32)> {
            by_name[name]
                .result
                .why
                .iter()
                .filter_map(|why| match why {
                    Why::Channel { channel, rank, .. } => Some((*channel, *rank)),
                    _ => None,
                })
                .collect()
        };
        let top = replayed
            .fused
            .first()
            .map(|name| channels_of(name))
            .unwrap_or_default();
        let lexical = |channel: &Channel| {
            matches!(channel, Channel::LexicalSegmented | Channel::LexicalNgram)
        };
        let graph_find = replayed.movable.iter().any(|at| *at >= replayed.head);
        let exact = top
            .iter()
            .filter(|(channel, rank)| lexical(channel) && *rank == 1)
            .count()
            == 2;
        let skips = [
            top.iter().any(|(channel, _)| lexical(channel)) && top.len() >= 2,
            top.len() >= 3,
            exact,
            top.len() >= 3 && !graph_find,
            // A graph find is the sign of a question that needs another
            // memory, and the pass is what brings it up -- so never skip it.
            exact && !graph_find,
        ];

        let fused = replayed.fused.clone();
        let shipped = replayed.order(Rule::Substitute);
        let mut orders: Vec<(Vec<String>, u64, u64)> =
            vec![(fused.clone(), 0, 0), (shipped.clone(), 0, shown)];

        // The small model scores exactly what the pass would show.
        let documents: Vec<String> = replayed
            .movable()
            .iter()
            .map(|name| {
                let hit = by_name[name];
                render(&hit.topic, &hit.state.content, hit.seed.as_deref(), true)
            })
            .collect();
        let mut cheap = vec![0.0f32; documents.len()];
        if !documents.is_empty() {
            let borrowed: Vec<&str> = documents.iter().map(String::as_str).collect();
            for ranked in small.rank(query, &borrowed).expect("rank") {
                cheap[ranked.position] = ranked.score;
            }
        }
        for n in CASCADE {
            let mut keep: Vec<usize> = (0..documents.len()).collect();
            keep.sort_by(|left, right| {
                cheap[*right].total_cmp(&cheap[*left]).then(left.cmp(right))
            });
            keep.truncate(n);
            keep.sort_unstable();
            let positions: Vec<usize> = keep.iter().map(|at| replayed.movable[*at]).collect();
            let mut best: Vec<usize> = (0..keep.len()).collect();
            best.sort_by(|left, right| {
                replayed.model[keep[*right]]
                    .total_cmp(&replayed.model[keep[*left]])
                    .then(left.cmp(right))
            });
            let order = if positions.len() < 2 {
                fused.clone()
            } else {
                pamin_engine::place(fused.clone(), &positions, replayed.head, &best)
            };
            orders.push((order, shown, keep.len() as u64));
        }
        for skip in skips {
            orders.push(if skip {
                (fused.clone(), 0, 0)
            } else {
                (shipped.clone(), 0, shown)
            });
        }

        for ((order, cheap_pairs, dear_pairs), (into, pairs)) in orders
            .iter()
            .zip(self.measured.iter_mut().zip(self.pairs.iter_mut()))
        {
            score(into.entry(group.to_string()).or_default(), order);
            pairs.0 += cheap_pairs;
            pairs.1 += dear_pairs;
        }
    }

    pub fn report(&self, title: &str) {
        report(title, &self.labels, Some(1), &self.measured);
        println!("  pairs a search, small model / large model");
        for ((label, _), (cheap, dear)) in self.labels.iter().zip(&self.pairs) {
            println!(
                "  {label:<44}  {:>6.1} / {:>6.1}",
                *cheap as f64 / self.queries.max(1) as f64,
                *dear as f64 / self.queries.max(1) as f64
            );
        }
        println!();
    }
}

/// Every row against the one that ships, group by group and as one
/// cross-validated choice. `measured` is indexed like `labels`.
pub fn report<T>(
    title: &str,
    labels: &[(String, T)],
    ship: Option<usize>,
    measured: &[BTreeMap<String, crate::scoring::Scores>],
) {
    let shipped = &measured[ship.expect("the shipped row is measured")];
    println!("\n  {title}");
    for (group, scores) in shipped {
        crate::channels::sweep_table(
            group,
            scores,
            labels,
            &measured
                .iter()
                .map(|row| row.get(group))
                .collect::<Vec<_>>(),
        );
    }
    crate::channels::cross_validated(
        title,
        &shipped.keys().map(String::as_str).collect::<Vec<_>>(),
        shipped,
        labels,
        ship,
        measured,
    );
    println!();
}

/// One candidate as the model is shown it: `name: content` when `named`, then
/// the seed's content when there is one. With both, the engine's own rendering,
/// which the CONTEXT arm asserts it reproduces.
fn render(name: &str, content: &str, seed: Option<&str>, named: bool) -> String {
    let mut text = if named {
        format!("{name}: {content}")
    } else {
        content.to_string()
    };
    if let Some(seed) = seed {
        text.push_str(". ");
        text.push_str(seed);
    }
    text
}

/// What the model is shown for each candidate, in the order [`in_context`]
/// returns them after `fusion alone`. [`SHIPPED`] names the one that ships.
pub const RENDERINGS: [&str; 4] = [
    "content",
    "name: content",
    "content, then the seed",
    "name: content, then the seed",
];

/// Which of [`RENDERINGS`] `search_reranked` uses: name, content and seed.
pub const SHIPPED: usize = 3;

/// The row of [`context_labels`] that ships: after `fusion alone`.
pub fn shipped_context() -> Option<usize> {
    Some(1 + SHIPPED)
}

/// The rows [`in_context`] produces: fusion alone, then each rendering.
pub fn context_labels() -> Vec<(String, ())> {
    std::iter::once("fusion alone")
        .chain(RENDERINGS)
        .map(|label| (label.to_string(), ()))
        .collect()
}

/// One query's order under each of [`context_labels`], rescoring the movable
/// candidates with `model` as each rendering presents them.
///
/// **Panics unless the shipped rendering reproduces the engine's own scores**:
/// a disagreement means the other rows are priced against a different model,
/// or different text, than the product used.
///
/// A candidate's seed is [`SearchHit::seed`], the memory the graph walked from
/// to reach it; on a corpus with no edges no candidate has one, and the seeded
/// renderings equal their unseeded counterparts. Returns how many movable
/// candidates were shown a seed.
pub fn in_context(
    hits: &[SearchHit],
    replayed: &Replayed,
    model: &mut pamin_index::Reranker,
    query: &str,
) -> (Vec<Vec<String>>, usize) {
    let content: HashMap<&str, &str> = hits
        .iter()
        .map(|hit| (hit.topic.as_str(), hit.state.content.as_str()))
        .collect();
    let seed: HashMap<&str, &str> = hits
        .iter()
        .filter_map(|hit| Some((hit.topic.as_str(), hit.seed.as_deref()?)))
        .collect();

    let movable = replayed.movable();
    let seeded = movable
        .iter()
        .filter(|name| seed.contains_key(*name))
        .count();
    let mut orders = vec![replayed.order(Rule::Blend {
        fusion: 1.0,
        scale: Scale::Rank,
    })];
    // The shipped rendering first. The model keeps a score cache keyed by
    // query and text, and renderings share most of their texts -- a candidate
    // with no seed reads the same with or without one -- so a later rendering
    // takes those candidates' scores from an earlier one's batches and sends
    // only the rest to the model. An int8 model quantizes activations per
    // batch, so that changes the scores: measured at up to half a logit. Scored
    // first, with nothing cached, the shipped rendering forms exactly the
    // engine's batches and the premise below can hold; the others are then the
    // approximation the reranking module's note already states.
    let mut rendered: Vec<Option<Vec<String>>> = vec![None; RENDERINGS.len()];
    let order = std::iter::once(SHIPPED).chain((0..RENDERINGS.len()).filter(|at| *at != SHIPPED));
    for rendering in order {
        let label = RENDERINGS[rendering];
        let (named, with_seed) = (rendering % 2 == 1, rendering >= 2);
        let documents: Vec<String> = movable
            .iter()
            .map(|name| {
                let seed = seed.get(name).copied().filter(|_| with_seed);
                render(name, content[name], seed, named)
            })
            .collect();
        let mut scores = vec![0.0; movable.len()];
        if !movable.is_empty() {
            let borrowed: Vec<&str> = documents.iter().map(String::as_str).collect();
            for ranked in model.rank(query, &borrowed).expect("rank") {
                scores[ranked.position] = f64::from(ranked.score);
            }
        }
        if rendering == SHIPPED {
            for (again, recorded) in scores.iter().zip(replayed.model()) {
                assert!(
                    (again - recorded).abs() < 1e-4,
                    "{label} scored {again} where the engine recorded {recorded} for {query:?}, \
                     so the other renderings would be priced against a different model"
                );
            }
        }
        rendered[rendering] = Some(replayed.rescored(scores).order(Rule::Substitute));
    }
    orders.extend(
        rendered
            .into_iter()
            .map(|order| order.expect("every rendering scored")),
    );
    (orders, seeded)
}

/// One query's fused order, which positions the model could move, and what it
/// gave each of them.
pub struct Replayed {
    /// Topic names in fused order.
    fused: Vec<String>,
    /// Positions in `fused` the tier is allowed to move, in fused order.
    movable: Vec<usize>,
    /// Per movable position, its fused score.
    fusion: Vec<f64>,
    /// Per movable position, the model's score.
    model: Vec<f64>,
    /// How deep the tier reads, which is where a graph find is inserted.
    head: usize,
}

/// Rebuilds one shipped search at `tier` and checks the rebuild.
///
/// **Panics unless replaying the shipped rule reproduces `hits`' order**, which
/// is the premise every other rule's figure rests on.
pub fn replay(hits: &[SearchHit], tier: Rerank) -> Replayed {
    let named: HashMap<_, &str> = hits
        .iter()
        .map(|hit| (hit.result.topic, hit.topic.as_str()))
        .collect();
    let model: HashMap<&str, f64> = hits
        .iter()
        .filter_map(|hit| {
            hit.result.why.iter().find_map(|why| match why {
                Why::Reranked { score } => Some((hit.topic.as_str(), f64::from(*score))),
                _ => None,
            })
        })
        .collect();

    let results = Fusion::default().fuse(&crate::channels::replay(hits));
    // The engine's own rule for what the model is shown, not a copy of it.
    let traces: Vec<&[Why]> = results.iter().map(|result| result.why.as_slice()).collect();
    let movable = pamin_engine::rerankable(&traces, tier);
    let fused: Vec<String> = results
        .iter()
        .map(|result| named[&result.topic].to_string())
        .collect();

    // The engine declines to rerank fewer than two, and then no candidate
    // carries a model score and every rule is fusion alone.
    let replayed = if movable.len() < 2 {
        assert!(
            model.is_empty(),
            "the model scored a query the replay says it could not move"
        );
        Replayed {
            fused,
            movable: Vec::new(),
            fusion: Vec::new(),
            model: Vec::new(),
            head: tier.depth(),
        }
    } else {
        assert_eq!(
            model.len(),
            movable.len(),
            "the model scored {} candidates and the replay found {} it could move",
            model.len(),
            movable.len()
        );
        Replayed {
            fusion: movable
                .iter()
                .map(|at| f64::from(results[*at].score))
                .collect(),
            model: movable
                .iter()
                .map(|at| model[fused[*at].as_str()])
                .collect(),
            head: tier.depth(),
            fused,
            movable,
        }
    };

    let engine: Vec<&str> = hits.iter().map(|hit| hit.topic.as_str()).collect();
    assert_eq!(
        replayed.order(Rule::Substitute),
        engine,
        "replaying the shipped rule did not reproduce the engine's order, so every other \
         rule's figure would be a reconstruction error"
    );
    replayed
}

impl Replayed {
    /// The names of the candidates the tier may move, in fused order -- what
    /// the model was shown, in the order it was shown them.
    pub fn movable(&self) -> Vec<&str> {
        self.movable
            .iter()
            .map(|at| self.fused[*at].as_str())
            .collect()
    }

    /// The model's scores as the engine recorded them, per movable candidate.
    pub fn model(&self) -> &[f64] {
        &self.model
    }

    /// The same query with the model's scores replaced, one per movable
    /// candidate -- for pricing what the model would have said had it been
    /// shown something else.
    pub fn rescored(&self, model: Vec<f64>) -> Replayed {
        assert_eq!(
            model.len(),
            self.movable.len(),
            "one score per movable candidate"
        );
        Replayed {
            fused: self.fused.clone(),
            movable: self.movable.clone(),
            fusion: self.fusion.clone(),
            model,
            head: self.head,
        }
    }

    /// The whole list under `rule`: unmovable positions stay, movable ones are
    /// refilled in the rule's order, ties to the earlier position.
    pub fn order(&self, rule: Rule) -> Vec<String> {
        let keys: Vec<f64> = match rule {
            Rule::Substitute => self.model.clone(),
            Rule::Blend { fusion, scale } => {
                let (first, second) =
                    (on(scale, &self.fusion, false), on(scale, &self.model, true));
                first
                    .iter()
                    .zip(&second)
                    .map(|(fused, model)| fusion * fused + (1.0 - fusion) * model)
                    .collect()
            }
        };
        let mut picks: Vec<usize> = (0..self.movable.len()).collect();
        picks.sort_by(|left, right| {
            keys[*right]
                .total_cmp(&keys[*left])
                .then_with(|| left.cmp(right))
        });
        pamin_engine::place(self.fused.clone(), &self.movable, self.head, &picks)
    }
}

/// `values` on `scale`, higher better. `sort` says whether they need ranking:
/// the fused scores arrive already in order, the model's do not.
fn on(scale: Scale, values: &[f64], sort: bool) -> Vec<f64> {
    let count = values.len();
    match scale {
        Scale::Score => {
            let low = values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = high - low;
            values
                .iter()
                .map(|value| {
                    if span > 0.0 {
                        (value - low) / span
                    } else {
                        0.0
                    }
                })
                .collect()
        }
        Scale::Rank => {
            let mut places: Vec<usize> = (0..count).collect();
            if sort {
                places.sort_by(|left, right| {
                    values[*right]
                        .total_cmp(&values[*left])
                        .then_with(|| left.cmp(right))
                });
            }
            let mut scaled = vec![0.0; count];
            let last = count.saturating_sub(1).max(1) as f64;
            for (place, at) in places.iter().enumerate() {
                scaled[*at] = 1.0 - place as f64 / last;
            }
            scaled
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replayed(fusion: &[f64], model: &[f64]) -> Replayed {
        Replayed {
            fused: ["lexical", "a", "b", "c"].map(String::from).to_vec(),
            movable: vec![1, 2, 3],
            fusion: fusion.to_vec(),
            model: model.to_vec(),
            head: 4,
        }
    }

    #[test]
    fn substitution_is_the_models_order_in_the_movable_slots() {
        let order = replayed(&[0.9, 0.5, 0.1], &[0.0, 1.0, 2.0]).order(Rule::Substitute);
        assert_eq!(order, ["lexical", "c", "b", "a"]);
    }

    #[test]
    fn a_blend_of_all_fusion_is_fusion_alone() {
        let order = replayed(&[0.9, 0.5, 0.1], &[0.0, 1.0, 2.0]).order(Rule::Blend {
            fusion: 1.0,
            scale: Scale::Rank,
        });
        assert_eq!(order, ["lexical", "a", "b", "c"]);
    }

    #[test]
    fn a_blend_keeps_what_fusion_was_sure_of_against_a_mild_model() {
        // The model barely prefers b; fusion strongly prefers a.
        let order = replayed(&[1.0, 0.0, 0.0], &[0.0, 0.1, -5.0]).order(Rule::Blend {
            fusion: 0.5,
            scale: Scale::Score,
        });
        assert_eq!(order, ["lexical", "a", "b", "c"]);
    }
}
