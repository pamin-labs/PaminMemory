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

/// Every rule against the one that ships, group by group and as one
/// cross-validated choice. `measured` is indexed like [`rules`].
pub fn report(title: &str, measured: &[BTreeMap<String, crate::scoring::Scores>]) {
    let rules = rules();
    let ship = shipped(&rules);
    let shipped = &measured[ship.expect("the shipped rule is measured")];
    println!("\n  rules for the reranker's scores, {title}");
    for (group, scores) in shipped {
        crate::channels::sweep_table(
            group,
            scores,
            &rules,
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
        &rules,
        ship,
        measured,
    );
    println!();
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
    let head = tier.depth().min(results.len());
    let movable: Vec<usize> = (0..head)
        .filter(|at| {
            !results[*at].why.iter().any(|why| {
                matches!(
                    why,
                    Why::Channel { channel, .. }
                        if *channel == Channel::LexicalSegmented
                            || *channel == Channel::LexicalNgram
                )
            })
        })
        .collect();
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
        let mut order = self.fused.clone();
        for (slot, pick) in self.movable.iter().zip(&picks) {
            order[*slot] = self.fused[self.movable[*pick]].clone();
        }
        order
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
