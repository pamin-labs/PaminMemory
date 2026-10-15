//! What the reranker's scores are worth under another rule for using them,
//! priced from the run that used them the shipped way.
//!
//! The accurate tier now blends the full head's fused and model scores.
//! The alternative is to substitute the model's order, throwing away the first stage's
//! opinion. The blend is a weighted sum of the first stage's
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

mod router;
pub use router::Router;

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
    /// The shipped blend, plus `graph` times the candidate's graph-channel
    /// score for candidates the graph reached. The graph score is calibrated,
    /// so it is added as it stands: a cross-encoder reads one candidate at a
    /// time and cannot see that it is relevant because another memory names
    /// it, which is what this keeps after the pass.
    GraphBlend { graph: f64 },
    /// The shipped blend over only the first `n` movable candidates in fused
    /// order; the rest keep the place fusion gave them. What a shallower depth
    /// would have cost, priced without rescoring, so two approximations:
    /// - the INT8 model's score for a pair depends slightly on its batch
    ///   neighbours, and the shipped pass's scores are reused;
    /// - a real depth of `n` would still insert strong graph-only candidates
    ///   from below `n`, which this drops, so on graph-bearing corpora it
    ///   understates a shallow pass.
    ///
    /// Neither applies to the full pass, and the second not to a corpus
    /// without edges.
    Prefix { n: usize },
    /// [`Rule::Prefix`] with `n` only when the fused list looks settled -- its
    /// first candidate leads the second by at least `lead` of its score, and
    /// both a lexical channel and the vector channel rank it in their top
    /// three -- and the shipped pass otherwise. `n` of zero skips the model.
    Confident { lead: f64, n: usize },
    /// Score the shortlist in the passes the reranker runs -- its own
    /// planner, shortest pairs first -- and stop after the first pass past
    /// the second that put nothing into the model's top `top` so far: the
    /// last pass says the pairs still to come are behind in the model too.
    /// An anytime stop, not a safe one -- a cross-encoder's score has no bound
    /// to stop against -- and an order by length, not by fusion, since that is
    /// the order the passes run in.
    Patience { top: usize },
}

/// Every rule measured, labelled. The shipped rule is found by value, not by
/// position, by [`shipped`].
pub fn rules() -> Vec<(String, Rule)> {
    let mut rules = vec![("substitute".to_string(), Rule::Substitute)];
    for scale in [Scale::Score, Scale::Rank] {
        for fusion in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] {
            rules.push((
                format!(
                    "blend {scale:?}, fusion {fusion:.1}{}",
                    if scale == Scale::Score && fusion == pamin_engine::RERANK_FUSION {
                        " (ships)"
                    } else {
                        ""
                    }
                )
                .to_lowercase(),
                Rule::Blend { fusion, scale },
            ));
        }
    }
    for graph in [0.05, 0.1, 0.2, 0.3, 0.5] {
        rules.push((
            format!("shipped blend + graph {graph:.2}"),
            Rule::GraphBlend { graph },
        ));
    }
    for n in [4, 8, 12, 16, 20] {
        rules.push((format!("shipped blend, first {n} only"), Rule::Prefix { n }));
    }
    for lead in [0.05, 0.1, 0.2] {
        for n in [0, 8] {
            rules.push((
                format!("settled (lead {lead:.2}): first {n}, else shipped"),
                Rule::Confident { lead, n },
            ));
        }
    }
    for top in [3, 5, 10] {
        rules.push((
            format!("patience: stop when a pass misses the top {top}"),
            Rule::Patience { top },
        ));
    }
    rules
}

/// Which row of [`rules`] ships.
pub fn shipped(rules: &[(String, Rule)]) -> Option<usize> {
    rules
        .iter()
        .position(|(_, rule)| *rule == shipped_rule(Rerank::default()))
}

fn shipped_rule(tier: Rerank) -> Rule {
    match tier {
        Rerank::Accurate => Rule::Blend {
            fusion: pamin_engine::RERANK_FUSION,
            scale: Scale::Score,
        },
        Rerank::Fast | Rerank::Off => Rule::Substitute,
    }
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
    router: Router,
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

fn cascade_order(replayed: &Replayed, keep: &[usize]) -> Vec<String> {
    if keep.len() < 2 {
        return replayed.fused.clone();
    }
    let positions: Vec<usize> = keep.iter().map(|at| replayed.movable[*at]).collect();
    let (fused, model): (Vec<f64>, Vec<f64>) = keep
        .iter()
        .map(|at| (replayed.fusion[*at], replayed.model[*at]))
        .unzip();
    let best = pamin_engine::score_blend_order(&fused, &model, pamin_engine::RERANK_FUSION);
    pamin_engine::place(replayed.fused.clone(), &positions, replayed.head, &best)
}

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
        crate::harness::shape::tables(labels.iter().map(|(label, _)| label));
        let rows = labels.len();
        Self {
            labels,
            measured: vec![BTreeMap::new(); rows],
            pairs: vec![(0, 0); rows],
            queries: 0,
            router: Router::default(),
        }
    }
}

impl Routes {
    /// One shipped search, its replay, and a small model to score what the
    /// pass showed. `score` scores one ranking into `group`; `question` names
    /// what is asked, so the learned router holds every asking of it out
    /// together.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &mut self,
        group: &str,
        question: &str,
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
        let shipped = replayed.order(shipped_rule(Rerank::default()));
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
        self.router
            .observe(group, question, query, hits, replayed, &cheap);
        for n in CASCADE {
            let mut keep: Vec<usize> = (0..documents.len()).collect();
            keep.sort_by(|left, right| {
                cheap[*right].total_cmp(&cheap[*left]).then(left.cmp(right))
            });
            keep.truncate(n);
            keep.sort_unstable();
            let order = cascade_order(replayed, &keep);
            orders.push((order, shown, keep.len() as u64));
        }
        for skip in skips {
            orders.push(if skip {
                (fused.clone(), 0, 0)
            } else {
                (shipped.clone(), 0, shown)
            });
        }

        for (((order, cheap_pairs, dear_pairs), (into, pairs)), (label, _)) in orders
            .iter()
            .zip(self.measured.iter_mut().zip(self.pairs.iter_mut()))
            .zip(&self.labels)
        {
            crate::harness::shape::table(label);
            score(into.entry(group.to_string()).or_default(), order);
            pairs.0 += cheap_pairs;
            pairs.1 += dear_pairs;
        }
    }

    pub fn report(&self, title: &str) {
        report(
            title,
            &self.labels,
            Some(1),
            &self.measured,
            crate::channels::Units::Questions,
        );
        println!("  pairs a search, small model / large model");
        for ((label, _), (cheap, dear)) in self.labels.iter().zip(&self.pairs) {
            println!(
                "  {label:<44}  {:>6.1} / {:>6.1}",
                *cheap as f64 / self.queries.max(1) as f64,
                *dear as f64 / self.queries.max(1) as f64
            );
        }
        self.router.report(&self.measured[0], &self.measured[1]);
        println!();
    }
}

/// Every rule in [`rules`], measured over shipped searches one at a time, with
/// what each costs in model pairs.
///
/// With `SUFFICIENCY_OUT` set, also writes one row a query -- group, whether
/// the shipped first result is relevant, the model's best score and its lead
/// over the second -- to that file, for asking whether those two numbers
/// predict a right first answer before anything exposes them. The scores are
/// the ones the product computed in its own batches, which is the condition
/// the withdrawn abstention experiment did not hold.
pub struct Priced {
    rules: Vec<(String, Rule)>,
    measured: Vec<BTreeMap<String, crate::scoring::Scores>>,
    pairs: Vec<u64>,
    queries: u64,
    /// How many of `rules` price the shipped model's own scores; the rest
    /// price `rescore`'s.
    firsts: usize,
    sufficiency: Option<std::io::BufWriter<std::fs::File>>,
    /// `SHORTLIST_OUT`: one JSON line a search with what the model was shown,
    /// for scoring the same shortlists with another model offline.
    shortlist: Option<std::io::BufWriter<std::fs::File>>,
    /// `RESCORE_IN`: another model's scores for those shortlists, one line a
    /// search in the order they were written, priced as extra rows.
    rescore: Option<Vec<Rescored>>,
    /// Per group, the unit each scored query belongs to, in `per_query`
    /// order, for the cross-validated choice and its paired test.
    clusters: BTreeMap<String, Vec<String>>,
    /// The shipped tier's reranker, loaded for its tokenizer and planner
    /// only, so [`Rule::Patience`] replays the passes the engine ran.
    reranker: pamin_index::Reranker,
    /// Searches checked by [`planned_like`](Self::planned_like), and why
    /// the replay's plan was not the engine's, if it ever was not.
    checked: u64,
    unlike: Option<String>,
}

/// The shape table `SUFFICIENCY_OUT`'s first-result check is scored into.
const SUFFICIENCY_TABLE: &str = "sufficiency: shipped first result";

/// Another model's scores for one dumped shortlist.
#[derive(serde::Deserialize)]
pub struct Rescored {
    /// Which model scored it, with revision and precision; one per file.
    model: String,
    query: String,
    names: Vec<String>,
    /// The texts the other model scored, as dumped, so a stale file whose
    /// names still match cannot be priced against changed documents.
    documents: Vec<String>,
    scores: Vec<f64>,
}

/// The rows priced over `model`'s scores when `RESCORE_IN` is set.
fn rescored_rules(model: &str) -> Vec<(String, Rule)> {
    vec![
        (
            format!("rescored by {model}: shipped blend"),
            shipped_rule(Rerank::default()),
        ),
        (format!("rescored by {model}: substitute"), Rule::Substitute),
    ]
}

impl Priced {
    /// `models` is the workspace's model cache, where the shipped tier's
    /// reranker is loaded from.
    pub fn new(models: &std::path::Path) -> Self {
        let reranker = pamin_index::Reranker::load(Rerank::default(), models)
            .expect("load the shipped reranker for its planner");
        let mut rules = rules();
        let shortlist = std::env::var("SHORTLIST_OUT").ok().map(|path| {
            std::io::BufWriter::new(
                std::fs::File::create(path).expect("create the shortlist output"),
            )
        });
        let rescore = std::env::var("RESCORE_IN").ok().map(|path| {
            let text = std::fs::read_to_string(path).expect("read the rescored shortlists");
            text.lines()
                .map(|line| serde_json::from_str(line).expect("parse a rescored shortlist"))
                .collect::<Vec<Rescored>>()
        });
        let firsts = rules.len();
        if let Some(rescore) = &rescore {
            let model = &rescore.first().expect("rescored shortlists").model;
            assert!(
                rescore.iter().all(|line| &line.model == model),
                "one RESCORE_IN file holds one model's scores"
            );
            rules.extend(rescored_rules(model));
        }
        let sufficiency = std::env::var("SUFFICIENCY_OUT").ok().map(|path| {
            let mut file = std::io::BufWriter::new(
                std::fs::File::create(path).expect("create the sufficiency output"),
            );
            use std::io::Write;
            writeln!(file, "group\tfirst_relevant\tbest\tgap\tscored").expect("write the header");
            file
        });
        crate::harness::shape::tables(rules.iter().map(|(label, _)| label));
        if sufficiency.is_some() {
            crate::harness::shape::tables([SUFFICIENCY_TABLE]);
        }
        Self {
            measured: vec![BTreeMap::new(); rules.len()],
            pairs: vec![0; rules.len()],
            queries: 0,
            rules,
            firsts,
            sufficiency,
            shortlist,
            rescore,
            clusters: BTreeMap::new(),
            reranker,
            checked: 0,
            unlike: None,
        }
    }

    /// One shipped search. `score` scores one ranking into its group or
    /// groups; `group` only labels the sufficiency row.
    ///
    /// `cluster` names the unit the search belongs to: searches that share
    /// one share a fold, and the paired test runs over each unit's mean. A
    /// LOCOMO conversation is one, since its questions share a corpus, index
    /// and graph. `None` makes the search its own unit, which also keeps the
    /// answer keys one search is scored under together.
    pub fn observe(
        &mut self,
        group: &str,
        query: &str,
        hits: &[SearchHit],
        replayed: &Replayed,
        cluster: Option<&str>,
        score: impl Fn(&mut BTreeMap<String, crate::scoring::Scores>, &[String]),
    ) {
        let at = self.queries as usize;
        let cluster = cluster.map_or_else(|| format!("search {at}"), str::to_string);
        let before: BTreeMap<String, usize> = self.measured[0]
            .iter()
            .map(|(group, scores)| (group.clone(), scores.per_query.len()))
            .collect();
        self.queries += 1;
        if let Some(file) = &mut self.shortlist {
            use std::io::Write;
            let names = replayed.movable();
            let documents = shown(hits, &names);
            let line = serde_json::json!({
                "group": group,
                "query": query,
                "names": names,
                "documents": documents,
                "model": replayed.model(),
            });
            writeln!(file, "{line}").expect("write a shortlist");
        }
        let rescored = self.rescore.as_ref().map(|rescore| {
            let other = rescore
                .get(at)
                .expect("a rescored shortlist for every search");
            assert_eq!(other.query, query, "rescored shortlists are out of step");
            assert_eq!(
                other.names,
                replayed.movable(),
                "the rescored shortlist is not the one this search showed the model"
            );
            assert_eq!(
                other.documents,
                shown(hits, &replayed.movable()),
                "the rescored documents are not the ones this search showed the model"
            );
            replayed.rescored(other.scores.clone())
        });
        let names = replayed.movable();
        let documents = shown(hits, &names);
        let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
        let planned =
            replayed
                .rescored(replayed.model.clone())
                .planned(&self.reranker, query, &documents);
        for (row, (((label, rule), into), pairs)) in self
            .rules
            .iter()
            .zip(&mut self.measured)
            .zip(&mut self.pairs)
            .enumerate()
        {
            let replayed = if row < self.firsts {
                &planned
            } else {
                rescored
                    .as_ref()
                    .expect("rescored rows only with RESCORE_IN")
            };
            crate::harness::shape::table(label);
            score(into, &replayed.order(*rule));
            *pairs += replayed.pairs(*rule) as u64;
        }
        // Every row scores the same groups, so the first says which grew.
        for (group, scores) in &self.measured[0] {
            let grown = scores.per_query.len() - before.get(group).copied().unwrap_or(0);
            self.clusters
                .entry(group.clone())
                .or_default()
                .extend(std::iter::repeat_n(cluster.clone(), grown));
        }
        if let Some(file) = &mut self.sufficiency {
            use std::io::Write;
            let shipped = replayed.order(shipped_rule(Rerank::default()));
            // A one-result list scores above zero exactly when that result is
            // relevant, whatever the corpus's own judgements are.
            let mut first = BTreeMap::new();
            crate::harness::shape::table(SUFFICIENCY_TABLE);
            score(&mut first, &shipped[..shipped.len().min(1)]);
            let first = first.values().any(|scores| scores.ndcg > 0.0);
            let mut model = replayed.model.clone();
            model.sort_by(|left, right| right.total_cmp(left));
            let (best, gap) = match model.as_slice() {
                [best, second, ..] => (*best, best - second),
                [best] => (*best, 0.0),
                [] => (f64::NAN, f64::NAN),
            };
            writeln!(
                file,
                "{group}\t{}\t{best}\t{gap}\t{}",
                u8::from(first),
                u8::from(!model.is_empty())
            )
            .expect("write a sufficiency row");
        }
    }

    /// Checks, after a search `engine` ran, that the replay plans passes as
    /// the engine's reranker does: the same tier on the same execution
    /// target (an NPU's provider and id included). The planner is the same code; what can differ is the
    /// model's batch limits, and those follow from the tier and the provider
    /// (a compiled CoreML session has its own) under this process's
    /// settings. A reranker that fell back mid-run -- CoreML to the CPU, say
    /// -- plans differently from the replay, and the Patience rows are then
    /// not reported rather than reported with the wrong pair counts.
    pub fn planned_like(&mut self, engine: &pamin_engine::Engine) {
        self.checked += 1;
        if self.unlike.is_some() {
            return;
        }
        let role = format!("reranking {}", self.reranker.tier().name());
        let replay = self.reranker.target();
        // Every plan the engine's reranker ran on so far: one, unless it fell
        // back mid-run, and then its passes were planned two ways.
        let mut ran: Vec<String> = engine
            .providers()
            .into_iter()
            .filter(|(name, _)| *name == role)
            .map(|(_, plan)| plan.target)
            .collect();
        ran.dedup();
        if ran != [replay.clone()] {
            self.unlike = Some(format!(
                "the engine's {role} ran on {}, and the replay plans for {replay}",
                if ran.is_empty() {
                    "nothing".to_string()
                } else {
                    ran.join(" then ")
                },
            ));
        }
    }

    pub fn report(&mut self, title: &str) {
        assert_eq!(
            self.checked, self.queries,
            "every search is checked with planned_like before Patience is priced"
        );
        if let Some(unlike) = &self.unlike {
            // Refused, not reported: the rows are scored, so the run's
            // coverage still holds, but their pair counts would be wrong.
            println!(
                "\n  patience rows not reported: {unlike}, so the passes they replay are not the ones that ran"
            );
            let keep: Vec<bool> = self
                .rules
                .iter()
                .map(|(_, rule)| !matches!(rule, Rule::Patience { .. }))
                .collect();
            fn kept<T>(values: &mut Vec<T>, keep: &[bool]) {
                let mut at = 0;
                values.retain(|_| {
                    at += 1;
                    keep[at - 1]
                });
            }
            self.firsts -= keep[..self.firsts].iter().filter(|kept| !**kept).count();
            kept(&mut self.rules, &keep);
            kept(&mut self.measured, &keep);
            kept(&mut self.pairs, &keep);
        }
        // Flushed with errors checked, so a truncated artifact fails the run
        // instead of being dropped silently with its writer.
        for file in [&mut self.sufficiency, &mut self.shortlist]
            .into_iter()
            .flatten()
        {
            use std::io::Write;
            file.flush().expect("flush a raw artifact");
        }
        report(
            title,
            &self.rules,
            shipped(&self.rules),
            &self.measured,
            crate::channels::Units::Clusters(&self.clusters),
        );
        println!("  model pairs a search");
        for ((label, _), pairs) in self.rules.iter().zip(&self.pairs) {
            println!(
                "  {label:<52}  {:>6.2}",
                *pairs as f64 / self.queries.max(1) as f64
            );
        }
    }
}

/// Every row against the one that ships, group by group and as one
/// cross-validated choice. `measured` is indexed like `labels`.
pub fn report<T>(
    title: &str,
    labels: &[(String, T)],
    ship: Option<usize>,
    measured: &[BTreeMap<String, crate::scoring::Scores>],
    units: crate::channels::Units,
) {
    let shipped = &measured[ship.expect("the shipped row is measured")];
    println!("\n  {title}");
    for (group, scores) in shipped {
        crate::channels::clustered_sweep_table(
            group,
            scores,
            labels,
            &measured
                .iter()
                .map(|row| row.get(group))
                .collect::<Vec<_>>(),
            &units.group_keys(group, scores),
        );
    }
    crate::channels::cross_validated(
        title,
        &shipped.keys().map(String::as_str).collect::<Vec<_>>(),
        shipped,
        labels,
        ship,
        measured,
        units,
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

/// What the shipped rendering shows the model for each of `names`.
fn shown(hits: &[SearchHit], names: &[&str]) -> Vec<String> {
    let found: HashMap<&str, &SearchHit> =
        hits.iter().map(|hit| (hit.topic.as_str(), hit)).collect();
    let (named, with_seed) = (SHIPPED % 2 == 1, SHIPPED >= 2);
    names
        .iter()
        .map(|name| {
            let hit = found[name];
            let seed = hit.seed.as_deref().filter(|_| with_seed);
            render(name, &hit.state.content, seed, named)
        })
        .collect()
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
    // Check the shipped rendering against the retained engine trace before
    // evaluating alternatives. Cached scores require an identical complete
    // ordered batch; changed renderings or batch membership are rescored.
    // INT8 scores can depend on that actual batch, but cache history does not
    // make later renderings approximations or require this evaluation order.
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
        rendered[rendering] = Some(
            replayed
                .rescored(scores)
                .order(shipped_rule(Rerank::default())),
        );
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
    /// Per movable position, its graph-channel score, zero when the graph did
    /// not reach it.
    graph: Vec<f64>,
    /// How deep the tier reads, which is where a graph find is inserted.
    head: usize,
    /// How far the fused first leads the fused second, as a share of the
    /// first's score; zero with fewer than two.
    lead: f64,
    /// Whether a lexical channel and the vector channel both rank the fused
    /// first in their top three.
    agree: bool,
    /// The reranker's passes over the movable candidates, as indices into
    /// them in the order it runs them, once [`Replayed::planned`] has asked
    /// it; what [`Rule::Patience`] replays.
    passes: Option<Vec<Vec<usize>>>,
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

    // The fusion this search ran, a profile's secondary vector channel
    // included, rather than the default.
    let fusion = crate::channels::effective_fusion(hits, &Fusion::default());
    let results = fusion.fuse(&crate::channels::replay(hits));
    // The engine's own rule for what the model is shown, not a copy of it.
    let traces: Vec<&[Why]> = results.iter().map(|result| result.why.as_slice()).collect();
    let movable = pamin_engine::rerankable(&traces, tier);
    let fused: Vec<String> = results
        .iter()
        .map(|result| named[&result.topic].to_string())
        .collect();

    let lead = match results.as_slice() {
        [first, second, ..] if first.score > 0.0 => {
            f64::from((first.score - second.score) / first.score)
        }
        _ => 0.0,
    };
    let agree = results.first().is_some_and(|first| {
        let near = |wanted: &dyn Fn(Channel) -> bool| {
            first.why.iter().any(|why| {
                matches!(why, Why::Channel { channel, rank, .. } if wanted(*channel) && *rank <= 3)
            })
        };
        near(&|channel| matches!(channel, Channel::LexicalSegmented | Channel::LexicalNgram))
            && near(&|channel| channel == Channel::Vector)
    });

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
            graph: Vec::new(),
            head: tier.depth(),
            lead,
            agree,
            passes: None,
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
            graph: movable
                .iter()
                .map(|at| {
                    results[*at]
                        .why
                        .iter()
                        .find_map(|why| match why {
                            Why::Channel {
                                channel: Channel::Graph,
                                score,
                                ..
                            } => Some(f64::from(score.unwrap_or(0.0))),
                            _ => None,
                        })
                        .unwrap_or(0.0)
                })
                .collect(),
            head: tier.depth(),
            lead,
            agree,
            passes: None,
            fused,
            movable,
        }
    };

    let engine: Vec<&str> = hits.iter().map(|hit| hit.topic.as_str()).collect();
    assert_eq!(
        replayed.order(shipped_rule(tier)),
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
            graph: self.graph.clone(),
            head: self.head,
            lead: self.lead,
            agree: self.agree,
            passes: self.passes.clone(),
        }
    }

    /// The same query with the passes `reranker` runs over what it was
    /// shown, `documents` in movable order: their encoded lengths through
    /// the planner [`pamin_index::Reranker::rank`] runs.
    pub fn planned(
        mut self,
        reranker: &pamin_index::Reranker,
        query: &str,
        documents: &[&str],
    ) -> Replayed {
        assert_eq!(
            documents.len(),
            self.movable.len(),
            "one document per movable candidate"
        );
        let lengths = reranker
            .lengths(query, documents)
            .expect("encode the shortlist");
        self.passes = Some(reranker.plan(&lengths));
        self
    }

    /// The whole list under `rule`: unmovable positions stay, movable ones are
    /// refilled in the rule's order, ties to the earlier position.
    pub fn order(&self, rule: Rule) -> Vec<String> {
        if let Some(keep) = self.scored(rule) {
            return cascade_order(self, &keep);
        }
        if let Rule::Blend {
            fusion,
            scale: Scale::Score,
        } = rule
        {
            let picks = pamin_engine::score_blend_order(&self.fusion, &self.model, fusion);
            return pamin_engine::place(self.fused.clone(), &self.movable, self.head, &picks);
        }
        let keys: Vec<f64> = match rule {
            Rule::Substitute => self.model.clone(),
            Rule::GraphBlend { graph } => {
                let scaled = |values: &[f64]| {
                    let low = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    values
                        .iter()
                        .map(|value| {
                            if high > low {
                                (value - low) / (high - low)
                            } else {
                                0.0
                            }
                        })
                        .collect::<Vec<_>>()
                };
                let (fused, model) = (scaled(&self.fusion), scaled(&self.model));
                let fusion = pamin_engine::RERANK_FUSION;
                (0..fused.len())
                    .map(|at| {
                        fusion * fused[at] + (1.0 - fusion) * model[at] + graph * self.graph[at]
                    })
                    .collect()
            }
            Rule::Blend {
                fusion,
                scale: Scale::Rank,
            } => {
                let (first, second) = (on(&self.fusion, false), on(&self.model, true));
                first
                    .iter()
                    .zip(&second)
                    .map(|(fused, model)| fusion * fused + (1.0 - fusion) * model)
                    .collect()
            }
            Rule::Blend {
                scale: Scale::Score,
                ..
            }
            | Rule::Prefix { .. }
            | Rule::Confident { .. }
            | Rule::Patience { .. } => unreachable!("handled above"),
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

impl Replayed {
    /// Which movable candidates a rule that scores only some of them has the
    /// model score, as indices into the movable ones; `None` for a rule that
    /// reads every score.
    fn scored(&self, rule: Rule) -> Option<Vec<usize>> {
        let count = self.movable.len();
        match rule {
            Rule::Prefix { n } => Some((0..n.min(count)).collect()),
            Rule::Confident { lead, n } => (self.lead >= lead && self.agree)
                .then(|| (0..n.min(count)).collect())
                .or_else(|| Some((0..count).collect())),
            Rule::Patience { top } => {
                let passes = self
                    .passes
                    .as_ref()
                    .expect("Patience replays the reranker's passes: plan them first");
                let mut scored: Vec<usize> = Vec::new();
                for (pass, positions) in passes.iter().enumerate() {
                    scored.extend(positions);
                    if pass < 2 {
                        continue;
                    }
                    let mut best = scored.clone();
                    best.sort_by(|left, right| {
                        self.model[*right]
                            .total_cmp(&self.model[*left])
                            .then(left.cmp(right))
                    });
                    best.truncate(top);
                    if !best.iter().any(|at| positions.contains(at)) {
                        break;
                    }
                }
                scored.sort_unstable();
                Some(scored)
            }
            Rule::Substitute | Rule::Blend { .. } | Rule::GraphBlend { .. } => None,
        }
    }

    /// How many pairs the model scores under `rule` -- what the reranker's
    /// time is proportional to, up to batching and length.
    pub fn pairs(&self, rule: Rule) -> usize {
        match self.scored(rule) {
            // The engine declines to rerank fewer than two.
            Some(keep) if keep.len() < 2 => 0,
            Some(keep) => keep.len(),
            None => self.movable.len(),
        }
    }
}

/// `values` on `scale`, higher better. `sort` says whether they need ranking:
/// the fused scores arrive already in order, the model's do not.
fn on(values: &[f64], sort: bool) -> Vec<f64> {
    let count = values.len();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn replayed(fusion: &[f64], model: &[f64]) -> Replayed {
        Replayed {
            fused: ["lexical", "a", "b", "c"].map(String::from).to_vec(),
            movable: vec![1, 2, 3],
            fusion: fusion.to_vec(),
            model: model.to_vec(),
            graph: vec![0.0; fusion.len()],
            head: 4,
            lead: 0.0,
            agree: false,
            passes: None,
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

    #[test]
    fn shipped_accurate_replay_blends_when_the_model_order_differs() {
        let scores = replayed(&[1.0, 0.0, 0.0], &[0.9, 1.0, 0.0]);
        assert_eq!(
            scores.order(shipped_rule(Rerank::Accurate)),
            ["lexical", "a", "b", "c"]
        );
        assert_eq!(
            scores.order(shipped_rule(Rerank::Fast)),
            ["lexical", "b", "a", "c"]
        );
    }

    #[test]
    fn cascade_keeps_the_shipped_blend_on_selected_candidates() {
        let scores = replayed(&[1.0, 0.0, 0.0], &[0.9, 1.0, 0.0]);
        assert_eq!(
            cascade_order(&scores, &[0, 1, 2]),
            scores.order(shipped_rule(Rerank::Accurate))
        );
    }

    #[test]
    fn scoring_every_candidate_first_is_the_shipped_pass() {
        let scores = replayed(&[1.0, 0.5, 0.0], &[0.0, 1.0, 2.0]);
        let shipped = scores.order(shipped_rule(Rerank::Accurate));
        assert_eq!(scores.order(Rule::Prefix { n: 3 }), shipped);
        assert_eq!(scores.pairs(Rule::Prefix { n: 3 }), 3);
        // Unsettled: the fused first leads by nothing and nothing agrees.
        assert_eq!(scores.order(Rule::Confident { lead: 0.1, n: 0 }), shipped);
        assert_eq!(scores.pairs(Rule::Confident { lead: 0.1, n: 0 }), 3);
    }

    #[test]
    fn a_prefix_too_short_to_rerank_is_fusion_alone() {
        let scores = replayed(&[1.0, 0.5, 0.0], &[0.0, 1.0, 2.0]);
        assert_eq!(scores.order(Rule::Prefix { n: 1 }), scores.fused);
        assert_eq!(scores.pairs(Rule::Prefix { n: 1 }), 0);
        let settled = Replayed {
            lead: 0.5,
            agree: true,
            ..scores.rescored(scores.model.clone())
        };
        assert_eq!(
            settled.order(Rule::Confident { lead: 0.1, n: 0 }),
            settled.fused
        );
        assert_eq!(settled.pairs(Rule::Confident { lead: 0.1, n: 0 }), 0);
    }

    #[test]
    fn patience_stops_after_a_pass_that_reaches_nothing() {
        let count = 12;
        let fusion: Vec<f64> = (0..count)
            .map(|at| 1.0 - at as f64 / count as f64)
            .collect();
        let mut scores = replayed(&fusion, &fusion);
        scores.fused = (0..=count).map(|at| at.to_string()).collect();
        scores.movable = (1..=count).collect();
        scores.graph = vec![0.0; count];
        scores.head = count + 1;
        // Passes in the order the reranker runs them, by length, not by
        // fusion.
        scores.passes = Some(vec![
            vec![11, 7, 3],
            vec![10, 6, 2],
            vec![9, 5, 1],
            vec![8, 4, 0],
        ]);
        // The model agrees with fusion, so the third pass puts candidate 1
        // into the top three and the fourth, holding 0, does too: all read.
        assert_eq!(scores.pairs(Rule::Patience { top: 3 }), count);
        // A model that prefers what the first two passes held stops after
        // the third, which reaches nothing, and never scores the fourth.
        let mut model = fusion.clone();
        for (at, score) in [(11, 9.0), (10, 8.0), (7, 7.0)] {
            model[at] = score;
        }
        let rescored = scores.rescored(model);
        assert_eq!(rescored.pairs(Rule::Patience { top: 3 }), 9);
        assert_eq!(
            rescored.scored(Rule::Patience { top: 3 }),
            Some(vec![1, 2, 3, 5, 6, 7, 9, 10, 11])
        );
    }
}
