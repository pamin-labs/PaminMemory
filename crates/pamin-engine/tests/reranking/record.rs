//! One replay record a search, for pricing a shortlist the product did not
//! ship without running the model again.
//!
//! `REPLAY_OUT` names the file; set, it runs the rules arm
//! ([`super::rules_asked`]) and writes one JSON line for every search that arm
//! makes. A line holds the whole fused list down to channel depth -- what
//! each channel ranked and scored every candidate, the graph path that
//! reached it, whether the shipped rule showed it to the model, and how every
//! answer key the search is scored under judges it -- and the model's score
//! for a **superset** of the shipped shortlist ([`superset`]).
//!
//! It replaces three dumps that could not price a shortlist between them:
//! the shortlists alone with their scores, the candidate features without
//! any, and a first-result sufficiency table derivable from either.
//!
//! **Why a superset makes a replay exact.** On the CPU the reranker scores
//! one pair a pass, so a pair's score does not depend on what else was in
//! the list. Any admission rule whose shortlist is inside the superset then
//! replays exactly from the line: its candidates' model scores are in it,
//! and the blend and placement are the engine's own
//! (`pamin_engine::score_blend_order`, `pamin_engine::place`), as
//! [`Line::order`] uses them. A rule that shows the model something outside
//! the superset cannot be priced from this file.
//!
//! The superset is scored through the engine's own reranker
//! (`Engine::rerank_scores`) after the shipped search has returned, so the
//! search itself -- its ranking and its timing -- is the product's.
//!
//! ## What each line asserts before it is written
//!
//! - every candidate the shipped rule showed the model is in the superset,
//!   and the superset scored it bit for bit as the shipped search did --
//!   the premise of the paragraph above, which a model scoring in batches,
//!   or on another target, would break;
//! - the line, written and read back, replays the shipped rule into the
//!   engine's own order and each answer key's nDCG@10 into the figure the
//!   harness scored for the shipped row, bit for bit.
//!
//! And [`Record::finish`] asserts that the superset scored at least the
//! pairs the shipped searches did.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;

use pamin_core::{Channel, Derivation, EdgeKind, Why};
use pamin_engine::SearchHit;

use super::Replayed;

/// The fused head the superset scores: ten past the shipped thirty, for
/// pricing a deeper head.
const HEAD: usize = 40;

/// Every candidate below the head that only the graph found is scored from
/// this graph score, half the shipped threshold, and with no cap -- for
/// pricing a graph quota or a lower floor.
const GRAPH_FROM: f32 = 0.25;

/// How many of the best-fused candidates below the head that only the
/// lexical channels found are scored, for pricing a lexical-only quota.
const LEXICAL_ONLY: usize = 5;

/// One search as the harness asked it.
pub struct Asked<'a> {
    /// What the harness groups the search under: its language on XQuAD-R,
    /// its category on LOCOMO.
    pub group: &'a str,
    /// Stable across runs and unique within the corpus, so a search can be
    /// paired across records.
    pub question: &'a str,
    pub query: &'a str,
    /// The language the harness knows the query is in, in the code it wrote
    /// its memories with; `None` where it does not know.
    pub language: Option<&'a str>,
    /// The unit the search belongs to: searches that share one share a fold
    /// and are averaged together for the paired test. `None` makes the search
    /// its own unit.
    pub cluster: Option<&'a str>,
}

/// One answer key a search is scored under, as the harness's own scoring
/// reads it.
pub struct Answers {
    /// The group the harness scores the search into under this key.
    pub key: String,
    /// How many items are relevant, found or not: what the ideal gain and
    /// the recall denominator are built from.
    pub judged: usize,
    pub relevant: HashSet<String>,
    /// What the key takes out of the ranking before scoring it: XQuAD-R's
    /// other correct answers.
    pub dropped: HashSet<String>,
}

/// One line of the record.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Line {
    /// SHA-256 over the run's question and document set (see
    /// `harness::shape::Shape::digest`).
    pub corpus: String,
    /// The commit, build, settings, models and plans that wrote the line: a
    /// corpus recorded twice is told apart by these, not by its digest.
    pub run: crate::harness::shape::Run,
    /// Which search of the run, from zero, in the order asked.
    pub search: usize,
    pub group: String,
    pub question: String,
    pub query: String,
    pub language: Option<String>,
    pub cluster: String,
    /// How deep the shipped rule reads: where a graph find is inserted.
    pub head: usize,
    /// Fusion's share of the blend.
    pub blend: f64,
    pub keys: Vec<Key>,
    /// Every fused candidate, in fused order.
    pub candidates: Vec<Candidate>,
}

/// An answer key, and how many items it judges relevant.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Key {
    pub key: String,
    pub judged: usize,
}

/// One fused candidate.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Candidate {
    pub topic: String,
    /// One-based.
    pub fused_rank: usize,
    pub fused_score: f32,
    /// By `Channel::as_str`, every channel that returned it.
    pub channels: BTreeMap<String, Placed>,
    /// For a candidate the graph reached, how. The graph channel's score is
    /// the path's strength times the origin's relevance; the trace carries
    /// neither factor apart.
    pub path: Option<Path>,
    pub language: Option<String>,
    /// The content's length in characters and the script most of its
    /// letters are in ([`script`]).
    pub chars: usize,
    pub script: String,
    /// Whether the shipped rule showed it to the model.
    pub shown: bool,
    /// The model's score, for a candidate in the superset.
    pub model: Option<f32>,
    /// What the model was shown for it, for a candidate in the superset: for
    /// scoring the same pairs with another model (`RESCORE_IN`).
    pub document: Option<String>,
    /// The keys it is relevant under, and the keys that take it out.
    pub relevant: Vec<String>,
    pub dropped: Vec<String>,
}

/// Where one channel ranked a candidate, and what it scored it.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Placed {
    pub rank: u32,
    pub score: Option<f32>,
}

/// The graph path that reached a candidate.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Path {
    /// The seed the walk started from.
    pub from: String,
    pub via: String,
    pub hops: u8,
    pub edge: EdgeKind,
    pub derivation: Derivation,
}

impl Line {
    /// The positions the shipped rule showed the model, ascending.
    pub fn shown(&self) -> Vec<usize> {
        (0..self.candidates.len())
            .filter(|at| self.candidates[*at].shown)
            .collect()
    }

    /// The whole list when the model is shown the candidates at `shown`
    /// (ascending): blended and placed as the engine does, the rest where
    /// fusion put them. Fewer than two is fusion alone, as in the engine.
    ///
    /// **Panics if one of them was not scored**: a shortlist outside the
    /// superset cannot be priced from the record.
    pub fn order(&self, shown: &[usize]) -> Vec<String> {
        let names: Vec<String> = self
            .candidates
            .iter()
            .map(|candidate| candidate.topic.clone())
            .collect();
        if shown.len() < 2 {
            return names;
        }
        let (fused, model): (Vec<f64>, Vec<f64>) = shown
            .iter()
            .map(|at| {
                let candidate = &self.candidates[*at];
                let model = candidate.model.unwrap_or_else(|| {
                    panic!("{} is outside the scored superset", candidate.topic)
                });
                (f64::from(candidate.fused_score), f64::from(model))
            })
            .unzip();
        let best = pamin_engine::score_blend_order(&fused, &model, self.blend);
        pamin_engine::place(names, shown, self.head, &best)
    }

    /// Each answer key's nDCG@10 for `order`, scored as the harness scores
    /// it: what the key drops taken out first.
    pub fn ndcg(&self, order: &[String]) -> BTreeMap<String, f64> {
        let by_name: HashMap<&str, &Candidate> = self
            .candidates
            .iter()
            .map(|candidate| (candidate.topic.as_str(), candidate))
            .collect();
        let candidate = |topic: &str| by_name.get(topic).copied();
        self.keys
            .iter()
            .map(|Key { key, judged }| {
                let kept: Vec<String> = order
                    .iter()
                    .filter(|topic| {
                        candidate(topic.as_str()).is_none_or(|it| !it.dropped.contains(key))
                    })
                    .cloned()
                    .collect();
                let ndcg = crate::scoring::Scores::default().add(&kept, *judged, |topic| {
                    candidate(topic).is_some_and(|it| it.relevant.contains(key))
                });
                (key.clone(), ndcg)
            })
            .collect()
    }
}

/// The open record, and what has been written to it. Lines go to a partial
/// file of its own beside `path`, which [`finish`](Self::finish) renames to
/// `path` only once every gate has passed, so a run that panics or stops
/// part way leaves no file a replay could take for a whole corpus.
pub struct Record {
    path: PathBuf,
    partial: PathBuf,
    out: BufWriter<File>,
    /// The run's corpus digest, from its published shape or named by the
    /// harness ([`super::Priced::corpus`]).
    pub corpus: Option<String>,
    /// Read once the first search has loaded the models it names.
    run: Option<crate::harness::shape::Run>,
    searches: usize,
    /// Pairs the shipped searches showed the model, and pairs the superset
    /// scored.
    shown: usize,
    scored: usize,
}

impl Record {
    /// The record `REPLAY_OUT` names, or `None` when it is not asked for.
    pub fn from_env() -> Option<Self> {
        let path = PathBuf::from(std::env::var_os("REPLAY_OUT")?);
        // Its own, so two runs naming one record never write into each other's.
        let partial = crate::harness::shape::unique(&path, "partial");
        let file = File::create(&partial)
            .unwrap_or_else(|error| panic!("creating {}: {error}", partial.display()));
        Some(Self {
            path,
            partial,
            out: BufWriter::new(file),
            corpus: crate::harness::shape::published_digest(),
            run: None,
            searches: 0,
            shown: 0,
            scored: 0,
        })
    }

    /// One shipped search, which `engine` ran and `replayed` rebuilt.
    /// `shipped` is each answer key's nDCG@10 as the harness scored the
    /// shipped rule's order.
    pub fn observe(
        &mut self,
        engine: &pamin_engine::Engine,
        asked: &Asked<'_>,
        hits: &[SearchHit],
        replayed: &Replayed,
        answers: &[Answers],
        shipped: &BTreeMap<String, f64>,
    ) {
        assert!(
            !replayed.fused.is_empty(),
            "{:?}: the search found nothing, so its line would replay vacuously; is the \
             project empty or the index stale?",
            asked.query
        );
        let by_name: HashMap<&str, &SearchHit> =
            hits.iter().map(|hit| (hit.topic.as_str(), hit)).collect();
        let mut candidates: Vec<Candidate> = replayed
            .fused
            .iter()
            .enumerate()
            .map(|(at, name)| candidate(at, by_name[name.as_str()], answers))
            .collect();
        for at in &replayed.movable {
            candidates[*at].shown = true;
        }

        let scored = superset(&candidates, replayed.head);
        assert!(
            replayed
                .movable
                .iter()
                .all(|at| scored.binary_search(at).is_ok()),
            "{:?}: the shipped shortlist is not inside the superset, so the record cannot \
             replay what shipped",
            asked.query
        );
        // The engine declines to rerank fewer than two, so neither does this.
        if scored.len() >= 2 {
            let names: Vec<&str> = scored
                .iter()
                .map(|at| candidates[*at].topic.as_str())
                .collect();
            let documents = super::shown(engine, asked.query, hits, &names);
            let borrowed: Vec<&str> = documents.iter().map(String::as_str).collect();
            // Equal scores below do not show independence by themselves: an
            // unchanged list can come back from the reranker's cache. The
            // plan does: one pair a pass is what makes a pair's score its own.
            let passes = engine
                .rerank_passes(asked.query, &borrowed)
                .expect("the engine's reranker scored this search, so it is loaded")
                .expect("plan the superset");
            assert!(
                passes.iter().all(|pass| pass.len() == 1),
                "{:?}: the reranker scores this superset in passes of {:?} pairs, so a \
                 pair's score depends on its neighbours and no shortlist replays exactly; \
                 record on the CPU, where it scores one pair a pass",
                asked.query,
                passes.iter().map(Vec::len).collect::<Vec<_>>()
            );
            let scores = engine
                .rerank_scores(asked.query, &borrowed)
                .expect("the engine's reranker scored this search, so it is loaded")
                .expect("score the superset");
            for ((at, document), score) in scored.iter().zip(documents).zip(scores) {
                candidates[*at].model = Some(score);
                candidates[*at].document = Some(document);
            }
        }
        for (at, recorded) in replayed.movable.iter().zip(&replayed.model) {
            let again = candidates[*at]
                .model
                .expect("inside the superset, asserted");
            assert_eq!(
                f64::from(again).to_bits(),
                recorded.to_bits(),
                "{:?}: the superset scored {} {again} where the shipped search scored \
                 {recorded}, so a pair's score depends on what it was scored beside and no \
                 shortlist replays exactly from this record",
                asked.query,
                candidates[*at].topic
            );
        }

        let line = Line {
            corpus: self.corpus.clone().expect(
                "REPLAY_OUT needs the corpus digest: publish a shape, or name it with \
                 `Priced::corpus`",
            ),
            run: self.run(engine, asked.query),
            search: self.searches,
            group: asked.group.to_string(),
            question: asked.question.to_string(),
            query: asked.query.to_string(),
            language: asked.language.map(str::to_string),
            cluster: asked
                .cluster
                .map_or_else(|| format!("search {}", self.searches), str::to_string),
            head: replayed.head,
            blend: pamin_engine::RERANK_FUSION,
            keys: answers
                .iter()
                .map(|answers| Key {
                    key: answers.key.clone(),
                    judged: answers.judged,
                })
                .collect(),
            candidates,
        };
        let text = serde_json::to_string(&line).expect("a line serialises");

        // The gate: the line as written, read back, is the shipped search.
        let back: Line = serde_json::from_str(&text).expect("a line reads back");
        for (written, read) in line.candidates.iter().zip(&back.candidates) {
            assert!(
                written.fused_score.to_bits() == read.fused_score.to_bits()
                    && written.model.map(f32::to_bits) == read.model.map(f32::to_bits),
                "{:?}: a score of {} did not read back as written",
                asked.query,
                written.topic
            );
        }
        let order = back.order(&back.shown());
        let engine_order: Vec<&str> = hits.iter().map(|hit| hit.topic.as_str()).collect();
        assert_eq!(
            order, engine_order,
            "{:?}: the record's shipped rule did not replay into the engine's order",
            asked.query
        );
        let replayed_ndcg = back.ndcg(&order);
        assert_eq!(
            replayed_ndcg.keys().collect::<Vec<_>>(),
            shipped.keys().collect::<Vec<_>>(),
            "{:?}: the record's answer keys are not the ones the harness scored",
            asked.query
        );
        for (key, ndcg) in &replayed_ndcg {
            assert_eq!(
                ndcg.to_bits(),
                shipped[key].to_bits(),
                "{:?}: the record scores the shipped order {ndcg} under {key} where the \
                 harness scored {}, so its labels are not the harness's",
                asked.query,
                shipped[key]
            );
        }

        writeln!(self.out, "{text}")
            .unwrap_or_else(|error| panic!("writing {}: {error}", self.partial.display()));
        self.searches += 1;
        self.shown += replayed.movable.len();
        self.scored += back
            .candidates
            .iter()
            .filter(|candidate| candidate.model.is_some())
            .count();
    }

    /// The run this search belongs to: read at the first search, and at
    /// every later one checked against the plans the engine has used since,
    /// so a fallback part way through fails the run rather than leaving lines
    /// that name plans their scores did not come from.
    fn run(&mut self, engine: &pamin_engine::Engine, query: &str) -> crate::harness::shape::Run {
        let engines = [("primary", engine)];
        let Some(run) = &self.run else {
            return self
                .run
                .insert(crate::harness::shape::run(&engines))
                .clone();
        };
        let (models, providers) = crate::harness::shape::loaded(&engines);
        assert!(
            models == run.models && providers == run.providers,
            "{query:?}: the engine now names models {models:?} on {providers:?}, where the \
             record's first search named {:?} on {:?}; one record holds one run's plans",
            run.models,
            run.providers
        );
        run.clone()
    }

    /// Flushes the record, asserts it holds a search and that the superset
    /// scored at least what the shipped searches did, and only then publishes
    /// it under its path.
    pub fn finish(&mut self) {
        self.out
            .flush()
            .unwrap_or_else(|error| panic!("flushing {}: {error}", self.partial.display()));
        assert!(
            self.searches > 0,
            "the record holds no search; a sample of none measures nothing"
        );
        assert!(
            self.scored >= self.shown,
            "the record scored {} pairs against the shipped {}",
            self.scored,
            self.shown
        );
        std::fs::rename(&self.partial, &self.path).unwrap_or_else(|error| {
            panic!(
                "publishing {} as {}: {error}",
                self.partial.display(),
                self.path.display()
            )
        });
        println!(
            "\n  replay record: {} searches, {} pairs scored for {} shipped, in {}",
            self.searches,
            self.scored,
            self.shown,
            self.path.display()
        );
    }
}

/// The fused candidate at `at`, as `hit`'s trace and `answers` describe it.
fn candidate(at: usize, hit: &SearchHit, answers: &[Answers]) -> Candidate {
    let mut channels = BTreeMap::new();
    let mut path = None;
    for why in &hit.result.why {
        match why {
            Why::Channel {
                channel,
                rank,
                score,
                ..
            } => {
                let placed = Placed {
                    rank: *rank,
                    score: *score,
                };
                assert!(
                    channels
                        .insert(channel.as_str().to_string(), placed)
                        .is_none(),
                    "{} is listed twice in {channel:?}",
                    hit.topic
                );
            }
            Why::Path {
                from,
                via,
                hops,
                edge,
                derivation,
                ..
            } => {
                path = Some(Path {
                    from: from.clone(),
                    via: via.clone(),
                    hops: *hops,
                    edge: *edge,
                    derivation: *derivation,
                });
            }
            Why::Reranked { .. } => {}
        }
    }
    let keys = |holds: &dyn Fn(&Answers) -> bool| -> Vec<String> {
        answers
            .iter()
            .filter(|answers| holds(answers))
            .map(|answers| answers.key.clone())
            .collect()
    };
    let content = &hit.state.content;
    Candidate {
        topic: hit.topic.clone(),
        fused_rank: at + 1,
        fused_score: hit.result.score,
        channels,
        path,
        language: hit.state.language.clone(),
        chars: content.chars().count(),
        script: script(content).to_string(),
        shown: false,
        model: None,
        document: None,
        relevant: keys(&|answers| answers.relevant.contains(&hit.topic)),
        dropped: keys(&|answers| answers.dropped.contains(&hit.topic)),
    }
}

/// The positions the record scores, ascending: the first [`HEAD`] (or the
/// shipped head, if deeper), every candidate below it only the graph found
/// at [`GRAPH_FROM`] or more, and the [`LEXICAL_ONLY`] best-fused below it
/// only the lexical channels found. Read off the line's own fields, so a
/// rule replayed from the record can say which candidates it may show.
fn superset(candidates: &[Candidate], head: usize) -> Vec<usize> {
    let head = HEAD.max(head).min(candidates.len());
    let only = |candidate: &Candidate, channels: &[Channel]| {
        candidate
            .channels
            .keys()
            .all(|name| channels.iter().any(|channel| channel.as_str() == name))
    };
    let graph = Channel::Graph.as_str();
    let mut positions: Vec<usize> = (0..head).collect();
    positions.extend((head..candidates.len()).filter(|at| {
        let candidate = &candidates[*at];
        only(candidate, &[Channel::Graph])
            && candidate.channels[graph].score.unwrap_or(0.0) >= GRAPH_FROM
    }));
    positions.extend(
        (head..candidates.len())
            .filter(|at| {
                only(
                    &candidates[*at],
                    &[Channel::LexicalSegmented, Channel::LexicalNgram],
                )
            })
            .take(LEXICAL_ONLY),
    );
    positions.sort_unstable();
    positions
}

/// The script most of a text's letters are written in.
///
/// Read off code-point blocks, which is all a script is. `none` when the text
/// has no letters. Ties go to the name that sorts first, so the answer is the
/// same on every run. Recorded rather than a language because a detector will
/// not commit to a language for a short text (see `Engine::search_reranked`),
/// and a script it can read off the code points.
pub fn script(text: &str) -> &'static str {
    let mut counts: [(&'static str, usize); 11] = [
        ("arabic", 0),
        ("cyrillic", 0),
        ("devanagari", 0),
        ("greek", 0),
        ("han", 0),
        ("hangul", 0),
        ("hebrew", 0),
        ("kana", 0),
        ("latin", 0),
        ("other", 0),
        ("thai", 0),
    ];
    for letter in text.chars().filter(|c| c.is_alphabetic()) {
        let name = match u32::from(letter) {
            0x0041..=0x024F | 0x1E00..=0x1EFF => "latin",
            0x0370..=0x03FF | 0x1F00..=0x1FFF => "greek",
            0x0400..=0x052F => "cyrillic",
            0x0590..=0x05FF => "hebrew",
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => "arabic",
            0x0900..=0x097F => "devanagari",
            0x0E00..=0x0E7F => "thai",
            0x3040..=0x30FF | 0x31F0..=0x31FF => "kana",
            0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => "hangul",
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => "han",
            _ => "other",
        };
        if let Some(entry) = counts.iter_mut().find(|(it, _)| *it == name) {
            entry.1 += 1;
        }
    }
    counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .max_by(|left, right| left.1.cmp(&right.1).then(right.0.cmp(left.0)))
        .map_or("none", |(name, _)| name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(topic: &str, fused: f32, channels: &[(Channel, f32)]) -> Candidate {
        Candidate {
            topic: topic.to_string(),
            fused_rank: 0,
            fused_score: fused,
            channels: channels
                .iter()
                .enumerate()
                .map(|(rank, (channel, score))| {
                    let placed = Placed {
                        rank: rank as u32 + 1,
                        score: Some(*score),
                    };
                    (channel.as_str().to_string(), placed)
                })
                .collect(),
            path: None,
            language: None,
            chars: 0,
            script: "none".to_string(),
            shown: false,
            model: None,
            document: None,
            relevant: Vec::new(),
            dropped: Vec::new(),
        }
    }

    #[test]
    fn the_superset_is_the_deeper_head_graph_finds_and_lexical_only_finds() {
        let vector = [(Channel::Vector, 0.5)];
        let lexical = [
            (Channel::LexicalSegmented, 3.0),
            (Channel::LexicalNgram, 2.0),
        ];
        let mut candidates: Vec<Candidate> = (0..45)
            .map(|at| found(&at.to_string(), 1.0, &vector))
            .collect();
        candidates.push(found("45", 0.1, &[(Channel::Graph, GRAPH_FROM)]));
        candidates.push(found("46", 0.1, &[(Channel::Graph, GRAPH_FROM / 2.0)]));
        // Found by the graph, but not only by it.
        candidates.push(found("47", 0.1, &[(Channel::Graph, 1.0), vector[0]]));
        candidates.extend((48..55).map(|at| found(&at.to_string(), 0.1, &lexical)));
        candidates.extend((55..60).map(|at| found(&at.to_string(), 0.1, &vector)));

        let mut expected: Vec<usize> = (0..HEAD).collect();
        expected.push(45);
        expected.extend(48..48 + LEXICAL_ONLY);
        assert_eq!(superset(&candidates, 30), expected);
        // A shipped head deeper than the superset's is read whole.
        assert_eq!(superset(&candidates, 60), (0..60).collect::<Vec<_>>());
    }

    #[test]
    fn a_line_replays_any_shortlist_it_scored_and_scores_it_as_the_harness_does() {
        let mut candidates = vec![
            found("a", 0.9, &[(Channel::Vector, 0.9)]),
            found("b", 0.5, &[(Channel::Vector, 0.5)]),
            found("c", 0.1, &[(Channel::Vector, 0.1)]),
        ];
        for (candidate, model) in candidates.iter_mut().zip([0.0, 1.0, 2.0]) {
            candidate.model = Some(model);
        }
        candidates[1].dropped = vec!["k".to_string()];
        candidates[2].relevant = vec!["k".to_string()];
        let line = Line {
            corpus: String::new(),
            run: crate::harness::shape::Run {
                commit: String::new(),
                noncanonical: None,
                build: String::new(),
                settings: Vec::new(),
                models: Vec::new(),
                providers: Vec::new(),
            },
            search: 0,
            group: String::new(),
            question: String::new(),
            query: String::new(),
            language: None,
            cluster: String::new(),
            head: 3,
            blend: 0.2,
            keys: vec![Key {
                key: "k".to_string(),
                judged: 1,
            }],
            candidates,
        };

        // The model reverses fusion and outweighs it.
        let shown = line.order(&[0, 1, 2]);
        assert_eq!(shown, ["c", "b", "a"]);
        assert_eq!(line.ndcg(&shown)["k"], 1.0);
        // Fewer than two shown is fusion alone; `b` is dropped before
        // scoring, so `c` is second.
        let fused = line.order(&[1]);
        assert_eq!(fused, ["a", "b", "c"]);
        assert_eq!(line.ndcg(&fused)["k"], 1.0 / 3f64.log2());
    }

    #[test]
    fn a_script_is_what_most_letters_are_written_in() {
        assert_eq!(script("how long are database backups kept"), "latin");
        assert_eq!(script("сколько хранятся резервные копии"), "cyrillic");
        assert_eq!(script("备份保留多久"), "han");
        assert_eq!(script("バックアップはいつまで"), "kana");
        assert_eq!(script("ข้อมูลสำรอง"), "thai");
        // Digits and punctuation are not letters, so they do not vote.
        assert_eq!(script("2024: 备份 v2"), "han");
        assert_eq!(script("42 -- 17"), "none");
    }
}
