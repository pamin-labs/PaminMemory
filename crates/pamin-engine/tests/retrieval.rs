//! What retrieval quality this engine actually has, in three numbers.
//!
//! Every default in this project — which embedding profile, how deep each
//! channel reaches, whether the n-gram field earns its index size, whether
//! quantizing the stored vectors costs anything — has been argued from
//! published benchmarks run on somebody else's corpus. This is the first thing
//! that measures them here.
//!
//! Ignored by default: it provisions PostgreSQL, downloads model weights, and
//! embeds the whole corpus. Run with
//! `cargo test -p pamin-engine --test retrieval -- --ignored --nocapture`.
//!
//! ## Three groups, reported separately, never summed
//!
//! | Group | The question | Why it is its own number |
//! |---|---|---|
//! | monolingual | query and memory in the same language | the baseline everything else is read against |
//! | cross-lingual | query and memory in *different* languages | the README's central claim, and the dimension models disagree on most |
//! | lexical | identifiers, error codes, paths, config keys | the only reason the n-gram channel exists |
//!
//! An aggregate would hide exactly what this is for. A model can lose six
//! points of cross-lingual recall and gain them back on lexical matching, and
//! the mean says nothing happened. That is not hypothetical: it is how a
//! model was nearly chosen for this project on a retrieval score assembled
//! mostly from tasks that were not multilingual at all.
//!
//! ## What this measures, and what it does not
//!
//! It measures this engine on this corpus. The corpus is small, hand-written,
//! and ours, so the numbers are comparable between two configurations run
//! against it and are not comparable with anything published.
//!
//! The cross-lingual group has a sharper limit worth stating plainly. Every
//! public cross-lingual benchmark — Belebele, MLQA, the bitext sets — is built
//! on *parallel* text: the same sentence in two languages. Real memories are
//! not translations of each other. So the corpus here deliberately is not
//! parallel either: each subject is written up in several languages, and each
//! of those memories states a *different fact* about the subject. A
//! cross-lingual query therefore has to reach a memory that shares its subject
//! and not its sentence, which is the thing we actually promise. It is still a
//! constructed corpus written by one author, and it cannot tell you what a real
//! workload does.
//!
//! ## What it measured when it was written
//!
//! 210 memories across ten languages, 137 queries. Same corpus, same depths,
//! the three shipped profiles:
//!
//! Every number below is from a clean workspace, and reruns reproduce them
//! exactly.
//!
//! | profile | model | cross nDCG@10 | cross recall@50 | mono nDCG@10 | lexical |
//! |---|---|---|---|---|---|
//! | `speed` | multilingual-e5-small | 0.167 | 0.707 | 0.989 | 1.000 |
//! | `balanced` (default) | multilingual-e5-base | 0.205 | 0.837 | 0.989 | 1.000 |
//! | `accuracy` | BGE-M3 | **0.357** | **0.962** | 0.983 | 1.000 |
//!
//! ## What it measured next
//!
//! Those numbers were fused with the constants the rank fusion literature
//! supplies: `k = 60`, every channel weighted alike. Both turned out to be
//! wrong for this shape of retrieval, and correcting them moved the weak group
//! further than changing the model did. On `balanced`:
//!
//! | fusion | cross nDCG@10 | cross recall@50 | mono nDCG@10 |
//! |---|---|---|---|
//! | k=60, equal weights | 0.2041 | 0.8372 | 0.9887 |
//! | k=60, lexical halved | 0.2245 | 0.8605 | 0.9892 |
//! | k=10, equal weights | 0.2404 | 0.8372 | 0.9940 |
//! | **k=10, lexical halved** | **0.3383** | 0.8512 | **0.9940** |
//!
//! The corrections compound, and neither is a trade: monolingual goes up.
//! Recall@50 barely moves, which is what you would expect -- `k` and the
//! weights reorder the candidates rather than change which ones there are.
//!
//! The `k` curve keeps improving all the way down (0.3599 at 5, 0.3874 at 1),
//! so this corpus can say that 60 is too flat for fifty-deep lists and cannot
//! say where the optimum is. `pamin_core::DEFAULT_K` explains why it stops at
//! ten rather than at the boundary the corpus prefers.
//!
//! ## And then the model, on the corrected fusion
//!
//! The two compound. Re-measuring the profiles with `k = 10` and the lexical
//! pair halved:
//!
//! | profile | model | cross nDCG@10 | cross recall@50 | mono | lexical |
//! |---|---|---|---|---|---|
//! | `balanced` | multilingual-e5-base, fp32 | 0.3383 | 0.8512 | 0.9940 | 1.000 |
//! | `accuracy`, as it was | BGE-M3, fp32 | 0.6720 | 0.9616 | 0.9940 | 1.000 |
//! | **`accuracy`, the default** | **BGE-M3, int8** | **0.6550** | **0.9500** | **0.9940** | **1.000** |
//!
//! Three times what the project shipped with before either change, and the
//! quantized weights give back 0.017 of it for a model that is 560 MB instead
//! of 2.2 GB and 35 ms a query instead of the order of magnitude the
//! full-precision export used to cost.
//!
//! Those three rows are a comparison between models, and they are frozen at
//! the fusion of the day they were taken: `k = 10` with the lexical pair at
//! half. The weight has been halved twice since, to a quarter and then to an
//! eighth, on evidence from two corpora this one cannot see -- see
//! `pamin_core::fusion`. What the bolded row's profile scores on the shipped
//! path today is what `FLOORS` records, 0.7773 cross-lingual; the model
//! ordering the table exists to show is unaffected, because the weight applies
//! to all three rows alike.
//!
//! The shape of the failures changes too, which the mean hides. On `balanced`
//! the worst cross-lingual queries score exactly zero: the relevant memory is
//! not in the top ten at all, and the results come back in the query's own
//! language about the wrong subject. On the default the worst is 0.246. There
//! is no longer a query on this corpus that misses outright.
//!
//! Three things fall out of that table, and none of them were visible before
//! it existed.
//!
//! **Cross-lingual retrieval is the weak channel by a wide margin.** On the
//! shipped default a cross-lingual query puts a relevant memory in the top ten
//! about a fifth as well as a same-language one does. The failures are
//! consistent in shape: the results come back in the query's own language,
//! about the wrong subject. That is a vector space clustering by language
//! rather than by meaning, and no amount of reranking fixes a memory that was
//! never returned.
//!
//! **The larger model buys almost all of it back**, at no measurable
//! monolingual cost — 0.357 against 0.205, and recall from 0.837 to 0.962,
//! while the same-language number moves by half a point in the other
//! direction. Whether that is worth its latency is a separate measurement.
//!
//! **The other two groups are at the ceiling and cannot detect much.**
//! Monolingual and lexical both sit at or near 1.000 on a corpus of this size,
//! so their floors below catch a collapse and nothing subtler. Making them
//! informative needs a larger corpus, not a different metric.

mod channels;
mod cold;
mod features;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use pamin_core::{Channel, Fusion};
use pamin_engine::{Depths, Engine, Write};
use pamin_index::{Access, Profile, Rerank, VectorIndex};
use pamin_store::Workspace;

use scoring::{NDCG_AT, RECALL_AT, Scores};

// `RECALL_AT` counts topics, not states. The index is keyed by topic state, so
// a topic with several versions occupies several results; the metric asks
// whether the topic was found, which means the search has to be given room for
// the duplicates before fifty distinct topics can come back -- hence
// `SEARCH_LIMIT` below.

/// How many results to ask for so that [`RECALL_AT`] distinct topics can fit.
const SEARCH_LIMIT: u32 = RECALL_AT as u32 * 2;

/// What each channel contributes before fusion, and how far the graph walks.
///
/// The defaults, deliberately: this measures the engine as shipped. Changing
/// them is an experiment to run against these numbers, not a way to improve
/// them.
const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

#[derive(serde::Deserialize)]
struct Memory {
    topic: String,
    content: String,
    /// The language the memory is written in. Recorded so a query can be
    /// classified as same-language or cross-language against it.
    language: String,
}

#[derive(serde::Deserialize)]
struct Query {
    /// `monolingual`, `cross_lingual`, or `lexical`.
    group: String,
    query: String,
    /// Topics that answer this query. Order does not matter; relevance is
    /// binary, because a hand-written corpus cannot honestly carry grades.
    relevant: Vec<String>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads model weights, and embeds the corpus"]
async fn retrieval_quality_by_group() {
    let corpus: Vec<Memory> = load("memories.json");
    let queries: Vec<Query> = load("queries.json");

    // A named workspace is reused; an unnamed one is thrown away. Embedding a
    // few hundred memories takes minutes, and a harness that pays that on every
    // run is a harness nobody runs twice in an afternoon.
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    // The default the product ships, so the floors below describe what a user
    // gets rather than what is quickest to measure.
    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).expect("a known profile");
    // The profile is part of the workspace identity, not just the corpus: an
    // index records the profile it was built with and refuses to open under
    // another, so two profiles measured against one corpus need two indexes.
    let project = format!("eval-{named}-{}", fingerprint(&corpus));

    let mut engine = Engine::open(
        &workspace,
        &project,
        profile,
        VectorIndex::default(),
        Access::ReadWrite,
    )
    .await
    .expect("open the engine");

    // The project name carries the corpus fingerprint, so a corpus that has
    // changed lands in a workspace that has never seen it and a corpus that has
    // not is written once and re-read. Writing is idempotent per topic either
    // way: the same content produces the same state.
    write_corpus(&mut engine, &corpus).await;

    // `FEATURES_OUT`: every candidate fusion saw, one row each, for fitting a
    // fusion offline. See `features`.
    if let Some(mut dump) = features::Features::from_env("own") {
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        for (index, query) in queries.iter().enumerate() {
            let hits = engine
                .search_fused(&query.query, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
            let asked = features::Asked {
                group: &query.group,
                question: &format!("q{index:03}"),
                text: &query.query,
                language: None,
                judged: relevant.len(),
            };
            dump.observe(&asked, &hits, WIDE, |topic| {
                Some(f64::from(relevant.contains(topic)))
            });
        }
        dump.finish(queries.len());
        return;
    }

    // `CHANNELS`: what each channel is worth alone, and what the fused list
    // looks like with each one taken away. See `channels`.
    //
    // This used to claim it was the only corpus of the three where the graph
    // channel returns anything, and therefore the only place that question has
    // an answer. Neither half survives inspection. The other two corpora name
    // their topics deliberately unlike their own text so that mention
    // derivation finds nothing -- so their graph is empty by design -- and this
    // corpus names its topics `<subject>_<language>`, which `name_sequence`
    // opens into a two-to-four token run that no memory's prose contains, so
    // its graph is empty too. `live_edges` now prints the census instead of
    // leaving the premise unstated.
    if std::env::var("CHANNELS").is_ok() {
        report_channels(&engine, &queries).await;
        return;
    }

    if let Some(variants) = channels::requested_variants() {
        let questions: Vec<(String, String)> = queries
            .iter()
            .map(|query| (query.query.clone(), query.group.clone()))
            .collect();
        channels::compare_reranked(
            &engine,
            "own corpus",
            &questions,
            SEARCH_LIMIT,
            DEPTHS,
            &variants,
            |index, into, hits| {
                let query = &queries[index];
                let mut ranked: Vec<String> = Vec::new();
                let mut seen = HashSet::new();
                for hit in hits {
                    if seen.insert(hit.topic.clone()) {
                        ranked.push(hit.topic.clone());
                    }
                }
                let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
                into.add(&ranked, relevant.len(), |topic| relevant.contains(topic));
            },
        )
        .await;
        return;
    }

    // `PASSAGES`: the same memories in a second project whose vectors embed
    // the topic's name, asked every question alongside this one. See
    // `channels::Paired`.
    if std::env::var("PASSAGES").is_ok() {
        let mut other = Engine::open(
            &workspace,
            &format!("{project}-named"),
            profile,
            VectorIndex::default(),
            Access::ReadWrite,
        )
        .await
        .expect("open the named project");
        write_corpus(&mut other, &corpus).await;
        assert_eq!(
            engine.passage(),
            pamin_index::Passage::Content,
            "the baseline project was built from content"
        );
        assert_eq!(
            other.passage(),
            pamin_index::Passage::Named,
            "the new project embeds names"
        );
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        let mut paired = channels::Paired::default();
        for query in &queries {
            let before = engine
                .search_fused(&query.query, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            let after = other
                .search_fused(&query.query, WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
            paired.observe(&query.group, &before, &after, |into, ranking| {
                into.add(ranking, relevant.len(), |topic| relevant.contains(topic));
            });
        }
        paired.report(&format!(
            "vectors embedding the topic name, own corpus, {named}"
        ));
        return;
    }

    // `ROUTES`: spending less on the reranker -- a cascade, and gates that
    // skip it -- priced against the shipped pass. See `reranking::Routes`.
    if std::env::var("ROUTES").is_ok() {
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        let tier = Rerank::default();
        let mut small = pamin_index::Reranker::load(Rerank::Fast, &workspace.root().join("models"))
            .expect("load the small reranker");
        let mut routes = reranking::Routes::default();
        for query in &queries {
            let hits = engine
                .search_reranked(&query.query, WIDE, DEPTHS, tier)
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits, tier);
            let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
            routes.observe(
                &query.group,
                &query.query,
                &hits,
                &replayed,
                &mut small,
                &query.query,
                |into, ranking| {
                    into.add(ranking, relevant.len(), |topic| relevant.contains(topic));
                },
            );
        }
        routes.report(&format!(
            "spending less on the {} reranker, own corpus, {named}",
            tier.name()
        ));
        return;
    }

    // `COLD`: where the first search after an idle release spends its time.
    // The harness's own engine is closed first, because it holds the index
    // this reopens and a read-write handle excludes every other. See `cold`.
    if std::env::var("COLD").is_ok() {
        let database = engine.database.clone();
        drop(engine);
        let cross: Vec<&str> = queries
            .iter()
            .filter(|query| query.group == "cross_lingual")
            .map(|query| query.query.as_str())
            .collect();
        cold::report(&database, &workspace, &project, profile, &cross).await;
        return;
    }

    // `RERANK_RULES` asks which candidates the reranker should be allowed to
    // move, which is a question the tier table cannot ask: it compares tiers
    // under one rule, and this compares rules under one tier.
    if std::env::var("RERANK_RULES").is_ok() {
        rerank_rules(&engine, &queries).await;
        return;
    }

    // `CONTEXT` asks what the reranker would say if it were shown what made a
    // candidate a candidate: its topic's name, and the memory whose edge
    // reached it. See `context`.
    if std::env::var("CONTEXT").is_ok() {
        context(&engine, &workspace, &queries).await;
        return;
    }

    if let Some(settings) = sweep() {
        println!("\n  setting              cross nDCG@10   mono nDCG@10   lexical nDCG@10");
        println!("  --------------------------------------------------------------------");
        let mut measured: Vec<(String, BTreeMap<String, Scores>)> = Vec::new();
        for (label, fusion) in settings {
            let mut groups: BTreeMap<String, Scores> = BTreeMap::new();
            let mut ignored = Vec::new();
            run(&engine, &queries, fusion, &mut groups, &mut ignored).await;
            println!(
                "  {label:<20}   {:>13.4}   {:>12.4}   {:>15.4}",
                groups["cross_lingual"].mean_ndcg(),
                groups["monolingual"].mean_ndcg(),
                groups["lexical"].mean_ndcg(),
            );
            measured.push((label, groups));
        }

        // Against the best cross-lingual row, query by query. This corpus has
        // 43 cross-lingual queries, so a tenth of a point is four of them, and
        // a table of means gives no way to see that.
        if let Some((best, top)) =
            statistics::baseline(&measured, |groups: &BTreeMap<String, Scores>| {
                groups["cross_lingual"].mean_ndcg()
            })
        {
            println!("\n  against {best}:");
            for (label, groups) in &measured {
                if label == best {
                    continue;
                }
                println!(
                    "  {label:<20}   {}",
                    statistics::compare(
                        &top["cross_lingual"].per_query,
                        &groups["cross_lingual"].per_query
                    )
                );
            }
        }
        println!();
        return;
    }

    // `TIERS`: every reranker tier through the shipped path, paired query by
    // query against the one that ships. This is the corpus with the relational
    // group, the one place a tier's cost to graph-reached answers is visible.
    if std::env::var("TIERS").is_ok() {
        report_tiers(&engine, &queries).await;
        return;
    }

    let (groups, mut worst) = shipped(&engine, &queries, Rerank::default()).await;
    report(&groups, &mut worst);

    // The floors describe one configuration, so they are only asserted against
    // it. Running another profile is an experiment, and an experiment that
    // scores below the shipped default is a result rather than a failure --
    // `speed` is genuinely worse cross-lingually and saying so is the point.
    if named != DEFAULT_PROFILE {
        println!("  {named} is not the default profile, so the floors are not asserted\n");
        return;
    }

    // Floors, not targets. Each is set below what this engine measured when the
    // harness was written, so it catches a regression rather than pinning a
    // number nobody chose. Raising one is a claim that the improvement is real
    // and repeatable; a run that beats a floor is not by itself either.
    for (group, floor_ndcg, floor_recall) in FLOORS {
        let scores = groups
            .get(*group)
            .unwrap_or_else(|| panic!("the corpus has no {group} queries"));
        assert!(
            scores.mean_ndcg() >= *floor_ndcg,
            "{group} nDCG@{NDCG_AT} fell to {:.4}, below the {floor_ndcg:.4} floor",
            scores.mean_ndcg()
        );
        assert!(
            scores.mean_recall() >= *floor_recall,
            "{group} recall@{RECALL_AT} fell to {:.4}, below the {floor_recall:.4} floor",
            scores.mean_recall()
        );
    }
}

/// The profile the floors below were measured against, and the product default.
const DEFAULT_PROFILE: &str = "accuracy";

/// Per group: nDCG@10 and recall@50 floors, for the default profile.
///
/// Roughly a tenth below what `balanced` measured when this was written, which
/// is wide enough that ordinary variation does not trip it and narrow enough
/// that losing a channel does. They are floors under the shipped default, not
/// targets and not a description of the best configuration — `accuracy` clears
/// the cross-lingual pair by a distance, which is the point of measuring all
/// three rather than pinning one.
const FLOORS: &[(&str, f64, f64)] = &[
    // 0.7773 / 0.9605 measured at the eighth lexical weight, against 0.7223 /
    // 0.9605 at the quarter it replaced: the change is ranking only, and this
    // group is where the weight is worth the most.
    ("cross_lingual", 0.69, 0.86),
    // 1.000 / 1.000 measured; at the ceiling, so this catches a collapse only.
    ("lexical", 0.95, 0.98),
    // 0.994 / 1.000 measured; likewise.
    ("monolingual", 0.94, 0.98),
    // 0.6068 / 1.0000 measured, at the graph weight this group's own sweep
    // chose. Roughly a tenth below, like the others.
    //
    // This is the one group whose floor guards a channel rather than the
    // pipeline: it is twenty queries whose answers are reachable across an
    // edge and hard to reach without one, so it falls when the graph channel
    // stops working -- which is exactly what the other three cannot see. It
    // is also twenty queries, so it is a collapse detector and not a
    // regression detector.
    ("relational", 0.54, 0.90),
];

/// Every query through `search_reranked` at `tier`, scored by group, with each
/// query's score, text and first three topics for the worst-first listing.
async fn shipped(
    engine: &Engine,
    queries: &[Query],
    tier: Rerank,
) -> (BTreeMap<String, Scores>, Vec<(f64, String, Vec<String>)>) {
    let mut groups: BTreeMap<String, Scores> = BTreeMap::new();
    let mut worst: Vec<(f64, String, Vec<String>)> = Vec::new();

    for query in queries {
        // `search_reranked`, because that is what `pamin search` calls and a
        // floor is only worth having over the path that ships. The sweep below
        // stays on `search_fused`, which is the one that takes a weighting.
        //
        // The tier reorders the top twenty of a hundred, so recall@50 cannot
        // move and only nDCG@10 can. On this corpus it is expected not to move
        // either -- nothing relevant here has ever sat below rank ten, which is
        // the finding that sent reranking to the external benchmark in the
        // first place -- but expected-not-to-move is still measured, because
        // otherwise nothing guards it.
        let hits = engine
            .search_reranked(&query.query, SEARCH_LIMIT, DEPTHS, tier)
            .await
            .expect("search");

        // A topic can appear through several of its states; the question is
        // whether the topic was found, so the first appearance is the rank.
        let mut ranked: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for hit in &hits {
            if seen.insert(hit.topic.clone()) {
                ranked.push(hit.topic.clone());
            }
        }

        let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
        let ndcg =
            groups
                .entry(query.group.clone())
                .or_default()
                .add(&ranked, relevant.len(), |topic| relevant.contains(topic));
        worst.push((
            ndcg,
            query.query.clone(),
            ranked.into_iter().take(3).collect(),
        ));
    }
    (groups, worst)
}

/// Each reranker tier through the shipped path, group by group, paired against
/// the tier that ships.
///
/// The external corpora answer which tier ranks passages best; they cannot say
/// what a tier does to an answer that is relevant because another memory
/// mentions it, since neither has an edge. This corpus's relational group is
/// the only place that cost can be read, and it is twenty queries, so the
/// paired counts are the number to read and the mean is not.
async fn report_tiers(engine: &Engine, queries: &[Query]) {
    let tiers = [Rerank::Off, Rerank::Fast, Rerank::Accurate];
    let mut priced: Vec<(Rerank, BTreeMap<String, Scores>, f64)> = Vec::new();
    for tier in tiers {
        let started = std::time::Instant::now();
        let (groups, _) = shipped(engine, queries, tier).await;
        let per_query = started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
        priced.push((tier, groups, per_query));
    }

    let (_, ships, _) = priced
        .iter()
        .find(|(tier, _, _)| *tier == Rerank::default())
        .expect("the shipped tier is one of those measured");
    for group in ships.keys() {
        println!("\n  {group}");
        println!(
            "  tier        nDCG@10   against {}",
            Rerank::default().name()
        );
        println!("  ----------------------------------------------------------------------");
        for (tier, groups, _) in &priced {
            let scores = &groups[group];
            let against = if *tier == Rerank::default() {
                String::new()
            } else {
                statistics::compare(&ships[group].per_query, &scores.per_query).to_string()
            };
            println!(
                "  {:<10}  {:>7.4}   {against}",
                tier.name(),
                scores.mean_ndcg()
            );
        }
    }
    println!();
    for (tier, _, per_query) in &priced {
        println!("  {:<10}  {per_query:.0} ms a query", tier.name());
    }
    println!(
        "  timed in a test harness, so read these against each other and not as a product figure\n"
    );
}

/// What the reranker is worth when it can see why a candidate is there.
///
/// **What prompted it.** The model is shown a memory's content and nothing
/// else -- not its topic's name, and not the edge that brought it into the
/// list. A relational answer is relevant *because* of that edge: "who gets
/// paged when a sev one escalates" is answered by `platform rota`, whose text
/// is "it pages ines on weekends and mikhail on weekdays", and the sentence
/// that connects the two is in another memory, "a sev one escalates to the
/// platform rota". Shown only the first, a cross-encoder has nothing to go on.
/// This is contextual retrieval's argument (prepend what the text leaves
/// implicit) applied at the one stage that reads text pairwise.
///
/// Four renderings of each movable candidate, scored by the shipped tier and
/// substituted by the shipped rule: the content alone, which must reproduce
/// the engine's own scores or the rest is a reconstruction error; the topic's
/// name before it; the memory the graph walked from after it, for a candidate
/// the graph reached; and both.
async fn context(engine: &Engine, workspace: &Workspace, queries: &[Query]) {
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let mut model = pamin_index::Reranker::load(tier, &workspace.root().join("models"))
        .expect("load the reranker");

    let labels = reranking::context_labels();
    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); labels.len()];
    let mut seeded = 0usize;
    for query in queries {
        let hits = engine
            .search_reranked(&query.query, WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);
        let (orders, shown) = reranking::in_context(&hits, &replayed, &mut model, &query.query);
        seeded += shown;

        let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
        for (order, into) in orders.iter().zip(&mut measured) {
            into.entry(query.group.clone())
                .or_default()
                .add(order, relevant.len(), |topic| relevant.contains(topic));
        }
    }
    reranking::report(
        &format!(
            "what the {} reranker is shown, own corpus; {seeded} candidates shown a seed",
            tier.name()
        ),
        &labels,
        reranking::shipped_context(),
        &measured,
    );
}

/// Every rule for using the reranker's scores, priced against the one that
/// ships, from one shipped run. See `reranking`.
///
/// **What prompted it.** The relational group scores lower through the shipped
/// path than through fusion alone: a relational answer is relevant because
/// *another* memory mentions it, which a cross-encoder reading the query and
/// that memory cannot see. Exempting graph-reached candidates from the model
/// was tried first and priced from this same replay: it recovered relational
/// (+0.0259, 4W/0L, p = 0.12) and cost cross-lingual (-0.0108, 0W/10L,
/// p = 0.0018), so it traded one group for another and was dropped. Blending
/// keeps fusion's opinion of every candidate rather than of some.
async fn rerank_rules(engine: &Engine, queries: &[Query]) {
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let rules = reranking::rules();

    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); rules.len()];
    for query in queries {
        let hits = engine
            .search_reranked(&query.query, WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);

        let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
        for ((_, rule), into) in rules.iter().zip(&mut measured) {
            into.entry(query.group.clone()).or_default().add(
                &replayed.order(*rule),
                relevant.len(),
                |topic| relevant.contains(topic),
            );
        }
    }

    reranking::report(
        &format!(
            "rules for the {} reranker's scores, own corpus",
            tier.name()
        ),
        &rules,
        reranking::shipped(&rules),
        &measured,
    );
}

/// Each channel alone, and each one removed, on the corpus this project wrote.
async fn report_channels(engine: &Engine, queries: &[Query]) {
    use pamin_core::Channel;

    /// Past four times the channel depth, so `take(limit)` cannot bite.
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;

    /// Every channel, so leaving one out is asked of all four.
    const CHANNELS: &[Channel] = &[
        Channel::LexicalSegmented,
        Channel::LexicalNgram,
        Channel::Vector,
        Channel::Graph,
    ];

    // The premise of every `graph` row below, taken before anything is scored
    // so that a zero is never mistaken for a measurement.
    let edges = channels::live_edges(engine).await;
    let total: i64 = edges.iter().map(|(_, count)| count).sum();
    println!("\n  live edges in this project: {total}");
    for (kind, count) in &edges {
        println!("    {kind:<14} {count:>8}");
    }
    // The `relational` group is in this corpus precisely so this cannot be
    // zero, and an assertion rather than a printed warning because a silent
    // zero is what turned a premise failure into a verdict about the design
    // once already. If mention derivation stops firing -- a change to
    // `name_sequence`, to the run bound, to the filter -- this fails here
    // instead of quietly handing the graph rows back to the corpus.
    assert!(
        total > 0,
        "no edges: the `relational` group's memories mention each other's topic names by \
         construction, so mention derivation has stopped asserting them and every graph row \
         below would be a property of this corpus rather than of the channel"
    );

    let mut alone: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut without: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut whole: BTreeMap<String, Scores> = BTreeMap::new();
    let mut lexical_agreement: Vec<f64> = Vec::new();

    // Every fusion setting worth pricing, scored from the same traces as the
    // rows above. One pass, the whole grid.
    let variants = channels::variants();
    let mut offline: Vec<BTreeMap<String, Scores>> =
        variants.iter().map(|_| BTreeMap::new()).collect();

    for query in queries {
        let hits = engine
            .search_fused(&query.query, WIDE, DEPTHS, Fusion::default())
            .await
            .expect("search");

        channels::enough_room(&hits, WIDE);
        channels::same_as_the_engine(&hits, &Fusion::default());

        let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
        // A topic can appear through several of its states; the question is
        // whether the topic was found, so the first appearance is the rank --
        // the same dedup the scoring arm does.
        let note = |into: &mut BTreeMap<String, Scores>, ranked: &[String]| {
            let mut deduped: Vec<String> = Vec::new();
            let mut seen = HashSet::new();
            for topic in ranked {
                if seen.insert(topic.clone()) {
                    deduped.push(topic.clone());
                }
            }
            into.entry(query.group.clone())
                .or_default()
                .add(&deduped, relevant.len(), |topic| relevant.contains(topic));
        };

        let each = channels::each_alone(&hits);
        for (channel, ranking) in &each {
            note(alone.entry(*channel).or_default(), ranking);
        }
        for missing in CHANNELS {
            let ranking = channels::as_if(&hits, &Fusion::default().without(*missing));
            note(without.entry(*missing).or_default(), &ranking);
        }
        note(
            &mut whole,
            &hits.iter().map(|hit| hit.topic.clone()).collect::<Vec<_>>(),
        );

        for ((_, fusion), into) in variants.iter().zip(&mut offline) {
            note(into, &channels::as_if(&hits, fusion));
        }

        if let (Some(segmented), Some(ngram)) = (
            each.get(&Channel::LexicalSegmented),
            each.get(&Channel::LexicalNgram),
        ) && let Some(tau) = channels::agreement(segmented, ngram)
        {
            lexical_agreement.push(tau);
        }
    }

    for group in whole.keys() {
        println!("\n  each channel on its own, {group}");
        println!("  channel               queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
        println!("  ---------------------------------------------------------------");
        for (channel, groups) in &alone {
            if let Some(scores) = groups.get(group) {
                // A graph row with no edges behind it is not a figure. Printing
                // the reason in the cell is what stops it being copied into a
                // table as though it were one.
                let premise = if *channel == Channel::Graph && total == 0 {
                    "   <- no edges; premise absent"
                } else {
                    ""
                };
                println!(
                    "  {:<20}   {:>7}   {:>7.4}   {:>9.4}{premise}",
                    format!("{channel:?}"),
                    scores.queries,
                    scores.mean_ndcg(),
                    scores.mean_recall()
                );
            }
        }
        if total == 0 && !alone.contains_key(&Channel::Graph) {
            println!(
                "  {:<20}   {:>7}   {:>7}   {:>9}   <- returned nothing on any query",
                "Graph", 0, "--", "--"
            );
        }
        println!(
            "  {:<20}   {:>7}   {:>7.4}   {:>9.4}",
            "all four fused",
            whole[group].queries,
            whole[group].mean_ndcg(),
            whole[group].mean_recall()
        );

        println!("\n  with one channel taken away, {group}:");
        for (channel, groups) in &without {
            if let Some(scores) = groups.get(group) {
                let premise = if *channel == Channel::Graph && total == 0 {
                    "   <- removing an empty channel; asserts nothing"
                } else {
                    ""
                };
                println!(
                    "  {:<20}   {}{premise}",
                    format!("{channel:?}"),
                    statistics::compare(&whole[group].per_query, &scores.per_query)
                );
            }
        }

        channels::sweep_table(
            group,
            &whole[group],
            &variants,
            &offline.iter().map(|row| row.get(group)).collect::<Vec<_>>(),
        );
    }

    channels::cross_validated(
        "own corpus",
        &whole.keys().map(String::as_str).collect::<Vec<_>>(),
        &whole,
        &variants,
        channels::shipped_row(&variants),
        &offline,
    );

    println!(
        "\n  the two lexical channels agree at Kendall tau {:.4} over {} queries\n",
        channels::mean(&lexical_agreement),
        lexical_agreement.len()
    );
}

/// The fusion settings to try when `SWEEP` is set, each labelled as printed.
///
/// First the two numbers fusion has, `k` and the lexical weight, which is the
/// pair this corpus settled once already. It is swept alongside the
/// cross-lingual harness rather than alone, because a setting that suits one
/// corpus and ruins the other is the outcome worth catching.
///
/// This is the corpus where a per-query rule could have been judged -- its three
/// groups are three different sets of queries, where the cross-lingual harness
/// scores one set twice. It is where the adaptive lexical weight was judged, and
/// it did not survive: the rule interpolated monotonically between the two
/// constants bounding it and never separated from them. The rows are gone with
/// the rule.
fn sweep() -> Option<Vec<(String, Fusion)>> {
    // `SWEEP=1` runs every row. Any other value keeps the rows whose label
    // contains it, because a row costs a pass over the whole query set and
    // re-checking one row should not cost twenty-four: `SWEEP="k=10 "` is one value of the
    // rank constant.
    let wanted = std::env::var("SWEEP").ok()?;
    let filter = (wanted != "1").then_some(wanted);
    let mut settings = Vec::new();
    for k in [5.0, 10.0, 20.0, 60.0] {
        for weight in [0.0, 0.125, 0.25, 0.5, 1.0] {
            settings.push((format!("k={k:.0} lex {weight:.3}"), fusion(k, weight)));
        }
    }
    if let Some(filter) = &filter {
        settings.retain(|(label, _)| label.contains(filter.as_str()));
        assert!(!settings.is_empty(), "SWEEP={filter:?} matched no row");
    }
    Some(settings)
}

fn fusion(k: f32, lexical: f32) -> Fusion {
    Fusion::default()
        .with_k(k)
        .with_weight(Channel::LexicalSegmented, lexical)
        .with_weight(Channel::LexicalNgram, lexical)
}

/// Scores every query under one fusion setting.
async fn run(
    engine: &Engine,
    queries: &[Query],
    fusion: Fusion,
    groups: &mut BTreeMap<String, Scores>,
    worst: &mut Vec<(f64, String, Vec<String>)>,
) {
    for query in queries {
        let hits = engine
            .search_fused(&query.query, SEARCH_LIMIT, DEPTHS, fusion.clone())
            .await
            .expect("search");

        let mut ranked: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for hit in &hits {
            if seen.insert(hit.topic.clone()) {
                ranked.push(hit.topic.clone());
            }
        }

        let relevant: HashSet<&str> = query.relevant.iter().map(String::as_str).collect();
        let ndcg =
            groups
                .entry(query.group.clone())
                .or_default()
                .add(&ranked, relevant.len(), |topic| relevant.contains(topic));
        worst.push((
            ndcg,
            query.query.clone(),
            ranked.into_iter().take(3).collect(),
        ));
    }
}

/// Writes every memory that is not already there, then runs the queue.
///
/// Skipping what exists is what makes a reused workspace mean the same thing
/// as a fresh one. Writing regardless appends a second identical version of
/// every topic, and the second run of the harness then measures a corpus twice
/// the size of the first — which is how this was found: the same profile
/// scored differently on consecutive runs, and the engine had not changed.
///
/// Drained once at the end rather than after each memory: the corpus is a bulk
/// import, and a drain per write rebuilds derived state far more often than
/// the data changes.
async fn write_corpus(engine: &mut Engine, corpus: &[Memory]) {
    let project = engine.project;
    let mut written = 0;

    for memory in corpus {
        let existing =
            pamin_store::repository::find_topic(engine.database.pool(), project, &memory.topic)
                .await
                .expect("look for the topic");
        if existing.is_some() {
            continue;
        }
        written += 1;
        engine
            .write(&Write {
                topic: &memory.topic,
                content: &memory.content,
                content_hash: &memory.content.len().to_string(),
                verdict: pamin_core::FilterDecision::Promoted,
                reason: "evaluation corpus",
                promoted: true,
                language: Some(&memory.language),
                language_confidence: None,
                observed_at: time::OffsetDateTime::now_utc(),
                validity: pamin_core::Validity::ALWAYS,
            })
            .await
            .unwrap_or_else(|error| panic!("writing {}: {error}", memory.topic));
    }

    let drained = engine
        .drain_cascade(pamin_engine::Owed::Everything)
        .await
        .expect("drain the cascade");
    assert_eq!(
        drained.pending, 0,
        "the corpus is not fully indexed: {} jobs still owed",
        drained.pending
    );
    if written > 0 {
        println!("  wrote {written} of {} memories", corpus.len());
    }
}

/// Prints the table this harness exists to produce.
fn report(groups: &BTreeMap<String, Scores>, worst: &mut [(f64, String, Vec<String>)]) {
    println!("\n  group           queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
    println!("  ------------------------------------------------");
    for (group, scores) in groups {
        println!(
            "  {:<14}  {:>7}   {:>7.4}   {:>9.4}",
            group,
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall()
        );
    }

    // The queries that went worst, because a mean says a group moved and these
    // say what moved it.
    worst.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!("\n  worst queries");
    for (ndcg, query, top) in worst.iter().take(8) {
        println!("  {ndcg:>6.3}  {query:<38}  → {}", top.join(", "));
    }
    println!();
}

fn load<T: serde::de::DeserializeOwned>(name: &str) -> Vec<T> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("corpus")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("reading {}: {error}", path.display());
    });
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parsing {name}: {error}"))
}

/// A short digest of the corpus, so a changed corpus gets a fresh project.
fn fingerprint(corpus: &[Memory]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for memory in corpus {
        for byte in memory.topic.bytes().chain(memory.content.bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}
