//! Where the first search after an idle release spends its time.
//!
//! A resident server gives back its indexes and both models after
//! `PAMIN_MODEL_IDLE`, so the next `pamin search` opens the project's index,
//! loads BGE-M3 and the reranker, and only then answers. That wait is the
//! latency a user sees after stepping away, and the parts of it are what
//! decide what is worth making concurrent. So each round here starts from what
//! a released server holds -- a database and nothing else -- and asks the
//! question the server asks, through `Engine::attached` and `search_reranked`
//! at the defaults `pamin search` passes.
//!
//! Three kinds of round, interleaved so a busy machine charges them alike:
//!
//! - **cold**: a fresh model registry, the index opened, one search at the
//!   shipped tier, then a second query on the now-warm engine;
//! - **serial**: the same, but the search asks for `off` first and the shipped
//!   tier second, which is the cold path with the reranker's load moved after
//!   retrieval -- what it was before `search_reranked` began loading the
//!   reranker at its top;
//! - **parts**: each model loaded alone, then both at once on two threads,
//!   and each model's first forward pass after its load.
//!
//! The process's other state is not reset: the operating system's page cache
//! holds the prepared model copies after the first round, as it does for a
//! server that released them half an hour ago on a machine with memory to
//! spare. A machine that has just booted pays reading them from disk as well,
//! which this does not measure.
//!
//! ## What it measured when it was written
//!
//! Seven runs of seven rounds, `accuracy` profile, `accurate` tier, on four
//! cores that other measurements were keeping two to three of busy the whole
//! time (load average 3.4 to 9.2). Each figure is the median of the seven
//! runs' medians, with the lowest and highest run median beside it:
//!
//! | step | median ms | run medians |
//! |---|---|---|
//! | open the index | 107 | 63-131 |
//! | first search, shipped tier | 2,240 | 1,733-2,755 |
//! | open + first search | 2,386 | 1,817-2,853 |
//! | the next query, warm | 618 | 549-690 |
//! | load BGE-M3 alone | 1,089 | 1,043-1,148 |
//! | load the `accurate` reranker alone | 977 | 916-1,011 |
//! | load both at once | 1,300 | 1,120-1,892 |
//! | first query embedding after a load | 40 | 35-41 |
//! | first rerank pass after a load | 582 | 505-669 |
//! | open + both, the reranker loaded after retrieval | 2,408 | 2,191-2,908 |
//!
//! So most of a cold search is the two loads run together, 1,300 ms, and then
//! what a warm search does, 618; the index adds 107, and the rest of the 2,386
//! is the loads and the first passes contending for cores. Loading the
//! reranker alongside retrieval rather than after it saved 262 ms at the
//! median of the seven runs' paired differences, which ranged from -67 to 545
//! -- positive in six of seven, and small next to the spread of the totals it
//! is taken from.

// Included by a harness that uses only some of it, like the other shared
// modules here.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use pamin_engine::{Depths, Engine, Models};
use pamin_index::{Access, Embedder, Profile, Rerank, Reranker, VectorIndex};
use pamin_store::{Database, Workspace};

/// What `pamin search` passes when a caller says nothing.
const LIMIT: u32 = 5;

/// Rounds of each kind, from `COLD_ROUNDS`. Seven by default: enough for a
/// median and a spread on a machine other work is sharing.
fn rounds() -> usize {
    std::env::var("COLD_ROUNDS")
        .ok()
        .and_then(|rounds| rounds.parse().ok())
        .filter(|rounds| *rounds > 0)
        .unwrap_or(7)
}

/// Timings of one kind, by name, in the order they were first recorded.
#[derive(Default)]
struct Samples(Vec<(&'static str, Vec<Duration>)>);

impl Samples {
    fn record(&mut self, name: &'static str, took: Duration) {
        match self.0.iter_mut().find(|(held, _)| *held == name) {
            Some((_, samples)) => samples.push(took),
            None => self.0.push((name, vec![took])),
        }
    }

    fn median(&self, name: &str) -> f64 {
        let mut samples = self.of(name);
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn of(&self, name: &str) -> Vec<f64> {
        self.0
            .iter()
            .find(|(held, _)| *held == name)
            .unwrap_or_else(|| panic!("nothing recorded as {name}"))
            .1
            .iter()
            .map(|took| took.as_secs_f64() * 1000.0)
            .collect()
    }

    fn print(&self) {
        println!("\n  step                                   median ms    min ms    max ms   n");
        println!("  --------------------------------------------------------------------------");
        for (name, _) in &self.0 {
            let samples = self.of(name);
            let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
            let max = samples.iter().copied().fold(0.0, f64::max);
            println!(
                "  {name:<38} {:>9.0}  {min:>8.0}  {max:>8.0}  {:>2}",
                self.median(name),
                samples.len()
            );
        }
    }
}

/// Runs every round and prints the table.
///
/// Two of `queries` are used: the first two the shipped tier reranks at the
/// default limit. The first is asked cold and the second of the warm engine,
/// so neither reads the other's cached vector or scores. Most queries are
/// not reranked at all -- the pass declines when no candidate it may move
/// would reach the caller's page -- and a round timed over one of those would
/// report a cold search without its most expensive step.
pub async fn report(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    queries: &[&str],
) {
    let tier = Rerank::default();
    let models = workspace.root().join("models");

    // Untimed: the first load on a machine writes each model's prepared copy,
    // which is a one-off and not what a released server pays.
    let queries = reranked(database, workspace, project, profile, tier, queries).await;
    println!(
        "\n  cold query {:?}, warm query {:?}",
        queries[0], queries[1]
    );

    let mut samples = Samples::default();
    let mut paired = Vec::new();
    for round in 0..rounds() {
        // The warmed round goes first every other time, so the two it is
        // paired with do not always run in the same order.
        let early = if round % 2 == 1 {
            Some(warmed(database, workspace, project, profile, tier, queries).await)
        } else {
            None
        };

        let (open, search, warm) = cold(database, workspace, project, profile, tier, queries).await;
        let cold_total = open + search;
        samples.record("cold: open the index", open);
        samples.record("cold: search, shipped tier", search);
        samples.record("cold: open + search", open + search);
        samples.record("warm: search, shipped tier", warm);

        let (open, off, then) = serial(database, workspace, project, profile, tier, queries).await;
        samples.record("serial: open the index", open);
        samples.record("serial: search, off", off);
        samples.record("serial: then the shipped tier", then);
        samples.record("serial: open + both", open + off + then);

        let (first, until, then) = match early {
            Some(early) => early,
            None => warmed(database, workspace, project, profile, tier, queries).await,
        };
        samples.record("warmed: open + search", first);
        samples.record("touched: until warm", until);
        samples.record("touched: then search", then);
        paired.push((cold_total.as_secs_f64() - first.as_secs_f64()) * 1e3);

        for (name, took) in parts(&models, profile, tier, queries) {
            samples.record(name, took);
        }
    }

    println!(
        "\n  cold search, own corpus, {} profile, {} tier, limit {LIMIT}, {} rounds",
        profile_name(profile),
        tier.name(),
        rounds()
    );
    samples.print();
    let saved = samples.median("serial: open + both") - samples.median("cold: open + search");
    println!("\n  loading the reranker alongside retrieval saves {saved:.0} ms at the median");
    paired.sort_by(f64::total_cmp);
    println!(
        "  warming at the request saves {:.0} ms at the median of {} paired rounds ({:.0} to {:.0})\n",
        paired[paired.len() / 2],
        paired.len(),
        paired[0],
        paired[paired.len() - 1]
    );
}

fn profile_name(profile: Profile) -> &'static str {
    match profile {
        Profile::Speed => "speed",
        Profile::Balanced => "balanced",
        Profile::Accuracy => "accuracy",
    }
}

/// The first two of `queries` that `tier` reranks, asked through the shipped
/// path at the default limit.
async fn reranked<'q>(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    tier: Rerank,
    queries: &[&'q str],
) -> [&'q str; 2] {
    let (engine, _) = open(database, workspace, project, profile).await;
    let mut found = Vec::new();
    for query in queries {
        let before = scored(&engine, tier);
        engine
            .search_reranked(query, LIMIT, Depths::default(), tier)
            .await
            .expect("search");
        if scored(&engine, tier) > before {
            found.push(*query);
            if let [first, second] = found[..] {
                return [first, second];
            }
        }
    }
    panic!("fewer than two queries are reranked at {}", tier.name());
}

/// Opens the project against a registry holding nothing, the way a server
/// that has released everything reopens it.
async fn open(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
) -> (Engine, Duration) {
    let started = Instant::now();
    let engine = Engine::attached(
        database.clone(),
        &Models::in_workspace(workspace),
        workspace,
        project,
        profile,
        VectorIndex::default(),
        Access::ReadWrite,
    )
    .await
    .expect("open the project");
    (engine, started.elapsed())
}

/// One shipped search, timed, asserting it did what it is timed as doing: hit
/// something, and hand the reranker at least `after` candidates in this
/// engine's lifetime. A search the reranker declined would be timed as a
/// rerank and cost nothing like one.
async fn search(engine: &Engine, query: &str, tier: Rerank, after: u64) -> Duration {
    let started = Instant::now();
    let hits = engine
        .search_reranked(query, LIMIT, Depths::default(), tier)
        .await
        .expect("search");
    let took = started.elapsed();
    assert!(!hits.is_empty(), "{query:?} found nothing");
    if tier != Rerank::Off {
        let scored = engine.reranked(tier).map_or(0, |counted| counted.scored);
        assert!(
            scored > after,
            "{query:?} was not reranked at {}, so it cannot stand for a search that is",
            tier.name()
        );
    }
    took
}

fn scored(engine: &Engine, tier: Rerank) -> u64 {
    engine.reranked(tier).map_or(0, |counted| counted.scored)
}

/// The product's cold path: open, then one search at the shipped tier, then a
/// different query on the warm engine.
async fn cold(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    tier: Rerank,
    [query, other]: [&str; 2],
) -> (Duration, Duration, Duration) {
    let (engine, open) = open(database, workspace, project, profile).await;
    assert!(
        engine.reranked(tier).is_none(),
        "the reranker was resident before the cold search"
    );
    let search_cold = search(&engine, query, tier, 0).await;
    let warm = search(&engine, other, tier, scored(&engine, tier)).await;
    (open, search_cold, warm)
}

/// The cold path with the server's warm-up, twice over.
///
/// First as it is when the search is itself the first request: a resident
/// server's `Session::warm` starts both models loading as the request
/// arrives, beside the index's open, and the search then waits for loads
/// already in flight. What is timed is open and search together, against the
/// `cold` round's.
///
/// Then as it is when anything else came first: the warm-up is started, the
/// time until both models are resident is taken -- how long an agent has to
/// spend reading before its search finds them -- and a search after that.
/// Each starts from a [`Models`] of its own, so nothing is loaded when it
/// begins, and the second asserts that the reranker was resident before its
/// search.
async fn warmed(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    tier: Rerank,
    [query, other]: [&str; 2],
) -> (Duration, Duration, Duration) {
    let first = {
        let models = Models::in_workspace(workspace);
        let started = Instant::now();
        let warming = {
            let models = models.clone();
            tokio::task::spawn_blocking(move || models.warm(profile, tier))
        };
        let engine = attached(database, &models, workspace, project, profile).await;
        search(&engine, query, tier, 0).await;
        let took = started.elapsed();
        warming.await.expect("the warm-up panicked").expect("warm");
        took
    };

    let models = Models::in_workspace(workspace);
    let started = Instant::now();
    let warming = {
        let models = models.clone();
        tokio::task::spawn_blocking(move || models.warm(profile, tier))
    };
    let engine = attached(database, &models, workspace, project, profile).await;
    warming.await.expect("the warm-up panicked").expect("warm");
    let until = started.elapsed();
    assert!(
        engine.reranked(tier).is_some(),
        "the warm-up finished without the reranker resident"
    );
    let then = search(&engine, other, tier, 0).await;
    (first, until, then)
}

/// The project against `models`, the way `Session::engine` opens it.
async fn attached(
    database: &Database,
    models: &Models,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
) -> Engine {
    Engine::attached(
        database.clone(),
        models,
        workspace,
        project,
        profile,
        VectorIndex::default(),
        Access::ReadWrite,
    )
    .await
    .expect("open the project")
}

/// The cold path with the reranker loaded only after retrieval: an `off`
/// search loads the embedder and retrieves, and the shipped tier asked next
/// finds the query's vector cached and loads the reranker.
async fn serial(
    database: &Database,
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    tier: Rerank,
    [query, _]: [&str; 2],
) -> (Duration, Duration, Duration) {
    let (engine, open) = open(database, workspace, project, profile).await;
    let off = search(&engine, query, Rerank::Off, 0).await;
    assert!(
        engine.reranked(tier).is_none(),
        "an `off` search loaded the reranker"
    );
    let then = search(&engine, query, tier, 0).await;
    (open, off, then)
}

/// Each model's load and first passes, outside the engine.
fn parts(
    models: &std::path::Path,
    profile: Profile,
    tier: Rerank,
    [query, other]: [&str; 2],
) -> Vec<(&'static str, Duration)> {
    let mut took = Vec::new();
    let mut time = |name, work: &mut dyn FnMut()| {
        let started = Instant::now();
        work();
        took.push((name, started.elapsed()));
    };

    let mut embedder = None;
    time("parts: load the embedder", &mut || {
        embedder = Some(Embedder::load(profile, models).expect("load the embedder"));
    });
    let mut embedder = embedder.expect("loaded");
    time("parts: first query embedding", &mut || {
        embedder.embed_query(query).expect("embed");
    });
    time("parts: second query embedding", &mut || {
        embedder.embed_query(other).expect("embed");
    });
    drop(embedder);

    // What a cold search hands it: up to the tier's depth of candidates.
    let documents: Vec<String> = (0..tier.depth())
        .map(|n| format!("{other} {n}: a memory the lexical channels did not find"))
        .collect();
    let documents: Vec<&str> = documents.iter().map(String::as_str).collect();
    let mut reranker = None;
    time("parts: load the reranker", &mut || {
        reranker = Some(Reranker::load(tier, models).expect("load the reranker"));
    });
    let mut reranker = reranker.expect("loaded");
    time("parts: first rerank pass", &mut || {
        reranker.rank(query, &documents).expect("rank");
    });
    time("parts: second rerank pass", &mut || {
        reranker.rank(other, &documents).expect("rank");
    });
    drop(reranker);

    // Dropped outside the timing: freeing the vocabulary is not loading it.
    let mut both = None;
    time("parts: load both at once", &mut || {
        both = Some(std::thread::scope(|scope| {
            let embedder = scope.spawn(|| Embedder::load(profile, models).expect("embedder"));
            let reranker = scope.spawn(|| Reranker::load(tier, models).expect("reranker"));
            (
                embedder.join().expect("the embedder's load panicked"),
                reranker.join().expect("the reranker's load panicked"),
            )
        }));
    });
    drop(both);
    took
}
