//! Retrieval on product search, graded by people, in three languages that are
//! not translations of each other.
//!
//! **What this corpus is here to settle.** Every fusion conclusion in this
//! repository rests on XQuAD-R, and XQuAD-R has a construction that produces
//! one of them by itself. Its two groups are the *same* 1,190 queries scored
//! against different answer keys -- the cross-lingual group's answers are the
//! ten translations of the sentence the same-language group's answer is -- so
//! any dial that strengthens lexical matching must help one group by exactly
//! the mechanism that hurts the other. That is why every dial there traces one
//! curve, and it is a property of the benchmark rather than of this fusion.
//!
//! The same-language half is the more suspect one, and it is the half that
//! justifies keeping the lexical channels at all. XQuAD-R is SQuAD's questions
//! translated, and SQuAD's questions are written *out of the answer's own
//! words* -- so a channel that matches wording is being asked a question built
//! to reward it. That is where the lexical pair's +0.1042 comes from, and
//! nothing here has ever checked it against queries somebody actually typed.
//!
//! Amazon ESCI is those queries. Real shopping searches against real product
//! listings, relevance judged by people on four ordered labels, in US English,
//! Spanish and Japanese -- three separate markets, not three versions of one
//! text. Nothing is parallel, nothing is translated, and no query was written
//! from the listing that answers it.
//!
//! **What it is not.** ESCI's judgements are *within* a locale: a Spanish query
//! is judged against Spanish listings and there is no link between a US
//! product and its Spanish counterpart. So this is three monolingual
//! benchmarks, not a cross-lingual one, and it cannot check the cross-lingual
//! side of anything. Saying so matters because the obvious misreading -- pool
//! the three locales, call it cross-lingual -- would produce a number with no
//! judgements under it.
//!
//! **Graded relevance, which no other corpus here has.** Exact, Substitute,
//! Complement, Irrelevant. Binary relevance cannot tell a ranking that put the
//! exact match first from one that put a substitute first, and both of this
//! project's other corpora are binary because a hand-written corpus cannot
//! honestly carry grades. See [`scoring::Scores::add_graded`].
//!
//! ## Running it
//!
//! Ignored by default. It pages the dataset out of the HuggingFace
//! datasets-server, caches each page, and embeds a few thousand listings.
//!
//! ```text
//! cargo test -p pamin-engine --test esci -- --ignored --nocapture
//! ```
//!
//! `CHANNELS=1` runs the channel diagnostic and the offline fusion sweep
//! instead of the search path, which is the arm this corpus was added for.

mod channels;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_core::{Channel, Fusion};
use pamin_engine::{Depths, Engine, Write};
use pamin_index::{Access, Profile, Rerank};
use pamin_store::Workspace;

use scoring::{NDCG_AT, RECALL_AT, Scores};

/// Where the rows come from.
///
/// The datasets-server rather than the parquet files, because reading parquet
/// would mean a new dependency for one test and `ci/budget.py` counts
/// dependencies. This returns JSON, which is already in the tree.
const ROWS: &str = "https://datasets-server.huggingface.co/rows\
                    ?dataset=tasksource%2Fesci&config=default&split=test";

/// How many rows a request asks for. The server's own maximum is 100.
const PER_PAGE: usize = 100;

/// How many pages to take.
///
/// A fixed number rather than "until enough queries", so two runs read the
/// same rows: the server pages by offset and the offsets are stable, but
/// "enough" would depend on how the labels happened to fall.
const PAGES: usize = 220;

/// The locales, which are the groups.
const LOCALES: [&str; 3] = ["us", "es", "jp"];

/// How deep to retrieve, and how deep each channel goes.
const DEPTH: usize = RECALL_AT + 1;

const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

const DEFAULT_PROFILE: &str = "accuracy";

/// A listing's key, so this corpus cannot collide with another in one
/// workspace.
const KEY: &str = "esci";

/// What each label is worth to the gain.
///
/// These are the dataset's own gains, not this repository's: `arXiv:2206.06588`
/// defines its Task 1 ranking metric with Exact 1, Substitute 0.1, Complement
/// 0.01 and Irrelevant 0, and those are the numbers below.
///
/// **One deviation, stated because it changes what a figure from here can be
/// compared with.** Task 1 is defined on the US locale alone; this applies its
/// gains to all three. The alternative is inventing a spacing per locale,
/// which is worse, but it does mean a Spanish or Japanese figure from this
/// harness has no published counterpart to be read against.
fn gain_of(label: &str) -> f64 {
    match label {
        "Exact" => 1.0,
        "Substitute" => 0.1,
        "Complement" => 0.01,
        "Irrelevant" => 0.0,
        other => panic!("unknown ESCI label {other}"),
    }
}

/// One judged (query, listing) pair, as the dataset gives it.
#[derive(serde::Deserialize)]
struct Row {
    query: String,
    query_id: u64,
    product_id: String,
    product_locale: String,
    esci_label: String,
    small_version: u8,
    product_text: Option<String>,
}

/// One listing, once, however many queries judged it.
struct Listing {
    key: String,
    text: String,
    locale: String,
}

/// One query and its judged listings.
struct Query {
    locale: String,
    text: String,
    /// Key to gain, for every listing this query was judged against.
    judged: HashMap<String, f64>,
}

impl Query {
    /// The gains a perfect ranking would have earned, descending.
    fn ideal(&self) -> Vec<f64> {
        let mut gains: Vec<f64> = self.judged.values().copied().collect();
        gains.sort_by(|left, right| right.partial_cmp(left).expect("no NaN gains"));
        gains
    }
}

struct Corpus {
    listings: Vec<Listing>,
    queries: Vec<Query>,
}

impl Corpus {
    /// Pages the dataset in, caches each page, and assembles it.
    ///
    /// Only `small_version` rows, which is the dataset's own reduced task-one
    /// subset, and only the locales above. The last query id seen in each
    /// locale is dropped, because a query whose listings straddle the last
    /// page fetched would be scored against part of its own judgements --
    /// which is a recall failure invented by the loader.
    fn load() -> Self {
        let dir = dataset_dir();
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));

        let mut rows: Vec<Row> = Vec::new();
        for page in 0..PAGES {
            let offset = page * PER_PAGE;
            let path = dir.join(format!("page-{offset:07}.json"));
            download(&path, &format!("{ROWS}&offset={offset}&length={PER_PAGE}"));

            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            let body: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));
            let page_rows = body["rows"]
                .as_array()
                .unwrap_or_else(|| panic!("{} has no rows: {text:.200}", path.display()));
            if page_rows.is_empty() {
                break;
            }
            for row in page_rows {
                let row: Row = serde_json::from_value(row["row"].clone())
                    .unwrap_or_else(|error| panic!("a row of {}: {error}", path.display()));
                rows.push(row);
            }
        }

        // The last query in each locale is incomplete by construction.
        let mut last: HashMap<&str, u64> = HashMap::new();
        for row in &rows {
            last.insert(
                LOCALES
                    .iter()
                    .find(|locale| **locale == row.product_locale)
                    .copied()
                    .unwrap_or(""),
                row.query_id,
            );
        }

        let mut listings: BTreeMap<String, Listing> = BTreeMap::new();
        let mut queries: BTreeMap<(String, u64), Query> = BTreeMap::new();

        for row in &rows {
            if row.small_version != 1 || !LOCALES.contains(&row.product_locale.as_str()) {
                continue;
            }
            if last.get(row.product_locale.as_str()) == Some(&row.query_id) {
                continue;
            }
            // A listing with no text cannot be indexed and cannot be judged
            // against, so the pair is dropped rather than indexed empty.
            let Some(text) = row
                .product_text
                .as_deref()
                .filter(|it| !it.trim().is_empty())
            else {
                continue;
            };

            let key = format!("{KEY}-{}:{}", row.product_locale, row.product_id);
            listings.entry(key.clone()).or_insert_with(|| Listing {
                key: key.clone(),
                // Listings run to thousands of characters of bullet points.
                // Cut on a character boundary, at what the embedders' window
                // holds, so a long listing is truncated rather than refused.
                text: text.chars().take(1_200).collect(),
                locale: row.product_locale.clone(),
            });

            queries
                .entry((row.product_locale.clone(), row.query_id))
                .or_insert_with(|| Query {
                    locale: row.product_locale.clone(),
                    text: row.query.clone(),
                    judged: HashMap::new(),
                })
                .judged
                .insert(key, gain_of(&row.esci_label));
        }

        // A query with nothing judged relevant cannot be scored -- it would
        // score zero whatever came back, which grades the corpus and not the
        // search.
        let queries: Vec<Query> = queries
            .into_values()
            .filter(|query| query.judged.values().any(|gain| *gain > 0.0))
            .collect();

        Self {
            listings: listings.into_values().collect(),
            queries,
        }
    }

    fn fingerprint(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for listing in &self.listings {
            for byte in listing.key.bytes().chain(listing.text.bytes()) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{hash:016x}")
    }

    fn describe(&self) {
        println!(
            "  {} listings, {} judged queries",
            self.listings.len(),
            self.queries.len()
        );
        for locale in LOCALES {
            let queries = self
                .queries
                .iter()
                .filter(|query| query.locale == locale)
                .count();
            let listings = self
                .listings
                .iter()
                .filter(|listing| listing.locale == locale)
                .count();
            let judged: usize = self
                .queries
                .iter()
                .filter(|query| query.locale == locale)
                .map(|query| query.judged.len())
                .sum();
            println!(
                "    {locale}: {queries} queries, {listings} listings, \
                 {:.1} judged a query",
                judged as f64 / queries.max(1) as f64
            );
        }
    }
}

fn dataset_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("ESCI_DIR") {
        return PathBuf::from(dir);
    }
    eval_home().join("esci")
}

fn eval_home() -> PathBuf {
    std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"))
}

/// Fetches one page if it is not already cached.
///
/// Through `curl` for the reason the other harnesses give: a dependency added
/// for a test CI never runs is a dependency the whole project carries. Lands
/// under a partial name and is renamed on success, so an interrupted page is
/// not mistaken for a complete one.
fn download(path: &Path, url: &str) {
    if path.exists() {
        return;
    }
    let partial = path.with_extension("part");
    let status = Command::new("curl")
        .args(["-sSLf", "--max-time", "300", "-o"])
        .arg(&partial)
        .arg(url)
        .status();
    assert!(
        matches!(status, Ok(status) if status.success()),
        "could not fetch {url}\n\
         The dataset is not vendored: ESCI is Amazon's, published under\n\
         apache-2.0 on HuggingFace, and this fetches it at run time. Make\n\
         `curl` and datasets-server.huggingface.co reachable, or point\n\
         ESCI_DIR at a directory of cached pages."
    );
    std::fs::rename(&partial, path).expect("name the downloaded page");
}

fn profile() -> (String, Profile) {
    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).unwrap_or_else(|| panic!("unknown profile {named}"));
    (named, profile)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads the dataset and model weights, and embeds the corpus"]
async fn search_ranks_product_listings() {
    let corpus = Corpus::load();
    corpus.describe();

    let (named, profile) = profile();
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    let project = format!("esci-{named}-{}", corpus.fingerprint());
    let engine = Engine::open(&workspace, &project, profile, Access::ReadWrite)
        .await
        .expect("open the engine");

    write_corpus(&engine, &corpus).await;

    if std::env::var("CHANNELS").is_ok() {
        report_channels(&engine, &corpus, &named).await;
        return;
    }

    let mut groups: BTreeMap<String, Scores> = BTreeMap::new();
    let started = std::time::Instant::now();
    for query in &corpus.queries {
        let hits = engine
            .search_reranked(&query.text, DEPTH as u32, DEPTHS, Rerank::default())
            .await
            .expect("search");
        let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
        score(
            groups.entry(query.locale.clone()).or_default(),
            query,
            &ranked,
        );
    }
    let per_query_ms =
        started.elapsed().as_secs_f64() * 1000.0 / corpus.queries.len().max(1) as f64;

    println!("\n  the shipped search path, {named}");
    println!("  locale   queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
    println!("  ----------------------------------------------");
    for (locale, scores) in &groups {
        println!(
            "  {locale:<6}   {:>7}   {:>7.4}   {:>9.4}",
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall()
        );
    }
    println!("  {per_query_ms:.0} ms per query\n");

    // No floors. They belong under a figure somebody has read, and this is the
    // first run of this corpus -- a floor written now would be a guess with an
    // assertion around it.
    println!("  no floors are asserted on this corpus yet: see the note in the source\n");
}

/// Scores one ranking for one query, on this corpus's graded judgements.
fn score(into: &mut Scores, query: &Query, ranked: &[String]) {
    into.add_graded(ranked, &query.ideal(), |key| {
        query.judged.get(key).copied().unwrap_or(0.0)
    });
}

/// The channel diagnostic and the offline fusion sweep.
///
/// The arm this corpus was added for. Same reconstruction the other two
/// harnesses use -- see [`channels`] -- so the settings priced here are the
/// same settings priced there and the comparison across corpora is a
/// comparison.
async fn report_channels(engine: &Engine, corpus: &Corpus, named: &str) {
    /// Past four times the channel depth, so `take(limit)` cannot bite.
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;

    const CHANNELS: &[Channel] = &[
        Channel::LexicalSegmented,
        Channel::LexicalNgram,
        Channel::Vector,
        Channel::Graph,
    ];

    let mut alone: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut without: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut whole: BTreeMap<String, Scores> = BTreeMap::new();

    let variants = channels::variants();
    let mut offline: Vec<BTreeMap<String, Scores>> =
        variants.iter().map(|_| BTreeMap::new()).collect();

    for query in &corpus.queries {
        let hits = engine
            .search_fused(&query.text, WIDE, DEPTHS, Fusion::default())
            .await
            .expect("search");

        channels::enough_room(&hits, WIDE);
        channels::same_as_the_engine(&hits, &Fusion::default());

        for (channel, ranking) in &channels::each_alone(&hits) {
            score(
                alone
                    .entry(*channel)
                    .or_default()
                    .entry(query.locale.clone())
                    .or_default(),
                query,
                ranking,
            );
        }
        for missing in CHANNELS {
            let ranking = channels::as_if(&hits, &Fusion::default().without(*missing));
            score(
                without
                    .entry(*missing)
                    .or_default()
                    .entry(query.locale.clone())
                    .or_default(),
                query,
                &ranking,
            );
        }

        let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
        score(
            whole.entry(query.locale.clone()).or_default(),
            query,
            &ranked,
        );

        for ((_, fusion), into) in variants.iter().zip(&mut offline) {
            score(
                into.entry(query.locale.clone()).or_default(),
                query,
                &channels::as_if(&hits, fusion),
            );
        }
    }

    for locale in LOCALES {
        let Some(group) = whole.get(locale) else {
            continue;
        };

        println!("\n  each channel on its own, {locale}, {named}");
        println!("  channel               queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
        println!("  ---------------------------------------------------------------");
        for (channel, groups) in &alone {
            let Some(scores) = groups.get(locale) else {
                continue;
            };
            println!(
                "  {:<20}   {:>7}   {:>7.4}   {:>9.4}",
                format!("{channel:?}"),
                scores.queries,
                scores.mean_ndcg(),
                scores.mean_recall()
            );
        }
        println!(
            "  {:<20}   {:>7}   {:>7.4}   {:>9.4}",
            "all four fused",
            group.queries,
            group.mean_ndcg(),
            group.mean_recall()
        );

        println!("\n  with one channel taken away, {locale}:");
        for missing in CHANNELS {
            let Some(scores) = without.get(missing).and_then(|it| it.get(locale)) else {
                continue;
            };
            println!(
                "  {:<20}   {:>+.4}  {}",
                format!("{missing:?}"),
                scores.mean_ndcg() - group.mean_ndcg(),
                statistics::compare(&group.per_query, &scores.per_query)
            );
        }

        channels::sweep_table(
            &format!("{locale}, {named}"),
            group,
            &variants,
            &offline
                .iter()
                .map(|row| row.get(locale))
                .collect::<Vec<_>>(),
        );
    }
    channels::cross_validated(
        &format!("ESCI, {named}"),
        &LOCALES,
        &whole,
        &variants,
        &offline,
    );
    println!();
}

/// Writes whatever is not already there.
///
/// Skipping what exists is what makes this resumable, and this corpus is large
/// enough that it matters: a run interrupted part way through embedding
/// continues rather than starting over.
async fn write_corpus(engine: &Engine, corpus: &Corpus) {
    let project = engine.project;
    let mut written = 0usize;

    for listing in &corpus.listings {
        let existing =
            pamin_store::repository::find_topic(engine.database.pool(), project, &listing.key)
                .await
                .expect("look for the topic");
        if existing.is_some() {
            continue;
        }
        written += 1;
        engine
            .write(&Write {
                topic: &listing.key,
                content: &listing.text,
                content_hash: &listing.text.len().to_string(),
                verdict: pamin_core::FilterDecision::Promoted,
                reason: "evaluation corpus",
                promoted: true,
                language: Some(&listing.locale),
                language_confidence: None,
                observed_at: time::OffsetDateTime::now_utc(),
                validity: pamin_core::Validity::ALWAYS,
            })
            .await
            .unwrap_or_else(|error| panic!("writing {}: {error}", listing.key));
        if written.is_multiple_of(2_000) {
            println!("  wrote {written} listings");
        }
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
        println!("  wrote {written} of {} listings", corpus.listings.len());
    }
}
