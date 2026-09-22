//! Cross-lingual recall, measured on somebody else's benchmark.
//!
//! [`retrieval.rs`](retrieval.rs) measures this engine on a corpus this
//! project wrote. That corpus settled the fusion constants and the default
//! model, and it has reached the end of what it can say: its own notes record
//! that two of its three groups sit at 1.000 and that "making them informative
//! needs a larger corpus, not a different metric". It also cannot answer the
//! question a reranker poses, because on 210 memories no relevant memory ever
//! falls below rank ten, so there is nothing left for a second pass to fix.
//!
//! This harness is the external half. It runs XQuAD-R, the retrieval form of
//! XQuAD published with LAReQA: 240 parallel paragraphs in eleven languages,
//! split into sentences, with 1,190 questions whose answer sentence is marked
//! in every language. A question asked in one language is answered by the same
//! sentence in all eleven, which makes "a query in one language reaches a
//! memory written in another" a thing with a number attached rather than a
//! claim in a README.
//!
//! ## Two tests, one corpus
//!
//! | test | what runs | what it answers |
//! |---|---|---|
//! | `the_model_reaches_across_languages` | the embedder alone, compared against every sentence in the pool | how far the embedding space itself gets |
//! | `search_reaches_across_languages` | the shipped search path: four channels, fusion, graph | how far the product gets |
//!
//! The gap between them is the fusion layer's net effect on cross-lingual
//! recall, which nothing here had measured. Both are wanted because either
//! alone misleads: the first cannot see a channel that dilutes the ranking,
//! and the second cannot separate a weak model from a weak fusion.
//!
//! The first deliberately does not go through the vector index. An approximate
//! index answers a slightly different question than the model does, and a
//! recall loss that could have come from either is a measurement of neither;
//! `pamin-index`'s own `recall.rs` is where the approximation is measured.
//!
//! ## Two groups, reported separately, never summed
//!
//! | group | the relevant sentences | why it is its own number |
//! |---|---|---|
//! | `same_language` | the answer sentence in the query's own language | the baseline the other is read against |
//! | `cross_lingual` | the answer sentence in the other ten languages, with the query's own removed from the ranking | the claim this project makes |
//!
//! Removing the same-language answer from the ranking is what makes the second
//! group honest. Left in, it takes a top rank on nearly every query and the
//! group scores well while answering the wrong question.
//!
//! Alongside the two metrics is a third number: how many relevant sentences
//! come back inside the shortlist but *below* rank ten. That is the whole
//! space a reranker could work in, and it is reported because a project that
//! keeps being asked whether to add one should be able to answer from a
//! measurement.
//!
//! ## What this measures, and what it does not
//!
//! XQuAD-R is parallel text: the eleven versions of a sentence are
//! translations of each other. Real memories are not, which is exactly the
//! limitation `retrieval.rs` was built to avoid, and this harness does not
//! avoid it. The two are complements. This one is external, published, larger
//! by two orders of magnitude, and comparable with other people's numbers; that
//! one is not parallel and is closer to the shape of the workload. A change
//! that helps on one and hurts on the other is a result, not a contradiction.
//!
//! ## What it measured when it was written
//!
//! 13,014 sentences in eleven languages, 1,190 queries, the default profile:
//!
//! | | group | nDCG@10 | recall@50 | relevant below rank 10 | queries with any |
//! |---|---|---|---|---|---|
//! | the model | cross-lingual | 0.6338 | 0.8951 | 3,273 | 965 of 1,190 |
//! | the model | same-language | 0.6748 | 0.9563 | 115 | 115 of 1,190 |
//! | fusion alone | cross-lingual | 0.6077 | 0.8960 | 3,598 | 1,062 of 1,190 |
//! | fusion alone | same-language | 0.7556 | 0.9580 | 74 | 74 of 1,190 |
//! | **the product** | cross-lingual | **0.6480** | 0.8960 | 3,097 | 1,009 of 1,190 |
//! | **the product** | same-language | **0.7495** | 0.9580 | 82 | 82 of 1,190 |
//!
//! "The product" is `search_reranked` at the default tier, which is what
//! `pamin search` calls. "Fusion alone" is `search_fused`, one stage short of
//! it. Both rows are here because the difference between them is the
//! reranker's, and for a while only the shorter one was measured.
//!
//! The two model rows are older than the other four: they predate the batching
//! fix that made `embed_passages` deterministic, and re-taken on deterministic
//! vectors the model scores 0.6335 / 0.8981 and 0.6787 / 0.9529 -- see
//! `MODEL_FLOORS`, which carries both pairs and why the difference is itself a
//! finding. Comparisons against the product below are read off the re-taken
//! pair.
//!
//! **Fusion is not one effect, it is two opposite ones, and they cancel in any
//! average.** It costs no recall at all -- 0.8951 cross-lingually for the
//! model and 0.8960 fused -- so the candidates the model reaches are still
//! there. What changes is the order, and it changes in opposite directions:
//! same-language nDCG@10 goes from 0.6748 to **0.7556**, because a question
//! and its answer sentence in one language share words and the two lexical
//! channels find them where a 1024-dimensional cosine does not; cross-lingual
//! nDCG@10 goes from 0.6338 to **0.6077**, because those same two channels
//! have nothing to match on across languages and spend part of the fused list
//! on the query's own language about the wrong subject.
//!
//! The reranker then buys back more than fusion cost cross-lingually, +0.0403,
//! and takes 0.0061 off same-language doing it. Both tiers and the latency they
//! cost are in the ADR; `TIERS=1` reproduces the comparison.
//!
//! **The whole stack now outranks the embedding model on both groups**, 0.6480
//! against 0.6335 cross-lingual and 0.7495 against 0.6787 same-language. It did
//! not when the lexical weight was a quarter: the product scored 0.6097
//! cross-lingual then, *below* the model it is built on, with the reranker
//! spending 226 ms a query buying back dilution fusion had introduced. That was
//! the open question this harness was written to answer, and halving the weight
//! is what answered it.
//!
//! How much of the list the lexical channels spend is what the fusion weight
//! decides, and this corpus is where the question was first visible. It is not
//! where it was settled: on parallel text the two groups are the same queries
//! scored twice, so every gain on one shows up as a loss on the other and the
//! corpus cannot arbitrate. MIRACL Swahili can -- 482 human-judged queries over
//! 131,924 real passages, one language -- and it put the eighth ahead of every
//! other weight tried, with the quarter scoring *below* the vector channel on
//! its own. The full three-corpus table is in `pamin_core::fusion`.
//!
//! Neither number is visible from one test, and neither is visible from the
//! corpus this project wrote, where both groups sat near the ceiling. What to
//! do about it is a sweep rather than a conclusion: the fusion weights were
//! settled where lexical carried signal for every query, and here it carries
//! signal for half of them and noise for the other half.
//!
//! The product row is insensitive to how the index is segmented, which is the
//! other thing worth knowing from it. Under the fusion that shipped before
//! this, the same corpus scored 0.4190 / 0.8558 in one segment with no graph
//! over it, 0.4266 / 0.8517 in six, and 0.4238 / 0.8547 in the four the engine
//! picks for a collection this size -- all within the third decimal of each
//! other, while a query went from 208 ms to 83. Segmenting buys latency and
//! costs no accuracy, which is not what approximate search usually trades.
//!
//! The last two columns are the ones that could not be obtained from the
//! corpus this project wrote. There, across all 137 queries, the count was
//! zero: every relevant memory was already inside the top ten, so no second
//! pass over the shortlist could have improved anything and the measurement
//! that said reranking did not help was measuring the corpus rather than the
//! reranker. Here four fifths of the cross-lingual queries leave a relevant
//! sentence stranded below rank ten.
//!
//! ## Running it
//!
//! Ignored by default: they download a dataset, download model weights, embed
//! thirteen thousand sentences, and for the second provision PostgreSQL and
//! index them.
//!
//! ```text
//! cargo test -p pamin-engine --test crosslingual -- --ignored --nocapture
//! ```
//!
//! The dataset is fetched with `curl` into `$PAMIN_EVAL_HOME/xquad-r`, or into
//! `$LAREQA_DIR` if that is set. It is not vendored: it is CC-BY-SA-4.0 and
//! this repository is Apache-2.0. Setting `PAMIN_EVAL_HOME` also keeps the
//! embeddings and the indexed workspace between runs, which is the difference
//! between minutes and most of an hour.
//!
//! `XQUAD_ALL_QUERIES=1` asks every question in all eleven languages, 13,090
//! queries. The default asks each question in one language, rotating through
//! the eleven, which is 1,190 queries covering every language and every
//! question and takes a eleventh of the time.

mod channels;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_core::{Channel, Fusion};
use pamin_engine::{Depths, Engine, Write};
use pamin_index::{Access, Embedder, Profile, Rerank};
use pamin_store::Workspace;

/// The languages XQuAD-R covers, in the order the rotation walks them.
const LANGUAGES: [&str; 11] = [
    "ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh",
];

/// Where the dataset comes from.
///
/// The published release, not the HuggingFace mirror of it: the mirror carries
/// XQuAD's paragraphs rather than LAReQA's sentence-level candidate pool, which
/// is a different and much easier task.
const SOURCE: &str =
    "https://raw.githubusercontent.com/google-research-datasets/lareqa/master/xquad-r";

use scoring::{NDCG_AT, RECALL_AT, Scores};
/// How deep to retrieve.
///
/// One more than [`RECALL_AT`], because the cross-lingual group drops the
/// query's own language from the ranking and still needs fifty left.
const DEPTH: usize = RECALL_AT + 1;

/// What each channel contributes before fusion, and how far the graph walks.
///
/// The shipped defaults, so the second test measures the product rather than a
/// configuration invented for the benchmark.
const DEPTHS: Depths = Depths {
    channel: 50,
    graph: 2,
};

/// The profile the floors were measured against, and the product default.
const DEFAULT_PROFILE: &str = "accuracy";

/// Per group: the nDCG@10 and recall@50 floors for the embedding space.
///
/// Floors under what was measured -- 0.6335 / 0.8981 cross-lingual and
/// 0.6787 / 0.9529 same-language -- by roughly a tenth, which is wide enough
/// that ordinary variation does not trip them and narrow enough that a
/// weaker model does. Floors, not targets: a run that beats one is not by
/// itself a reason to raise it.
///
/// Re-taken on deterministic vectors. This arm calls `embed_passages`, and
/// on the default profile a batch used to perturb every vector in it, so the
/// figures here described embeddings the product no longer produces. They
/// moved by less than this harness's own run-to-run spread -- the previous
/// pair was 0.6338 / 0.8951 and 0.6748 / 0.9563 -- which is the finding
/// rather than a reason to skip the re-run: the perturbation was systematic
/// enough to leave the ranking alone, and that is why nothing caught it.
const MODEL_FLOORS: &[(&str, f64, f64)] =
    &[("cross_lingual", 0.57, 0.80), ("same_language", 0.60, 0.86)];

// ---------------------------------------------------------------------------
// The corpus
// ---------------------------------------------------------------------------

/// One candidate sentence, and the key that identifies it everywhere.
struct Sentence {
    /// `language:paragraph:index`, unique across the whole pool. Also the topic
    /// name the second test writes it under, which is why it has to survive
    /// segmentation as one token and match nothing in the text.
    key: String,
    text: String,
    language: &'static str,
}

/// One question, and the sentence that answers it in each language.
///
/// Assembled by the dataset's own question id, which is the same string in all
/// eleven files and is the only thing that aligns them: the sentence splits are
/// not parallel, so a Thai paragraph holds 852 sentences where the German one
/// holds 1,276 and nothing can be matched by position.
struct Question {
    /// The question text, per language.
    asked: HashMap<&'static str, String>,
    /// The key of the answering sentence, per language.
    answers: HashMap<&'static str, String>,
}

/// Everything loaded from the dataset.
struct Corpus {
    sentences: Vec<Sentence>,
    questions: Vec<Question>,
}

/// One query: a question, and the language it is asked in.
struct Query<'a> {
    question: &'a Question,
    language: &'static str,
}

impl Query<'_> {
    fn text(&self) -> &str {
        &self.question.asked[self.language]
    }

    /// The relevant sentence keys for one group, and what to drop first.
    ///
    /// The cross-lingual group answers "can a query reach a memory written in
    /// another language", so the memory written in its own language is removed
    /// from the ranking rather than merely uncounted. Left in the ranking it
    /// occupies a top position on nearly every query, and the group reports a
    /// number that is mostly about same-language retrieval.
    fn relevant(&self, group: &str) -> (HashSet<&str>, Option<&str>) {
        let own = self.question.answers[self.language].as_str();
        match group {
            "same_language" => (HashSet::from([own]), None),
            "cross_lingual" => (
                self.question
                    .answers
                    .iter()
                    .filter(|(language, _)| **language != self.language)
                    .map(|(_, key)| key.as_str())
                    .collect(),
                Some(own),
            ),
            other => panic!("unknown group {other}"),
        }
    }
}

/// The groups, in the order they are reported.
const GROUPS: [&str; 2] = ["cross_lingual", "same_language"];

impl Corpus {
    /// Reads the eleven language files, fetching them if they are not there.
    fn load() -> Self {
        let dir = dataset_dir();
        fetch(&dir);

        let mut sentences = Vec::new();
        // Keyed by question id, because that is what is shared across the
        // eleven files; nothing else in them is.
        let mut asked: HashMap<String, HashMap<&'static str, String>> = HashMap::new();
        let mut answers: HashMap<String, HashMap<&'static str, String>> = HashMap::new();

        for language in LANGUAGES {
            let path = dir.join(format!("{language}.json"));
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
            let file: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|error| panic!("parsing {}: {error}", path.display()));

            let paragraphs = file["data"]
                .as_array()
                .expect("data")
                .iter()
                .flat_map(|article| article["paragraphs"].as_array().expect("paragraphs"));

            for (position, paragraph) in paragraphs.enumerate() {
                let breaks: Vec<(usize, usize)> = paragraph["sentence_breaks"]
                    .as_array()
                    .expect("sentence_breaks")
                    .iter()
                    .map(|span| {
                        let span = span.as_array().expect("a span");
                        (
                            span[0].as_u64().expect("start") as usize,
                            span[1].as_u64().expect("end") as usize,
                        )
                    })
                    .collect();

                for (index, sentence) in paragraph["sentences"]
                    .as_array()
                    .expect("sentences")
                    .iter()
                    .enumerate()
                {
                    sentences.push(Sentence {
                        key: format!("{language}:{position}:{index}"),
                        // Several of the Thai and Chinese sentences start with
                        // a byte-order mark, which is not part of the sentence
                        // and would be indexed as though it were.
                        text: sentence
                            .as_str()
                            .expect("a sentence")
                            .trim_start_matches('\u{feff}')
                            .to_string(),
                        language,
                    });
                }

                for question in paragraph["qas"].as_array().expect("qas") {
                    let id = question["id"].as_str().expect("an id").to_string();
                    let start = question["answers"][0]["answer_start"]
                        .as_u64()
                        .expect("an answer offset") as usize;
                    let index = breaks
                        .iter()
                        .position(|(from, to)| (*from..*to).contains(&start))
                        .unwrap_or_else(|| {
                            panic!("{id} in {language} answers outside every sentence")
                        });

                    asked.entry(id.clone()).or_default().insert(
                        language,
                        question["question"]
                            .as_str()
                            .expect("a question")
                            .to_string(),
                    );
                    answers
                        .entry(id)
                        .or_default()
                        .insert(language, format!("{language}:{position}:{index}"));
                }
            }
        }

        // Sorted, so the rotation below assigns the same language to the same
        // question on every run and two runs measure the same query set.
        let mut ids: Vec<String> = answers.keys().cloned().collect();
        ids.sort();

        let questions: Vec<Question> = ids
            .into_iter()
            .map(|id| Question {
                asked: asked.remove(&id).expect("the question text"),
                answers: answers.remove(&id).expect("the answering sentences"),
            })
            // A question the eleven files do not agree on cannot be scored
            // cross-lingually. There are none today; this is what would happen
            // if the dataset gained a language it does not cover everywhere.
            .filter(|question| question.answers.len() == LANGUAGES.len())
            .collect();

        assert!(
            !questions.is_empty(),
            "no question is present in all {} languages",
            LANGUAGES.len()
        );
        Self {
            sentences,
            questions,
        }
    }

    /// The queries to run.
    ///
    /// Every question in every language is 13,090 queries and a forward pass
    /// each, which is most of an hour before anything is scored. Rotating
    /// instead -- the nth question asked in the nth language -- keeps every
    /// question and every language and costs an eleventh of that. The rotation
    /// is positional rather than random so that a rerun compares against the
    /// same queries.
    fn queries(&self) -> Vec<Query<'_>> {
        let all = std::env::var("XQUAD_ALL_QUERIES").is_ok();
        self.questions
            .iter()
            .enumerate()
            .flat_map(|(position, question)| {
                let languages: Vec<&'static str> = if all {
                    LANGUAGES.to_vec()
                } else {
                    vec![LANGUAGES[position % LANGUAGES.len()]]
                };
                languages
                    .into_iter()
                    .map(move |language| Query { question, language })
            })
            .collect()
    }

    /// A short digest, so a dataset that has changed does not reuse a cache.
    fn fingerprint(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for sentence in &self.sentences {
            for byte in sentence.key.bytes().chain(sentence.text.bytes()) {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        format!("{hash:016x}")
    }
}

/// Where the dataset lives.
fn dataset_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LAREQA_DIR") {
        return PathBuf::from(dir);
    }
    let base = std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"));
    base.join("xquad-r")
}

/// Downloads the language files that are not there yet.
///
/// Through `curl` rather than an HTTP client, because the workspace has none
/// that reaches an arbitrary URL -- the one the model loader uses speaks only
/// to HuggingFace -- and a dependency added for a test that CI never runs is a
/// dependency the whole project carries. Each file lands under a partial name
/// and is renamed on success, so an interrupted download is not mistaken for a
/// complete one on the next run.
fn fetch(dir: &Path) {
    std::fs::create_dir_all(dir)
        .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));

    for language in LANGUAGES {
        let path = dir.join(format!("{language}.json"));
        if path.exists() {
            continue;
        }

        let partial = dir.join(format!("{language}.json.part"));
        let url = format!("{SOURCE}/{language}.json");
        let status = Command::new("curl")
            .args(["-sSLf", "--max-time", "300", "-o"])
            .arg(&partial)
            .arg(&url)
            .status();

        let fetched = matches!(status, Ok(status) if status.success());
        assert!(
            fetched,
            "could not fetch {url}\n\
             The dataset is not vendored: it is CC-BY-SA-4.0 and this repository is Apache-2.0.\n\
             Download the eleven language files by hand and point LAREQA_DIR at the directory,\n\
             or make `curl` and {SOURCE} reachable."
        );
        std::fs::rename(&partial, &path).expect("name the downloaded file");
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Scores one ranking into every group.
fn score(groups: &mut BTreeMap<String, Scores>, query: &Query<'_>, ranked: &[String]) {
    for group in GROUPS {
        let (relevant, drop) = query.relevant(group);
        let kept: Vec<String> = match drop {
            None => ranked.to_vec(),
            Some(key) => ranked
                .iter()
                .filter(|hit| hit.as_str() != key)
                .cloned()
                .collect(),
        };
        groups
            .entry(group.to_string())
            .or_default()
            .add(&kept, relevant.len(), |topic| relevant.contains(topic));
    }
}

/// The channel diagnostic, plus the two things only this corpus can answer.
///
/// It has eleven languages and parallel sentences, so it is the only corpus
/// here that can say what a channel is worth *per language*, and the only one
/// where "did the reranker keep the query's own language on top" is a question
/// with two possible answers.
async fn report_channels(engine: &Engine, queries: &[Query<'_>], named: &str) {
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

    let mut alone: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut without: BTreeMap<Channel, BTreeMap<String, Scores>> = BTreeMap::new();
    let mut whole: BTreeMap<String, Scores> = BTreeMap::new();
    let mut by_language: BTreeMap<String, Scores> = BTreeMap::new();
    let mut lexical_agreement: Vec<f64> = Vec::new();

    // Every fusion setting worth pricing, scored from the same traces as the
    // rows above. One pass, the whole grid.
    let variants = channels::variants();
    let mut offline: Vec<BTreeMap<String, Scores>> =
        variants.iter().map(|_| BTreeMap::new()).collect();

    for query in queries {
        let hits = engine
            .search_fused(query.text(), WIDE, DEPTHS, Fusion::default())
            .await
            .expect("search");

        channels::enough_room(&hits, WIDE);
        channels::same_as_the_engine(&hits, &Fusion::default());

        let each = channels::each_alone(&hits);
        for (channel, ranking) in &each {
            score(alone.entry(*channel).or_default(), query, ranking);
        }
        for missing in CHANNELS {
            let ranking = channels::as_if(&hits, &Fusion::default().without(*missing));
            score(without.entry(*missing).or_default(), query, &ranking);
        }

        let ranked: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
        score(&mut whole, query, &ranked);

        for ((_, fusion), into) in variants.iter().zip(&mut offline) {
            score(into, query, &channels::as_if(&hits, fusion));
        }

        // The per-language table, on a key the two-group map does not use, so
        // nothing that indexes that map by bare group name is disturbed.
        let mut one = BTreeMap::new();
        score(&mut one, query, &ranked);
        for (group, scores) in one {
            by_language
                .entry(format!("{group}/{}", query.language))
                .or_default()
                .absorb(scores);
        }

        if let (Some(segmented), Some(ngram)) = (
            each.get(&Channel::LexicalSegmented),
            each.get(&Channel::LexicalNgram),
        ) && let Some(tau) = channels::agreement(segmented, ngram)
        {
            lexical_agreement.push(tau);
        }
    }

    for group in GROUPS {
        println!("\n  each channel on its own, {group}, {named}");
        println!("  channel               queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
        println!("  ---------------------------------------------------------------");
        for (channel, groups) in &alone {
            let scores = &groups[group];
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
            whole[group].queries,
            whole[group].mean_ndcg(),
            whole[group].mean_recall()
        );

        println!("\n  with one channel taken away, {group}:");
        for (channel, groups) in &without {
            println!(
                "  {:<20}   {}",
                format!("{channel:?}"),
                statistics::compare(&whole[group].per_query, &groups[group].per_query)
            );
        }
    }

    for group in GROUPS {
        println!("\n  every fusion setting against the one that ships, {group}, {named}");
        println!(
            "  setting                        nDCG@{NDCG_AT}   recall@{RECALL_AT}   against shipped"
        );
        println!("  ---------------------------------------------------------------------------");
        for ((label, _), scores) in variants.iter().zip(&offline) {
            println!(
                "  {label:<28}   {:>7.4}   {:>9.4}   {}",
                scores[group].mean_ndcg(),
                scores[group].mean_recall(),
                statistics::compare(&whole[group].per_query, &scores[group].per_query)
            );
        }
    }

    println!("\n  per language, all four fused, {named}");
    println!("  group and language     queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}");
    println!("  ----------------------------------------------------------------");
    for (key, scores) in &by_language {
        println!(
            "  {key:<20}   {:>7}   {:>7.4}   {:>9.4}",
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall()
        );
    }

    println!(
        "\n  the two lexical channels agree at Kendall tau {:.4} over {} queries\n",
        channels::mean(&lexical_agreement),
        lexical_agreement.len()
    );
}

/// Prints the table this harness exists to produce.
fn report(title: &str, groups: &BTreeMap<String, Scores>, per_query_ms: f64) {
    println!("\n  {title}");
    println!(
        "  group            queries   nDCG@{NDCG_AT}   recall@{RECALL_AT}   below rank {NDCG_AT}   queries with any"
    );
    println!("  ---------------------------------------------------------------------------------");
    for (group, scores) in groups {
        println!(
            "  {:<14}   {:>7}   {:>7.4}   {:>9.4}   {:>13}   {:>16}",
            group,
            scores.queries,
            scores.mean_ndcg(),
            scores.mean_recall(),
            scores.deep,
            scores.with_work,
        );
    }
    println!("  {per_query_ms:.0} ms per query\n");
}

/// What the reranker did during a run, printed beside the scores.
///
/// A row of numbers nothing in this project had: how many candidates reached
/// the model against how many were offered it, how long they were, and how
/// often the score cache answered instead. Each of the three gates a decision
/// recorded as deferred -- see [`pamin_index::Reranked`] -- and each was an
/// inference from what the corpus is until this printed it.
fn report_reranking(engine: &Engine, tier: Rerank, queries: usize) {
    let Some(counted) = engine.reranked(tier) else {
        println!("  the {} tier was never loaded\n", tier.name());
        return;
    };
    if counted.offered == 0 {
        println!("  the {} tier scored nothing\n", tier.name());
        return;
    }
    println!(
        "  {}: {:.1} candidates a query reached the model of {:.1} offered, \
         {:.0} characters each, longest {}, cache {:.1}% of {} lookups",
        tier.name(),
        counted.scored as f64 / queries as f64,
        counted.offered as f64 / queries as f64,
        counted.characters as f64 / counted.scored.max(1) as f64,
        counted.longest,
        100.0 * (counted.offered - counted.scored) as f64 / counted.offered as f64,
        counted.offered,
    );
}

/// Asserts the floors, unless this is a run of some other profile.
fn assert_floors(named: &str, groups: &BTreeMap<String, Scores>, floors: &[(&str, f64, f64)]) {
    if named != DEFAULT_PROFILE {
        println!("  {named} is not the default profile, so the floors are not asserted\n");
        return;
    }

    for (group, floor_ndcg, floor_recall) in floors {
        let scores = groups
            .get(*group)
            .unwrap_or_else(|| panic!("nothing was scored for {group}"));
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

/// The fusion settings to try when `SWEEP` is set, each labelled as printed.
///
/// First the two numbers fusion has. Both were settled on the corpus this
/// project wrote, where the lexical pair carried signal for every query; this
/// corpus is the first place they can be read against one where it carries
/// signal for half of them and noise for the other half.
///
/// The adaptive block that used to follow is gone with the rule it measured:
/// scaling the lexical pair by how far it agreed with the vector channel
/// interpolated monotonically between the two constants it was bounded by, and
/// the rows never separated from them. This corpus could never have settled it
/// anyway -- its two groups are the same 1,190 queries scored twice, once with
/// the same-language answer struck out of the ranking, so one per-query
/// decision serves both groups and a rule cannot be credited with telling them
/// apart.
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
            settings.push((
                format!("k={k:.0} lex {weight:.3}"),
                fusion(Some((k, weight))),
            ));
        }
    }
    if let Some(filter) = &filter {
        settings.retain(|(label, _)| label.contains(filter.as_str()));
        assert!(!settings.is_empty(), "SWEEP={filter:?} matched no row");
    }
    Some(settings)
}

/// The fusion a sweep step runs, or the shipped one.
fn fusion(setting: Option<(f32, f32)>) -> Fusion {
    match setting {
        None => Fusion::default(),
        Some((k, lexical)) => Fusion::default()
            .with_k(k)
            .with_weight(Channel::LexicalSegmented, lexical)
            .with_weight(Channel::LexicalNgram, lexical),
    }
}

/// The profile to measure.
fn profile() -> (String, Profile) {
    let named = std::env::var("PAMIN_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());
    let profile = Profile::parse(&named).expect("a known profile");
    (named, profile)
}

// ---------------------------------------------------------------------------
// The embedding space on its own
// ---------------------------------------------------------------------------

#[test]
#[ignore = "downloads a dataset and model weights, and embeds thirteen thousand sentences"]
fn the_model_reaches_across_languages() {
    let corpus = Corpus::load();
    let queries = corpus.queries();
    let (named, profile) = profile();
    println!(
        "  {} sentences in {} languages, {} queries, profile {named}",
        corpus.sentences.len(),
        LANGUAGES.len(),
        queries.len()
    );

    let home = std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"));
    let mut embedder =
        Embedder::load(profile, &home.join("models")).expect("load the embedding model");

    let width = profile.dimensions() as usize;
    let vectors = embed_corpus(&mut embedder, &corpus, &home, width);

    let mut groups = BTreeMap::new();
    let started = std::time::Instant::now();
    for query in &queries {
        let vector = unit(embedder.embed_query(query.text()).expect("embed the query"));

        // Exhaustive rather than through the index: this test asks what the
        // embedding space contains, and an approximate index answers a
        // different question that would be indistinguishable from a worse
        // model. `pamin-index`'s `recall.rs` measures the approximation.
        let mut scored: Vec<(f32, usize)> = (0..corpus.sentences.len())
            .map(|document| {
                let stored = &vectors[document * width..(document + 1) * width];
                let similarity = vector.iter().zip(stored).map(|(a, b)| a * b).sum();
                (similarity, document)
            })
            .collect();
        scored.select_nth_unstable_by(DEPTH, |a, b| b.0.total_cmp(&a.0));
        scored.truncate(DEPTH);
        scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

        let ranked: Vec<String> = scored
            .into_iter()
            .map(|(_, document)| corpus.sentences[document].key.clone())
            .collect();
        score(&mut groups, query, &ranked);
    }

    report(
        &format!("the embedding space alone, {named}"),
        &groups,
        started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64,
    );
    assert_floors(&named, &groups, MODEL_FLOORS);
}

/// Embeds every sentence, reusing and resuming a cache on disk.
///
/// Resumable rather than merely cached: this is twenty minutes of forward
/// passes, and a machine that loses the process halfway -- a container that
/// restarts, a laptop that sleeps -- would otherwise start again from nothing.
/// Vectors are appended in batches, so the length of the file is how many are
/// done.
fn embed_corpus(embedder: &mut Embedder, corpus: &Corpus, home: &Path, width: usize) -> Vec<f32> {
    /// How many sentences go through one forward pass.
    const BATCH: usize = 32;

    let path = home.join(format!(
        "xquad-{}-{}.f32",
        embedder.profile().model_id().replace('/', "-"),
        corpus.fingerprint()
    ));
    std::fs::create_dir_all(home).expect("the evaluation directory");

    let bytes = std::fs::metadata(&path)
        .map(|file| file.len() as usize)
        .unwrap_or(0);
    let mut done = bytes / 4 / width;
    if done > corpus.sentences.len() {
        // A cache from a wider profile under the same name cannot happen -- the
        // name carries the model -- but a truncated write can leave a partial
        // vector, and half a vector is worse than none.
        done = 0;
    }
    let mut vectors: Vec<f32> = std::fs::read(&path)
        .map(|raw| {
            raw[..done * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect()
        })
        .unwrap_or_default();
    if done > 0 {
        println!("  reusing {done} embedded sentences");
    }

    let resumed = done;
    let started = std::time::Instant::now();
    while done < corpus.sentences.len() {
        let batch: Vec<&str> = corpus.sentences[done..]
            .iter()
            .take(BATCH)
            .map(|sentence| sentence.text.as_str())
            .collect();
        let embedded = embedder.embed_passages(&batch).expect("embed the corpus");

        let mut appended = Vec::with_capacity(batch.len() * width * 4);
        for vector in embedded {
            let vector = unit(vector);
            for value in &vector {
                appended.extend_from_slice(&value.to_le_bytes());
            }
            vectors.extend_from_slice(&vector);
        }
        append(&path, &appended);

        done += batch.len();
        if done.is_multiple_of(BATCH * 50) {
            println!(
                "  embedded {done}/{} at {:.0}/s",
                corpus.sentences.len(),
                (done - resumed) as f64 / started.elapsed().as_secs_f64()
            );
        }
    }

    vectors
}

fn append(path: &Path, bytes: &[u8]) {
    use std::io::Write as _;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|error| panic!("opening {}: {error}", path.display()));
    file.write_all(bytes).expect("write the vector cache");
}

/// Scales a vector to unit length, so a dot product is a cosine.
fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

// ---------------------------------------------------------------------------
// The whole search path
// ---------------------------------------------------------------------------

/// The floors for the whole search path, as the product calls it.
///
/// A tenth below 0.6480 / 0.8960 cross-lingual, which is what
/// `search_reranked` scores at the default tier.
///
/// The same-language pair is deliberately *not* a tenth. It measures 0.7495 /
/// 0.9580 and the floor stays at the 0.71 set when the lexical weight was a
/// quarter and this group scored 0.7971: halving that weight cost this group
/// 0.0476 and bought 0.0383 cross-lingual here, 0.0550 cross-lingual on the
/// own corpus and the best MIRACL score of the five weights tried (see
/// `pamin_core::fusion`). That leaves about a twentieth of margin instead of a
/// tenth, and lowering the floor to restore the tenth would be moving a guard
/// to fit the regression it exists to catch. A twentieth is enough here
/// because this measurement is exactly repeatable: fixed corpus, fixed index,
/// fixed model, greedy pass.
///
/// Both pairs now sit *above* the model's own floors. The cross-lingual pair
/// did not until the lexical weight was halved -- the product used to rank
/// below the model it is built on, on the group the model is best at -- which
/// is the single most important thing this harness has found: see the table in
/// the module notes.
const SEARCH_FLOORS: &[(&str, f64, f64)] =
    &[("cross_lingual", 0.58, 0.80), ("same_language", 0.71, 0.86)];

/// The least the reranker must be worth, in cross-lingual nDCG@10.
///
/// A floor cannot carry this. The tenth of margin every other floor here uses
/// is wider than the reranker's own contribution -- fusion alone scores 0.6077
/// and the default tier 0.6480, so a floor set a tenth below the tier still
/// passes with the reranker switched off entirely. Losing it would be silent.
///
/// So the floor test scores fusion alone as well and asserts the gap. Measured
/// at 0.0403, and the measurement is exactly repeatable: three runs of all
/// three tiers returned the same four decimals every time, because the corpus,
/// the index and the model are all fixed and the pass is greedy. Half of what
/// was measured, so that this fails when the reranker stops working rather than
/// when it works slightly less well.
///
/// The gap is a corpus's opinion, not the reranker's worth in general. On
/// MIRACL Swahili the same default tier scores 0.6730 against 0.6882 for
/// fusion alone -- it *costs* 0.0152 there, at 226 ms a query. Those two are
/// the `speed` profile rather than this one, because `accuracy` is ten hours
/// of indexing for that corpus, so the pair is comparable with each other and
/// not with the figures above. What it establishes is that part of what the
/// reranker buys here is the dilution fusion introduced, and this corpus --
/// parallel translations, half its queries answered in another language -- is
/// the one where that dilution is largest.
const RERANK_IS_WORTH: f64 = 0.020;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads a dataset and model weights, and indexes thirteen thousand sentences"]
async fn search_reaches_across_languages() {
    let corpus = Corpus::load();
    let queries = corpus.queries();
    let (named, profile) = profile();

    // A named workspace is reused; an unnamed one is thrown away. Indexing
    // thirteen thousand sentences is minutes, and a harness that pays that on
    // every run is a harness nobody runs twice in an afternoon.
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    // The profile is part of the workspace identity: an index records the
    // profile it was built with and refuses to open under another.
    let project = format!("xquad-{named}-{}", corpus.fingerprint());
    let engine = Engine::open(&workspace, &project, profile, Access::ReadWrite)
        .await
        .expect("open the engine");

    write_corpus(&engine, &corpus).await;

    if let Some(settings) = sweep() {
        println!("\n  setting              cross nDCG@10   same nDCG@10   cross recall@50");
        println!("  --------------------------------------------------------------------");
        let mut measured: Vec<(String, BTreeMap<String, Scores>)> = Vec::new();
        for (label, fusion) in settings {
            let groups = run(&engine, &queries, Route::Fused(fusion)).await;
            println!(
                "  {label:<20}   {:>13.4}   {:>12.4}   {:>15.4}",
                groups["cross_lingual"].mean_ndcg(),
                groups["same_language"].mean_ndcg(),
                groups["cross_lingual"].mean_recall(),
            );
            measured.push((label, groups));
        }

        // A sweep that prints only means says which row is highest and not
        // whether it is distinguishable from the others. The rows here are
        // separated by thousandths on one group and by tenths on the other, so
        // the means are worth very different amounts in the two columns, and
        // the table cannot show that. Priced against the best cross-lingual row,
        // query by query, in both groups -- because the weight this sweep
        // settles is the one that trades one group against the other.
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
                    "  {label:<20}   cross {}",
                    statistics::compare(
                        &top["cross_lingual"].per_query,
                        &groups["cross_lingual"].per_query
                    )
                );
                println!(
                    "  {:<20}   same  {}",
                    "",
                    statistics::compare(
                        &top["same_language"].per_query,
                        &groups["same_language"].per_query
                    )
                );
            }
        }
        println!();
        return;
    }

    // `CHANNELS` reports what each channel is worth on its own, what the fused
    // list looks like with each one taken away, how far the two lexical
    // channels agree with each other, and whether the reranker keeps a query's
    // own language at the top. One run, not four: the trace carries every
    // channel's rank for every candidate. See `channels`.
    if std::env::var("CHANNELS").is_ok() {
        report_channels(&engine, &queries, &named).await;
        return;
    }

    // `AT_LIMIT` asks the shipped path for as many results as a person asks
    // for, and reports what the reranker was made to do rather than how well
    // it did it. Every other arm here asks for fifty-one so that recall@50 can
    // be scored, which puts the reranker's whole twenty-deep head inside what
    // is read; at five, most of that head is past it, and `can_be_seen` then
    // declines the pass entirely on the queries where none of the candidates
    // it may move is inside the five. How often that is, is the number this
    // arm exists for, and it shows up as candidates never offered to the
    // model.
    if let Ok(value) = std::env::var("AT_LIMIT") {
        let limit: u32 = value.parse().expect("AT_LIMIT is a number of results");
        for tier in [Rerank::Off, Rerank::default()] {
            let started = std::time::Instant::now();
            let _ = run(&engine, &queries, Route::AtLimit(tier, limit)).await;
            println!(
                "\n  {:?} at --limit {limit}, {named}: {:.0} ms per query",
                tier,
                started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64
            );
            report_reranking(&engine, tier, queries.len());
        }
        return;
    }

    // `TIERS` compares the reranker's settings against each other on this
    // path, which is the only place the comparison means anything: the tier
    // reorders what fusion produced, so a number for it has to come from the
    // same pipeline that produced the ordering.
    if std::env::var("TIERS").is_ok() {
        for tier in [Rerank::Off, Rerank::Fast, Rerank::Accurate] {
            let started = std::time::Instant::now();
            let groups = run(&engine, &queries, Route::Shipped(tier)).await;
            report(
                &format!("{tier:?}, {named}"),
                &groups,
                started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64,
            );
            report_reranking(&engine, tier, queries.len());
        }
        return;
    }

    let started = std::time::Instant::now();
    let groups = run(&engine, &queries, Route::Shipped(Rerank::default())).await;
    report(
        &format!("the shipped search path, {named}"),
        &groups,
        started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64,
    );
    report_reranking(&engine, Rerank::default(), queries.len());
    assert_floors(&named, &groups, SEARCH_FLOORS);

    // Fusion alone, to price the reranker. Forty seconds against the four
    // minutes above, and without it nothing here notices the reranker going
    // missing -- see `RERANK_IS_WORTH`.
    if named == DEFAULT_PROFILE {
        let alone = run(&engine, &queries, Route::Shipped(Rerank::Off)).await;
        let (with, without) = (
            groups["cross_lingual"].mean_ndcg(),
            alone["cross_lingual"].mean_ndcg(),
        );
        // Query by query as well as mean against mean. A mean cannot tell the
        // two cases apart -- every query moving a little, and one query moving
        // a lot -- and this project has been reading differences of this size
        // as findings without ever checking which case it was in. See
        // `statistics`.
        let paired = statistics::compare(
            &alone["cross_lingual"].per_query,
            &groups["cross_lingual"].per_query,
        );
        let same = statistics::compare(
            &alone["same_language"].per_query,
            &groups["same_language"].per_query,
        );
        println!("  reranking is worth {paired} cross-lingual nDCG@{NDCG_AT}");
        println!("  and {same} same-language\n");

        assert!(
            paired.is_significant(),
            "reranking moved cross-lingual nDCG@{NDCG_AT} by {paired} -- a \
             difference of means with nothing under it. The tier is supposed to \
             be worth something, not to be indistinguishable from leaving it off"
        );
        assert!(
            with - without >= RERANK_IS_WORTH,
            "reranking moved cross-lingual nDCG@{NDCG_AT} by {:+.4}, under the \
             {RERANK_IS_WORTH:.4} it is supposed to be worth: {without:.4} without it, \
             {with:.4} with",
            with - without
        );
    }
}

/// Which of the engine's two search entry points a scoring run drives.
///
/// Both exist because they answer different questions and neither can answer
/// the other's. `search_reranked` is what `pamin search` calls, so it is the
/// only thing a floor can guard -- but it fixes `Fusion::default()` internally,
/// so a weight sweep cannot go through it. `search_fused` takes the weighting
/// and stops before the reranker.
///
/// Keeping the sweep on the fused path is not a compromise: what the sweep
/// tunes is the fusion, and measuring it through a reranker that reorders the
/// top twenty afterwards would attribute the reranker's work to the weight.
#[derive(Clone)]
enum Route {
    /// The product's own entry point, reranker and all.
    Shipped(Rerank),
    /// The same entry point, asked for as many results as a person asks for.
    ///
    /// `DEPTH` is fifty-one so that recall@50 can be scored, and the whole of
    /// the reranker's twenty-deep head is inside that. `pamin search` defaults
    /// to five, where most of that head is past what the caller reads --
    /// which is a different amount of work for the same query and was never
    /// measured. The scores from this route are not comparable with anything
    /// else on this corpus and it does not report them.
    AtLimit(Rerank, u32),
    /// Fusion alone, at a weighting the caller chooses.
    Fused(Fusion),
}

/// Scores every query through one of the engine's search paths.
async fn run<'a>(engine: &Engine, queries: &[Query<'a>], route: Route) -> BTreeMap<String, Scores> {
    let mut groups = BTreeMap::new();
    for query in queries {
        let hits = match &route {
            // `DEPTH` is fifty-one and a tier's depth is twenty, so
            // `fused_for` keeps the fifty-one this scores at and the reranker
            // reorders the head. recall@50 is therefore the same list either
            // way and only the ordering moves, which is what the tier claims
            // to change.
            Route::Shipped(rerank) => {
                engine
                    .search_reranked(query.text(), DEPTH as u32, DEPTHS, *rerank)
                    .await
            }
            Route::AtLimit(rerank, limit) => {
                engine
                    .search_reranked(query.text(), *limit, DEPTHS, *rerank)
                    .await
            }
            Route::Fused(fusion) => {
                engine
                    .search_fused(query.text(), DEPTH as u32, DEPTHS, fusion.clone())
                    .await
            }
        }
        .expect("search");
        let ranked: Vec<String> = hits.into_iter().map(|hit| hit.topic).collect();
        score(&mut groups, query, &ranked);
    }
    groups
}

/// Writes every sentence that is not already a topic, then runs the queue.
///
/// Each sentence is its own topic, named by its key. The key is deliberately
/// unlike anything in the text: mention derivation looks for topic names inside
/// content, and a corpus whose sentences named each other would measure the
/// graph channel on relationships the dataset does not assert.
async fn write_corpus(engine: &Engine, corpus: &Corpus) {
    let project = engine.project;
    let mut written = 0;

    for sentence in &corpus.sentences {
        let existing =
            pamin_store::repository::find_topic(engine.database.pool(), project, &sentence.key)
                .await
                .expect("look for the topic");
        if existing.is_some() {
            continue;
        }
        written += 1;
        engine
            .write(&Write {
                topic: &sentence.key,
                content: &sentence.text,
                content_hash: &sentence.text.len().to_string(),
                verdict: pamin_core::FilterDecision::Promoted,
                reason: "cross-lingual evaluation corpus",
                promoted: true,
                // The dataset's own two-letter tags, which are not what the
                // product writes: `pamin write` takes its language from
                // `detect_language`, and that returns ISO-639-3 -- `eng` where
                // this says `en`. Nothing compares the two today, and the rule
                // that would have (a fusion weight that knew the query's
                // language) was measured and dropped. Left as the dataset has
                // it rather than translated, because changing it would mean
                // re-indexing thirteen thousand sentences to alter a column no
                // reader consults. Anything that starts consulting it should
                // fix this first.
                language: Some(sentence.language),
                language_confidence: None,
                observed_at: time::OffsetDateTime::now_utc(),
                validity: pamin_core::Validity::ALWAYS,
            })
            .await
            .unwrap_or_else(|error| panic!("writing {}: {error}", sentence.key));
    }

    if written > 0 {
        println!("  wrote {written} of {} sentences", corpus.sentences.len());
    }
    let started = std::time::Instant::now();
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
        println!(
            "  ran {} cascade jobs in {:.0}s",
            drained.completed,
            started.elapsed().as_secs_f64()
        );
    }
}
