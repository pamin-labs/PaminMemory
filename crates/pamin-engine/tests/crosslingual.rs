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
mod features;
mod reranking;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_core::{Channel, Fusion, Why};
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
    /// The dataset's own id, the same in all eleven files.
    id: String,
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
                id,
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
        score_group(
            groups.entry(group.to_string()).or_default(),
            query,
            group,
            ranked,
        );
    }
}

/// Scores one ranking for one group, dropping what that group drops.
fn score_group(into: &mut Scores, query: &Query<'_>, group: &str, ranked: &[String]) {
    let (relevant, drop) = query.relevant(group);
    let kept: Vec<String> = match drop {
        None => ranked.to_vec(),
        Some(key) => ranked
            .iter()
            .filter(|hit| hit.as_str() != key)
            .cloned()
            .collect(),
    };
    into.add(&kept, relevant.len(), |topic| relevant.contains(topic));
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
    let mut dense_by_language: BTreeMap<String, Scores> = BTreeMap::new();
    let mut lexical_agreement: Vec<f64> = Vec::new();
    // Per channel: how many of its top ten are in the query's own language,
    // and how many it returned there at all. See the table this prints.
    let mut head: BTreeMap<Channel, (usize, usize)> = BTreeMap::new();

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

        // The same split for the vector channel alone, which is what the
        // whole fusion loses to on this group. Without it the per-language
        // table says where the fusion is weak and not where it is *worse than
        // not fusing*, and those are different questions.
        if let Some(vector) = each.get(&Channel::Vector) {
            let mut one = BTreeMap::new();
            score(&mut one, query, vector);
            for (group, scores) in one {
                dense_by_language
                    .entry(format!("{group}/{}", query.language))
                    .or_default()
                    .absorb(scores);
            }
        }

        // What language each channel puts in its own head. The candidate keys
        // are `{language}:{paragraph}:{sentence}`, so this is read off the key
        // rather than guessed from the text.
        for (channel, ranking) in &each {
            let entry = head.entry(*channel).or_insert((0, 0));
            for key in ranking.iter().take(NDCG_AT) {
                entry.1 += 1;
                if key.split(':').next() == Some(query.language) {
                    entry.0 += 1;
                }
            }
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
        channels::sweep_table(
            &format!("{group}, {named}"),
            &whole[group],
            &variants,
            &offline.iter().map(|row| row.get(group)).collect::<Vec<_>>(),
        );
    }
    channels::cross_validated(
        &format!("XQuAD-R, {named}"),
        &GROUPS,
        &whole,
        &variants,
        channels::shipped_row(&variants),
        &offline,
    );

    // Per language, against the vector channel alone rather than against
    // nothing. Fusing all four ranks below the vector channel by itself on
    // the cross-lingual group, and the only question that decides whether the
    // remedy is a global constant or a per-candidate rule is whether that
    // deficit is spread across the eleven languages or concentrated in a few.
    // A deficit concentrated in one script family would argue for a rule
    // about scripts; one that is everywhere argues the mechanism is the
    // arithmetic and not the languages.
    println!("\n  per language, fused against the vector channel alone, {named}");
    println!(
        "  group and language     queries   nDCG@{NDCG_AT}   dense@{NDCG_AT}   fused-dense   recall@{RECALL_AT}"
    );
    println!("  --------------------------------------------------------------------------------");
    for (key, scores) in &by_language {
        let dense = dense_by_language.get(key).map(Scores::mean_ndcg);
        println!(
            "  {key:<20}   {:>7}   {:>7.4}   {:>7}   {:>11}   {:>9.4}",
            scores.queries,
            scores.mean_ndcg(),
            dense.map_or_else(|| "--".into(), |it| format!("{it:.4}")),
            dense.map_or_else(
                || "--".into(),
                |it| format!("{:+.4}", scores.mean_ndcg() - it)
            ),
            scores.mean_recall()
        );
    }

    // And what language each channel's own head is in. On the cross-lingual
    // group the query's own language is *never* the answer -- the one gold
    // sentence in it is removed from the ranking -- so a channel whose head is
    // mostly the query's own language is spending its head on candidates that
    // cannot be right, and it is the additive promotion of exactly those that
    // `Fusion::needing_support` is a candidate remedy for.
    println!("  what language each channel's top {NDCG_AT} is in, {named}");
    println!("  channel                own language   of returned   share");
    println!("  ----------------------------------------------------------");
    for (channel, (own, total)) in &head {
        println!(
            "  {:<20}   {own:>12}   {total:>11}   {:>5.1}%",
            format!("{channel:?}"),
            100.0 * *own as f64 / *total.max(&1) as f64
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

    // `PASSAGES`: the same memories in a second project whose vectors embed
    // the topic's name, asked every question alongside this one. See
    // `channels::Paired`.
    if std::env::var("PASSAGES").is_ok() {
        let other = Engine::open(
            &workspace,
            &format!("{project}-named"),
            profile,
            Access::ReadWrite,
        )
        .await
        .expect("open the named project");
        write_corpus(&other, &corpus).await;
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
                .search_fused(query.text(), WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            let after = other
                .search_fused(query.text(), WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            for group in GROUPS {
                paired.observe(group, &before, &after, |into, ranking| {
                    score_group(into, query, group, ranking)
                });
            }
        }
        paired.report(&format!(
            "vectors embedding the topic name, XQuAD-R, {named}"
        ));
        return;
    }

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

    // `FEATURES_OUT`: every candidate fusion saw, one row each, for fitting a
    // fusion offline. See `features`. Each query is written in both groups,
    // and its question id is the dataset's, so every asking of one question
    // can be kept in one fold.
    if let Some(mut dump) = features::Features::from_env("xquad-r") {
        const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
        for query in &queries {
            let hits = engine
                .search_fused(query.text(), WIDE, DEPTHS, Fusion::default())
                .await
                .expect("search");
            for group in GROUPS {
                let (relevant, drop) = query.relevant(group);
                let asked = features::Asked {
                    group,
                    question: &format!("{}/{}", query.question.id, query.language),
                    text: query.text(),
                    language: Some(query.language),
                    judged: relevant.len(),
                };
                dump.observe(&asked, &hits, WIDE, |topic| {
                    (drop != Some(topic)).then(|| f64::from(relevant.contains(topic)))
                });
            }
        }
        dump.finish(queries.len() * GROUPS.len());
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

    // `RERANK_RULES` prices other rules for using the shipped tier's scores
    // -- blending them with fusion's rather than substituting -- from one
    // shipped run. See `reranking`.
    if std::env::var("RERANK_RULES").is_ok() {
        rerank_rules(&engine, &queries, &named).await;
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
                .search_reranked(query.text(), WIDE, DEPTHS, tier)
                .await
                .expect("search");
            channels::enough_room(&hits, WIDE);
            let replayed = reranking::replay(&hits, tier);
            for group in GROUPS {
                routes.observe(
                    group,
                    &query.question.answers["en"],
                    &hits,
                    &replayed,
                    &mut small,
                    query.text(),
                    |into, ranking| score_group(into, query, group, ranking),
                );
            }
        }
        routes.report(&format!(
            "spending less on the {} reranker, XQuAD-R, {named}",
            tier.name()
        ));
        return;
    }

    // `CONTEXT`: the shipped tier shown each candidate's name and seed. See
    // `reranking::in_context`.
    if std::env::var("CONTEXT").is_ok() {
        context(&engine, &workspace, &queries, &named).await;
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

    // `LABELS` produces the utility label a router would predict and asks
    // what, visible before the pass, predicts it.
    if std::env::var("LABELS").is_ok() {
        labels(&engine, &queries, &named).await;
        return;
    }

    // `CALIBRATE` asks whether the shipped tier's score can be made into a
    // probability, which is the cheap half of what the typed judge is for.
    if std::env::var("CALIBRATE").is_ok() {
        calibration(&engine, &queries, &named).await;
        return;
    }

    // `GATE` asks whether the pass should run at all, which is a different
    // question from which tier runs it and is measured separately.
    if std::env::var("GATE").is_ok() {
        gate_sweep(&engine, &queries, &named).await;
        return;
    }

    // `PAIRS` asks how few candidates the pass needs to be shown, which is
    // the one lever proportional to its cost rather than all-or-nothing.
    if std::env::var("PAIRS").is_ok() {
        pairs_sweep(&engine, &queries, &named).await;
        return;
    }

    // `TIERS` compares the reranker's settings against each other on this
    // path, which is the only place the comparison means anything: the tier
    // reorders what fusion produced, so a number for it has to come from the
    // same pipeline that produced the ordering.
    if std::env::var("TIERS").is_ok() {
        // Every tier that loads a model, plus `off` as the baseline the others
        // are read against. `noncommercial` downloads CC-BY-NC-4.0 weights,
        // which is a thing a measurement may do and a product may not without
        // being asked -- the gate is on the command, and this is the harness
        // that prices the tier the gate exists for.
        // `TIERS=1` runs every tier; `TIERS=fast,accurate` runs those, with
        // `off` always first because every row is also priced against it. The
        // whole set is about two hours, and the question is usually about two.
        let wanted = std::env::var("TIERS").expect("checked above");
        let tiers: Vec<Rerank> = if wanted == "1" {
            vec![
                Rerank::Off,
                Rerank::Fast,
                Rerank::Balanced,
                Rerank::Accurate,
                Rerank::Noncommercial,
            ]
        } else {
            std::iter::once(Rerank::Off)
                .chain(wanted.split(',').map(|name| {
                    Rerank::parse(name.trim())
                        .unwrap_or_else(|| panic!("TIERS names an unknown tier: {name}"))
                }))
                .collect()
        };

        // Kept so the tiers can be compared against each other with paired
        // counts rather than by subtracting two means. `off` is the first,
        // which is what every row is priced against.
        // The cost is kept, not just printed, because the oracle below is about
        // spending it: a router's whole claim is that some queries need a
        // cheaper tier than others, and that claim cannot be priced without
        // knowing what each tier cost in this same run.
        let mut priced: Vec<(Rerank, f64, BTreeMap<String, Scores>)> = Vec::new();

        for tier in tiers {
            let started = std::time::Instant::now();
            let groups = run(&engine, &queries, Route::Shipped(tier)).await;
            let cost = started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
            report(&format!("{tier:?}, {named}"), &groups, cost);
            report_reranking(&engine, tier, queries.len());
            priced.push((tier, cost, groups));
        }

        let (_, _, baseline) = &priced[0];
        for group in GROUPS {
            println!("\n  every tier against reranking off, {group}, {named}");
            println!("  tier                 nDCG@{NDCG_AT}   recall@{RECALL_AT}   against off");
            println!("  ---------------------------------------------------------------------");
            for (tier, _, groups) in &priced[1..] {
                println!(
                    "  {:<18}   {:>7.4}   {:>9.4}   {}",
                    tier.name(),
                    groups[group].mean_ndcg(),
                    groups[group].mean_recall(),
                    statistics::compare(&baseline[group].per_query, &groups[group].per_query)
                );
            }
        }

        // Against the tier that ships, directly. "Each against `off`" cannot
        // answer whether the default is the right tier: two tiers each compared
        // with a third are not compared with each other, and a paired test
        // needs the pair. This is the table a change of default rests on.
        if let Some((_, _, shipped)) = priced.iter().find(|(tier, ..)| *tier == Rerank::default()) {
            for group in GROUPS {
                println!(
                    "\n  every tier against the one that ships ({}), {group}, {named}",
                    Rerank::default().name()
                );
                println!(
                    "  tier                 nDCG@{NDCG_AT}   recall@{RECALL_AT}   ms   against shipped"
                );
                println!(
                    "  -------------------------------------------------------------------------------"
                );
                for (tier, cost, groups) in &priced {
                    if *tier == Rerank::default() {
                        continue;
                    }
                    println!(
                        "  {:<18}   {:>7.4}   {:>9.4}   {:>5.0}   {}",
                        tier.name(),
                        groups[group].mean_ndcg(),
                        groups[group].mean_recall(),
                        cost,
                        statistics::compare(&shipped[group].per_query, &groups[group].per_query)
                    );
                }
            }
        }

        oracle(&priced, &named);
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

/// The utility labels a router would have to predict, and what predicts them.
///
/// The oracle next door said the routing line is alive on latency and nearly
/// dead on accuracy: over six tiers a perfect router could add at most +0.0404
/// cross-lingual and +0.0029 same-language, while saving 62% and 85% of the
/// time. And it said why — **`off` is the cheapest sufficient tier on 43.4% of
/// cross-lingual queries and 98.4% of same-language ones.** Nearly half, and
/// almost all, need no reranking at all.
///
/// That is also the diagnosis of why the hand-picked gate failed. The oracle
/// knows *which* queries; lexical coverage did not. The 2026 work this borrows
/// from says the same thing in its method: a utility-based label per query and
/// a router trained on it, not a proxy somebody chose. So this arm produces the
/// label and then asks, feature by feature, whether anything available *before*
/// the pass predicts it.
///
/// **Three tiers, not six, and that is a narrowing rather than a shortcut.**
/// `typed` measured *below* `off` and is closed; `balanced` costs four times
/// `fast`'s parameters for a quarter of its gain. What is left — nothing,
/// cheap, expensive — is what a shipped router would choose between, and is
/// the same triple the published router used.
///
/// Every feature here is read from the **`off`** search, because that is all a
/// router can see: it runs before the pass it is deciding about.
async fn labels(engine: &Engine, queries: &[Query<'_>], named: &str) {
    /// How close to the best a tier must be to count as sufficient. The same
    /// hundredth the oracle uses, and for the same reason.
    const ENOUGH: f64 = 0.01;

    /// The tiers a router would choose between, cheapest first, so the first
    /// sufficient one is the cheapest sufficient one.
    const LADDER: [Rerank; 3] = [Rerank::Off, Rerank::Fast, Rerank::Accurate];

    /// What a feature is measured against: whether the pass was needed at all.
    /// Binary rather than three-way because that is the decision worth most --
    /// 43.4% and 98.4% of queries skipping the pass entirely is where the 62%
    /// and 85% live.
    struct Sample {
        group: String,
        /// `true` when `off` was sufficient: the pass buys nothing.
        skippable: bool,
        features: [f64; FEATURES.len()],
    }

    const FEATURES: [&str; 7] = [
        "fused score at rank 1",
        "margin, rank 1 over rank 2",
        "margin, rank 1 over rank 10",
        "candidates the pass may touch",
        "lexical share of the head",
        "channels proposing rank 1",
        "query tokens",
    ];

    let head = Rerank::default().depth();
    let mut samples: Vec<Sample> = Vec::new();
    let mut per_tier: Vec<BTreeMap<String, Scores>> =
        LADDER.iter().map(|_| BTreeMap::new()).collect();
    let mut chosen: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    // By position rather than keyed by tier: `Rerank` is deliberately not
    // `Ord`, and the ladder's order is the price order anyway.
    let mut cost = vec![0.0f64; LADDER.len()];

    // Measured in this run, so the saving below is priced against what these
    // tiers cost here rather than against a figure from another run.
    for (at, tier) in LADDER.iter().enumerate() {
        let started = std::time::Instant::now();
        let groups = run(engine, queries, Route::Shipped(*tier)).await;
        cost[at] = started.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
        per_tier[at] = groups;
    }

    // The features, from one more pass at `off` -- which is the cheap tier, and
    // the only ordering a router gets to look at.
    for (index, query) in queries.iter().enumerate() {
        let hits = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, Rerank::Off)
            .await
            .expect("search with the reranker off");

        let scored_at = |rank: usize| -> f64 {
            hits.get(rank)
                .map(|hit| f64::from(hit.result.score))
                .unwrap_or(0.0)
        };
        let lexical = |hit: &pamin_engine::SearchHit| {
            hit.result.why.iter().any(|why| {
                matches!(
                    why,
                    Why::Channel { channel, .. }
                        if *channel == Channel::LexicalSegmented
                            || *channel == Channel::LexicalNgram
                )
            })
        };
        let depth = head.min(hits.len());
        let touchable = hits[..depth].iter().filter(|hit| !lexical(hit)).count();
        let window = NDCG_AT.min(hits.len()).max(1);
        let lexical_share =
            hits[..window].iter().filter(|hit| lexical(hit)).count() as f64 / window as f64;
        let channels_at_1 = hits
            .first()
            .map(|hit| {
                hit.result
                    .why
                    .iter()
                    .filter(|why| matches!(why, Why::Channel { .. }))
                    .count()
            })
            .unwrap_or(0);

        let features = [
            scored_at(0),
            scored_at(0) - scored_at(1),
            scored_at(0) - scored_at(9),
            touchable as f64,
            lexical_share,
            channels_at_1 as f64,
            query.text().split_whitespace().count() as f64,
        ];

        for group in GROUPS {
            // The label: the cheapest tier within ENOUGH of this query's best.
            let scores: Vec<f64> = per_tier
                .iter()
                .map(|by| by[group].per_query[index])
                .collect();
            let best = scores.iter().copied().fold(f64::MIN, f64::max);
            let cheapest = scores
                .iter()
                .position(|score| *score >= best - ENOUGH)
                .expect("the best tier suffices for itself");
            chosen
                .entry(group.to_string())
                .or_insert_with(|| vec![0; LADDER.len()])[cheapest] += 1;

            samples.push(Sample {
                group: group.to_string(),
                skippable: cheapest == 0,
                features,
            });
        }
    }

    for group in GROUPS {
        let mine: Vec<&Sample> = samples.iter().filter(|s| s.group == group).collect();
        let skippable = mine.iter().filter(|s| s.skippable).count();
        let picked = &chosen[group];

        println!("\n  what a router would have to predict, {group}, {named}");
        for (at, tier) in LADDER.iter().enumerate() {
            println!(
                "    {:<10} cheapest-sufficient on {:>5.1}% of queries ({:>6.0} ms)",
                tier.name(),
                100.0 * picked[at] as f64 / mine.len() as f64,
                cost[at]
            );
        }
        println!(
            "    the pass is skippable on {:.1}% of them",
            100.0 * skippable as f64 / mine.len() as f64
        );

        // Separation per feature, as the AUC of ranking skippable above the
        // rest. 0.5 is a feature that says nothing; 1.0 or 0.0 would be a
        // feature that decides it. Reported for both directions, because a
        // feature that predicts the *opposite* is just as useful.
        println!(
            "\n  does anything visible before the pass predict it, {group}\n  \
             feature                          mean when skippable   when not        AUC"
        );
        println!("  ----------------------------------------------------------------------------");
        for (at, name) in FEATURES.iter().enumerate() {
            let yes: Vec<f64> = mine
                .iter()
                .filter(|s| s.skippable)
                .map(|s| s.features[at])
                .collect();
            let no: Vec<f64> = mine
                .iter()
                .filter(|s| !s.skippable)
                .map(|s| s.features[at])
                .collect();
            if yes.is_empty() || no.is_empty() {
                continue;
            }
            // Mann-Whitney: the share of (skippable, not) pairs the feature
            // orders correctly, ties counting half. That is the AUC, and it
            // needs no model and no threshold.
            let mut correct = 0.0;
            for a in &yes {
                for b in &no {
                    correct += match a.partial_cmp(b) {
                        Some(std::cmp::Ordering::Greater) => 1.0,
                        Some(std::cmp::Ordering::Equal) => 0.5,
                        _ => 0.0,
                    };
                }
            }
            let auc = correct / (yes.len() * no.len()) as f64;
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            println!(
                "  {name:<32} {:>19.4} {:>14.4} {:>10.4}",
                mean(&yes),
                mean(&no),
                auc
            );
        }
        println!(
            "\n  An AUC at 0.5 is a feature that says nothing. Far from 0.5 in either\n  \
             direction is a feature a rule could use -- below 0.5 means the feature\n  \
             predicts the opposite, which is equally usable.\n"
        );
    }
}

/// Whether the shipped reranker's score can be turned into a probability.
///
/// This is the cheap half of the question the typed judge was added to answer,
/// and it should be asked first: **a cross-encoder can be calibrated too.** If
/// fitting two parameters onto the tier already running produces a usable
/// probability, then the thresholding, abstention and weighted-sum-fusion work
/// all become reachable with no new model, no new download and no
/// per-question-shape temperature to maintain. If it does not, a
/// purpose-trained judge very likely will not rescue it either, because
/// calibration is corpus-specific and cross-corpus comparability is the exact
/// property being bought.
///
/// The dataset already exists and needed no new run: `Why::Reranked` records
/// what the model scored every candidate it saw, and the corpus supplies
/// whether each was relevant. What was missing until that entry existed was not
/// the model — it was that the score was computed and thrown away.
///
/// **Fit and test are disjoint by query, not by candidate.** Splitting by
/// candidate would put two candidates of the same query on opposite sides, and
/// a cross-encoder's scores within one shortlist are correlated — so the test
/// half would be partly memorised and the calibration error would read far
/// better than it is.
async fn calibration(engine: &Engine, queries: &[Query<'_>], named: &str) {
    /// Bins for the reliability curve and the calibration error.
    const BINS: usize = 10;
    /// Gradient steps for the two-parameter fit, and the step size. Plain
    /// gradient descent rather than Newton: two parameters, a convex loss, and
    /// a fixed step count is deterministic where a convergence test is not.
    const STEPS: usize = 4_000;
    const RATE: f64 = 0.05;

    let tier = Rerank::default();
    // (score, relevant) per group, split by query parity.
    let mut fit: BTreeMap<String, Vec<(f64, bool)>> = BTreeMap::new();
    let mut test: BTreeMap<String, Vec<(f64, bool)>> = BTreeMap::new();

    for (at, query) in queries.iter().enumerate() {
        let hits = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, tier)
            .await
            .expect("search at the default tier");

        for group in GROUPS {
            let (relevant, drop) = query.relevant(group);
            let into = if at % 2 == 0 { &mut fit } else { &mut test };
            let pairs = into.entry(group.to_string()).or_default();
            for hit in &hits {
                if drop == Some(hit.topic.as_str()) {
                    continue;
                }
                let scored = hit.result.why.iter().find_map(|why| match why {
                    Why::Reranked { score } => Some(*score),
                    Why::Channel { .. } | Why::Path { .. } => None,
                });
                if let Some(score) = scored {
                    pairs.push((f64::from(score), relevant.contains(hit.topic.as_str())));
                }
            }
        }
    }

    for group in GROUPS {
        let Some(training) = fit.get(group) else {
            continue;
        };
        let Some(held) = test.get(group) else {
            continue;
        };
        if training.is_empty() || held.is_empty() {
            continue;
        }

        // Platt: P = sigmoid(-(a * score + b)), fitted by minimising the
        // negative log likelihood. `a` starts negative so the initial map is
        // increasing in the score, which is the direction every score in this
        // project runs.
        //
        // **With the label smoothing Platt's method is defined with, which a
        // first version of this left out and was wrong without.** A
        // cross-encoder's scores are close to separable -- relevant candidates
        // score high, the rest low, with little overlap -- and maximum
        // likelihood on separable data drives the slope to infinity. Fitted raw
        // it produced a coefficient of -81 cross-lingual and -318
        // same-language, which is a hard threshold wearing a sigmoid's clothes:
        // 9,129 of 9,175 held-out candidates landed in the two end bins, and
        // the calibration error got *worse* than the untransformed sigmoid,
        // 0.0905 to 0.2146. That was a property of the fit, not of the score.
        //
        // Platt's targets are `(n+1)/(n+2)` for the positives and `1/(m+2)`
        // for the negatives rather than 1 and 0, which bounds the likelihood
        // and so bounds the slope.
        let positives = training.iter().filter(|(_, relevant)| *relevant).count() as f64;
        let negatives = training.len() as f64 - positives;
        let high = (positives + 1.0) / (positives + 2.0);
        let low = 1.0 / (negatives + 2.0);

        let (mut a, mut b) = (-1.0f64, 0.0f64);
        for _ in 0..STEPS {
            let (mut da, mut db) = (0.0f64, 0.0f64);
            for (score, relevant) in training {
                let p = 1.0 / (1.0 + (a * score + b).exp());
                let error = p - if *relevant { high } else { low };
                da -= error * score;
                db -= error;
            }
            let n = training.len() as f64;
            a += RATE * da / n;
            b += RATE * db / n;
        }

        let platt = |score: f64| 1.0 / (1.0 + (a * score + b).exp());

        // Isotonic regression as the second arm, because it *cannot* diverge:
        // it is monotone and non-parametric, so separable data gives it a step
        // and nothing worse. Pool-adjacent-violators over the training pairs
        // sorted by score, then a lookup by interval. Included because a
        // parametric fit failing tells you about the parametric family, not
        // about whether the score can be calibrated at all.
        let mut sorted: Vec<(f64, f64)> = training
            .iter()
            .map(|(score, relevant)| (*score, if *relevant { 1.0 } else { 0.0 }))
            .collect();
        sorted.sort_by(|left, right| left.0.total_cmp(&right.0));
        // Blocks of (sum, count, boundary score), merged while a block's mean
        // is below its predecessor's.
        let mut blocks: Vec<(f64, f64, f64)> = Vec::with_capacity(sorted.len());
        for (score, label) in &sorted {
            blocks.push((*label, 1.0, *score));
            while blocks.len() >= 2 {
                let (sum_b, n_b, _) = blocks[blocks.len() - 1];
                let (sum_a, n_a, at_a) = blocks[blocks.len() - 2];
                if sum_b / n_b >= sum_a / n_a {
                    break;
                }
                blocks.pop();
                let last = blocks.len() - 1;
                blocks[last] = (sum_a + sum_b, n_a + n_b, at_a);
            }
        }
        // The boundary of a block is the lowest score in it, so a lookup finds
        // the last block starting at or below the score.
        let isotonic = |score: f64| -> f64 {
            match blocks.binary_search_by(|(_, _, at)| at.total_cmp(&score)) {
                Ok(at) => blocks[at].0 / blocks[at].1,
                Err(0) => blocks.first().map_or(0.0, |(s, n, _)| s / n),
                Err(at) => blocks[at - 1].0 / blocks[at - 1].1,
            }
        };

        let probability = &platt;

        // Expected calibration error on the held-out half, before and after,
        // with "before" being a sigmoid of the raw score -- the thing somebody
        // would reach for if they assumed the logit was already a probability.
        let error_of = |map: &dyn Fn(f64) -> f64| -> (f64, Vec<(f64, f64, usize)>) {
            let mut bins = vec![(0.0f64, 0usize, 0usize); BINS];
            for (score, relevant) in held {
                let p = map(*score);
                let at = ((p * BINS as f64) as usize).min(BINS - 1);
                bins[at].0 += p;
                bins[at].1 += usize::from(*relevant);
                bins[at].2 += 1;
            }
            let total = held.len() as f64;
            let mut ece = 0.0;
            let mut curve = Vec::new();
            for (sum, positives, count) in &bins {
                if *count == 0 {
                    continue;
                }
                let confidence = sum / *count as f64;
                let accuracy = *positives as f64 / *count as f64;
                ece += (*count as f64 / total) * (confidence - accuracy).abs();
                curve.push((confidence, accuracy, *count));
            }
            (ece, curve)
        };

        let raw = |score: f64| 1.0 / (1.0 + (-score).exp());
        let (before, _) = error_of(&raw);
        let (after, curve) = error_of(probability);
        let (iso, iso_curve) = error_of(&isotonic);

        // The base rate is reported because it decides what a good calibration
        // error even looks like: a group where one candidate in fifty is
        // relevant and a group where five are cannot share a threshold, and a
        // fit that ignores it will look broken on the sparse group while being
        // right about the score.
        let base = held.iter().filter(|(_, relevant)| *relevant).count() as f64 / held.len() as f64;

        println!("\n  calibrating the {} tier, {group}, {named}", tier.name());
        println!(
            "  {} pairs fitted, {} held out, disjoint by query, {:.1}% relevant",
            training.len(),
            held.len(),
            100.0 * base
        );
        println!("  Platt, smoothed: P = sigmoid(-({a:.4} * score + {b:.4}))");
        println!("  expected calibration error, held out:");
        println!("    raw sigmoid of the logit   {before:.4}");
        println!(
            "    Platt                      {after:.4}   ({:+.4})",
            after - before
        );
        println!(
            "    isotonic                   {iso:.4}   ({:+.4})",
            iso - before
        );
        println!("  reliability under the better of the two, held out:");
        println!("    predicted   observed   candidates");
        for (confidence, accuracy, count) in if iso < after { iso_curve } else { curve } {
            println!("    {confidence:>9.3}   {accuracy:>8.3}   {count:>10}");
        }
    }

    println!(
        "\n  a fit is only worth what it transfers: this one is fitted and tested on one \n  \
         corpus, so it bounds the within-corpus case and says nothing about another. \n  \
         The transfer test belongs on a corpus this was not fitted on.\n"
    );
}

/// The ceiling on routing between tiers, before anybody builds a router.
///
/// Both 2025–2026 results this project took its adaptive-reranking ideas from
/// start here, and the per-query gate measured next door skipped it and failed.
/// A router's claim is that some queries need a cheaper tier than others, so the
/// question that decides whether any router can help is: **if an oracle picked
/// the cheapest tier that was good enough for each query, what would that
/// buy?** Two numbers answer it, and neither needs a model:
///
/// - **The oracle's score**, taking the best tier per query. That is the
///   unreachable ceiling — a perfect router cannot beat it — so if it sits close
///   to always running the best single tier, routing has nothing to win on
///   accuracy.
/// - **The cheapest-sufficient cost**, taking the cheapest tier within
///   [`ENOUGH`] of that query's best. That is the unreachable floor on latency,
///   and if it sits close to the best single tier's cost, routing has nothing to
///   win there either.
///
/// Both are oracles: they read the answer key. Nothing achievable does this
/// well, so a gap here is an upper bound on a router's value and the absence of
/// a gap is a **kill condition** rather than a disappointment.
///
/// Free, because the tier sweep above already holds every tier's per-query
/// score and every tier's measured cost from the same run. Priced across runs
/// this would be worthless, which is why it lives inside the arm rather than
/// beside it.
fn oracle(priced: &[(Rerank, f64, BTreeMap<String, Scores>)], named: &str) {
    /// How close to the best a tier has to be to count as good enough.
    ///
    /// Not zero. At zero a tier is "insufficient" for a query it loses by a
    /// ten-thousandth, which would route on noise and read as though the
    /// expensive tier were needed everywhere. A hundredth of nDCG is smaller
    /// than every gain this project has ever acted on and larger than the
    /// fourth-decimal movement it has measured between identical runs.
    const ENOUGH: f64 = 0.01;

    for group in GROUPS {
        let queries = priced[0].2[group].per_query.len();
        if priced
            .iter()
            .any(|(_, _, by)| by[group].per_query.len() != queries)
        {
            println!("\n  the tiers scored different numbers of {group} queries; no oracle");
            continue;
        }

        let mut best_total = 0.0;
        let mut cheapest_cost = 0.0;
        // How often each tier is the cheapest one that suffices, which is the
        // distribution a router would have to predict.
        let mut chosen = vec![0usize; priced.len()];

        for query in 0..queries {
            let best = priced
                .iter()
                .map(|(_, _, by)| by[group].per_query[query])
                .fold(f64::MIN, f64::max);
            best_total += best;

            // Cheapest by measured cost, not by tier order, because the order
            // the sweep happens to run in is not a price list.
            let (at, cost) = priced
                .iter()
                .enumerate()
                .filter(|(_, (_, _, by))| by[group].per_query[query] >= best - ENOUGH)
                .map(|(at, (_, cost, _))| (at, *cost))
                .min_by(|left, right| left.1.total_cmp(&right.1))
                .expect("the best tier always suffices for itself");
            cheapest_cost += cost;
            chosen[at] += 1;
        }

        let queries = queries as f64;
        let oracle_score = best_total / queries;
        let routed_cost = cheapest_cost / queries;

        // The best single tier on this group, which is what a router has to
        // beat rather than `off`. Beating `off` is not the claim.
        let (strongest, strongest_cost, strongest_scores) = priced
            .iter()
            .max_by(|left, right| {
                left.2[group]
                    .mean_ndcg()
                    .total_cmp(&right.2[group].mean_ndcg())
            })
            .expect("at least one tier");

        println!("\n  the ceiling on routing between tiers, {group}, {named}");
        println!("  ----------------------------------------------------------------------");
        println!(
            "  an oracle picking the best tier per query      {oracle_score:>7.4} nDCG@{NDCG_AT}"
        );
        println!(
            "  always the strongest single tier ({:<13}) {:>7.4} nDCG@{NDCG_AT}   {:>7.0} ms",
            strongest.name(),
            strongest_scores[group].mean_ndcg(),
            strongest_cost
        );
        println!(
            "  so a perfect router could add at most          {:>+7.4}",
            oracle_score - strongest_scores[group].mean_ndcg()
        );
        println!(
            "  an oracle picking the cheapest sufficient tier {:>7.0} ms a query, against \
             {:>4.0} ms",
            routed_cost, strongest_cost
        );
        println!(
            "  so a perfect router could save at most         {:>7.0} ms ({:.0}%)",
            strongest_cost - routed_cost,
            100.0 * (strongest_cost - routed_cost) / strongest_cost
        );
        println!("  and it would have to predict this distribution:");
        for (at, (tier, cost, _)) in priced.iter().enumerate() {
            if chosen[at] > 0 {
                println!(
                    "    {:<14} cheapest-sufficient on {:>5.1}% of queries ({:>6.0} ms)",
                    tier.name(),
                    100.0 * chosen[at] as f64 / queries,
                    cost
                );
            }
        }
    }
}

/// Two ways of deciding not to rerank, both swept from one pass.
///
/// The reranker is 260 ms of a 359 ms search and is *significantly negative*
/// on the same-language group -- -0.0060 at p = 0.0008, nineteen queries worse
/// against three better. So a rule that declines the pass is the only lever on
/// this project's backlog that buys latency and accuracy at once, and there are
/// two shapes it could take. Both are measured here rather than argued about.
///
/// **Per query, on lexical coverage.** A candidate-level language test was
/// tried before and rejected for a sound reason: `detect_language` returns
/// nothing for a query as short as "how does deployment work". The signal used
/// instead needs no model and no detector -- what share of the fused head at
/// least one lexical channel already proposed. If the lexical channels own the
/// head there is nothing for a cross-encoder to recover, and the pass is cost.
///
/// **Per candidate, on the model's own score.** The 2025 threshold result this
/// record cites -- dropping candidates below a cut rather than reordering all
/// of them, scoring 0.975 against 0.969 -- was not measurable here until
/// `Why::Reranked` existed. Two families of cut are tried because they test
/// different claims. An *absolute* cut on the logit is only meaningful if the
/// score carries calibration the documentation says it does not, so it is
/// included precisely to find out whether that warning bites. A *relative* cut,
/// against the best-scoring candidate of the same query, is expressible
/// whatever the calibration.
///
/// One pass produces both. Each query is searched twice, once at `off` and
/// once at the default tier, and every threshold in both families is then
/// scored from those two orderings -- so adding a threshold costs nothing and
/// the arms cannot drift apart by being taken in different runs.
async fn gate_sweep(engine: &Engine, queries: &[Query<'_>], named: &str) {
    /// Share of the fused head the lexical channels must own before the pass
    /// is declined. `0.0` declines always, `1.01` never -- the two ends
    /// reproduce `off` and the shipped tier, which is the check that the sweep
    /// is wired to the same orderings the baselines came from.
    const COVERAGE: [f64; 7] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.01];

    /// Absolute cuts on the cross-encoder's logit. Included to test the
    /// documented claim that this number is not comparable across queries: if
    /// a fixed cut works, the claim is too strong.
    const ABSOLUTE: [f32; 5] = [-8.0, -6.0, -4.0, -2.0, 0.0];

    /// Relative cuts: drop a reranked candidate scoring this far below the
    /// best-scoring reranked candidate of the same query. Expressible whatever
    /// the calibration.
    const RELATIVE: [f32; 4] = [2.0, 4.0, 6.0, 8.0];

    /// How deep the coverage signal looks, which is the depth nDCG scores.
    const HEAD: usize = NDCG_AT;

    let tier = Rerank::default();
    let mut ungated: BTreeMap<String, Scores> = BTreeMap::new();
    let mut plain: BTreeMap<String, Scores> = BTreeMap::new();
    let mut by_coverage: Vec<BTreeMap<String, Scores>> =
        COVERAGE.iter().map(|_| BTreeMap::new()).collect();
    let mut by_absolute: Vec<BTreeMap<String, Scores>> =
        ABSOLUTE.iter().map(|_| BTreeMap::new()).collect();
    let mut by_relative: Vec<BTreeMap<String, Scores>> =
        RELATIVE.iter().map(|_| BTreeMap::new()).collect();
    let mut declined = vec![0usize; COVERAGE.len()];
    let mut dropped_absolute = vec![0usize; ABSOLUTE.len()];
    let mut dropped_relative = vec![0usize; RELATIVE.len()];
    let mut reranked_total = 0usize;

    for query in queries {
        let off = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, Rerank::Off)
            .await
            .expect("search with the reranker off");
        let on = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, tier)
            .await
            .expect("search at the default tier");

        let names = |hits: &[pamin_engine::SearchHit]| -> Vec<String> {
            hits.iter().map(|hit| hit.topic.clone()).collect()
        };
        let off_order = names(&off);
        let on_order = names(&on);
        score(&mut plain, query, &off_order);
        score(&mut ungated, query, &on_order);

        // The signal, read off the ordering the gate would see: the gate runs
        // before the pass, so it can only know what fusion produced.
        let head = HEAD.min(off.len());
        let lexical = off[..head]
            .iter()
            .filter(|hit| {
                hit.result.why.iter().any(|why| {
                    matches!(
                        why,
                        Why::Channel { channel, .. }
                            if *channel == Channel::LexicalSegmented
                                || *channel == Channel::LexicalNgram
                    )
                })
            })
            .count();
        let coverage = if head == 0 {
            0.0
        } else {
            lexical as f64 / head as f64
        };

        for (at, threshold) in COVERAGE.iter().enumerate() {
            let decline = coverage >= *threshold;
            if decline {
                declined[at] += 1;
            }
            score(
                &mut by_coverage[at],
                query,
                if decline { &off_order } else { &on_order },
            );
        }

        // The model's own scores, from the arm that actually ran it.
        let scored = |hit: &pamin_engine::SearchHit| -> Option<f32> {
            hit.result.why.iter().find_map(|why| match why {
                Why::Reranked { score } => Some(*score),
                Why::Channel { .. } | Why::Path { .. } => None,
            })
        };
        let best = on.iter().filter_map(scored).fold(f32::MIN, f32::max);
        reranked_total += on.iter().filter(|hit| scored(hit).is_some()).count();

        // A candidate the model never saw is never dropped: the cut is a
        // judgement the model made, and it did not make one about those.
        let keeping = |cut: f32| -> Vec<String> {
            on.iter()
                .filter(|hit| scored(hit).is_none_or(|score| score >= cut))
                .map(|hit| hit.topic.clone())
                .collect()
        };

        for (at, cut) in ABSOLUTE.iter().enumerate() {
            let kept = keeping(*cut);
            dropped_absolute[at] += on_order.len() - kept.len();
            score(&mut by_absolute[at], query, &kept);
        }
        for (at, margin) in RELATIVE.iter().enumerate() {
            let kept = keeping(best - *margin);
            dropped_relative[at] += on_order.len() - kept.len();
            score(&mut by_relative[at], query, &kept);
        }
    }

    for group in GROUPS {
        println!("\n  gating the reranker per query, {group}, {named}");
        println!(
            "  the lexical channels own this much of the head -> decline the pass\n  \
             coverage >=   declined   nDCG@{NDCG_AT}   recall@{RECALL_AT}   against the ungated tier"
        );
        println!(
            "  ------------------------------------------------------------------------------"
        );
        for (at, threshold) in COVERAGE.iter().enumerate() {
            println!(
                "  {threshold:>11.2}   {:>7.1}%   {:>7.4}   {:>9.4}   {}",
                100.0 * declined[at] as f64 / queries.len() as f64,
                by_coverage[at][group].mean_ndcg(),
                by_coverage[at][group].mean_recall(),
                statistics::compare(&ungated[group].per_query, &by_coverage[at][group].per_query)
            );
        }
        println!(
            "  {:>11}   {:>7}   {:>7.4}   {:>9.4}   the tier, ungated",
            "--",
            "--",
            ungated[group].mean_ndcg(),
            ungated[group].mean_recall()
        );
        println!(
            "  {:>11}   {:>7}   {:>7.4}   {:>9.4}   reranking off",
            "--",
            "--",
            plain[group].mean_ndcg(),
            plain[group].mean_recall()
        );

        println!("\n  dropping candidates instead of reordering them, {group}, {named}");
        println!(
            "  cut             dropped   nDCG@{NDCG_AT}   recall@{RECALL_AT}   against the ungated tier"
        );
        println!(
            "  ------------------------------------------------------------------------------"
        );
        for (at, cut) in ABSOLUTE.iter().enumerate() {
            println!(
                "  score < {cut:<7.1} {:>7.1}%   {:>7.4}   {:>9.4}   {}",
                100.0 * dropped_absolute[at] as f64 / reranked_total.max(1) as f64,
                by_absolute[at][group].mean_ndcg(),
                by_absolute[at][group].mean_recall(),
                statistics::compare(&ungated[group].per_query, &by_absolute[at][group].per_query)
            );
        }
        for (at, margin) in RELATIVE.iter().enumerate() {
            println!(
                "  best - {margin:<8.1} {:>7.1}%   {:>7.4}   {:>9.4}   {}",
                100.0 * dropped_relative[at] as f64 / reranked_total.max(1) as f64,
                by_relative[at][group].mean_ndcg(),
                by_relative[at][group].mean_recall(),
                statistics::compare(&ungated[group].per_query, &by_relative[at][group].per_query)
            );
        }
    }

    // The premise of both tables. A sweep whose ends do not reproduce the two
    // baselines is reading something other than the orderings it thinks it is.
    let last = COVERAGE.len() - 1;
    for group in GROUPS {
        assert_eq!(
            by_coverage[0][group].mean_ndcg(),
            plain[group].mean_ndcg(),
            "declining every query did not reproduce reranking off on {group}"
        );
        assert_eq!(
            by_coverage[last][group].mean_ndcg(),
            ungated[group].mean_ndcg(),
            "declining no query did not reproduce the ungated tier on {group}"
        );
    }
    println!(
        "\n  both ends of the coverage sweep reproduce their baselines, and {reranked_total} \
         candidates over {} queries reached the model\n",
        queries.len()
    );
}

/// How few pairs the reranker can be shown before its gain goes.
///
/// The per-query gate measured next door can only take the whole pass or none
/// of it, and on parallel text it cannot even do that -- the two groups are the
/// same queries. This is the other shape of the same idea and it has neither
/// problem: the pass costs about 16.9 ms a pair on four cores, linear in pairs,
/// so showing the model half as many candidates halves the 260 ms whatever the
/// corpus looks like. The question is what accuracy that costs per pair saved.
///
/// **Exact rather than approximate, and from one pass.** `Why::Reranked` now
/// records which candidates the model saw and what it scored each one, and the
/// write-back permutes those candidates among the positions they already held.
/// So "what if only the best `n` of them had been sent" is computable: the rest
/// keep their fused positions, which is what they would have done in a run that
/// never sent them. The two ends of the sweep assert the reconstruction rather
/// than trusting it -- a cap of zero must reproduce `--rerank off` and an
/// uncapped run must reproduce the shipped tier, orderings included.
///
/// One caveat the reconstruction cannot remove. `Reranker::rank` sorts
/// candidates by length before batching, so a capped run forms different
/// batches, and a quantized model scores a pair slightly differently depending
/// on what it shared a tensor with -- measured elsewhere in this repository at
/// 0.0006 of nDCG. The scores replayed here are the ones the uncapped batches
/// produced. So a chosen setting is confirmed by a real timed run, and this
/// sweep is for choosing which setting to confirm.
async fn pairs_sweep(engine: &Engine, queries: &[Query<'_>], named: &str) {
    /// At most this many confined candidates reach the model, best-fused
    /// first. `usize::MAX` is today's rule: every confined candidate in the
    /// head.
    const CAPS: [usize; 8] = [0, 2, 4, 6, 8, 10, 12, usize::MAX];

    let tier = Rerank::default();
    let head = tier.depth();

    let mut shipped: BTreeMap<String, Scores> = BTreeMap::new();
    let mut fused_only: BTreeMap<String, Scores> = BTreeMap::new();
    let mut capped: Vec<BTreeMap<String, Scores>> = CAPS.iter().map(|_| BTreeMap::new()).collect();
    let mut pairs = vec![0usize; CAPS.len()];
    let mut ran = 0usize;

    for query in queries {
        let off = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, Rerank::Off)
            .await
            .expect("search with the reranker off");
        let on = engine
            .search_reranked(query.text(), DEPTH as u32, DEPTHS, tier)
            .await
            .expect("search at the default tier");

        let fused: Vec<String> = off.iter().map(|hit| hit.topic.clone()).collect();
        let shipped_order: Vec<String> = on.iter().map(|hit| hit.topic.clone()).collect();
        score(&mut fused_only, query, &fused);
        score(&mut shipped, query, &shipped_order);

        // What the model scored each candidate it saw, keyed by topic because
        // the position it holds is what the sweep is varying.
        let scored: HashMap<&str, f32> = on
            .iter()
            .filter_map(|hit| {
                hit.result
                    .why
                    .iter()
                    .find_map(|why| match why {
                        Why::Reranked { score } => Some(*score),
                        Why::Channel { .. } | Why::Path { .. } => None,
                    })
                    .map(|score| (hit.topic.as_str(), score))
            })
            .collect();

        // The confined set, as the engine builds it: positions inside the
        // tier's depth that no lexical channel proposed, ascending, so the
        // first is the best-fused candidate the pass could move.
        let confined: Vec<usize> = (0..head.min(off.len()))
            .filter(|position| {
                !off[*position].result.why.iter().any(|why| {
                    matches!(
                        why,
                        Why::Channel { channel, .. }
                            if *channel == Channel::LexicalSegmented
                                || *channel == Channel::LexicalNgram
                    )
                })
            })
            .collect();
        // `can_be_seen`: fewer than two candidates cannot be reordered, and a
        // pass whose best candidate is past the caller's limit changes nothing
        // the caller reads. When it declines, every cap gives the fused order.
        let pass_ran = confined.len() >= 2 && confined[0] < DEPTH;
        if pass_ran {
            ran += 1;
        }

        for (at, cap) in CAPS.iter().enumerate() {
            if !pass_ran {
                score(&mut capped[at], query, &fused);
                continue;
            }
            let sent = &confined[..(*cap).min(confined.len())];
            pairs[at] += sent.len();

            let mut order: Vec<usize> = sent.to_vec();
            // Score descending, then by the position the candidate held --
            // which is the order `rank` was handed them in, so this is the
            // same tie-break the engine applies.
            order.sort_by(|left, right| {
                let of = |position: &usize| {
                    scored
                        .get(fused[*position].as_str())
                        .copied()
                        .unwrap_or(f32::MIN)
                };
                of(right).total_cmp(&of(left)).then_with(|| left.cmp(right))
            });

            let mut ranked = fused.clone();
            for (slot, from) in sent.iter().zip(&order) {
                ranked[*slot] = fused[*from].clone();
            }
            score(&mut capped[at], query, &ranked);
        }
    }

    for group in GROUPS {
        println!("\n  how few pairs the reranker needs, {group}, {named}");
        println!(
            "  pairs sent   per query   nDCG@{NDCG_AT}   recall@{RECALL_AT}   against the shipped pass"
        );
        println!(
            "  ------------------------------------------------------------------------------"
        );
        for (at, cap) in CAPS.iter().enumerate() {
            let label = if *cap == usize::MAX {
                "all (ships)".to_string()
            } else {
                format!("at most {cap}")
            };
            println!(
                "  {label:<12} {:>9.2}   {:>7.4}   {:>9.4}   {}",
                pairs[at] as f64 / queries.len() as f64,
                capped[at][group].mean_ndcg(),
                capped[at][group].mean_recall(),
                statistics::compare(&shipped[group].per_query, &capped[at][group].per_query)
            );
        }
    }

    // The premise, asserted on the orderings rather than on the means: a
    // reconstruction that agrees on average and not per query would hide
    // exactly the mistake this is checking for.
    let last = CAPS.len() - 1;
    for group in GROUPS {
        assert_eq!(
            capped[0][group].per_query, fused_only[group].per_query,
            "sending no pairs did not reproduce reranking off on {group}"
        );
        assert_eq!(
            capped[last][group].per_query, shipped[group].per_query,
            "sending every pair did not reproduce the shipped pass on {group}, so the \
             reconstruction of the write-back is wrong and every row above is wrong with it"
        );
    }
    println!(
        "\n  the reconstruction reproduces both ends per query, and the pass ran on {ran} of {} \
         queries\n",
        queries.len()
    );
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

/// Every rule in `reranking::rules` over the shipped tier's own scores.
async fn rerank_rules(engine: &Engine, queries: &[Query<'_>], named: &str) {
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let rules = reranking::rules();
    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); rules.len()];
    for query in queries {
        let hits = engine
            .search_reranked(query.text(), WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);
        for ((_, rule), into) in rules.iter().zip(&mut measured) {
            score(into, query, &replayed.order(*rule));
        }
    }
    reranking::report(
        &format!(
            "rules for the {} reranker's scores, XQuAD-R, {named}",
            tier.name()
        ),
        &rules,
        reranking::shipped(&rules),
        &measured,
    );
}

/// What the shipped tier is worth when it is shown each candidate's topic
/// name, and the memory the graph reached it from. See `reranking::in_context`.
///
/// This corpus's names are keys chosen to be unlike any text, so the named
/// renderings ask what a name that carries nothing costs; it has no edges, so
/// the seeded renderings equal their unseeded counterparts.
async fn context(engine: &Engine, workspace: &Workspace, queries: &[Query<'_>], named: &str) {
    const WIDE: u32 = 4 * DEPTHS.channel + 4 * DEPTHS.channel / 2;
    let tier = Rerank::default();
    let mut model = pamin_index::Reranker::load(tier, &workspace.root().join("models"))
        .expect("load the reranker");
    let labels = reranking::context_labels();
    let mut measured: Vec<BTreeMap<String, Scores>> = vec![BTreeMap::new(); labels.len()];
    for query in queries {
        let hits = engine
            .search_reranked(query.text(), WIDE, DEPTHS, tier)
            .await
            .expect("search");
        channels::enough_room(&hits, WIDE);
        let replayed = reranking::replay(&hits, tier);
        let (orders, _) = reranking::in_context(&hits, &replayed, &mut model, query.text());
        for (order, into) in orders.iter().zip(&mut measured) {
            score(into, query, order);
        }
    }
    reranking::report(
        &format!(
            "what the {} reranker is shown, XQuAD-R, {named}",
            tier.name()
        ),
        &labels,
        reranking::shipped_context(),
        &measured,
    );
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
