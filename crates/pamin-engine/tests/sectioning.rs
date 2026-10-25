//! What the size of a section is worth to retrieval: Qasper, through `pamin
//! ingest`'s cutting and `pamin search`'s entry point.
//!
//! **What this is here to settle.** `pamin ingest` cuts a file into sections
//! sized so the reranker reads all of each beside its name -- about two
//! hundred tokens -- and that size was chosen from the reranker's window and
//! from published chunking studies, not from a measurement here. This asks
//! whether a section of another size retrieves better on documents nobody
//! here wrote, and what recording a file whole, without cutting it, would
//! cost.
//!
//! **The corpus.** Qasper (Dasigi et al., NAACL 2021): NLP papers in their
//! sections and paragraphs, and questions about each one written by people who
//! had read only its title and abstract, answered by others who cited the
//! paragraphs that hold the answer. Each paper becomes one Markdown file -- its
//! title, its abstract, and every section under a heading of its depth -- and
//! goes through `sections` as a file from a repository would. A question is
//! kept when some answer cites at least one paragraph of text; figures and
//! tables are left out as evidence, and an evidence paragraph that cannot be
//! found in its paper is dropped and counted.
//!
//! **What is measured.** Every question through `Engine::search_reranked`, the
//! function `pamin search` calls, at the product's depths. Two scores:
//!
//! - **evidence recall within a budget**, the primary one. The hits are read
//!   in rank order, as a caller filling a context window would, until
//!   [`BUDGET`] tokens of content are taken; an evidence paragraph counts when
//!   at least [`COVERED`] of its distinct words are among what was taken. A
//!   fixed budget is what makes sizes comparable: a larger section holds more
//!   text, and without a budget it would be credited for the size and not for
//!   the retrieval.
//! - **nDCG@10**, with a memory relevant when it covers an evidence paragraph
//!   by the same rule, and the ideal built from every memory of the paper that
//!   does.
//!
//! **The arms**, each a project of its own built from the same papers:
//! `whole`, every paper one memory, which is what recording files without
//! cutting them would be; `shipped`, what `sections` cuts; and the same cut at
//! fixed caps of [`CAPS`] tokens, through `sections_capped`.
//!
//! **The rule, written before the table was read.** The shipped size stays
//! unless the setting a five-fold cross-validation chooses -- folds by paper,
//! choosing the highest mean budget recall on the other folds -- beats it on
//! held-out questions with p < 0.05, and every fold chooses the same setting.
//! Otherwise the sweep has found nothing to change. Every arm's difference
//! from `shipped` is also reported with its family-adjusted p.
//!
//! **Crowding.** For each arm, the same rankings with at most [`PER_FILE`]
//! sections of one file kept, scored the same way and paired against the
//! arm's own rankings. Qasper asks about one paper at a time, so a limit can
//! only cost here; what it would cost is the number this reports.
//!
//! **Licence.** Qasper is CC BY 4.0. It is fetched at run time from the
//! HuggingFace datasets-server, not vendored.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p pamin-engine --test sectioning -- --ignored --nocapture
//! ```
//!
//! | variable | what it does |
//! |---|---|
//! | `QASPER_DIR` | where cached pages are, instead of fetching them |
//! | `QASPER_PAPERS` | how many papers to draw, default [`PAPERS`] |
//! | `EVAL_SEED` | the seed papers are drawn with |
//! | `SECTIONING_OUT` | where every question's scores are written, default under `PAMIN_EVAL_HOME` |

mod harness;
mod scoring;
mod statistics;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

use pamin_engine::Engine;
use pamin_index::sections::{Section, sections, sections_capped};
use pamin_index::spans::TokenCount;
use pamin_index::{Access, SpanTokens, VectorIndex};
use pamin_store::Workspace;

use scoring::Scores;

/// Where the rows come from: the test split, 416 papers.
const ROWS: &str = "https://datasets-server.huggingface.co/rows\
                    ?dataset=allenai%2Fqasper&config=qasper&split=test";

/// The server's own maximum.
const PER_PAGE: usize = 100;

/// Every paper in the split.
const SPLIT: usize = 416;

/// Papers drawn by default: enough questions to see a difference of a few
/// points, few enough sections to index every arm in under an hour on four
/// cores.
const PAPERS: usize = 100;

/// Tokens of content a caller is taken to read, in rank order.
const BUDGET: usize = 1_000;

/// The share of an evidence paragraph's distinct words that has to be read for
/// it to count as found.
const COVERED: f64 = 0.5;

/// The fixed caps swept beside what ships.
const CAPS: [usize; 3] = [128, 320, 448];

/// Sections of one file kept by the crowding arm.
const PER_FILE: usize = 3;

/// How many hits a search asks for: enough to fill the budget at the smallest
/// cap and to leave the crowding arm something to promote.
const LIMIT: u32 = 30;

/// The directory every paper is ingested from, recorded with its sections as
/// `pamin ingest` records a repository's root.
const ROOT: &str = "/qasper";

struct Paper {
    id: String,
    /// The name its sections go by: `qasper/{id}.md`.
    name: String,
    markdown: String,
}

struct Question {
    paper: String,
    /// The name of the paper's file, which a hit has to be cut from to count.
    file: String,
    text: String,
    /// Each annotator's evidence: the content words of every paragraph they
    /// cited. A question is answered as well as its best-served annotator,
    /// as Qasper scores it, rather than only when every annotator's
    /// paragraphs are all found.
    evidence: Vec<Vec<HashSet<String>>>,
}

/// One way of cutting the papers.
#[derive(Clone, Copy)]
enum Arm {
    Whole,
    Shipped,
    Capped(usize),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Self::Whole => "whole".into(),
            Self::Shipped => "shipped".into(),
            Self::Capped(cap) => format!("cap{cap}"),
        }
    }

    fn cut(self, paper: &Paper, counter: &dyn TokenCount) -> Vec<Section> {
        match self {
            Self::Whole => vec![Section {
                name: paper.name.clone(),
                content: paper.markdown.clone(),
            }],
            Self::Shipped => sections(&paper.name, &paper.markdown, counter),
            Self::Capped(cap) => sections_capped(&paper.name, &paper.markdown, counter, cap),
        }
    }
}

/// What one arm scored, question by question.
#[derive(serde::Serialize)]
struct Scored {
    arm: String,
    memories: usize,
    /// Memories this run recorded; fewer than `memories` means the project was
    /// built by an earlier run, and `ingest_s` timed nothing.
    written: usize,
    budget_recall: Vec<f64>,
    ndcg: Vec<f64>,
    crowded_budget_recall: Vec<f64>,
    /// Questions whose hits held at least the budget: the rest were scored on
    /// less reading than the budget allows.
    filled: usize,
    search_ms: Vec<f64>,
    ingest_s: Option<f64>,
}

/// Words too common to say whether a paragraph was found: half the distinct
/// words of a short paragraph are often these, and any text has them.
const STOPWORDS: [&str; 40] = [
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "these", "those", "from",
    "into", "than", "then", "they", "their", "there", "which", "while", "where", "when", "what",
    "who", "has", "have", "had", "not", "but", "can", "our", "its", "also", "such", "both", "each",
    "all", "any", "more", "most",
];

/// The distinct content words of `text`, lowercased: what coverage is counted
/// in. Words of fewer than three characters and [`STOPWORDS`] are left out.
fn words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.chars().count() >= 3)
        .map(str::to_lowercase)
        .filter(|word| !STOPWORDS.contains(&word.as_str()))
        .collect()
}

/// Whether `found` holds at least [`COVERED`] of `evidence`.
fn covers(found: &HashSet<String>, evidence: &HashSet<String>) -> bool {
    !evidence.is_empty()
        && evidence.iter().filter(|word| found.contains(*word)).count() as f64
            >= COVERED * evidence.len() as f64
}

/// The share of `question`'s evidence found within [`BUDGET`] tokens of
/// `hits`, read in order -- the best-served annotator's -- and whether the
/// hits held the whole budget. The last hit read is cut at the budget; every
/// hit spends it, and only a hit from the question's own paper can find its
/// evidence.
fn budget_recall(
    question: &Question,
    hits: &[(&str, &str)],
    counter: &dyn TokenCount,
) -> (f64, bool) {
    let mut left = BUDGET;
    let mut read = HashSet::new();
    for (name, content) in hits {
        if left == 0 {
            break;
        }
        let offsets = counter.offsets(content);
        let taken = if offsets.len() <= left {
            *content
        } else {
            &content[..offsets[left].start]
        };
        left = left.saturating_sub(offsets.len());
        if file_of(name) == question.file {
            read.extend(words(taken));
        }
    }
    let best = question
        .evidence
        .iter()
        .map(|cited| {
            cited
                .iter()
                .filter(|paragraph| covers(&read, paragraph))
                .count() as f64
                / cited.len() as f64
        })
        .fold(0.0, f64::max);
    (best, left == 0)
}

/// Whether a memory is from `question`'s paper and covers any paragraph an
/// annotator cited.
fn relevant(question: &Question, name: &str, content: &str) -> bool {
    if file_of(name) != question.file {
        return false;
    }
    let found = words(content);
    question
        .evidence
        .iter()
        .flatten()
        .any(|paragraph| covers(&found, paragraph))
}

/// The file a section was cut from: its name before its headings and parts.
fn file_of(name: &str) -> &str {
    let file = name.split(" › ").next().unwrap_or(name);
    match file.find(".md") {
        Some(at) => &file[..at + 3],
        None => file,
    }
}

/// `hits` with at most [`PER_FILE`] of any one file.
fn spread<'a>(hits: &[(&'a str, &'a str)]) -> Vec<(&'a str, &'a str)> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    hits.iter()
        .filter(|(name, _)| {
            let count = seen.entry(file_of(name)).or_default();
            *count += 1;
            *count <= PER_FILE
        })
        .copied()
        .collect()
}

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
         The dataset is not vendored: Qasper is CC BY 4.0 and this fetches it\n\
         at run time. Make `curl` and datasets-server.huggingface.co reachable,\n\
         or point QASPER_DIR at a directory of cached pages."
    );
    std::fs::rename(&partial, path).expect("name the downloaded page");
}

/// A paper as Markdown: its title, its abstract, and each section under a
/// heading as deep as its place in the paper -- Qasper names a subsection
/// `Parent ::: Child`.
fn markdown(row: &serde_json::Value) -> String {
    let mut out = format!(
        "# {}\n\n## Abstract\n\n{}\n",
        row["title"].as_str().unwrap_or_default().trim(),
        row["abstract"].as_str().unwrap_or_default().trim()
    );
    let full = &row["full_text"];
    let names = full["section_name"].as_array().cloned().unwrap_or_default();
    let paragraphs = full["paragraphs"].as_array().cloned().unwrap_or_default();
    let mut open: Vec<String> = Vec::new();
    for (name, paragraphs) in names.iter().zip(&paragraphs) {
        let path: Vec<String> = name
            .as_str()
            .unwrap_or_default()
            .split(":::")
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect();
        for (depth, part) in path.iter().enumerate() {
            if open.get(depth) == Some(part) {
                continue;
            }
            open.truncate(depth);
            open.push(part.clone());
            out.push_str(&format!("\n{} {part}\n", "#".repeat((depth + 2).min(6))));
        }
        open.truncate(path.len());
        for paragraph in paragraphs.as_array().into_iter().flatten() {
            let paragraph = paragraph.as_str().unwrap_or_default().trim();
            if !paragraph.is_empty() {
                out.push_str(&format!("\n{paragraph}\n"));
            }
        }
    }
    out
}

/// The papers and questions, drawn by a seeded hash so a run can be repeated.
fn corpus() -> (Vec<Paper>, Vec<Question>) {
    let dir = harness::dataset_dir("QASPER_DIR", "qasper");
    std::fs::create_dir_all(&dir).expect("dataset directory");
    let mut rows = Vec::new();
    for offset in (0..SPLIT).step_by(PER_PAGE) {
        let path = dir.join(format!("test-{offset}.json"));
        download(&path, &format!("{ROWS}&offset={offset}&length={PER_PAGE}"));
        let page: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read a page")).expect("a page");
        rows.extend(
            page["rows"]
                .as_array()
                .expect("rows")
                .iter()
                .map(|row| row["row"].clone()),
        );
    }
    assert_eq!(rows.len(), SPLIT, "a short read of the split");

    let wanted: usize = std::env::var("QASPER_PAPERS")
        .ok()
        .and_then(|papers| papers.parse().ok())
        .unwrap_or(PAPERS);
    let seed = harness::shape::seed();
    let draw = |id: &str| {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (seed, id).hash(&mut hasher);
        hasher.finish()
    };
    rows.sort_by_key(|row| draw(row["id"].as_str().unwrap_or_default()));
    rows.truncate(wanted);

    let mut papers = Vec::new();
    let mut questions = Vec::new();
    let (mut cited, mut lost) = (0usize, 0usize);
    for row in &rows {
        let id = row["id"].as_str().expect("an id").to_string();
        let markdown = markdown(row);
        let text = markdown.split_whitespace().collect::<Vec<_>>().join(" ");
        let qas = &row["qas"];
        let asked = qas["question"].as_array().cloned().unwrap_or_default();
        let answered = qas["answers"].as_array().cloned().unwrap_or_default();
        for (question, answers) in asked.iter().zip(&answered) {
            let mut evidence = Vec::new();
            for answer in answers["answer"].as_array().into_iter().flatten() {
                if answer["unanswerable"].as_bool() == Some(true) {
                    continue;
                }
                let mut found = BTreeSet::new();
                for paragraph in answer["evidence"].as_array().into_iter().flatten() {
                    let paragraph = paragraph.as_str().unwrap_or_default().trim();
                    if paragraph.is_empty() || paragraph.starts_with("FLOAT SELECTED") {
                        continue;
                    }
                    cited += 1;
                    let flat = paragraph.split_whitespace().collect::<Vec<_>>().join(" ");
                    if text.contains(&flat) {
                        found.insert(flat);
                    } else {
                        lost += 1;
                    }
                }
                let found: Vec<HashSet<String>> = found
                    .iter()
                    .map(|paragraph| words(paragraph))
                    .filter(|paragraph| !paragraph.is_empty())
                    .collect();
                if !found.is_empty() {
                    evidence.push(found);
                }
            }
            if evidence.is_empty() {
                continue;
            }
            questions.push(Question {
                paper: id.clone(),
                file: format!("qasper/{id}.md"),
                text: question.as_str().unwrap_or_default().trim().to_string(),
                evidence,
            });
        }
        papers.push(Paper {
            name: format!("qasper/{id}.md"),
            id,
            markdown,
        });
    }
    println!(
        "  {} papers, {} questions; {lost} of {cited} cited paragraphs not found in their paper",
        papers.len(),
        questions.len()
    );
    // The premise: evidence is judged by finding it, so most of it has to be
    // findable in the papers as they are written here.
    assert!(
        lost * 10 <= cited,
        "{lost} of {cited} evidence paragraphs are not in their papers as written"
    );
    (papers, questions)
}

/// One arm: its project built and drained, every question asked.
async fn run(
    arm: Arm,
    workspace: &Workspace,
    papers: &[Paper],
    questions: &[Question],
    counter: &dyn TokenCount,
    fingerprint: u64,
) -> Scored {
    let label = arm.label();
    let cut: Vec<(String, Vec<Section>)> = papers
        .iter()
        .map(|paper| (paper.id.clone(), arm.cut(paper, counter)))
        .collect();
    let memories: usize = cut.iter().map(|(_, sections)| sections.len()).sum();
    let names: HashSet<&str> = cut
        .iter()
        .flat_map(|(_, sections)| sections.iter().map(|section| section.name.as_str()))
        .collect();
    assert_eq!(
        names.len(),
        memories,
        "{label}: two sections under one name"
    );

    let engine = Engine::open(
        workspace,
        &format!("qasper-{label}-{fingerprint:016x}"),
        VectorIndex::default(),
        Access::ReadWrite,
        &harness::settings(),
    )
    .await
    .expect("open the engine");

    let started = Instant::now();
    let mut written = 0;
    for (_, sections) in &cut {
        for section in sections {
            let recorded = engine
                .remember_section(
                    &section.name,
                    &section.content,
                    pamin_core::Validity::ALWAYS,
                    ROOT,
                )
                .await
                .expect("record a section");
            written += usize::from(recorded.is_some());
        }
    }
    harness::drain(&engine).await;
    // Timed only when this run built the project: one an earlier run built is
    // read back unchanged, and the time is the time to find that out.
    let ingest_s = (written == memories).then(|| started.elapsed().as_secs_f64());
    if ingest_s.is_none() {
        println!(
            "  {label}: {written} of {memories} memories written; the rest were already there"
        );
    }
    // The premise of every score below: the project holds this arm's memories,
    // all of them, and nothing else.
    assert_eq!(
        engine.indexed_documents().expect("count documents") as usize,
        memories,
        "{label}: the index does not hold the arm's memories"
    );

    // Every memory of a paper that covers some of a question's evidence: the
    // ideal nDCG is built from these.
    let by_paper: BTreeMap<&str, &Vec<Section>> = cut
        .iter()
        .map(|(id, sections)| (id.as_str(), sections))
        .collect();

    let mut scores = Scores::default();
    let (mut budget, mut crowded, mut search_ms) = (Vec::new(), Vec::new(), Vec::new());
    let mut filled = 0;
    for question in questions {
        let started = Instant::now();
        let hits = engine
            .search_reranked(&question.text, LIMIT, harness::depths())
            .await
            .expect("search");
        search_ms.push(started.elapsed().as_secs_f64() * 1e3);

        let ranked: Vec<(&str, &str)> = hits
            .iter()
            .map(|hit| (hit.topic.as_str(), hit.state.content.as_str()))
            .collect();
        let (recall, full) = budget_recall(question, &ranked, counter);
        budget.push(recall);
        filled += usize::from(full);
        crowded.push(budget_recall(question, &spread(&ranked), counter).0);

        let judged = by_paper[question.paper.as_str()]
            .iter()
            .filter(|section| relevant(question, &section.name, &section.content))
            .count();
        let relevant_names: HashSet<&str> = ranked
            .iter()
            .filter(|(name, content)| relevant(question, name, content))
            .map(|(name, _)| *name)
            .collect();
        let topics: Vec<String> = ranked.iter().map(|(name, _)| name.to_string()).collect();
        scores.add(&topics, judged, |name| relevant_names.contains(name));
    }

    Scored {
        arm: label,
        memories,
        written,
        budget_recall: budget,
        ndcg: scores.per_query,
        crowded_budget_recall: crowded,
        filled,
        search_ms,
        ingest_s,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn percentile(values: &[f64], at: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() - 1) as f64 * at).round() as usize]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres, downloads the dataset and model weights, and embeds the corpus five times"]
async fn a_sections_size_against_retrieval() {
    let (papers, questions) = corpus();
    assert!(
        statistics::gate(
            "Qasper, a section size against the shipped one",
            questions.len()
        ),
        "too few questions to compare sizes"
    );

    let home = harness::eval_home();
    std::fs::create_dir_all(&home).expect("eval home");
    let workspace = Workspace::at(&home);
    let counter = SpanTokens::load(&workspace.models_dir()).expect("BGE-M3's tokenizer");
    let fingerprint = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for paper in &papers {
            paper.id.hash(&mut hasher);
        }
        hasher.finish()
    };

    let arms: Vec<Arm> = [Arm::Whole, Arm::Shipped]
        .into_iter()
        .chain(CAPS.iter().map(|cap| Arm::Capped(*cap)))
        .collect();
    let mut scored = Vec::new();
    for arm in arms {
        let one = run(arm, &workspace, &papers, &questions, &counter, fingerprint).await;
        println!(
            "  {:<8} {:>6} memories  budget recall {:.4}  nDCG@10 {:.4}  per file {:.4}  \
             budget filled {}/{}  search p50 {:.0} ms p95 {:.0} ms  ingest {}",
            one.arm,
            one.memories,
            mean(&one.budget_recall),
            mean(&one.ndcg),
            mean(&one.crowded_budget_recall),
            one.filled,
            questions.len(),
            percentile(&one.search_ms, 0.5),
            percentile(&one.search_ms, 0.95),
            one.ingest_s.map_or(
                "not timed: built by an earlier run".to_string(),
                |s| format!("{s:.0} s")
            )
        );
        scored.push(one);
    }

    let out = std::env::var("SECTIONING_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| home.join(format!("sectioning-{fingerprint:016x}.json")));
    std::fs::write(&out, serde_json::to_vec_pretty(&scored).expect("serialize"))
        .expect("write the scores");
    println!("  every question's scores: {}", out.display());

    let shipped = scored
        .iter()
        .position(|one| one.arm == "shipped")
        .expect("the shipped arm");
    let others: Vec<usize> = (0..scored.len()).filter(|at| *at != shipped).collect();

    // Questions about one paper share its sections and its index, so they are
    // not independent: every test is over papers, each a cluster of its
    // questions' scores, as the repository's other clustered corpora are.
    let groups = vec![0; questions.len()];
    let keys: Vec<String> = questions
        .iter()
        .map(|question| question.paper.clone())
        .collect();
    let by_paper = |scores: &[f64]| statistics::cluster_macro_contributions(scores, &groups, &keys);

    println!("\n  against shipped, paired by paper:");
    let adjusted = statistics::family(
        &by_paper(&scored[shipped].budget_recall),
        &others
            .iter()
            .map(|at| by_paper(&scored[*at].budget_recall))
            .collect::<Vec<_>>(),
    );
    for (at, p) in others.iter().zip(&adjusted) {
        let budget = statistics::compare(
            &by_paper(&scored[shipped].budget_recall),
            &by_paper(&scored[*at].budget_recall),
        );
        let ndcg = statistics::compare(
            &by_paper(&scored[shipped].ndcg),
            &by_paper(&scored[*at].ndcg),
        );
        println!(
            "  {:<8} budget recall {budget} (family p {p:.4})  nDCG@10 {ndcg}",
            scored[*at].arm
        );
    }
    println!("\n  a limit of {PER_FILE} sections a file, against each arm's own rankings:");
    for one in &scored {
        println!(
            "  {:<8} {}",
            one.arm,
            statistics::compare(
                &by_paper(&one.budget_recall),
                &by_paper(&one.crowded_budget_recall)
            )
        );
    }

    // The rule: cross-validated by paper, choosing the highest mean budget
    // recall; ship a change only if the held-out score beats shipped
    // significantly, over papers, and every fold agreed.
    let folds = statistics::cluster_folds(&groups, &keys, 5);
    let settings: Vec<Vec<f64>> = scored.iter().map(|one| one.budget_recall.clone()).collect();
    let selected = statistics::cross_validate(&settings, &groups, &folds, |table| {
        (0..table.len())
            .max_by(|left, right| table[*left][0].total_cmp(&table[*right][0]))
            .expect("a setting")
    });
    let held = statistics::compare(
        &by_paper(&scored[shipped].budget_recall),
        &by_paper(&selected.held_out),
    );
    let chosen: Vec<&str> = selected
        .chosen
        .iter()
        .map(|at| scored[*at].arm.as_str())
        .collect();
    let agreed = selected.chosen.windows(2).all(|pair| pair[0] == pair[1]);
    println!("\n  cross-validated choice by fold: {chosen:?}; held out against shipped: {held}");
    let change =
        agreed && held.is_significant() && held.mean > 0.0 && selected.chosen[0] != shipped;
    println!(
        "  rule: {}",
        if change {
            format!("change the size to {}", scored[selected.chosen[0]].arm)
        } else {
            "keep the shipped size: the sweep found nothing to change".to_string()
        }
    );
}

#[test]
fn a_file_is_read_from_a_sections_name() {
    assert_eq!(
        file_of("qasper/1909.00001.md › Intro (2)"),
        "qasper/1909.00001.md"
    );
    assert_eq!(file_of("qasper/1909.00001.md"), "qasper/1909.00001.md");
}

#[test]
fn coverage_is_a_share_of_distinct_words() {
    let evidence = words("The model uses a BiLSTM encoder and a CRF decoder.");
    assert_eq!(evidence.len(), 6, "{evidence:?}");
    assert!(covers(&words("uses a bilstm encoder"), &evidence));
    assert!(!covers(&words("a CRF"), &evidence));
    // Common words carry nothing: text made of them finds nothing.
    assert!(!covers(
        &words("the and with that this from which"),
        &evidence
    ));
    assert!(!covers(&words("anything"), &HashSet::new()));
}

#[test]
fn a_paper_becomes_markdown_with_nested_headings() {
    let row = serde_json::json!({
        "title": "A Paper",
        "abstract": "What it does.",
        "full_text": {
            "section_name": ["Intro", "Data ::: Collection", "Data ::: Cleaning"],
            "paragraphs": [["First."], ["Gathered."], ["Cleaned."]]
        }
    });
    assert_eq!(
        markdown(&row),
        "# A Paper\n\n## Abstract\n\nWhat it does.\n\n## Intro\n\nFirst.\n\n## Data\n\n\
         ### Collection\n\nGathered.\n\n### Cleaning\n\nCleaned.\n"
    );
}

/// One token a word, for the scoring tests.
struct Spaces;

impl TokenCount for Spaces {
    fn offsets(&self, text: &str) -> Vec<std::ops::Range<usize>> {
        let mut offsets = Vec::new();
        let mut start = None;
        for (at, c) in text.char_indices() {
            match (c.is_whitespace(), start) {
                (false, None) => start = Some(at),
                (true, Some(from)) => {
                    offsets.push(from..at);
                    start = None;
                }
                _ => {}
            }
        }
        offsets.extend(start.map(|from| from..text.len()));
        offsets
    }
}

fn question(evidence: &[&[&str]]) -> Question {
    Question {
        paper: "p".into(),
        file: "qasper/p.md".into(),
        text: String::new(),
        evidence: evidence
            .iter()
            .map(|cited| cited.iter().map(|paragraph| words(paragraph)).collect())
            .collect(),
    }
}

/// Only the question's own paper can find its evidence, though every hit spends
/// the budget; and the question scores as its best-served annotator.
#[test]
fn budget_recall_reads_the_right_paper_within_the_budget() {
    let asked = question(&[
        &["bilstm encoder crf decoder", "attention heads pruned"],
        &["dropout rate tuned"],
    ]);
    let filler = "word ".repeat(BUDGET);

    // Found, in the right paper.
    let (found, full) = budget_recall(
        &asked,
        &[("qasper/p.md › Model", "a bilstm encoder with a crf decoder")],
        &Spaces,
    );
    assert_eq!(found, 0.5);
    assert!(!full, "seven words do not fill the budget");

    // The same words in another paper find nothing.
    let (found, _) = budget_recall(
        &asked,
        &[("qasper/q.md › Model", "a bilstm encoder with a crf decoder")],
        &Spaces,
    );
    assert_eq!(found, 0.0);

    // Behind a hit that spends the whole budget, nothing is read.
    let (found, full) = budget_recall(
        &asked,
        &[
            ("qasper/q.md", filler.as_str()),
            ("qasper/p.md › Model", "dropout rate tuned"),
        ],
        &Spaces,
    );
    assert_eq!(found, 0.0);
    assert!(full);

    // The second annotator is fully served, so the question is.
    let (found, _) = budget_recall(
        &asked,
        &[("qasper/p.md › Training", "the dropout rate was tuned")],
        &Spaces,
    );
    assert_eq!(found, 1.0);
}
