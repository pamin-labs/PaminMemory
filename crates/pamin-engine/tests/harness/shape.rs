//! What a run was asked, written down so that two runs can be shown to be the
//! same run -- or shown not to be.
//!
//! Three things every harness owes its reader, and that each used to leave
//! implicit:
//!
//! - **Which questions and documents, exactly.** A project name carries a
//!   64-bit fingerprint of the documents, which is enough to keep a cache
//!   apart and not enough to say two logs measured the same thing: it does
//!   not cover the questions, the answer key, or which of them a sample drew.
//!   [`Shape::digest`] is a SHA-256 over all of those.
//! - **How they were chosen.** "The first 600" of MuSiQue were all two-hop,
//!   because the dataset is ordered by hop count, and nothing in the log said
//!   so. A sample here is a seeded hash, stratified, drawn without looking at
//!   any outcome, and the seed is printed beside the digest.
//! - **What was not scored, and why.** A question a harness cannot grade was
//!   dropped where it was found, and a reader had no way to know there were
//!   any. Each is now listed with its reason, and every scored table is
//!   checked against `asked - excluded` before it is believed.
//!
//! The discipline is the Jev Decision Index's: a pinned corpus digest, a
//! deterministic seeded sample that keeps linked groups whole, and explicit
//! coverage with no silent replacement of failures.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// The seed a run uses when `EVAL_SEED` does not name one.
///
/// Fixed rather than drawn, so that a sample is the same sample on every
/// machine and a rerun is a rerun. Changing it is how to ask whether a result
/// survives a different draw.
pub const SEED: u64 = 20_261_012;

/// `EVAL_SEED`, or [`SEED`].
pub fn seed() -> u64 {
    std::env::var("EVAL_SEED")
        .map(|value| value.parse().expect("EVAL_SEED is an unsigned integer"))
        .unwrap_or(SEED)
}

/// Where `key` falls in the seeded order: SHA-256 of the seed and the key.
///
/// A hash rather than a shuffle, so an item's place depends on nothing but its
/// own key -- not on which other items were loaded, nor in what order -- and a
/// larger sample of one stratum contains every smaller one.
fn order(seed: u64, key: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(seed.to_be_bytes());
    hash.update(key.as_bytes());
    hash.finalize().into()
}

/// One of `options`, chosen by the seed and `key` alone.
pub fn choose<'a, T>(seed: u64, key: &str, options: &'a [T]) -> &'a T {
    assert!(!options.is_empty(), "nothing to choose from for {key}");
    let hash = order(seed, key);
    let at = u64::from_be_bytes(hash[..8].try_into().expect("eight bytes"));
    &options[(at % options.len() as u64) as usize]
}

/// `wanted` of `items`, drawn per stratum in proportion to its size, without
/// replacement, by the seeded order of each item's key.
///
/// Proportional, so the sample's mix is the dataset's and a figure over all
/// of it means what it would over the whole; each stratum is still reported on
/// its own. Kept in the order the items came in, so a harness that walks them
/// walks them as the dataset lists them.
///
/// **Allocated one unit at a time, so a larger sample contains every smaller
/// one.** Each unit goes to the stratum with the largest `size / (quota + 1)`
/// (D'Hondt), ties to the stratum whose name hashes first under the seed. The
/// first version split by largest remainder, which is not monotone in the
/// sample size: strata of 1, 3 and 3 got (1, 1, 1) at three and (0, 2, 2) at
/// four, so growing a sample could drop a question it had asked -- and empty
/// a stratum the caller asserts is reached. Units handed out in sequence are
/// never taken back, and each stratum's members are taken in the same seeded
/// order, so that cannot happen.
///
/// An item is whatever the caller keeps together -- one MuSiQue question with
/// its twenty paragraphs, one LOCOMO conversation -- so a linked group is
/// drawn or left whole, never split.
pub fn stratified<T>(
    items: Vec<T>,
    wanted: usize,
    seed: u64,
    stratum: impl Fn(&T) -> String,
    key: impl Fn(&T) -> String,
) -> Vec<T> {
    assert!(
        wanted > 0 && wanted <= items.len(),
        "a sample of {wanted} from {} items",
        items.len()
    );
    let mut strata: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (at, item) in items.iter().enumerate() {
        strata.entry(stratum(item)).or_default().push(at);
    }

    let sizes: Vec<usize> = strata.values().map(Vec::len).collect();
    let ties: Vec<[u8; 32]> = strata.keys().map(|name| order(seed, name)).collect();
    let mut quotas = vec![0usize; sizes.len()];
    for _ in 0..wanted {
        // `a / (qa + 1)` against `b / (qb + 1)`, cross-multiplied so the
        // comparison is exact.
        let next = (0..sizes.len())
            .filter(|at| quotas[*at] < sizes[*at])
            .max_by(|a, b| {
                (sizes[*a] * (quotas[*b] + 1))
                    .cmp(&(sizes[*b] * (quotas[*a] + 1)))
                    .then(ties[*b].cmp(&ties[*a]))
            })
            .expect("fewer wanted than there are items");
        quotas[next] += 1;
    }

    let mut kept = vec![false; items.len()];
    for (members, quota) in strata.values().zip(quotas) {
        let mut members = members.clone();
        members.sort_by_cached_key(|at| order(seed, &key(&items[*at])));
        for at in members.into_iter().take(quota) {
            kept[at] = true;
        }
    }
    items
        .into_iter()
        .zip(kept)
        .filter_map(|(item, kept)| kept.then_some(item))
        .collect()
}

/// A question the selection drew and the harness cannot score.
#[derive(Clone, serde::Serialize)]
pub struct Excluded {
    pub id: String,
    pub group: String,
    pub reason: String,
}

/// What the open shape has counted since it was published: the tables the
/// running arm said it would print, the one it is scoring into now, and how
/// many times each `(table, group, query id)` was scored.
///
/// Kept here rather than in a field a harness threads through, because the
/// places a query is scored are a harness's scoring function and the
/// callbacks a dozen diagnostic arms hand it, several layers down; each
/// harness calls [`tally`] once, where its answer key is read, and every arm
/// is covered by construction. `None` while no shape is open, so unit tests
/// that score nothing published are not counted.
///
/// Per thread rather than per process, because libtest runs tests on threads
/// of one process: two harness tests run together -- `--ignored` without
/// `--exact` -- each count into their own. A test's body is polled on its own
/// thread, `#[tokio::test]` included, and nothing here scores from a spawned
/// task; a score that ever did would go uncounted, and fail the run as never
/// scored rather than pass it.
///
/// By table because a count per query cannot see a table that never ran: an
/// arm that prints three tables and fills two scores every query twice, which
/// is as uniform as three. Each arm names its tables with [`tables`] before
/// it scores, and [`table`] says which one a score goes to.
#[derive(Default)]
struct Tally {
    declared: BTreeSet<String>,
    current: String,
    counts: BTreeMap<(String, String, String), usize>,
}

thread_local! {
    static TALLY: RefCell<Option<Tally>> = const { RefCell::new(None) };
}

/// Runs `f` on this thread's tally.
fn tallied<R>(f: impl FnOnce(&mut Option<Tally>) -> R) -> R {
    TALLY.with(|tally| f(&mut tally.borrow_mut()))
}

/// Starts counting on this thread, for a shape just published.
fn open_tally() {
    tallied(|open| {
        // Two shapes on one thread would count into one tally and fail each
        // other's check with each other's queries; say so instead.
        assert!(
            open.is_none(),
            "this test already published a shape and is counting its coverage"
        );
        *open = Some(Tally::default());
    });
}

/// `PAMIN_` variables that say only where something is, or what to log, and
/// change no result -- and the four `PAMIN_ENGINE_` variables `build.rs` sets
/// with `cargo:rustc-env`, which Cargo also sets on every test it runs: they
/// describe the build, which the record names as its source and build.
/// Every other `PAMIN_` variable is taken to change what is measured.
const NOT_SETTINGS: [&str; 15] = [
    "PAMIN_EMBEDDING_TEST_OUTPUT",
    "PAMIN_ENGINE_BUILD_ENVIRONMENT",
    "PAMIN_ENGINE_COMMIT",
    "PAMIN_ENGINE_DIRTY",
    "PAMIN_ENGINE_SOURCES",
    "PAMIN_EVAL_HOME",
    "PAMIN_HOME",
    "PAMIN_LOG",
    "PAMIN_MATRIX_OUTPUT",
    "PAMIN_NATIVE_TEST_CACHE",
    "PAMIN_POSTGRES_DIR",
    "PAMIN_PREPARED_CACHE",
    "PAMIN_RERANK_TEST_OUTPUT",
    "PAMIN_SHARED_CACHE",
    "PAMIN_TEST_MODEL_CACHE",
];

/// Variables outside `PAMIN_` that every harness's record is named by,
/// whichever harness it is. `SIGNIFICANCE_GATE`, which `statistics::gate`
/// reads, decides whether a run's significance gate was applied at all.
/// `HF_ENDPOINT` is where the product downloads its models from. The rest
/// set how many threads the runtime, the tokenizer and the model's kernels
/// use: they change timings, and a reduction split over a different number
/// of threads can round differently. With them, every variable starting
/// with one of [`ALWAYS_PREFIXES`], and the CUDA device selection
/// ([`pamin_index::CUDA_SELECTION`], the product's own list, which names its
/// CUDA plans).
///
/// `every_environment_read_is_classified` holds the product to this: each
/// variable its code reads is here, under a prefix, or in
/// [`NOT_IDENTITY`] with the reason it changes no result.
const ALWAYS_SETTINGS: [&str; 6] = [
    "SIGNIFICANCE_GATE",
    "HF_ENDPOINT",
    "TOKIO_WORKER_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
];

/// Families of variables the native libraries read at run time: ONNX
/// Runtime's, the OpenMP, MKL and OpenBLAS thread pools', and the ROCm
/// device selection (`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`).
const ALWAYS_PREFIXES: [&str; 7] = ["ORT_", "OMP_", "MKL_", "KMP_", "OPENBLAS_", "HIP_", "ROCR_"];

/// Variables the product reads that change no result or timing, with why.
/// Anything else it reads names the record.
const NOT_IDENTITY: [(&str, &str); 6] = [
    ("HF_HOME", "where downloaded models are cached, not which"),
    ("LOCALAPPDATA", "where the WinML runtime is unpacked"),
    ("OUT_DIR", "a build directory, compiled in"),
    (
        "COMPUTERNAME",
        "the host's name, for a plan cache's key; the hardware is recorded",
    ),
    (
        "PROCESSOR_IDENTIFIER",
        "the CPU, recorded as the hardware's cpu",
    ),
    ("HOSTNAME", "who holds a cascade claim in a shared store"),
];

/// Variables outside `PAMIN_` the evaluation tests read that choose what a
/// run measures: an arm's knobs, a sample's size, a replay's input. Every one
/// that is set names the record, whichever harness reads it, so no harness
/// keeps a list of its own that can miss one.
/// `every_evaluation_read_is_classified` holds the tests to this.
const EVAL_SETTINGS: [&str; 30] = [
    "AT_LIMIT",
    "CALIBRATE",
    "CHANNELS",
    "COLD",
    "COLD_ROUNDS",
    "CONTEXT",
    "DEPTH_VARIANTS",
    "EFFORTS",
    "ENTITIES",
    "FEATURES_OUT",
    "GATE",
    "GRAPHS",
    "LABELS",
    "LOCOMO_CONVERSATIONS",
    "MEMORY",
    "MIRACL_MAX_DOCS",
    "MUSIQUE_QUESTIONS",
    "MUSIQUE_SAMPLE",
    "PAIRS",
    "PASSAGES",
    "REACH",
    "RERANK_RULES",
    "RESCORE_IN",
    "RESHAPE",
    "ROUTES",
    "SUFFICIENCY_OUT",
    "SWEEP",
    "SWEEP_AGAINST",
    "TIERS",
    "XQUAD_ALL_QUERIES",
];

/// Variables the evaluation tests read that change nothing a record is
/// named by, with why.
const EVAL_NOT_IDENTITY: [(&str, &str); 7] = [
    ("CHANNELS_OUT", "where a table is written"),
    (
        "SHORTLIST_OUT",
        "where the shortlists a reranker saw are written",
    ),
    (
        "ESCI_DIR",
        "where a dataset is; what it holds is the corpus digest",
    ),
    (
        "LAREQA_DIR",
        "where a dataset is; what it holds is the corpus digest",
    ),
    (
        "LOCOMO_DIR",
        "where a dataset is; what it holds is the corpus digest",
    ),
    (
        "MIRACL_DIR",
        "where a dataset is; what it holds is the corpus digest",
    ),
    (
        "MUSIQUE_DIR",
        "where a dataset is; what it holds is the corpus digest",
    ),
];

/// Environment reads in the evaluation tests whose name is not a literal at
/// the read, with where the names come from.
const EVAL_INDIRECT: [(&str, &str); 9] = [
    (
        "pamin-engine/tests/harness/mod.rs: std::env::var(var)",
        "dataset_dir's variable, a literal at every dataset_dir call",
    ),
    (
        "pamin-engine/tests/harness/mod.rs: std::env::var_os(name)",
        "KnobSweep's knob, a literal at every KnobSweep::claim call, checked unset",
    ),
    (
        "pamin-engine/tests/locomo.rs: std::env::var_os(name)",
        "PAMIN_RERANK_DEPTH, PAMIN_RERANK_MAX_TOKENS and PAMIN_SEARCH_EFFORT, asserted unset",
    ),
    (
        "pamin-index/tests/prepared.rs: std::env::var(ARM)",
        "PAMIN_PREPARED_ARM, a measuring child's arm",
    ),
    (
        "pamin-index/tests/prepared.rs: std::env::var(CACHE)",
        "PAMIN_PREPARED_CACHE, a measuring child's cache",
    ),
    (
        "pamin-index/tests/shared_vocabulary.rs: std::env::var(ARM)",
        "PAMIN_SHARED_ARM, a measuring child's arm",
    ),
    (
        "pamin-index/tests/shared_vocabulary.rs: std::env::var(CACHE)",
        "PAMIN_SHARED_CACHE, a measuring child's cache",
    ),
    (
        "pamin-index/tests/shared_vocabulary.rs: std::env::var(name)",
        "PAMIN_SHARED_*_COPY, the prepared copies a child loads",
    ),
    (
        "pamin-index/tests/shared_vocabulary.rs: std::env::var_os(tuned)",
        "PAMIN_RERANK_* batch limits, asserted unset",
    ),
];

fn identity_variable(name: &str) -> bool {
    ALWAYS_SETTINGS.contains(&name)
        || EVAL_SETTINGS.contains(&name)
        || pamin_index::CUDA_SELECTION.contains(&name)
        || ALWAYS_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

/// What makes this run an arm other than the shipped one, as sorted
/// `NAME=value` lines: each set variable [`ALWAYS_SETTINGS`] or
/// [`EVAL_SETTINGS`] covers, and every set `PAMIN_` variable but
/// [`NOT_SETTINGS`].
///
/// The `PAMIN_` variables are found rather than listed. A list was kept per
/// harness, and the product's runtime knobs -- `PAMIN_RERANK_DEPTH`,
/// `PAMIN_SEARCH_EFFORT` -- were never on it, so a run with one set was
/// recorded as the shipped run, over the shipped run's record. The
/// harnesses' own knobs were listed per harness as well, and MuSiQue's list
/// left out `RERANK_RULES`, so they are one shared list now.
fn settings(set: impl IntoIterator<Item = (String, String)>) -> Vec<String> {
    let mut settings: Vec<String> = set
        .into_iter()
        .filter(|(name, _)| {
            identity_variable(name)
                || (name.starts_with("PAMIN_") && !NOT_SETTINGS.contains(&name.as_str()))
        })
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    settings.sort();
    settings
}

/// This process's environment, lossily as text, for [`settings`].
fn environment() -> impl Iterator<Item = (String, String)> {
    std::env::vars_os().map(|(name, value)| {
        (
            name.to_string_lossy().into_owned(),
            value.to_string_lossy().into_owned(),
        )
    })
}

/// The `PAMIN_` variables the product reads on the path from a text to its
/// vector and from the vectors to an index: those that change how a model
/// executes ([`pamin_index::EXECUTION_SETTINGS`], which its plan keys are
/// named by too) and those that change what a drain builds. A vector
/// cache's producer and an index's stamp are named by these and by
/// [`NATIVE_SETTINGS`], and by nothing else a run sets.
///
/// An allowlist, so a knob added elsewhere -- the reranker's, the search's
/// -- cannot discard a cache or unvouch an index whose vectors it never
/// touched. `every_product_pamin_read_is_placed` holds the product to it:
/// every `PAMIN_` variable the product reads is here or in
/// [`NOT_PRODUCER`] with why.
fn producer_pamin(name: &str) -> bool {
    pamin_index::EXECUTION_SETTINGS.contains(&name) || INDEX_SETTINGS.contains(&name)
}

/// The `PAMIN_` variables that change what an index build makes of the
/// vectors: when a drain builds the vector graph.
const INDEX_SETTINGS: [&str; 1] = ["PAMIN_UNINDEXED_BUDGET"];

/// The `PAMIN_` variables the product reads off that path, with why: they
/// name a run's record (see [`settings`]), not its vectors.
const NOT_PRODUCER: [(&str, &str); 11] = [
    ("PAMIN_HOME", "where a workspace is"),
    (
        "PAMIN_INFERENCE_BUILD",
        "compiled in, and named by the producer's build line",
    ),
    (
        "PAMIN_INFERENCE_CARGO",
        "compiled in, and named by the producer's build line",
    ),
    ("PAMIN_JIT", "how Postgres plans the store's queries"),
    ("PAMIN_MODEL_IDLE", "when an idle model is released"),
    ("PAMIN_POSTGRES_DIR", "where the Postgres binaries are"),
    ("PAMIN_RERANK_BATCH", "the reranker's batches"),
    ("PAMIN_RERANK_BATCH_TOKENS", "the reranker's batches"),
    (
        "PAMIN_RERANK_DEPTH",
        "how many candidates the reranker sees",
    ),
    ("PAMIN_RERANK_MAX_TOKENS", "the reranker's input length"),
    (
        "PAMIN_SEARCH_EFFORT",
        "how hard a query searches the vector graph",
    ),
];

/// The variables outside `PAMIN_` that change what a model computes: the
/// native libraries' families ([`ALWAYS_PREFIXES`]), the CUDA device
/// selection, and the thread counts of the tokenizer and the kernels, as a
/// reduction split another way can round another way.
const NATIVE_SETTINGS: [&str; 3] = [
    "OMP_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
];

/// The [`settings`] set now that can change what a text embeds to or what
/// an index of the vectors holds: only those [`producer_setting`] allows.
/// A cache of model output, and an index's stamp, are named by these;
/// every other setting -- the harnesses' knobs, the significance gate, the
/// reranker's and the search's -- names the record only.
pub fn runtime_settings() -> Vec<String> {
    runtime(settings(environment()))
}

fn runtime(settings: Vec<String>) -> Vec<String> {
    settings
        .into_iter()
        .filter(|line| {
            producer_setting(line.split_once('=').map_or(line.as_str(), |(name, _)| name))
        })
        .collect()
}

/// Whether the variable `name` can change what an embedding producer makes.
fn producer_setting(name: &str) -> bool {
    producer_pamin(name)
        || NATIVE_SETTINGS.contains(&name)
        || pamin_index::CUDA_SELECTION.contains(&name)
        || ALWAYS_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

/// The arm's name in a record's file name: `shipped` when nothing is set,
/// else a short hash of [`settings`], which can hold paths and run long. The
/// settings themselves are printed and recorded in full.
fn arm(settings: &[String]) -> String {
    if settings.is_empty() {
        return "shipped".to_string();
    }
    let hash = Sha256::digest(settings.join("\n").as_bytes());
    let short: String = hash[..6].iter().map(|byte| format!("{byte:02x}")).collect();
    format!("arm-{short}")
}

/// What a record can say about the code that ran.
///
/// The commit is the one the binary was compiled from (`PAMIN_ENGINE_COMMIT`,
/// which `build.rs` sets), not whatever is checked out when it runs: a binary
/// built at one commit and run after the checkout moved would otherwise name
/// code it does not contain. Only a run that can vouch for that commit writes
/// the canonical record; see [`provenance`].
#[derive(Debug, PartialEq)]
struct Source {
    /// The compiled commit, or `unknown` for a build outside a git checkout.
    commit: String,
    /// Why this run cannot vouch for `commit`, if it cannot: its record is
    /// then written beside the canonical one, never over it.
    noncanonical: Option<String>,
}

/// `source`, not canonical when `settings` turn the significance gate off: a
/// run that skipped its gate cannot stand for one that passed it, however
/// clean its checkout.
fn ungated(mut source: Source, settings: &[String]) -> Source {
    if settings
        .iter()
        .any(|setting| setting == "SIGNIFICANCE_GATE=off")
    {
        const OFF: &str = "significance gate off";
        source.noncanonical = Some(match source.noncanonical {
            None => OFF.to_string(),
            Some(reason) => format!("{reason}; {OFF}"),
        });
    }
    source
}

/// [`provenance`] of this binary and the checkout it runs in, asking `git`
/// without failing the run when there is no `git` or no checkout.
fn source() -> Source {
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(["-C", env!("CARGO_MANIFEST_DIR")])
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
    };
    let head = git(&["rev-parse", "HEAD"])
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|head| head.trim().to_string());
    let tracked = git(&["diff", "--quiet", "HEAD"])
        .map(|_| false)
        .or_else(|| {
            // `--quiet` exits 1 for changes, which `git` above reads as a
            // failure; ask again for the list to tell the two apart.
            git(&["diff", "--name-only", "HEAD"]).map(|output| !output.stdout.is_empty())
        });
    // Untracked and ignored files the build compiles in, as the build script
    // counts them.
    let args = super::compiled::status_args(env!("PAMIN_ENGINE_SOURCES"));
    let untracked = git(&args.iter().map(String::as_str).collect::<Vec<_>>())
        .map(|output| super::compiled::compiled_in(&String::from_utf8_lossy(&output.stdout)));
    let changed = match (tracked, untracked) {
        (Some(false), Some(false)) => Some(false),
        (Some(true), _) | (_, Some(true)) => Some(true),
        _ => None,
    };
    // Anything but an explicit "false" counts as dirty, which errs towards
    // a noncanonical record.
    let built_dirty = env!("PAMIN_ENGINE_DIRTY") != "false";
    provenance(
        env!("PAMIN_ENGINE_COMMIT"),
        built_dirty,
        head.as_deref(),
        changed,
    )
}

/// Whether a run compiled at `built`, from a checkout that was `built_dirty`
/// or not, may write the canonical record, given the checkout's `head` and
/// whether it has `changed` files a build compiles in (`None` where `git`
/// could not say): tracked changes anywhere, and untracked or ignored files
/// under the build script's sources.
///
/// Only a clean build run from a clean checkout at the compiled commit can:
/// edits compiled in may since have been restored, changes at run time may
/// or may not be what was compiled, and a build with no commit, or
/// a checkout `git` cannot read, names nothing a reader could rebuild.
fn provenance(built: &str, built_dirty: bool, head: Option<&str>, changed: Option<bool>) -> Source {
    let noncanonical = if built == "unknown" {
        Some("compiled outside a git checkout, so no commit names it".to_string())
    } else if built_dirty {
        Some(
            "compiled from a checkout with changes a build compiles in, which no commit names"
                .to_string(),
        )
    } else if head.is_none() {
        Some("git could not read the checkout it runs in".to_string())
    } else if head != Some(built) {
        Some(format!(
            "the checkout is at {}, not the compiled commit",
            head.unwrap_or_default()
        ))
    } else {
        match changed {
            Some(false) => None,
            Some(true) => Some(
                "the checkout has changes a build compiles in, which may not be the compiled ones"
                    .to_string(),
            ),
            None => Some("git could not list the checkout's changes".to_string()),
        }
    };
    Source {
        commit: built.to_string(),
        noncanonical,
    }
}

/// A commit in a record's name: its first 12 characters.
fn short(commit: &str) -> &str {
    &commit[..commit.len().min(12)]
}

/// Prints `line` to stdout, ignoring a failure: `println!` panics on a
/// closed stdout, which a record must not be lost to, nor an unwind turned
/// into an abort by.
fn report(line: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stdout(), "{line}");
}

/// A path beside `path` that no other writer uses: named by this process and
/// the time, then `suffix`.
fn unique(path: &Path, suffix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    path.with_extension(format!("{}.{nanos}.{suffix}", std::process::id()))
}

/// Where one run keeps its record until its coverage holds: beside the
/// record, so a run that fails never replaces the last one that completed,
/// and its own, so two runs of one record never share or remove each other's.
fn in_progress(record: &Path) -> PathBuf {
    unique(record, "inprogress.json")
}

/// Whether anything at all was scored into `table`.
fn filled_any(tally: &Tally, table: &str) -> bool {
    tally.counts.keys().any(|(into, ..)| into == table)
}

/// One part of a record's file name, with anything a file system might
/// object to replaced.
fn file_part(part: &str) -> String {
    part.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "=+,._".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Declares tables the running arm will score every query into, before it
/// scores any. A declared table nothing is scored into fails the run, and so
/// does a score into a table never declared.
pub fn tables<S: ToString>(names: impl IntoIterator<Item = S>) {
    tallied(|open| {
        if let Some(tally) = open.as_mut() {
            tally
                .declared
                .extend(names.into_iter().map(|name| name.to_string()));
        }
    });
}

/// Names the table the scores that follow are counted toward.
pub fn table(name: impl ToString) {
    tallied(|open| {
        if let Some(tally) = open.as_mut() {
            tally.current = name.to_string();
        }
    });
}

/// Records that query `id` was scored once in `group`, in the current
/// [`table`]. Called where a harness reads its answer key, so a query graded
/// is a query counted.
pub fn tally(group: &str, id: &str) {
    tallied(|open| {
        if let Some(tally) = open.as_mut() {
            let key = (tally.current.clone(), group.to_string(), id.to_string());
            *tally.counts.entry(key).or_default() += 1;
        }
    });
}

/// One run's question and document set, how it was chosen, and its coverage.
///
/// Coverage is checked when the shape is dropped -- at the end of the test,
/// whichever arm ran and wherever it returned. The first version checked in
/// the shipped arm's reporting and nowhere else, so every diagnostic arm
/// returned before the check and was believed unchecked.
pub struct Shape {
    name: String,
    selection: String,
    seed: Option<u64>,
    queries: Vec<[u8; 32]>,
    documents: Vec<[u8; 32]>,
    /// Per group, the id of every query the selection drew, excluded or not.
    asked: BTreeMap<String, BTreeSet<String>>,
    excluded: Vec<Excluded>,
    /// What the check found, recorded beside the rest once it has run.
    coverage: Option<serde_json::Value>,
    /// Where [`publish`](Self::publish) will write the record, without its
    /// extension or the execution providers, which are known only once the
    /// models have loaded (see [`ran_on`](Self::ran_on)); set only on a
    /// published shape, which is also the one whose coverage is checked.
    record: Option<PathBuf>,
    /// This run's [`in_progress`] file, set beside `record`.
    progress: PathBuf,
    /// Which test, profile and arm the run was, as `publish` was told.
    run: [String; 3],
    /// The compiled commit, and whether this run may write the canonical
    /// record, from [`source`], once published.
    source: Source,
    /// What set the arm apart, from [`settings`].
    settings: Vec<String>,
    /// Asks the run which execution provider each of its models resolved
    /// to, as `role=provider` lines; see [`ran_on`](Self::ran_on).
    probe: Option<Probe>,
    /// Models the harness loaded itself, beside the engine's, from
    /// [`also_ran`](Self::also_ran): each asked for its plans when the
    /// providers are resolved, as `probe` is.
    loaded: Vec<Probe>,
    /// What `probe` answered, with `loaded`, once the coverage is checked.
    providers: Vec<String>,
    /// The model each of those roles ran, `role=repository@commit`, read
    /// with them.
    models: Vec<String>,
    /// Vectors the run read from a cache rather than embedding, from
    /// [`cached`](Self::cached).
    caches: Vec<String>,
    /// What model loads in this process had fetched and prepared when
    /// [`observe_models`](Self::observe_models) first looked, before the
    /// models loaded.
    activity_before: Option<pamin_index::LoadActivity>,
    /// Reads those counts: `LoadActivity::now`, or a test's own.
    activity: fn() -> pamin_index::LoadActivity,
    /// This process's turn at loading models, from that first look until
    /// the record is written, so no other test's loads fall in between
    /// (see `harness::ModelsTurn`).
    turn: Option<super::ModelsTurn>,
    /// Per role, where the load found its plan -- calibrated by the run,
    /// before it, or none to calibrate -- read with the providers.
    calibration: Vec<String>,
    /// The search indexes the run scored, from [`indexed`](Self::indexed),
    /// each checked against who embedded it once the providers are known.
    indexes: Vec<Indexed>,
    /// What those checks found, `label: state by producer`.
    indexed: Vec<String>,
    /// The machine, read at `publish`; accelerators once `providers` are.
    hardware: Hardware,
}

/// The stamp of a build begun by `producer` and not yet finished.
fn pending(producer: &str) -> String {
    format!("pending\n{producer}")
}

/// Replaces the stamp at `path` with `stamp` whole, or says why it could
/// not, leaving what was there.
fn write_stamp(path: &Path, stamp: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(path.parent().expect("a parent"))?;
    let partial = path.with_extension("partial");
    let written = std::fs::write(&partial, stamp).and_then(|()| std::fs::rename(&partial, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
}

/// Where the load of `role` found `plan`, as the selection says
/// (`pamin_index::PlanOrigin`): whether the run paid for a calibration.
fn calibration(role: &str, plan: &pamin_index::PlanUse) -> String {
    use pamin_index::PlanOrigin;
    let provider = provider(plan);
    match plan.origin {
        PlanOrigin::CalibratedNow => format!("calibrated {role}={provider}: by the run"),
        PlanOrigin::PersistedRecord => {
            format!("calibrated {role}={provider}: before the run, from its plan file")
        }
        PlanOrigin::ProcessCache => {
            format!("calibrated {role}={provider}: before the run, in this process")
        }
        PlanOrigin::CpuOnly => format!("cpu-only {role}={provider}: no calibration"),
        PlanOrigin::Deferred => format!("deferred {role}={provider}: calibration deferred"),
        PlanOrigin::Unplanned => format!("unplanned {role}={provider}: no calibration"),
    }
}

/// A search index a run scored, as [`Shape::indexed`] registered it.
struct Indexed {
    label: String,
    /// The run's embedding producer now (see `harness::vectors`), or `None`
    /// while its embedder has not loaded.
    producer: Box<dyn Fn() -> Option<String> + Send>,
    /// Where the producer that built the index is kept.
    stamp: PathBuf,
    /// Documents the index held before this run wrote any.
    found: usize,
    total: usize,
    /// The stamp this run's producer writes, as it was before the run wrote
    /// anything; set by [`Shape::begin_index`].
    start: Option<String>,
    /// This index's build lock, held from counting what it holds until the
    /// run is over (see [`Shape::claim_index`]).
    building: Option<std::fs::File>,
    /// Counts the documents the index holds now: asked again before a
    /// build's stamp is promoted.
    count: Box<dyn Fn() -> usize + Send>,
}

/// Everything that tells one run's record from another's, serialised
/// canonically into the record and hashed into its file name.
///
/// The file name was built from hand-picked parts, and every review found one
/// missing -- the providers' roles, then the build profile. Now the readable
/// prefix is only for people, and the name's hash covers this whole struct,
/// so a field added here is in the name without anyone remembering to put it
/// there. `every_identity_field_names_the_record` fails for a field that
/// does not change the hash.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
struct Identity {
    harness: String,
    test: String,
    profile: String,
    arm: String,
    /// The arm's settings in full, `SIGNIFICANCE_GATE` included.
    settings: Vec<String>,
    commit: String,
    /// Why the run cannot vouch for its commit -- a dirty build or
    /// checkout, a moved checkout, the gate off -- if it cannot.
    noncanonical_reason: Option<String>,
    corpus_sha256: String,
    sequence_sha256: String,
    selection: String,
    seed: Option<u64>,
    /// `engine/role=provider`: which engine, which model, where it ran.
    providers: Vec<String>,
    /// `engine/role=repository@commit`: the revision each of those models
    /// was read at, the commit the model cache's ref resolved to when it
    /// loaded. XQuAD-R pins its models by a manifest the commit names;
    /// without this, a run after a moved `refs/main` would take another
    /// revision's weights under the same record name.
    models: Vec<String>,
    /// `file: reused/total from producer`: vectors a run read from a cache
    /// rather than embedding, and the digest of who embedded them (see
    /// `harness::vectors`). A run that embedded everything itself says so
    /// too, with nothing reused.
    caches: Vec<String>,
    /// Whether the model loads fetched or prepared anything, and how much,
    /// and per role whether its plan was calibrated before the run: a run
    /// that downloads, prepares or calibrates times that too, and a rerun
    /// over a warm home is another condition.
    warmth: Vec<String>,
    /// `label: built|resumed|reused by producer`: each search index the run
    /// scored, and the digest of the embedding producer it was built by --
    /// a reused index's document vectors are not the current plan's merely
    /// because its queries are.
    indexes: Vec<String>,
    build: Build,
    hardware: Hardware,
}

/// The machine a run ran on, as far as it can be read cheaply.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
struct Hardware {
    /// The CPU's model name, or `unknown`.
    cpu: String,
    /// Logical cores this process may use.
    logical_cores: usize,
    /// Accelerators a provider ran on, by name where the platform says.
    /// When a model ran on CUDA, the GPUs visible to it -- index, UUID, PCI
    /// bus, name and driver, the inventory the product's CUDA plans are
    /// named by ([`pamin_index::cuda_devices`]) -- narrowed to the ones
    /// `CUDA_VISIBLE_DEVICES` selects where that can be read off. CoreML
    /// runs on the Apple chip `cpu` names; DirectML is named only as itself.
    accelerators: Vec<String>,
}

impl Hardware {
    /// The CPU and core count; accelerators are added once the providers
    /// are known, by [`Hardware::accelerators_for`].
    fn detect() -> Self {
        Self {
            cpu: cpu_model().unwrap_or_else(|| "unknown".into()),
            logical_cores: std::thread::available_parallelism().map_or(0, usize::from),
            accelerators: Vec::new(),
        }
    }

    /// The accelerators behind `providers` (`engine/role=provider` lines).
    ///
    /// With `probe` false -- on a failed run's unwind, which must not wait
    /// on `nvidia-smi` before its coverage is on disk -- the CUDA inventory
    /// is taken only from an earlier probe in this process, else the line
    /// says it was not probed.
    fn accelerators_for(providers: &[String], probe: bool) -> Vec<String> {
        let ran = |name: &str| {
            providers.iter().any(|line| {
                line.rsplit_once('=').is_some_and(|(_, on)| {
                    let on = on.split('@').next().unwrap_or(on);
                    on == name
                        || on
                            .strip_prefix(name)
                            .is_some_and(|key| key.starts_with(':'))
                })
            })
        };
        let mut accelerators = Vec::new();
        if ran("cuda") {
            static INVENTORY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
            let inventory = if probe {
                Some(INVENTORY.get_or_init(pamin_index::cuda_devices))
            } else {
                INVENTORY.get()
            };
            match inventory {
                Some(inventory) => {
                    let read = |name: &str| std::env::var(name).ok();
                    accelerators.extend(visible_cuda(
                        inventory.as_deref(),
                        read("CUDA_VISIBLE_DEVICES").as_deref(),
                        read("CUDA_DEVICE_ORDER").as_deref(),
                    ));
                }
                None => accelerators.push("cuda: not probed (failed run)".to_string()),
            }
        }
        for other in ["coreml", "directml", "npu"] {
            if ran(other) {
                accelerators.push(other.to_string());
            }
        }
        accelerators
    }
}

/// The GPUs of `inventory` (`nvidia-smi` lines: index, UUID, PCI bus, name,
/// driver) that `visible` -- `CUDA_VISIBLE_DEVICES` -- selects, in its order,
/// as `cuda: ` lines. An entry naming a UUID, or a prefix of one, picks that
/// GPU. An index picks the GPU `nvidia-smi` numbers so only under
/// `CUDA_DEVICE_ORDER=PCI_BUS_ID`, the order `nvidia-smi` numbers in; under
/// CUDA's default, fastest first, an index is not read off here, and the
/// whole inventory is recorded as unresolved, beside the selection itself
/// in the settings. (`NVIDIA_VISIBLE_DEVICES` is applied by the container,
/// before `nvidia-smi` sees any GPU.)
fn visible_cuda(
    inventory: Option<&str>,
    visible: Option<&str>,
    order: Option<&str>,
) -> Vec<String> {
    let Some(inventory) = inventory else {
        return vec!["cuda: unknown".to_string()];
    };
    let lines: Vec<&str> = inventory.lines().map(str::trim).collect();
    let field = |line: &str, at: usize| line.split(',').nth(at).map(str::trim).map(str::to_string);
    let Some(visible) = visible else {
        return lines.iter().map(|line| format!("cuda: {line}")).collect();
    };
    let by_bus = order == Some("PCI_BUS_ID");
    let picked: Option<Vec<String>> = visible
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            lines
                .iter()
                .find(|line| {
                    if entry.starts_with("GPU-") {
                        field(line, 1).is_some_and(|uuid| uuid.starts_with(entry))
                    } else {
                        by_bus && field(line, 0).is_some_and(|index| index == entry)
                    }
                })
                .map(|line| format!("cuda: {line}"))
        })
        .collect();
    picked.unwrap_or_else(|| {
        lines
            .iter()
            .map(|line| format!("cuda (selection unresolved): {line}"))
            .collect()
    })
}

/// The CPU's model name: `/proc/cpuinfo` on Linux, `sysctl` on macOS, the
/// environment on Windows.
fn cpu_model() -> Option<String> {
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        return info
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                matches!(key.trim(), "model name" | "Hardware" | "Model")
                    .then(|| value.trim().to_string())
            })
            .filter(|name| !name.is_empty());
    }
    if cfg!(target_os = "macos") {
        return std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .filter(|name| !name.is_empty());
    }
    std::env::var("PROCESSOR_IDENTIFIER").ok()
}

/// What the harness binary was compiled as.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
struct Build {
    /// pamin-index's build fingerprint, verbatim
    /// ([`pamin_index::BUILD_FINGERPRINT`]): its sources, the workspace
    /// manifests and Cargo config, target, profile, opt-level, debug
    /// setting, rustflags, target features, Cargo profile overrides and
    /// rustc's version. Whatever pamin-index adds to it names records too.
    inference: String,
    /// The same build, readable: each Cargo profile override set and each
    /// target `cfg` (`CARGO_CFG_*`, the target features among them), as
    /// `NAME=value`.
    cargo: Vec<String>,
    /// The `ORT_*`, `ZVEC_*` and rustc-wrapper build variables that were
    /// set, as `NAME=value`: what chooses the native libraries linked in,
    /// which the fingerprint does not hold.
    environment: Vec<String>,
}

/// The SHA-256 of the build a record names ([`Build`]), for a cache to name
/// the build that wrote it.
pub fn build_digest() -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&Build::compiled()).expect("serialise the build"))
    )
}

impl Build {
    fn compiled() -> Self {
        let lines = |joined: &str| -> Vec<String> {
            joined
                .split('\x1f')
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        };
        Self {
            inference: pamin_index::BUILD_FINGERPRINT.to_string(),
            cargo: lines(pamin_index::BUILD_CARGO),
            environment: lines(env!("PAMIN_ENGINE_BUILD_ENVIRONMENT")),
        }
    }
}

impl Identity {
    /// The identity's canonical form: its fields in declaration order.
    fn canonical(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("serialise the identity")
    }

    /// The first twelve hex digits of the canonical form's sha256.
    fn short(&self) -> String {
        Sha256::digest(self.canonical())[..6]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// A probe for [`Shape::ran_on`] over every engine a run searched, each
/// under its label (`primary`, `second`, ...): each one's
/// [`Engine::providers`](pamin_engine::Engine::providers) as
/// `label/role`, kept apart, sorted, once each engine's outstanding
/// loads have finished ([`Engine::settle_loads`](pamin_engine::Engine::settle_loads)).
/// It never panics, and waits only for those loads.
pub fn engines(
    engines: &[(&str, &pamin_engine::Engine)],
) -> impl Fn() -> Vec<(String, pamin_index::PlanUse)> + Send + 'static {
    let engines: Vec<(String, pamin_engine::Engine)> = engines
        .iter()
        .map(|(label, engine)| (label.to_string(), (*engine).clone()))
        .collect();
    move || {
        // A load a search started and returned before needing -- a reranked
        // search that found nothing -- may still be running. It is waited
        // for here, before the plans are read and before the record reads
        // what the loads fetched, so it is neither missed nor left running
        // into the next run's turn. Only the harness waits: a search never
        // does.
        for (_, engine) in &engines {
            futures::executor::block_on(engine.settle_loads());
        }
        let mut providers: Vec<(String, pamin_index::PlanUse)> = engines
            .iter()
            .flat_map(|(label, engine)| {
                engine
                    .providers()
                    .into_iter()
                    .map(move |(role, plan)| (format!("{label}/{role}"), plan))
            })
            .collect();
        providers.sort();
        providers
    }
}

/// One plan a model ran on, as a record names it: its target, the first
/// sixteen hex digits of its plan key's hash, and `!unstable` for a plan
/// that does not name its hardware stably across processes.
pub fn provider(plan: &pamin_index::PlanUse) -> String {
    format!(
        "{}@{}{}",
        plan.target,
        plan.plan.get(..16).unwrap_or(&plan.plan),
        if plan.stable { "" } else { UNSTABLE }
    )
}

const UNSTABLE: &str = "!unstable";

/// Why a run's plans keep its record from being canonical, if they do: a
/// plan whose hardware is not named stably across processes (a DirectML
/// adapter ordinal, an NPU id -- the plans pamin-index will not persist),
/// or a model whose plan changed during the run, as a fallback does.
fn plan_reasons(providers: &[(String, String)]) -> Vec<String> {
    let mut reasons = Vec::new();
    let unstable: BTreeSet<&str> = providers
        .iter()
        .filter_map(|(_, provider)| provider.strip_suffix(UNSTABLE))
        .map(|provider| provider.split(['@', ':']).next().unwrap_or(provider))
        .collect();
    if !unstable.is_empty() {
        let names: Vec<&str> = unstable.into_iter().collect();
        reasons.push(format!(
            "unstable accelerator identity ({})",
            names.join(", ")
        ));
    }
    let mut roles: BTreeMap<&str, usize> = BTreeMap::new();
    for (role, _) in providers {
        *roles.entry(role.as_str()).or_default() += 1;
    }
    if roles.values().any(|plans| *plans > 1) {
        reasons.push("fallback during run".to_string());
    }
    reasons
}

/// How a shape asks its run which execution plans, and which model
/// revisions, it ran on.
type Probe = Box<dyn Fn() -> Vec<(String, pamin_index::PlanUse)> + Send>;

/// A model the harness loaded and uses itself, shared with the record so
/// that [`Shape::also_ran`] can read its plans once the run is over.
pub type Owned<M> = std::sync::Arc<std::sync::Mutex<M>>;

/// `model`, held so that a record can read its plans later.
pub fn owned<M>(model: M) -> Owned<M> {
    std::sync::Arc::new(std::sync::Mutex::new(model))
}

/// The model, for the harness to use. A run that panicked while holding it
/// left nothing half-written that a plan list depends on, so a poisoned
/// lock is taken as it is.
pub fn lock<M>(model: &Owned<M>) -> std::sync::MutexGuard<'_, M> {
    model
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// One record's digest: its fields, separated so that no two field lists
/// share a byte string.
fn record(kind: &str, fields: &[&str]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(kind.as_bytes());
    for field in fields {
        hash.update([0x1f]);
        hash.update((field.len() as u64).to_be_bytes());
        hash.update(field.as_bytes());
    }
    hash.finalize().into()
}

/// The groups a query is asked in, sorted and joined, as a record field:
/// a query moved to another group is scored and counted there, so it is
/// another question set.
fn assigned(groups: &[&str]) -> String {
    let mut sorted = groups.to_vec();
    sorted.sort_unstable();
    sorted.join("\u{1f}")
}

/// At most this many ids are named in a coverage failure; the count says how
/// many there were.
const NAMED: usize = 5;

fn named(ids: &[&str]) -> String {
    let shown: Vec<&str> = ids.iter().take(NAMED).copied().collect();
    format!(
        "{} ({}{})",
        ids.len(),
        shown.join(", "),
        if ids.len() > NAMED { ", ..." } else { "" }
    )
}

impl Shape {
    /// `selection` says how the questions were chosen, in words a log reader
    /// can check; `seed` is the seed it used, if it used one.
    pub fn new(name: &str, selection: String, seed: Option<u64>) -> Self {
        Self {
            name: name.to_string(),
            selection,
            seed,
            queries: Vec::new(),
            documents: Vec::new(),
            asked: BTreeMap::new(),
            excluded: Vec::new(),
            coverage: None,
            record: None,
            progress: PathBuf::new(),
            run: Default::default(),
            source: Source {
                commit: String::new(),
                noncanonical: None,
            },
            settings: Vec::new(),
            probe: None,
            loaded: Vec::new(),
            providers: Vec::new(),
            models: Vec::new(),
            caches: Vec::new(),
            activity_before: None,
            activity: pamin_index::LoadActivity::now,
            turn: None,
            calibration: Vec::new(),
            indexes: Vec::new(),
            indexed: Vec::new(),
            hardware: Hardware::default(),
        }
    }

    /// How to ask, once the run is over, which execution plan each of its
    /// models ran on, as `(role, plan)` pairs: where it ran, and which
    /// revision of the model. The answer names the record and is kept in
    /// it: two runs that differ only in what `PAMIN_DEVICE=auto` found on
    /// the machine, or in the commit a model cache's `refs/main` resolved
    /// to, are two records, not one written over the other. Asked at the end
    /// rather than now, because the models load lazily -- a published shape
    /// comes before the first search, and on a fresh home before the first
    /// download -- and an arm loads only the tiers it uses. A run that never
    /// says fails its coverage.
    pub fn ran_on(
        &mut self,
        probe: impl Fn() -> Vec<(String, pamin_index::PlanUse)> + Send + 'static,
    ) {
        self.probe = Some(Box::new(probe));
    }

    /// A model the harness loaded itself rather than through an engine --
    /// a second reranker an arm prices against the shipped one -- as `role`,
    /// recorded with what [`ran_on`](Self::ran_on)'s probe says.
    ///
    /// `plans` is read when the providers are resolved, after the workload,
    /// as an engine's are: read at load, it missed a fallback the model took
    /// later, and the run was taken as canonical. It never waits: while the
    /// model is in use -- which only a run unwinding past a held guard
    /// could see -- the plans it had at load stand in.
    pub fn also_ran<M: Send + 'static>(
        &mut self,
        role: &str,
        model: &Owned<M>,
        plans: fn(&M) -> Vec<pamin_index::PlanUse>,
    ) {
        let role = format!("harness/{role}");
        let model = std::sync::Arc::clone(model);
        let loaded = plans(&lock(&model));
        self.loaded.push(Box::new(move || {
            let now = match model.try_lock() {
                Ok(model) => plans(&model),
                Err(std::sync::TryLockError::Poisoned(model)) => plans(&model.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => loaded.clone(),
            };
            now.into_iter().map(|plan| (role.clone(), plan)).collect()
        }));
    }

    /// That `reused` of the `total` vectors in the cache `file` were read
    /// from it rather than embedded, by `producer` (see
    /// `harness::vectors::producer`). It names the record: a run scored on
    /// cached vectors is not taken for one that embedded them.
    pub fn cached(&mut self, file: &str, producer: &str, reused: usize, total: usize) {
        let producer: String = Sha256::digest(producer.as_bytes())[..8]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        self.caches
            .push(format!("{file}: {reused}/{total} from {producer}"));
    }

    /// Observes, before any model loads, what loads in this process have
    /// fetched and prepared so far. At the end, what the loads fetched and
    /// prepared since -- tokenizer files, accelerator exports, mapped
    /// copies, rewritten graphs and all -- says whether the run paid for
    /// them in its timings, and each plan says where its load found it:
    /// the outcome of the loads, not a prediction of it.
    pub fn observe_models(&mut self) {
        if self.activity_before.is_none() {
            self.take_turn();
            self.activity_before = Some((self.activity)());
        }
    }

    /// Takes this process's turn at loading models, if this run does not
    /// hold it already (see `harness::ModelsTurn`).
    fn take_turn(&mut self) {
        if self.turn.is_none() {
            self.turn = Some(super::load_models());
        }
    }

    /// Reads the settings the run is under, holding the turn first: another
    /// test's experimental arm sets the environment only while it holds the
    /// turn, so the settings read are this run's own, not that arm's.
    fn read_settings(&mut self) {
        self.take_turn();
        self.settings = settings(environment());
    }

    /// What the model loads fetched and prepared since
    /// [`observe_models`](Self::observe_models) looked, and per role whether
    /// its execution plan was calibrated before the run or by it: a run
    /// that downloads, prepares or calibrates times that too.
    fn observed_warmth(&self) -> Vec<String> {
        let mut warmth: Vec<String> = self
            .activity_before
            .map(|before| {
                let done = (self.activity)().since(before);
                if done == pamin_index::LoadActivity::default() {
                    "models: warm: nothing fetched or prepared".to_string()
                } else {
                    format!(
                        "models: cold: fetched {} / prepared {}",
                        done.fetched, done.prepared
                    )
                }
            })
            .into_iter()
            .collect();
        warmth.extend(self.calibration.iter().cloned());
        warmth
    }

    /// The search index `engine` scores, under `label`, before this run
    /// writes to it, and how many of the corpus's `total` documents it
    /// holds, counted once this run holds its build lock.
    /// `engine`'s embedder must be loaded (`Engine::load_embedder`), so the
    /// producer that will embed what the run writes is known.
    ///
    /// The index is bound to the embedding producer that built it -- plans,
    /// model revisions, runtime settings and build, as a vector cache is
    /// named (`harness::vectors`) -- and the commit, in a stamp kept beside
    /// it under `workspace`. An empty index is stamped now, before anything
    /// is written, so a build that is interrupted resumes under its stamp.
    /// One that holds documents must carry the run's stamp, or the run is
    /// not canonical: its document vectors may come from another plan than
    /// its query vectors. Deleting the project rebuilds it. A run that is not
    /// canonical removes the stamp and writes none.
    pub fn indexed(
        &mut self,
        label: &str,
        engine: &pamin_engine::Engine,
        workspace: &Path,
        total: usize,
    ) {
        let stamp = workspace
            .join("index")
            .join(format!("{}.eval-producer", engine.project));
        let counting = engine.clone();
        let engine = engine.clone();
        self.claim_index(
            label,
            stamp,
            Box::new(move || super::vectors::index_producer(&engine)),
            Box::new(move || counting.indexed_documents().expect("count documents") as usize),
            total,
        );
    }

    /// Takes the index's build lock, counts what it holds, and begins it
    /// (see [`begin_index`](Self::begin_index)), holding the lock until this
    /// run is over.
    ///
    /// One build at a time, across processes sharing an evaluation home:
    /// two runs over one empty project each counted nothing, each stamped
    /// it pending as theirs and wrote into it together, and the one whose
    /// stamp was left vouched for documents the other embedded. Held from
    /// the count through the stamp's promotion, a second run waits and then
    /// counts what the first built.
    fn claim_index(
        &mut self,
        label: &str,
        stamp: PathBuf,
        producer: Box<dyn Fn() -> Option<String> + Send>,
        count: Box<dyn Fn() -> usize + Send>,
        total: usize,
    ) {
        // An index this run has already claimed is locked by it: a second
        // lock on another descriptor would wait on this run itself.
        let held = self
            .indexes
            .iter_mut()
            .find(|kept| kept.stamp == stamp)
            .and_then(|kept| kept.building.take());
        let building = held.unwrap_or_else(|| super::vectors::lock(&stamp));
        let found = count();
        self.begin_index(Indexed {
            label: label.to_string(),
            producer,
            stamp,
            found,
            total,
            start: None,
            building: Some(building),
            count,
        });
    }

    /// What a stamp holds for `producer`: it, and the commit -- what an
    /// index holds beyond its vectors, the mention graph and the passages,
    /// is derived by code outside the inference build the producer names.
    fn stamp_for(&self, producer: &str) -> String {
        format!("{producer}\ncommit={}", self.source.commit)
    }

    /// Marks an empty index pending before it is written; removes the stamp
    /// before a run that is not canonical writes into the index, or before
    /// writing into an index that is not this run's own pending build; and
    /// keeps the index to check.
    ///
    /// A pending stamp vouches for nothing: the embedder can fall back to
    /// another plan between the start and the end of a build, and a build
    /// interrupted after that holds vectors from both. Only
    /// [`check_indexes`](Self::check_indexes), at the end of a build that
    /// began empty under one producer and kept it, makes the stamp vouch.
    fn begin_index(&mut self, mut index: Indexed) {
        index.start = (index.producer)().map(|producer| self.stamp_for(&producer));
        if self.source.noncanonical.is_some() {
            // Only a run that writes invalidates: a complete index it just
            // reads keeps what a canonical run vouched for.
            if index.found < index.total {
                let _ = std::fs::remove_file(&index.stamp);
            }
        } else if index.found == 0
            && let Some(start) = &index.start
        {
            // Unwritten, the index is not this run's pending build, and the
            // check at its end says so; nothing more is needed here.
            if write_stamp(&index.stamp, &pending(start)).is_err() {
                let _ = std::fs::remove_file(&index.stamp);
            }
        } else if index.found < index.total
            && std::fs::read_to_string(&index.stamp).ok() != index.start.as_deref().map(pending)
        {
            // About to write into an index stamped otherwise: the stamp
            // goes first, so an interruption cannot leave this run's
            // documents vouched for by it.
            let _ = std::fs::remove_file(&index.stamp);
        }
        self.indexes.retain(|kept| kept.label != index.label);
        self.indexes.push(index);
    }

    /// Checks each index against its stamp (see [`indexed`](Self::indexed)),
    /// making a pending stamp vouch at the end of the build that wrote it,
    /// returning a line each and why the run cannot be canonical, if any
    /// cannot be vouched for.
    fn check_indexes(&self) -> (Vec<String>, Vec<String>) {
        let short = |producer: &str| -> String {
            Sha256::digest(producer.as_bytes())[..8]
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        };
        let mut lines = Vec::new();
        let mut reasons = Vec::new();
        for index in &self.indexes {
            let state = match index.found {
                0 => "built",
                found if found >= index.total => "reused",
                _ => "resumed",
            };
            let Some(producer) = (index.producer)() else {
                lines.push(format!("{}: {state}, no embedder loaded", index.label));
                continue;
            };
            let now = self.stamp_for(&producer);
            if self.source.noncanonical.is_some() {
                // A run that cannot vouch for its source -- a dirty build
                // names its clean HEAD -- vouches for no index either. One
                // that wrote into it, or finds it stamped for another
                // producer, invalidates the stamp; one that only read a
                // matching, vouched index leaves it to the canonical runs.
                let read_only = index.found >= index.total
                    && std::fs::read_to_string(&index.stamp).ok().as_deref() == Some(now.as_str());
                if !read_only {
                    let _ = std::fs::remove_file(&index.stamp);
                }
                lines.push(format!(
                    "{}: {state} by {}, not {}: the run is not canonical",
                    index.label,
                    short(&now),
                    if read_only { "vouched" } else { "stamped" }
                ));
                continue;
            }
            if index.start.as_deref() != Some(now.as_str()) {
                // What it wrote was embedded by more than one producer.
                let _ = std::fs::remove_file(&index.stamp);
                lines.push(format!(
                    "{}: {state}, producer changed during the run to {}",
                    index.label,
                    short(&now)
                ));
                reasons.push(format!(
                    "{} index written by more than one embedding producer",
                    index.label
                ));
                continue;
            }
            let stamped = std::fs::read_to_string(&index.stamp).ok();
            if index.found == 0 && std::thread::panicking() {
                // A run that failed may have failed before its build was
                // done -- after the documents, before the edges derived from
                // them -- so a pending stamp stays pending.
                lines.push(format!(
                    "{}: {state} by {}, not stamped: the run failed",
                    index.label,
                    short(&now)
                ));
            } else if index.found == 0 && stamped.as_deref() == Some(pending(&now).as_str()) {
                let held = (index.count)();
                if held < index.total {
                    // Begun empty, but not holding the corpus now: a ledger
                    // that already held every memory -- the index directory
                    // deleted without it -- writes nothing again, and the
                    // index stays as empty as it began. The stamp stays
                    // pending.
                    lines.push(format!(
                        "{}: {state} by {}, not stamped: it holds {held} of {}",
                        index.label,
                        short(&now),
                        index.total
                    ));
                    reasons.push(format!(
                        "{} index incomplete: {held} of {}",
                        index.label, index.total
                    ));
                    continue;
                }
                // Built whole from empty by one producer, and holding the
                // corpus: now it vouches -- once the stamp that says so is
                // on disk. Unpublished, the stamp stays pending, and the run
                // vouches for nothing.
                match write_stamp(&index.stamp, &now) {
                    Ok(()) => lines.push(format!("{}: {state} by {}", index.label, short(&now))),
                    Err(error) => {
                        lines.push(format!(
                            "{}: {state} by {}, not stamped: {error}",
                            index.label,
                            short(&now)
                        ));
                        reasons.push(format!(
                            "{} index stamp could not be published",
                            index.label
                        ));
                    }
                }
            } else if index.found >= index.total && stamped.as_deref() == Some(now.as_str()) {
                lines.push(format!("{}: {state} by {}", index.label, short(&now)));
            } else if index.found > 0 && index.found < index.total {
                // What the interrupted session wrote is not recorded by
                // producer, so a fallback during it cannot be ruled out.
                lines.push(format!(
                    "{}: {state} by {}, its earlier documents unverified",
                    index.label,
                    short(&now)
                ));
                reasons.push(format!(
                    "{} index resumed from an interrupted build",
                    index.label
                ));
            } else {
                lines.push(format!(
                    "{}: {state} by {}, run by {}",
                    index.label,
                    stamped.as_deref().map_or("unknown".into(), short),
                    short(&now)
                ));
                reasons.push(
                    if stamped.is_some_and(|stamp| stamp.starts_with("pending\n")) {
                        format!("{} index left pending by an unfinished build", index.label)
                    } else {
                        format!("{} index built by another embedding producer", index.label)
                    },
                );
            }
        }
        (lines, reasons)
    }

    /// Asks the probe, if there is one, and keeps its answer with `loaded`
    /// as `engine/role=provider` lines and `engine/role=repository@commit`
    /// model lines; whether there was a probe to ask.
    /// The probe never panics (see [`engines`]), so a failing run asks it
    /// too.
    ///
    /// `probe_hardware` says whether the hardware behind them may be probed now (see
    /// [`Hardware::accelerators_for`]); a failed run's unwind passes false.
    fn resolve_providers(&mut self, probe_hardware: bool) -> bool {
        let Some(probe) = &self.probe else {
            return false;
        };
        let mut plans = probe();
        for loaded in &self.loaded {
            plans.extend(loaded());
        }
        if self.activity_before.is_some() {
            let mut calibration: Vec<String> = plans
                .iter()
                .map(|(role, plan)| calibration(role, plan))
                .collect();
            calibration.sort();
            calibration.dedup();
            self.calibration = calibration;
        }
        let mut models: Vec<String> = plans
            .iter()
            .map(|(role, plan)| format!("{role}={}", plan.revision))
            .collect();
        models.sort();
        models.dedup();
        self.models = models;
        let mut providers: Vec<(String, String)> = plans
            .iter()
            .map(|(role, plan)| (role.clone(), provider(plan)))
            .collect();
        providers.sort();
        providers.dedup();
        let (indexed, unvouched) = self.check_indexes();
        self.indexed = indexed;
        for reason in plan_reasons(&providers).into_iter().chain(unvouched) {
            let noncanonical = self.source.noncanonical.get_or_insert_with(String::new);
            if !noncanonical.split("; ").any(|known| known == reason) {
                if !noncanonical.is_empty() {
                    noncanonical.push_str("; ");
                }
                noncanonical.push_str(&reason);
            }
        }
        self.providers = providers
            .into_iter()
            .map(|(role, provider)| format!("{role}={provider}"))
            .collect();
        self.hardware.accelerators = Hardware::accelerators_for(&self.providers, probe_hardware);
        true
    }

    /// One document, by every field that reaches the index.
    pub fn document(&mut self, fields: &[&str]) {
        self.documents.push(record("document", fields));
    }

    /// One question, asked in each of `groups` under `id` -- the id the
    /// harness passes to [`tally`] -- by every other field that decides how it
    /// is scored: its text and its answer key.
    pub fn query(&mut self, groups: &[&str], id: &str, fields: &[&str]) {
        let assigned = assigned(groups);
        let mut all = vec![id, assigned.as_str()];
        all.extend(fields);
        self.queries.push(record("query", &all));
        for group in groups {
            let fresh = self
                .asked
                .entry((*group).to_string())
                .or_default()
                .insert(id.to_string());
            assert!(fresh, "{}: query {id} selected twice in {group}", self.name);
        }
    }

    /// A question the selection drew and the harness will not score. It
    /// counts as asked, so the gap shows in every coverage line.
    pub fn exclude(&mut self, groups: &[&str], id: &str, reason: &str) {
        self.queries
            .push(record("excluded", &[id, &assigned(groups), reason]));
        for group in groups {
            let fresh = self
                .asked
                .entry((*group).to_string())
                .or_default()
                .insert(id.to_string());
            assert!(fresh, "{}: query {id} selected twice in {group}", self.name);
            self.excluded.push(Excluded {
                id: id.to_string(),
                group: (*group).to_string(),
                reason: reason.to_string(),
            });
        }
    }

    /// SHA-256 over the exact question and document set.
    ///
    /// Over the sorted record digests, so it names a *set*: two loaders that
    /// list the same documents in a different order agree, and a change to
    /// one character of one answer key does not.
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        for (label, records) in [("queries", &self.queries), ("documents", &self.documents)] {
            let mut sorted = records.clone();
            sorted.sort_unstable();
            hash.update(label.as_bytes());
            hash.update((sorted.len() as u64).to_be_bytes());
            for one in sorted {
                hash.update(one);
            }
        }
        format!("{:x}", hash.finalize())
    }

    /// SHA-256 over the questions in the order they were given, which is the
    /// order every arm asks them in, and the documents in the order they
    /// were given, which is the order they are written in.
    ///
    /// The [`digest`](Self::digest) names a set, so two runs over the same
    /// questions or documents in a different order share it -- and the
    /// questions are asked one after another, so order is a condition of
    /// every latency and cache figure a run reports, while the order the
    /// documents are inserted in shapes the graph and the vector index a
    /// search walks. This names both sequences, and is part of the record's
    /// name beside the digest, so the runs are kept apart.
    pub fn sequence(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"sequence");
        for records in [&self.queries, &self.documents] {
            hash.update((records.len() as u64).to_be_bytes());
            for one in records {
                hash.update(one);
            }
        }
        format!("{:x}", hash.finalize())
    }

    /// Prints the digest, the selection, the seed and every exclusion,
    /// records them under `<eval home>/shape/`, and starts counting what is
    /// scored. Called once, before anything is searched, so a run that dies
    /// still said what it was running.
    ///
    /// The record is named by `test`, `profile` and arm as well as the
    /// digest, the arm being every setting that is set (see [`settings`]).
    /// It was named by the digest alone, and
    /// every arm over one corpus wrote the same file, so the last run's
    /// coverage replaced the others'. It is named by the evaluated commit
    /// too, so a run of another revision keeps its own record rather than
    /// replacing this one's.
    pub fn publish(&mut self, test: &str, profile: &str) {
        let digest = self.digest();
        self.read_settings();
        let arm = arm(&self.settings);
        self.run = [test.to_string(), profile.to_string(), arm.clone()];
        let name = self.name.clone();
        let name = &name;
        self.source = ungated(source(), &self.settings);
        self.hardware = Hardware::detect();
        let sequence = self.sequence();

        // The in-progress record is on disk before anything is reported, so a
        // run whose output fails still leaves the snapshot it was asked for.
        let dir = super::eval_home().join("shape");
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
        let file = [
            name.as_str(),
            test,
            profile,
            &arm,
            short(&self.source.commit),
            &digest[..16],
            &sequence[..8],
        ]
        .map(file_part)
        .join("-");
        self.progress = in_progress(&dir.join(format!("{file}.json")));
        let record = dir.join(file);
        self.write(&self.progress);

        report(&format!(
            "  shape {name}: {test}, profile {profile}, arm {arm}"
        ));
        report(&format!("  shape {name}: commit {}", self.source.commit));
        if let Some(reason) = &self.source.noncanonical {
            report(&format!("  shape {name}: not canonical: {reason}"));
        }
        for setting in &self.settings {
            report(&format!("  shape {name}: arm {arm} sets {setting}"));
        }
        report(&format!(
            "  shape {name}: corpus sha256 {digest} over {} questions and {} documents",
            self.queries.len(),
            self.documents.len()
        ));
        report(&format!(
            "  shape {name}: asked in the order sha256 {sequence}"
        ));
        match self.seed {
            Some(seed) => report(&format!("  shape {name}: {}, seed {seed}", self.selection)),
            None => report(&format!("  shape {name}: {}, no seed", self.selection)),
        }
        let asked: Vec<String> = self
            .asked
            .iter()
            .map(|(group, ids)| format!("{group} {}", ids.len()))
            .collect();
        report(&format!(
            "  shape {name}: asked {}; excluded {}",
            asked.join(", "),
            self.excluded.len()
        ));
        for excluded in &self.excluded {
            report(&format!(
                "  excluded {name} {} [{}]: {}",
                excluded.id, excluded.group, excluded.reason
            ));
        }
        report(&format!(
            "  shape {name}: recording in {}, and in {}-on-<providers>.json once its coverage holds",
            self.progress.display(),
            record.display()
        ));
        self.record = Some(record);
        open_tally();
    }

    /// Checks what `tally` counted against what was asked: every declared
    /// table holds, per group, each query selected and not excluded exactly
    /// once, and nothing else; and nothing was scored outside them.
    ///
    /// Ids rather than counts, because a table that lost one query and
    /// counted another twice has the right count. Per table, because a count
    /// per query cannot see a table that never ran.
    ///
    /// Returns a line per group, and every failure found.
    fn coverage(&self, tally: &Tally) -> (Vec<String>, Vec<String>, serde_json::Value) {
        let mut lines = Vec::new();
        let mut failures = Vec::new();
        let mut recorded = serde_json::Map::new();
        if tally.declared.is_empty() {
            failures.push("the arm declared no table, so nothing it scored can be checked".into());
        }
        let filled: BTreeSet<&str> = tally
            .counts
            .keys()
            .map(|(table, ..)| table.as_str())
            .collect();
        for table in &tally.declared {
            if !filled.contains(table.as_str()) {
                failures.push(format!("{table}: declared and never scored into"));
            }
        }
        for table in filled {
            if !tally.declared.contains(table) {
                failures.push(format!(
                    "scored into {table:?}, which the arm never declared"
                ));
            }
        }

        for (group, asked) in &self.asked {
            let excluded: BTreeSet<&str> = self
                .excluded
                .iter()
                .filter(|it| &it.group == group)
                .map(|it| it.id.as_str())
                .collect();
            let expected: BTreeSet<&str> = asked
                .iter()
                .map(String::as_str)
                .filter(|id| !excluded.contains(id))
                .collect();
            // Scored exactly once in every declared table.
            // Nothing declared is nothing checked, so nothing counts as whole.
            let mut whole = if tally.declared.is_empty() {
                BTreeSet::new()
            } else {
                expected.clone()
            };
            for table in &tally.declared {
                let scored: BTreeMap<&str, usize> = tally
                    .counts
                    .iter()
                    .filter(|((into, of, _), _)| into == table && of == group)
                    .map(|((_, _, id), count)| (id.as_str(), *count))
                    .collect();
                whole.retain(|id| scored.get(id) == Some(&1));
                if !filled_any(tally, table) {
                    // Reported once above, not once per group.
                    continue;
                }
                let missing: Vec<&str> = expected
                    .iter()
                    .copied()
                    .filter(|id| !scored.contains_key(id))
                    .collect();
                let doubled: Vec<&str> = scored
                    .iter()
                    .filter(|(id, count)| expected.contains(*id) && **count > 1)
                    .map(|(id, _)| *id)
                    .collect();
                let unexpected: Vec<&str> = scored
                    .keys()
                    .copied()
                    .filter(|id| !expected.contains(id))
                    .collect();
                for (what, ids) in [
                    ("never scored", &missing),
                    ("scored more than once", &doubled),
                    ("scored though excluded or never selected", &unexpected),
                ] {
                    if !ids.is_empty() {
                        failures.push(format!("{table}, {group}: {what}: {}", named(ids)));
                    }
                }
            }

            lines.push(format!(
                "{group} asked {} scored {} x{} tables excluded {}",
                asked.len(),
                whole.len(),
                tally.declared.len(),
                excluded.len()
            ));
            recorded.insert(
                group.clone(),
                serde_json::json!({
                    "asked": asked.len(),
                    "scored": whole.len(),
                    "excluded": excluded.len(),
                }),
            );
        }
        let strays: BTreeSet<&str> = tally
            .counts
            .keys()
            .map(|(_, group, _)| group.as_str())
            .filter(|group| !self.asked.contains_key(*group))
            .collect();
        for group in strays {
            failures.push(format!("scored in {group}, which nothing was asked in"));
        }
        let recorded = serde_json::json!({
            "tables": tally.declared,
            "groups": recorded,
        });
        (lines, failures, recorded)
    }

    /// Stops counting, prints and records the coverage, and fails the run if
    /// it is not exactly what was asked less what was excluded.
    fn close(&mut self) {
        let tally = tallied(Option::take).unwrap_or_default();
        let failures = self.finish(&tally);
        assert!(
            failures.is_empty(),
            "{} coverage is not what was asked:\n  {}",
            self.name,
            failures.join("\n  ")
        );
    }

    /// Checks and records the coverage, returning every failure. The record
    /// itself is written only when there are none: a run that fails leaves
    /// its in-progress file and the last completed record as it was.
    fn finish(&mut self, tally: &Tally) -> Vec<String> {
        let (lines, mut failures, recorded) = self.coverage(tally);
        self.coverage = Some(recorded);
        let Some(record) = self.record.clone() else {
            report(&format!("  coverage {}: {}", self.name, lines.join("; ")));
            return failures;
        };
        if !self.resolve_providers(true) {
            failures.push(
                "the run never said which execution providers it ran on (Shape::ran_on)".into(),
            );
        }
        let path = if failures.is_empty() {
            self.named(&record)
        } else {
            self.progress.clone()
        };
        if let Err(error) = self.persist(&path, "", &lines) {
            panic!("{error}");
        }
        if failures.is_empty() {
            let _ = std::fs::remove_file(&self.progress);
            report(&format!(
                "  shape {}: ran on {}; recorded in {}",
                self.name,
                self.providers.join(", "),
                path.display()
            ));
        }
        failures
    }

    /// The record's file: `record`, the providers it ran on, and for a run
    /// that cannot vouch for its commit, a name of its own beside the
    /// canonical record and never over it, nor over another such run's.
    fn named(&self, record: &Path) -> PathBuf {
        let canonical = PathBuf::from(format!(
            "{}-{}.json",
            record.display(),
            self.identity().short()
        ));
        match self.source.noncanonical {
            None => canonical,
            Some(_) => unique(&canonical, "noncanonical.json"),
        }
    }

    /// Everything that tells this run apart from another; see [`Identity`].
    fn identity(&self) -> Identity {
        let [test, profile, arm] = self.run.clone();
        Identity {
            harness: self.name.clone(),
            test,
            profile,
            arm,
            settings: self.settings.clone(),
            commit: self.source.commit.clone(),
            noncanonical_reason: self.source.noncanonical.clone(),
            corpus_sha256: self.digest(),
            sequence_sha256: self.sequence(),
            selection: self.selection.clone(),
            seed: self.seed,
            providers: self.providers.clone(),
            models: self.models.clone(),
            caches: self.caches.clone(),
            warmth: self.observed_warmth(),
            indexes: self.indexed.clone(),
            build: Build::compiled(),
            hardware: self.hardware.clone(),
        }
    }

    /// Writes the record to `path`, and only then reports `lines` and any
    /// write failure, through [`report`]: on every path, a broken stdout can
    /// neither stop the record being written nor start a second panic.
    fn persist(&self, path: &Path, label: &str, lines: &[String]) -> Result<(), String> {
        let written = self.try_write(path);
        report(&format!(
            "  coverage {}{label}: {}",
            self.name,
            lines.join("; ")
        ));
        if let Err(error) = &written {
            use std::io::Write;
            let _ = writeln!(std::io::stderr(), "  shape {}: {error}", self.name);
        }
        written
    }

    fn write(&self, path: &Path) {
        if let Err(error) = self.try_write(path) {
            panic!("{error}");
        }
    }

    /// [`write`](Self::write), returning the failure rather than panicking,
    /// for a run that is already panicking.
    fn try_write(&self, path: &Path) -> Result<(), String> {
        let asked: BTreeMap<&String, usize> = self
            .asked
            .iter()
            .map(|(group, ids)| (group, ids.len()))
            .collect();
        let [test, profile, arm] = &self.run;
        let body = serde_json::json!({
            "harness": self.name,
            "test": test,
            "profile": profile,
            "arm": arm,
            "source_commit": self.source.commit,
            "canonical": self.source.noncanonical.is_none(),
            "noncanonical_reason": self.source.noncanonical,
            "settings": self.settings,
            "providers": self.providers,
            "models": self.models,
            "warmth": self.observed_warmth(),
            "indexes": self.indexed,
            "identity": self.identity(),
            "identity_sha256_12": self.identity().short(),
            "corpus_sha256": self.digest(),
            "sequence_sha256": self.sequence(),
            "selection": self.selection,
            "seed": self.seed,
            "questions": self.queries.len(),
            "documents": self.documents.len(),
            "asked": asked,
            "excluded": self.excluded,
            "scored": self.coverage,
        });
        // Beside it and renamed over it, so a run killed mid-write leaves the
        // last whole record rather than half of a new one. Named by process
        // and time, so two runs writing one record never share a partial
        // file; the last rename wins with a whole record either way.
        let partial = unique(path, "part");
        let written = std::fs::write(
            &partial,
            serde_json::to_vec_pretty(&body).expect("serialise"),
        )
        .map_err(|error| format!("writing {}: {error}", partial.display()))
        .and_then(|()| {
            std::fs::rename(&partial, path)
                .map_err(|error| format!("renaming onto {}: {error}", path.display()))
        });
        if written.is_err() {
            let _ = std::fs::remove_file(&partial);
        }
        written
    }
}

impl Drop for Shape {
    fn drop(&mut self) {
        if self.record.is_none() {
            return;
        }
        if std::thread::panicking() {
            // The run already failed -- in scoring, or at a floor after every
            // query was scored. Stop counting, record what was scored in the
            // in-progress file only, so a failed floor still says what its
            // numbers were over, and leave the failure as the one reported:
            // a second panic here would abort the process, so the write's
            // failure is reported, not raised (see `persist`).
            let Some(tally) = tallied(Option::take) else {
                return;
            };
            let (lines, _, recorded) = self.coverage(&tally);
            self.coverage = Some(recorded);
            // What it ran on, too: the probe never panics and waits only for
            // loads already running, and no hardware is probed here -- only
            // what this process already read -- so nothing but those loads
            // stands between the failure and the file.
            self.resolve_providers(false);
            let progress = self.progress.clone();
            let _ = self.persist(&progress, " (failed run)", &lines);
            return;
        }
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items() -> Vec<(String, String)> {
        // Ordered by stratum, the way MuSiQue is ordered by hop count, so a
        // prefix would see only the first.
        (0..60)
            .map(|at| (format!("a{at}"), "a".to_string()))
            .chain((0..30).map(|at| (format!("b{at}"), "b".to_string())))
            .chain((0..10).map(|at| (format!("c{at}"), "c".to_string())))
            .collect()
    }

    fn sample(wanted: usize, seed: u64) -> Vec<(String, String)> {
        stratified(
            items(),
            wanted,
            seed,
            |item| item.1.clone(),
            |item| item.0.clone(),
        )
    }

    #[test]
    fn a_sample_is_proportional_to_every_stratum() {
        let drawn = sample(20, SEED);
        let count = |name: &str| drawn.iter().filter(|item| item.1 == name).count();
        assert_eq!((count("a"), count("b"), count("c")), (12, 6, 2));
    }

    #[test]
    fn a_sample_depends_on_the_seed_and_nothing_else() {
        assert_eq!(sample(20, SEED), sample(20, SEED));
        assert_ne!(sample(20, SEED), sample(20, SEED + 1));
        let mut reversed = items();
        reversed.reverse();
        let mut backwards = stratified(
            reversed,
            20,
            SEED,
            |item| item.1.clone(),
            |item| item.0.clone(),
        );
        backwards.reverse();
        assert_eq!(
            backwards,
            sample(20, SEED),
            "the input order chose the sample"
        );
    }

    /// Every size against the next, over the strata that broke largest
    /// remainder -- 1, 3 and 3 -- and over the uneven ones above.
    #[test]
    fn a_larger_sample_contains_every_smaller_one() {
        let small_strata: Vec<(String, String)> = [("a", 1), ("b", 3), ("c", 3)]
            .into_iter()
            .flat_map(|(name, count)| {
                (0..count).map(move |at| (format!("{name}{at}"), name.to_string()))
            })
            .collect();
        for all in [small_strata, items()] {
            let draw = |wanted| {
                stratified(
                    all.clone(),
                    wanted,
                    SEED,
                    |item| item.1.clone(),
                    |item| item.0.clone(),
                )
            };
            let mut previous = Vec::new();
            for wanted in 1..=all.len() {
                let drawn = draw(wanted);
                assert_eq!(drawn.len(), wanted);
                assert!(
                    previous.iter().all(|item| drawn.contains(item)),
                    "a sample of {wanted} dropped part of the sample of {}",
                    wanted - 1
                );
                previous = drawn;
            }
        }
    }

    #[test]
    fn the_digest_names_a_set_and_sees_one_character() {
        let build = |documents: &[&str], answer: &str| {
            let mut shape = Shape::new("test", String::new(), None);
            for document in documents {
                shape.document(&["key", document]);
            }
            shape.query(&["group"], "q1", &["a question", answer]);
            shape.digest()
        };
        assert_eq!(build(&["x", "y"], "k"), build(&["y", "x"], "k"));
        assert_ne!(build(&["x", "y"], "k"), build(&["x", "y"], "K"));
        assert_ne!(build(&["x", "y"], "k"), build(&["x", "z"], "k"));
    }

    /// The same questions in another order are the same set and a different
    /// sequence.
    #[test]
    fn the_sequence_sees_the_order_the_digest_does_not() {
        let build = |ids: &[&str]| {
            let mut shape = Shape::new("test", String::new(), None);
            for id in ids {
                shape.query(&["group"], id, &["a question"]);
            }
            (shape.digest(), shape.sequence())
        };
        let (forward, backward) = (build(&["q1", "q2"]), build(&["q2", "q1"]));
        assert_eq!(forward.0, backward.0);
        assert_ne!(forward.1, backward.1);
        assert_eq!(forward, build(&["q1", "q2"]));
    }

    /// The same documents written in another order are the same set and a
    /// different sequence: insertion order shapes the index searched.
    #[test]
    fn the_sequence_sees_the_document_order() {
        let build = |documents: &[&str]| {
            let mut shape = Shape::new("test", String::new(), None);
            for document in documents {
                shape.document(&[document, "text"]);
            }
            shape.query(&["group"], "q1", &["a question"]);
            (shape.digest(), shape.sequence())
        };
        let (forward, backward) = (build(&["d1", "d2"]), build(&["d2", "d1"]));
        assert_eq!(forward.0, backward.0);
        assert_ne!(forward.1, backward.1);
    }

    /// A query moved to another group, its text and answer key unchanged,
    /// is another question set; the order groups are listed in is not.
    #[test]
    fn the_digest_sees_which_groups_a_query_is_asked_in() {
        let build = |groups: &[&str]| {
            let mut shape = Shape::new("test", String::new(), None);
            shape.query(groups, "q1", &["a question", "answer"]);
            shape.exclude(groups, "q2", "unanswerable");
            shape.digest()
        };
        assert_ne!(build(&["2hop"]), build(&["3hop"]));
        assert_ne!(build(&["a"]), build(&["a", "b"]));
        assert_eq!(build(&["a", "b"]), build(&["b", "a"]));
    }

    /// An index is stamped pending before it is first written and vouched
    /// for only once the build that began it empty finishes under one
    /// producer; an interrupted build -- which may have fallen back to
    /// another plan before it stopped -- is never vouched for, resumed or
    /// reused; and a run that is not canonical, or whose producer changed
    /// while it wrote, leaves no stamp.
    #[test]
    fn an_index_is_vouched_for_only_by_its_own_stamp() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let index = |producer: &'static str, found: usize| Indexed {
            label: "primary".into(),
            producer: Box::new(move || Some(producer.to_string())),
            stamp: stamp.clone(),
            found,
            total: 4,
            start: None,
            building: None,
            // The build wrote the whole corpus.
            count: Box::new(|| 4),
        };
        // One run: published at `commit`, canonical or not, its index begun
        // before writing; and, unless interrupted, checked at the end.
        let run = |commit: &str, dirty: bool, indexed: Indexed, finished: bool| {
            let mut shape = Shape::new("test", String::new(), None);
            shape.source.commit = commit.into();
            shape.source.noncanonical = dirty.then(|| "dirty build".to_string());
            shape.begin_index(indexed);
            finished.then(|| shape.check_indexes())
        };
        let stamped = || std::fs::read_to_string(&stamp).ok();

        // Unstamped documents from before: not vouched for.
        let (lines, reasons) = run("a", false, index("plan=cpu@a", 2), true).unwrap();
        assert_eq!(
            reasons,
            ["primary index resumed from an interrupted build"],
            "{lines:?}"
        );

        // A build interrupted before it finished is stamped pending, which
        // vouches for nothing: it may have fallen back to another plan
        // before it stopped -- from A to B here -- and its documents do not
        // say which.
        let fallen = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fallen);
        let falling = Indexed {
            producer: Box::new(move || {
                Some(
                    if flag.load(Ordering::SeqCst) {
                        "plan=cuda@b"
                    } else {
                        "plan=cpu@a"
                    }
                    .into(),
                )
            }),
            ..index("", 0)
        };
        assert!(run("a", false, falling, false).is_none());
        fallen.store(true, Ordering::SeqCst);
        assert_eq!(stamped().as_deref(), Some("pending\nplan=cpu@a\ncommit=a"));
        let (lines, reasons) = run("a", false, index("plan=cpu@a", 2), true).unwrap();
        assert_eq!(
            reasons,
            ["primary index resumed from an interrupted build"],
            "{lines:?}"
        );
        let (_, reasons) = run("a", false, index("plan=cpu@a", 4), true).unwrap();
        assert_eq!(
            reasons,
            ["primary index left pending by an unfinished build"]
        );

        // Built whole from empty by one producer: vouched for, and so is
        // every reuse after it.
        let (lines, reasons) = run("a", false, index("plan=cpu@a", 0), true).unwrap();
        assert!(reasons.is_empty(), "{lines:?}");
        assert_eq!(stamped().as_deref(), Some("plan=cpu@a\ncommit=a"));
        let (lines, reasons) = run("a", false, index("plan=cpu@a", 4), true).unwrap();
        assert!(reasons.is_empty(), "{lines:?}");
        assert!(lines[0].starts_with("primary: reused by "), "{lines:?}");

        // Another commit reusing it whole writes nothing: not vouched for,
        // and the stamp still names who built it.
        let (_, reasons) = run("b", false, index("plan=cpu@a", 4), true).unwrap();
        assert_eq!(reasons.len(), 1);
        assert_eq!(stamped().as_deref(), Some("plan=cpu@a\ncommit=a"));

        // Another plan resuming it, interrupted before the end: the stamp
        // went before anything was written, so nothing it wrote is vouched
        // for by the first producer, then or after.
        assert!(run("a", false, index("plan=cuda@b", 2), false).is_none());
        assert_eq!(stamped(), None, "a mixed index kept the first stamp");
        let (_, reasons) = run("a", false, index("plan=cpu@a", 4), true).unwrap();
        assert_eq!(
            reasons,
            ["primary index built by another embedding producer"],
            "a mixed index was vouched for"
        );

        // A producer that changed while the run wrote: stamp removed.
        let moved = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&moved);
        let changing = Indexed {
            producer: Box::new(move || {
                Some(
                    if flag.load(Ordering::SeqCst) {
                        "plan=cpu@a"
                    } else {
                        "plan=cuda@b"
                    }
                    .into(),
                )
            }),
            ..index("", 0)
        };
        let mut shape = Shape::new("test", String::new(), None);
        shape.source.commit = "a".into();
        shape.begin_index(changing);
        moved.store(true, Ordering::SeqCst);
        let (_, reasons) = shape.check_indexes();
        assert_eq!(
            reasons,
            ["primary index written by more than one embedding producer"]
        );
        assert_eq!(stamped(), None);

        // Built by a dirty build, which names the same clean commit: no
        // stamp, so a canonical run over it is not vouched for.
        assert!(run("a", false, index("plan=cpu@a", 0), true).is_some());
        let (lines, reasons) = run("a", true, index("plan=cpu@a", 0), true).unwrap();
        assert!(reasons.is_empty(), "{lines:?}");
        assert!(lines[0].contains("not stamped"), "{lines:?}");
        assert_eq!(stamped(), None, "a dirty build's index was stamped");
        let (_, reasons) = run("a", false, index("plan=cpu@a", 4), true).unwrap();
        assert_eq!(reasons.len(), 1, "a dirty build's index was vouched for");
    }

    /// A run that is not canonical -- a gate-only run, say -- and only reads
    /// a complete index vouched for its producer leaves the stamp to the
    /// canonical runs; one that finds it stamped for another producer does
    /// not.
    #[test]
    fn a_noncanonical_read_keeps_a_matching_stamp() {
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let index = |producer: &'static str| Indexed {
            label: "primary".into(),
            producer: Box::new(move || Some(producer.to_string())),
            stamp: stamp.clone(),
            found: 4,
            total: 4,
            start: None,
            building: None,
            count: Box::new(|| 4),
        };
        let read = |noncanonical: Option<&str>, indexed: Indexed| {
            let mut shape = Shape::new("test", String::new(), None);
            shape.source.commit = "a".into();
            shape.source.noncanonical = noncanonical.map(str::to_string);
            shape.begin_index(indexed);
            shape.check_indexes()
        };
        write_stamp(&stamp, "plan=cpu@a\ncommit=a").expect("stamp");

        let (lines, reasons) = read(Some("significance gate only"), index("plan=cpu@a"));
        assert!(reasons.is_empty(), "{lines:?}");
        assert!(lines[0].contains("not vouched"), "{lines:?}");
        let (lines, reasons) = read(None, index("plan=cpu@a"));
        assert!(
            reasons.is_empty(),
            "the gate-only run invalidated it: {lines:?}"
        );

        let (_, _) = read(Some("significance gate only"), index("plan=cuda@b"));
        assert_eq!(std::fs::read_to_string(&stamp).ok(), None);
    }

    /// A run that fails after building an index -- its documents written,
    /// what is derived from them perhaps not -- leaves the stamp pending,
    /// so no later run vouches for what it left.
    #[test]
    fn a_failed_run_leaves_its_index_pending() {
        struct Failing(Shape);
        impl Drop for Failing {
            fn drop(&mut self) {
                self.0.check_indexes();
            }
        }
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let index = |found: usize| Indexed {
            label: "primary".into(),
            producer: Box::new(|| Some("plan=cpu@a".into())),
            stamp: stamp.clone(),
            found,
            total: 4,
            start: None,
            building: None,
            // The build wrote the whole corpus.
            count: Box::new(|| 4),
        };
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut shape = Shape::new("test", String::new(), None);
            shape.source.commit = "a".into();
            shape.begin_index(index(0));
            let _failing = Failing(shape);
            panic!("deriving the edges failed after every document was indexed");
        }));
        assert!(failed.is_err());
        let pending = std::fs::read_to_string(&stamp).ok();
        assert_eq!(pending.as_deref(), Some("pending\nplan=cpu@a\ncommit=a"));

        let mut shape = Shape::new("test", String::new(), None);
        shape.source.commit = "a".into();
        shape.begin_index(index(4));
        let (_, reasons) = shape.check_indexes();
        assert_eq!(
            reasons,
            ["primary index left pending by an unfinished build"]
        );
    }

    /// An index begun empty that does not hold the corpus at the end is not
    /// vouched for: the ledger held every memory already -- the index
    /// directory deleted without it -- so every write was skipped and the
    /// projection stayed empty. Its stamp stays pending.
    #[test]
    fn an_index_still_empty_at_the_end_is_not_vouched_for() {
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let mut shape = Shape::new("test", String::new(), None);
        shape.source.commit = "a".into();
        // Counted empty at the start, and nothing written since.
        shape.claim_index(
            "primary",
            stamp.clone(),
            Box::new(|| Some("plan=cpu@a".into())),
            Box::new(|| 0),
            4,
        );
        let (lines, reasons) = shape.check_indexes();
        assert_eq!(reasons, ["primary index incomplete: 0 of 4"], "{lines:?}");
        assert!(lines[0].contains("not stamped"), "{lines:?}");
        assert_eq!(
            std::fs::read_to_string(&stamp).ok().as_deref(),
            Some("pending\nplan=cpu@a\ncommit=a")
        );
    }

    /// A build whose final stamp cannot be published vouches for nothing:
    /// the run is not canonical and the stamp on disk stays pending.
    #[test]
    fn an_unpublished_stamp_vouches_for_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let mut shape = Shape::new("test", String::new(), None);
        shape.source.commit = "a".into();
        shape.begin_index(Indexed {
            label: "primary".into(),
            producer: Box::new(|| Some("plan=cpu@a".into())),
            stamp: stamp.clone(),
            found: 0,
            total: 4,
            start: None,
            building: None,
            // The build wrote the whole corpus.
            count: Box::new(|| 4),
        });
        // The publication's partial file cannot be written, root or not.
        std::fs::create_dir(stamp.with_extension("partial")).expect("block the write");
        let (lines, reasons) = shape.check_indexes();
        assert_eq!(
            reasons,
            ["primary index stamp could not be published"],
            "{lines:?}"
        );
        assert!(lines[0].contains("not stamped"), "{lines:?}");
        assert_eq!(
            std::fs::read_to_string(&stamp).ok().as_deref(),
            Some("pending\nplan=cpu@a\ncommit=a")
        );
    }

    /// Two runs building one index take turns: the second waits for the
    /// first's build lock, then counts what the first built, and the stamp
    /// is the promoting build's, not overwritten by a pending claim.
    #[test]
    fn index_builds_take_turns_and_the_stamp_is_the_promoter_s() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let dir = tempfile::tempdir().expect("temp dir");
        let stamp = dir.path().join("index").join("p.eval-producer");
        let documents = Arc::new(AtomicUsize::new(0));
        let claim = |shape: &mut Shape, documents: Arc<AtomicUsize>, stamp: PathBuf| {
            shape.source.commit = "a".into();
            shape.claim_index(
                "primary",
                stamp,
                Box::new(|| Some("plan=cpu@a".into())),
                Box::new(move || documents.load(Ordering::SeqCst)),
                4,
            );
        };

        let mut first = Shape::new("test", String::new(), None);
        claim(&mut first, Arc::clone(&documents), stamp.clone());
        assert_eq!(
            std::fs::read_to_string(&stamp).ok().as_deref(),
            Some("pending\nplan=cpu@a\ncommit=a")
        );
        let (claimed, waiting) = std::sync::mpsc::channel();
        let second = std::thread::spawn({
            let documents = Arc::clone(&documents);
            let stamp = stamp.clone();
            move || {
                let mut second = Shape::new("test", String::new(), None);
                claim(&mut second, documents, stamp);
                claimed.send(()).expect("say it claimed");
                second.check_indexes()
            }
        });
        assert!(
            waiting
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "a second build claimed the index while the first held it"
        );
        // The first build writes every document and promotes its stamp.
        documents.store(4, Ordering::SeqCst);
        let (_, reasons) = first.check_indexes();
        assert!(reasons.is_empty(), "{reasons:?}");
        drop(first);

        let (lines, reasons) = second.join().expect("the second run");
        assert!(reasons.is_empty(), "{reasons:?}");
        assert!(lines[0].starts_with("primary: reused by "), "{lines:?}");
        assert_eq!(
            std::fs::read_to_string(&stamp).ok().as_deref(),
            Some("plan=cpu@a\ncommit=a")
        );
    }

    /// Two evaluations in one process take turns at loading models: one
    /// that fetches while the other waits does not fall in the other's
    /// interval, so the other is not named cold by it -- nor is its record
    /// written over an interval holding another run's loads.
    #[test]
    fn evaluations_take_turns_at_loading_models() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static FETCHED: AtomicU64 = AtomicU64::new(0);
        fn activity() -> pamin_index::LoadActivity {
            pamin_index::LoadActivity {
                fetched: FETCHED.load(Ordering::SeqCst),
                prepared: 0,
            }
        }
        let observe = || {
            let mut shape = Shape::new("test", String::new(), None);
            shape.activity = activity;
            shape.observe_models();
            shape
        };
        let fetching = observe();
        let (looked, waiting) = std::sync::mpsc::channel();
        let other = std::thread::spawn({
            move || {
                let shape = observe();
                looked.send(()).expect("say it looked");
                shape.observed_warmth()
            }
        });
        // The other evaluation cannot look while this one loads.
        assert!(
            waiting
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "two evaluations loaded models at once"
        );
        FETCHED.fetch_add(3, Ordering::SeqCst);
        assert_eq!(
            fetching.observed_warmth(),
            ["models: cold: fetched 3 / prepared 0"]
        );
        drop(fetching);
        assert_eq!(
            other.join().expect("the other evaluation"),
            ["models: warm: nothing fetched or prepared"]
        );
    }

    /// A run reads its settings only once it holds the turn, so it waits
    /// out another test's experimental arm -- which sets the environment
    /// only while it holds the turn -- rather than recording that arm's
    /// transient settings as its own.
    #[test]
    fn settings_are_read_holding_the_turn() {
        let arm = super::super::load_models();
        let (read, waiting) = std::sync::mpsc::channel();
        let other = std::thread::spawn(move || {
            let mut shape = Shape::new("test", String::new(), None);
            shape.read_settings();
            read.send(()).expect("say it read");
            shape.turn.is_some()
        });
        assert!(
            waiting
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "settings were read while another test held the turn"
        );
        drop(arm);
        assert!(
            other.join().expect("the other run"),
            "read without the turn"
        );
    }

    /// Code that sets the environment for an experimental arm takes the
    /// turn itself when its test holds none -- a comparison run from a test
    /// with no evaluation and no model-loading turn -- and reuses its
    /// thread's turn when it does, rather than waiting on itself; and it
    /// waits while another thread holds it.
    #[test]
    fn an_environment_arm_takes_the_turn_it_needs() {
        use super::super::{environment_turn, holds_turn, load_models};
        let (taken, waiting) = std::sync::mpsc::channel();
        let held = load_models();
        let unheld = std::thread::spawn(move || {
            // No turn of its own: it waits for one, then holds it.
            let turn = environment_turn();
            taken.send(()).expect("say it took the turn");
            let holding = turn.is_some() && holds_turn();
            // Holding it, it takes no second one.
            let nested = environment_turn().is_none();
            drop(turn);
            (holding, nested, holds_turn())
        });
        assert!(
            waiting
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "the environment was set while another thread held the turn"
        );
        assert!(environment_turn().is_none(), "the holder waited on itself");
        drop(held);
        assert_eq!(
            unheld.join().expect("the other thread"),
            (true, true, false)
        );
    }

    /// A run whose model loads fetched or prepared anything is cold, one
    /// whose loads found everything in place is warm, and each plan is
    /// recorded as its load found it: calibrated by the run, before it (from
    /// its file or in this process), or with nothing to calibrate.
    #[test]
    fn warmth_is_observed_from_the_loads() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static FETCHED: AtomicU64 = AtomicU64::new(0);
        fn activity() -> pamin_index::LoadActivity {
            pamin_index::LoadActivity {
                fetched: FETCHED.load(Ordering::SeqCst),
                prepared: 0,
            }
        }
        let observe = || {
            let mut shape = Shape::new("test", String::new(), None);
            shape.activity = activity;
            shape.observe_models();
            shape
        };
        // A load that fetches: cold.
        let first = observe();
        FETCHED.fetch_add(2, Ordering::SeqCst);
        assert_eq!(
            first.observed_warmth(),
            ["models: cold: fetched 2 / prepared 0"]
        );
        drop(first);
        // A second load, finding what the first fetched: warm.
        assert_eq!(
            observe().observed_warmth(),
            ["models: warm: nothing fetched or prepared"]
        );
        // Loads are counted from the first look, not a later one.
        let mut shape = observe();
        FETCHED.fetch_add(1, Ordering::SeqCst);
        shape.observe_models();
        assert_eq!(
            shape.observed_warmth(),
            ["models: cold: fetched 1 / prepared 0"]
        );
        drop(shape);

        // Each plan as its load found it: a process-cached reuse is not a
        // calibration by the run, nor is a plan read from its file.
        use pamin_index::PlanOrigin;
        let from = |role: &str, digit: &str, origin| {
            (
                role.to_string(),
                pamin_index::PlanUse {
                    origin,
                    ..used("cpu", digit, REVISION)
                },
            )
        };
        let mut shape = observe();
        shape.ran_on(move || {
            vec![
                from("a", "1", PlanOrigin::CalibratedNow),
                from("b", "2", PlanOrigin::PersistedRecord),
                from("c", "3", PlanOrigin::ProcessCache),
                from("d", "4", PlanOrigin::CpuOnly),
                from("e", "5", PlanOrigin::Deferred),
                from("f", "6", PlanOrigin::Unplanned),
            ]
        });
        assert!(shape.resolve_providers(false));
        let at = |digit: &str| digit.repeat(16);
        assert_eq!(
            shape.observed_warmth()[1..],
            [
                format!("calibrated a=cpu@{}: by the run", at("1")),
                format!(
                    "calibrated b=cpu@{}: before the run, from its plan file",
                    at("2")
                ),
                format!(
                    "calibrated c=cpu@{}: before the run, in this process",
                    at("3")
                ),
                format!("cpu-only d=cpu@{}: no calibration", at("4")),
                format!("deferred e=cpu@{}: calibration deferred", at("5")),
                format!("unplanned f=cpu@{}: no calibration", at("6")),
            ]
        );
    }

    /// A git object id: 40 lowercase hex digits under SHA-1, 64 under
    /// SHA-256.
    fn object_id(commit: &str) -> bool {
        matches!(commit.len(), 40 | 64)
            && commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }

    /// Only a clean checkout at the compiled commit writes the canonical
    /// record; everything else says why it cannot.
    #[test]
    fn only_a_clean_checkout_at_the_compiled_commit_is_canonical() {
        let (a, b) = ("a".repeat(40), "b".repeat(40));
        assert_eq!(
            provenance(&a, false, Some(&a), Some(false)),
            Source {
                commit: a.clone(),
                noncanonical: None
            }
        );
        let reason = |source: Source| {
            assert!(object_id(&source.commit), "{source:?}");
            source.noncanonical.expect("not canonical")
        };
        // A binary built at one commit and run after the checkout moved
        // names the commit it was built at.
        assert!(reason(provenance(&a, false, Some(&b), Some(false))).contains(&b));
        // A dirty checkout may or may not be what was compiled.
        assert!(
            reason(provenance(&a, false, Some(&a), Some(true)))
                .contains("changes a build compiles in")
        );
        // Edits compiled in and restored before the run: the checkout looks
        // clean at the compiled commit, but the binary is not that commit.
        assert!(reason(provenance(&a, true, Some(&a), Some(false))).contains("compiled from"));
        assert!(reason(provenance(&a, false, Some(&a), None)).contains("could not list"));
        assert!(reason(provenance(&a, false, None, None)).contains("could not read"));
        // A source archive: no commit at build time, and no git at run time.
        let archive = provenance("unknown", true, None, None);
        assert_eq!(archive.commit, "unknown");
        assert!(
            archive
                .noncanonical
                .expect("not canonical")
                .contains("outside a git")
        );
    }

    /// Every file the workspace compiles in by a literal `include_str!` or
    /// `include_bytes!` path lies under a root the build script watches, so
    /// an edit to it reruns the script and moves the dirty flag. An include
    /// built from `env!` or `concat!`, such as an `OUT_DIR` file, is a build
    /// output, not a source.
    #[test]
    fn the_build_script_watches_every_compiled_in_file() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = root.canonicalize().expect("workspace root");
        let roots: Vec<PathBuf> = env!("PAMIN_ENGINE_SOURCES")
            .split(':')
            .map(|path| root.join(path))
            .collect();
        let mut files = vec![root.join("crates")];
        let mut included = 0;
        while let Some(path) = files.pop() {
            if path.is_dir() {
                let entries = std::fs::read_dir(&path).expect("read a source directory");
                files.extend(entries.map(|entry| entry.expect("a directory entry").path()));
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read a source file");
            // Built here so this file's own text does not match.
            for macro_call in ["str", "bytes"].map(|kind| format!("include_{kind}!(")) {
                for (at, _) in text.match_indices(&macro_call) {
                    let rest = text[at + macro_call.len()..].trim_start();
                    let Some(literal) = rest.strip_prefix('"') else {
                        continue;
                    };
                    let relative = &literal[..literal.find('"').expect("a closed literal")];
                    let file = path.parent().expect("a parent").join(relative);
                    let file = file
                        .canonicalize()
                        .unwrap_or_else(|error| panic!("{}: {relative}: {error}", path.display()));
                    assert!(
                        roots.iter().any(|root| file.starts_with(root)),
                        "{} compiles in {}, which build.rs does not watch",
                        path.display(),
                        file.display()
                    );
                    included += 1;
                }
            }
        }
        // The benchmark manifests and the store's migrations, at least.
        assert!(included > 10, "found {included} includes");
    }

    /// Reading the provenance never fails a run, with or without git.
    #[test]
    fn the_record_names_the_compiled_commit() {
        let source = source();
        assert_eq!(source.commit, env!("PAMIN_ENGINE_COMMIT"));
        if source.commit != "unknown" {
            assert!(object_id(&source.commit), "{source:?}");
            assert_eq!(short(&source.commit), &source.commit[..12]);
        }
        assert_eq!(short("unknown"), "unknown");
    }

    /// Two queries asked, one excluded, in one group.
    fn asked() -> Shape {
        let mut shape = Shape::new("test", String::new(), None);
        shape.query(&["g"], "q1", &[]);
        shape.query(&["g"], "q2", &[]);
        shape.exclude(&["g"], "q3", "no judged answer");
        shape
    }

    /// How often each query was scored: table, group, id, count.
    type Scored<'a> = &'a [(&'a str, &'a str, &'a str, usize)];

    fn failures(declared: &[&str], scored: Scored<'_>) -> Vec<String> {
        let tally = Tally {
            declared: declared.iter().map(|table| table.to_string()).collect(),
            current: String::new(),
            counts: scored
                .iter()
                .map(|(table, group, id, count)| {
                    (
                        (table.to_string(), group.to_string(), id.to_string()),
                        *count,
                    )
                })
                .collect(),
        };
        asked().coverage(&tally).1
    }

    #[test]
    fn coverage_counts_exclusions_as_asked_and_not_scored() {
        // Two tables, each holding both queries: what an arm with an `off`
        // and a shipped row produces.
        assert_eq!(
            failures(
                &["off", "on"],
                &[
                    ("off", "g", "q1", 1),
                    ("off", "g", "q2", 1),
                    ("on", "g", "q1", 1),
                    ("on", "g", "q2", 1),
                ]
            ),
            Vec::<String>::new()
        );
    }

    /// Two tests publishing at once, as libtest runs them, each count only
    /// what they scored.
    #[test]
    fn two_tests_count_their_own_coverage() {
        let run = |table: &'static str, ids: &'static [&'static str]| {
            std::thread::spawn(move || {
                open_tally();
                tables([table]);
                self::table(table);
                for id in ids {
                    tally("g", id);
                }
                tallied(Option::take).expect("an open tally")
            })
        };
        let first = run("first", &["q1", "q2"]);
        let second = run("second", &["q3"]);
        let (first, second) = (first.join().unwrap(), second.join().unwrap());
        assert_eq!(first.declared, ["first".to_string()].into());
        assert_eq!(first.counts.len(), 2);
        assert!(first.counts.keys().all(|(table, ..)| table == "first"));
        assert_eq!(second.declared, ["second".to_string()].into());
        assert_eq!(second.counts.len(), 1);
        // Nothing leaked onto a thread that published nothing.
        assert!(tallied(|open| open.is_none()));
    }

    /// A run that declared no table -- one that failed before its arm began --
    /// records nothing scored, not everything.
    #[test]
    fn a_run_that_declared_no_table_scored_nothing() {
        let (lines, failures, recorded) = asked().coverage(&Tally::default());
        assert_eq!(recorded["groups"]["g"]["scored"], 0, "{recorded}");
        assert!(lines[0].contains("scored 0"), "{lines:?}");
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("declared no table"))
        );
    }

    /// A run that scored everything and then failed a floor records what it
    /// scored, in the in-progress file and not the record.
    #[test]
    fn a_run_that_fails_after_scoring_keeps_its_coverage() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let record = dir.path().join("test-run.json");
        let progress = in_progress(&record);
        let (into, kept_in) = (record.clone(), progress.clone());
        let failed = std::thread::spawn(move || {
            let mut shape = asked();
            shape.record = Some(into);
            shape.progress = kept_in;
            open_tally();
            tables(["t"]);
            table("t");
            tally("g", "q1");
            tally("g", "q2");
            panic!("below the floor");
        })
        .join();
        assert!(failed.is_err());
        assert!(!record.exists(), "a failed run wrote the record");
        let kept: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&progress).expect("the in-progress file"))
                .expect("JSON");
        assert_eq!(kept["scored"]["groups"]["g"]["scored"], 2, "{kept}");
    }

    /// A completed record survives a rerun of the same run that fails.
    #[test]
    fn a_failed_rerun_leaves_the_completed_record() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let base = dir.path().join("test-run");
        let record = {
            let mut shape = asked();
            shape.providers = vec![
                format!("embedding=cpu@{}", "a".repeat(16)),
                format!("reranking accurate=coreml@{}", "b".repeat(16)),
            ];
            shape.models = vec![
                format!("embedding={REVISION}"),
                format!("reranking accurate={REVISION}"),
            ];
            shape.hardware.accelerators = vec!["coreml".to_string()];
            shape.named(&base)
        };
        let run = |scored: Scored<'_>| {
            let mut shape = asked();
            shape.record = Some(base.clone());
            shape.progress = in_progress(&record);
            shape.ran_on(|| {
                vec![
                    ("embedding".into(), used("cpu", "a", REVISION)),
                    ("reranking accurate".into(), used("coreml", "b", REVISION)),
                ]
            });
            shape.write(&shape.progress);
            let tally = Tally {
                declared: ["t".to_string()].into(),
                current: String::new(),
                counts: scored
                    .iter()
                    .map(|(table, group, id, count)| {
                        (
                            (table.to_string(), group.to_string(), id.to_string()),
                            *count,
                        )
                    })
                    .collect(),
            };
            let failures = shape.finish(&tally);
            // Finished here, so dropping it checks nothing more.
            shape.record = None;
            (failures, shape.progress.clone())
        };

        // A failed run's in-progress file, left beside the record as a
        // concurrent run's would be while it is still going.
        let (failures, other) = run(&[("t", "g", "q1", 1)]);
        assert!(!failures.is_empty());
        assert!(other.exists(), "the failed run left nothing");

        let (failures, own) = run(&[("t", "g", "q1", 1), ("t", "g", "q2", 1)]);
        assert!(failures.is_empty());
        assert_ne!(own, other, "two runs shared an in-progress file");
        assert!(!own.exists(), "a completed run left its in-progress file");
        assert!(other.exists(), "a completed run removed another run's");
        let completed = std::fs::read(&record).expect("the completed record");
        let kept: serde_json::Value = serde_json::from_slice(&completed).expect("JSON");
        assert_eq!(
            kept["providers"],
            serde_json::json!([
                format!("embedding=cpu@{}", "a".repeat(16)),
                format!("reranking accurate=coreml@{}", "b".repeat(16)),
            ])
        );
        assert_eq!(
            kept["models"],
            serde_json::json!([
                format!("embedding={REVISION}"),
                format!("reranking accurate={REVISION}"),
            ])
        );

        let (failures, again) = run(&[("t", "g", "q1", 1)]);
        assert!(!failures.is_empty());
        assert_eq!(std::fs::read(&record).expect("the record"), completed);
        assert!(again.exists(), "the failed run left nothing");
    }

    /// The revision the test plans' model was read at.
    const REVISION: &str = "someone/model@c0ffee";

    /// A plan on `target`, its key `digit` repeated, of the model at
    /// `revision`.
    fn used(target: &str, digit: &str, revision: &str) -> pamin_index::PlanUse {
        pamin_index::PlanUse {
            target: target.into(),
            plan: digit.repeat(64),
            stable: true,
            revision: revision.into(),
            origin: pamin_index::PlanOrigin::CalibratedNow,
        }
    }

    fn identity() -> Identity {
        Identity {
            harness: "h".into(),
            test: "t".into(),
            profile: "accuracy".into(),
            arm: "shipped".into(),
            settings: Vec::new(),
            commit: "a".repeat(40),
            noncanonical_reason: None,
            corpus_sha256: "c".repeat(64),
            sequence_sha256: "s".repeat(64),
            selection: "all".into(),
            seed: None,
            providers: vec!["embedding=cpu".into(), "reranking accurate=coreml".into()],
            models: vec![
                format!("embedding={REVISION}"),
                format!("reranking accurate={REVISION}"),
            ],
            caches: Vec::new(),
            warmth: vec!["models: cold: fetched 1 / prepared 0".into()],
            indexes: Vec::new(),
            build: Build {
                inference: "f".repeat(64),
                cargo: vec!["CARGO_CFG_TARGET_FEATURE=fxsr,sse,sse2".into()],
                environment: Vec::new(),
            },
            hardware: Hardware {
                cpu: "Example CPU".into(),
                logical_cores: 4,
                accelerators: vec!["coreml".into()],
            },
        }
    }

    /// Two runs that differ only in which role ran where, or only in how
    /// the harness was compiled, are two records.
    #[test]
    fn swapped_roles_and_build_profiles_are_different_records() {
        let base = identity();
        let swapped = Identity {
            providers: vec!["embedding=coreml".into(), "reranking accurate=cpu".into()],
            ..base.clone()
        };
        let debug = Identity {
            build: Build {
                inference: "d".repeat(64),
                ..base.build.clone()
            },
            ..base.clone()
        };
        let names: BTreeSet<String> = [&base, &swapped, &debug].map(Identity::short).into();
        assert_eq!(names.len(), 3, "{names:?}");
        assert_eq!(base.short().len(), 12);
        assert_eq!(base.short(), identity().short(), "the name is stable");
    }

    /// Every field of the identity changes the name, and every field is
    /// changed here: a field added to `Identity` without a line below fails.
    #[test]
    fn every_identity_field_names_the_record() {
        type Mutation = (&'static str, fn(&mut Identity));
        let mutations: Vec<Mutation> = vec![
            ("harness", |id| id.harness.push('x')),
            ("test", |id| id.test.push('x')),
            ("profile", |id| id.profile.push('x')),
            ("arm", |id| id.arm.push('x')),
            ("settings", |id| {
                id.settings.push("SIGNIFICANCE_GATE=off".into())
            }),
            ("commit", |id| id.commit = "b".repeat(40)),
            ("noncanonical_reason", |id| {
                id.noncanonical_reason = Some("dirty".into())
            }),
            ("corpus_sha256", |id| id.corpus_sha256.push('x')),
            ("sequence_sha256", |id| id.sequence_sha256.push('x')),
            ("selection", |id| id.selection.push('x')),
            ("seed", |id| id.seed = Some(1)),
            ("providers", |id| id.providers.reverse()),
            ("models", |id| {
                id.models[0] = "embedding=someone/model@decaf".into()
            }),
            ("caches", |id| {
                id.caches
                    .push("xquad.f32: 0/10 from 0123456789abcdef".into())
            }),
            ("warmth", |id| {
                id.warmth[0] = "models: warm: nothing fetched or prepared".into()
            }),
            ("indexes", |id| {
                id.indexes.push("primary: reused by 01234567".into())
            }),
            ("hardware.cpu", |id| id.hardware.cpu.push('x')),
            ("hardware.logical_cores", |id| {
                id.hardware.logical_cores += 1
            }),
            ("hardware.accelerators", |id| {
                id.hardware.accelerators.clear()
            }),
            ("build.inference", |id| id.build.inference = "d".repeat(64)),
            ("build.cargo", |id| {
                id.build.cargo.push("CARGO_PROFILE_RELEASE_LTO=fat".into())
            }),
            ("build.environment", |id| {
                id.build.environment.push("ORT_LIB_LOCATION=/ort".into())
            }),
        ];
        let base = identity();
        for (field, mutate) in &mutations {
            let mut changed = base.clone();
            mutate(&mut changed);
            assert_ne!(
                changed.short(),
                base.short(),
                "{field} does not name the record"
            );
        }
        let value = serde_json::to_value(&base).expect("JSON");
        let leaves = |value: &serde_json::Value| -> usize {
            value
                .as_object()
                .expect("an object")
                .values()
                .map(|field| field.as_object().map_or(1, |inner| inner.len()))
                .sum()
        };
        assert_eq!(
            leaves(&value),
            mutations.len(),
            "a field of Identity has no mutation here"
        );
    }

    /// Accelerators are read off what each engine's models ran on, with the
    /// engine label and role kept apart.
    #[test]
    fn accelerators_follow_the_providers() {
        let providers = [
            "primary/embedding=cpu".to_string(),
            "second/reranking accurate=coreml".to_string(),
        ];
        assert_eq!(Hardware::accelerators_for(&providers, true), ["coreml"]);
        assert!(Hardware::accelerators_for(&providers[..1], true).is_empty());
        assert!(Hardware::detect().logical_cores > 0);
    }

    /// A failed run's unwind never probes the GPUs: what a CUDA run ran on
    /// is said to be unprobed unless this process already read it.
    #[test]
    fn the_unwind_does_not_probe_the_gpus() {
        let providers = ["primary/embedding=cuda".to_string()];
        assert_eq!(
            Hardware::accelerators_for(&providers, false),
            ["cuda: not probed (failed run)"]
        );
    }

    /// A provider is recorded as the execution target its plans are keyed
    /// by, so two NPUs that differ in provider or in id are two records, and
    /// either is still an NPU among the accelerators.
    #[test]
    fn npu_targets_name_the_record() {
        let on = |target: &str| Identity {
            providers: vec![format!("primary/embedding={target}")],
            ..identity()
        };
        let names: BTreeSet<String> = [
            on("npu:qnn:0"),
            on("npu:qnn:1"),
            on("npu:openvino:0"),
            on("npu"),
        ]
        .iter()
        .map(Identity::short)
        .collect();
        assert_eq!(names.len(), 4, "{names:?}");
        let providers = ["primary/embedding=npu:qnn:1".to_string()];
        assert_eq!(Hardware::accelerators_for(&providers, true), ["npu"]);
    }

    /// The build a record names is pamin-index's own fingerprint, so a field
    /// pamin-index adds to it reaches the record, and the readable part says
    /// what the target resolved to.
    #[test]
    fn the_build_is_pamin_index_s_fingerprint() {
        let build = Build::compiled();
        assert_eq!(build.inference, pamin_index::BUILD_FINGERPRINT);
        assert_eq!(build.inference.len(), 64, "{}", build.inference);
        assert!(
            build
                .cargo
                .iter()
                .any(|line| line.starts_with("CARGO_CFG_TARGET_ARCH=")),
            "{:?}",
            build.cargo
        );
    }

    /// A model the harness owns is asked for its plans when the record
    /// resolves its providers, so a fallback it took after it was
    /// registered makes the run non-canonical, as an engine's does.
    #[test]
    fn a_harness_model_s_later_fallback_is_recorded() {
        let plan = |target: &str, digit: &str| used(target, digit, REVISION);
        let model = owned(vec![plan("cuda:0", "a")]);
        let mut shape = asked();
        shape.ran_on(Vec::new);
        shape.also_ran("context reranker", &model, |plans| plans.clone());
        // The workload: the model falls back to the CPU after it was
        // registered.
        lock(&model).push(plan("cpu", "b"));
        assert!(shape.resolve_providers(false));
        assert_eq!(
            shape.providers,
            [
                format!("harness/context reranker=cpu@{}", "b".repeat(16)),
                format!("harness/context reranker=cuda:0@{}", "a".repeat(16)),
            ]
        );
        assert!(
            shape
                .source
                .noncanonical
                .as_deref()
                .is_some_and(|reason| reason.contains("fallback during run")),
            "{:?}",
            shape.source.noncanonical
        );
    }

    /// The same plans over a model whose cache ref moved are another
    /// record: a rerun after `refs/main` resolves to a new commit keeps the
    /// earlier revision's record rather than replacing it.
    #[test]
    fn a_moved_model_revision_is_a_different_record() {
        let named = |revision: &str| {
            let read = revision.to_string();
            let mut shape = asked();
            shape.ran_on(move || vec![("primary/embedding".into(), used("cpu", "a", &read))]);
            assert!(shape.resolve_providers(false));
            assert_eq!(shape.models, [format!("primary/embedding={revision}")]);
            shape.identity().short()
        };
        assert_ne!(named("someone/model@c0ffee"), named("someone/model@decaf"));
    }

    /// A run whose model fell back names both plans, so it is a different
    /// record from a run that only ever ran on the CPU, and neither it nor a
    /// run on a plan without a stable hardware identity is canonical.
    #[test]
    fn a_fallback_or_an_unstable_plan_is_not_canonical() {
        let fell_back = vec![
            ("primary/embedding".to_string(), "cuda@aaaa".to_string()),
            ("primary/embedding".to_string(), "cpu@bbbb".to_string()),
        ];
        let cpu = vec![("primary/embedding".to_string(), "cpu@bbbb".to_string())];
        let lines = |providers: &[(String, String)]| -> Vec<String> {
            providers
                .iter()
                .map(|(role, provider)| format!("{role}={provider}"))
                .collect()
        };
        let named = |providers: &[(String, String)]| {
            Identity {
                providers: lines(providers),
                ..identity()
            }
            .short()
        };
        assert_ne!(named(&fell_back), named(&cpu));
        assert_eq!(plan_reasons(&fell_back), ["fallback during run"]);
        assert!(plan_reasons(&cpu).is_empty());
        let unstable = vec![(
            "primary/embedding".to_string(),
            format!("directml@cccc{UNSTABLE}"),
        )];
        assert_eq!(
            plan_reasons(&unstable),
            ["unstable accelerator identity (directml)"]
        );
        let plan = pamin_index::PlanUse {
            stable: false,
            ..used("npu:qnn:0", "d", REVISION)
        };
        assert_eq!(
            provider(&plan),
            format!("npu:qnn:0@{}{UNSTABLE}", "d".repeat(16))
        );

        // And the record says so, as it does for an ungated run.
        let mut shape = asked();
        shape.ran_on(|| {
            vec![(
                "primary/embedding".to_string(),
                pamin_index::PlanUse {
                    stable: false,
                    ..used("directml", "c", REVISION)
                },
            )]
        });
        assert!(shape.resolve_providers(false));
        assert!(
            shape
                .source
                .noncanonical
                .as_deref()
                .is_some_and(|reason| reason.contains("unstable accelerator identity")),
            "{:?}",
            shape.source.noncanonical
        );
    }

    /// The GPUs a CUDA run could see are the ones `CUDA_VISIBLE_DEVICES`
    /// selects, by UUID always and by index only in `nvidia-smi`'s order.
    #[test]
    fn the_visible_gpus_are_the_selected_ones() {
        let inventory = "0, GPU-aaa, 0000:01:00.0, A100, 580.0\n\
                         1, GPU-bbb, 0000:02:00.0, H100, 580.0";
        let all = visible_cuda(Some(inventory), None, None);
        assert_eq!(all.len(), 2);
        assert_eq!(
            visible_cuda(Some(inventory), Some("GPU-bb"), None),
            ["cuda: 1, GPU-bbb, 0000:02:00.0, H100, 580.0"]
        );
        assert_eq!(
            visible_cuda(Some(inventory), Some("1,0"), Some("PCI_BUS_ID")),
            [all[1].clone(), all[0].clone()]
        );
        let unresolved = visible_cuda(Some(inventory), Some("1"), None);
        assert_eq!(unresolved.len(), 2);
        assert!(unresolved.iter().all(|line| line.contains("unresolved")));
        assert_eq!(visible_cuda(None, Some("0"), None), ["cuda: unknown"]);
    }

    /// The reranker's knobs and the search's effort name the record but not
    /// the producer: a tuned arm reuses the shipped run's vector cache and
    /// index, whose vectors no such knob touched.
    #[test]
    fn reranker_and_search_knobs_name_the_record_not_the_producer() {
        for knob in [
            "PAMIN_RERANK_DEPTH",
            "PAMIN_RERANK_BATCH",
            "PAMIN_RERANK_BATCH_TOKENS",
            "PAMIN_RERANK_MAX_TOKENS",
            "PAMIN_SEARCH_EFFORT",
        ] {
            let set = settings([(knob.to_string(), "7".to_string())]);
            assert_eq!(set, [format!("{knob}=7")], "{knob} names the record");
            assert!(runtime(set).is_empty(), "{knob} names the producer");
        }
    }

    /// Every `PAMIN_` variable the product reads is placed: on the embedding
    /// and index-build path ([`producer_pamin`]), naming vector caches and
    /// index stamps, or off it ([`NOT_PRODUCER`]) with why. A new read
    /// fails here until it is placed, so a knob on that path cannot be left
    /// out of the producer, nor one off it put in by default. Names are the
    /// quoted `PAMIN_` literals of the product crates' sources, outside
    /// comments and test modules, which is how every read is written.
    #[test]
    fn every_product_pamin_read_is_placed() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates directory");
        let mut found = BTreeSet::new();
        for name in ["pamin-index", "pamin-engine", "pamin-store", "pamin-core"] {
            pamin_literals(&crates.join(name).join("src"), &mut found);
        }
        assert!(
            found.contains("PAMIN_DEVICE") && found.contains("PAMIN_RERANK_DEPTH"),
            "this test is not reading the source: {found:?}"
        );
        let placed =
            |name: &str| producer_pamin(name) || NOT_PRODUCER.iter().any(|(off, _)| *off == name);
        let unplaced: Vec<&String> = found.iter().filter(|name| !placed(name)).collect();
        assert!(
            unplaced.is_empty(),
            "read by the product, and in neither producer_pamin nor NOT_PRODUCER: \
             {unplaced:?}"
        );
        let gone: Vec<&str> = pamin_index::EXECUTION_SETTINGS
            .iter()
            .chain(&INDEX_SETTINGS)
            .copied()
            .chain(NOT_PRODUCER.iter().map(|(name, _)| *name))
            .filter(|name| !found.contains(*name))
            .collect();
        assert!(gone.is_empty(), "placed but no longer read: {gone:?}");
    }

    /// The quoted `PAMIN_` literals in every `.rs` file under `dir`, outside
    /// comment lines and the file's test module.
    fn pamin_literals(dir: &Path, into: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("read a source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                pamin_literals(&path, into);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read a source file");
            let product = source
                .find("#[cfg(test)]\nmod tests")
                .map_or(source.as_str(), |at| &source[..at]);
            for line in product.lines() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                for (at, _) in line.match_indices("\"PAMIN_") {
                    let rest = &line[at + 1..];
                    let Some(end) = rest.find('"') else { continue };
                    let literal = &rest[..end];
                    // A bare prefix -- a `starts_with("PAMIN_")` -- names none.
                    if literal.len() > "PAMIN_".len()
                        && literal
                            .chars()
                            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                    {
                        into.insert(literal.to_string());
                    }
                }
            }
        }
    }

    /// Every environment variable the product reads outside `PAMIN_` either
    /// names the record or is listed in [`NOT_IDENTITY`] with why it
    /// changes nothing measured: reading a new one without deciding fails
    /// here. Names are taken from the string literals of the source files
    /// that read the environment, outside their tests, which is how every
    /// read is written; build scripts are left out, because what they read
    /// is compiled in and recorded as the build.
    #[test]
    fn every_environment_read_is_classified() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates directory");
        let mut found = BTreeSet::new();
        for name in [
            "pamin-index",
            "pamin-engine",
            "pamin-store",
            "pamin-cli",
            "pamin-core",
        ] {
            environment_reads(&crates.join(name).join("src"), &mut found);
        }
        assert!(
            found.contains("HF_ENDPOINT") && found.contains("CUDA_VISIBLE_DEVICES"),
            "this test is not reading the source: {found:?}"
        );
        let unclassified: Vec<&String> = found
            .iter()
            .filter(|name| !identity_variable(name))
            .filter(|name| !NOT_IDENTITY.iter().any(|(excluded, _)| excluded == name))
            .collect();
        assert!(
            unclassified.is_empty(),
            "read by the product, and neither naming the record nor in NOT_IDENTITY: \
             {unclassified:?}"
        );
        let gone: Vec<&str> = NOT_IDENTITY
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !found.contains(*name))
            .collect();
        assert!(
            gone.is_empty(),
            "in NOT_IDENTITY but no longer read: {gone:?}"
        );
    }

    /// Every environment variable the evaluation tests read outside `PAMIN_`
    /// -- each `var("...")`, `var_os("...")` and `dataset_dir("...")` under a
    /// crate's `tests/` -- names the record or is excluded with a reason, and
    /// a read whose name is not a literal is listed with where its names
    /// come from: reading a new one without deciding fails here.
    #[test]
    fn every_evaluation_read_is_classified() {
        let crates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates directory");
        let mut found = BTreeSet::new();
        let mut indirect = BTreeSet::new();
        for entry in std::fs::read_dir(crates).expect("read the crates") {
            let tests = entry.expect("a crate").path().join("tests");
            if tests.is_dir() {
                evaluation_reads(crates, &tests, &mut found, &mut indirect);
            }
        }
        assert!(
            found.contains("RERANK_RULES") && found.contains("MUSIQUE_DIR"),
            "this test is not reading the tests: {found:?}"
        );
        let excluded = |name: &str| {
            NOT_IDENTITY
                .iter()
                .chain(&EVAL_NOT_IDENTITY)
                .any(|(excluded, _)| *excluded == name)
        };
        let unclassified: Vec<&String> = found
            .iter()
            .filter(|name| !name.starts_with("PAMIN_"))
            .filter(|name| !identity_variable(name) && !excluded(name))
            .collect();
        assert!(
            unclassified.is_empty(),
            "read by an evaluation test, and neither naming the record (EVAL_SETTINGS) nor \
             in EVAL_NOT_IDENTITY: {unclassified:?}"
        );
        let listed: BTreeSet<String> = EVAL_INDIRECT
            .iter()
            .map(|(read, _)| read.to_string())
            .collect();
        assert_eq!(
            indirect, listed,
            "a read with a computed name is not in EVAL_INDIRECT, or is gone"
        );
        let gone: Vec<&str> = EVAL_SETTINGS
            .iter()
            .copied()
            .chain(EVAL_NOT_IDENTITY.iter().map(|(name, _)| *name))
            .filter(|name| !found.contains(*name))
            .collect();
        assert!(gone.is_empty(), "listed but no test reads them: {gone:?}");
    }

    /// The names of the environment reads in every `.rs` file under `dir`,
    /// and, as `path: expression`, the reads whose name is not a literal --
    /// leaving out the ones in this file, whose names are its own lists'.
    fn evaluation_reads(
        root: &Path,
        dir: &Path,
        found: &mut BTreeSet<String>,
        indirect: &mut BTreeSet<String>,
    ) {
        for entry in std::fs::read_dir(dir).expect("read a test directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                evaluation_reads(root, &path, found, indirect);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let relative = path.strip_prefix(root).expect("under the crates");
            if relative.ends_with("harness/shape.rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read a test file");
            for call in ["env::var(", "env::var_os(", "dataset_dir("] {
                for (at, _) in source.match_indices(call) {
                    let argument = source[at + call.len()..].trim_start();
                    let name = argument
                        .strip_prefix('"')
                        .and_then(|rest| rest.split_once('"'))
                        .map(|(name, _)| name);
                    match name {
                        Some(name) => {
                            found.insert(name.to_string());
                        }
                        None if call != "dataset_dir(" => {
                            let expression = argument.split(')').next().unwrap_or_default();
                            indirect.insert(format!(
                                "{}: std::{call}{expression})",
                                relative.display()
                            ));
                        }
                        None => {}
                    }
                }
            }
        }
    }

    /// The upper-case, non-`PAMIN_` string literals of every source file
    /// under `dir` that reads the environment, up to its test module.
    fn environment_reads(dir: &Path, into: &mut BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).expect("read a source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                environment_reads(&path, into);
                continue;
            }
            if path.extension().is_none_or(|extension| extension != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read a source file");
            if !["env::var", "var_os(", "env::vars"]
                .iter()
                .any(|read| source.contains(read))
            {
                continue;
            }
            let product = source
                .find("#[cfg(test)]\nmod tests")
                .map_or(source.as_str(), |at| &source[..at]);
            for (at, _) in product.match_indices('"') {
                let rest = &product[at + 1..];
                let Some(end) = rest.find('"') else { continue };
                let literal = &rest[..end];
                let variable = literal.len() > 1
                    && literal.starts_with(|first: char| first.is_ascii_uppercase())
                    && literal
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
                if variable && !literal.starts_with("PAMIN_") {
                    into.insert(literal.to_string());
                }
            }
        }
    }

    /// A published run that never says what it ran on fails, and leaves its
    /// in-progress file rather than a record named by no provider.
    #[test]
    fn a_run_that_never_names_its_providers_fails() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let mut shape = asked();
        shape.record = Some(dir.path().join("test-run"));
        shape.progress = in_progress(&dir.path().join("test-run.json"));
        let tally = Tally {
            declared: ["t".to_string()].into(),
            current: String::new(),
            counts: [("q1", 1), ("q2", 1)]
                .map(|(id, count)| (("t".to_string(), "g".to_string(), id.to_string()), count))
                .into(),
        };
        let failures = shape.finish(&tally);
        shape.record = None;
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("execution providers")),
            "{failures:?}"
        );
        assert!(shape.progress.exists());
        assert_eq!(std::fs::read_dir(dir.path()).expect("list").count(), 1);
    }

    /// An arm is named by everything set that changes a result, and by
    /// nothing else.
    #[test]
    fn an_arm_is_named_by_every_setting_that_changes_a_result() {
        let named = |set: &[(&str, &str)]| {
            let set = set
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()));
            arm(&settings(set))
        };
        let shipped = named(&[("PAMIN_EVAL_HOME", "/data"), ("HOME", "/root")]);
        assert_eq!(shipped, "shipped");
        // What `cargo test` puts in every test's environment from `build.rs`.
        let built = named(&[
            ("PAMIN_ENGINE_COMMIT", env!("PAMIN_ENGINE_COMMIT")),
            ("PAMIN_ENGINE_DIRTY", env!("PAMIN_ENGINE_DIRTY")),
            ("PAMIN_ENGINE_SOURCES", env!("PAMIN_ENGINE_SOURCES")),
            (
                "PAMIN_ENGINE_BUILD_ENVIRONMENT",
                env!("PAMIN_ENGINE_BUILD_ENVIRONMENT"),
            ),
        ]);
        assert_eq!(built, "shipped");
        let depth = named(&[("PAMIN_RERANK_DEPTH", "20")]);
        let effort = named(&[("PAMIN_SEARCH_EFFORT", "low")]);
        let both = named(&[("PAMIN_SEARCH_EFFORT", "low"), ("PAMIN_RERANK_DEPTH", "20")]);
        let swept = named(&[("SWEEP", "1"), ("PAMIN_EVAL_HOME", "/data")]);
        let distinct: BTreeSet<&String> = [&shipped, &depth, &effort, &both, &swept].into();
        assert_eq!(distinct.len(), 5, "{distinct:?}");
        // Found in any order, named the same.
        assert_eq!(
            both,
            named(&[("PAMIN_RERANK_DEPTH", "20"), ("PAMIN_SEARCH_EFFORT", "low")])
        );
        assert_eq!(
            settings([("PAMIN_DEVICE".to_string(), "cpu".to_string())]),
            ["PAMIN_DEVICE=cpu"]
        );
        // What a model computes is named by the native libraries' settings
        // and the product's embedding path, not by the harness's own knobs,
        // the gate, the reranker's or the search's, nor a variable nothing
        // on that path reads.
        let set = |name: &str, value: &str| (name.to_string(), value.to_string());
        assert_eq!(
            runtime(settings([
                set("OMP_NUM_THREADS", "1"),
                set("MKL_NUM_THREADS", "2"),
                set("OPENBLAS_NUM_THREADS", "3"),
                set("ORT_LOGGING", "x"),
                set("PAMIN_DEVICE", "cpu"),
                set("PAMIN_EVAL_HOME", "/data"),
                set("XQUAD_ALL_QUERIES", "1"),
                set("MUSIQUE_SAMPLE", "60"),
                set("SIGNIFICANCE_GATE", "off"),
                set("PAMIN_RERANK_DEPTH", "20"),
                set("PAMIN_RERANK_BATCH", "8"),
                set("PAMIN_RERANK_BATCH_TOKENS", "4096"),
                set("PAMIN_RERANK_MAX_TOKENS", "256"),
                set("PAMIN_SEARCH_EFFORT", "100"),
                set("PAMIN_SOMETHING_NEW", "1"),
                set("TOKIO_WORKER_THREADS", "2"),
            ])),
            [
                "MKL_NUM_THREADS=2",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=3",
                "ORT_LOGGING=x",
                "PAMIN_DEVICE=cpu",
            ]
        );
        // Named by every harness, though no harness lists it -- the record,
        // not the model output (above).
        let ungated = named(&[("SIGNIFICANCE_GATE", "off")]);
        assert!(!distinct.contains(&ungated), "{ungated}");
        for threads in [
            "TOKIO_WORKER_THREADS",
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
        ] {
            assert_ne!(named(&[(threads, "1")]), shipped, "{threads}");
        }
        assert_ne!(named(&[("TOKENIZERS_PARALLELISM", "false")]), shipped);
    }

    /// A run with its significance gate off is never the canonical record.
    #[test]
    fn a_run_without_its_gate_is_not_canonical() {
        let clean = || Source {
            commit: "a".repeat(40),
            noncanonical: None,
        };
        let off = ["SIGNIFICANCE_GATE=off".to_string()];
        assert_eq!(
            ungated(clean(), &off).noncanonical.as_deref(),
            Some("significance gate off")
        );
        assert_eq!(ungated(clean(), &[]), clean());
        let dirty = Source {
            noncanonical: Some("dirty".into()),
            ..clean()
        };
        assert_eq!(
            ungated(dirty, &off).noncanonical.as_deref(),
            Some("dirty; significance gate off")
        );
    }

    /// Each way a table can be over the wrong set -- including the one a
    /// count cannot see, one query lost and another counted twice.
    #[test]
    fn coverage_fails_on_any_query_out_of_place() {
        let cases: [(&[&str], Scored<'_>, &str); 9] = [
            (&["t"], &[("t", "g", "q1", 1)], "t, g: never scored: 1 (q2)"),
            (
                &["t"],
                &[
                    ("t", "g", "q1", 1),
                    ("t", "g", "q2", 1),
                    ("t", "g", "q4", 1),
                ],
                "excluded or never selected: 1 (q4)",
            ),
            (
                &["t"],
                &[
                    ("t", "g", "q1", 1),
                    ("t", "g", "q2", 1),
                    ("t", "g", "q3", 1),
                ],
                "excluded or never selected: 1 (q3)",
            ),
            (
                &["t"],
                &[("t", "g", "q1", 2), ("t", "g", "q2", 1)],
                "t, g: scored more than once: 1 (q1)",
            ),
            // Four scores over two tables, as two clean tables would be --
            // but q2 is missing from one and q1 is in it twice.
            (
                &["a", "b"],
                &[
                    ("a", "g", "q1", 1),
                    ("a", "g", "q2", 1),
                    ("b", "g", "q1", 2),
                ],
                "b, g: never scored: 1 (q2)",
            ),
            // A whole table that never ran: every query scored once, the same
            // number of times as every other, which a count cannot tell from
            // an arm that has one table.
            (
                &["off", "on"],
                &[("off", "g", "q1", 1), ("off", "g", "q2", 1)],
                "on: declared and never scored into",
            ),
            (
                &["a"],
                &[
                    ("a", "g", "q1", 1),
                    ("a", "g", "q2", 1),
                    ("b", "g", "q1", 1),
                ],
                "scored into \"b\", which the arm never declared",
            ),
            (
                &["a"],
                &[
                    ("a", "g", "q1", 1),
                    ("a", "g", "q2", 1),
                    ("a", "h", "q1", 1),
                ],
                "scored in h, which nothing was asked in",
            ),
            (&[], &[], "declared no table"),
        ];
        for (declared, scored, expected) in cases {
            let found = failures(declared, scored);
            assert!(
                found.iter().any(|failure| failure.contains(expected)),
                "{scored:?}: {found:?}"
            );
        }
    }
}
