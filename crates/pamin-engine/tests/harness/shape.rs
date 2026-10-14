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
/// change no result -- plus `PAMIN_PROFILE`, which the record names already.
/// Every other `PAMIN_` variable is taken to change what is measured.
const NOT_SETTINGS: [&str; 12] = [
    "PAMIN_EMBEDDING_TEST_OUTPUT",
    "PAMIN_EVAL_HOME",
    "PAMIN_HOME",
    "PAMIN_LOG",
    "PAMIN_MATRIX_OUTPUT",
    "PAMIN_NATIVE_TEST_CACHE",
    "PAMIN_POSTGRES_DIR",
    "PAMIN_PREPARED_CACHE",
    "PAMIN_PROFILE",
    "PAMIN_RERANK_TEST_OUTPUT",
    "PAMIN_SHARED_CACHE",
    "PAMIN_TEST_MODEL_CACHE",
];

/// What makes this run an arm other than the shipped one, as sorted
/// `NAME=value` lines: each of the harness's own `vars` that is set, and
/// every set `PAMIN_` variable but [`NOT_SETTINGS`].
///
/// The `PAMIN_` variables are found rather than listed. A list was kept per
/// harness, and the product's runtime knobs -- `PAMIN_RERANK_DEPTH`,
/// `PAMIN_SEARCH_EFFORT` -- were never on it, so a run with one set was
/// recorded as the shipped run, over the shipped run's record.
fn settings(vars: &[&str], set: impl IntoIterator<Item = (String, String)>) -> Vec<String> {
    let mut settings: Vec<String> = set
        .into_iter()
        .filter(|(name, _)| {
            vars.contains(&name.as_str())
                || (name.starts_with("PAMIN_") && !NOT_SETTINGS.contains(&name.as_str()))
        })
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    settings.sort();
    settings
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
    /// Where [`publish`](Self::publish) wrote the record; set only on a
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
        }
    }

    /// One document, by every field that reaches the index.
    pub fn document(&mut self, fields: &[&str]) {
        self.documents.push(record("document", fields));
    }

    /// One question, asked in each of `groups` under `id` -- the id the
    /// harness passes to [`tally`] -- by every other field that decides how it
    /// is scored: its text and its answer key.
    pub fn query(&mut self, groups: &[&str], id: &str, fields: &[&str]) {
        let mut all = vec![id];
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
        self.queries.push(record("excluded", &[id, reason]));
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
    /// order every arm asks them in.
    ///
    /// The [`digest`](Self::digest) names a set, so two runs over the same
    /// questions in a different order share it -- and the questions are asked
    /// one after another, so order is a condition of every latency and cache
    /// figure a run reports. This names the sequence, and is part of the
    /// record's name beside the digest, so the two runs are kept apart.
    pub fn sequence(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"sequence");
        hash.update((self.queries.len() as u64).to_be_bytes());
        for one in &self.queries {
            hash.update(one);
        }
        format!("{:x}", hash.finalize())
    }

    /// Prints the digest, the selection, the seed and every exclusion,
    /// records them under `<eval home>/shape/`, and starts counting what is
    /// scored. Called once, before anything is searched, so a run that dies
    /// still said what it was running.
    ///
    /// The record is named by `test`, `profile` and arm as well as the
    /// digest, the arm being the harness's `vars` and the `PAMIN_` settings
    /// that are set (see [`settings`]). It was named by the digest alone, and
    /// every arm over one corpus wrote the same file, so the last run's
    /// coverage replaced the others'. It is named by the evaluated commit
    /// too, so a run of another revision keeps its own record rather than
    /// replacing this one's.
    pub fn publish(&mut self, test: &str, profile: &str, vars: &[&str]) {
        let digest = self.digest();
        let set = std::env::vars_os().map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                value.to_string_lossy().into_owned(),
            )
        });
        self.settings = settings(vars, set);
        let arm = arm(&self.settings);
        self.run = [test.to_string(), profile.to_string(), arm.clone()];
        let name = self.name.clone();
        let name = &name;
        self.source = source();
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
        let canonical = dir.join(format!("{file}.json"));
        // Beside the canonical record and never over it, nor over another
        // such run's.
        let record = match self.source.noncanonical {
            None => canonical,
            Some(_) => unique(&canonical, "noncanonical.json"),
        };
        self.progress = in_progress(&record);
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
            "  shape {name}: recording in {}, and in {} once its coverage holds",
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
        let (lines, failures, recorded) = self.coverage(tally);
        self.coverage = Some(recorded);
        let Some(record) = self.record.clone() else {
            report(&format!("  coverage {}: {}", self.name, lines.join("; ")));
            return failures;
        };
        let path = if failures.is_empty() {
            record
        } else {
            self.progress.clone()
        };
        if let Err(error) = self.persist(&path, "", &lines) {
            panic!("{error}");
        }
        if failures.is_empty() {
            let _ = std::fs::remove_file(&self.progress);
        }
        failures
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
        let record = dir.path().join("test-run.json");
        let run = |scored: Scored<'_>| {
            let mut shape = asked();
            shape.record = Some(record.clone());
            shape.progress = in_progress(&record);
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

        let (failures, again) = run(&[("t", "g", "q1", 1)]);
        assert!(!failures.is_empty());
        assert_eq!(std::fs::read(&record).expect("the record"), completed);
        assert!(again.exists(), "the failed run left nothing");
    }

    /// An arm is named by everything set that changes a result, and by
    /// nothing else.
    #[test]
    fn an_arm_is_named_by_every_setting_that_changes_a_result() {
        let named = |set: &[(&str, &str)]| {
            let set = set
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()));
            arm(&settings(&["SWEEP"], set))
        };
        let shipped = named(&[
            ("PAMIN_EVAL_HOME", "/data"),
            ("PAMIN_PROFILE", "fast"),
            ("HOME", "/root"),
        ]);
        assert_eq!(shipped, "shipped");
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
            settings(&[], [("PAMIN_DEVICE".to_string(), "cpu".to_string())]),
            ["PAMIN_DEVICE=cpu"]
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
