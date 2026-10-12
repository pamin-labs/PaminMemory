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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};

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

/// Every `(group, query id)` scored since the open shape was published, and
/// how many times.
///
/// Process-wide rather than a field a harness threads through, because the
/// places a query is scored are a harness's scoring function and the
/// callbacks a dozen diagnostic arms hand it, several layers down; each
/// harness calls [`tally`] once, where its answer key is read, and every arm
/// is covered by construction. `None` while no shape is open, so unit tests
/// that score nothing published are not counted.
static TALLY: Mutex<Option<BTreeMap<(String, String), usize>>> = Mutex::new(None);

fn tallied() -> std::sync::MutexGuard<'static, Option<BTreeMap<(String, String), usize>>> {
    TALLY.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Which arm of a harness this run is: each of `vars` that is set, with its
/// value when the value is short and plain (`TIERS=fast`), else its name
/// alone (`FEATURES_OUT`, a path). `shipped` when none is set.
pub fn arm(vars: &[&str]) -> String {
    let set: Vec<String> = vars
        .iter()
        .filter_map(|var| {
            let value = std::env::var(var).ok()?;
            let plain = value.len() <= 16
                && value
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || ",._".contains(c));
            Some(if plain {
                format!("{var}={value}")
            } else {
                (*var).to_string()
            })
        })
        .collect();
    if set.is_empty() {
        "shipped".to_string()
    } else {
        set.join("+")
    }
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

/// Records that query `id` was scored once in `group`. Called where a harness
/// reads its answer key, so a query graded is a query counted.
pub fn tally(group: &str, id: &str) {
    if let Some(counts) = tallied().as_mut() {
        *counts
            .entry((group.to_string(), id.to_string()))
            .or_default() += 1;
    }
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
    /// Which test, profile and arm the run was, as `publish` was told.
    run: [String; 3],
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
            run: Default::default(),
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

    /// Prints the digest, the selection, the seed and every exclusion,
    /// records them under `<eval home>/shape/`, and starts counting what is
    /// scored. Called once, before anything is searched, so a run that dies
    /// still said what it was running.
    ///
    /// The record is named by `test`, `profile` and `arm` (see [`arm`]) as
    /// well as the digest. It was named by the digest alone, and every arm
    /// over one corpus wrote the same file, so the last run's coverage
    /// replaced the others'.
    pub fn publish(&mut self, test: &str, profile: &str, arm: &str) {
        let digest = self.digest();
        self.run = [test.to_string(), profile.to_string(), arm.to_string()];
        let name = &self.name;
        println!("  shape {name}: {test}, profile {profile}, arm {arm}");
        println!(
            "  shape {name}: corpus sha256 {digest} over {} questions and {} documents",
            self.queries.len(),
            self.documents.len()
        );
        match self.seed {
            Some(seed) => println!("  shape {name}: {}, seed {seed}", self.selection),
            None => println!("  shape {name}: {}, no seed", self.selection),
        }
        let asked: Vec<String> = self
            .asked
            .iter()
            .map(|(group, ids)| format!("{group} {}", ids.len()))
            .collect();
        println!(
            "  shape {name}: asked {}; excluded {}",
            asked.join(", "),
            self.excluded.len()
        );
        for excluded in &self.excluded {
            println!(
                "  excluded {name} {} [{}]: {}",
                excluded.id, excluded.group, excluded.reason
            );
        }

        let dir = super::eval_home().join("shape");
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|error| panic!("creating {}: {error}", dir.display()));
        let file = [name.as_str(), test, profile, arm, &digest[..16]]
            .map(file_part)
            .join("-");
        self.record = Some(dir.join(format!("{file}.json")));
        self.write();
        println!(
            "  shape {name}: recorded in {}",
            self.record.as_ref().expect("just set").display()
        );

        // Two harness tests in one process would count into one tally and
        // fail each other's check with each other's queries; say so instead.
        let mut open = tallied();
        assert!(
            open.is_none(),
            "another harness test is already counting coverage in this process; \
             run one at a time with --exact"
        );
        *open = Some(BTreeMap::new());
    }

    /// Checks `counts` -- `(group, id)` to times scored -- against what was
    /// asked: per group, the queries selected and not excluded, each scored
    /// the same number of times, and nothing else.
    ///
    /// The same number of times because an arm scores each query into every
    /// table it prints -- three tiers, or every channel taken away in turn --
    /// so a query with fewer is missing from some table and one with more was
    /// counted twice in one. Either way that table's mean is not over the set
    /// the digest names. Ids rather than counts, because a table that lost one
    /// query and counted another twice has the right count.
    ///
    /// Returns a line per group, and every failure found.
    fn coverage(
        &self,
        counts: &BTreeMap<(String, String), usize>,
    ) -> (Vec<String>, Vec<String>, serde_json::Value) {
        let mut lines = Vec::new();
        let mut failures = Vec::new();
        let mut recorded = serde_json::Map::new();
        for (group, asked) in &self.asked {
            let excluded: BTreeSet<&str> = self
                .excluded
                .iter()
                .filter(|it| &it.group == group)
                .map(|it| it.id.as_str())
                .collect();
            let scored: BTreeMap<&str, usize> = counts
                .iter()
                .filter(|((scored_in, _), _)| scored_in == group)
                .map(|((_, id), count)| (id.as_str(), *count))
                .collect();
            let expected: Vec<&str> = asked
                .iter()
                .map(String::as_str)
                .filter(|id| !excluded.contains(id))
                .collect();

            // How many tables an arm scores each query into: the commonest
            // count, the larger on a tie.
            let mut frequency: BTreeMap<usize, usize> = BTreeMap::new();
            for id in &expected {
                if let Some(count) = scored.get(id) {
                    *frequency.entry(*count).or_default() += 1;
                }
            }
            let tables = frequency
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)))
                .map_or(0, |(count, _)| *count);

            let missing: Vec<&str> = expected
                .iter()
                .copied()
                .filter(|id| !scored.contains_key(id))
                .collect();
            let short: Vec<&str> = expected
                .iter()
                .copied()
                .filter(|id| scored.get(id).is_some_and(|count| *count < tables))
                .collect();
            let doubled: Vec<&str> = expected
                .iter()
                .copied()
                .filter(|id| scored.get(id).is_some_and(|count| *count > tables))
                .collect();
            let unexpected: Vec<&str> = scored
                .keys()
                .copied()
                .filter(|id| !expected.contains(id))
                .collect();
            for (what, ids) in [
                ("never scored", &missing),
                ("scored in fewer tables than the rest", &short),
                ("scored more often than the rest", &doubled),
                ("scored though excluded or never selected", &unexpected),
            ] {
                if !ids.is_empty() {
                    failures.push(format!("{group}: {what}: {}", named(ids)));
                }
            }

            let got = expected.len() - missing.len();
            lines.push(format!(
                "{group} asked {} scored {got} x{tables} excluded {}",
                asked.len(),
                excluded.len()
            ));
            recorded.insert(
                group.clone(),
                serde_json::json!({
                    "asked": asked.len(),
                    "scored": got,
                    "tables": tables,
                    "excluded": excluded.len(),
                }),
            );
        }
        let strays: BTreeSet<&str> = counts
            .keys()
            .map(|(group, _)| group.as_str())
            .filter(|group| !self.asked.contains_key(*group))
            .collect();
        for group in strays {
            failures.push(format!("scored in {group}, which nothing was asked in"));
        }
        (lines, failures, serde_json::Value::Object(recorded))
    }

    /// Stops counting, prints and records the coverage, and fails the run if
    /// it is not exactly what was asked less what was excluded.
    fn close(&mut self) {
        let counts = tallied().take().unwrap_or_default();
        let (lines, failures, recorded) = self.coverage(&counts);
        println!("  coverage {}: {}", self.name, lines.join("; "));
        self.coverage = Some(recorded);
        self.write();
        assert!(
            failures.is_empty(),
            "{} coverage is not what was asked:\n  {}",
            self.name,
            failures.join("\n  ")
        );
    }

    fn write(&self) {
        let Some(path) = &self.record else {
            return;
        };
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
            "corpus_sha256": self.digest(),
            "selection": self.selection,
            "seed": self.seed,
            "questions": self.queries.len(),
            "documents": self.documents.len(),
            "asked": asked,
            "excluded": self.excluded,
            "scored": self.coverage,
        });
        // Beside it and renamed over it, so a run killed mid-write leaves the
        // last whole record rather than half of a new one.
        let partial = path.with_extension("json.part");
        std::fs::write(
            &partial,
            serde_json::to_vec_pretty(&body).expect("serialise"),
        )
        .unwrap_or_else(|error| panic!("writing {}: {error}", partial.display()));
        std::fs::rename(&partial, path)
            .unwrap_or_else(|error| panic!("renaming onto {}: {error}", path.display()));
    }
}

impl Drop for Shape {
    fn drop(&mut self) {
        if self.record.is_none() {
            return;
        }
        if std::thread::panicking() {
            // The run already failed; stop counting so nothing after it in
            // this process is counted against it, and leave the failure as
            // the one reported.
            tallied().take();
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

    /// Two queries asked, one excluded, in one group.
    fn asked() -> Shape {
        let mut shape = Shape::new("test", String::new(), None);
        shape.query(&["g"], "q1", &[]);
        shape.query(&["g"], "q2", &[]);
        shape.exclude(&["g"], "q3", "no judged answer");
        shape
    }

    fn failures(scored: &[(&str, &str, usize)]) -> Vec<String> {
        let counts = scored
            .iter()
            .map(|(group, id, count)| ((group.to_string(), id.to_string()), *count))
            .collect();
        asked().coverage(&counts).1
    }

    #[test]
    fn coverage_counts_exclusions_as_asked_and_not_scored() {
        // Two tables, each holding both queries: what an arm with an `off`
        // and a shipped row produces.
        assert_eq!(
            failures(&[("g", "q1", 2), ("g", "q2", 2)]),
            Vec::<String>::new()
        );
    }

    /// Each way a table can be over the wrong set -- including the one a
    /// count cannot see, one query lost and another counted twice.
    #[test]
    fn coverage_fails_on_any_query_out_of_place() {
        let cases: [(&[(&str, &str, usize)], &str); 6] = [
            (&[("g", "q1", 1)], "never scored: 1 (q2)"),
            (
                &[("g", "q1", 3), ("g", "q2", 3), ("g", "q4", 1)],
                "excluded or never selected: 1 (q4)",
            ),
            (
                &[("g", "q1", 2), ("g", "q2", 2), ("g", "q3", 2)],
                "excluded or never selected: 1 (q3)",
            ),
            (
                &[("g", "q1", 2), ("g", "q2", 1), ("g", "q5", 2)],
                "fewer tables than the rest: 1 (q2)",
            ),
            // Four scores over two tables, as two clean tables would be --
            // but q2 is missing from one and q1 is in another twice.
            (
                &[("g", "q1", 3), ("g", "q2", 1)],
                "fewer tables than the rest: 1 (q2)",
            ),
            (
                &[("g", "q1", 1), ("g", "q2", 1), ("h", "q1", 1)],
                "scored in h, which nothing was asked in",
            ),
        ];
        for (scored, expected) in cases {
            let found = failures(scored);
            assert!(
                found.iter().any(|failure| failure.contains(expected)),
                "{scored:?}: {found:?}"
            );
        }
    }
}
