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

use std::collections::BTreeMap;
use std::path::PathBuf;

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

/// One run's question and document set, how it was chosen, and its coverage.
pub struct Shape {
    name: String,
    selection: String,
    seed: Option<u64>,
    queries: Vec<[u8; 32]>,
    documents: Vec<[u8; 32]>,
    asked: BTreeMap<String, usize>,
    excluded: Vec<Excluded>,
    /// Per scored table: group to scored count. Recorded beside the rest.
    coverage: BTreeMap<String, BTreeMap<String, usize>>,
    /// Where [`publish`](Self::publish) wrote the record, so later checks
    /// rewrite it with their coverage.
    record: Option<PathBuf>,
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
            coverage: BTreeMap::new(),
            record: None,
        }
    }

    /// One document, by every field that reaches the index.
    pub fn document(&mut self, fields: &[&str]) {
        self.documents.push(record("document", fields));
    }

    /// One question, asked in each of `groups`, by every field that decides
    /// how it is scored: its id, its text, and its answer key.
    pub fn query(&mut self, groups: &[&str], fields: &[&str]) {
        self.queries.push(record("query", fields));
        for group in groups {
            *self.asked.entry((*group).to_string()).or_default() += 1;
        }
    }

    /// A question the selection drew and the harness will not score. It
    /// counts as asked, so the gap shows in every coverage line.
    pub fn exclude(&mut self, groups: &[&str], id: &str, reason: &str) {
        self.queries.push(record("excluded", &[id, reason]));
        for group in groups {
            *self.asked.entry((*group).to_string()).or_default() += 1;
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

    /// Prints the digest, the selection, the seed and every exclusion, and
    /// records them under `<eval home>/shape/`. Called once, before anything
    /// is searched, so a run that dies still said what it was running.
    pub fn publish(&mut self) {
        let digest = self.digest();
        let name = &self.name;
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
            .map(|(group, count)| format!("{group} {count}"))
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
        self.record = Some(dir.join(format!("{name}-{}.json", &digest[..16])));
        self.write();
        println!(
            "  shape {name}: recorded in {}",
            self.record.as_ref().expect("just set").display()
        );
    }

    /// Asserts that `scored` -- group and how many queries a table scored in
    /// it -- covers exactly what was asked less what was excluded, then prints
    /// and records the coverage.
    ///
    /// A table that scored fewer has lost questions somewhere, and one that
    /// scored more has counted something twice; either way its means are not
    /// over the set the digest names, and it fails here instead of printing.
    pub fn check<'a>(&mut self, table: &str, scored: impl IntoIterator<Item = (&'a str, usize)>) {
        let scored: BTreeMap<String, usize> = scored
            .into_iter()
            .map(|(group, count)| (group.to_string(), count))
            .collect();
        let mut line = Vec::new();
        for (group, asked) in &self.asked {
            let excluded = self.excluded.iter().filter(|it| &it.group == group).count();
            let got = scored.get(group).copied().unwrap_or(0);
            assert_eq!(
                got,
                asked - excluded,
                "{} {table}: {group} scored {got} of {asked} asked with {excluded} excluded",
                self.name
            );
            line.push(format!(
                "{group} asked {asked} scored {got} excluded {excluded}"
            ));
        }
        for group in scored.keys() {
            assert!(
                self.asked.contains_key(group),
                "{} {table}: scored a group nothing was asked in: {group}",
                self.name
            );
        }
        println!("  coverage {}, {table}: {}", self.name, line.join("; "));
        self.coverage.insert(table.to_string(), scored);
        self.write();
    }

    fn write(&self) {
        let Some(path) = &self.record else {
            return;
        };
        let body = serde_json::json!({
            "harness": self.name,
            "corpus_sha256": self.digest(),
            "selection": self.selection,
            "seed": self.seed,
            "questions": self.queries.len(),
            "documents": self.documents.len(),
            "asked": self.asked,
            "excluded": self.excluded,
            "scored": self.coverage,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&body).expect("serialise"))
            .unwrap_or_else(|error| panic!("writing {}: {error}", path.display()));
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
            shape.query(&["group"], &["q1", "a question", answer]);
            shape.digest()
        };
        assert_eq!(build(&["x", "y"], "k"), build(&["y", "x"], "k"));
        assert_ne!(build(&["x", "y"], "k"), build(&["x", "y"], "K"));
        assert_ne!(build(&["x", "y"], "k"), build(&["x", "z"], "k"));
    }

    #[test]
    fn coverage_counts_exclusions_as_asked_and_not_scored() {
        let mut shape = Shape::new("test", String::new(), None);
        shape.query(&["g"], &["q1"]);
        shape.query(&["g"], &["q2"]);
        shape.exclude(&["g"], "q3", "no judged answer");
        shape.check("table", [("g", 2)]);
    }

    #[test]
    #[should_panic(expected = "g scored 1 of 3 asked with 1 excluded")]
    fn a_table_that_lost_a_question_fails() {
        let mut shape = Shape::new("test", String::new(), None);
        shape.query(&["g"], &["q1"]);
        shape.query(&["g"], &["q2"]);
        shape.exclude(&["g"], "q3", "no judged answer");
        shape.check("table", [("g", 1)]);
    }
}
