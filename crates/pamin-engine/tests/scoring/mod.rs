//! How a ranking is scored, once, for all three corpora.
//!
//! The three harnesses each had their own `Scores` with the same five fields
//! and the same arithmetic, differing only in how they asked whether a rank was
//! relevant -- one held `HashSet<&str>`, one held `HashSet<String>`, and the
//! third computed nDCG and recall in free functions before handing them over.
//! Three copies of a discounted cumulative gain is three chances for a corpus
//! to be scored slightly differently from the corpus it is being compared
//! against, which is the one thing a comparison cannot survive.
//!
//! So relevance arrives as a predicate and a count, which is all the arithmetic
//! ever needed from it, and each harness keeps its own judgements where they
//! belong -- in the corpus loader that understands them.

// Not every harness uses every helper here -- three test binaries include this
// file and each one's compilation sees the others' helpers as dead. The
// alternative is a helper per harness, which is what this module exists to undo.
#![allow(dead_code)]
/// Where the ranking quality is read.
///
/// Ten, because that is roughly what fits in a context window a caller would
/// actually paste, and because the reranker only reorders; anything it could
/// fix has to already be inside the shortlist.
pub const NDCG_AT: usize = 10;

/// Where recall is read.
///
/// Fifty, one channel's depth. What is here and below rank ten is the space a
/// second pass could still recover; what is not here at all is lost to every
/// pass.
pub const RECALL_AT: usize = 50;

/// Running totals over one set of queries.
///
/// Two metrics, reported side by side because they fail differently and a
/// reranker can only fix one of them: nothing recovers a memory that was never
/// returned. nDCG is normalized discounted cumulative gain over *binary*
/// relevance -- binary because a hand-written corpus cannot honestly carry
/// graded relevance. "This memory answers the query" is a judgement one author
/// can make consistently; "this one answers it 0.7 as well" is not.
#[derive(Clone, Default)]
pub struct Scores {
    pub queries: usize,
    pub ndcg: f64,
    pub recall: f64,
    /// Relevant results that came back inside the shortlist but below rank ten
    /// -- everything a second pass over the shortlist could still fix.
    pub deep: usize,
    /// Queries with at least one of those.
    pub with_work: usize,
    /// Every query's own nDCG, in the order they were scored.
    ///
    /// Kept beside the running total because a mean cannot be tested and a list
    /// can. Two runs over the same queries in the same order pair up entry by
    /// entry, which is what `statistics::compare` needs to say whether a
    /// difference of means is a result -- see that module for why this project
    /// stopped reporting the means alone.
    pub per_query: Vec<f64>,
    /// Every query's own recall, in the same order, for the same reason.
    pub per_query_recall: Vec<f64>,
}

impl Scores {
    /// Scores one ranking.
    ///
    /// `relevant` answers whether one returned key is relevant and `judged` is
    /// how many relevant keys there are in total -- which is not the same as
    /// how many came back, and is what the ideal gain and the recall
    /// denominator are both built from. A query with nothing judged relevant
    /// scores zero on both rather than dividing by it: it is a query the corpus
    /// cannot grade, and grading it one would reward returning nothing.
    ///
    /// Returns this query's own nDCG, for the harnesses that also want to name
    /// the queries that went worst.
    pub fn add(
        &mut self,
        ranked: &[String],
        judged: usize,
        relevant: impl Fn(&str) -> bool,
    ) -> f64 {
        // Binary relevance is graded relevance whose gains are all one, and
        // whose ideal is `judged` of them. Expressed that way rather than
        // computed separately, so the two paths cannot drift: a corpus scored
        // by one and compared against a corpus scored by the other is the
        // failure this module exists to prevent.
        self.add_graded(ranked, &vec![1.0; judged], |key| f64::from(relevant(key)))
    }

    /// Scores one ranking against graded judgements.
    ///
    /// `ideal` is every judged document's gain, **sorted descending**, which
    /// is what the perfect ranking would have earned; `gain` is what one
    /// returned key is worth, and zero for one that is not judged relevant at
    /// all. Gains are the caller's, not this module's: a corpus that publishes
    /// four labels decides what they are worth, and the usual `2^label - 1` is
    /// a choice about that corpus rather than a property of nDCG.
    ///
    /// Recall and the `deep` count read "relevant" as "gain above zero",
    /// because recall has no graded form -- a document is either returned or
    /// it is not.
    pub fn add_graded(
        &mut self,
        ranked: &[String],
        ideal: &[f64],
        gain: impl Fn(&str) -> f64,
    ) -> f64 {
        let at = |rank: usize| gain(ranked[rank].as_str());
        let discount = |rank: usize| 1.0 / ((rank + 2) as f64).log2();

        let gained: f64 = (0..ranked.len().min(NDCG_AT))
            .map(|rank| at(rank) * discount(rank))
            .sum();
        // Asserted rather than sorted here: sorting silently would hide a
        // caller that built the ideal from the ranking instead of from the
        // judgements, which is how a recall failure gets reported as a perfect
        // ranking.
        debug_assert!(
            ideal.windows(2).all(|pair| pair[0] >= pair[1]),
            "the ideal gains must be sorted descending"
        );
        let best: f64 = ideal
            .iter()
            .take(NDCG_AT)
            .enumerate()
            .map(|(rank, gain)| gain * discount(rank))
            .sum();

        let judged = ideal.iter().filter(|gain| **gain > 0.0).count();
        let found = (0..ranked.len().min(RECALL_AT))
            .filter(|rank| at(*rank) > 0.0)
            .count();
        let deep = (NDCG_AT..ranked.len().min(RECALL_AT))
            .filter(|rank| at(*rank) > 0.0)
            .count();

        self.queries += 1;
        let ndcg = if best == 0.0 { 0.0 } else { gained / best };
        self.ndcg += ndcg;
        self.per_query.push(ndcg);
        let recall = if judged == 0 {
            0.0
        } else {
            found as f64 / judged as f64
        };
        self.recall += recall;
        self.per_query_recall.push(recall);
        self.deep += deep;
        self.with_work += usize::from(deep > 0);
        ndcg
    }

    /// Folds another set of the same queries in, for a harness that scores a
    /// ranking into more than one table.
    pub fn absorb(&mut self, other: Scores) {
        self.queries += other.queries;
        self.ndcg += other.ndcg;
        self.recall += other.recall;
        self.deep += other.deep;
        self.with_work += other.with_work;
        self.per_query.extend(other.per_query);
        self.per_query_recall.extend(other.per_query_recall);
    }

    pub fn mean_ndcg(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.ndcg / self.queries as f64
    }

    pub fn mean_recall(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        self.recall / self.queries as f64
    }
}

/// Where a run's per-question scores are kept, in `root`, for `project`.
///
/// Named by the project, which each harness names by profile and corpus
/// fingerprint, so a file can only be paired with a run over the same
/// questions.
pub fn saved(root: &std::path::Path, project: &str) -> std::path::PathBuf {
    root.join(format!("shipped-{project}.tsv"))
}

/// Writes every group's per-question nDCG and recall, one line a question in
/// the order they were scored.
///
/// So that a run on another project over the same questions can be paired
/// with this one later, rather than holding both projects open in one process
/// -- two models and two indexes -- or at one moment, when another run may hold
/// one of them. Written whole and renamed into place, so a run that dies
/// leaves the previous file rather than half of a new one.
///
/// The first line names what the project's vectors were embedded from, because
/// two projects over one corpus can differ in that as well as in their model:
/// an index built before names were embedded goes on being written from
/// content, so one profile's old project and another profile's new one differ
/// by two changes, and a pairing of them credits the model with both.
/// [`against`] refuses that pair.
pub fn save(
    path: &std::path::Path,
    passage: pamin_index::Passage,
    groups: &std::collections::BTreeMap<String, Scores>,
) {
    let mut lines = format!("passage\t{passage:?}\n");
    for (group, scores) in groups {
        for (ndcg, recall) in scores.per_query.iter().zip(&scores.per_query_recall) {
            // `{}` on an `f64` is its shortest round-trip form, so what is
            // read back is the same value, not one rounded to the table's.
            lines.push_str(&format!("{group}\t{ndcg}\t{recall}\n"));
        }
    }
    let partial = path.with_extension("partial");
    std::fs::write(&partial, lines)
        .unwrap_or_else(|error| panic!("writing {}: {error}", partial.display()));
    std::fs::rename(&partial, path)
        .unwrap_or_else(|error| panic!("naming {}: {error}", path.display()));
    println!("  per-question scores saved to {}", path.display());
}

/// A run [`save`] kept: what its project's vectors were embedded from, and
/// its scores.
pub struct Saved {
    pub passage: String,
    pub groups: std::collections::BTreeMap<String, Scores>,
}

/// Reads back what [`save`] wrote. `deep` and `with_work` are not kept, so
/// they read as zero.
pub fn load(path: &std::path::Path) -> Saved {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let mut lines = text.lines();
    let passage = match lines.next().and_then(|line| line.split_once('\t')) {
        Some(("passage", passage)) => passage.to_string(),
        _ => panic!(
            "{} does not say what its vectors were embedded from, so it cannot be \
             paired with anything: run it again",
            path.display()
        ),
    };
    let mut groups: std::collections::BTreeMap<String, Scores> = Default::default();
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        let [group, ndcg, recall] = fields[..] else {
            panic!("{}: not a saved score: {line:?}", path.display());
        };
        let value = |field: &str| -> f64 {
            field
                .parse()
                .unwrap_or_else(|_| panic!("{}: not a number: {line:?}", path.display()))
        };
        let into = groups.entry(group.to_string()).or_default();
        into.queries += 1;
        into.ndcg += value(ndcg);
        into.recall += value(recall);
        into.per_query.push(value(ndcg));
        into.per_query_recall.push(value(recall));
    }
    Saved { passage, groups }
}

/// This run against a saved one over the same questions, query by query, in
/// both metrics: the table a change of model rests on.
///
/// Panics when a group is missing from either side or holds a different
/// number of questions -- see `statistics::compare` -- because a pairing of
/// two different question sets is not a comparison of anything. And when the
/// two projects embedded different text, because that pairing measures two
/// changes and names one: the first pplx trial paired its new project with
/// BGE-M3's, built from content before names were embedded, and credited the
/// model with the encoding's share.
pub fn against(
    title: &str,
    before: &Saved,
    passage: pamin_index::Passage,
    after: &std::collections::BTreeMap<String, Scores>,
) {
    use crate::statistics::compare;

    assert_eq!(
        before.passage,
        format!("{passage:?}"),
        "the saved run's project embedded {} passages and this one {passage:?}, so \
         the pairing would measure the encoding as well as the change it names. \
         Rebuild the older project -- a fresh workspace, or `pamin reindex` -- and \
         run it again",
        before.passage
    );
    let before = &before.groups;

    let mut names: Vec<&String> = before.keys().chain(after.keys()).collect();
    names.sort();
    names.dedup();
    println!("\n  {title}");
    println!("  group            queries   nDCG@{NDCG_AT} before    after   paired");
    for group in &names {
        let (Some(was), Some(is)) = (before.get(*group), after.get(*group)) else {
            panic!("{group} was scored on one side only");
        };
        println!(
            "  {group:<15}  {:>7}   {:>12.4}   {:>6.4}   {}",
            is.queries,
            was.mean_ndcg(),
            is.mean_ndcg(),
            compare(&was.per_query, &is.per_query)
        );
    }
    println!("  group            queries   recall@{RECALL_AT} before  after   paired");
    for group in &names {
        let (was, is) = (&before[*group], &after[*group]);
        println!(
            "  {group:<15}  {:>7}   {:>12.4}   {:>6.4}   {}",
            is.queries,
            was.mean_recall(),
            is.mean_recall(),
            compare(&was.per_query_recall, &is.per_query_recall)
        );
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranking(topics: &[&str]) -> Vec<String> {
        topics.iter().map(|topic| topic.to_string()).collect()
    }

    #[test]
    fn the_perfect_ranking_scores_one() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["a", "b", "z"]), 2, |topic| topic != "z");
        assert!((scores.mean_ndcg() - 1.0).abs() < 1e-12);
        assert!((scores.mean_recall() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_ranking_that_found_nothing_scores_zero() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["x", "y"]), 2, |_| false);
        assert_eq!(scores.mean_ndcg(), 0.0);
        assert_eq!(scores.mean_recall(), 0.0);
    }

    /// The ideal is built from how many are relevant, not how many came back.
    ///
    /// Otherwise a query that returned one of five relevant results and put it
    /// first would score a perfect one, which is how a recall failure gets
    /// reported as a ranking success.
    #[test]
    fn missing_most_of_the_answer_is_not_a_perfect_ranking() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["a"]), 5, |topic| topic == "a");
        assert!(scores.mean_ndcg() < 0.4, "{}", scores.mean_ndcg());
        assert!((scores.mean_recall() - 0.2).abs() < 1e-12);
    }

    /// A query the corpus cannot grade scores zero rather than dividing by it.
    #[test]
    fn a_query_with_nothing_judged_relevant_is_not_a_free_point() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["a", "b"]), 0, |_| true);
        assert_eq!(scores.mean_ndcg(), 0.0);
        assert_eq!(scores.mean_recall(), 0.0);
    }

    /// A grade that is worth more belongs higher, and nDCG says so.
    ///
    /// The property binary relevance cannot express and the reason a graded
    /// corpus is worth having: two rankings that return the same two documents
    /// in opposite orders score the same under binary relevance and differently
    /// here.
    #[test]
    fn a_better_grade_belongs_higher() {
        let ideal = [3.0, 1.0];
        let gain = |key: &str| match key {
            "best" => 3.0,
            "fair" => 1.0,
            _ => 0.0,
        };

        let mut right = Scores::default();
        right.add_graded(&ranking(&["best", "fair"]), &ideal, gain);
        let mut wrong = Scores::default();
        wrong.add_graded(&ranking(&["fair", "best"]), &ideal, gain);

        assert!((right.mean_ndcg() - 1.0).abs() < 1e-12);
        assert!(
            wrong.mean_ndcg() < right.mean_ndcg(),
            "{} against {}",
            wrong.mean_ndcg(),
            right.mean_ndcg()
        );
        // And binary relevance cannot tell them apart, which is the point.
        let mut flat = Scores::default();
        flat.add(&ranking(&["fair", "best"]), 2, |key| gain(key) > 0.0);
        assert!((flat.mean_ndcg() - 1.0).abs() < 1e-12);
    }

    /// The binary arithmetic, pinned to a number rather than to itself.
    ///
    /// `add` is a call into `add_graded` now, so a test that compared the two
    /// could not fail -- it would be asserting that a function equals itself.
    /// What can fail is the arithmetic changing, and this is the value the
    /// three corpora's floors were set under: one relevant result at rank one
    /// out of five relevant in total is `1 / (sum over five of 1/log2(r+2))`.
    #[test]
    fn the_binary_arithmetic_is_what_the_floors_were_set_under() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["a", "x", "y"]), 5, |topic| topic == "a");
        assert!(
            (scores.mean_ndcg() - 0.339_160_2).abs() < 1e-6,
            "{}",
            scores.mean_ndcg()
        );
        assert!((scores.mean_recall() - 0.2).abs() < 1e-12);
    }

    /// What is saved reads back as the same values, per question and in
    /// order, so a pairing of two runs compares what each measured rather
    /// than a rounding of it.
    #[test]
    fn saved_scores_read_back_exactly() {
        let mut first = Scores::default();
        first.add(&ranking(&["a", "x", "y"]), 5, |topic| topic == "a");
        first.add(&ranking(&["x", "b"]), 3, |topic| topic == "b");
        let mut second = Scores::default();
        second.add(&ranking(&["x", "c"]), 1, |topic| topic == "c");
        let groups = std::collections::BTreeMap::from([
            ("first".to_string(), first),
            ("second".to_string(), second),
        ]);

        let dir = tempfile::tempdir().expect("temp dir");
        let path = saved(dir.path(), "project");
        save(&path, pamin_index::Passage::Named, &groups);
        let read = load(&path);

        assert_eq!(read.passage, "Named");
        assert_eq!(read.groups.len(), 2);
        for (group, scores) in &groups {
            let read = &read.groups[group];
            assert_eq!(read.queries, scores.queries);
            assert_eq!(read.per_query, scores.per_query);
            assert_eq!(read.per_query_recall, scores.per_query_recall);
            assert_eq!(read.mean_ndcg(), scores.mean_ndcg());
        }
    }

    /// A run over a project embedded from content is not paired with one
    /// embedded from names: the difference would be two changes, reported
    /// as one.
    #[test]
    #[should_panic(expected = "embedded Content passages and this one Named")]
    fn runs_over_different_encodings_are_not_paired() {
        let mut scores = Scores::default();
        scores.add(&ranking(&["a"]), 1, |topic| topic == "a");
        let groups = std::collections::BTreeMap::from([("group".to_string(), scores)]);

        let dir = tempfile::tempdir().expect("temp dir");
        let path = saved(dir.path(), "project");
        save(&path, pamin_index::Passage::Content, &groups);
        against("title", &load(&path), pamin_index::Passage::Named, &groups);
    }

    /// What is inside the shortlist but below rank ten is a reranker's to fix.
    #[test]
    fn relevant_results_below_the_cutoff_are_counted_as_recoverable() {
        let mut ranked = ranking(&["x"; 12]);
        ranked[11] = "a".to_string();

        let mut scores = Scores::default();
        scores.add(&ranked, 1, |topic| topic == "a");
        assert_eq!((scores.deep, scores.with_work), (1, 1));
        assert_eq!(scores.mean_ndcg(), 0.0, "and it is not in the top ten");
        assert_eq!(scores.mean_recall(), 1.0, "but it was returned");
    }
}
