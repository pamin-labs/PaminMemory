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
#[derive(Default)]
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
        let hit = |rank: usize| relevant(ranked[rank].as_str());

        let gained: f64 = (0..ranked.len().min(NDCG_AT))
            .filter(|rank| hit(*rank))
            .map(|rank| 1.0 / ((rank + 2) as f64).log2())
            .sum();
        let ideal: f64 = (0..judged.min(NDCG_AT))
            .map(|rank| 1.0 / ((rank + 2) as f64).log2())
            .sum();

        let found = (0..ranked.len().min(RECALL_AT))
            .filter(|rank| hit(*rank))
            .count();
        let deep = (NDCG_AT..ranked.len().min(RECALL_AT))
            .filter(|rank| hit(*rank))
            .count();

        self.queries += 1;
        let ndcg = if ideal == 0.0 { 0.0 } else { gained / ideal };
        self.ndcg += ndcg;
        self.per_query.push(ndcg);
        self.recall += if judged == 0 {
            0.0
        } else {
            found as f64 / judged as f64
        };
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
