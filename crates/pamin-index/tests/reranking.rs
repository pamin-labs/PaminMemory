//! Runs each reranker tier's real model.
//!
//! Ignored by default: the first run downloads weights. Run with
//! `cargo test -p pamin-index --test reranking -- --ignored --nocapture`.
//!
//! What this is for is narrower than accuracy, and worth stating because the
//! accuracy harnesses are elsewhere and are expensive. A tier is six matched
//! arms plus a repository name plus an export filename, and every part of that
//! is a string the compiler cannot check. A wrong export name fails loudly at
//! download; a *loadable* export whose graph the runtime cannot drive fails at
//! the first score, and an export that loads and scores but was never the
//! reranker anyone meant fails nowhere at all.
//!
//! So this asks each tier the one question that separates those three: given a
//! query and two documents, one relevant and one not, does the model put the
//! relevant one first. That is a check on the plumbing, not a measurement of
//! the model -- a reranker that cannot do this on an unambiguous pair is
//! misconfigured rather than weak -- and it is the premise every figure the
//! corpora report is standing on.

use pamin_index::{Licence, Rerank, Reranker};

/// Every tier that loads a model, so a new one is measured only after it has
/// been shown to work at all.
const TIERS: &[Rerank] = &[
    Rerank::Fast,
    Rerank::Accurate,
    Rerank::Balanced,
    Rerank::Typed,
];

#[test]
#[ignore = "downloads reranker model weights"]
fn every_tier_puts_the_relevant_document_first() {
    let dir = tempfile::tempdir().expect("temp dir");

    let query = "how does the deployment pipeline handle a failed migration";
    let relevant = "When a migration fails the deployment pipeline rolls the \
                    release back and leaves the previous version serving.";
    let unrelated = "The office coffee machine is descaled on the first Monday \
                     of each month.";

    for tier in TIERS {
        let mut reranker =
            Reranker::load(*tier, &dir.path().join("models")).expect("load the reranker");

        // Both orders, because a model whose output is being read off the
        // wrong axis would pass one of them by luck. Passing both means the
        // scores follow the documents rather than the positions.
        for documents in [[relevant, unrelated], [unrelated, relevant]] {
            let ranked = reranker
                .rank(query, &documents)
                .expect("score the candidates");

            assert_eq!(
                ranked.len(),
                documents.len(),
                "the {} tier returned {} of {} candidates",
                tier.name(),
                ranked.len(),
                documents.len()
            );
            assert_eq!(
                documents[ranked[0].position],
                relevant,
                "the {} tier ranked the unrelated document first, given {documents:?} -- \
                 the export loads and scores, so this is a wiring question rather than a \
                 quality one",
                tier.name()
            );

            // The order now carries the scores that produced it, so the
            // stronger claim is available for free: the ordering follows a
            // real separation rather than the stable tie-break. A model
            // scoring both candidates identically would satisfy the assertion
            // above by input order alone.
            assert!(
                ranked[0].score > ranked[1].score,
                "the {} tier scored both documents the same ({} and {}), so the order \
                 above came from the tie-break rather than from the model",
                tier.name(),
                ranked[0].score,
                ranked[1].score
            );
        }

        println!("  {} scores the pair the right way round", tier.name());
    }
}

/// The non-commercial tier is not in `TIERS` and this says why in a test.
///
/// It downloads weights a commercial user may not use, so it is not swept up
/// by a loop over "every tier that loads a model". Anything added to `TIERS`
/// has to be permissive, and this is the assertion that makes that a rule
/// rather than a habit.
#[test]
fn nothing_in_the_download_sweep_is_non_commercial() {
    for tier in TIERS {
        assert_eq!(
            tier.licence(),
            Some(Licence::Permissive),
            "{} is in the sweep and is not permissive",
            tier.name()
        );
    }
}
