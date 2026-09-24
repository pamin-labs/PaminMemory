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

use pamin_index::{Rerank, Reranker};

/// Every tier that loads a model, so a new one is measured only after it has
/// been shown to work at all.
const TIERS: &[Rerank] = &[Rerank::Fast, Rerank::Accurate];

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

/// Whatever device the load picks, it orders like the CPU.
///
/// On a GPU the reranker runs a different export in a different precision --
/// fp16 rather than int8 -- so its scores are not the CPU's and are not
/// expected to be. What must hold is the ordering on candidates the model
/// separates clearly, which is the only thing a search reads. On a machine
/// with no accelerator both loads land on the CPU and this checks that the
/// fallback runs the CPU's own export; on a GPU it is the equivalence check a
/// user of the accelerator is relying on.
///
/// One test, not two, because `PAMIN_DEVICE` is process-wide and a second
/// test flipping it in parallel would race this one.
#[test]
#[ignore = "downloads reranker model weights"]
fn every_device_orders_like_the_cpu() {
    let dir = tempfile::tempdir().expect("temp dir");
    let models = dir.path().join("models");

    let query = "how does the deployment pipeline handle a failed migration";
    let documents = [
        "The office coffee machine is descaled on the first Monday of each month.",
        "When a migration fails the deployment pipeline rolls the release back and \
         leaves the previous version serving.",
        "Deployments are frozen on Fridays after noon.",
        "Database migrations run before the new version takes traffic.",
    ];

    let order = |reranker: &mut Reranker| -> Vec<usize> {
        reranker
            .rank(query, &documents)
            .expect("score the candidates")
            .iter()
            .map(|ranked| ranked.position)
            .collect()
    };

    let mut chosen = Reranker::load(Rerank::Fast, &models).expect("load on any device");
    let chosen_order = order(&mut chosen);

    // SAFETY: nothing else in this binary reads the environment concurrently;
    // see the note above.
    unsafe { std::env::set_var("PAMIN_DEVICE", "cpu") };
    let mut cpu = Reranker::load(Rerank::Fast, &models).expect("load on the cpu");
    unsafe { std::env::remove_var("PAMIN_DEVICE") };

    assert_eq!(
        cpu.device(),
        pamin_index::Device::Cpu,
        "PAMIN_DEVICE=cpu was not honoured"
    );
    assert_eq!(
        chosen_order,
        order(&mut cpu),
        "the {} device ordered four clearly separated candidates differently from the cpu",
        chosen.device().name()
    );
    println!("  the build chose {}", chosen.device().name());
}
