//! Runs each reranker tier's real model.
//!
//! Ignored by default: the first run downloads weights. Run with
//! `cargo test -p pamin-index --test reranking -- --ignored --nocapture`.
//! On a host expected to accelerate, set `PAMIN_EXPECT_DEVICE=directml`,
//! `cuda` or `coreml` when running `every_device_orders_like_the_cpu` so a
//! failed accelerator load cannot pass unnoticed as a CPU-only comparison.
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
/// An accelerator runs the unquantized export while the CPU runs an int8
/// export, so their scores are not identical and are not
/// expected to be. What must hold is the ordering on candidates the model
/// separates clearly, which is the only thing a search reads. On a machine
/// with no accelerator both loads land on the CPU and this checks that the
/// fallback runs the CPU's own export; on a GPU it is the equivalence check a
/// user of the accelerator is relying on.
///
/// Each arm starts in its own process, before native inference threads exist.
/// `PAMIN_TEST_MODEL_CACHE` reuses an existing cache for both tiers and arms.
#[test]
#[ignore = "downloads reranker model weights"]
fn every_device_orders_like_the_cpu() {
    let temporary = tempfile::tempdir().expect("temp dir");
    let models = std::env::var_os("PAMIN_TEST_MODEL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().join("models"));

    if let Some(output) = std::env::var_os("PAMIN_RERANK_TEST_OUTPUT") {
        let name = std::env::var("PAMIN_RERANK_TEST_TIER").expect("parent supplies a tier");
        let tier = *TIERS
            .iter()
            .find(|tier| tier.name() == name)
            .expect("supported tier");
        let mut model = Reranker::load(tier, &models).expect("load reranker");
        if let Ok(expected) = std::env::var("PAMIN_EXPECT_DEVICE") {
            assert_eq!(
                model.device().name(),
                expected,
                "unexpected provider fallback"
            );
        }
        let documents = [
            "The office coffee machine is descaled on the first Monday of each month.",
            "When a migration fails the deployment pipeline rolls the release back and leaves the previous version serving.",
            "Deployments are frozen on Fridays after noon.",
            "Database migrations run before the new version takes traffic.",
        ];
        let ranked = model
            .rank(
                "how does the deployment pipeline handle a failed migration",
                &documents,
            )
            .expect("score candidates");
        assert_eq!(ranked.len(), documents.len());
        assert!(ranked.iter().all(|ranked| ranked.score.is_finite()));
        assert_eq!(ranked[0].position, 1, "the relevant document is not first");
        assert!(
            ranked[0].score > ranked[1].score,
            "a tie cannot establish model plumbing"
        );
        let order: Vec<_> = ranked.iter().map(|ranked| ranked.position).collect();
        std::fs::write(
            output,
            serde_json::to_vec(&(model.device().name(), order)).unwrap(),
        )
        .unwrap();
        return;
    }

    for tier in TIERS {
        let run = |cpu: bool| {
            let output = temporary
                .path()
                .join(if cpu { "cpu.json" } else { "chosen.json" });
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "every_device_orders_like_the_cpu",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                ])
                .env("PAMIN_RERANK_TEST_OUTPUT", &output)
                .env("PAMIN_RERANK_TEST_TIER", tier.name())
                .env("PAMIN_TEST_MODEL_CACHE", &models);
            if cpu {
                command
                    .env("PAMIN_DEVICE", "cpu")
                    .env("PAMIN_EXPECT_DEVICE", "cpu");
            }
            assert!(
                command.status().expect("start arm").success(),
                "{} arm failed for {}",
                if cpu { "CPU" } else { "chosen" },
                tier.name()
            );
            let result: (String, Vec<usize>) =
                serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
            assert_eq!(result.1.len(), 4);
            result
        };
        let chosen = run(false);
        let cpu = run(true);
        assert_eq!(
            chosen.1,
            cpu.1,
            "{} on {} ordered the candidates differently from CPU",
            tier.name(),
            chosen.0
        );
        println!("{} chose {} and orders like CPU", tier.name(), chosen.0);
    }
}

/// The short/long buckets must keep logical rows and physical work distinct.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "loads the accurate CoreML model; reuses PAMIN_TEST_MODEL_CACHE"]
fn coreml_buckets_preserve_partial_batches_and_work() {
    let temporary = tempfile::tempdir().unwrap();
    let cache = std::env::var_os("PAMIN_TEST_MODEL_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temporary.path().join("models"));
    let mut model = Reranker::load(Rerank::Accurate, &cache).expect("load accurate model");
    assert_eq!(
        model.device(),
        pamin_index::Device::CoreMl,
        "the premise fell back to CPU"
    );
    assert_eq!(
        model.counted().maximum_tokens,
        256,
        "unset token-limit sweep overrides"
    );
    let mut documents = vec![
        "short document one".to_string(),
        "short document two".to_string(),
        "short document three".to_string(),
    ];
    for index in 0..4 {
        documents.push(format!(
            "{} record {index}",
            "database migration rollback ".repeat(300)
        ));
    }
    let borrowed: Vec<_> = documents.iter().map(String::as_str).collect();
    let ranked = model
        .rank("failed migration", &borrowed)
        .expect("score both buckets");
    assert_eq!(
        ranked.len(),
        7,
        "physical padding rows escaped into results"
    );
    let positions: std::collections::BTreeSet<_> = ranked.iter().map(|r| r.position).collect();
    assert_eq!(positions, (0..7).collect());
    assert!(ranked.iter().all(|r| r.score.is_finite()));
    let work = model.counted();
    assert_eq!(work.scored, 7);
    assert_eq!(work.batches, 3);
    assert_eq!(
        work.padded_tokens,
        3 * 512,
        "work counts only logical padding rather than model input"
    );
    let alone = model
        .rank("another failed migration", &["one new short document"])
        .expect("single-row bucket");
    assert_eq!(alone.len(), 1);
    assert_eq!(model.counted().padded_tokens - work.padded_tokens, 512);
}
