use std::time::Instant;
use pamin_index::{Device, Rerank, Reranker};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "measures cached Accurate CoreML scoring on fixed candidates"]
fn fixed_candidate_rank() {
    let input = std::fs::read(std::env::var("PAD_INPUT").unwrap()).unwrap();
    assert_eq!(format!("{:x}", Sha256::digest(&input)), "cd17aba46410010832ce681008bd22a93e11d0a2b2cc1325f4b7a2e6328d7ac5");
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&input).unwrap();
    assert_eq!(rows.len(), 1190);
    assert!(std::env::var_os("HF_HOME").is_none());
    let _ = tracing_subscriber::fmt().with_writer(std::io::stdout).try_init();
    let models = std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").unwrap()).join("models");
    let mut model = Reranker::load(Rerank::Accurate, &models).unwrap();
    assert_eq!(model.device(), Device::CoreMl);
    assert_eq!(model.counted().maximum_tokens, 256);
    for i in 0..8 {
        let query = format!("measurement warmup {i}: migration failure");
        for document in ["a migration rolls back".to_owned(), "database migration rollback ".repeat(20), "database migration rollback ".repeat(300)] {
            model.rank(&query, &[document.as_str()]).unwrap();
        }
    }
    let mut results = Vec::new();
    for row in rows.iter().step_by(20) {
        let candidates = row["candidates"].as_array().unwrap();
        let documents: Vec<_> = candidates.iter().map(|value| value["shown"].as_str().unwrap()).collect();
        assert_eq!(documents.len(), 30);
        let before = model.counted();
        let start = Instant::now();
        let ranked = model.rank(row["text"].as_str().unwrap(), &documents).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let after = model.counted();
        assert_eq!(after.scored - before.scored, 30, "cached pair in measurement");
        let mut scores = vec![f32::NAN; 30];
        for scored in ranked { scores[scored.position] = scored.score; }
        assert!(scores.iter().all(|score| score.is_finite()));
        results.push(serde_json::json!({"id": row["id"], "seconds": seconds, "scores": scores, "encode_us": after.encode_us-before.encode_us, "forward_us": after.forward_us-before.forward_us, "batches":after.batches-before.batches,"padded_tokens":after.padded_tokens-before.padded_tokens}));
    }
    assert_eq!(results.len(), 60);
    std::fs::write(std::env::var("PAD_OUTPUT").unwrap(), serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}
