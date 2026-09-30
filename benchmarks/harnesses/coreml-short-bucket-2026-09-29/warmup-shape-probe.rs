use pamin_index::{Device, Rerank, Reranker};
use sha2::{Digest, Sha256};

#[test]
#[ignore = "loads the pinned Accurate model to inspect actual warmup token lengths"]
fn original_warmups_reach_each_bucket() {
    let models = std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").unwrap()).join("models");
    let tokenizer = models.join("models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/tokenizer.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(std::fs::read(tokenizer).unwrap())),
        "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3"
    );
    let mut reranker = Reranker::load(Rerank::Accurate, &models).unwrap();
    assert_eq!(reranker.device(), Device::CoreMl);
    let cases = [
        ("short", "a migration rolls back".to_string(), 1..=64, 256),
        ("medium", "database migration rollback ".repeat(20), 65..=128, 512),
        ("long", "database migration rollback ".repeat(300), 129..=256, 512),
    ];
    for i in 0..8 {
        let query = format!("measurement warmup {i}: migration failure");
        for (name, document, interval, physical) in &cases {
            let before = reranker.counted();
            reranker.rank(&query, &[document.as_str()]).unwrap();
            let after = reranker.counted();
            let logical = after.tokens - before.tokens;
            let padded = after.padded_tokens - before.padded_tokens;
            assert_eq!(after.scored - before.scored, 1, "{name} was cached");
            assert!(interval.contains(&logical), "{name} had {logical} tokens");
            assert_eq!(padded, *physical, "{name} reached the wrong bucket");
            println!("{name} {i}: logical={logical} physical={padded}");
        }
    }
}
