use std::collections::BTreeMap;
use std::io::Write;
use std::sync::{Arc, Mutex};

use pamin_index::{Device, Rerank, Reranker};
use sha2::{Digest, Sha256};

struct LogWriter(Arc<Mutex<Vec<u8>>>);

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
#[ignore = "loads the pinned Accurate model to inspect actual warmup token lengths"]
fn original_warmups_reach_each_bucket() {
    let models = std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").unwrap()).join("models");
    assert!(std::env::var_os("HF_HOME").is_none(), "unset HF_HOME");
    assert!(
        std::env::var_os("HF_ENDPOINT").is_none(),
        "unset HF_ENDPOINT"
    );
    let repository = models.join("models--onnx-community--bge-reranker-v2-m3-ONNX");
    let revision = "6f5ff65298512715a1e669753bc754d2bc8f367b";
    let selected = repository.join("refs/main");
    assert_eq!(std::fs::read_to_string(&selected).unwrap().trim(), revision);
    let tokenizer = repository
        .join("snapshots")
        .join(revision)
        .join("tokenizer.json");
    let tokenizer_hash = || format!("{:x}", Sha256::digest(std::fs::read(&tokenizer).unwrap()));
    assert_eq!(
        tokenizer_hash(),
        "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3"
    );
    let logs = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&logs);
    tracing_subscriber::fmt()
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || LogWriter(Arc::clone(&writer)))
        .try_init()
        .unwrap();
    let mut reranker = Reranker::load(Rerank::Accurate, &models).unwrap();
    assert_eq!(reranker.device(), Device::CoreMl);
    assert_eq!(reranker.counted().maximum_tokens, 256);
    assert_eq!(std::fs::read_to_string(&selected).unwrap().trim(), revision);
    assert_eq!(
        tokenizer_hash(),
        "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3"
    );
    let recorded = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
    let assignments: Vec<BTreeMap<String, usize>> = recorded
        .lines()
        .filter_map(|line| line.split_once("assigned_nodes=").map(|(_, nodes)| nodes))
        .map(|nodes| serde_json::from_str(nodes).unwrap())
        .collect();
    assert_eq!(
        assignments.len(),
        3,
        "expected three static scoring sessions: {recorded}"
    );
    for (index, nodes) in assignments.iter().enumerate() {
        assert!(
            nodes.get("CoreMLExecutionProvider").copied().unwrap_or(0) > 0,
            "static scoring session {index} has no CoreML nodes: {nodes:?}"
        );
        println!("CoreML scoring session {index}: {nodes:?}");
    }
    let cases = [
        ("short", "a migration rolls back".to_string(), 1..=64, 256),
        (
            "medium",
            "database migration rollback ".repeat(20),
            65..=128,
            512,
        ),
        (
            "long",
            "database migration rollback ".repeat(300),
            129..=256,
            512,
        ),
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
