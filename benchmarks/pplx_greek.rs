//! Copied to ignored tests/scratch_pplx_greek.rs by the reproduction recipe.

use std::collections::HashSet;

use pamin_engine::{Depths, Engine, SearchHit};
use pamin_index::{Access, Profile, Rerank, VectorIndex};
use pamin_store::Workspace;

fn score(hits: &[SearchHit], relevant: &HashSet<&str>, drop: &HashSet<&str>) -> (f64, f64) {
    let kept: Vec<&str> = hits
        .iter()
        .map(|hit| hit.topic.as_str())
        .filter(|name| !drop.contains(name))
        .collect();
    let discount = |rank: usize| 1.0 / ((rank + 2) as f64).log2();
    let best: f64 = (0..relevant.len().min(10)).map(discount).sum();
    let gained: f64 = kept
        .iter()
        .take(10)
        .enumerate()
        .filter(|(_, name)| relevant.contains(**name))
        .map(|(rank, _)| discount(rank))
        .sum();
    let found = kept
        .iter()
        .take(50)
        .filter(|name| relevant.contains(**name))
        .count();
    (gained / best, found as f64 / relevant.len() as f64)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "opens existing BGE and pplx XQuAD-R indexes and reranks Greek queries"]
async fn greek_questions_on_the_shipped_search_path() {
    let home = std::path::PathBuf::from(std::env::var("PAMIN_EVAL_HOME").expect("evaluation home"));
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(home.join("pplx-xquad-manifest.json")).unwrap())
            .unwrap();
    let all = manifest["queries"].as_array().unwrap();
    assert_eq!(all.len(), 1_190);
    let greek: Vec<_> = all
        .iter()
        .filter(|query| query["language"] == "el")
        .collect();
    assert_eq!(greek.len(), 108);

    let workspace = Workspace::at(&home);
    let bge = Engine::open(
        &workspace,
        "xquad-accuracy-24ad7f1862182925",
        Profile::Accuracy,
        VectorIndex::Memory,
        Access::ReadWrite,
    )
    .await
    .unwrap();
    let pplx = Engine::open(
        &workspace,
        "xquad-pplx-24ad7f1862182925",
        Profile::Pplx,
        VectorIndex::Memory,
        Access::ReadWrite,
    )
    .await
    .unwrap();
    assert_eq!(bge.indexed_documents().unwrap(), 13_014);
    assert_eq!(pplx.indexed_documents().unwrap(), 13_014);
    assert_eq!(bge.passage(), pplx.passage(), "passage encoding differs");

    let depths = Depths {
        channel: 50,
        graph: 2,
    };
    let mut output = Vec::new();
    for query in greek {
        let text = query["text"].as_str().unwrap();
        let answers = query["answers"].as_object().unwrap();
        let own = answers["el"].as_str().unwrap();
        let translations: HashSet<&str> = answers
            .iter()
            .filter(|(language, _)| language.as_str() != "el")
            .map(|(_, key)| key.as_str().unwrap())
            .collect();
        assert_eq!(translations.len(), 10);
        let bge_hits = bge
            .search_reranked(text, 60, depths, Rerank::Accurate)
            .await
            .unwrap();
        let pplx_hits = pplx
            .search_reranked(text, 60, depths, Rerank::Accurate)
            .await
            .unwrap();
        let bge_cross = score(&bge_hits, &translations, &HashSet::from([own]));
        let pplx_cross = score(&pplx_hits, &translations, &HashSet::from([own]));
        let bge_same = score(&bge_hits, &HashSet::from([own]), &translations);
        let pplx_same = score(&pplx_hits, &HashSet::from([own]), &translations);
        output.push(serde_json::json!({
            "id": query["id"],
            "cross": {"bge": bge_cross, "pplx": pplx_cross},
            "same": {"bge": bge_same, "pplx": pplx_same},
        }));
    }
    for group in ["cross", "same"] {
        let bge: Vec<f64> = output
            .iter()
            .map(|row| row[group]["bge"][0].as_f64().unwrap())
            .collect();
        let pplx: Vec<f64> = output
            .iter()
            .map(|row| row[group]["pplx"][0].as_f64().unwrap())
            .collect();
        let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
        let wins = pplx
            .iter()
            .zip(&bge)
            .filter(|(new, old)| **new > **old + 1e-9)
            .count();
        let losses = pplx
            .iter()
            .zip(&bge)
            .filter(|(new, old)| **new + 1e-9 < **old)
            .count();
        println!(
            "Greek {group}: BGE {:.4}, pplx {:.4}, delta {:+.4}, W/L {wins}/{losses}",
            mean(&bge),
            mean(&pplx),
            mean(&pplx) - mean(&bge)
        );
    }
    std::fs::write(
        home.join("pplx-greek-paired.json"),
        serde_json::to_vec_pretty(&output).unwrap(),
    )
    .unwrap();
}
