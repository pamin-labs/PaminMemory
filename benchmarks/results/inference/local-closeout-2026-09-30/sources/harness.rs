
#[tokio::test(flavor = "multi_thread")]
#[ignore = "scratch shared-machine backend comparison at product entry point"]
async fn scratch_backend_repeat() {
    assert!(std::env::var_os("HF_HOME").is_none());
    report_loaded_graphs();
    let corpus = Corpus::load();
    let queries = corpus.queries();
    assert_eq!(queries.len(),1190);
    let workspace=Workspace::at(std::env::var("PAMIN_EVAL_HOME").unwrap());
    let identity=pinned_embedding_id(&workspace.root().join("models"));
    let project=xquad_project_name("accuracy",&corpus.fingerprint(),&identity);
    let engine=Engine::open(&workspace,&project,pamin_index::Profile::Accuracy,VectorIndex::Memory,Access::ReadWrite).await.unwrap();
    assert_eq!(engine.indexed_documents().unwrap(),13014);
    assert_eq!(engine.vector_index_completeness().unwrap(),1.0);
    let tier=if std::env::var("PAMIN_MATRIX_TIER").as_deref()==Ok("fast") {Rerank::Fast} else {Rerank::Accurate};
    // Fixed sample and disjoint warmups chosen before looking at timings.
    for index in 1..9 {
        let hits=engine.search_reranked(queries[index].text(),60,DEPTHS,tier).await.unwrap();
        assert!(!hits.is_empty());
    }
    let mut rows=Vec::new();
    for index in (0..1190).step_by(50) {
        let before=engine.reranked(tier).unwrap();
        let start=std::time::Instant::now();
        let hits=engine.search_reranked(queries[index].text(),60,DEPTHS,tier).await.unwrap();
        let seconds=start.elapsed().as_secs_f64();
        let after=engine.reranked(tier).unwrap();
        let offered=after.offered-before.offered;
        let scored=after.scored-before.scored;
        assert!(offered>0 && scored==offered,"reranking must be a cache miss");
        assert_eq!(after.maximum_tokens,256);
        let ranked:Vec<String>=hits.into_iter().map(|hit|hit.topic).collect();
        assert!(!ranked.is_empty());
        let mut scores=BTreeMap::new();score(&mut scores,&queries[index],&ranked);
        rows.push(serde_json::json!({"query":index,"seconds":seconds,"ranked":ranked,"scores":scores,"offered":offered,"scored":scored,"encode_us":after.encode_us-before.encode_us,"forward_us":after.forward_us-before.forward_us}));
        println!("backend sample {}/24",rows.len());
        std::fs::write(std::env::var("PAMIN_MATRIX_OUTPUT").unwrap(),serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    }
    assert_eq!(rows.len(),24);
}
