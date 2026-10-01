//! End-to-end invariant: optional dense spaces survive write/query/reindex.
mod harness;
use pamin_core::{Channel, Why};
use pamin_engine::{Depths, Engine};
use pamin_index::{Access, Profile, Rerank, VectorIndex};
use pamin_store::Workspace;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "uses complete cached models and a live evaluation PostgreSQL workspace"]
async fn dual_profile_survives_real_write_search_and_reindex() {
    let workspace = Workspace::at(harness::eval_home());
    let project = format!(
        "dual-lifecycle-{}",
        time::OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let engine = Engine::open(
        &workspace,
        &project,
        Profile::DualAccuracy,
        VectorIndex::Memory,
        Access::ReadWrite,
    )
    .await
    .unwrap();
    for (topic, text) in [
        (
            "rollback policy",
            "A failed migration rolls back the deployment transaction.",
        ),
        (
            "office supplies",
            "The office coffee machine needs descaling.",
        ),
        ("数据库回滚", "数据库迁移失败时回滚事务，并恢复旧版本。"),
    ] {
        harness::write_absent(&engine, topic, text, "eng", "dual lifecycle invariant").await;
    }
    harness::drain(&engine).await;
    assert_eq!(engine.indexed_documents().unwrap(), 3);
    let query = "How does a failed migration roll back?";
    let before = engine
        .search_reranked(query, 10, Depths::default(), Rerank::Accurate)
        .await
        .unwrap();
    assert!(!before.is_empty());
    assert!(
        before
            .iter()
            .any(|hit| hit.result.why.iter().any(|why| matches!(
                why,
                Why::Channel {
                    channel: Channel::VectorSecondary,
                    ..
                }
            ))),
        "second vector space was not queried"
    );
    let topics: Vec<_> = before.iter().map(|hit| hit.topic.clone()).collect();
    engine.reindex().await.unwrap();
    let after = engine
        .search_reranked(query, 10, Depths::default(), Rerank::Accurate)
        .await
        .unwrap();
    assert_eq!(
        after
            .iter()
            .map(|hit| hit.topic.clone())
            .collect::<Vec<_>>(),
        topics
    );
    assert_eq!(engine.indexed_documents().unwrap(), 3);
}
