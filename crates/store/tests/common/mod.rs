//! Helpers more than one of the store's test files needs.

// Each including test uses some of it.
#![allow(dead_code)]

/// The `PAMIN_MEMORY_EVAL_*` knobs, shared with every crate's tests.
#[path = "../../../index/tests/support/eval.rs"]
pub mod eval;

/// The store settings a test measures: the defaults, and `PAMIN_MEMORY_EVAL_JIT` and
/// `PAMIN_MEMORY_EVAL_POSTGRES_DIR`.
pub fn store() -> memory_store::StoreSettings {
    eval::check();
    memory_store::StoreSettings {
        jit: eval::switch("PAMIN_MEMORY_EVAL_JIT").unwrap_or_default(),
        postgres_dir: eval::path("PAMIN_MEMORY_EVAL_POSTGRES_DIR"),
        database_url: None,
        database_connections: None,
    }
}

use memory_core::{ProjectId, Topic, TopicId};
use memory_store::repository;

/// Returns the topic with this name, creating it if it does not exist, and
/// files no name for it.
///
/// The product creates a topic only inside a write, and files its name in the
/// same statement (`repository::create_topic_named`). These tests need topics
/// with no memory under them -- the ends of an edge, the subject of a queued
/// job -- and several record names themselves, in forms a name filed here
/// would be mistaken for. So the insert lives beside them rather than in the
/// store's API, which has no other caller for it.
pub async fn ensure_topic(
    connection: &mut sqlx::PgConnection,
    project: ProjectId,
    name: &str,
) -> memory_store::Result<Topic> {
    if let Some(topic) = repository::find_topic(&mut *connection, project, name).await? {
        return Ok(topic);
    }
    sqlx::query(
        "INSERT INTO topics (id, project_id, name, created_at)
         VALUES ($1, $2, $3, now())
         ON CONFLICT (project_id, name) DO NOTHING",
    )
    .bind(TopicId::new().0)
    .bind(project.0)
    .bind(name)
    .execute(&mut *connection)
    .await?;
    Ok(repository::find_topic(&mut *connection, project, name)
        .await?
        .expect("the topic was just created, by this call or another"))
}
