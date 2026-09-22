//! Whether recording topic names in one statement matches doing it one at a time.
//!
//! Ignored by default: it provisions PostgreSQL. Run with
//! `cargo test -p pamin-store --test namebatch -- --ignored --nocapture`.
//!
//! A rebuild records every topic in the project, and it used to do that with a
//! round trip per topic -- thirteen thousand of them on the corpora this
//! project measures, against a table with no more rows than that. The batched
//! form is one `unnest` statement, which is the shape the *reads* on this table
//! already use.
//!
//! What this asserts is the only thing that makes the substitution safe: the
//! two forms leave the table in the same state, including the `ON CONFLICT`
//! path. That path is not incidental -- a rebuild runs against a table that
//! already has these rows, so the update branch is the one a rebuild takes
//! every time after the first.
//!
//! The failure this is really for is an `unnest` whose arrays disagree in
//! order. Three parallel arrays are zipped by position, and a batch that
//! recorded the right keys against the wrong topics would leave every row
//! populated and every name wrong -- so `widest_topic_name` would still answer,
//! `topics_named_by` would still match something, and mention derivation would
//! quietly derive edges to the wrong topics. Nothing downstream can notice
//! that, which is why it is checked here by name and not by row count.

use pamin_store::{Connections, Database, Workspace, repository};

/// Names with different token counts, so the widest is unambiguous, and one
/// with a non-ASCII key, because the column is text and the tokenizer above
/// this layer produces whatever the language gives it.
const NAMES: &[(&str, usize)] = &[
    ("deploy pipeline en", 3),
    ("oncall rota", 2),
    ("database backup policy en", 4),
    ("部署 流水线", 2),
];

async fn recorded(
    database: &Database,
    project: pamin_core::ProjectId,
) -> Vec<(uuid::Uuid, String, i16)> {
    sqlx::query_as(
        "SELECT topic_id, name_key, token_count FROM topic_name_tokens
          WHERE project_id = $1 ORDER BY name_key",
    )
    .bind(project.0)
    .fetch_all(database.pool())
    .await
    .expect("read the recorded names")
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn one_statement_records_what_one_per_topic_did() {
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };

    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    // Two projects rather than two runs against one, so the comparison is a
    // comparison and not a sequence: the batched write must not be reading
    // anything the row-at-a-time write left behind.
    let singly = repository::ensure_project(
        database.pool(),
        &format!("namebatch-single-{}", uuid::Uuid::new_v4()),
    )
    .await
    .expect("ensure project")
    .id;
    let batched = repository::ensure_project(
        database.pool(),
        &format!("namebatch-batch-{}", uuid::Uuid::new_v4()),
    )
    .await
    .expect("ensure project")
    .id;

    // The same topic identifiers in both projects, so the rows can be compared
    // by identity rather than by position.
    let topics: Vec<pamin_core::TopicId> = (0..NAMES.len())
        .map(|_| pamin_core::TopicId::from(uuid::Uuid::new_v4()))
        .collect();

    for (topic, (key, tokens)) in topics.iter().zip(NAMES) {
        repository::record_topic_name(database.pool(), singly, *topic, key, *tokens)
            .await
            .expect("record one name");
    }

    let names: Vec<(pamin_core::TopicId, String, usize)> = topics
        .iter()
        .zip(NAMES)
        .map(|(topic, (key, tokens))| (*topic, (*key).to_string(), *tokens))
        .collect();
    repository::record_topic_names(database.pool(), batched, &names)
        .await
        .expect("record the names in one statement");

    assert_eq!(
        recorded(&database, singly).await,
        recorded(&database, batched).await,
        "the batched write recorded something different from the row-at-a-time write"
    );

    // And the update branch, which is the one a rebuild takes every time after
    // the first. Re-recording with a changed key has to overwrite rather than
    // conflict, and has to do it for every row of the batch.
    let renamed: Vec<(pamin_core::TopicId, String, usize)> = names
        .iter()
        .map(|(topic, key, tokens)| (*topic, format!("{key} again"), tokens + 1))
        .collect();
    for (topic, key, tokens) in &renamed {
        repository::record_topic_name(database.pool(), singly, *topic, key, *tokens)
            .await
            .expect("re-record one name");
    }
    repository::record_topic_names(database.pool(), batched, &renamed)
        .await
        .expect("re-record the names in one statement");

    let after = recorded(&database, batched).await;
    assert_eq!(
        recorded(&database, singly).await,
        after,
        "the batched write took a different conflict path from the row-at-a-time write"
    );
    assert!(
        after.iter().all(|(_, key, _)| key.ends_with("again")),
        "the conflict clause did not update every row of the batch: {after:?}"
    );

    // Empty is a real call: a project with no topics rebuilds to no names, and
    // an `unnest` of three empty arrays must be a no-op rather than an error.
    repository::record_topic_names(database.pool(), batched, &[])
        .await
        .expect("an empty batch is a no-op");
    assert_eq!(recorded(&database, batched).await, after);

    println!("  the batched write matches the row-at-a-time write, insert and update");
}
