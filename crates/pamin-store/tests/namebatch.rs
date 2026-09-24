//! Whether recording topic names in one statement records each name against
//! its own topic.
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
//! What this asserts is the only thing that makes it safe: every row holds the
//! key and count given for its own topic, including through the `ON CONFLICT`
//! path. That path is not incidental -- a rebuild runs against a table that
//! already has these rows, so the update branch is the one a rebuild takes
//! every time after the first. It used to compare against the one-row-at-a-time
//! form, which is gone: a write now files its name with the topic, in the
//! statement that creates it. The expected rows are written out instead.
//!
//! The failure this is really for is an `unnest` whose arrays disagree in
//! order. Three parallel arrays are zipped by position, and a batch that
//! recorded the right keys against the wrong topics would leave every row
//! populated and every name wrong -- so `widest_topic_name` would still answer,
//! `topics_named_by` would still match something, and mention derivation would
//! quietly derive edges to the wrong topics. Nothing downstream can notice
//! that, which is why it is checked here by name and not by row count.

mod common;

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

/// One topic per name in `NAMES`, in that order.
///
/// `topic_name_tokens` references `topics`, so a name can only be recorded
/// for a topic that exists, and a topic's identifier is unique across
/// projects -- each project needs topics of its own.
async fn topics_in(
    database: &Database,
    project: pamin_core::ProjectId,
) -> Vec<pamin_core::TopicId> {
    let mut connection = database.pool().acquire().await.expect("connection");
    let mut topics = Vec::new();
    for (name, _) in NAMES {
        let topic = common::ensure_topic(&mut connection, project, name)
            .await
            .expect("create the topic");
        topics.push(topic.id);
    }
    topics
}

/// The rows recorded for `project`, each topic given as its position in
/// `topics` and in that order, so a key recorded against the wrong topic reads
/// back at the wrong position.
async fn recorded(
    database: &Database,
    project: pamin_core::ProjectId,
    topics: &[pamin_core::TopicId],
) -> Vec<(usize, String, i16)> {
    let rows: Vec<(uuid::Uuid, String, i16)> = sqlx::query_as(
        "SELECT topic_id, name_key, token_count FROM topic_name_tokens
          WHERE project_id = $1",
    )
    .bind(project.0)
    .fetch_all(database.pool())
    .await
    .expect("read the recorded names");
    let mut rows = rows
        .into_iter()
        .map(|(topic, key, tokens)| {
            let position = topics
                .iter()
                .position(|candidate| candidate.0 == topic)
                .expect("every recorded topic is one of the project's");
            (position, key, tokens)
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|(position, _, _)| *position);
    rows
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres"]
async fn one_statement_records_each_name_against_its_topic() {
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

    let batched = repository::ensure_project(
        database.pool(),
        &format!("namebatch-batch-{}", uuid::Uuid::now_v7()),
    )
    .await
    .expect("ensure project")
    .id;
    let batched_topics = topics_in(&database, batched).await;
    assert!(
        recorded(&database, batched, &batched_topics)
            .await
            .is_empty(),
        "the topics came with names, so the insert branch would not be tested"
    );

    // What each position should hold.
    let expected = |names: &[(pamin_core::TopicId, String, usize)]| {
        names
            .iter()
            .enumerate()
            .map(|(position, (_, key, tokens))| (position, key.clone(), *tokens as i16))
            .collect::<Vec<_>>()
    };

    let names: Vec<(pamin_core::TopicId, String, usize)> = batched_topics
        .iter()
        .zip(NAMES)
        .map(|(topic, (key, tokens))| (*topic, (*key).to_string(), *tokens))
        .collect();
    repository::record_topic_names(database.pool(), batched, &names)
        .await
        .expect("record the names in one statement");

    assert_eq!(
        recorded(&database, batched, &batched_topics).await,
        expected(&names),
        "the batched write recorded a name against some other topic"
    );

    // And the update branch, which is the one a rebuild takes every time after
    // the first. Re-recording with a changed key has to overwrite rather than
    // conflict, and has to do it for every row of the batch.
    let renamed: Vec<(pamin_core::TopicId, String, usize)> = names
        .iter()
        .map(|(topic, key, tokens)| (*topic, format!("{key} again"), tokens + 1))
        .collect();
    repository::record_topic_names(database.pool(), batched, &renamed)
        .await
        .expect("re-record the names in one statement");

    let after = recorded(&database, batched, &batched_topics).await;
    assert_eq!(
        after,
        expected(&renamed),
        "the batched write's conflict path left a row it should have updated"
    );

    // Empty is a real call: a project with no topics rebuilds to no names, and
    // an `unnest` of three empty arrays must be a no-op rather than an error.
    repository::record_topic_names(database.pool(), batched, &[])
        .await
        .expect("an empty batch is a no-op");
    assert_eq!(recorded(&database, batched, &batched_topics).await, after);

    println!("  the batched write records each name against its topic, insert and update");
}
