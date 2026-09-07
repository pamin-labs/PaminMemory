//! Drives the projection index against the real engine.

use pamin_core::TopicStateId;
use pamin_index::{Profile, ProjectionIndex};

const PROFILE: Profile = Profile::Speed;

/// A stand-in embedding. These tests exercise the lexical channels, so the
/// vector only has to be the right width.
fn stub() -> Vec<f32> {
    vec![0.1; PROFILE.dimensions() as usize]
}

fn id(byte: u8) -> TopicStateId {
    TopicStateId(uuid::Uuid::from_bytes([byte; 16]))
}

#[test]
fn lexical_recall_works_across_languages_and_on_exact_strings() {
    let dir = tempfile::tempdir().expect("temp dir");
    let index =
        ProjectionIndex::open(dir.path(), &dir.path().join("legacy"), PROFILE).expect("open index");

    let english = id(1);
    let chinese = id(2);
    let japanese = id(3);
    let thai = id(4);
    let identifier = id(5);

    index
        .upsert(english, "the deployment pipeline runs on ci", &stub())
        .expect("upsert english");
    index
        .upsert(chinese, "部署流水线运行在持续集成上", &stub())
        .expect("upsert chinese");
    index
        .upsert(
            japanese,
            "デプロイパイプラインは東京で動いています",
            &stub(),
        )
        .expect("upsert japanese");
    index
        .upsert(thai, "ท่อการปรับใช้ทำงานอยู่", &stub())
        .expect("upsert thai");
    index
        .upsert(
            identifier,
            "see crates/pamin-store/src/database.rs for error E1234",
            &stub(),
        )
        .expect("upsert identifier");
    index.flush().expect("flush");

    // Each language is searched in its own words, which is the whole point of
    // segmenting before indexing rather than falling back to n-grams.
    let hits = index.recall_segmented("deployment", 10).expect("english");
    assert!(hits.contains(&english), "english recall failed: {hits:?}");

    let hits = index.recall_segmented("流水线", 10).expect("chinese");
    assert!(hits.contains(&chinese), "chinese recall failed: {hits:?}");

    let hits = index.recall_segmented("東京", 10).expect("japanese");
    assert!(hits.contains(&japanese), "japanese recall failed: {hits:?}");

    let hits = index.recall_segmented("ทำงาน", 10).expect("thai");
    assert!(hits.contains(&thai), "thai recall failed: {hits:?}");

    // The n-gram field catches substrings of a path or an error code, which
    // word segmentation splits apart.
    let hits = index.recall_ngram("database.rs", 10).expect("path");
    assert!(hits.contains(&identifier), "path recall failed: {hits:?}");

    let hits = index.recall_ngram("E1234", 10).expect("error code");
    assert!(
        hits.contains(&identifier),
        "error code recall failed: {hits:?}"
    );

    assert!(
        index.recall_segmented("   ", 10).expect("blank").is_empty(),
        "a blank query should match nothing rather than everything"
    );
}

#[test]
fn discarding_the_directory_leaves_an_empty_index() {
    let dir = tempfile::tempdir().expect("temp dir");

    let index =
        ProjectionIndex::open(dir.path(), &dir.path().join("legacy"), PROFILE).expect("open index");
    index
        .upsert(id(1), "the deployment pipeline", &stub())
        .expect("upsert");
    index.flush().expect("flush");
    assert!(!index.recall_segmented("deployment", 10).unwrap().is_empty());
    drop(index);

    ProjectionIndex::discard(dir.path()).expect("discard");

    let rebuilt = ProjectionIndex::open(dir.path(), &dir.path().join("legacy"), PROFILE)
        .expect("reopen index");
    assert!(
        rebuilt
            .recall_segmented("deployment", 10)
            .unwrap()
            .is_empty(),
        "a discarded index must come back empty, ready to rebuild from postgres"
    );
}

#[test]
fn a_pre_split_workspace_is_reported_rather_than_searched() {
    // Before projects had their own directory there was one shared collection.
    // Opening a project's empty directory beside it would return nothing and
    // look like an empty workspace, which is the worst of the three outcomes:
    // wrong, silent, and indistinguishable from correct.
    let dir = tempfile::tempdir().expect("temp dir");
    let legacy = dir.path().join("legacy");
    std::fs::create_dir_all(&legacy).expect("legacy layout");

    let opened = ProjectionIndex::open(&dir.path().join("project"), &legacy, PROFILE);
    let Err(error) = opened else {
        panic!("a shared layout must not be opened silently");
    };
    assert!(
        error.to_string().contains("reindex"),
        "the error has to say how to fix it, got {error}"
    );
}
