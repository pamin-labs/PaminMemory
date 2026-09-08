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

/// A second command waits for the index instead of being turned away.
///
/// The engine takes the collection's file lock exclusively and non-blocking, so
/// a second opener is refused rather than queued. Agents run this CLI
/// concurrently by design, and a refusal turns an ordinary overlap into a
/// failed command.
///
/// The waiting side runs on the spawned thread because an open collection is
/// not `Send`, so the one being held has to stay where it was opened.
#[test]
fn a_second_opener_waits_for_the_index_rather_than_failing() {
    const HELD_FOR: std::time::Duration = std::time::Duration::from_millis(150);

    let dir = tempfile::tempdir().expect("temp dir");
    let legacy = dir.path().join("legacy");
    let held = ProjectionIndex::open(dir.path(), &legacy, PROFILE).expect("open index");

    let waiting = {
        let dir = dir.path().to_path_buf();
        let legacy = legacy.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let opened = ProjectionIndex::open(&dir, &legacy, PROFILE).is_ok();
            (opened, started.elapsed())
        })
    };

    // Released while the second opener is still retrying, which is what makes
    // this a test of waiting rather than of the deadline.
    std::thread::sleep(HELD_FOR);
    drop(held);

    let (opened, waited) = waiting.join().expect("waiting opener");
    assert!(
        opened,
        "a second opener should wait for the lock, not be refused"
    );
    assert!(
        waited >= HELD_FOR,
        "the second open returned before the first released the lock"
    );
}
