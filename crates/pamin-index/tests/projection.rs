//! Drives the projection index against the real engine.

use pamin_core::TopicStateId;
use pamin_index::{Access, Profile, Projection, ProjectionIndex};

const PROFILE: Profile = Profile::Speed;

/// A stand-in embedding. These tests exercise the lexical channels, so the
/// vector only has to be the right width.
fn stub() -> Vec<f32> {
    vec![0.1; PROFILE.dimensions() as usize]
}

fn id(byte: u8) -> TopicStateId {
    TopicStateId(uuid::Uuid::from_bytes([byte; 16]))
}

/// A distinct identifier per number, for the tests that write many documents.
fn numbered(n: u128) -> TopicStateId {
    TopicStateId(uuid::Uuid::from_u128(n))
}

#[test]
fn lexical_recall_works_across_languages_and_on_exact_strings() {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
    )
    .expect("open index");

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

    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
    )
    .expect("open index");
    index
        .upsert(id(1), "the deployment pipeline", &stub())
        .expect("upsert");
    index.flush().expect("flush");
    assert!(!index.recall_segmented("deployment", 10).unwrap().is_empty());
    drop(index);

    ProjectionIndex::discard(dir.path()).expect("discard");

    let rebuilt = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
    )
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

    let opened = ProjectionIndex::open(
        &dir.path().join("project"),
        &legacy,
        PROFILE,
        Access::ReadWrite,
    );
    let Err(error) = opened else {
        panic!("a shared layout must not be opened silently");
    };
    assert!(
        error.to_string().contains("reindex"),
        "the error has to say how to fix it, got {error}"
    );
}

/// Two searches can hold the index at the same time.
///
/// Every open used to be read-write, so the engine took the directory lock
/// exclusively and two agents searching one project at the same time meant one
/// search and one wait -- for a pair of commands that write nothing. A shared
/// lock is what the read path actually needs.
#[test]
fn two_readers_hold_the_index_at_once() {
    let dir = tempfile::tempdir().expect("temp dir");
    let legacy = dir.path().join("legacy");

    let first = ProjectionIndex::open(dir.path(), &legacy, PROFILE, Access::ReadOnly)
        .expect("first reader");
    let second = ProjectionIndex::open(dir.path(), &legacy, PROFILE, Access::ReadOnly)
        .expect("a second reader should not have to wait for the first");

    // Both are usable, not merely open.
    for reader in [&first, &second] {
        assert!(
            reader
                .recall_segmented("anything", 1)
                .expect("recall")
                .is_empty(),
            "an empty index should return nothing rather than fail"
        );
    }
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
    let held =
        ProjectionIndex::open(dir.path(), &legacy, PROFILE, Access::ReadWrite).expect("open index");

    let waiting = {
        let dir = dir.path().to_path_buf();
        let legacy = legacy.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let opened = ProjectionIndex::open(&dir, &legacy, PROFILE, Access::ReadWrite).is_ok();
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

/// Everything written is still recallable after the vector index is built.
///
/// Building the graph rewrites the vector storage, and zvec has an open report
/// -- alibaba/zvec#724, against 0.6 and 0.7 -- of that step dropping the last
/// documents of a collection. What makes it worth a standing test rather than a
/// note is how it fails: the dropped documents keep appearing in the document
/// count and in scalar reads, so only a vector query can tell, and re-running
/// the build does not bring them back.
///
/// It does not reproduce here, across five rounds of write-delete-build in the
/// shape the cascade produces. That is a reason to call `optimize`, not a
/// reason to stop checking: this is the gate that says so on every run and on
/// every version of the engine we upgrade to.
#[test]
fn building_the_vector_index_loses_nothing() {
    const BATCH: usize = 400;
    const ROUNDS: usize = 5;

    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
    )
    .expect("open index");

    let mut live: Vec<u128> = Vec::new();
    let mut written = 0u128;

    for round in 0..ROUNDS {
        for _ in 0..BATCH {
            written += 1;
            index
                .upsert(
                    numbered(written),
                    &format!("memory number {written}"),
                    &separated(written),
                )
                .expect("upsert");
            live.push(written);
        }
        index.flush().expect("flush");
        index.optimize().expect("build the vector index");

        assert_eq!(
            index.vector_index_completeness().expect("completeness"),
            1.0,
            "round {round} left part of the collection outside the vector index"
        );

        // Checked by vector rather than by count: the failure this guards
        // against leaves the count right and the vector index wrong.
        let unreachable: Vec<u128> = live
            .iter()
            .copied()
            .filter(|written| {
                !index
                    // Three rather than one: search is approximate, and this
                    // is asking whether the document is there at all, not
                    // whether it ranks first.
                    .recall_vector(&separated(*written), 3)
                    .expect("recall")
                    .contains(&numbered(*written))
            })
            .collect();

        assert!(
            unreachable.is_empty(),
            "round {round}: {} of {} documents survived the build in name only, \
             starting at {:?}",
            unreachable.len(),
            live.len(),
            unreachable.first()
        );
    }
}

/// A deterministic unit vector far from every other one this function makes.
///
/// Nearest-neighbour search is approximate, so vectors that sit close together
/// go missing from a result for ordinary reasons and would make the check above
/// mean nothing. These are drawn symmetrically about zero and normalised, so
/// any two of them are nearly orthogonal and a document that does not answer
/// its own vector is a document that is not there.
fn separated(seed: u128) -> Vec<f32> {
    let mut state = (seed as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(1);
    let mut vector = Vec::with_capacity(PROFILE.dimensions() as usize);
    for _ in 0..PROFILE.dimensions() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        vector.push((state >> 40) as f32 / 16_777_216.0 - 0.5);
    }

    let length: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    vector.iter().map(|value| value / length).collect()
}

/// Asking for a name means every word of it, not any word of it.
///
/// Backfill asks "which memories name this topic", and confirms each candidate
/// exactly afterwards. Ranked disjunction answers a different question -- what
/// is most relevant to these words -- and fills a bounded list with documents
/// carrying only the common half of the name, each of which then costs a
/// confirmation that rejects it.
///
/// What this shows is the semantics and the candidate count: a name nothing
/// carries every word of returns nothing at all, and a name one document
/// carries returns that document alone rather than it plus four decoys. It does
/// not show the failure it exists to prevent -- a real match pushed off the end
/// of the list by decoys -- because the document carrying both words also
/// scores highest, so at any size a test can build the disjunction still finds
/// it. That case needs a corpus, not a fixture.
#[test]
fn asking_for_a_name_requires_every_word_of_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
    )
    .expect("open index");

    let both = id(1);
    index
        .upsert(both, "the release process is documented here", &stub())
        .expect("upsert the one that names it");
    // Documents carrying only the common word, which is the situation a real
    // project is always in.
    for n in 2..40u8 {
        index
            .upsert(
                id(n),
                "another release note about the release we cut this week",
                &stub(),
            )
            .expect("upsert a decoy");
    }
    index.flush().expect("flush");

    let named = index
        .recall_naming("release process", 5)
        .expect("recall by name");
    assert_eq!(
        named,
        vec![both],
        "only the document carrying every word of the name should be a candidate"
    );

    // The same corpus and the same limit through the channel that ranks. Every
    // extra entry here is a candidate the caller would confirm and reject.
    let ranked = index
        .recall_segmented("release process", 5)
        .expect("recall by relevance");
    assert!(
        ranked.len() > named.len(),
        "the ranking channel should be the one that returns decoys: {ranked:?}"
    );

    // The names this store actually holds are identifiers and CJK, not English
    // bigrams, and both are segmented before they are indexed. A conjunction is
    // over whatever that segmentation produced, so a name that splits into
    // several tokens has to still find the document it splits the same way in.
    let identifier = id(200);
    let chinese = id(201);
    index
        .upsert(
            identifier,
            "everything goes out through argo_cd now",
            &stub(),
        )
        .expect("upsert an identifier");
    index
        .upsert(chinese, "部署流水线运行在持续集成上", &stub())
        .expect("upsert chinese");
    index.flush().expect("flush");

    assert!(
        index
            .recall_naming("argo_cd", 5)
            .expect("recall an identifier name")
            .contains(&identifier),
        "a name that segments into several tokens should still find its memory"
    );
    assert!(
        index
            .recall_naming("流水线", 5)
            .expect("recall a chinese name")
            .contains(&chinese),
        "a name with no spaces in it should still find its memory"
    );

    // A name no document carries every word of matches nothing, rather than
    // matching everything that carries part of it.
    let absent = index
        .recall_naming("release ceremony", 5)
        .expect("recall an absent name");
    assert!(
        absent.is_empty(),
        "a conjunction nothing satisfies should return nothing: {absent:?}"
    );
}
