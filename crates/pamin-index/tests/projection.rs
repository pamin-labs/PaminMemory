//! Drives the projection index against the real engine.

use std::collections::HashMap;

use pamin_core::{Scored, TopicId};
use pamin_index::{Access, Profile, Projection, ProjectionIndex, VectorIndex};

const PROFILE: Profile = Profile::Speed;

/// A stand-in embedding. These tests exercise the lexical channels, so the
/// vector only has to be the right width.
fn stub() -> Vec<f32> {
    vec![0.1; PROFILE.dimensions() as usize]
}

fn id(byte: u8) -> TopicId {
    TopicId(uuid::Uuid::from_bytes([byte; 16]))
}

/// A distinct identifier per number, for the tests that write many documents.
fn numbered(n: u128) -> TopicId {
    TopicId(uuid::Uuid::from_u128(n))
}

/// Whether a channel returned this topic at all, at any rank.
fn holds(candidates: &[Scored], topic: TopicId) -> bool {
    candidates.iter().any(|candidate| candidate.topic == topic)
}

/// Every channel scores what it returns, and returns it in that order.
///
/// The scores were being dropped on the floor: `collect_scored` reads them from
/// the same result set the ranking already came out of, so nothing here is
/// asking the index to work harder -- it is asking whether the accessor was
/// wired to anything at all. A channel that returned `Some(0.0)` for every
/// candidate would pass a test that only checked the ranking, and would make
/// any judgement of that channel's confidence a judgement of the constant zero.
#[test]
fn every_channel_scores_what_it_returns_and_ranks_by_it() {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    // Distinct embeddings, which is load-bearing: written with the same stub
    // vector, every document is exactly as near the query as every other, so
    // the vector channel reports one constant and an assertion that it orders
    // by its score passes without checking anything. That is how this test
    // first shipped, and it left the one channel whose metric could have been
    // a distance rather than a similarity unchecked.
    let leaning = |towards: usize| {
        let mut vector = vec![0.1; PROFILE.dimensions() as usize];
        vector[towards] = 1.0;
        vector
    };
    for (n, text) in [
        "the deployment pipeline runs on every merge to main",
        "the deployment pipeline is described in database.rs",
        "an unrelated memory about the weather",
    ]
    .iter()
    .enumerate()
    {
        index
            .upsert(numbered(n as u128 + 1), text, &leaning(n))
            .expect("upsert");
    }
    index.flush().expect("flush");

    // Nearest the first document by construction, so the vector channel has a
    // real ordering to report and a real best candidate.
    let query = leaning(0);
    for (channel, candidates) in [
        (
            "segmented",
            index.recall_segmented("deployment pipeline", 10).unwrap(),
        ),
        ("ngram", index.recall_ngram("pipeline", 10).unwrap()),
        ("vector", index.recall_vector(&query, 10).unwrap()),
    ] {
        assert!(
            !candidates.is_empty(),
            "{channel} returned nothing to score"
        );

        let scores: Vec<f32> = candidates
            .iter()
            .map(|candidate| {
                candidate
                    .score
                    .unwrap_or_else(|| panic!("{channel} returned a candidate with no score"))
            })
            .collect();

        assert!(
            scores.windows(2).all(|pair| pair[0] >= pair[1]),
            "{channel} is not ordered by the score it reports, so the score is a distance \
             rather than a similarity and anything that sums it is summing it backwards: \
             {scores:?}"
        );
        assert!(
            scores.windows(2).any(|pair| pair[0] > pair[1]),
            "{channel} reported the same score for every candidate, so ordering by it asserts \
             nothing: {scores:?}"
        );
        assert!(
            scores.iter().any(|score| *score != 0.0),
            "{channel} reported zero for every candidate, which is what an \
             unwired accessor looks like: {scores:?}"
        );
    }
}

#[test]
fn lexical_recall_works_across_languages_and_on_exact_strings() {
    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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
    assert!(holds(&hits, english), "english recall failed: {hits:?}");

    let hits = index.recall_segmented("流水线", 10).expect("chinese");
    assert!(holds(&hits, chinese), "chinese recall failed: {hits:?}");

    let hits = index.recall_segmented("東京", 10).expect("japanese");
    assert!(holds(&hits, japanese), "japanese recall failed: {hits:?}");

    let hits = index.recall_segmented("ทำงาน", 10).expect("thai");
    assert!(holds(&hits, thai), "thai recall failed: {hits:?}");

    // The n-gram field catches substrings of a path or an error code, which
    // word segmentation splits apart.
    let hits = index.recall_ngram("database.rs", 10).expect("path");
    assert!(holds(&hits, identifier), "path recall failed: {hits:?}");

    let hits = index.recall_ngram("E1234", 10).expect("error code");
    assert!(
        holds(&hits, identifier),
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
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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

    let first = ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadOnly,
        0,
    )
    .expect("first reader");
    let second = ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadOnly,
        0,
    )
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
    let held = ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    let waiting = {
        let dir = dir.path().to_path_buf();
        let legacy = legacy.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let opened = ProjectionIndex::open(
                &dir,
                &legacy,
                PROFILE,
                VectorIndex::default(),
                Access::ReadWrite,
                0,
            )
            .is_ok();
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
/// Building the graph rewrites the vector storage, and zvec has a report --
/// alibaba/zvec#724, against 0.6 and 0.7 -- of that step dropping the last
/// documents of a collection. It is fixed upstream and not released: #731
/// merged as `31d88ea`, and the newest published version is 0.7.0. What makes
/// it worth a standing test rather than a note is how it fails: the dropped
/// documents keep appearing in the document count and in scalar reads, so only
/// a vector query can tell, and re-running the build does not bring them back.
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
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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
                    .iter()
                    .any(|candidate| candidate.topic == numbered(*written))
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
        VectorIndex::default(),
        Access::ReadWrite,
        0,
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

/// An index from before a document meant a topic refuses to open.
///
/// This is the one upgrade failure with no symptom. The identifiers in an index
/// keyed by topic state are read as topic identifiers, match nothing, and every
/// channel comes back empty -- so `search` returns no results, reports no error,
/// and looks exactly like a workspace nobody has written to. The embedding model
/// check does not catch it: a profile whose model did not change opens happily
/// and answers nothing.
#[test]
fn an_index_keyed_the_old_way_says_so_rather_than_answering_nothing() {
    let dir = tempfile::tempdir().expect("temp dir");
    let legacy = dir.path().join("legacy");

    ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("build one");

    // What the marker held before there was a grain to record: the model, and
    // nothing else.
    std::fs::write(dir.path().join("profile"), PROFILE.model_id()).expect("age the marker");

    let message = match ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    ) {
        Ok(_) => panic!("an index of the wrong shape must not open"),
        Err(error) => error.to_string(),
    };
    assert!(
        message.contains("topic state") && message.contains("reindex"),
        "the refusal has to say what is wrong and what to run: {message}"
    );
}

/// An index built before names were embedded keeps being written the way it
/// was built, and a new one embeds names.
///
/// The upgrade has to be silent in the right direction: an existing workspace
/// must neither refuse to open nor start mixing `name: content` vectors into
/// an index of content vectors, whose distances would then compare two
/// different encodings and look fine. So the marker's missing line reads as
/// the old encoding, and only a fresh index -- which is what `pamin reindex`
/// builds -- gets the new one.
#[test]
fn an_index_built_before_names_keeps_its_encoding() {
    use pamin_index::{Passage, Projection};

    let dir = tempfile::tempdir().expect("temp dir");
    let legacy = dir.path().join("legacy");

    let fresh = ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("build one");
    assert_eq!(fresh.passage(), Passage::Named, "a new index embeds names");
    drop(fresh);

    // What the marker held before the encoding was recorded.
    std::fs::write(
        dir.path().join("profile"),
        format!(
            "{}\ntopic\n{}",
            PROFILE.model_id(),
            VectorIndex::default().label()
        ),
    )
    .expect("age the marker");
    let aged = ProjectionIndex::open(
        dir.path(),
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("an index from before the encoding line still opens");
    assert_eq!(
        aged.passage(),
        Passage::Content,
        "an index with no encoding line was built from content and must stay so"
    );

    assert_eq!(
        Passage::Named.render("platform rota", "it pages ines"),
        "platform rota: it pages ines"
    );
    assert_eq!(
        Passage::Content.render("platform rota", "it pages ines"),
        "it pages ines"
    );
}

/// Rewriting the same few memories does not make the index grow for ever.
///
/// The index spreads across about two more files with every write, whatever it
/// holds: the count follows writes, not documents. Nothing noticed, because the
/// only thing that compacted it was gated on documents -- so ten memories
/// rewritten a few hundred times reached eight hundred files and a workspace
/// used normally for a week died of `Too many open files` on an ordinary
/// descriptor limit.
///
/// This writes what such a workspace writes and asks how many files are left.
/// Without a budget it fails on the count long before the assertion means
/// anything about compaction; with one it settles wherever the policy says.
#[test]
fn rewriting_the_same_memories_leaves_a_bounded_number_of_files() {
    fn files(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .expect("read the index directory")
            .flatten()
            .map(|entry| match entry.file_type() {
                Ok(kind) if kind.is_dir() => files(&entry.path()),
                _ => 1,
            })
            .sum()
    }

    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    // Ten topics, two hundred writes, one flush each -- which is what the
    // cascade does for an interactive write, and why the count follows writes.
    for round in 0..200u128 {
        index
            .upsert(
                numbered(round % 10),
                &format!("writer recorded round {round} of the shared index run"),
                &stub(),
            )
            .expect("upsert");
        index.flush().expect("flush");

        if pamin_index::is_fragmented(index.file_count().expect("count files")) {
            index.optimize().expect("optimize");
        }
    }

    let left = files(dir.path());
    assert!(
        left <= 512,
        "the index settled at {left} files, which is not a bound"
    );
    assert_eq!(
        index.document_count().expect("documents"),
        10,
        "the files were compacted away along with the memories"
    );
}

/// Splitting text does not wait for whoever is using the index.
///
/// The composition layer holds the projection behind a mutex, for a defect in
/// the search engine that has nothing to do with segmentation. Handing out the
/// segmenter as a borrow made that lock cover tokenizing too: five callers took
/// it for work the index was not doing, and two held it a long time -- one
/// segments every candidate a probe returned, the other every topic name in the
/// project. Every search on that project queued behind them.
///
/// So the segmenter is a handle, and this is what says so: a caller takes one,
/// somebody else locks the projection and keeps it, and the tokenizing still
/// finishes. Before the change this does not compile rather than failing an
/// assertion -- the borrow's lifetime was the guard's, so there was no way to
/// write the second half of it. That is the stronger form of the same claim.
#[test]
fn tokenizing_does_not_wait_for_the_index() {
    use std::sync::{Arc, Mutex};

    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    // The shape the composition layer holds it in, so this is the lock the test
    // is actually about and not a stand-in for it.
    let index: Arc<Mutex<Box<dyn Projection + Send + Sync>>> =
        Arc::new(Mutex::new(Box::new(index)));

    let segmenter = index.lock().expect("lock the index").segmenter();
    let held = index.lock().expect("lock the index");

    let tokens = std::thread::spawn(move || segmenter.name_sequence("index lock contention"))
        .join()
        .expect("the segmenting thread panicked");

    assert_eq!(
        tokens,
        vec!["index", "lock", "contention"],
        "the segmenter handed out by the projection stopped tokenizing"
    );

    // Named rather than dropped at the end of scope, so it is visible that the
    // lock was still held for all of the above.
    drop(held);
}

/// A rebuild reuses a vector only where it is certainly the one it would compute.
///
/// The set-aside index lends a topic's stored vector when the text stored with
/// it is the state's text exactly; a changed memory, or one the old index never
/// held, is embedded again. An index whose vectors came from another encoding
/// lends nothing and is gone, so nothing can mix two embedding spaces.
#[test]
fn a_rebuild_reuses_only_the_vectors_of_unchanged_text() {
    use pamin_index::Previous;

    let root = tempfile::tempdir().expect("temp dir");
    let dir = root.path().join("index");
    let legacy = root.path().join("legacy");
    let mut vector = stub();
    vector[1] = 0.5;

    let index = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open");
    index
        .upsert_batch(&[
            (
                id(1),
                "the release train leaves on thursdays",
                vector.as_slice(),
            ),
            (id(2), "the oncall rota rotates weekly", stub().as_slice()),
        ])
        .expect("write");
    index.flush().expect("flush");
    drop(index);

    let previous = Previous::set_aside(&dir, PROFILE)
        .expect("set aside")
        .expect("an index built now lends its vectors");
    assert!(!dir.exists(), "the rebuild starts from an empty directory");

    let wanted = HashMap::from([
        (id(1), "the release train leaves on thursdays"),
        (id(2), "the oncall rota rotates every fortnight"),
        (id(3), "a topic the old index never held"),
    ]);
    assert_eq!(previous.lends(&wanted).expect("count"), 1);
    let lent = lent(&previous, &wanted);
    assert_eq!(
        lent.get(&id(1)).map(|(_, vector)| vector.clone()),
        Some(pamin_index::as_stored(&vector)),
        "unchanged text keeps its vector"
    );
    assert!(!lent.contains_key(&id(2)), "changed text is embedded again");
    assert!(
        !lent.contains_key(&id(3)),
        "a topic the old index lacks is embedded"
    );
    previous.discard().expect("discard");
    assert!(!dir.with_extension("previous").exists());

    // Built from content alone: its vectors are not what an index built now
    // computes, so it lends nothing and is discarded.
    let index = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open");
    drop(index);
    std::fs::write(
        dir.join("profile"),
        format!(
            "{}\ntopic\n{}\ncontent",
            PROFILE.model_id(),
            VectorIndex::default().label()
        ),
    )
    .expect("age the marker");
    assert!(
        Previous::set_aside(&dir, PROFILE)
            .expect("set aside")
            .is_none(),
        "vectors from another encoding were lent"
    );
    assert!(!dir.exists() && !dir.with_extension("previous").exists());
}

/// Everything a set-aside index lends for `wanted`, by topic.
fn lent(
    previous: &pamin_index::Previous,
    wanted: &HashMap<TopicId, &str>,
) -> HashMap<TopicId, (String, Vec<f32>)> {
    let mut lent = HashMap::new();
    let topics = previous
        .lend(wanted, 256, |documents| {
            for (topic, content, vector) in documents {
                lent.insert(*topic, (content.to_string(), vector.to_vec()));
            }
            Ok(())
        })
        .expect("lend");
    assert_eq!(
        topics.len(),
        lent.len(),
        "the topics reported lent are not the ones handed over"
    );
    lent
}

/// A rebuild lends, through one pass of the engine's document iterator,
/// exactly what reading each topic back by key returns: the same documents,
/// the same text and bit-for-bit the same vectors, and nothing whose text the
/// ledger has since changed, that was deleted, or that the index never held.
///
/// Keyed reads are the reference because they are what a rebuild lent through
/// before, and what a reshape still reads with. More documents than one batch
/// of the lend, with edits and deletions in the index, so the iterator walks
/// more than one segment's worth of what a real index holds.
#[test]
fn a_rebuild_lends_through_the_iterator_what_keyed_reads_return() {
    use pamin_index::Previous;

    const HELD: u128 = 600;
    let root = tempfile::tempdir().expect("temp dir");
    let dir = root.path().join("index");
    let legacy = root.path().join("legacy");
    let text = |n: u128| format!("memory number {n} is about kiln{n}");

    let index = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open");
    let texts: Vec<String> = (1..=HELD).map(text).collect();
    let vectors: Vec<Vec<f32>> = (1..=HELD).map(separated).collect();
    index
        .upsert_batch(
            &(1..=HELD)
                .map(|n| {
                    let at = (n - 1) as usize;
                    (numbered(n), texts[at].as_str(), vectors[at].as_slice())
                })
                .collect::<Vec<_>>(),
        )
        .expect("write");
    index.flush().expect("flush");
    for n in 10..20 {
        index
            .upsert(
                numbered(n),
                &format!("memory {n} was edited"),
                &separated(n + 1000),
            )
            .expect("edit");
    }
    let deleted: Vec<TopicId> = (30..40).map(numbered).collect();
    index.delete(&deleted).expect("delete");
    index.flush().expect("flush");

    // What the index holds, read back by key.
    let every: Vec<TopicId> = (1..=HELD + 10).map(numbered).collect();
    let by_key: HashMap<TopicId, pamin_index::Stored> = every
        .iter()
        .zip(index.stored(&every).expect("read back by key"))
        .filter_map(|(topic, stored)| stored.map(|stored| (*topic, stored)))
        .collect();
    drop(index);

    // What the ledger says now: the index's text for most topics, a text the
    // index does not hold for a few, and topics it never held at all.
    let ledger: HashMap<TopicId, String> = every
        .iter()
        .map(|topic| {
            let n = topic.0.as_u128();
            let content = match by_key.get(topic) {
                Some(_) if (50..60).contains(&n) => format!("memory {n} changed in the ledger"),
                Some(stored) => stored.content.clone(),
                None => text(n),
            };
            (*topic, content)
        })
        .collect();
    let wanted: HashMap<TopicId, &str> = ledger
        .iter()
        .map(|(topic, content)| (*topic, content.as_str()))
        .collect();
    let expected: HashMap<TopicId, (String, Vec<f32>)> = by_key
        .into_iter()
        .filter(|(topic, stored)| wanted[topic] == stored.content)
        .map(|(topic, stored)| (topic, (stored.content, stored.embedding)))
        .collect();
    assert_eq!(
        expected.len(),
        (HELD - 10 - 10) as usize,
        "the fixture does not hold what it means to"
    );

    let previous = Previous::set_aside(&dir, PROFILE)
        .expect("set aside")
        .expect("an index built now lends its vectors");
    assert_eq!(previous.lends(&wanted).expect("count"), expected.len());
    let lent = lent(&previous, &wanted);
    assert_eq!(lent.len(), expected.len());
    for (topic, want) in &expected {
        assert!(
            lent.get(topic) == Some(want),
            "topic {topic} was lent as something other than what a keyed read returns"
        );
    }
    previous.discard().expect("discard");
}

/// A document reads back as it was written -- the text and the vector both --
/// whether or not a flush has reached it yet, and a topic the index does not
/// hold reads back as absent rather than as an error.
///
/// Unflushed is the case that matters. A copy taken from a served index reads
/// whatever the writes since the last flush left in its buffer, and a read
/// that skipped the buffer would copy a memory as it was before its last edit.
#[test]
fn a_document_reads_back_as_it_was_written() {
    use pamin_index::Stored;

    let dir = tempfile::tempdir().expect("temp dir");
    let index = ProjectionIndex::open(
        dir.path(),
        &dir.path().join("legacy"),
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    index
        .upsert(
            id(1),
            "the release train leaves on thursdays",
            &separated(1),
        )
        .expect("upsert");
    index.flush().expect("flush");
    index
        .upsert(id(2), "the oncall rota rotates weekly", &separated(2))
        .expect("upsert, left unflushed");
    index
        .upsert(id(1), "the release train leaves on fridays", &separated(3))
        .expect("an edit, left unflushed");

    let stored = index.stored(&[id(1), id(2), id(3)]).expect("read back");
    assert_eq!(
        stored,
        vec![
            Some(Stored {
                content: "the release train leaves on fridays".to_string(),
                embedding: pamin_index::as_stored(&separated(3)),
            }),
            Some(Stored {
                content: "the oncall rota rotates weekly".to_string(),
                embedding: pamin_index::as_stored(&separated(2)),
            }),
            None,
        ],
        "an unflushed edit or write read back as something other than itself"
    );

    index.delete(&[id(2)]).expect("delete");
    assert_eq!(
        index.stored(&[id(2)]).expect("read back"),
        vec![None],
        "a deleted document still reads back"
    );
}

/// Marks `dir` as an index built before key spellings were recorded, so what
/// opens there next spells every key as the topic's identifier -- which is
/// what every index built before then holds.
fn keyed_as_written(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("index dir");
    std::fs::write(
        dir.join("profile"),
        format!(
            "{}\ntopic\n{}\nnamed",
            PROFILE.model_id(),
            VectorIndex::default().label()
        ),
    )
    .expect("age the marker");
}

/// The files the engine's key map is spread across.
///
/// The engine keeps the map from primary key to row in a RocksDB instance of
/// its own beside the segments, and never compacts it.
fn key_map_files(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir.join("memories"))
        .expect("read the collection")
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("idmap"))
        .flat_map(|map| std::fs::read_dir(map.path()).expect("read the key map"))
        .flatten()
        .filter(|file| file.file_name().to_string_lossy().ends_with(".sst"))
        .count()
}

/// An index whose keys are the identifiers as written goes on answering, and
/// the rebuild that replaces it reads its vectors back through that spelling.
///
/// Every workspace built before key spellings were recorded holds one. Read
/// with the new spelling it would match nothing and say nothing -- the silence
/// `DOCUMENT_GRAIN` exists to turn into an error -- so the marker's missing
/// line has to be read as the old spelling, and this reads it the other way
/// too, to show the line is what decides.
#[test]
fn an_index_keyed_by_the_identifier_as_written_keeps_answering() {
    use pamin_index::Previous;

    let root = tempfile::tempdir().expect("temp dir");
    let dir = root.path().join("index");
    let legacy = root.path().join("legacy");
    let topics: Vec<TopicId> = (0..3).map(|_| TopicId::new()).collect();
    let texts = [
        "the release train leaves on thursdays",
        "the oncall rota rotates weekly",
        "the staging cluster is rebuilt nightly",
    ];

    keyed_as_written(&dir);
    let index = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("open");
    let documents: Vec<Vec<f32>> = (0..3).map(|n| separated(n + 40)).collect();
    index
        .upsert_batch(
            &(0..3)
                .map(|n| (topics[n], texts[n], documents[n].as_slice()))
                .collect::<Vec<_>>(),
        )
        .expect("write");
    index.flush().expect("flush");
    index.delete(&[topics[2]]).expect("delete");
    index.flush().expect("flush");
    drop(index);

    let index = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("reopen");
    assert!(holds(
        &index.recall_segmented("release train", 10).expect("recall"),
        topics[0]
    ));
    assert!(holds(
        &index.recall_vector(&documents[1], 10).expect("recall"),
        topics[1]
    ));
    assert!(
        !holds(
            &index
                .recall_segmented("staging cluster", 10)
                .expect("recall"),
            topics[2]
        ),
        "a deletion by the old spelling did not delete"
    );
    assert_eq!(
        index.stored(&[topics[1]]).expect("read back")[0]
            .as_ref()
            .map(|stored| stored.content.as_str()),
        Some(texts[1])
    );
    drop(index);
    assert!(
        std::fs::read_to_string(dir.join("profile"))
            .expect("marker")
            .lines()
            .count()
            == 4,
        "opening an old index relabelled it"
    );

    // The rebuild's side: it lends from this index and writes a new one.
    let previous = Previous::set_aside(&dir, PROFILE)
        .expect("set aside")
        .expect("an index whose vectors are current lends them");
    let wanted = HashMap::from([(topics[0], texts[0]), (topics[1], texts[1])]);
    assert_eq!(
        previous.lends(&wanted).expect("count"),
        2,
        "the set-aside index was read with a spelling it was not written in"
    );
    let rebuilt = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("rebuild");
    let lent = previous
        .lend(&wanted, 256, |documents| rebuilt.upsert_batch(documents))
        .expect("lend");
    assert_eq!(lent.len(), 2);
    rebuilt.flush().expect("flush");
    previous.discard().expect("discard");
    assert!(holds(
        &rebuilt.recall_vector(&documents[1], 10).expect("recall"),
        topics[1]
    ));
    drop(rebuilt);
    assert_eq!(
        std::fs::read_to_string(dir.join("profile"))
            .expect("marker")
            .lines()
            .last(),
        Some("reversed-keys"),
        "a rebuilt index keeps the old spelling"
    );

    // Read the other way, the same documents answer as other topics entirely.
    std::fs::write(
        dir.join("profile"),
        format!(
            "{}\ntopic\n{}\nnamed\ntopic-keys",
            PROFILE.model_id(),
            VectorIndex::default().label()
        ),
    )
    .expect("mislabel the marker");
    let misread = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadOnly,
        0,
    )
    .expect("open");
    let found = misread
        .recall_segmented("release train", 10)
        .expect("recall");
    assert!(
        !found.is_empty() && !holds(&found, topics[0]),
        "the spelling the marker names is not the one the index is read with"
    );
}

/// Every channel returns the same topics, in the same order and with the same
/// scores, whichever way the index spells its keys.
///
/// The spelling decides where a key lands in the engine's key map, and
/// nothing else should move: a document's row, its segment and every score
/// follow the order it was written in. Two indexes are written the same
/// documents in the same batches, one keyed each way, and every channel's
/// top fifty is compared for every query.
#[test]
fn every_channel_answers_the_same_whichever_way_keys_are_spelled() {
    const WORDS: [&str; 24] = [
        "kiln", "harbor", "ledger", "orchid", "copper", "lantern", "meadow", "quartz", "saffron",
        "tundra", "violet", "walnut", "anchor", "bramble", "cinder", "dune", "ember", "fjord",
        "garnet", "hollow", "ivory", "juniper", "kestrel", "lagoon",
    ];
    const DOCUMENTS: usize = 640;
    let text = |n: usize| {
        let mut state = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let words: Vec<&str> = (0..9)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                WORDS[(state % WORDS.len() as u64) as usize]
            })
            .collect();
        format!("note {n} {}", words.join(" "))
    };
    let topics: Vec<TopicId> = (0..DOCUMENTS).map(|_| TopicId::new()).collect();
    let texts: Vec<String> = (0..DOCUMENTS).map(text).collect();
    let vectors: Vec<Vec<f32>> = (0..DOCUMENTS).map(|n| separated(n as u128)).collect();

    let root = tempfile::tempdir().expect("temp dir");
    let legacy = root.path().join("legacy");
    let build = |dir: &std::path::Path| {
        let index = ProjectionIndex::open(
            dir,
            &legacy,
            PROFILE,
            VectorIndex::default(),
            Access::ReadWrite,
            0,
        )
        .expect("open");
        // Sixty-four to a flush, as the cascade writes, with every eighth
        // batch rewriting a document written before.
        for (batch, chunk) in (0..DOCUMENTS).collect::<Vec<_>>().chunks(64).enumerate() {
            let documents: Vec<(TopicId, &str, &[f32])> = chunk
                .iter()
                .map(|&n| (topics[n], texts[n].as_str(), vectors[n].as_slice()))
                .collect();
            index.upsert_batch(&documents).expect("write");
            if batch % 8 == 7 {
                index
                    .upsert(topics[batch], "note rewritten kiln harbor", &vectors[batch])
                    .expect("rewrite");
            }
            index.flush().expect("flush");
        }
        index.delete(&[topics[3]]).expect("delete");
        index.flush().expect("flush");
        index
    };
    let written = root.path().join("written");
    keyed_as_written(&written);
    let written = build(&written);
    let reversed = build(&root.path().join("reversed"));

    let answers = |index: &ProjectionIndex, query: usize| {
        let words = WORDS[query % WORDS.len()];
        let pair = format!("{words} {}", WORDS[(query * 7 + 3) % WORDS.len()]);
        [
            index.recall_segmented(&pair, 50).expect("segmented"),
            index.recall_ngram(words, 50).expect("ngram"),
            index
                .recall_naming(&pair, 50)
                .expect("naming")
                .into_iter()
                .map(|topic| Scored::new(topic, 0.0))
                .collect(),
            index
                .recall_vector(&vectors[query * 17 % DOCUMENTS], 50)
                .expect("vector"),
        ]
    };
    for query in 0..30 {
        let (before, after) = (answers(&written, query), answers(&reversed, query));
        for (channel, (before, after)) in ["segmented", "ngram", "naming", "vector"]
            .iter()
            .zip(before.iter().zip(&after))
        {
            // The premise: two empty lists agree about nothing.
            assert!(
                !before.is_empty(),
                "query {query} found nothing on {channel}"
            );
            let spelled = |list: &[Scored]| -> Vec<(TopicId, u32)> {
                list.iter()
                    .map(|hit| (hit.topic, hit.score.map_or(0, f32::to_bits)))
                    .collect()
            };
            assert_eq!(
                spelled(before),
                spelled(after),
                "query {query} answered differently on {channel}"
            );
        }
    }
}

/// Topics written in the order they were created leave the engine's key map a
/// bounded number of files, however many flushes wrote them.
///
/// Identifiers are time-ordered, so each flush wrote keys above every key
/// before it. The engine flushes its key map and never compacts it, and
/// RocksDB moves a file that overlaps nothing below it down a level without
/// merging, so every flush left a file of its own for good -- 200 over 12,800
/// memories written through the engine sixty-four to a drain. Spelling the key with its random
/// bytes first makes every flush overlap the last, and RocksDB merges them.
///
/// Both spellings are written, and the old one is asserted to accumulate:
/// that is what shows the count is measuring the key map at all, and if it
/// stops accumulating the engine has started compacting and this workaround
/// can go.
#[test]
fn topics_written_in_order_leave_a_bounded_key_map() {
    const FLUSHES: usize = 60;

    let root = tempfile::tempdir().expect("temp dir");
    let legacy = root.path().join("legacy");
    let files = |dir: &std::path::Path| {
        let index = ProjectionIndex::open(
            dir,
            &legacy,
            PROFILE,
            VectorIndex::default(),
            Access::ReadWrite,
            0,
        )
        .expect("open");
        for flush in 0..FLUSHES {
            let documents: Vec<(TopicId, String)> = (0..4)
                .map(|n| (TopicId::new(), format!("flush {flush} memory {n}")))
                .collect();
            let vector = stub();
            index
                .upsert_batch(
                    &documents
                        .iter()
                        .map(|(topic, text)| (*topic, text.as_str(), vector.as_slice()))
                        .collect::<Vec<_>>(),
                )
                .expect("write");
            index.flush().expect("flush");
        }
        drop(index);
        key_map_files(dir)
    };

    let written = root.path().join("written");
    keyed_as_written(&written);
    let as_written = files(&written);
    let reversed = files(&root.path().join("reversed"));

    assert!(
        as_written >= FLUSHES / 2,
        "the premise: keys in creation order left {as_written} key-map files over \
         {FLUSHES} flushes, so this is not counting what accumulates"
    );
    assert!(
        reversed <= 8,
        "reversed keys left {reversed} key-map files over {FLUSHES} flushes \
         (keys as written left {as_written})"
    );
}

/// Each vector index keeps the vector it was given, as half precision stores
/// it, records itself, and refuses to be opened as the other.
///
/// Reading back what was stored is what a reshape copies and a rebuild lends,
/// so a project can move between the two without embedding anything again.
/// And the refusal is the marker doing its job: an index searched with the
/// other one's parameters is the silent failure ADR 0001 records.
#[test]
fn each_vector_index_reads_back_what_it_stored_and_is_opened_only_as_itself() {
    const DOCUMENTS: u128 = 300;

    for index in VectorIndex::ALL {
        let root = tempfile::tempdir().expect("temp dir");
        let dir = root.path().join("index");
        let legacy = root.path().join("legacy");
        let built = ProjectionIndex::open(&dir, &legacy, PROFILE, index, Access::ReadWrite, 0)
            .expect("open");
        let topics: Vec<TopicId> = (1..=DOCUMENTS).map(numbered).collect();
        let texts: Vec<String> = (1..=DOCUMENTS)
            .map(|n| format!("memory number {n}"))
            .collect();
        let vectors: Vec<Vec<f32>> = (1..=DOCUMENTS).map(separated).collect();
        built
            .upsert_batch(
                &(0..DOCUMENTS as usize)
                    .map(|at| (topics[at], texts[at].as_str(), vectors[at].as_slice()))
                    .collect::<Vec<_>>(),
            )
            .expect("write");
        built.flush().expect("flush");
        built.optimize().expect("build");
        assert_eq!(built.vector_index(), index);
        assert_eq!(
            built.vector_index_completeness().expect("completeness"),
            1.0,
            "{index:?} left documents outside its index"
        );
        for (at, got) in built.stored(&topics).expect("read back").iter().enumerate() {
            let got = got.as_ref().expect("held");
            assert_eq!(got.content, texts[at]);
            assert!(
                got.embedding == pamin_index::as_stored(&vectors[at]),
                "{index:?} read back something other than what it stored"
            );
        }
        drop(built);

        assert_eq!(
            std::fs::read_to_string(dir.join("profile"))
                .expect("marker")
                .lines()
                .nth(2),
            Some(index.label()),
            "the marker does not record the vector index"
        );
        let other = VectorIndex::ALL
            .into_iter()
            .find(|other| *other != index)
            .expect("two of them");
        let refused =
            match ProjectionIndex::open(&dir, &legacy, PROFILE, other, Access::ReadWrite, 0) {
                Ok(_) => panic!("{index:?} opened as {other:?}"),
                Err(error) => error.to_string(),
            };
        assert!(
            refused.contains(index.label()) && refused.contains("reindex"),
            "the refusal has to name what the index is and what to run: {refused}"
        );

        let reopened = ProjectionIndex::open(&dir, &legacy, PROFILE, index, Access::ReadWrite, 0)
            .expect("reopen as itself");
        for at in (0..DOCUMENTS as usize).step_by(10) {
            assert!(
                holds(
                    &reopened.recall_vector(&vectors[at], 3).expect("recall"),
                    topics[at]
                ),
                "{index:?}: a document does not answer its own vector after reopening"
            );
        }
    }
}

/// Clustered vectors and queries drawn from the same clusters, at the profile's
/// width, so near neighbours are near for a reason and recall means something.
fn clustered(documents: usize, queries: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    const CLUSTERS: usize = 40;
    let mut state: u64 = 0x5eed;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    };
    let unit = |vector: Vec<f32>| {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        vector.into_iter().map(|x| x / norm).collect::<Vec<f32>>()
    };
    let dimensions = PROFILE.dimensions() as usize;
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| unit((0..dimensions).map(|_| next()).collect()))
        .collect();
    let mut around =
        |centroid: &Vec<f32>| unit(centroid.iter().map(|x| x + 0.35 * next()).collect());
    let documents = (0..documents)
        .map(|at| around(&centroids[at % CLUSTERS]))
        .collect();
    let queries = (0..queries)
        .map(|at| around(&centroids[(at * 7) % CLUSTERS]))
        .collect();
    (documents, queries)
}

/// Each vector index returns the nearest documents, not merely near ones, once
/// it is built.
///
/// A small `recall.rs`, small enough to run on every change, for the failure
/// with no other symptom: an index returning plausible neighbours that are not
/// the nearest. The first quantization attempt returned 0.000 recall@10 and
/// int8 with the engine's rotation 0.053, both with no error; half precision
/// scored by the engine alone lost 0.03 on 50,000 clustered vectors, which the
/// rescore takes back. Here `disk` measures 0.9967 and `memory` 1.0000; the
/// floor is below both and far above any of those failures.
#[test]
fn each_vector_index_returns_the_nearest_documents() {
    const DOCUMENTS: usize = 2_000;
    const QUERIES: usize = 30;
    const TOP: usize = 10;
    const FLOOR: f64 = 0.95;

    let (documents, queries) = clustered(DOCUMENTS, QUERIES);
    let truth: Vec<Vec<TopicId>> = queries
        .iter()
        .map(|query| {
            let mut scored: Vec<(f32, usize)> = documents
                .iter()
                .enumerate()
                .map(|(at, document)| (document.iter().zip(query).map(|(a, b)| a * b).sum(), at))
                .collect();
            scored.sort_by(|left, right| right.0.total_cmp(&left.0));
            scored
                .iter()
                .take(TOP)
                .map(|(_, at)| numbered(*at as u128 + 1))
                .collect()
        })
        .collect();

    for index in VectorIndex::ALL {
        let dir = tempfile::tempdir().expect("temp dir");
        let built = ProjectionIndex::open(
            &dir.path().join("index"),
            &dir.path().join("legacy"),
            PROFILE,
            index,
            Access::ReadWrite,
            0,
        )
        .expect("open");
        built
            .upsert_batch(
                &documents
                    .iter()
                    .enumerate()
                    .map(|(at, vector)| (numbered(at as u128 + 1), "", vector.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .expect("write");
        built.flush().expect("flush");
        built.optimize().expect("build");
        assert!(
            built.vector_index_completeness().expect("completeness") > 0.99,
            "{index:?}: the index has to be built for this to measure it rather than a scan"
        );

        let found: usize = queries
            .iter()
            .zip(&truth)
            .map(|(query, want)| {
                built
                    .recall_vector(query, TOP as u32)
                    .expect("recall")
                    .iter()
                    .filter(|hit| want.contains(&hit.topic))
                    .count()
            })
            .sum();
        let recall = found as f64 / (QUERIES * TOP) as f64;
        println!("  {index:?}: recall@{TOP} {recall:.4}");
        assert!(
            recall >= FLOOR,
            "{index:?} returned recall@{TOP} {recall:.4}, below {FLOOR}"
        );
    }
}

/// An fp32 index built before the vector indexes existed is refused with a
/// message naming `pamin reindex`, and the rebuild lends its vectors rather
/// than embedding them again.
///
/// Every workspace that predates this holds one. Searching it with half-
/// precision parameters is the silent wrong answer the marker exists to
/// prevent, so it is refused; and the rebuild that moves it reads its fp32
/// vectors and stores each as half precision would have stored the model's.
#[test]
fn an_fp32_index_from_before_is_refused_and_lends_to_its_rebuild() {
    use pamin_index::Previous;
    use zvec_rust::{Collection, CollectionSchema, DataType, Doc, FieldSchema, IndexParams};

    let root = tempfile::tempdir().expect("temp dir");
    let dir = root.path().join("index");
    let legacy = root.path().join("legacy");
    std::fs::create_dir_all(&dir).expect("index dir");
    // What an index built before looked like: the same fields, the vector
    // field fp32 under an HNSW graph, and a marker with its storage line.
    let schema = CollectionSchema::builder("memories")
        .add_field(FieldSchema::new("id", DataType::String, false, 0).expect("field"))
        .add_indexed_field(
            "content_segmented",
            DataType::String,
            IndexParams::fts(Some("standard"), Some(&["lowercase"]), None).expect("fts"),
        )
        .add_indexed_field(
            "content_ngram",
            DataType::String,
            IndexParams::fts(Some("ngram"), None, None).expect("fts"),
        )
        .add_vector_field(
            "embedding",
            DataType::VectorFp32,
            PROFILE.dimensions(),
            IndexParams::hnsw(zvec_rust::MetricType::Cosine, 32, 500).expect("hnsw"),
        )
        .build()
        .expect("schema");
    let texts = [
        "the release train leaves on thursdays",
        "the oncall rota rotates weekly",
    ];
    let vectors: Vec<Vec<f32>> = (0..2).map(|n| separated(n + 70)).collect();
    {
        let collection =
            Collection::create_and_open(&dir.join("memories").to_string_lossy(), &schema, None)
                .expect("create");
        let docs: Vec<Doc> = (0..2)
            .map(|n| {
                // Keys spelled as the topic is written, which the marker's
                // missing key line says.
                let key = id(n as u8 + 1).to_string();
                let mut doc = Doc::new().expect("doc");
                doc.set_pk(&key);
                doc.add_string("id", &key).expect("id");
                doc.add_string("content_segmented", texts[n]).expect("text");
                doc.add_string("content_ngram", texts[n]).expect("text");
                doc.add_vector_f32("embedding", &vectors[n])
                    .expect("vector");
                doc
            })
            .collect();
        collection
            .upsert(&docs.iter().collect::<Vec<_>>())
            .expect("write");
        collection.flush().expect("flush");
    }
    std::fs::write(
        dir.join("profile"),
        format!("{}\ntopic\nfp32\nnamed", PROFILE.model_id()),
    )
    .expect("marker");

    for index in VectorIndex::ALL {
        let refused =
            match ProjectionIndex::open(&dir, &legacy, PROFILE, index, Access::ReadWrite, 0) {
                Ok(_) => panic!("an fp32 index opened as {index:?}"),
                Err(error) => error.to_string(),
            };
        assert!(
            refused.contains("fp32") && refused.contains("reindex"),
            "the refusal has to name what the index is and what to run: {refused}"
        );
    }

    let previous = Previous::set_aside(&dir, PROFILE)
        .expect("set aside")
        .expect("an fp32 index of the same model and encoding lends its vectors");
    let wanted = HashMap::from([(id(1), texts[0]), (id(2), texts[1])]);
    assert_eq!(previous.lends(&wanted).expect("count"), 2);
    let rebuilt = ProjectionIndex::open(
        &dir,
        &legacy,
        PROFILE,
        VectorIndex::default(),
        Access::ReadWrite,
        0,
    )
    .expect("rebuild");
    previous
        .lend(&wanted, 256, |documents| rebuilt.upsert_batch(documents))
        .expect("lend");
    rebuilt.flush().expect("flush");
    previous.discard().expect("discard");
    let stored = rebuilt.stored(&[id(1), id(2)]).expect("read back");
    for (n, got) in stored.iter().enumerate() {
        assert!(
            got.as_ref().map(|stored| &stored.embedding)
                == Some(&pamin_index::as_stored(&vectors[n])),
            "the rebuild did not store the lent vector as half precision stores it"
        );
    }
    assert!(holds(
        &rebuilt.recall_vector(&vectors[1], 3).expect("recall"),
        id(2)
    ));
}
