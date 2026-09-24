//! Reshaping an index while it is being served.
//!
//! A collection records its segment size when it is created, and a project is
//! created empty, so every project that grew holds one segment per
//! [`segment_documents`](crate::segment_documents)`(0)` documents however large it became -- and each
//! segment keeps its own full-text stores resident. Fifty thousand documents
//! open at 1,292 MB in 25 segments and at 308 MB in 4, and the MIRACL workspace
//! spent 3,283 MB opening its index before any model was loaded. Recreating
//! the collection is the only fix: setting the size on an open one changes
//! nothing.
//!
//! [`Previous`](crate::Previous) made that a copy rather than a re-embedding,
//! and this makes it a copy nobody has to wait for. The served index is read a
//! batch at a time under its owner's lock and written into a sibling outside
//! it, so a search waits behind one batch rather than behind the rebuild. The
//! sibling's graph is built with no lock held at all, which is the part that
//! takes minutes. Writes keep arriving throughout, and the served index is
//! wrapped for the length of the copy so each one records the topic it
//! touched; those topics are read again and applied to the sibling before it
//! replaces the served one, the last of them under the lock so that nothing
//! can be written between that catch-up and the swap.
//!
//! The swap closes both indexes before moving either directory, because
//! Windows will not rename a directory something holds open, and it moves the
//! served one aside rather than deleting it until the copy has opened in its
//! place. A process that dies between the two renames leaves the served index
//! aside and nothing where it was, which [`Reshape::recover`] puts back.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pamin_core::{Scored, TopicId};

use crate::embedding::Profile;
use crate::error::{IndexError, Result};
use crate::projection::{Passage, Projection, ProjectionIndex, Segmentation, Stored};
use crate::segmentation::Segmenter;

/// A served projection: one handle, behind the lock its owner serializes every
/// call on.
///
/// The lock is the owner's, not this crate's -- the engine takes it because
/// concurrent calls wedge the index (see `Engine::index`) -- but a reshape has
/// to take it too, a batch at a time, and has to replace what it holds.
pub type Held = Mutex<Arc<dyn Projection + Send + Sync>>;

/// How many documents a copy reads under the lock at a time.
///
/// Bounded so that a search arriving mid-copy waits behind one read of this
/// many documents rather than behind the copy. The size is a choice and not a
/// measurement: it is what a rebuild writes at a time, and nothing here has
/// timed a read of it.
const COPY_BATCH: usize = 256;

/// How long a swap waits before looking again for a handle to be let go of.
///
/// The only handles on a served index outside its lock are the upkeep
/// worker's, taken for one `optimize` -- and while a reshape runs, the
/// recording wrapper answers `optimize` without doing anything, so the one a
/// swap can find is one that started before the reshape did. That ends on its
/// own; this is how often the swap checks.
const PATIENCE: Duration = Duration::from_millis(100);

/// Where the copy is built, beside the index it will replace.
fn sibling(live: &Path) -> PathBuf {
    live.with_extension("reshaping")
}

/// Where the served index is moved while the copy takes its place.
///
/// Not `.previous`, which a rebuild sets aside to lend its vectors from: a
/// rebuild discards what it finds there, and the two have to be able to fail
/// without either tidying away the other's index.
fn replaced(live: &Path) -> PathBuf {
    live.with_extension("replaced")
}

fn lock(held: &Held) -> std::sync::MutexGuard<'_, Arc<dyn Projection + Send + Sync>> {
    held.lock().expect("the index lock is poisoned")
}

/// Whether `served` is this recording, compared by address.
fn is(served: &Arc<dyn Projection + Send + Sync>, recording: &Arc<Recording>) -> bool {
    std::ptr::addr_eq(Arc::as_ptr(served), Arc::as_ptr(recording))
}

/// What a reshape did.
#[derive(Clone, Copy, Debug)]
pub struct Reshaped {
    /// The shape the index was in.
    pub before: Segmentation,
    /// The shape it is in now.
    pub after: Segmentation,
    /// Documents copied a batch at a time.
    pub copied: u64,
    /// Topics written while the copy ran, read again and applied to it.
    pub caught_up: u64,
}

/// A copy of a served index in the shape the policy wants, on its way to
/// replacing it.
///
/// Four steps, each its own call so the caller can do what it has to between
/// them -- the engine reads which topics to copy from the ledger, which is
/// async, after [`begin`](Self::begin) and before [`copy`](Self::copy):
///
/// 1. [`begin`](Self::begin) creates the sibling and starts recording writes;
/// 2. [`copy`](Self::copy) copies the topics it is given, a batch per lock;
/// 3. [`build`](Self::build) flushes the sibling and builds its graph with no
///    lock held, then applies what was written meanwhile;
/// 4. [`swap`](Self::swap) applies the last writes under the lock and puts the
///    sibling where the served index was.
///
/// Dropped before the swap, it abandons the copy: the served index is
/// unwrapped and goes on as it was, and the sibling is discarded.
pub struct Reshape {
    held: Arc<Held>,
    /// The wrapper installed in `held`; `None` once the swap has let go of it.
    recording: Option<Arc<Recording>>,
    /// The copy; `None` once the swap has closed it.
    next: Option<ProjectionIndex>,
    live: PathBuf,
    profile: Profile,
    before: Segmentation,
    copied: u64,
    caught_up: u64,
}

impl Reshape {
    /// Starts reshaping the index served from `held`, which lives at `live`,
    /// if its shape is worth the work; `None` if it is not.
    ///
    /// The caller has to list the topics to copy *after* this returns. A topic
    /// written before the recording starts is in the served index by then, so
    /// listing afterwards finds it; one written after is recorded. Listing
    /// first would miss a topic created and indexed between the two.
    pub fn begin(held: &Arc<Held>, live: &Path, profile: Profile) -> Result<Option<Self>> {
        let before = lock(held).segmentation()?;
        if !before.is_worth_rebuilding() {
            return Ok(None);
        }

        let next = ProjectionIndex::create_beside(live, &sibling(live), profile, before.documents)?;
        let recording = {
            let mut served = lock(held);
            let recording = Arc::new(Recording {
                inner: Arc::clone(&served),
                touched: Mutex::default(),
            });
            *served = Arc::clone(&recording) as Arc<dyn Projection + Send + Sync>;
            recording
        };

        Ok(Some(Self {
            held: Arc::clone(held),
            recording: Some(recording),
            next: Some(next),
            live: live.to_path_buf(),
            profile,
            before,
            copied: 0,
            caught_up: 0,
        }))
    }

    /// Copies these topics from the served index into the sibling.
    ///
    /// The lock is held for each batch's read and not for its write, so the
    /// served index is never held for longer than one read of [`COPY_BATCH`]
    /// documents. A topic the served index does not hold is skipped: the
    /// ledger lists a topic whose write has not reached the index yet, and
    /// that write is recorded when it does.
    pub fn copy(&mut self, topics: &[TopicId]) -> Result<()> {
        for chunk in topics.chunks(COPY_BATCH) {
            let stored = lock(&self.held).stored(chunk)?;
            self.copied += self.apply(chunk, &stored)?;
        }
        Ok(())
    }

    /// Makes the sibling durable and builds its graph, then applies what was
    /// written while that ran.
    ///
    /// No lock is held for the build, which is the slow part -- 8.1 s for a
    /// graph over ten thousand documents, 124 s over fifty thousand -- and the
    /// served index answers everything meanwhile. The catch-up after it is
    /// what keeps the one [`swap`](Self::swap) does under the lock small.
    pub fn build(&mut self) -> Result<()> {
        let next = self.next();
        next.flush()?;
        next.optimize()?;
        self.catch_up()
    }

    /// Applies what was written since the last catch-up, a batch per lock.
    fn catch_up(&mut self) -> Result<()> {
        let touched = self.recording().take();
        for chunk in touched.chunks(COPY_BATCH) {
            let stored = lock(&self.held).stored(chunk)?;
            self.apply(chunk, &stored)?;
        }
        self.caught_up += touched.len() as u64;
        Ok(())
    }

    /// Puts the sibling where the served index was, and serves it.
    ///
    /// Under the lock throughout, so nothing is written between the last
    /// catch-up and the swap and nothing is read from an index being moved.
    /// Everything holding the served index has to have let go of it before it
    /// can be closed, and a handle the upkeep worker still holds is waited out
    /// with the lock released, catching up again each time it is retaken.
    ///
    /// If the sibling cannot be put in place or will not open, the served
    /// index is moved back and reopened and the error is returned. Only if
    /// that reopen fails too is the project left without an index, and then
    /// every call on it says so rather than answering from nothing.
    pub fn swap(mut self) -> Result<Reshaped> {
        let held = Arc::clone(&self.held);
        loop {
            let mut served = lock(&held);
            let recording = Arc::clone(self.recording());
            if !is(&served, &recording) {
                return Err(IndexError::Engine(
                    "the served index was replaced while it was being reshaped".to_string(),
                ));
            }

            let touched = recording.take();
            for chunk in touched.chunks(COPY_BATCH) {
                let stored = served.stored(chunk)?;
                self.apply(chunk, &stored)?;
            }
            self.caught_up += touched.len() as u64;
            self.next().flush()?;

            // Three holders of the wrapper -- the lock's, this one's and the
            // clone just taken -- and one of the index inside it. Nothing can
            // take another while the lock is held; anything more is a handle
            // somebody took before and has not let go of yet.
            if Arc::strong_count(&recording) > 3 || Arc::strong_count(&recording.inner) > 1 {
                drop(served);
                drop(recording);
                std::thread::sleep(PATIENCE);
                continue;
            }

            // Durable before it is closed, so that putting it back, if the
            // swap fails, reopens exactly what was served.
            served.flush()?;
            *served = Arc::new(Closed {
                segmenter: served.segmenter(),
                passage: served.passage(),
                reason: "it is being swapped for its reshaped copy".to_string(),
            });
            drop(recording);
            // The last holders of each, so these close them.
            drop(self.recording.take());
            drop(self.next.take());

            match exchange(&self.live, self.profile) {
                Ok(index) => {
                    *served = Arc::new(index);
                    let after = served.segmentation();
                    drop(served);
                    // Past the swap, so failing to delete the old copy is not
                    // failing to reshape: `recover` removes it on the next open.
                    if let Err(error) = ProjectionIndex::discard(&replaced(&self.live)) {
                        tracing::warn!(%error, "could not delete the index a reshape replaced");
                    }
                    return Ok(Reshaped {
                        before: self.before,
                        after: after?,
                        copied: self.copied,
                        caught_up: self.caught_up,
                    });
                }
                Err(error) => {
                    match ProjectionIndex::reopen(&self.live, self.profile) {
                        Ok(index) => *served = Arc::new(index),
                        Err(reopening) => {
                            *served = Arc::new(Closed {
                                segmenter: served.segmenter(),
                                passage: served.passage(),
                                reason: format!("{error}, and reopening it failed: {reopening}"),
                            })
                        }
                    }
                    return Err(error);
                }
            }
        }
    }

    /// Puts back an index a swap moved aside and did not replace, and deletes
    /// one a finished swap left behind.
    ///
    /// For whoever opens `live` next. Between a swap's two renames the served
    /// index is aside and nothing is where it was, and an open that found
    /// nothing there would create an empty index -- a project with no memories
    /// and no error.
    pub fn recover(live: &Path) -> Result<()> {
        let replaced = replaced(live);
        if !std::fs::exists(&replaced)? {
            return Ok(());
        }
        if std::fs::exists(live)? {
            ProjectionIndex::discard(&replaced)
        } else {
            std::fs::rename(&replaced, live)?;
            Ok(())
        }
    }

    /// Writes what the served index holds for these topics into the sibling:
    /// each one it holds, and the removal of each one it does not.
    ///
    /// Returns how many it wrote.
    fn apply(&self, topics: &[TopicId], stored: &[Option<Stored>]) -> Result<u64> {
        let mut present: Vec<(TopicId, &str, &[f32])> = Vec::with_capacity(topics.len());
        let mut absent: Vec<TopicId> = Vec::new();
        for (topic, document) in topics.iter().zip(stored) {
            match document {
                Some(document) => present.push((
                    *topic,
                    document.content.as_str(),
                    document.embedding.as_slice(),
                )),
                None => absent.push(*topic),
            }
        }

        let next = self.next();
        next.upsert_batch(&present)?;
        if !absent.is_empty() {
            next.delete(&absent)?;
        }
        Ok(present.len() as u64)
    }

    fn next(&self) -> &ProjectionIndex {
        self.next
            .as_ref()
            .expect("the copy is open until the swap closes it")
    }

    fn recording(&self) -> &Arc<Recording> {
        self.recording
            .as_ref()
            .expect("the recording is installed until the swap removes it")
    }
}

impl Drop for Reshape {
    /// Abandons the copy, if the swap has not taken it.
    fn drop(&mut self) {
        if let Some(recording) = self.recording.take() {
            let mut served = self.held.lock().unwrap_or_else(PoisonError::into_inner);
            if is(&served, &recording) {
                *served = Arc::clone(&recording.inner);
            }
        }
        drop(self.next.take());
        // Left for the next reshape to discard if this fails: it discards the
        // sibling before creating one.
        if let Err(error) = ProjectionIndex::discard(&sibling(&self.live)) {
            tracing::warn!(%error, "could not delete an abandoned reshape's copy");
        }
    }
}

/// Moves the sibling into `live` and opens it, moving the served index aside.
///
/// Both indexes are closed by now. If the second rename or the open fails,
/// the served index is moved back before returning; if moving it back fails
/// as well, [`Reshape::recover`] does it on the next open.
fn exchange(live: &Path, profile: Profile) -> Result<ProjectionIndex> {
    let replaced = replaced(live);
    ProjectionIndex::discard(&replaced)?;
    std::fs::rename(live, &replaced)?;
    if let Err(error) = std::fs::rename(sibling(live), live) {
        std::fs::rename(&replaced, live)?;
        return Err(error.into());
    }
    ProjectionIndex::reopen(live, profile).or_else(|error| {
        ProjectionIndex::discard(live)?;
        std::fs::rename(&replaced, live)?;
        Err(error)
    })
}

/// The served index, recording which topics are written through it.
///
/// A wrapper installed where the served index was, rather than a list every
/// write path has to remember to feed: every write the owner makes goes
/// through what the lock holds, so none can be missed, including one added
/// after this was written.
struct Recording {
    inner: Arc<dyn Projection + Send + Sync>,
    touched: Mutex<HashSet<TopicId>>,
}

impl Recording {
    fn touch(&self, topics: impl IntoIterator<Item = TopicId>) {
        self.touched
            .lock()
            .expect("the reshape's record of writes is poisoned")
            .extend(topics);
    }

    /// The topics written since the last call, each once.
    fn take(&self) -> Vec<TopicId> {
        std::mem::take(
            &mut *self
                .touched
                .lock()
                .expect("the reshape's record of writes is poisoned"),
        )
        .into_iter()
        .collect()
    }
}

impl Projection for Recording {
    fn segmenter(&self) -> Arc<Segmenter> {
        self.inner.segmenter()
    }

    fn upsert(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<()> {
        self.touch([topic]);
        self.inner.upsert(topic, content, embedding)
    }

    fn upsert_batch(&self, documents: &[(TopicId, &str, &[f32])]) -> Result<()> {
        self.touch(documents.iter().map(|(topic, _, _)| *topic));
        self.inner.upsert_batch(documents)
    }

    fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<Scored>> {
        self.inner.recall_segmented(query, limit)
    }

    fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<Scored>> {
        self.inner.recall_ngram(query, limit)
    }

    fn recall_naming(&self, name: &str, limit: u32) -> Result<Vec<TopicId>> {
        self.inner.recall_naming(name, limit)
    }

    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<Scored>> {
        self.inner.recall_vector(embedding, limit)
    }

    fn stored(&self, topics: &[TopicId]) -> Result<Vec<Option<Stored>>> {
        self.inner.stored(topics)
    }

    fn delete(&self, topics: &[TopicId]) -> Result<()> {
        self.touch(topics.iter().copied());
        self.inner.delete(topics)
    }

    fn flush(&self) -> Result<()> {
        self.inner.flush()
    }

    /// Nothing, while the copy that will replace this index builds its own.
    ///
    /// Compacting an index about to be deleted is minutes of work for nothing,
    /// and it holds a handle the swap would have to wait out.
    fn optimize(&self) -> Result<()> {
        Ok(())
    }

    fn vector_index_completeness(&self) -> Result<f32> {
        self.inner.vector_index_completeness()
    }

    fn document_count(&self) -> Result<u64> {
        self.inner.document_count()
    }

    fn file_count(&self) -> Result<u64> {
        self.inner.file_count()
    }

    fn segmentation(&self) -> Result<Segmentation> {
        self.inner.segmentation()
    }

    fn passage(&self) -> Passage {
        self.inner.passage()
    }
}

/// What the lock holds while no index is open behind it.
///
/// Only observable if a swap failed and the served index would not reopen;
/// during a swap the lock is held and nobody can see it. Every call then says
/// why rather than answering from an index that is not there.
struct Closed {
    segmenter: Arc<Segmenter>,
    passage: Passage,
    reason: String,
}

impl Closed {
    fn refuse<T>(&self) -> Result<T> {
        Err(IndexError::Unavailable(self.reason.clone()))
    }
}

impl Projection for Closed {
    fn segmenter(&self) -> Arc<Segmenter> {
        Arc::clone(&self.segmenter)
    }

    fn upsert(&self, _: TopicId, _: &str, _: &[f32]) -> Result<()> {
        self.refuse()
    }

    fn upsert_batch(&self, _: &[(TopicId, &str, &[f32])]) -> Result<()> {
        self.refuse()
    }

    fn recall_segmented(&self, _: &str, _: u32) -> Result<Vec<Scored>> {
        self.refuse()
    }

    fn recall_ngram(&self, _: &str, _: u32) -> Result<Vec<Scored>> {
        self.refuse()
    }

    fn recall_naming(&self, _: &str, _: u32) -> Result<Vec<TopicId>> {
        self.refuse()
    }

    fn recall_vector(&self, _: &[f32], _: u32) -> Result<Vec<Scored>> {
        self.refuse()
    }

    fn stored(&self, _: &[TopicId]) -> Result<Vec<Option<Stored>>> {
        self.refuse()
    }

    fn delete(&self, _: &[TopicId]) -> Result<()> {
        self.refuse()
    }

    fn flush(&self) -> Result<()> {
        self.refuse()
    }

    fn optimize(&self) -> Result<()> {
        self.refuse()
    }

    fn vector_index_completeness(&self) -> Result<f32> {
        self.refuse()
    }

    fn document_count(&self) -> Result<u64> {
        self.refuse()
    }

    fn file_count(&self) -> Result<u64> {
        self.refuse()
    }

    fn segmentation(&self) -> Result<Segmentation> {
        self.refuse()
    }

    fn passage(&self) -> Passage {
        self.passage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{Access, segment_documents};

    const PROFILE: Profile = Profile::Speed;

    /// The smallest segment the engine will create -- it refuses anything
    /// under a thousand -- so that [`DOCUMENTS`] are four segments where the
    /// policy wants one: the shape a grown project is in, without writing
    /// fifty thousand documents to reach it.
    const TINY_SEGMENT: u64 = 1_000;

    const DOCUMENTS: u128 = 3_001;

    fn topic(n: u128) -> TopicId {
        TopicId(uuid::Uuid::from_u128(n))
    }

    fn content(n: u128) -> String {
        format!("memory number {n} is about kiln{n}")
    }

    /// A deterministic unit vector nearly orthogonal to every other one this
    /// makes, so a document that does not answer its own vector is missing.
    fn vector(seed: u128) -> Vec<f32> {
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

    /// A served index at `live` in the shape a grown project is in, holding
    /// `DOCUMENTS` documents, with its encoding aged to content alone so the
    /// copy has to keep an encoding that is not the one a new index gets.
    fn served(live: &Path) -> Arc<Held> {
        std::fs::create_dir_all(live).expect("index dir");
        std::fs::write(
            live.join("profile"),
            format!("{}\ntopic\nfp32\ncontent", PROFILE.model_id()),
        )
        .expect("marker");
        let index = ProjectionIndex::open_sized(live, PROFILE, Access::ReadWrite, TINY_SEGMENT)
            .expect("open the served index");
        let contents: Vec<String> = (1..=DOCUMENTS).map(content).collect();
        let vectors: Vec<Vec<f32>> = (1..=DOCUMENTS).map(vector).collect();
        let documents: Vec<(TopicId, &str, &[f32])> = (1..=DOCUMENTS)
            .map(|n| {
                let at = (n - 1) as usize;
                (topic(n), contents[at].as_str(), vectors[at].as_slice())
            })
            .collect();
        index.upsert_batch(&documents).expect("write");
        index.flush().expect("flush");
        Arc::new(Mutex::new(Arc::new(index)))
    }

    fn every_topic() -> Vec<TopicId> {
        (1..=DOCUMENTS).map(topic).collect()
    }

    /// A served index is copied into the shape the policy wants and swapped
    /// in, and afterwards holds exactly what the served one held -- including
    /// what was written while the copy ran.
    ///
    /// The writes are the point. An edit and a deletion land after the copy
    /// and before the build, and a new topic after the build and before the
    /// swap. Taking out the catch-up the swap does under the lock fails this
    /// by losing the new topic, and taking out both loses all three. The one
    /// after the build only shortens the swap's -- the record is not cleared
    /// until something applies it -- so taking out that one alone changes
    /// nothing this can see, which is what it should do.
    #[test]
    fn a_reshaped_index_holds_what_the_served_one_did_in_the_shape_the_policy_wants() {
        let root = tempfile::tempdir().expect("temp dir");
        let live = root.path().join("index");
        let held = served(&live);
        let marker = std::fs::read_to_string(live.join("profile")).expect("marker");

        let shape = lock(&held).segmentation().expect("shape");
        assert!(
            shape.segments() == 4 && shape.wanted() == 1 && shape.is_worth_rebuilding(),
            "the fixture is not in a grown project's shape: {shape:?}"
        );
        let words_before = lock(&held).recall_ngram("kiln4", 50).expect("recall");
        assert!(
            words_before.iter().any(|hit| hit.topic == topic(4)),
            "the fixture's lexical channel finds nothing to compare against"
        );

        let mut reshape = Reshape::begin(&held, &live, PROFILE)
            .expect("begin")
            .expect("four segments where one would do is worth reshaping");
        reshape.copy(&every_topic()).expect("copy");

        // Written between the copy and the build, through the lock the way
        // the engine writes.
        {
            let served = lock(&held);
            served
                .upsert(topic(5), "memory number 5 was edited", &vector(1005))
                .expect("edit");
            served.delete(&[topic(7)]).expect("delete");
        }
        reshape.build().expect("build");
        // And between the build and the swap.
        lock(&held)
            .upsert(
                topic(DOCUMENTS + 1),
                &content(DOCUMENTS + 1),
                &vector(DOCUMENTS + 1),
            )
            .expect("write a new topic");

        let reshaped = reshape.swap().expect("swap");
        assert_eq!(reshaped.copied, DOCUMENTS as u64);
        assert_eq!(reshaped.after.documents, DOCUMENTS as u64);
        assert_eq!(
            reshaped.after.segments(),
            reshaped.after.wanted(),
            "the copy is not in the shape the policy wants: {:?}",
            reshaped.after
        );
        assert!(!reshaped.after.is_worth_rebuilding());

        let served = lock(&held);
        assert_eq!(served.document_count().expect("count"), DOCUMENTS as u64);
        assert_eq!(
            served.passage(),
            Passage::Content,
            "the copy was relabelled with an encoding its vectors were not embedded in"
        );
        assert_eq!(
            std::fs::read_to_string(live.join("profile")).expect("marker"),
            marker,
            "the copy does not record what the served index recorded"
        );

        let mut expected: Vec<(TopicId, Option<Stored>)> = (1..=DOCUMENTS + 1)
            .map(|n| {
                (
                    topic(n),
                    Some(Stored {
                        content: content(n),
                        embedding: vector(n),
                    }),
                )
            })
            .collect();
        expected[4].1 = Some(Stored {
            content: "memory number 5 was edited".to_string(),
            embedding: vector(1005),
        });
        expected[6].1 = None;
        let topics: Vec<TopicId> = expected.iter().map(|(topic, _)| *topic).collect();
        let stored = served.stored(&topics).expect("read back");
        for ((topic, want), got) in expected.iter().zip(&stored) {
            assert!(
                got == want,
                "topic {topic} did not survive the reshape: it holds {:?}",
                got.as_ref().map(|document| &document.content)
            );
        }

        // Searches still find what they found, by vector and by word.
        for n in (1..=DOCUMENTS + 1).filter(|n| *n != 5 && *n != 7) {
            let hits = served.recall_vector(&vector(n), 3).expect("recall");
            assert!(
                hits.iter().any(|hit| hit.topic == topic(n)),
                "topic {n} no longer answers its own vector"
            );
        }
        let words_after = served.recall_ngram("kiln4", 50).expect("recall");
        let mut before: Vec<TopicId> = words_before.iter().map(|hit| hit.topic).collect();
        let mut after: Vec<TopicId> = words_after.iter().map(|hit| hit.topic).collect();
        before.sort();
        after.sort();
        assert_eq!(after, before, "a lexical search finds different topics");
        drop(served);

        assert!(!sibling(&live).exists(), "the copy's directory is left");
        assert!(!replaced(&live).exists(), "the replaced index is left");

        // On disk, not only in the handle: what reopens is the copy.
        drop(held);
        let reopened = ProjectionIndex::reopen(&live, PROFILE).expect("reopen");
        assert_eq!(reopened.document_count().expect("count"), DOCUMENTS as u64);
        assert!(
            !reopened
                .segmentation()
                .expect("shape")
                .is_worth_rebuilding()
        );
    }

    /// A swap waits for a handle taken before the reshape began, rather than
    /// closing an index somebody is still using.
    #[test]
    fn a_swap_waits_for_a_handle_on_the_served_index_to_be_let_go() {
        let root = tempfile::tempdir().expect("temp dir");
        let live = root.path().join("index");
        let held = served(&live);
        // What the upkeep worker takes for an `optimize`.
        let upkeep = Arc::clone(&*lock(&held));

        let mut reshape = Reshape::begin(&held, &live, PROFILE)
            .expect("begin")
            .expect("worth reshaping");
        reshape.copy(&every_topic()).expect("copy");
        reshape.build().expect("build");

        let releasing = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            drop(upkeep);
        });
        let reshaped = reshape.swap().expect("swap");
        releasing.join().expect("release");
        assert!(!reshaped.after.is_worth_rebuilding());
        assert_eq!(
            lock(&held).document_count().expect("count"),
            DOCUMENTS as u64
        );
    }

    /// A reshape dropped before its swap leaves the served index as it was:
    /// the same handle in the lock, answering, and no copy on disk.
    #[test]
    fn an_abandoned_reshape_leaves_the_served_index_as_it_was() {
        let root = tempfile::tempdir().expect("temp dir");
        let live = root.path().join("index");
        let held = served(&live);
        let original = Arc::as_ptr(&*lock(&held));

        let mut reshape = Reshape::begin(&held, &live, PROFILE)
            .expect("begin")
            .expect("worth reshaping");
        reshape.copy(&every_topic()).expect("copy");
        assert!(sibling(&live).exists(), "the copy is being built beside it");
        drop(reshape);

        let served = lock(&held);
        assert!(
            std::ptr::addr_eq(Arc::as_ptr(&*served), original),
            "the served index is still wrapped, or was replaced"
        );
        assert_eq!(served.document_count().expect("count"), DOCUMENTS as u64);
        assert!(!sibling(&live).exists(), "the abandoned copy is left");
    }

    /// An index already in the shape the policy wants is left alone.
    #[test]
    fn an_index_in_shape_is_not_reshaped() {
        let root = tempfile::tempdir().expect("temp dir");
        let live = root.path().join("index");
        let index =
            ProjectionIndex::open_sized(&live, PROFILE, Access::ReadWrite, segment_documents(0))
                .expect("open");
        let held: Arc<Held> = Arc::new(Mutex::new(Arc::new(index)));
        assert!(
            Reshape::begin(&held, &live, PROFILE)
                .expect("begin")
                .is_none()
        );
        assert!(!sibling(&live).exists());
    }

    /// A swap interrupted between its two renames leaves the served index
    /// aside and nothing where it was; recovering puts it back, and after a
    /// swap that finished it deletes only the leftover.
    #[test]
    fn an_index_a_swap_left_aside_is_put_back() {
        let root = tempfile::tempdir().expect("temp dir");
        let live = root.path().join("index");
        drop(served(&live));

        std::fs::rename(&live, replaced(&live)).expect("interrupt a swap");
        Reshape::recover(&live).expect("recover");
        let index = ProjectionIndex::reopen(&live, PROFILE).expect("the index is back");
        assert_eq!(index.document_count().expect("count"), DOCUMENTS as u64);
        drop(index);

        std::fs::create_dir_all(replaced(&live)).expect("a finished swap's leftover");
        Reshape::recover(&live).expect("recover");
        assert!(live.exists() && !replaced(&live).exists());
    }
}
