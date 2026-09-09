//! The projection index: two lexical fields and one vector field in one engine.
//!
//! Everything here is derived. Losing the whole directory costs a reindex from
//! PostgreSQL, which is what makes depending on a pre-1.0 engine reasonable: a
//! breaking change is a rebuild rather than a migration.
//!
//! The engine offers a hybrid search helper that fuses its own channels. It is
//! deliberately unused. The graph channel lives in PostgreSQL where this engine
//! cannot see it, so an engine-fused list would be fused again against the graph
//! list, weighting its members twice, and the per-channel ranks every result has
//! to report would already be gone.

use std::path::Path;
use std::sync::Once;
use std::time::{Duration, Instant};

use pamin_core::TopicStateId;

use crate::embedding::Profile;
use zvec_rust::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, Fts,
    FtsQueryParams, IndexParams, MetricType, SearchQuery,
};

use crate::error::{IndexError, Result};
use crate::segmentation::Segmenter;

const COLLECTION: &str = "memories";

/// Word-level recall, fed pre-segmented text so every language tokenizes well.
const FIELD_SEGMENTED: &str = "content_segmented";
/// Substring recall over raw text: paths, error codes, identifiers.
const FIELD_NGRAM: &str = "content_ngram";
const FIELD_VECTOR: &str = "embedding";

static INITIALIZE: Once = Once::new();

/// How many documents the engine accepts in one write.
///
/// Its own limit, not a tuning choice: a larger batch is refused outright.
const WRITE_BATCH: usize = 1024;

/// What a handle on the index is allowed to do with it.
///
/// The engine locks the collection's directory, and which lock it takes follows
/// from this. Several read-only handles coexist; a read-write handle excludes
/// every other handle, readers included. Commands therefore have to say which
/// they need, because asking for more than they use is what turns two
/// simultaneous searches into one search and one failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Queries. Shared with other readers.
    ReadOnly,
    /// Writes and rebuilds. Exclusive.
    ReadWrite,
}

/// What the layer above needs a projection to do.
///
/// The architecture decision that chose this engine mitigated its pre-1.0 risk
/// by keeping it behind a boundary, and named this trait as the boundary. It
/// was never written, so what actually stood between the composition layer and
/// a specific engine was one concrete type. Every method here is one the layer
/// above calls; nothing is here for a caller that does not exist.
///
/// Everything a projection holds is derived, so replacing one is a rebuild
/// rather than a migration. That is what makes an engine swap an addition:
/// a second implementation of this can be built and measured against the first
/// on the same corpus, which is what the evaluation harness needs and what
/// deciding to swap would require evidence from.
pub trait Projection {
    /// The segmenter this projection tokenizes with.
    ///
    /// On the trait because anything comparing text against indexed content has
    /// to split it the way the index did. A projection that tokenizes one way
    /// and hands out a segmenter that tokenizes another is an index nothing
    /// matches against.
    fn segmenter(&self) -> &Segmenter;

    /// Adds or replaces one topic state.
    fn upsert(&self, topic_state: TopicStateId, content: &str, embedding: &[f32]) -> Result<()>;

    /// Adds or replaces many.
    fn upsert_batch(&self, documents: &[(TopicStateId, &str, &[f32])]) -> Result<()>;

    /// Word-level lexical recall, best first.
    fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>>;

    /// Substring lexical recall over raw text, best first.
    fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>>;

    /// Lexical recall for documents containing every word of a name.
    ///
    /// A conjunction rather than [`Projection::recall_segmented`]'s ranking of
    /// anything that matches at all. The question behind it is not "what is
    /// most relevant to this name" but "which memories name this thing", and a
    /// two-word name answered by either word alone fills the candidates with
    /// documents carrying only the common half -- so a real match falls off the
    /// end of a bounded list. The caller still confirms each candidate exactly.
    fn recall_naming(&self, name: &str, limit: u32) -> Result<Vec<TopicStateId>>;

    /// Semantic recall over dense embeddings, nearest first.
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<TopicStateId>>;

    /// Removes these topic states.
    ///
    /// The projection had no way to shrink: the only route out was deleting the
    /// whole directory. A soft-deleted state therefore stayed in every channel's
    /// candidate budget, so removing content from the ledger quietly reduced how
    /// much a search could find.
    fn delete(&self, states: &[TopicStateId]) -> Result<()>;

    /// Makes buffered writes visible to later queries.
    fn flush(&self) -> Result<()>;

    /// Builds whatever structure makes recall faster than a scan.
    fn optimize(&self) -> Result<()>;

    /// How much of the collection that structure covers, from 0.0 to 1.0.
    fn vector_index_completeness(&self) -> Result<f32>;

    /// How many documents the projection holds.
    fn document_count(&self) -> Result<u64>;
}

/// A lexical or vector index over topic states.
pub struct ProjectionIndex {
    collection: Collection,
    segmenter: Segmenter,
}

impl ProjectionIndex {
    /// Opens the index at `dir`, creating it if absent.
    ///
    /// The profile is recorded on creation and checked on every reopen. Mixing
    /// embedding spaces in one index produces distances that mean nothing, and
    /// nothing about the resulting rankings would look wrong, so this is
    /// enforced rather than documented. Changing profile requires a reindex.
    pub fn open(dir: &Path, legacy_dir: &Path, profile: Profile, access: Access) -> Result<Self> {
        // A workspace built before projects had their own directory holds one
        // shared collection. Opening this project's empty directory beside it
        // would return nothing and look like an empty workspace, so it is
        // reported instead.
        if std::fs::exists(legacy_dir)? {
            return Err(IndexError::LegacyLayout);
        }

        std::fs::create_dir_all(dir)?;
        let marker = dir.join("profile");
        match std::fs::read_to_string(&marker) {
            Ok(recorded) if recorded.trim() != profile.model_id() => {
                return Err(IndexError::ProfileMismatch {
                    indexed: recorded.trim().to_string(),
                    requested: profile.model_id().to_string(),
                });
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                std::fs::write(&marker, profile.model_id())?;
            }
            Err(error) => return Err(error.into()),
        }

        Self::open_with_dimensions(dir, profile.dimensions(), access)
    }

    fn open_with_dimensions(dir: &Path, dimensions: u32, access: Access) -> Result<Self> {
        INITIALIZE.call_once(|| {
            let _ = zvec_rust::initialize(None);
        });

        std::fs::create_dir_all(dir)?;
        let path = dir.join(COLLECTION);

        let schema = CollectionSchema::builder(COLLECTION)
            .add_field(FieldSchema::new("id", DataType::String, false, 0)?)
            // Input is already segmented, so the engine only has to split on
            // the spaces we produced.
            .add_indexed_field(
                FIELD_SEGMENTED,
                DataType::String,
                IndexParams::fts(Some("standard"), Some(&["lowercase"]), None)?,
            )
            // Dictionary-free by construction, which is what makes it a usable
            // fallback for text no segmenter handled well, and what catches
            // substrings that segmentation splits apart.
            .add_indexed_field(
                FIELD_NGRAM,
                DataType::String,
                IndexParams::fts(Some("ngram"), None, None)?,
            )
            .add_vector_field(
                FIELD_VECTOR,
                DataType::VectorFp32,
                dimensions,
                IndexParams::hnsw(MetricType::Cosine, 16, 100)?,
            )
            .build()?;

        // The engine refuses to create over an existing path, so reopen when
        // the collection is already there. Every command after the first opens
        // rather than creates.
        //
        // Creating is a write however the caller means to use the result, so a
        // read-only open of a workspace nothing has been written to yet still
        // creates first and reopens. That is one extra open, once in a
        // workspace's life, and the alternative is `pamin search` failing on a
        // workspace that is merely empty.
        let path = path.to_string_lossy().to_string();
        let collection = open_contended(|| {
            if !std::fs::exists(&path)? {
                Collection::create_and_open(&path, &schema, None)?;
            }
            Ok(Collection::open(&path, options(access)?.as_ref())?)
        })?;

        Ok(Self {
            collection,
            segmenter: Segmenter::new(),
        })
    }

    fn document(&self, topic_state: TopicStateId, content: &str, embedding: &[f32]) -> Result<Doc> {
        let mut doc = Doc::new()?;
        let key = topic_state.to_string();
        doc.set_pk(&key);
        doc.add_string("id", &key)?;
        doc.add_string(FIELD_SEGMENTED, &self.segmenter.segment_for_index(content))?;
        doc.add_string(FIELD_NGRAM, content)?;
        doc.add_vector_f32(FIELD_VECTOR, embedding)?;
        Ok(doc)
    }

    fn recall_text(&self, field: &str, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        self.recall_fts(field, query, limit, false)
    }

    /// Lexical recall, either ranking whatever matches or requiring every term.
    fn recall_fts(
        &self,
        field: &str,
        query: &str,
        limit: u32,
        every_term: bool,
    ) -> Result<Vec<TopicStateId>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut fts = Fts::new()?;
        fts.set_match_string(query)?;
        let mut search = SearchQuery::fts(field, &fts, limit as i32)?;
        if every_term {
            search.set_fts_params(FtsQueryParams::new(Some("AND"))?)?;
        }

        // Only ranks leave this function. The engine's BM25 scores are not
        // comparable with vector distances, and rank fusion is what lets the
        // two be combined without pretending they are.
        Ok(collect_ids(self.collection.query(&search)?))
    }

    /// Deletes the index directory so the next open starts empty.
    ///
    /// Rebuilding is the intended way to clear it. The projection carries no
    /// state PostgreSQL cannot reproduce, so discarding the directory is both
    /// the simplest reset and a standing demonstration that the index is
    /// disposable.
    pub fn discard(dir: &Path) -> Result<()> {
        match std::fs::remove_dir_all(dir) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error.into()),
        }
    }
}

impl Projection for ProjectionIndex {
    /// The segmenter this index tokenizes with.
    ///
    /// Shared rather than duplicated so that anything comparing text against
    /// indexed content splits it the same way this index did. A second
    /// segmenter would be the same code today and a divergence the first time
    /// either side changed.
    fn segmenter(&self) -> &Segmenter {
        &self.segmenter
    }

    /// Adds or replaces many topic states.
    ///
    /// Chunked because the engine refuses a write of more than [`WRITE_BATCH`]
    /// documents. Rebuilding used to write one document per call, which pays
    /// the per-call cost once per document; here it is once per batch.
    ///
    /// No flush: a caller writing in batches decides when the result becomes
    /// visible, and flushing between batches would make that decision for them
    /// once per batch.
    fn upsert_batch(&self, documents: &[(TopicStateId, &str, &[f32])]) -> Result<()> {
        for chunk in documents.chunks(WRITE_BATCH) {
            let docs = chunk
                .iter()
                .map(|(topic_state, content, embedding)| {
                    self.document(*topic_state, content, embedding)
                })
                .collect::<Result<Vec<_>>>()?;

            let refs: Vec<&Doc> = docs.iter().collect();
            self.collection.upsert(&refs)?;
        }

        Ok(())
    }

    /// Adds or replaces one topic state.
    ///
    /// The embedding is required rather than optional. The engine enforces it,
    /// and it is the right constraint: a document indexed without one is
    /// invisible to the vector channel, which would show up as unexplained
    /// recall gaps rather than as an error.
    fn upsert(&self, topic_state: TopicStateId, content: &str, embedding: &[f32]) -> Result<()> {
        let doc = self.document(topic_state, content, embedding)?;
        self.collection.upsert(&[&doc])?;
        Ok(())
    }

    /// Removes these topic states.
    fn delete(&self, states: &[TopicStateId]) -> Result<()> {
        for chunk in states.chunks(WRITE_BATCH) {
            let keys: Vec<String> = chunk.iter().map(ToString::to_string).collect();
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            self.collection.delete(&keys)?;
        }

        Ok(())
    }

    /// Word-level lexical recall, ranked by BM25.
    ///
    /// The query is segmented by the same function that segmented the documents.
    /// Tokenizing the two differently is the standard way to build an index that
    /// never matches.
    fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        let segmented = self.segmenter.segment_for_index(query);
        self.recall_text(FIELD_SEGMENTED, &segmented, limit)
    }

    /// Substring lexical recall over raw text, ranked by BM25.
    fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        self.recall_text(FIELD_NGRAM, query, limit)
    }

    /// Word-level recall requiring every word of the name.
    fn recall_naming(&self, name: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        let segmented = self.segmenter.segment_for_index(name);
        self.recall_fts(FIELD_SEGMENTED, &segmented, limit, true)
    }

    /// Semantic recall over dense embeddings.
    ///
    /// Returns ranks only, like the lexical channels. A cosine distance and a
    /// BM25 score are different quantities, and keeping both as ranks is what
    /// lets one fusion step combine them.
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<TopicStateId>> {
        let search = SearchQuery::new(FIELD_VECTOR, embedding, limit as i32)?;
        Ok(collect_ids(self.collection.query(&search)?))
    }

    /// Flushes buffered writes so a later query sees them.
    fn flush(&self) -> Result<()> {
        self.collection.flush()?;
        Ok(())
    }

    /// Builds the vector index over everything written since the last call.
    ///
    /// Documents land in a flat buffer that vector search scans exhaustively,
    /// and only this moves them into the graph. Nothing in this project had
    /// ever called it, so the vector channel had been running a brute-force
    /// scan of the whole project on every query while the HNSW parameters it
    /// was configured with described a structure that was never built. Recall
    /// was right, which is why it went unnoticed.
    ///
    /// It is not free and does not belong on the write path: it runs over
    /// everything unindexed, so a write that happened to trigger it would pay
    /// for every write before it. `reindex` calls it because a rebuild is
    /// already the expensive operation; incremental writes wait for the
    /// cascade worker, which is where a threshold on
    /// [`vector_index_completeness`](Self::vector_index_completeness) belongs.
    fn optimize(&self) -> Result<()> {
        self.collection.optimize()?;
        Ok(())
    }

    /// How much of the collection the vector index covers, from 0.0 to 1.0.
    ///
    /// The share of documents [`optimize`](Self::optimize) has taken in. Below
    /// 1.0 the remainder is still answered by the flat buffer -- correctly, and
    /// at a cost that grows with the project.
    fn vector_index_completeness(&self) -> Result<f32> {
        Ok(self
            .collection
            .stats()?
            .indexes
            .iter()
            .find(|index| index.name == FIELD_VECTOR)
            .map_or(0.0, |index| index.completeness))
    }

    /// How many documents the index holds.
    fn document_count(&self) -> Result<u64> {
        Ok(self.collection.stats()?.doc_count)
    }
}

/// The engine's open options for this access mode.
///
/// `None` for read-write, which is what the engine defaults to, so the common
/// path allocates nothing.
fn options(access: Access) -> Result<Option<CollectionOptions>> {
    match access {
        Access::ReadWrite => Ok(None),
        Access::ReadOnly => {
            let mut options = CollectionOptions::new()?;
            options.set_read_only(true)?;
            Ok(Some(options))
        }
    }
}

/// How long to keep trying for the index's file lock before giving up.
///
/// Long enough to outlast the other command, short enough that a caller who is
/// actually stuck finds out quickly. A `pamin search` holds the lock for the
/// length of one query.
const LOCK_BUDGET: Duration = Duration::from_millis(2_000);

/// Opens the collection, waiting out another process that holds its lock.
///
/// The engine takes the lock non-blocking and exclusive, so two commands
/// running at once do not queue -- the second is refused outright. Agents drive
/// this CLI concurrently by design, so a plain refusal turns an ordinary
/// overlap into a failed command. Retrying with backoff is not the eventual
/// answer, which is opening read-only for queries and holding the index in one
/// process, but it is what makes an overlap survivable today.
///
/// The jitter matters more than the backoff: several commands started together
/// by one agent would otherwise retry in step forever.
fn open_contended(mut open: impl FnMut() -> Result<Collection>) -> Result<Collection> {
    let deadline = Instant::now() + LOCK_BUDGET;
    let mut wait = Duration::from_millis(5);

    loop {
        let error = match open() {
            Ok(collection) => return Ok(collection),
            Err(error) if is_lock_conflict(&error) => error,
            // Anything else is a real failure and retrying only delays it.
            Err(error) => return Err(error),
        };

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(IndexError::Busy(error.to_string()));
        }

        std::thread::sleep(jittered(wait).min(remaining));
        wait = (wait * 2).min(Duration::from_millis(200));
    }
}

/// Whether this is another process holding the index rather than a real fault.
///
/// Matched on the engine's message because its error code for this is
/// `InternalError`, which it also uses for faults worth reporting rather than
/// waiting out. If a future version words it differently this stops recognising
/// the conflict and the caller sees the refusal directly, which is the
/// behaviour that preceded this function.
fn is_lock_conflict(error: &IndexError) -> bool {
    matches!(error, IndexError::Engine(message) if message.contains("Can't lock"))
}

/// Spreads a wait over roughly half to all of `wait`.
///
/// The low bits of the clock, rather than a random number generator: this needs
/// to decorrelate a handful of processes that started together, not to be
/// unpredictable.
fn jittered(wait: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.subsec_nanos());
    wait / 2 + (wait / 2).mul_f64(f64::from(nanos % 1_000) / 1_000.0)
}

fn collect_ids(docs: Vec<Doc>) -> Vec<TopicStateId> {
    docs.iter()
        .filter_map(|doc| doc.get_pk())
        .filter_map(|pk| uuid::Uuid::parse_str(pk).ok())
        .map(TopicStateId::from)
        .collect()
}
