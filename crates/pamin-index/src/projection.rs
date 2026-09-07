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

use pamin_core::TopicStateId;

use crate::embedding::Profile;
use zvec_rust::{
    Collection, CollectionSchema, DataType, Doc, FieldSchema, Fts, IndexParams, MetricType,
    SearchQuery,
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
    pub fn open(dir: &Path, profile: Profile) -> Result<Self> {
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

        Self::open_with_dimensions(dir, profile.dimensions())
    }

    fn open_with_dimensions(dir: &Path, dimensions: u32) -> Result<Self> {
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
        let path = path.to_string_lossy().to_string();
        let collection = if std::fs::exists(&path)? {
            Collection::open(&path, None)?
        } else {
            Collection::create_and_open(&path, &schema, None)?
        };

        Ok(Self {
            collection,
            segmenter: Segmenter::new(),
        })
    }

    /// The segmenter this index tokenizes with.
    ///
    /// Shared rather than duplicated so that anything comparing text against
    /// indexed content splits it the same way this index did. A second
    /// segmenter would be the same code today and a divergence the first time
    /// either side changed.
    pub fn segmenter(&self) -> &Segmenter {
        &self.segmenter
    }

    /// Adds or replaces one topic state.
    ///
    /// The embedding is required rather than optional. The engine enforces it,
    /// and it is the right constraint: a document indexed without one is
    /// invisible to the vector channel, which would show up as unexplained
    /// recall gaps rather than as an error.
    pub fn upsert(
        &self,
        topic_state: TopicStateId,
        content: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let mut doc = Doc::new()?;
        let key = topic_state.to_string();
        doc.set_pk(&key);
        doc.add_string("id", &key)?;
        doc.add_string(FIELD_SEGMENTED, &self.segmenter.segment_for_index(content))?;
        doc.add_string(FIELD_NGRAM, content)?;
        doc.add_vector_f32(FIELD_VECTOR, embedding)?;

        self.collection.upsert(&[&doc])?;
        Ok(())
    }

    /// Word-level lexical recall, ranked by BM25.
    ///
    /// The query is segmented by the same function that segmented the documents.
    /// Tokenizing the two differently is the standard way to build an index that
    /// never matches.
    pub fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        let segmented = self.segmenter.segment_for_index(query);
        self.recall_text(FIELD_SEGMENTED, &segmented, limit)
    }

    /// Substring lexical recall over raw text, ranked by BM25.
    pub fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        self.recall_text(FIELD_NGRAM, query, limit)
    }

    fn recall_text(&self, field: &str, query: &str, limit: u32) -> Result<Vec<TopicStateId>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let mut fts = Fts::new()?;
        fts.set_match_string(query)?;
        let search = SearchQuery::fts(field, &fts, limit as i32)?;

        // Only ranks leave this function. The engine's BM25 scores are not
        // comparable with vector distances, and rank fusion is what lets the
        // two be combined without pretending they are.
        Ok(collect_ids(self.collection.query(&search)?))
    }

    /// Semantic recall over dense embeddings.
    ///
    /// Returns ranks only, like the lexical channels. A cosine distance and a
    /// BM25 score are different quantities, and keeping both as ranks is what
    /// lets one fusion step combine them.
    pub fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<TopicStateId>> {
        let search = SearchQuery::new(FIELD_VECTOR, embedding, limit as i32)?;
        Ok(collect_ids(self.collection.query(&search)?))
    }

    /// Flushes buffered writes so a later query sees them.
    pub fn flush(&self) -> Result<()> {
        self.collection.flush()?;
        Ok(())
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

fn collect_ids(docs: Vec<Doc>) -> Vec<TopicStateId> {
    docs.iter()
        .filter_map(|doc| doc.get_pk())
        .filter_map(|pk| uuid::Uuid::parse_str(pk).ok())
        .map(TopicStateId::from)
        .collect()
}
