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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use pamin_core::{Scored, TopicId};

use crate::embedding::{DIMENSIONS, MODEL_ID};
use zvec_rust::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc,
    FieldSchema, Fts, FtsQueryParams, HnswQueryParams, IndexParams, MetricType, SearchQuery,
};

use crate::error::{IndexError, Result};
use crate::segmentation::Segmenter;
use crate::vector_math::dot;

const COLLECTION: &str = "memories";

/// Kept inside the collection so deleting/rebuilding it also discards its floor.
const OPTIMIZED_FILES: &str = ".pamin-optimized-files";

fn count_files(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            if entry.file_name() == OPTIMIZED_FILES {
                return 0;
            }
            match entry.file_type() {
                Ok(kind) if kind.is_dir() => count_files(&entry.path()),
                Ok(_) => 1,
                Err(_) => 0,
            }
        })
        .sum()
}

fn optimized_files_marker(dir: &Path) -> std::path::PathBuf {
    dir.join(COLLECTION).join(OPTIMIZED_FILES)
}

fn saved_optimize_floor(dir: &Path) -> u64 {
    std::fs::read_to_string(optimized_files_marker(dir))
        .ok()
        .and_then(|text| text.strip_prefix("v1 ")?.trim().parse::<u64>().ok())
        .filter(|floor| *floor > 0 && *floor <= count_files(dir))
        .unwrap_or(0)
}

/// The primary key field, and the only one any query here reads back.
///
/// Every channel returns ranks -- the caller resolves what a topic stands for
/// against the ledger -- so the text and the vector the engine would otherwise
/// send back cross the boundary only to be dropped.
const FIELD_ID: &str = "id";

/// Word-level recall, fed pre-segmented text so every language tokenizes well.
const FIELD_SEGMENTED: &str = "content_segmented";
/// Substring recall over raw text: paths, error codes, identifiers.
const FIELD_NGRAM: &str = "content_ngram";
const FIELD_VECTOR: &str = "embedding";
/// What produced a document's vectors ([`Stored::encoding`]). Nullable, and
/// absent from a collection created before it was recorded.
const FIELD_ENCODING: &str = "encoding";

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

/// A topic, its text and its vector: a memory indexed whole, as its one span.
pub type VectorDocument<'a> = (TopicId, &'a str, &'a [f32]);

/// Span `ordinal` of a topic, its text and its vector.
///
/// A memory is indexed as the spans it is cut into (see [`crate::spans`]),
/// each its own document, numbered from 0 with no gap. A memory that fits in
/// one span is one document under ordinal 0, keyed as its topic always was.
pub type SpanDocument<'a> = (TopicId, u32, &'a str, &'a [f32]);
type PendingSpanDocument<'a> = (TopicId, u32, &'a str, Vec<f32>);

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
    ///
    /// A handle rather than a borrow, so a caller can keep tokenizing after it
    /// has let go of the projection. Splitting text touches nothing the
    /// projection owns, and the lock the composition layer holds the projection
    /// behind is there for an engine defect the segmenter has no part in; a
    /// borrow would keep that lock held for work that never needed it.
    fn segmenter(&self) -> Arc<Segmenter>;

    /// Adds or replaces one topic, as one span.
    fn upsert(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<()> {
        self.upsert_batch(&[(topic, content, embedding)])
    }

    /// Adds or replaces many, each as one span.
    fn upsert_batch(&self, documents: &[VectorDocument<'_>]) -> Result<()> {
        self.upsert_encoded(documents, None)
    }

    /// [`upsert_batch`](Self::upsert_batch), recording beside every
    /// document what produced its vectors ([`Stored::encoding`]).
    fn upsert_encoded(
        &self,
        documents: &[VectorDocument<'_>],
        encoding: Option<&str>,
    ) -> Result<()> {
        let spans: Vec<SpanDocument<'_>> = documents
            .iter()
            .map(|(topic, content, embedding)| (*topic, 0, *content, *embedding))
            .collect();
        self.replace(&spans, encoding)
    }

    /// Adds or replaces whole topics: `documents` holds every span of each
    /// topic in it, and whatever spans past those an earlier write of the
    /// topic left behind are removed. A memory rewritten shorter would
    /// otherwise go on being found by text it no longer has.
    fn replace(&self, documents: &[SpanDocument<'_>], encoding: Option<&str>) -> Result<()> {
        self.write_spans(documents, encoding)?;
        let mut ends: BTreeMap<TopicId, u32> = BTreeMap::new();
        for (topic, ordinal, ..) in documents {
            let end = ends.entry(*topic).or_default();
            *end = (*end).max(ordinal + 1);
        }
        self.truncate(&ends.into_iter().collect::<Vec<_>>())
    }

    /// Writes these spans, recording beside each what produced its vector
    /// ([`Stored::encoding`]), and leaves every other span alone.
    ///
    /// So a topic can be written in parts -- a rebuild lends some of its spans
    /// and embeds the rest -- without the first part cutting the topic short;
    /// [`replace`](Self::replace) is the write that says where a topic ends.
    fn write_spans(&self, documents: &[SpanDocument<'_>], encoding: Option<&str>) -> Result<()>;

    /// Removes every span of each topic from ordinal `from` on, for each
    /// `(topic, from)`.
    fn truncate(&self, topics: &[(TopicId, u32)]) -> Result<()>;

    /// Word-level lexical recall, best first.
    fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<Scored>>;

    /// Substring lexical recall over raw text, best first.
    fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<Scored>>;

    /// Lexical recall for documents containing every word of a name.
    ///
    /// A conjunction rather than [`Projection::recall_segmented`]'s ranking of
    /// anything that matches at all. The question behind it is not "what is
    /// most relevant to this name" but "which memories name this thing", and a
    /// two-word name answered by either word alone fills the candidates with
    /// documents carrying only the common half -- so a real match falls off the
    /// end of a bounded list. The caller still confirms each candidate exactly.
    ///
    /// Topics rather than the [`Scored`] the recall channels return, and that is
    /// the point of the difference: the caller confirms every candidate exactly,
    /// so the ranking is thrown away and a score would be a number nothing
    /// reads. The three channels above feed fusion, which is the only thing here
    /// that has a use for one.
    fn recall_naming(&self, name: &str, limit: u32) -> Result<Vec<TopicId>>;

    /// Semantic recall over dense embeddings, nearest first, at the shipped
    /// [`SEARCH_EFFORT`].
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<Scored>> {
        self.recall_vector_at(embedding, limit, SEARCH_EFFORT)
    }

    /// [`recall_vector`](Self::recall_vector) with the in-memory graph
    /// searched `effort` wide: a search's own setting, so a sweep over it
    /// varies the search and nothing else in the process. The on-disk graph
    /// takes its width from the candidates asked for and ignores this.
    fn recall_vector_at(&self, embedding: &[f32], limit: u32, effort: u32) -> Result<Vec<Scored>>;

    /// What this index holds for these topics, as it was written, in the
    /// order asked: each topic's spans in order, and none for a topic it does
    /// not hold.
    ///
    /// Reading a document back is what lets an index be copied rather than
    /// rebuilt: the vector is the one expensive part of a document, and the
    /// index already holds it. A write the index has buffered and not yet
    /// flushed is read back like any other, because a query sees it too.
    fn stored(&self, topics: &[TopicId]) -> Result<Vec<Vec<Stored>>>;

    /// Whether this index already holds each topic with exactly the spans
    /// given for it -- each with exactly its text, and none past them --
    /// embedded under exactly `encoding`, in the order asked.
    ///
    /// The rule a rebuild lends a stored vector by ([`Previous::lend`]), asked
    /// of a served index, and narrowed to the plan that produced the vector: a
    /// document holding the same text was embedded from it under this index's
    /// passage encoding, but only one recorded with the same
    /// [`Stored::encoding`] holds the vector embedding it now would write. A
    /// vector from another execution provider or plan is not bit-identical to
    /// it, so it is never a match, and neither is one whose encoding was not
    /// recorded. Read through [`stored`](Self::stored), so a buffered write
    /// counts, a deleted topic does not, and a document missing a vector it
    /// should have is an error rather than a match.
    fn holds(&self, wanted: &[(TopicId, &[&str])], encoding: &str) -> Result<Vec<bool>> {
        let topics: Vec<TopicId> = wanted.iter().map(|(topic, _)| *topic).collect();
        Ok(self
            .stored(&topics)?
            .iter()
            .zip(wanted)
            .map(|(stored, (_, texts))| {
                holds_spans(
                    stored
                        .iter()
                        .map(|span| (Some(span.content.as_str()), span.encoding.as_deref())),
                    texts,
                    encoding,
                )
            })
            .collect())
    }

    /// Removes these topics, every span of each.
    ///
    /// The projection had no way to shrink: the only route out was deleting the
    /// whole directory. A soft-deleted state therefore stayed in every channel's
    /// candidate budget, so removing content from the ledger quietly reduced how
    /// much a search could find.
    fn delete(&self, topics: &[TopicId]) -> Result<()> {
        let from: Vec<(TopicId, u32)> = topics.iter().map(|topic| (*topic, 0)).collect();
        self.truncate(&from)
    }

    /// Makes buffered writes visible to later queries.
    fn flush(&self) -> Result<()>;

    /// Builds whatever structure makes recall faster than a scan.
    fn optimize(&self) -> Result<()>;

    /// How much of the collection that structure covers, from 0.0 to 1.0.
    fn vector_index_completeness(&self) -> Result<f32>;

    /// How many documents the projection holds.
    fn document_count(&self) -> Result<u64>;

    /// How many files the projection is spread across.
    ///
    /// The resource itself rather than a proxy for it. Every one of these is
    /// held open while the index is, so this is what a descriptor limit is
    /// counting, and it is what decides when the index is asked to tidy up.
    fn file_count(&self) -> Result<u64>;

    /// How many files [`optimize`](Self::optimize) left the last time it ran
    /// for this collection, or zero if no valid floor was saved. See
    /// [`is_fragmented`].
    fn files_after_optimize(&self) -> u64;

    /// How many vector blocks flushes have left that no compaction has merged.
    ///
    /// What disk is spent on, where [`file_count`](Self::file_count) is what
    /// descriptors are spent on. See [`wastes_disk`].
    fn unmerged_blocks(&self) -> Result<u64>;

    /// How this index is segmented, against what the policy would choose.
    fn segmentation(&self) -> Result<Segmentation>;

    /// What text this index's vectors are embedded from, which every write to
    /// it has to follow.
    fn passage(&self) -> Passage;

    /// How a memory written to this index is cut into documents, which every
    /// write to it has to follow.
    fn grain(&self) -> Grain;

    /// Which vector index this one was built with.
    fn vector_index(&self) -> VectorIndex;
}

/// One span of a memory as an index holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stored {
    /// The span's text, exactly as written: for a memory of one span, the
    /// memory's whole content. The segmented field is derived from it, so it
    /// is all a copy needs to rebuild both lexical fields.
    pub content: String,
    /// The vector, as it was embedded under the index's [`Passage`] and then
    /// stored: each component rounded to half precision (see [`crate::as_stored`]).
    pub embedding: Vec<f32>,
    /// What produced the vectors, as [`crate::Embedder::encoding`] names it:
    /// the execution plan of the model. `None` when nothing recorded it --
    /// one written before this was recorded, one lent by a rebuild from a
    /// document that had none, or any in an index created before then -- or
    /// when the plan was not one a record can name.
    pub encoding: Option<String>,
}

/// A lexical or vector index over topics.
pub struct ProjectionIndex {
    collection: Collection,
    segmenter: Arc<Segmenter>,
    dir: std::path::PathBuf,
    /// Which vector index this collection was built with, as its marker
    /// records -- and so how a query asks it.
    index: VectorIndex,
    /// What this index's vectors were embedded from. See [`Passage`].
    passage: Passage,
    /// How this index spells a topic as a primary key. See [`Keys`].
    keys: Keys,
    /// How a memory written to this index is cut. See [`Grain`].
    grain: Grain,
    /// Whether this collection has [`FIELD_ENCODING`]: one created before it
    /// existed does not, and its documents are written and read without it.
    encodings: bool,
    /// What [`Projection::files_after_optimize`] reports.
    optimized_files: AtomicU64,
}

/// What text a document's vector was embedded from.
///
/// Recorded in the index beside the model, for the same reason the model is:
/// two encodings in one index produce distances that mean nothing and look
/// fine. An index built before this existed has no line for it and was built
/// from content alone, which is what it keeps being written with -- so an
/// existing workspace goes on working unchanged, and `pamin reindex`, which
/// builds a fresh index, is what moves it to the current encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Passage {
    /// The memory's content alone. Every index built before names were.
    Content,
    /// `name: content`. A memory's text leaves implicit what its topic's name
    /// says -- a paragraph under "Green (Steve Hillage album)" never names the
    /// album -- and the vector cannot use what it was not shown. Measured on
    /// MuSiQue, embedding the vector channel's documents this way lifts its
    /// nDCG@10 from 0.6221 to 0.6516 (247 questions better, 145 worse).
    Named,
}

impl Passage {
    fn label(self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::Named => "named",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label.trim() {
            "content" => Some(Self::Content),
            "named" => Some(Self::Named),
            _ => None,
        }
    }

    /// The text a memory is embedded from under this encoding.
    pub fn render(self, name: &str, content: &str) -> String {
        match self {
            Self::Content => content.to_string(),
            Self::Named => format!("{name}: {content}"),
        }
    }
}

/// The encoding a new index is built with.
const PASSAGE: Passage = Passage::Named;

/// How a topic's identifier is spelled as the engine's primary key.
///
/// The engine maps primary keys to its own row numbers in a RocksDB instance
/// that it flushes and never compacts, and RocksDB moves a file whose keys
/// overlap nothing below it down a level without merging it. Identifiers are
/// time-ordered (see `pamin_core::id`), so every flush wrote a range of keys
/// above everything before it and left one file behind that nothing ever
/// merged. Measured through `Engine::write` and a drain per sixty-four new
/// memories, 12,800 of them on the `speed` profile: 200 files in the map, one
/// per round, against 1 spelled this way. After an `optimize`, which does not
/// touch the map, that was 252 files in the index against 108 and 215
/// descriptors held by an open against 72 -- and the map's files count
/// against [`MAX_FILES`] like any other, so they spend a budget no
/// compaction can give back. Opening took the same time either way.
///
/// Only a flush that adds and nothing else does it. One that also rewrites an
/// older topic overlaps what came before and is merged with it, so with a
/// quarter of the writes being edits the map held one or two files whatever
/// the spelling. Importing, or a stretch of new memories, is what adds only.
///
/// Recorded in the marker, like the [`Passage`], because reading one spelling
/// as the other matches nothing. What a document stands for is unchanged --
/// that is [`DOCUMENT_GRAIN`] -- so an index keyed the old way opens and goes
/// on being written that way, and a rebuild or a reshape, which write every
/// document again, key the new one the new way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keys {
    /// The identifier as written. Every index built before this existed.
    Topic,
    /// The identifier's sixteen bytes in reverse order, so the random bytes
    /// at its end lead and its timestamp trails. Its own inverse, and a
    /// permutation of the bytes rather than a hash, so a key is exactly one
    /// topic and reads back without anything stored beside it.
    Reversed,
}

impl Keys {
    fn label(self) -> &'static str {
        match self {
            Self::Topic => "topic-keys",
            Self::Reversed => "reversed-keys",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label.trim() {
            "topic-keys" => Some(Self::Topic),
            "reversed-keys" => Some(Self::Reversed),
            _ => None,
        }
    }

    /// The primary key `topic`'s first span is stored under.
    #[cfg(test)]
    fn key(self, topic: TopicId) -> String {
        self.span_key(topic, 0)
    }

    /// The primary key span `ordinal` of `topic` is stored under: the
    /// topic's own for the first, so a memory of one span is keyed as a topic
    /// always was, and `{key}#{ordinal}` for the rest.
    fn span_key(self, topic: TopicId, ordinal: u32) -> String {
        let key = self.spell(topic.0).to_string();
        match ordinal {
            0 => key,
            ordinal => format!("{key}#{ordinal}"),
        }
    }

    /// The topic stored under `key`, whichever of its spans the key is, or
    /// `None` for a key this crate did not write.
    #[cfg(test)]
    fn topic(self, key: &str) -> Option<TopicId> {
        self.span(key).map(|(topic, _)| topic)
    }

    /// The topic and span ordinal stored under `key`, or `None` for a key
    /// this crate did not write. Only the spelling
    /// [`span_key`](Self::span_key) writes is read: no `#0`, no leading zero.
    ///
    /// Every channel reads its topics through this, and drops a key it
    /// returns `None` for without a word -- so a span key read as anything
    /// but its topic would lose that span from search silently.
    fn span(self, key: &str) -> Option<(TopicId, u32)> {
        let (key, ordinal) = match key.split_once('#') {
            None => (key, 0),
            Some((key, ordinal)) => (
                key,
                ordinal
                    .parse::<u32>()
                    .ok()
                    .filter(|&parsed| parsed > 0 && parsed.to_string() == ordinal)?,
            ),
        };
        let topic = uuid::Uuid::parse_str(key).ok()?;
        Some((TopicId::from(self.spell(topic)), ordinal))
    }

    /// Both directions at once, since reversing is its own inverse.
    fn spell(self, id: uuid::Uuid) -> uuid::Uuid {
        match self {
            Self::Topic => id,
            Self::Reversed => {
                let mut bytes = *id.as_bytes();
                bytes.reverse();
                uuid::Uuid::from_bytes(bytes)
            }
        }
    }
}

/// How a new index spells its keys.
const KEYS: Keys = Keys::Reversed;

/// How many segments a collection is aimed at.
///
/// Four, measured. Building 100,000 documents at several segment sizes, against
/// exact search:
///
/// ```text
///   segments   build s   query ms   recall@10
///          1     314.3      11.62      0.8940
///          4     163.0       9.04      0.9920
///         10      80.9      16.90      0.9990
///         40      23.6      22.02      1.0000
/// ```
///
/// One segment is worse than four in three directions at once, and past four
/// the per-segment cost of a query -- about 0.36 ms each -- outgrows what the
/// smaller graphs save. So the count is held near four and the size follows the
/// collection, rather than the other way round.
const TARGET_SEGMENTS: u64 = 4;

/// The largest segment worth sealing, in documents.
///
/// A sealed segment has one graph built over it, once, and that build is a
/// background job. Building is superlinear -- 8.1 s at ten thousand documents,
/// 124 s at fifty thousand, 325.8 s at a hundred thousand -- so the size at which
/// a build stops being a background job and starts being an outage is what caps
/// this. A quarter of a million extrapolates to about twenty minutes, which is
/// the most that should ever be owed to one segment.
///
/// A project past a million documents therefore runs more than four segments
/// rather than larger ones, which is the right way round: the query cost of a
/// segment is linear and the build cost of one is not.
const LARGEST_SEGMENT: u64 = 250_000;

/// The smallest, which is also what every project grown from empty holds.
///
/// A collection records its segment size when it is created, and a workspace
/// is created before anything is written to it, so this floor -- not the
/// division by [`TARGET_SEGMENTS`] -- is the segment size of nearly every
/// project anyone has. It was 2,000, which put 131,924 documents in 66
/// segments, and a segment is not free to hold open: each keeps its own
/// full-text stores resident. Fifty thousand documents open at 1,292 MB in 25
/// segments and at 308 MB in 4, and the MIRACL workspace spent 3,283 MB on
/// opening its index before any model was loaded.
///
/// Ten thousand is the largest size in the table below at which a segment's
/// graph still agrees with an exhaustive scan exactly, and the size at which
/// scanning the segment being written costs 2.7 ms. Twenty-five thousand
/// would cost less memory again and give up recall -- 0.9830 -- which is
/// the axis this project will not trade. So 131,924 documents are 14 segments
/// rather than 66, recall is what it was, and a new project is still one.
const SMALLEST_SEGMENT: u64 = 10_000;

/// How many documents a segment should hold, for a collection of this size.
///
/// This one number is the whole vector-maintenance policy, because the engine
/// makes it do two jobs. Documents land in the segment being written and are
/// searched by scanning them; the segment seals at this size, and only a sealed
/// segment gets a graph built over it. So the size decides both what a query
/// scans and what a build costs, and there is no separate question of when to
/// build -- the answer is "whenever a segment has sealed without one".
///
/// Scanning is not a fallback, it is the faster thing to do at small sizes.
/// Measured on the default profile, one graph against an exhaustive scan:
///
/// ```text
///  documents   scan ms   graph ms   build s   agreement
///      1,000      0.57       0.66       0.3      1.0000
///     10,000      2.70       3.10       8.1      1.0000
///     25,000      5.78       5.76      40.9      0.9830
///     50,000     20.85      10.08     124.0      0.9540
///    100,000     39.58      11.24     325.8      0.8920
/// ```
///
/// The cost of segmenting at all is that BM25 statistics are per segment, so a
/// term's rarity is measured against a segment rather than the project. On the
/// cross-lingual benchmark, six segments against one over the same 13,014
/// sentences moved same-language nDCG@10 from 0.8558 to 0.8517 and recall@50
/// from 0.9639 to 0.9655, while a query went from 208 ms to 63 ms.
///
/// A collection records this when it is created, so a project that has grown
/// by orders of magnitude keeps the size it was created with until something
/// recreates it: a server reshapes it on its own once
/// [`Segmentation::is_worth_rebuilding`] says so, and `pamin reindex` rebuilds
/// it at once.
pub fn segment_documents(documents: u64) -> u64 {
    (documents / TARGET_SEGMENTS).clamp(SMALLEST_SEGMENT, LARGEST_SEGMENT)
}

/// How an index is segmented, against what the policy would choose now.
///
/// The size is recorded when the collection is created and a workspace is
/// created empty, so every grown project records [`SMALLEST_SEGMENT`] and
/// holds one segment per ten thousand documents rather than the four the
/// policy aims at -- 14 over 131,924. Measured over fifty thousand, 25
/// segments answer a query in 39.8 ms where four answer in 16.9.
///
/// Acted on, not only reported. A server checks each open project's shape
/// from its upkeep loop and, when [`is_worth_rebuilding`](Self::is_worth_rebuilding)
/// says so, reshapes the index in the background: a copy taken while the
/// index is served, with every vector reused and the lock held a batch at a
/// time -- see [`crate::Reshape`]. A process holding the index without a
/// server, such as an evaluation harness, has nothing doing that on its own.
/// `pamin cascade drain` reports the shape, and `pamin reindex` rebuilds it,
/// also without embedding anything again; see [`Previous`].
#[derive(Clone, Copy, Debug)]
pub struct Segmentation {
    /// Documents the collection holds.
    pub documents: u64,
    /// Documents a segment holds, as the collection recorded at creation.
    pub recorded: u64,
}

impl Segmentation {
    /// Segments this many documents fall into at the recorded size.
    pub fn segments(&self) -> u64 {
        self.documents.div_ceil(self.recorded.max(1))
    }

    /// Segments the policy would choose for the count it holds now.
    pub fn wanted(&self) -> u64 {
        self.documents
            .div_ceil(segment_documents(self.documents).max(1))
    }

    /// Whether rebuilding would measurably help.
    ///
    /// Twice the target rather than any difference at all, because the target
    /// is a floor as well as a ceiling. Measured over the same fifty thousand
    /// documents, recall@10 against exact search:
    ///
    /// ```text
    ///   segments   recall@10   a query   build
    ///         25      1.0000    39.8 ms    52 s
    ///          4      0.9980    16.9 ms   129 s
    ///          2      0.9880    26.4 ms   221 s
    ///          1      0.9510    16.6 ms   415 s
    /// ```
    ///
    /// **The recall column is the one to read.** It falls monotonically as the
    /// segments grow, reaching 0.9510 at one -- below `recall.rs`'s own 0.97
    /// floor -- so aiming at fewer segments than the policy wants trades
    /// accuracy away, and that is the reason this reports only an excess. The
    /// latency column is not reliable at this resolution: 26.4 ms for two
    /// segments sits above both one and four, which is not a shape anything
    /// physical would produce, and these arms ran while a 131,924-passage
    /// index build had the machine. What survives that is the 25-segment row,
    /// which is 2.4x the four-segment one and reproduced across two runs.
    pub fn is_worth_rebuilding(&self) -> bool {
        self.segments() > 2 * self.wanted().max(1)
    }
}

/// How many files an index may be spread across before it is compacted.
///
/// This is the merge policy, and its shape is not ours: Lucene's
/// `TieredMergePolicy` merges on segments per tier rather than on documents,
/// Qdrant runs an optimizer continuously, and an engine given no such budget
/// pays for it. A write leaves about two files behind whatever the collection
/// holds, so without one the count grows without bound -- ten documents
/// rewritten two hundred times reached eight hundred and forty files, and a
/// workspace used normally for a week died of `Too many open files`.
///
/// The budget is in files rather than writes or documents because files are
/// the resource: the index holds them open, and what runs out is descriptors.
/// Two hundred and fifty-six is one such budget entirely -- the smallest
/// default a supported platform sets -- which is the size at which one index
/// is something a process can hold several of.
///
/// It is also, measured, the point where holding the budget stops costing
/// anything. Two hundred writes over ten topics:
///
/// ```text
///     budget   files held   elapsed   against no compaction
///       none   840, rising     28.3 s                     --
///        256       47..253     27.8 s                  +0.0
///        512      197..442     32.1 s                   +13%
///        128        60..109     37.5 s                  +32%
///      every         16..18     91.1 s                  +221%
/// ```
///
/// The last row is what an engine without a merge policy does when it is asked
/// on every change, and it is not a straw man -- it was the first thing tried.
/// The rows are not monotone between 256 and 512 because at that end the
/// difference is smaller than the run-to-run spread, which is itself the
/// finding: past a couple of hundred files the cost of compacting is no longer
/// what decides the number, so the resource is.
///
/// Those numbers were measured while every write flushed the index, which is
/// what produced the files. Now that a write applies its document and leaves
/// the flush to the server, this is a bound rather than a working limit: three
/// thousand writes through a server leave 136 files and nothing is ever
/// compacted, where two thousand flushed one at a time left 10,031. It is kept
/// for the cases that still reach it -- an import, and a harness writing to an
/// index with no server behind it -- and because a bound that is not being
/// approached is the one worth having.
const MAX_FILES: u64 = 256;

/// Whether an index is spread across more files than it should be.
///
/// `floor` is what the last `optimize` left. When that floor is already above
/// [`MAX_FILES`], the index has to grow by a quarter of the budget before
/// optimizing again. Below the budget, the file ceiling still applies. Without that, an
/// index `optimize` cannot bring under the budget asked for it after every
/// drain that did work: the scalar files of sealed segments are never merged,
/// so an index with two or more sealed segments can stay above the budget for
/// good -- MIRACL's 131,924 passages hold 2,317 files -- and every open adds
/// two empty write-ahead logs. Measured on a 12,814-document index its own
/// `optimize` left at 257 files, one write per five-second upkeep tick
/// re-queued `optimize` 8 times in 60 ticks, about a second of processor time
/// each under `memory`.
///
/// An index without a valid saved floor still uses the budget alone, so legacy
/// indexes are asked once and learn their floor from that.
pub fn is_fragmented(files: u64, floor: u64) -> bool {
    files
        > if floor > MAX_FILES {
            floor.saturating_add(MAX_FILES / 4)
        } else {
            MAX_FILES
        }
}

/// How many unmerged vector blocks an index may hold whatever its size.
///
/// A flush writes the vectors it carries into a block of their own, and the
/// engine sizes that block for a segment rather than for what is in it: 5 MB
/// apparent and 1.06 MB allocated whether it holds one document or forty. So
/// a project written a memory at a time spends a megabyte of disk a flush
/// until something compacts it, and [`MAX_FILES`] is not that something for a
/// small one: a flush leaves about six files, so the file budget is reached
/// after some forty flushes. Measured on the `speed` profile, one write and
/// one flush at a time, with nothing compacting until the end:
///
/// ```text
///   writes   files   allocated   compacted   after closing
///       40     238     45.5 MB      3.4 MB          1.9 MB
///      300   1,538    333.3 MB      7.0 MB          5.6 MB
/// ```
///
/// Compacting changed none of the thirty top-ten lists the three channels
/// returned for ten queries, at eight, forty and three hundred writes.
///
/// Eight blocks is about eight megabytes, which is what a project may carry
/// before anything is spent on it: small enough that a hundred small projects
/// are not four gigabytes of padding, and large enough that one written a
/// memory at a time is not compacted on every flush.
const BLOCK_FLOOR: u64 = 8;

/// Documents per unmerged block an index may carry above [`BLOCK_FLOOR`].
///
/// Scales the allowance with the index, so the waste stays in proportion to
/// what compacting it costs. A compacted index is about 9 KB a document on
/// the shipping profile -- 116 MB over 13,014 -- and a block about a
/// megabyte, so one block per 128 documents lets the waste grow to roughly the
/// size of the index itself before a compaction is asked for. Past about five
/// thousand documents [`MAX_FILES`] is reached first, so this changes nothing
/// for a project that size or larger.
const DOCUMENTS_PER_BLOCK: u64 = 128;

/// Whether an index holds more unmerged vector blocks than its size warrants.
pub fn wastes_disk(blocks: u64, documents: u64) -> bool {
    blocks > BLOCK_FLOOR.max(documents / DOCUMENTS_PER_BLOCK)
}

/// How many documents may sit outside the vector graph before one is built.
///
/// `segment_documents` says the maintenance question has one answer --
/// "whenever a segment has sealed without a graph" -- and the cascade asked it
/// with the wrong instrument. It gated the graph on the file count alone
/// (`is_fragmented`), and that was sound while every write flushed: files grew
/// about two per write, so the budget was reached every sixty or so and a
/// sealed segment never waited long for its graph. Once a write left the flush
/// to the server that stopped being true -- 136 files after three thousand
/// writes, against a budget of 256 -- so the trigger moved from every sixty
/// writes to roughly every five and a half thousand, and a project below that
/// had no graph at all. `cascade.rs` carried the old justification for another
/// release; this is what replaced it.
///
/// The bound is on the *unindexed remainder* rather than on the project,
/// because the remainder is what a query scans and is therefore the thing with
/// a cost. An earlier attempt at this was a threshold on the project's size --
/// a hundred thousand documents -- and it never fired, because how much has
/// ever been written says nothing about how much is outside the graph.
///
/// The value is the crossover in `segment_documents`' own table, where a scan
/// and a graph cost the same: 5.78 ms against 5.76 at twenty-five thousand
/// documents. Below it the scan is the faster of the two and a build would be
/// work spent to go slower, so waiting is right; above it the scan is what the
/// graph exists to replace. So the remainder a query may scan is held at the
/// point where scanning it stops being the cheaper thing, which is a cost
/// rather than a size, and it is read off a measurement already in this file
/// rather than chosen.
pub const UNINDEXED_BUDGET: std::num::NonZeroU64 = std::num::NonZeroU64::new(25_000).unwrap();

/// Whether enough documents sit outside the vector graph to be worth building.
///
/// Takes the completeness rather than reading it, so the policy is testable
/// without an index and the caller does the one FFI call it already makes.
///
/// `budget` is [`UNINDEXED_BUDGET`] wherever the product calls this. It is an
/// argument so a test can lower it: the assertion worth having is that a
/// drain leaves a graph behind, and asserting it at the shipped budget would
/// mean writing and embedding twenty-five thousand memories to see it.
///
/// Never zero: a budget of none would find every drain lagging, so a
/// project with anything queued would be optimized on every pass.
pub fn vector_index_lags(documents: u64, completeness: f32, budget: std::num::NonZeroU64) -> bool {
    let covered = (documents as f64 * f64::from(completeness.clamp(0.0, 1.0))) as u64;
    documents.saturating_sub(covered) >= budget.get()
}

/// What one document in this index stands for.
///
/// Recorded beside the model because an index keyed by something else is not
/// stale, it is silently empty: the old scheme's identifiers are read as the
/// new scheme's, match nothing, and every search comes back with no results
/// and no error anywhere. A grain this build does not know -- the topic state
/// an index was once keyed by -- is refused with a message naming
/// `pamin reindex`.
///
/// Like the [`Passage`], an index keeps the grain it was built with, and a
/// rebuild builds the one its caller asks for. Either grain is read the same
/// way: a document is a span of a topic, keyed by [`Keys::span_key`], and a
/// memory that is one span is keyed and embedded as a whole topic always was.
/// What the grain decides is how a memory is cut when it is written, and so
/// how deep an exact channel asks to find enough topics ([`deepest`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grain {
    /// A whole memory, cut nowhere: one document per topic. Every index built
    /// before spans.
    Topic,
    /// The spans a memory is cut into by [`crate::spans::pack`], each of at
    /// most this many tokens.
    Spans(std::num::NonZeroU32),
}

/// The grain a new or rebuilt index is cut at: spans of at most 200 tokens.
///
/// A memory written the way an agent writes one -- a whole session -- runs
/// past the 256 tokens the reranker reads and the 512 the embedder does, so
/// indexed whole it was judged by its opening. Measured on LoCoMo with one
/// memory per session (1,442 questions, all ten conversations), spans of 200
/// against whole memories: nDCG@10 0.527 to 0.873, every category better
/// (p = 0.002 over conversations), and the first five hits return 56% fewer
/// bytes, for an index 56% larger on disk. Spans of 128 were no better on
/// nDCG@10 (+0.008, p = 0.25), covered the answer less often in the first
/// five hits (-0.053, p = 0.002) and made the index larger still. A memory
/// short enough to be one span is indexed exactly as before.
impl Default for Grain {
    fn default() -> Self {
        Self::Spans(std::num::NonZeroU32::new(200).expect("200 is not zero"))
    }
}

impl Grain {
    fn label(self) -> String {
        match self {
            Self::Topic => "topic".to_string(),
            Self::Spans(cap) => format!("span/{cap}/v1"),
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label.trim() {
            "topic" => Some(Self::Topic),
            label => label
                .strip_prefix("span/")?
                .strip_suffix("/v1")?
                .parse()
                .ok()
                .map(Self::Spans),
        }
    }

    /// The most tokens a span holds, or `None` for a memory indexed whole.
    pub fn cap(self) -> Option<usize> {
        match self {
            Self::Topic => None,
            Self::Spans(cap) => Some(cap.get() as usize),
        }
    }
}

/// How many neighbours each document keeps in the vector graph.
///
/// Measured, on 50,000 clustered 1024-dimensional vectors, against exact
/// nearest neighbours:
///
/// | m | ef_construction | ef | recall@10 | per query |
/// |---|---|---|---|---|
/// | 16 | 100 | 300 (default) | 0.689 | 2.4 ms |
/// | 16 | 500 | 300 | 0.708 | 2.3 ms |
/// | 16 | 500 | 1200 | 0.917 | 7.9 ms |
/// | 16 | 500 | 2048 | 0.952 | 12.6 ms |
/// | 32 | 500 | 300 | 0.862 | 4.3 ms |
/// | **32** | **500** | **700** | **0.952** | **9.5 ms** |
/// | 32 | 500 | 1200 | 0.985 | 13.4 ms |
///
/// The first row is what this shipped: nearly a third of a query's true
/// nearest neighbours missed, on a corpus far smaller than the ones this store
/// is for. Nothing reported it, because a vector channel returning the wrong
/// neighbours returns plausible ones.
///
/// Sixteen to thirty-two doubles the graph, and the graph is the part of an
/// index that quantizing the payload does not shrink. It is still the right
/// trade. `ef` alone can buy most of the recall back on a smaller graph -- 16
/// reaches 0.952 at ef 2048 -- but 2048 is the top of the range the engine
/// accepts, and recall falls as a project grows (the same configuration scores
/// 0.984 at five thousand documents and 0.708 at fifty thousand), so a
/// configuration that needs the maximum at fifty thousand has nothing left at
/// seven million.
const GRAPH_DEGREE: i32 = 32;

// The setting itself -- its two names and its default -- is in
// `pamin_core::settings`, which the command-line client shares without linking
// this crate; what each one builds and asks for is here.
pub use pamin_core::settings::VectorIndex;

/// What one [`VectorIndex`] builds and asks for.
trait Parameters {
    fn params(self) -> Result<IndexParams>;
    fn ask(self, search: &mut SearchQuery, candidates: u32, effort: u32) -> Result<()>;
}

impl Parameters for VectorIndex {
    /// The parameters a collection is created with.
    fn params(self) -> Result<IndexParams> {
        Ok(match self {
            Self::Disk => IndexParams::diskann(
                MetricType::Cosine,
                DISKANN_DEGREE,
                DISKANN_BUILD_LIST,
                DISKANN_PQ_CHUNKS,
            )?,
            Self::Memory => IndexParams::hnsw(MetricType::Cosine, GRAPH_DEGREE, GRAPH_EFFORT)?,
        })
    }

    /// Asks `search` for this index, wide enough for `candidates`, the
    /// in-memory graph searched `effort` wide.
    fn ask(self, search: &mut SearchQuery, candidates: u32, effort: u32) -> Result<()> {
        match self {
            Self::Disk => search.set_diskann_params(DiskannQueryParams::new(
                DISKANN_SEARCH_LIST.max(candidates as i32),
            ))?,
            Self::Memory => {
                let effort = i32::try_from(effort).unwrap_or(i32::MAX);
                search.set_hnsw_params(HnswQueryParams::new(effort, 0.0, false, false))?
            }
        }
        Ok(())
    }
}

/// What a marker with no storage line records: every index built before the
/// line existed stored its vectors as fp32.
const LEGACY_STORAGE: &str = "fp32";

/// How many neighbours each document keeps in the on-disk graph.
///
/// Sixty-four, which is where every DiskANN figure here was taken. The build
/// is what this index costs most -- 922 to 1,024 s for 50,000 clustered
/// 1024-dimensional vectors in four segments on a shared four-core machine,
/// against 58 s for [`VectorIndex::Memory`] -- so a degree that recalled as
/// much for less build would be the one to take; none was measured to.
const DISKANN_DEGREE: i32 = 64;

/// How wide the build searches to place each document in the on-disk graph.
///
/// One hundred, within the audited 0.7.2 prebuilt engine's build-list ceiling.
/// A historical run requested 200 but did not record the effective native
/// value. The audited prebuilt clamps this parameter to 100; the source-build
/// fallback follows upstream's default branch and can differ. Those historical
/// timings cannot establish the effect of widening it. See ADR 0001.
const DISKANN_BUILD_LIST: i32 = 100;

/// Requested product-quantization chunks for on-disk graph navigation.
///
/// Zero is the measured requested setting, not proof of disabled PQ. The
/// 0.7.2 release's referenced native builder treats zero as an automatic
/// dimension/2 count; historical runs did not record the effective count.
/// The requested-64 arm answered in about a third of the time at the same
/// width and recalled 0.9200 at ten and 0.8629 at fifty, against 0.9985 and
/// 0.9975 for requested zero, and built in 1,859 s against 922 to 1,024.
/// These observations do not establish a no-PQ versus PQ comparison. See
/// ADR 0001 and the native-source audit for provenance and its limits.
const DISKANN_PQ_CHUNKS: i32 = 0;

/// How wide a query searches the on-disk graph, at the least: one asking for
/// more candidates than this searches as wide as it asks.
///
/// Measured over 50,000 clustered 1024-dimensional vectors in four segments,
/// with the rescore, against exact search -- fp32 HNSW recalls 0.9980 at ten
/// and 0.9974 at fifty there:
///
/// | width | recall@10 | recall@50 | a query, shared machine |
/// | --- | --- | --- | --- |
/// | 300 | 0.9810 | 0.9715 | 20 ms |
/// | 500 | 0.9955 | 0.9901 | 62 ms |
/// | 800 | 0.9975 | 0.9952 | 82 ms |
/// | **1,200** | **0.9985** | **0.9975** | **114 ms** |
/// | 2,000 | 0.9985 | 0.9986 | 186 ms |
///
/// The graph is what limits it rather than the arithmetic -- the rescore adds
/// at most 0.0015 at any width -- and 1,200 is the narrowest measured to stay
/// within 0.002 of fp32 at both depths. The milliseconds are from a machine
/// at load 12 to 13 and are a direction, not a figure; see ADR 0001.
const DISKANN_SEARCH_LIST: i32 = 1_200;

/// How many candidates either index is asked for per result kept, before they
/// are ranked again in f32.
///
/// The engine scores a half-precision field by multiplying and accumulating
/// in half precision on a CPU with AVX-512 FP16 (upstream,
/// `inner_product_distance_batch_impl_fp16_avx512fp16.cc`), so its scores are
/// off by about 4e-4 where rounding the vectors moves a cosine by about 1e-5,
/// and an HNSW graph over such a field recalled 0.9650 of the exact top ten on
/// 50,000 clustered vectors where fp32 recalled 0.9975. Reading the stored
/// vectors back with the candidates and ranking them by an exact f32 cosine
/// removes the arithmetic's error and leaves only the rounding's. Twice the
/// candidates is where recall stops rising: at k = 10 it restored 0.9970 on
/// the synthetic set and 0.9998 on MIRACL's 131,924 passages -- what exact
/// search over the rounded vectors recalls -- and four times gained nothing.
const RESCORE: u32 = 2;

/// How hard the build works to place each document in the graph.
///
/// Five hundred is the engine's own default and this had been at 100. It costs
/// build time and nothing at query time: 50,000 documents take 21 s at 16/100
/// and 126 s at 32/500, and a rebuild of a large project is measured in hours
/// either way.
const GRAPH_EFFORT: i32 = 500;

// The width a search asks for by default, with what settled it; in
// `pamin_core::search` beside the other defaults the command line shows.
pub use pamin_core::search::SEARCH_EFFORT;

impl ProjectionIndex {
    /// Opens the index at `dir`, creating it if absent.
    ///
    /// What the index was built for is recorded on creation and checked on
    /// every reopen: the embedding model, because mixing embedding spaces
    /// produces distances that mean nothing, and what a document stands for,
    /// because reading one scheme's keys as another's matches nothing at all.
    /// Neither failure looks like a failure -- one returns plausible rankings
    /// from meaningless distances and the other returns no results and no
    /// error -- so both are enforced rather than documented. So is the vector
    /// index, for the same reason: see [`VectorIndex`].
    pub fn open(
        dir: &Path,
        legacy_dir: &Path,
        index: VectorIndex,
        access: Access,
        documents: u64,
    ) -> Result<Self> {
        Self::open_as(dir, legacy_dir, index, access, documents, Grain::default())
    }

    /// [`open`](Self::open), creating an absent index with `grain`; an index
    /// that exists keeps the grain it was built with.
    pub fn open_as(
        dir: &Path,
        legacy_dir: &Path,
        index: VectorIndex,
        access: Access,
        documents: u64,
        grain: Grain,
    ) -> Result<Self> {
        // A workspace built before projects had their own directory holds one
        // shared collection. Opening this project's empty directory beside it
        // would return nothing and look like an empty workspace, so it is
        // reported instead.
        if std::fs::exists(legacy_dir)? {
            return Err(IndexError::LegacyLayout);
        }

        Self::open_sized(dir, index, access, segment_documents(documents), grain)
    }

    /// The vector index the index at `dir` was built with, if there is one
    /// there that this build can open.
    ///
    /// For a caller that has to open an index nobody asked it to and so has
    /// no vector index in hand: the resident server, bringing up to date a
    /// project that was written to by a process that is gone. Opening under a
    /// guess is not an option -- a wrong guess is refused, and a guess at a
    /// directory with no index creates one.
    ///
    /// `None` for no index, and for one recorded with a vector index no build
    /// makes any more, which only a rebuild can open. An index recorded with
    /// another embedding model is [`IndexError::ModelMismatch`], so that a
    /// caller can say which and that `pamin reindex` rebuilds it, rather than
    /// passing over it as if there were nothing there.
    pub fn built_for(dir: &Path) -> Result<Option<VectorIndex>> {
        let Some(recorded) = Marker::read(dir)? else {
            return Ok(None);
        };
        if recorded.model != MODEL_ID {
            return Err(IndexError::ModelMismatch {
                indexed: recorded.model,
            });
        }
        Ok(VectorIndex::parse(&recorded.storage))
    }

    /// Opens the index at `dir`, creating it with segments of `segment`
    /// documents if absent.
    ///
    /// The size is the caller's here rather than derived from a count, because
    /// the reshape's tests have to build an index in the shape a grown project
    /// is in -- many segments -- without writing fifty thousand documents.
    ///
    /// `grain` is what an index created here is built with; one that exists
    /// keeps its own.
    pub(crate) fn open_sized(
        dir: &Path,
        index: VectorIndex,
        access: Access,
        segment: u64,
        grain: Grain,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (passage, keys, grain) = match Marker::read(dir)? {
            Some(recorded) => {
                if recorded.model != MODEL_ID {
                    return Err(IndexError::ModelMismatch {
                        indexed: recorded.model,
                    });
                }
                let Some(grain) = Grain::parse(&recorded.grain) else {
                    return Err(IndexError::GrainMismatch {
                        // A marker with no grain line was written before there
                        // was one, and everything written then was keyed by
                        // state.
                        indexed: if recorded.grain.is_empty() {
                            "topic state".to_string()
                        } else {
                            recorded.grain
                        },
                        expected: Grain::Topic.label(),
                    });
                };
                if recorded.storage != index.label() {
                    return Err(IndexError::VectorIndexMismatch {
                        indexed: recorded.storage,
                        requested: index.label().to_string(),
                    });
                }
                (recorded.passage, recorded.keys, grain)
            }
            None => {
                Marker::current(index, grain).write(dir)?;
                (PASSAGE, KEYS, grain)
            }
        };

        let mut opened = Self::open_with_dimensions(dir, DIMENSIONS, index, access, segment)?;
        opened.passage = passage;
        opened.keys = keys;
        opened.grain = grain;
        Ok(opened)
    }

    /// Creates an empty index at `dir` that records what the one at `source`
    /// does, sized for `documents`.
    ///
    /// The marker is copied rather than written afresh, so the copy keeps the
    /// source's encoding: a copy of an index whose vectors were embedded from
    /// content alone must go on being written that way, and writing the
    /// current marker would label those vectors `name: content`. Reopening
    /// then checks the copied marker against the model like any other open.
    ///
    /// Except for the [`Keys`]: the copy is written a document at a time
    /// through this index, which spells every key itself, so nothing of the
    /// source's spelling survives into it and the copy takes the current one.
    ///
    /// Whatever is at `dir` already is discarded first: it can only be a copy
    /// that did not finish.
    pub(crate) fn create_beside(
        source: &Path,
        dir: &Path,
        index: VectorIndex,
        documents: u64,
    ) -> Result<Self> {
        Self::discard(dir)?;
        std::fs::create_dir_all(dir)?;
        let recorded = Marker::read(source)?.ok_or_else(|| {
            IndexError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no marker at {}", source.display()),
            ))
        })?;
        Marker {
            keys: KEYS,
            ..recorded
        }
        .write(dir)?;
        Self::open_sized(
            dir,
            index,
            Access::ReadWrite,
            segment_documents(documents),
            Grain::default(),
        )
    }

    /// Opens the index at `dir` for writing, refusing to create one.
    ///
    /// For reopening an index after its directory was moved into place, where
    /// finding nothing there is a fault: an open that created an empty index
    /// would hand the caller a project with no memories and no error.
    pub(crate) fn reopen(dir: &Path, index: VectorIndex) -> Result<Self> {
        if !std::fs::exists(dir.join(COLLECTION))? {
            return Err(IndexError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no index at {}", dir.display()),
            )));
        }
        Self::open_sized(
            dir,
            index,
            Access::ReadWrite,
            segment_documents(0),
            Grain::default(),
        )
    }

    fn open_with_dimensions(
        dir: &Path,
        dimensions: u32,
        index: VectorIndex,
        access: Access,
        segment: u64,
    ) -> Result<Self> {
        // The engine's defaults, and an explicit `memory_limit` was measured
        // rather than assumed away. The limit sizes the buffer pool the engine
        // reads vectors through when a collection was created with mmap off,
        // and every collection here is created with the default, mmap on, so
        // nothing reads it. Over XQuAD-R's 13,014 documents, 256 MiB, 2 GiB
        // and none opened at the same resident set (243 MB), returned the same
        // results for 400 vector and 400 lexical queries, and built in the
        // same time and peak within noise (25.2-26.4 s, 826-873 MB). ADR 0001.
        INITIALIZE.call_once(|| {
            let _ = zvec_rust::initialize(None);
        });

        std::fs::create_dir_all(dir)?;
        let path = dir.join(COLLECTION);

        let schema = CollectionSchema::builder(COLLECTION)
            .add_field(FieldSchema::new(FIELD_ID, DataType::String, false, 0)?)
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
            // Not indexed: read back by key, never searched.
            .add_field(FieldSchema::new(FIELD_ENCODING, DataType::String, true, 0)?)
            // Half precision, and read back by the rescore; see `RESCORE`.
            .add_vector_field(
                FIELD_VECTOR,
                DataType::VectorFp16,
                dimensions,
                index.params()?,
            )
            .max_doc_count_per_segment(segment)
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
        let encodings = collection.schema()?.has_field(FIELD_ENCODING);

        Ok(Self {
            collection,
            segmenter: Arc::new(Segmenter::new()),
            dir: dir.to_path_buf(),
            index,
            passage: PASSAGE,
            keys: KEYS,
            grain: Grain::default(),
            encodings,
            optimized_files: AtomicU64::new(saved_optimize_floor(dir)),
        })
    }

    fn document(
        &self,
        (topic, ordinal, content, embedding): SpanDocument<'_>,
        encoding: Option<&str>,
    ) -> Result<Doc> {
        let mut doc = Doc::new()?;
        let key = self.keys.span_key(topic, ordinal);
        doc.set_pk(&key);
        doc.add_string(FIELD_ID, &key)?;
        doc.add_string(FIELD_SEGMENTED, &self.segmenter.segment_for_index(content))?;
        doc.add_string(FIELD_NGRAM, content)?;
        crate::half::add(&mut doc, FIELD_VECTOR, embedding)?;
        if let Some(encoding) = encoding.filter(|_| self.encodings) {
            doc.add_string(FIELD_ENCODING, encoding)?;
        }
        Ok(doc)
    }

    fn recall_text(&self, field: &str, query: &str, limit: u32) -> Result<Vec<Scored>> {
        self.recall_fts(field, query, limit, false)
    }

    fn recall_dense(
        &self,
        field: &str,
        embedding: &[f32],
        limit: u32,
        effort: u32,
    ) -> Result<Vec<Scored>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let length = dot(embedding, embedding).sqrt();
        deepening(limit, |depth| {
            let candidates = depth.saturating_mul(RESCORE);
            let mut search =
                SearchQuery::new(field, &crate::half::query(embedding), candidates as i32)?;
            search.set_output_fields(&[FIELD_ID])?;
            search.set_include_vector(true)?;
            self.index.ask(&mut search, candidates, effort)?;

            let docs = self.collection.query(&search)?;
            let full = docs.len() >= candidates as usize;
            let mut scored = Vec::with_capacity(docs.len());
            for doc in docs {
                let Some((topic, ordinal)) = doc.get_pk().and_then(|key| self.keys.span(key))
                else {
                    continue;
                };
                let vector = crate::half::get(&doc, field)?.ok_or_else(|| {
                    IndexError::Engine(format!(
                        "a candidate for topic {topic} came without its {field} vector"
                    ))
                })?;
                let lengths = length * dot(&vector, &vector).sqrt();
                let similarity = if lengths > 0.0 {
                    dot(embedding, &vector) / lengths
                } else {
                    0.0
                };
                scored.push((topic, ordinal, similarity));
            }
            // Stable, so spans the cosine cannot tell apart keep the order
            // the index proposed them in, as topics always did.
            scored.sort_by(|left, right| right.2.total_cmp(&left.2));
            Ok((collapse(scored), full))
        })
    }

    /// Lexical recall, either ranking whatever matches or requiring every term.
    fn recall_fts(
        &self,
        field: &str,
        query: &str,
        limit: u32,
        every_term: bool,
    ) -> Result<Vec<Scored>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        deepest(self.grain, limit, |depth| {
            let mut fts = Fts::new()?;
            fts.set_match_string(query)?;
            let mut search = SearchQuery::fts(field, &fts, depth as i32)?;
            search.set_output_fields(&[FIELD_ID])?;
            if every_term {
                search.set_fts_params(FtsQueryParams::new(Some("AND"))?)?;
            }

            // The BM25 score leaves with each candidate. It is still not
            // comparable with a vector distance, and fusion still combines the
            // channels by rank for exactly that reason -- but this function used to
            // destroy the score instead of merely declining to compare it, which
            // left every layer above unable to tell a channel that found the answer
            // from one that returned the least bad of fifty wrong documents. Reading
            // it costs one accessor per candidate on a result set already in memory.
            // BM25, where larger is already better.
            let docs = self.collection.query(&search)?;
            // Equal scores in the order of their keys, not the engine's: which
            // of them leads a deeper list depends on how the index is laid out
            // in segments, and a reshape must not change what a search finds.
            let mut scored = collect_scored(self.keys, docs, |score| score);
            scored.sort_by(|left, right| {
                right
                    .2
                    .total_cmp(&left.2)
                    .then_with(|| (left.0, left.1).cmp(&(right.0, right.1)))
            });
            Ok(collapse(scored))
        })
    }

    /// The scalar fields [`fetch`](Self::fetch) reads back: the text, and what
    /// produced the vectors when this collection records it.
    fn stored_fields(&self) -> &'static [&'static str] {
        if self.encodings {
            &[FIELD_NGRAM, FIELD_ENCODING]
        } else {
            &[FIELD_NGRAM]
        }
    }

    /// What produced a fetched document's vectors, if that was recorded.
    fn encoding_of(&self, doc: &Doc) -> Result<Option<String>> {
        if !self.encodings || !doc.has_field(FIELD_ENCODING) || doc.is_field_null(FIELD_ENCODING) {
            return Ok(None);
        }
        Ok(doc.get_string(FIELD_ENCODING)?)
    }

    /// The documents stored under these topics, keyed by the topic each is.
    ///
    /// How a reshape reads the index it is copying, and it has to be keyed:
    /// it reads a served index that goes on taking writes, a batch at a time
    /// under the owner's lock, and afterwards reads again exactly the topics
    /// written meanwhile. The engine's document iterator, which a rebuild
    /// lends through (see [`Previous`]), would seal a writable collection's
    /// open segment each time one was made and block its `optimize` while
    /// open. The text comes from the n-gram field, which holds the content
    /// verbatim; the segmented one is derived from it. `vectors` says whether
    /// to bring the vectors along, which only a caller that copies them needs.
    ///
    /// Every span of each topic, in order. Ordinals are written from 0 with
    /// no gap, so a topic's spans end at the first ordinal it does not hold,
    /// and each round asks only for the next ordinal of the topics that
    /// still had one.
    fn fetch(&self, topics: &[TopicId], vectors: bool) -> Result<HashMap<TopicId, Vec<Doc>>> {
        let mut stored: HashMap<TopicId, Vec<Doc>> = HashMap::with_capacity(topics.len());
        let mut asked = HashSet::with_capacity(topics.len());
        let mut next: Vec<(TopicId, u32)> = topics
            .iter()
            .filter(|topic| asked.insert(**topic))
            .map(|topic| (*topic, 0))
            .collect();
        while !next.is_empty() {
            let mut found = self.fetch_keys(&next, Some(self.stored_fields()), vectors)?;
            next.retain(|span| found.contains_key(span));
            for span in &mut next {
                let doc = found.remove(span).expect("retained as found");
                stored.entry(span.0).or_default().push(doc);
                span.1 += 1;
            }
        }
        Ok(stored)
    }

    /// The documents stored under these spans' keys, by span.
    fn fetch_keys(
        &self,
        spans: &[(TopicId, u32)],
        fields: Option<&[&str]>,
        vectors: bool,
    ) -> Result<HashMap<(TopicId, u32), Doc>> {
        let keys: Vec<String> = spans
            .iter()
            .map(|(topic, ordinal)| self.keys.span_key(*topic, *ordinal))
            .collect();
        let mut found = HashMap::with_capacity(keys.len());
        for chunk in keys.chunks(WRITE_BATCH) {
            let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
            for doc in self
                .collection
                .fetch_with_options(&chunk, fields, vectors)?
            {
                if let Some(span) = doc.get_pk().and_then(|key| self.keys.span(key)) {
                    found.insert(span, doc);
                }
            }
        }
        Ok(found)
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

/// What an index records it was built for, in its `profile` file.
///
/// Five lines: the embedding model, what a document stands for, how vectors
/// are stored, what text they were embedded from, and how a topic is spelled
/// as a key. A line an older index does not have reads as what that index was
/// built with: no storage line is `fp32`, no passage line is content alone, no
/// key line is the topic as written.
///
/// The storage line is kept as written rather than parsed into a
/// [`VectorIndex`], because an index built before those existed records
/// something else -- `fp32`, or whatever a sweep built -- and has to be named
/// when it is refused and read when a rebuild lends from it.
struct Marker {
    model: String,
    grain: String,
    storage: String,
    passage: Passage,
    keys: Keys,
}

impl Marker {
    const FILE: &str = "profile";

    /// What an index built now, with this vector index, is.
    fn current(index: VectorIndex, grain: Grain) -> Self {
        Self {
            model: MODEL_ID.to_string(),
            grain: grain.label(),
            storage: index.label().to_string(),
            passage: PASSAGE,
            keys: KEYS,
        }
    }

    /// The marker in `dir`, or `None` for an index that has none yet.
    fn read(dir: &Path) -> Result<Option<Self>> {
        let recorded = match std::fs::read_to_string(dir.join(Self::FILE)) {
            Ok(recorded) => recorded,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut lines = recorded.trim().lines();
        Ok(Some(Self {
            model: lines.next().unwrap_or_default().trim().to_string(),
            grain: lines.next().unwrap_or_default().trim().to_string(),
            storage: lines.next().map_or(LEGACY_STORAGE, str::trim).to_string(),
            passage: lines
                .next()
                .and_then(Passage::parse)
                .unwrap_or(Passage::Content),
            keys: lines.next().and_then(Keys::parse).unwrap_or(Keys::Topic),
        }))
    }

    fn write(&self, dir: &Path) -> Result<()> {
        std::fs::write(
            dir.join(Self::FILE),
            format!(
                "{}\n{}\n{}\n{}\n{}",
                self.model,
                self.grain,
                self.storage,
                self.passage.label(),
                self.keys.label()
            ),
        )?;
        Ok(())
    }

    /// Whether a vector this index holds is the vector an index built now
    /// would compute for the same text: same model, same encoding.
    ///
    /// Not the same [`Grain`]: a vector is embedded from a span's header and
    /// text whichever grain cut it, and a memory is lent only when the index
    /// holds exactly the spans the rebuild would write ([`holds_spans`]) -- so
    /// a memory a span index holds whole lends to a topic index, and the
    /// reverse, and nothing cut differently lends at all.
    ///
    /// Not the same [`Keys`]: how a key is spelled says nothing about the
    /// vector stored under it, and the index lending it is read through its
    /// own spelling. Nor the same storage: every index built now stores half
    /// precision, and a vector read from any of them -- or from an fp32 index
    /// built before -- rounds to the same codes the model's own output would.
    fn matches(&self, other: &Self) -> bool {
        self.model == other.model && self.passage == other.passage
    }

    /// Whether the index this describes stores half-precision vectors, as
    /// every [`VectorIndex`] does; an index built before them stored fp32.
    fn is_half(&self) -> bool {
        VectorIndex::parse(&self.storage).is_some()
    }
}

/// A project's index, moved aside by a rebuild so the rebuild can reuse the
/// vectors it already holds.
///
/// A rebuild restates the index from the ledger, and the ledger stays the
/// authority: every topic's current state is read from it and written again.
/// What the old index can still supply is the one expensive part, the vector,
/// and only where it is certainly the vector the rebuild would compute -- the
/// stored text is the state's text exactly, and the marker says the same
/// model, storage and encoding produced it. A topic's name is fixed for its
/// life, so under the `name: content` encoding the same content is the same
/// passage.
///
/// That turns reshaping an index into copying it. A project grown from empty
/// held one segment per two thousand documents -- 66 over 131,924 -- and
/// every segment keeps its own full-text store resident: opening fifty thousand
/// documents in 25 segments costs 1,292 MB where 4 cost 308 MB. A rebuild that
/// had to embed every memory again took hours at that size and needed the
/// model; one that reuses what is stored takes neither.
pub struct Previous {
    index: ProjectionIndex,
    dir: std::path::PathBuf,
    /// Whether its vectors are half precision; an index from before the
    /// [`VectorIndex`]es stores fp32.
    half: bool,
}

impl Previous {
    /// Moves the index in `dir` aside and opens it to lend its vectors.
    ///
    /// `None`, with the directory discarded, when there is no index or when
    /// its vectors were computed some other way than an index built now would
    /// compute them. A directory left aside by a rebuild that did not finish is
    /// discarded first: what is in `dir` is then that rebuild's partial output
    /// or its finished one, and the ledger reproduces either.
    pub fn set_aside(dir: &Path) -> Result<Option<Self>> {
        let aside = dir.with_extension("previous");
        ProjectionIndex::discard(&aside)?;
        let current = Marker::current(VectorIndex::default(), Grain::default());
        let Some(recorded) = Marker::read(dir)?.filter(|recorded| {
            recorded.matches(&current) && Grain::parse(&recorded.grain).is_some()
        }) else {
            ProjectionIndex::discard(dir)?;
            return Ok(None);
        };
        std::fs::rename(dir, &aside)?;
        // The collection exists, so the vector index passed here only fills a
        // schema the engine does not read.
        let mut index = ProjectionIndex::open_with_dimensions(
            &aside,
            DIMENSIONS,
            VectorIndex::default(),
            Access::ReadOnly,
            segment_documents(0),
        )?;
        index.passage = PASSAGE;
        index.keys = recorded.keys;
        index.grain = Grain::parse(&recorded.grain).expect("parsed when it was set aside");
        Ok(Some(Self {
            index,
            dir: aside,
            half: recorded.is_half(),
        }))
    }

    /// How many of these topics [`lend`](Self::lend) would supply, without
    /// reading a vector.
    ///
    /// `wanted` maps each topic to the texts of the spans the rebuild will
    /// write for it, in order.
    pub fn lends(&self, wanted: &HashMap<TopicId, Vec<&str>>) -> Result<usize> {
        Ok(self.whole(wanted)?.len())
    }

    /// Hands `take` every document this index can lend, at most `batch` at a
    /// time, as the span, the text wanted for it and its stored vector, with
    /// what produced those vectors ([`Stored::encoding`]); returns the topics
    /// it lent, every span of each.
    ///
    /// Each batch shares one encoding, so the caller can record it as this
    /// index did: a rebuild that dropped it would leave every lent document
    /// one the next sync of it has to embed again. The bound holds across
    /// encodings, not per encoding: a long-lived index can carry many, and
    /// waiting for each to fill a batch would hold most of the collection's
    /// vectors at once. So whenever `batch` documents are waiting in all,
    /// every encoding's are handed over.
    ///
    /// A topic is lent when this index holds it with exactly the spans in
    /// `wanted` (see [`holds_spans`]); everything else is the caller's to
    /// embed, and nothing of it is lent. The documents are written with
    /// [`Projection::write_spans`], so a topic whose spans arrive in two
    /// batches is not cut short by the first. In the order this
    /// index holds its documents rather than the caller's, because that is
    /// the order they are read in: one pass over the collection rather than a
    /// keyed fetch per batch. Over MIRACL's 131,924 documents a pass takes
    /// 0.28 s this way against 1.46-1.59 s of keyed fetches, with the same
    /// resident set and identical vectors and text (ADR 0001).
    pub fn lend(
        &self,
        wanted: &HashMap<TopicId, Vec<&str>>,
        batch: usize,
        mut take: impl FnMut(&[SpanDocument<'_>], Option<&str>) -> Result<()>,
    ) -> Result<HashSet<TopicId>> {
        let lent = self.whole(wanted)?;
        let mut pending: BTreeMap<Option<String>, Vec<PendingSpanDocument<'_>>> = BTreeMap::new();
        let mut waiting = 0;
        let mut hand_over =
            |encoding: &Option<String>, pending: &mut Vec<PendingSpanDocument<'_>>| {
                let documents: Vec<SpanDocument<'_>> = pending
                    .iter()
                    .map(|(topic, ordinal, content, vector)| {
                        (*topic, *ordinal, *content, vector.as_slice())
                    })
                    .collect();
                take(&documents, encoding.as_deref())?;
                pending.clear();
                Ok::<_, IndexError>(())
            };
        self.each(true, |(topic, ordinal), doc| {
            if !lent.contains(&topic) {
                return Ok(());
            }
            // The vector field is not nullable and every write here carries
            // one, so a document without it is an engine fault -- and the
            // engine's iterator fails before handing one over anyway.
            let vector = if self.half {
                crate::half::get(&doc, FIELD_VECTOR)?
            } else {
                doc.get_vector_f32(FIELD_VECTOR)?
            };
            let vector = vector.ok_or_else(|| {
                IndexError::Engine(format!("the document for topic {topic} has no vector"))
            })?;
            let encoding = self.index.encoding_of(&doc)?;
            pending.entry(encoding).or_default().push((
                topic,
                ordinal,
                wanted[&topic][ordinal as usize],
                vector,
            ));
            waiting += 1;
            if waiting == batch {
                for (encoding, group) in &mut pending {
                    if !group.is_empty() {
                        hand_over(encoding, group)?;
                    }
                }
                waiting = 0;
            }
            Ok(())
        })?;
        for (encoding, group) in &mut pending {
            if !group.is_empty() {
                hand_over(encoding, group)?;
            }
        }
        Ok(lent)
    }

    /// The topics of `wanted` this index holds with exactly the spans given
    /// for them: every one, each with its text, and none past them.
    ///
    /// A pass of its own, without vectors, before any is lent: a topic is
    /// lent whole or not at all, and whether it is whole is known only once
    /// every document has been seen.
    fn whole(&self, wanted: &HashMap<TopicId, Vec<&str>>) -> Result<HashSet<TopicId>> {
        let mut matched: HashMap<TopicId, usize> = HashMap::new();
        let mut spoiled: HashSet<TopicId> = HashSet::new();
        self.each(false, |(topic, ordinal), doc| {
            let Some(texts) = wanted.get(&topic) else {
                return Ok(());
            };
            match texts.get(ordinal as usize) {
                Some(text) if holds_exactly(doc.get_string(FIELD_NGRAM)?.as_deref(), text) => {
                    *matched.entry(topic).or_default() += 1;
                }
                _ => {
                    spoiled.insert(topic);
                }
            }
            Ok(())
        })?;
        Ok(matched
            .into_iter()
            .filter(|(topic, count)| *count == wanted[topic].len() && !spoiled.contains(topic))
            .map(|(topic, _)| topic)
            .collect())
    }

    /// Calls `visit` for every document this index holds, with the span it
    /// is.
    ///
    /// Through the engine's document iterator, which reads a snapshot and has
    /// three constraints this meets by construction: made on a writable
    /// collection it seals the segment being written, which this index does
    /// not have because it was opened read-only; nothing can `optimize` a
    /// collection while one is open, and nothing optimizes a set-aside index;
    /// and it fails on a document without a vector when vectors are asked
    /// for, which the schema does not allow.
    fn each(
        &self,
        vectors: bool,
        mut visit: impl FnMut((TopicId, u32), Doc) -> Result<()>,
    ) -> Result<()> {
        let index = &self.index;
        for doc in index
            .collection
            .iter_with_options(Some(index.stored_fields()), vectors)?
        {
            let doc = doc?;
            let Some(span) = doc.get_pk().and_then(|key| index.keys.span(key)) else {
                continue;
            };
            visit(span, doc)?;
        }
        Ok(())
    }

    /// Deletes the set-aside index, once the rebuild no longer needs it.
    pub fn discard(self) -> Result<()> {
        let Self { index, dir, .. } = self;
        drop(index);
        ProjectionIndex::discard(&dir)
    }
}

/// Whether a document holding `stored` as its text already holds `wanted`.
///
/// Raw text, not normalised: the lexical fields are derived from this text and
/// the vector was embedded from it, so a difference in whitespace or case is a
/// different document. One rule for both a rebuild lending a vector and a
/// served index skipping a write, so the two cannot drift apart.
fn holds_exactly(stored: Option<&str>, wanted: &str) -> bool {
    stored == Some(wanted)
}

impl Projection for ProjectionIndex {
    fn passage(&self) -> Passage {
        self.passage
    }

    fn grain(&self) -> Grain {
        self.grain
    }

    fn vector_index(&self) -> VectorIndex {
        self.index
    }

    /// The segmenter this index tokenizes with.
    ///
    /// Shared rather than duplicated so that anything comparing text against
    /// indexed content splits it the same way this index did. A second
    /// segmenter would be the same code today and a divergence the first time
    /// either side changed.
    fn segmenter(&self) -> Arc<Segmenter> {
        Arc::clone(&self.segmenter)
    }

    /// Writes these spans.
    ///
    /// Chunked because the engine refuses a write of more than [`WRITE_BATCH`]
    /// documents. Rebuilding used to write one document per call, which pays
    /// the per-call cost once per document; here it is once per batch.
    ///
    /// No flush: a caller writing in batches decides when the result becomes
    /// visible, and flushing between batches would make that decision for them
    /// once per batch.
    fn write_spans(&self, documents: &[SpanDocument<'_>], encoding: Option<&str>) -> Result<()> {
        for chunk in documents.chunks(WRITE_BATCH) {
            let docs = chunk
                .iter()
                .map(|document| self.document(*document, encoding))
                .collect::<Result<Vec<_>>>()?;

            let refs: Vec<&Doc> = docs.iter().collect();
            self.collection.upsert(&refs)?;
        }

        Ok(())
    }

    /// One keyed read per ordinal, of only the topics that still had one:
    /// ordinals have no gap, so a topic's spans end at the first one missing,
    /// and what is deleted is only what was found. A write of memories that
    /// were one span and still are reads one key each and deletes nothing.
    fn truncate(&self, topics: &[(TopicId, u32)]) -> Result<()> {
        let mut next: Vec<(TopicId, u32)> = topics.to_vec();
        while !next.is_empty() {
            let found = self.fetch_keys(&next, Some(&[FIELD_ID]), false)?;
            next.retain(|span| found.contains_key(span));
            for chunk in next.chunks(WRITE_BATCH) {
                let keys: Vec<String> = chunk
                    .iter()
                    .map(|(topic, ordinal)| self.keys.span_key(*topic, *ordinal))
                    .collect();
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                self.collection.delete(&keys)?;
            }
            for span in &mut next {
                span.1 += 1;
            }
        }

        Ok(())
    }

    /// A document without its text or its vector is refused rather than read
    /// as absent: a copy that took it for absent would drop it without a word.
    fn stored(&self, topics: &[TopicId]) -> Result<Vec<Vec<Stored>>> {
        let mut stored = self.fetch(topics, true)?;
        topics
            .iter()
            .map(|topic| {
                stored
                    .remove(topic)
                    .unwrap_or_default()
                    .iter()
                    .map(|doc| {
                        match (
                            doc.get_string(FIELD_NGRAM)?,
                            crate::half::get(doc, FIELD_VECTOR)?,
                        ) {
                            (Some(content), Some(embedding)) => Ok(Stored {
                                content,
                                embedding,
                                encoding: self.encoding_of(doc)?,
                            }),
                            _ => Err(IndexError::Engine(format!(
                                "a document for topic {topic} came back without its text or its \
                                 vector"
                            ))),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Without the vectors: the answer compares only the text and the
    /// encoding, and a sync asks it of every batch, most of which hold
    /// nothing. Read through [`stored`](Projection::stored), it fetched and
    /// decoded a primary and a secondary vector per document only to drop
    /// them. A buffered write still counts and a deleted topic still does
    /// not, because both go through the same keyed fetch.
    fn holds(&self, wanted: &[(TopicId, &[&str])], encoding: &str) -> Result<Vec<bool>> {
        let topics: Vec<TopicId> = wanted.iter().map(|(topic, _)| *topic).collect();
        let stored = self.fetch(&topics, false)?;
        wanted
            .iter()
            .map(|(topic, texts)| {
                let docs = stored.get(topic).map_or(&[][..], Vec::as_slice);
                let read = docs
                    .iter()
                    .map(|doc| Ok((doc.get_string(FIELD_NGRAM)?, self.encoding_of(doc)?)))
                    .collect::<Result<Vec<_>>>()?;
                Ok(holds_spans(
                    read.iter()
                        .map(|(text, encoding)| (text.as_deref(), encoding.as_deref())),
                    texts,
                    encoding,
                ))
            })
            .collect()
    }

    /// Word-level lexical recall, ranked by BM25.
    ///
    /// The query is segmented by the same function that segmented the documents.
    /// Tokenizing the two differently is the standard way to build an index that
    /// never matches.
    fn recall_segmented(&self, query: &str, limit: u32) -> Result<Vec<Scored>> {
        let segmented = self.segmenter.segment_for_index(query);
        self.recall_text(FIELD_SEGMENTED, &segmented, limit)
    }

    /// Substring lexical recall over raw text, ranked by BM25.
    fn recall_ngram(&self, query: &str, limit: u32) -> Result<Vec<Scored>> {
        self.recall_text(FIELD_NGRAM, query, limit)
    }

    /// Word-level recall requiring every word of the name.
    fn recall_naming(&self, name: &str, limit: u32) -> Result<Vec<TopicId>> {
        let segmented = self.segmenter.segment_for_index(name);
        Ok(self
            .recall_fts(FIELD_SEGMENTED, &segmented, limit, true)?
            .into_iter()
            .map(|candidate| candidate.topic)
            .collect())
    }

    /// Semantic recall over dense embeddings.
    ///
    /// Carries the similarity the index computed, as the lexical channels carry
    /// their BM25 scores. Fusion still combines the four channels by rank -- a
    /// similarity and a BM25 score are different quantities and summing them
    /// directly would be meaningless -- but each channel's own scores are the
    /// only evidence of whether *that* channel is confident, which is a question
    /// ranks cannot answer. See [`pamin_core::ChannelResults`].
    ///
    /// The index proposes [`RESCORE`] times `limit` candidates with their
    /// stored vectors, and they are ranked again here by an exact f32 cosine
    /// against the query as the model produced it; the engine's own scores
    /// over half-precision vectors are not good enough to rank by. The score
    /// each keeps is that cosine similarity, where larger is better as
    /// `Scored` requires.
    fn recall_vector_at(&self, embedding: &[f32], limit: u32, effort: u32) -> Result<Vec<Scored>> {
        self.recall_dense(FIELD_VECTOR, embedding, limit, effort)
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
        let marker = optimized_files_marker(&self.dir);
        // Our metadata is not held open by zvec and does not spend its file
        // budget; count_files excludes it from both the floor and the trigger.
        let floor = self.file_count()?;
        std::fs::write(marker, format!("v1 {floor}\n"))?;
        self.optimized_files.store(floor, Ordering::Relaxed);
        Ok(())
    }

    /// How much of the collection the vector index covers, from 0.0 to 1.0.
    ///
    /// The share of documents [`optimize`](Self::optimize) has taken in. Below
    /// 1.0 the remainder is still answered by the flat buffer -- correctly, and
    /// at a cost that grows with the project.
    fn vector_index_completeness(&self) -> Result<f32> {
        let stats = self.collection.stats()?;
        Ok(f32::min(
            1.0,
            stats
                .indexes
                .iter()
                .find(|index| index.name == FIELD_VECTOR)
                .map_or(0.0, |index| index.completeness),
        ))
    }

    /// How many documents the index holds.
    fn document_count(&self) -> Result<u64> {
        Ok(self.collection.stats()?.doc_count)
    }

    fn segmentation(&self) -> Result<Segmentation> {
        Ok(Segmentation {
            documents: self.collection.stats()?.doc_count,
            // What the collection actually recorded, not what the policy would
            // have chosen: the point of reporting this is that the two differ.
            recorded: self.collection.schema()?.max_doc_count_per_segment(),
        })
    }

    /// How many files the index is spread across, counted from the directory.
    ///
    /// The engine reports documents and index completeness and nothing about
    /// files, so this is read from the filesystem -- which is no worse a source,
    /// since the number that matters is the one the operating system will
    /// count. A directory read of a few hundred entries is well under a
    /// millisecond and happens once per drain.
    ///
    /// A directory that cannot be read counts as nothing to do. This decides
    /// whether to schedule maintenance, and failing a write over it would be a
    /// worse answer than scheduling it a little late.
    fn file_count(&self) -> Result<u64> {
        Ok(count_files(&self.dir))
    }

    fn files_after_optimize(&self) -> u64 {
        self.optimized_files.load(Ordering::Relaxed)
    }

    /// Counted from the directory, as [`file_count`](Self::file_count) is:
    /// the engine keeps each segment in a directory of its own, writes one
    /// `embedding.index.<n>.proxima` there per flush, and leaves one per
    /// segment once it has compacted. So every block past a segment's first is
    /// one a compaction would merge.
    fn unmerged_blocks(&self) -> Result<u64> {
        let Ok(segments) = std::fs::read_dir(self.dir.join(COLLECTION)) else {
            return Ok(0);
        };
        Ok(segments
            .flatten()
            .filter(|segment| segment.file_type().is_ok_and(|kind| kind.is_dir()))
            .map(|segment| {
                let names: Vec<String> = std::fs::read_dir(segment.path()).map_or_else(
                    |_| Vec::new(),
                    |files| {
                        files
                            .flatten()
                            .map(|file| file.file_name().to_string_lossy().into_owned())
                            .collect()
                    },
                );
                let prefix = format!("{FIELD_VECTOR}.index.");
                let blocks = names
                    .iter()
                    .filter(|name| name.starts_with(&prefix) && name.ends_with(".proxima"))
                    .count() as u64;
                blocks.saturating_sub(1)
            })
            .sum())
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
/// Long enough to outlast a server on its way out, short enough that a caller
/// who is actually stuck finds out quickly.
const LOCK_BUDGET: Duration = Duration::from_millis(2_000);

/// Opens the collection, waiting out another process that holds its lock.
///
/// The engine takes the lock non-blocking and exclusive, so a second opener
/// does not queue -- it is refused outright. This used to be for `pamin`
/// commands overlapping, each opening the index itself. They no longer open
/// it: the workspace's server does, and it is the only process the CLI opens
/// an index in. What is left is a handover between two processes, and
/// it is still worth waiting out:
///
/// - **A server being replaced.** `pamin stop` and then any command, or a
///   client that found a server from another build, stops one server and
///   starts another. The old one
///   removes its socket and then exits, and its lock goes only when the kernel
///   closes its descriptors -- after unmapping an address space that runs to
///   gigabytes. The client watches the socket, not the process, so the new
///   server can reach the index while the old one still holds it.
/// - **A process outside the CLI** holding the same workspace, such as an
///   evaluation harness driving an engine directly.
///
/// The jitter is for the second case: several processes started together would
/// otherwise retry in step forever.
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

/// The topics a query returned, best first, each with the score it was ranked by.
///
/// A document whose primary key does not parse is dropped rather than reported.
/// The key is a UUID this crate wrote, so an unparseable one means the index is
/// corrupt in a way a single query cannot act on, and failing recall over it
/// would take the whole search down for one bad row.
/// `orient` turns the engine's number into one where larger is better, which
/// is what [`Scored`] requires of every channel. It is the identity for BM25
/// and `1 - score` for a cosine index, and it is a parameter rather than a
/// branch on the field so that adding a channel cannot forget it.
fn collect_scored(
    keys: Keys,
    docs: Vec<Doc>,
    orient: impl Fn(f32) -> f32,
) -> Vec<(TopicId, u32, f32)> {
    docs.iter()
        .filter_map(|doc| {
            let (topic, ordinal) = keys.span(doc.get_pk()?)?;
            Some((topic, ordinal, orient(doc.get_score())))
        })
        .collect()
}

/// Each topic once, by the best of its spans, best first.
///
/// What an index channel ranks is spans and what fusion ranks is topics, so a
/// topic is as good as its best span -- the max over spans -- and its other
/// spans give up their places to the next topic. A list where every topic is
/// one span comes back as it went in.
fn collapse(spans: Vec<(TopicId, u32, f32)>) -> Vec<Scored> {
    let mut seen = HashSet::with_capacity(spans.len());
    spans
        .into_iter()
        .filter(|(topic, ..)| seen.insert(*topic))
        .map(|(topic, ordinal, score)| Scored::new(topic, score).in_span(ordinal))
        .collect()
}

/// How much deeper than asked a channel will look for `limit` distinct
/// topics, at most.
const DEEPEST: u32 = 8;

/// `limit` topics from `ask`, which ranks `depth` spans and collapses them
/// ([`collapse`]), also saying whether the index had that many to give.
///
/// Spans of one topic crowd others out of a list of `depth`, so a list that
/// collapsed to fewer than `limit` topics while the index still had more to
/// give is asked again twice as deep, up to [`DEEPEST`] times what was asked.
/// Only then: where every topic is one span nothing is ever asked twice, so
/// the candidates are the ones a topic-grain index proposed -- a deeper ask
/// changes what an approximate search proposes and what the rescore sees.
fn deepening(
    limit: u32,
    mut ask: impl FnMut(u32) -> Result<(Vec<Scored>, bool)>,
) -> Result<Vec<Scored>> {
    let mut depth = limit;
    loop {
        let (mut topics, full) = ask(depth)?;
        if topics.len() >= limit as usize || !full || depth >= limit.saturating_mul(DEEPEST) {
            topics.truncate(limit as usize);
            return Ok(topics);
        }
        depth = depth.saturating_mul(2);
    }
}

/// `limit` topics from one ask of `ask`, for a channel whose ranking is exact.
///
/// Where a memory can be several spans, the ask is as deep as [`deepening`]
/// ever goes. An exact ranking at any depth is the start of the ranking at
/// every deeper one, so one ask [`DEEPEST`] times deep keeps exactly the topics
/// that deepening until enough of them collapsed out would have kept, and it
/// is cheaper: a lexical ask costs what scoring the spans that share a term
/// with the query costs, and deepening paid that two or three times over
/// whenever one memory's spans crowded a list. Where every memory is one
/// document, nothing crowds, and a deeper ask only scores more of them before
/// the engine can stop: the depth asked is `limit`, as it always was.
fn deepest(
    grain: Grain,
    limit: u32,
    ask: impl FnOnce(u32) -> Result<Vec<Scored>>,
) -> Result<Vec<Scored>> {
    let depth = match grain {
        Grain::Topic => limit,
        Grain::Spans(_) => limit.saturating_mul(DEEPEST),
    };
    let mut topics = ask(depth)?;
    topics.truncate(limit as usize);
    Ok(topics)
}

/// Whether the spans an index holds for a topic -- their texts and what
/// produced their vectors, in order -- are exactly `wanted`, all embedded
/// under `encoding`: the same number, each with exactly its text.
///
/// Whole topics, not span by span. A span's vector is embedded from its
/// header as well as its text, and the header is read off the spans before
/// it -- the heading it sits under, the turn it continues -- so the same text
/// is the same passage only when everything before it is the same too. A
/// topic's name never changes, so with every span the same, every header is.
fn holds_spans<'a>(
    stored: impl ExactSizeIterator<Item = (Option<&'a str>, Option<&'a str>)>,
    wanted: &[&str],
    encoding: &str,
) -> bool {
    stored.len() == wanted.len()
        && !wanted.is_empty()
        && stored.zip(wanted).all(|((text, recorded), wanted)| {
            holds_exactly(text, wanted) && recorded == Some(encoding)
        })
}

#[cfg(test)]
mod upkeep {
    use super::{
        Segmentation, UNINDEXED_BUDGET, is_fragmented, segment_documents, vector_index_lags,
        wastes_disk,
    };

    /// A workspace that grew from empty holds the floor's segments, and the
    /// report says so; one built knowing its size does not.
    ///
    /// The defect written down, the way the two above are. A collection
    /// records its segment size at creation and a workspace is created before
    /// anything is written to it, so `segment_documents` is asked about zero
    /// documents and clamped to the floor -- which means the division by
    /// `TARGET_SEGMENTS` never runs for a project anyone has, and 131,924
    /// documents land in 14 segments rather than four -- 66 before the floor
    /// was raised.
    #[test]
    fn a_project_grown_from_empty_holds_the_floors_segments_and_the_report_says_so() {
        let grown = Segmentation {
            documents: 131_924,
            recorded: segment_documents(0),
        };
        assert_eq!(
            grown.recorded, 10_000,
            "an empty collection records the floor"
        );
        assert_eq!(grown.segments(), 14);
        assert_eq!(grown.wanted(), 4);
        assert!(
            grown.is_worth_rebuilding(),
            "fourteen segments where four would do was not reported"
        );

        let rebuilt = Segmentation {
            documents: 131_924,
            recorded: segment_documents(131_924),
        };
        assert_eq!(rebuilt.segments(), 4);
        assert!(
            !rebuilt.is_worth_rebuilding(),
            "an index already at the target was reported as worth rebuilding"
        );
    }

    /// Fewer segments than wanted is not reported, because it is not better.
    ///
    /// Measured over fifty thousand documents, recall@10 falls monotonically
    /// as the segments grow -- 1.0000 at twenty-five, 0.9980 at four, 0.9880
    /// at two, 0.9510 at one, which is below `recall.rs`'s floor. So the
    /// target is a floor as well as a ceiling, and a report that said "fewer
    /// than four, rebuild" would be advising hours of work for a regression.
    #[test]
    fn fewer_segments_than_wanted_is_not_worth_rebuilding() {
        let coarse = Segmentation {
            documents: 50_000,
            recorded: 50_000,
        };
        assert_eq!(coarse.segments(), 1);
        assert_eq!(coarse.wanted(), 4);
        assert!(!coarse.is_worth_rebuilding());
    }

    /// A little over the target is not worth hours either.
    #[test]
    fn a_few_more_segments_than_wanted_is_not_worth_rebuilding() {
        let close = Segmentation {
            documents: 50_000,
            recorded: 8_000,
        };
        assert_eq!(close.segments(), 7);
        assert_eq!(close.wanted(), 4);
        assert!(
            !close.is_worth_rebuilding(),
            "seven segments against four is not a rebuild"
        );
    }

    /// The two maintenance conditions, at the numbers a served workspace
    /// actually reaches.
    ///
    /// This is the defect written down. `projection.rs` measured that three
    /// thousand writes through a server leave 136 files, and 136 is inside the
    /// 256-file budget -- so for a release the only condition that could queue
    /// `optimize` was one that a served workspace does not trip, and the vector
    /// channel answered from an exhaustive scan with nothing reporting it.
    ///
    /// Asserted together rather than apart, because either one alone passes on
    /// the broken code: the point is that at one set of numbers the file
    /// condition is silent and the graph condition is not.
    #[test]
    fn a_served_workspace_trips_the_graph_condition_and_not_the_file_one() {
        // Measured, not chosen: the file count after three thousand writes
        // that left their flush to the server.
        assert!(
            !is_fragmented(136, 0),
            "136 files is inside the budget, which is why this condition \
             cannot be the graph's"
        );
        assert!(
            vector_index_lags(30_000, 0.0, UNINDEXED_BUDGET),
            "thirty thousand documents outside the graph has to queue a build"
        );
    }

    /// Below the crossover, waiting is right rather than merely tolerable.
    #[test]
    fn a_remainder_cheaper_to_scan_than_to_index_waits() {
        assert!(!vector_index_lags(1_000, 0.0, UNINDEXED_BUDGET));
        assert!(!vector_index_lags(100_000, 0.9, UNINDEXED_BUDGET));
        // The bound is on the remainder, not on the project: the size that
        // never fired as a threshold is fully indexed here.
        assert!(!vector_index_lags(1_000_000, 1.0, UNINDEXED_BUDGET));
    }

    /// A small index is compacted after a few flushes, not after the file
    /// budget's forty; a large one keeps a larger allowance of blocks.
    #[test]
    fn unmerged_blocks_are_bounded_by_the_index_size() {
        assert!(!wastes_disk(8, 1), "eight blocks is the floor");
        assert!(wastes_disk(9, 43), "a ninth on a small index is waste");
        assert!(
            !is_fragmented(9 * 6, 0),
            "and the file budget would not have asked yet"
        );
        assert!(!wastes_disk(100, 13_014), "a large index is allowed more");
        assert!(wastes_disk(102, 13_014));
    }

    /// An index `optimize` cannot bring under the budget is not asked again
    /// until it has grown.
    ///
    /// The first assertion is the one the budget alone failed: an index its
    /// own `optimize` left at 300 files was asked for another after every
    /// drain that did work, and the answer was the same 300 files.
    #[test]
    fn an_index_optimize_cannot_shrink_is_not_asked_again_until_it_grows() {
        assert!(
            !is_fragmented(300, 300),
            "nothing has grown since the last optimize"
        );
        assert!(
            !is_fragmented(300 + 2 * 10, 300),
            "ten reopens' write-ahead logs are not growth worth an optimize"
        );
        assert!(
            is_fragmented(300 + 65, 300),
            "a quarter of the budget past the floor is"
        );
        assert!(!is_fragmented(2_317, 2_317), "nor is MIRACL's floor");
    }

    /// Under the budget the floor changes nothing, and an index with no valid
    /// saved floor is held to the budget alone.
    #[test]
    fn below_the_budget_the_floor_changes_nothing() {
        assert!(!is_fragmented(256, 0));
        assert!(
            is_fragmented(257, 0),
            "an index with no saved floor is asked once"
        );
        assert!(!is_fragmented(256, 240));
        assert!(
            is_fragmented(257, 240),
            "a prior optimize below the ceiling cannot raise that ceiling"
        );
        assert!(
            is_fragmented(257, 256),
            "the ceiling itself is not a floor above it"
        );
        assert!(
            is_fragmented(257, 40),
            "an index optimize left small keeps the plain budget"
        );
        assert!(!is_fragmented(u64::MAX, u64::MAX), "the floor saturates");
    }

    /// A completeness outside 0.0..=1.0 must not read as a negative remainder.
    #[test]
    fn a_nonsense_completeness_does_not_wrap() {
        assert!(vector_index_lags(30_000, -1.0, UNINDEXED_BUDGET));
        assert!(!vector_index_lags(30_000, 2.0, UNINDEXED_BUDGET));
    }
}

#[cfg(test)]
mod keys {
    use super::{KEYS, Keys};
    use pamin_core::TopicId;

    /// Every spelling reads back as exactly the topic it was written for.
    ///
    /// A key that read back as some other topic would not fail anywhere: the
    /// channels would return a plausible identifier, the ledger would resolve
    /// it or drop it, and a memory would go missing from search with no error.
    #[test]
    fn a_key_reads_back_as_the_topic_it_was_written_for() {
        let mut topics: Vec<TopicId> = (0..1_000).map(|_| TopicId::new()).collect();
        topics.extend([
            TopicId(uuid::Uuid::nil()),
            TopicId(uuid::Uuid::max()),
            TopicId(uuid::Uuid::from_u128(1)),
            TopicId(uuid::Uuid::from_u128(
                0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
            )),
        ]);
        for keys in [Keys::Topic, Keys::Reversed] {
            assert_eq!(Keys::parse(keys.label()), Some(keys));
            for topic in &topics {
                assert_eq!(keys.topic(&keys.key(*topic)), Some(*topic), "{keys:?}");
            }
            assert_eq!(keys.topic("not a key"), None);
        }

        // The old spelling is the identifier as written, which is what every
        // index without a key line holds.
        let topic = topics[0];
        assert_eq!(Keys::Topic.key(topic), topic.to_string());
        assert_ne!(Keys::Reversed.key(topic), topic.to_string());
        assert_eq!(KEYS, Keys::Reversed, "a new index spells keys reversed");
    }

    /// A span's key reads back as its topic and ordinal, and as its topic
    /// alone where only that is asked for. A key `topic` failed to strip
    /// would be dropped by every channel without an error.
    #[test]
    fn a_span_key_reads_back_as_its_topic_and_ordinal() {
        let topics = [
            TopicId::new(),
            TopicId(uuid::Uuid::nil()),
            TopicId(uuid::Uuid::max()),
        ];
        for keys in [Keys::Topic, Keys::Reversed] {
            for topic in topics {
                assert_eq!(keys.span_key(topic, 0), keys.key(topic));
                for ordinal in [0, 1, 2, 9, 10, 4_000, u32::MAX] {
                    let key = keys.span_key(topic, ordinal);
                    assert_eq!(keys.span(&key), Some((topic, ordinal)), "{key}");
                    assert_eq!(keys.topic(&key), Some(topic), "{key}");
                }
                let key = keys.key(topic);
                for suffix in ["#0", "#01", "#", "#-1", "#+1", "#1#2", "#x", "#4294967296"] {
                    assert_eq!(keys.span(&format!("{key}{suffix}")), None, "{suffix}");
                }
            }
        }
        let topic = topics[0];
        assert_eq!(Keys::Topic.span_key(topic, 3), format!("{topic}#3"));
    }

    /// Topics created one after another do not make keys that ascend.
    ///
    /// Which is the whole point: the key map only merges files whose key
    /// ranges overlap, so keys that each land above the last are what left
    /// one file per flush behind.
    #[test]
    fn topics_created_in_order_do_not_make_keys_in_order() {
        let topics: Vec<TopicId> = (0..1_000).map(|_| TopicId::new()).collect();
        let ascending = |keys: Keys| {
            let spelled: Vec<String> = topics.iter().map(|topic| keys.key(*topic)).collect();
            spelled.windows(2).filter(|pair| pair[0] < pair[1]).count()
        };

        assert_eq!(
            ascending(Keys::Topic),
            topics.len() - 1,
            "the premise: identifiers created in order ascend"
        );
        // Random order ascends about half the time: 499.5 of 999 pairs, with
        // a standard deviation of about nine, so these bounds are ten of them
        // either side and still nowhere near the 999 of the premise.
        let reversed = ascending(Keys::Reversed);
        assert!(
            (400..600).contains(&reversed),
            "reversed keys ascended at {reversed} of {} pairs",
            topics.len() - 1
        );
    }
}

#[cfg(test)]
mod marker {
    use super::{IndexError, Marker, ProjectionIndex, VectorIndex};

    /// What an index was built with reads back as the vector index that built
    /// it.
    ///
    /// The server opens a project nobody asked for under this answer, and
    /// the answer being wrong has no quiet failure: the open is refused, or
    /// -- for a directory with no index -- an empty one is created under the
    /// guess. So every vector index is written and read back, and an empty
    /// directory has to answer that there is nothing to open.
    #[test]
    fn an_index_reads_back_what_it_was_built_with() {
        for index in VectorIndex::ALL {
            let dir = tempfile::tempdir().expect("a directory");
            Marker::current(index, super::Grain::default())
                .write(dir.path())
                .expect("marking");
            assert_eq!(
                ProjectionIndex::built_for(dir.path()).expect("reading"),
                Some(index)
            );
        }

        // An index from before the vector indexes names nothing to open it
        // under: only a rebuild can.
        let legacy = tempfile::tempdir().expect("a directory");
        std::fs::write(
            legacy.path().join("profile"),
            format!("{}\ntopic\nfp32\nnamed", super::MODEL_ID),
        )
        .expect("marking");
        assert_eq!(
            ProjectionIndex::built_for(legacy.path()).expect("reading"),
            None
        );

        // An index from a retired profile is not passed over as if it were
        // not there: it is named, so a caller can say `pamin reindex`.
        let retired = tempfile::tempdir().expect("a directory");
        Marker {
            model: "intfloat/multilingual-e5-small+p1".to_string(),
            ..Marker::current(VectorIndex::default(), super::Grain::default())
        }
        .write(retired.path())
        .expect("marking");
        assert!(matches!(
            ProjectionIndex::built_for(retired.path()),
            Err(IndexError::ModelMismatch { indexed }) if indexed == "intfloat/multilingual-e5-small+p1"
        ));

        let empty = tempfile::tempdir().expect("a directory");
        assert_eq!(
            ProjectionIndex::built_for(empty.path()).expect("reading"),
            None,
            "a directory with no index named something to open it under"
        );
    }
}
