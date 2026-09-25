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

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use pamin_core::{Scored, TopicId};

use crate::embedding::Profile;
use zvec_rust::{
    Collection, CollectionOptions, CollectionSchema, DataType, DiskannQueryParams, Doc,
    FieldSchema, Fts, FtsQueryParams, HnswQueryParams, IndexParams, MetricType, SearchQuery,
};

use crate::error::{IndexError, Result};
use crate::segmentation::Segmenter;

const COLLECTION: &str = "memories";

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
    ///
    /// A handle rather than a borrow, so a caller can keep tokenizing after it
    /// has let go of the projection. Splitting text touches nothing the
    /// projection owns, and the lock the composition layer holds the projection
    /// behind is there for an engine defect the segmenter has no part in; a
    /// borrow would keep that lock held for work that never needed it.
    fn segmenter(&self) -> Arc<Segmenter>;

    /// Adds or replaces one topic.
    fn upsert(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<()>;

    /// Adds or replaces many.
    fn upsert_batch(&self, documents: &[(TopicId, &str, &[f32])]) -> Result<()>;

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

    /// Semantic recall over dense embeddings, nearest first.
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<Scored>>;

    /// What this index holds for these topics, as it was written, in the
    /// order asked; `None` for a topic it does not hold.
    ///
    /// Reading a document back is what lets an index be copied rather than
    /// rebuilt: the vector is the one expensive part of a document, and the
    /// index already holds it. A write the index has buffered and not yet
    /// flushed is read back like any other, because a query sees it too.
    fn stored(&self, topics: &[TopicId]) -> Result<Vec<Option<Stored>>>;

    /// Removes these topics.
    ///
    /// The projection had no way to shrink: the only route out was deleting the
    /// whole directory. A soft-deleted state therefore stayed in every channel's
    /// candidate budget, so removing content from the ledger quietly reduced how
    /// much a search could find.
    fn delete(&self, topics: &[TopicId]) -> Result<()>;

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

    /// Which vector index this one was built with.
    fn vector_index(&self) -> VectorIndex;
}

/// One document as an index holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stored {
    /// The memory's text, exactly as written. The segmented field is derived
    /// from it, so it is all a copy needs to rebuild both lexical fields.
    pub content: String,
    /// The vector, as it was embedded under the index's [`Passage`] and then
    /// stored: each component rounded to half precision (see [`crate::as_stored`]).
    pub embedding: Vec<f32>,
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

    /// The primary key `topic` is stored under.
    fn key(self, topic: TopicId) -> String {
        self.spell(topic.0).to_string()
    }

    /// The topic stored under `key`, or `None` for a key this crate did not
    /// write.
    fn topic(self, key: &str) -> Option<TopicId> {
        uuid::Uuid::parse_str(key)
            .ok()
            .map(|parsed| TopicId::from(self.spell(parsed)))
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
pub fn is_fragmented(files: u64) -> bool {
    files > MAX_FILES
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
const UNINDEXED_BUDGET: u64 = 25_000;

/// Whether enough documents sit outside the vector graph to be worth building.
///
/// Takes the completeness rather than reading it, so the policy is testable
/// without an index and the caller does the one FFI call it already makes.
pub fn vector_index_lags(documents: u64, completeness: f32) -> bool {
    let covered = (documents as f64 * f64::from(completeness.clamp(0.0, 1.0))) as u64;
    documents.saturating_sub(covered) >= unindexed_budget()
}

/// The budget, or whatever a harness set it to.
///
/// Same shape and the same reason as `reranking::tuned`: the assertion worth
/// having here is that a drain leaves a graph behind, and asserting it at the
/// shipped budget would mean writing and embedding twenty-five thousand
/// memories to see it. Unset means the constant, so nothing a user runs is
/// affected. Undocumented on purpose -- it exists so a test does not have to
/// edit the tree.
fn unindexed_budget() -> u64 {
    std::env::var("PAMIN_UNINDEXED_BUDGET")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(UNINDEXED_BUDGET)
}

/// What one document in this index stands for.
///
/// Recorded beside the model because an index keyed by something else is not
/// stale, it is silently empty: the old scheme's identifiers are read as the
/// new scheme's, match nothing, and every search comes back with no results
/// and no error anywhere. Changing what a document is keyed by means changing
/// this, which turns that silence into a message naming `pamin reindex`.
const DOCUMENT_GRAIN: &str = "topic";

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

/// Which vector index a project's index is built with, and what its marker
/// records.
///
/// A setting (`--vector-index`, `PAMIN_VECTOR_INDEX`) with two values, because
/// the choice is a trade between resident memory on one side and query and
/// build time on the other that only the person running a project can make.
/// Both store vectors in half precision and both are searched the same way:
/// the index proposes [`RESCORE`] times the candidates asked for, with their
/// stored vectors, and those are ranked again by an exact f32 cosine. The two
/// differ in their index and query parameters and in nothing else. ADR 0001,
/// "Two vector indexes, both half precision", has the measurements.
///
/// The marker records which one an index was built with, and an index is
/// never searched as the other: opening one under the other is refused with a
/// message naming `pamin reindex`, as a profile change is. Reading an index
/// built one way as another is the silent-wrong-answer shape ADR 0001 records
/// from the first attempt at quantization, which returned recall@10 of 0.000
/// with no error. `pamin reindex` lends every vector the old index holds to
/// the new one (see [`Previous`]), so changing costs a build and no embedding.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum VectorIndex {
    /// A DiskANN graph kept on disk and read per query.
    ///
    /// Holds almost nothing resident, for a project whose memory is scarce.
    /// Not the default: every `optimize` after a working drain rebuilds a
    /// great deal of the graph (114 to 310 s after 64 new documents on 25,000,
    /// against about a second for `memory`), and upkeep issues one whenever a
    /// drain has done work.
    Disk,
    /// An HNSW graph held in memory.
    ///
    /// The default: the fastest to query and to build, and the smallest on
    /// disk, at the cost of holding the graph and the vectors resident --
    /// still half what the full-precision graph held.
    #[default]
    Memory,
}

impl VectorIndex {
    /// Both, in the order the setting documents them.
    pub const ALL: [Self; 2] = [Self::Memory, Self::Disk];

    /// The name the setting takes and the marker records.
    pub fn label(self) -> &'static str {
        match self {
            Self::Disk => "disk",
            Self::Memory => "memory",
        }
    }

    /// The index a name stands for, `None` for any other name -- including
    /// what an index built before these existed records.
    pub fn parse(label: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|index| index.label() == label.trim())
    }

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

    /// Asks `search` for this index, wide enough for `candidates`.
    fn ask(self, search: &mut SearchQuery, candidates: u32) -> Result<()> {
        match self {
            Self::Disk => search.set_diskann_params(DiskannQueryParams::new(
                DISKANN_SEARCH_LIST.max(candidates as i32),
            ))?,
            Self::Memory => {
                search.set_hnsw_params(HnswQueryParams::new(search_effort(), 0.0, false, false))?
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
/// One hundred. Twice that took 1,416 s rather than 922 to 1,024 over the same
/// 50,000 vectors and recalled no more at any search width: 0.9970 against
/// 0.9975 at 800.
const DISKANN_BUILD_LIST: i32 = 100;

/// Product-quantization chunks for in-memory navigation of the on-disk graph;
/// zero keeps none, and every step reads full vectors from disk.
///
/// None. Sixty-four chunks answered in about a third of the time at the same
/// width and recalled 0.9200 at ten and 0.8629 at fifty where none recalls
/// 0.9985 and 0.9975 -- navigating by codes loses neighbours the rescore
/// cannot find again -- and took 1,859 s to build rather than 922 to 1,024.
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

/// How wide a query searches the graph.
///
/// The engine defaults to 300 and this had never been set, so every query took
/// that default without anything saying so. Seven hundred is the first value
/// measured to reach 0.95 recall against exact search, which is the target --
/// the last few points cost more than the rest put together, and a query
/// spends 35 ms embedding before it gets here.
///
/// **Checked again once segments grew, and it holds on real text.** A reshape
/// takes a project from 10,000 documents a segment to a quarter of the
/// collection, and on 132,000 synthetic clustered vectors four such segments
/// reach only 0.9758 recall@50 against exact search at 700 (0.9976 at 2,000;
/// the engine refuses more than 2,048). On MIRACL's 131,924 real passages in
/// four segments, every one of 482 questions returns the same results at 700
/// and at 2,000, fused and through the reranker -- zero wins, zero losses (the
/// `EFFORTS` arm of `pamin-engine/tests/monolingual.rs`). Synthetic clusters
/// are harder to search than real embeddings, so the width stays.
///
/// **And again over half-precision vectors with the rescore**, which is what
/// [`VectorIndex::Memory`] searches, against the bar that fp32 at 700 sets --
/// within 0.002 of its recall on 50,000 clustered vectors, 0.001 on MIRACL:
///
/// | width | synthetic @10 | synthetic @50 | MIRACL @10 | MIRACL @50 |
/// | --- | --- | --- | --- | --- |
/// | fp32 at 700 | 0.9980 | 0.9974 | 1.0000 | 0.9999 |
/// | 200 | 0.9665 | 0.9458 | 0.9996 | 0.9980 |
/// | 300 | 0.9855 | 0.9769 | 0.9998 | 0.9988 |
/// | 500 | 0.9965 | 0.9927 | 1.0000 | 0.9993 |
/// | **700** | **0.9965** | **0.9965** | **1.0000** | **0.9997** |
///
/// A narrower width saves about a millisecond a query and fails the bar at
/// fifty on both sets, so 700 stays for this index too.
const SEARCH_EFFORT: i32 = 700;

/// Overrides [`SEARCH_EFFORT`], for the sweep that settles it.
const PAMIN_SEARCH_EFFORT: &str = "PAMIN_SEARCH_EFFORT";

fn search_effort() -> i32 {
    std::env::var(PAMIN_SEARCH_EFFORT)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|effort| *effort > 0)
        .unwrap_or(SEARCH_EFFORT)
}

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
        profile: Profile,
        index: VectorIndex,
        access: Access,
        documents: u64,
    ) -> Result<Self> {
        // A workspace built before projects had their own directory holds one
        // shared collection. Opening this project's empty directory beside it
        // would return nothing and look like an empty workspace, so it is
        // reported instead.
        if std::fs::exists(legacy_dir)? {
            return Err(IndexError::LegacyLayout);
        }

        Self::open_sized(dir, profile, index, access, segment_documents(documents))
    }

    /// The profile the index at `dir` was built with, if there is one there.
    ///
    /// For a caller that has to open an index nobody asked it to and so has
    /// no profile in hand: the resident server, bringing up to date a project
    /// that was written to by a process that is gone. Opening under a guess is
    /// not an option -- a wrong guess is refused, and a guess at a directory
    /// with no index creates one for that profile.
    ///
    /// `None` for no index, and for one recorded with a model no profile runs
    /// any more or a vector index no build makes any more, which only a
    /// rebuild can open.
    pub fn built_for(dir: &Path) -> Result<Option<(Profile, VectorIndex)>> {
        Ok(Marker::read(dir)?.and_then(|recorded| {
            let profile = [Profile::Speed, Profile::Balanced, Profile::Accuracy]
                .into_iter()
                .find(|profile| profile.model_id() == recorded.model)?;
            Some((profile, VectorIndex::parse(&recorded.storage)?))
        }))
    }

    /// Opens the index at `dir`, creating it with segments of `segment`
    /// documents if absent.
    ///
    /// The size is the caller's here rather than derived from a count, because
    /// the reshape's tests have to build an index in the shape a grown project
    /// is in -- many segments -- without writing fifty thousand documents.
    pub(crate) fn open_sized(
        dir: &Path,
        profile: Profile,
        index: VectorIndex,
        access: Access,
        segment: u64,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let (passage, keys) = match Marker::read(dir)? {
            Some(recorded) => {
                if recorded.model != profile.model_id() {
                    return Err(IndexError::ProfileMismatch {
                        indexed: recorded.model,
                        requested: profile.model_id().to_string(),
                    });
                }
                if recorded.grain != DOCUMENT_GRAIN {
                    return Err(IndexError::GrainMismatch {
                        // A marker with no grain line was written before there
                        // was one, and everything written then was keyed by
                        // state.
                        indexed: if recorded.grain.is_empty() {
                            "topic state".to_string()
                        } else {
                            recorded.grain
                        },
                        expected: DOCUMENT_GRAIN.to_string(),
                    });
                }
                if recorded.storage != index.label() {
                    return Err(IndexError::VectorIndexMismatch {
                        indexed: recorded.storage,
                        requested: index.label().to_string(),
                    });
                }
                (recorded.passage, recorded.keys)
            }
            None => {
                Marker::current(profile, index).write(dir)?;
                (PASSAGE, KEYS)
            }
        };

        let mut opened =
            Self::open_with_dimensions(dir, profile.dimensions(), index, access, segment)?;
        opened.passage = passage;
        opened.keys = keys;
        Ok(opened)
    }

    /// Creates an empty index at `dir` that records what the one at `source`
    /// does, sized for `documents`.
    ///
    /// The marker is copied rather than written afresh, so the copy keeps the
    /// source's encoding: a copy of an index whose vectors were embedded from
    /// content alone must go on being written that way, and writing the
    /// current marker would label those vectors `name: content`. Reopening
    /// then checks the copied marker against `profile` like any other open.
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
        profile: Profile,
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
            profile,
            index,
            Access::ReadWrite,
            segment_documents(documents),
        )
    }

    /// Opens the index at `dir` for writing, refusing to create one.
    ///
    /// For reopening an index after its directory was moved into place, where
    /// finding nothing there is a fault: an open that created an empty index
    /// would hand the caller a project with no memories and no error.
    pub(crate) fn reopen(dir: &Path, profile: Profile, index: VectorIndex) -> Result<Self> {
        if !std::fs::exists(dir.join(COLLECTION))? {
            return Err(IndexError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no index at {}", dir.display()),
            )));
        }
        Self::open_sized(dir, profile, index, Access::ReadWrite, segment_documents(0))
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

        Ok(Self {
            collection,
            segmenter: Arc::new(Segmenter::new()),
            dir: dir.to_path_buf(),
            index,
            passage: PASSAGE,
            keys: KEYS,
        })
    }

    fn document(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<Doc> {
        let mut doc = Doc::new()?;
        let key = self.keys.key(topic);
        doc.set_pk(&key);
        doc.add_string(FIELD_ID, &key)?;
        doc.add_string(FIELD_SEGMENTED, &self.segmenter.segment_for_index(content))?;
        doc.add_string(FIELD_NGRAM, content)?;
        crate::half::add(&mut doc, FIELD_VECTOR, embedding)?;
        Ok(doc)
    }

    fn recall_text(&self, field: &str, query: &str, limit: u32) -> Result<Vec<Scored>> {
        self.recall_fts(field, query, limit, false)
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

        let mut fts = Fts::new()?;
        fts.set_match_string(query)?;
        let mut search = SearchQuery::fts(field, &fts, limit as i32)?;
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
        Ok(collect_scored(
            self.keys,
            self.collection.query(&search)?,
            |score| score,
        ))
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
    /// verbatim; the segmented one is derived from it.
    fn fetch(&self, topics: &[TopicId]) -> Result<HashMap<TopicId, Doc>> {
        let keys: Vec<String> = topics.iter().map(|topic| self.keys.key(*topic)).collect();
        let mut stored: HashMap<TopicId, Doc> = HashMap::with_capacity(keys.len());
        for chunk in keys.chunks(WRITE_BATCH) {
            let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
            for doc in self
                .collection
                .fetch_with_options(&chunk, Some(&[FIELD_NGRAM]), true)?
            {
                if let Some(topic) = doc.get_pk().and_then(|key| self.keys.topic(key)) {
                    stored.insert(topic, doc);
                }
            }
        }
        Ok(stored)
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

    /// What an index built now, for this profile and vector index, is.
    fn current(profile: Profile, index: VectorIndex) -> Self {
        Self {
            model: profile.model_id().to_string(),
            grain: DOCUMENT_GRAIN.to_string(),
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
    /// would compute for the same text: same model, same encoding, same grain.
    ///
    /// Not the same [`Keys`]: how a key is spelled says nothing about the
    /// vector stored under it, and the index lending it is read through its
    /// own spelling. Nor the same storage: every index built now stores half
    /// precision, and a vector read from any of them -- or from an fp32 index
    /// built before -- rounds to the same codes the model's own output would.
    fn matches(&self, other: &Self) -> bool {
        self.model == other.model && self.grain == other.grain && self.passage == other.passage
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
    pub fn set_aside(dir: &Path, profile: Profile) -> Result<Option<Self>> {
        let aside = dir.with_extension("previous");
        ProjectionIndex::discard(&aside)?;
        let current = Marker::current(profile, VectorIndex::default());
        let Some(recorded) = Marker::read(dir)?.filter(|recorded| recorded.matches(&current))
        else {
            ProjectionIndex::discard(dir)?;
            return Ok(None);
        };
        std::fs::rename(dir, &aside)?;
        // The collection exists, so the vector index passed here only fills a
        // schema the engine does not read.
        let mut index = ProjectionIndex::open_with_dimensions(
            &aside,
            profile.dimensions(),
            VectorIndex::default(),
            Access::ReadOnly,
            segment_documents(0),
        )?;
        index.passage = PASSAGE;
        index.keys = recorded.keys;
        Ok(Some(Self {
            index,
            dir: aside,
            half: recorded.is_half(),
        }))
    }

    /// How many of these topics [`lend`](Self::lend) would supply, without
    /// reading a vector.
    ///
    /// `wanted` maps each topic to the text the rebuild will write for it.
    pub fn lends(&self, wanted: &HashMap<TopicId, &str>) -> Result<usize> {
        let mut lendable = 0;
        self.each(wanted, false, |_, _| {
            lendable += 1;
            Ok(())
        })?;
        Ok(lendable)
    }

    /// Hands `take` every document this index can lend, `batch` at a time,
    /// as the topic, the text wanted for it and its stored vector; returns
    /// the topics it lent.
    ///
    /// A topic is lent when this index holds it with exactly the text in
    /// `wanted`; everything else is the caller's to embed. In the order this
    /// index holds its documents rather than the caller's, because that is
    /// the order they are read in: one pass over the collection rather than a
    /// keyed fetch per batch. Over MIRACL's 131,924 documents a pass takes
    /// 0.28 s this way against 1.46-1.59 s of keyed fetches, with the same
    /// resident set and identical vectors and text (ADR 0001).
    pub fn lend(
        &self,
        wanted: &HashMap<TopicId, &str>,
        batch: usize,
        mut take: impl FnMut(&[(TopicId, &str, &[f32])]) -> Result<()>,
    ) -> Result<HashSet<TopicId>> {
        let mut lent = HashSet::new();
        let mut pending: Vec<(TopicId, &str, Vec<f32>)> = Vec::with_capacity(batch);
        let mut hand_over = |pending: &mut Vec<(TopicId, &str, Vec<f32>)>| {
            let documents: Vec<(TopicId, &str, &[f32])> = pending
                .iter()
                .map(|(topic, content, vector)| (*topic, *content, vector.as_slice()))
                .collect();
            take(&documents)?;
            pending.clear();
            Ok::<_, IndexError>(())
        };
        self.each(wanted, true, |topic, doc| {
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
            lent.insert(topic);
            pending.push((topic, wanted[&topic], vector));
            if pending.len() == batch {
                hand_over(&mut pending)?;
            }
            Ok(())
        })?;
        if !pending.is_empty() {
            hand_over(&mut pending)?;
        }
        Ok(lent)
    }

    /// Calls `visit` for every document this index holds with exactly the
    /// text `wanted` gives its topic.
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
        wanted: &HashMap<TopicId, &str>,
        vectors: bool,
        mut visit: impl FnMut(TopicId, Doc) -> Result<()>,
    ) -> Result<()> {
        let index = &self.index;
        for doc in index
            .collection
            .iter_with_options(Some(&[FIELD_NGRAM]), vectors)?
        {
            let doc = doc?;
            let Some(topic) = doc.get_pk().and_then(|key| index.keys.topic(key)) else {
                continue;
            };
            let Some(content) = wanted.get(&topic) else {
                continue;
            };
            if doc.get_string(FIELD_NGRAM)?.as_deref() == Some(*content) {
                visit(topic, doc)?;
            }
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

impl Projection for ProjectionIndex {
    fn passage(&self) -> Passage {
        self.passage
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

    /// Adds or replaces many topics.
    ///
    /// Chunked because the engine refuses a write of more than [`WRITE_BATCH`]
    /// documents. Rebuilding used to write one document per call, which pays
    /// the per-call cost once per document; here it is once per batch.
    ///
    /// No flush: a caller writing in batches decides when the result becomes
    /// visible, and flushing between batches would make that decision for them
    /// once per batch.
    fn upsert_batch(&self, documents: &[(TopicId, &str, &[f32])]) -> Result<()> {
        for chunk in documents.chunks(WRITE_BATCH) {
            let docs = chunk
                .iter()
                .map(|(topic, content, embedding)| self.document(*topic, content, embedding))
                .collect::<Result<Vec<_>>>()?;

            let refs: Vec<&Doc> = docs.iter().collect();
            self.collection.upsert(&refs)?;
        }

        Ok(())
    }

    /// Adds or replaces one topic.
    ///
    /// The embedding is required rather than optional. The engine enforces it,
    /// and it is the right constraint: a document indexed without one is
    /// invisible to the vector channel, which would show up as unexplained
    /// recall gaps rather than as an error.
    fn upsert(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<()> {
        let doc = self.document(topic, content, embedding)?;
        self.collection.upsert(&[&doc])?;
        Ok(())
    }

    /// A document without its text or its vector is refused rather than read
    /// as absent: a copy that took it for absent would drop it without a word.
    fn stored(&self, topics: &[TopicId]) -> Result<Vec<Option<Stored>>> {
        let stored = self.fetch(topics)?;
        topics
            .iter()
            .map(|topic| {
                let Some(doc) = stored.get(topic) else {
                    return Ok(None);
                };
                match (
                    doc.get_string(FIELD_NGRAM)?,
                    crate::half::get(doc, FIELD_VECTOR)?,
                ) {
                    (Some(content), Some(embedding)) => Ok(Some(Stored { content, embedding })),
                    _ => Err(IndexError::Engine(format!(
                        "the document for topic {topic} came back without its text or its vector"
                    ))),
                }
            })
            .collect()
    }

    /// Removes these topics.
    fn delete(&self, topics: &[TopicId]) -> Result<()> {
        for chunk in topics.chunks(WRITE_BATCH) {
            let keys: Vec<String> = chunk.iter().map(|topic| self.keys.key(*topic)).collect();
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
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<Scored>> {
        let candidates = limit.saturating_mul(RESCORE);
        let mut search = SearchQuery::new(
            FIELD_VECTOR,
            &crate::half::query(embedding),
            candidates as i32,
        )?;
        search.set_output_fields(&[FIELD_ID])?;
        search.set_include_vector(true)?;
        self.index.ask(&mut search, candidates)?;

        let length = dot(embedding, embedding).sqrt();
        let mut scored = Vec::with_capacity(candidates as usize);
        for doc in self.collection.query(&search)? {
            let Some(topic) = doc.get_pk().and_then(|key| self.keys.topic(key)) else {
                continue;
            };
            let vector = crate::half::get(&doc, FIELD_VECTOR)?.ok_or_else(|| {
                IndexError::Engine(format!(
                    "a candidate for topic {topic} came without its vector"
                ))
            })?;
            let lengths = length * dot(&vector, &vector).sqrt();
            let similarity = if lengths > 0.0 {
                dot(embedding, &vector) / lengths
            } else {
                0.0
            };
            scored.push(Scored::new(topic, similarity));
        }
        scored.sort_by(|left, right| {
            right
                .score
                .unwrap_or(f32::MIN)
                .total_cmp(&left.score.unwrap_or(f32::MIN))
        });
        scored.truncate(limit as usize);
        Ok(scored)
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
    fn segmentation(&self) -> Result<Segmentation> {
        Ok(Segmentation {
            documents: self.collection.stats()?.doc_count,
            // What the collection actually recorded, not what the policy would
            // have chosen: the point of reporting this is that the two differ.
            recorded: self.collection.schema()?.max_doc_count_per_segment(),
        })
    }

    fn file_count(&self) -> Result<u64> {
        fn walk(dir: &std::path::Path) -> u64 {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return 0;
            };
            entries
                .flatten()
                .map(|entry| match entry.file_type() {
                    Ok(kind) if kind.is_dir() => walk(&entry.path()),
                    Ok(_) => 1,
                    Err(_) => 0,
                })
                .sum()
        }

        Ok(walk(&self.dir))
    }

    /// Counted from the directory, as [`file_count`](Self::file_count) is:
    /// the engine keeps each segment in a directory of its own, writes one
    /// `embedding.index.<n>.proxima` there per flush, and leaves one per
    /// segment once it has compacted. So every block past a segment's first is
    /// one a compaction would merge.
    fn unmerged_blocks(&self) -> Result<u64> {
        let prefix = format!("{FIELD_VECTOR}.index.");
        let Ok(segments) = std::fs::read_dir(self.dir.join(COLLECTION)) else {
            return Ok(0);
        };
        Ok(segments
            .flatten()
            .filter(|segment| segment.file_type().is_ok_and(|kind| kind.is_dir()))
            .map(|segment| {
                let blocks = std::fs::read_dir(segment.path()).map_or(0, |files| {
                    files
                        .flatten()
                        .filter(|file| {
                            let name = file.file_name();
                            let name = name.to_string_lossy();
                            name.starts_with(&prefix) && name.ends_with(".proxima")
                        })
                        .count() as u64
                });
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
/// The dot product, accumulated in f32.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn collect_scored(keys: Keys, docs: Vec<Doc>, orient: impl Fn(f32) -> f32) -> Vec<Scored> {
    docs.iter()
        .filter_map(|doc| {
            let topic = keys.topic(doc.get_pk()?)?;
            Some(Scored::new(topic, orient(doc.get_score())))
        })
        .collect()
}

#[cfg(test)]
mod upkeep {
    use super::{Segmentation, is_fragmented, segment_documents, vector_index_lags, wastes_disk};

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
            !is_fragmented(136),
            "136 files is inside the budget, which is why this condition \
             cannot be the graph's"
        );
        assert!(
            vector_index_lags(30_000, 0.0),
            "thirty thousand documents outside the graph has to queue a build"
        );
    }

    /// Below the crossover, waiting is right rather than merely tolerable.
    #[test]
    fn a_remainder_cheaper_to_scan_than_to_index_waits() {
        assert!(!vector_index_lags(1_000, 0.0));
        assert!(!vector_index_lags(100_000, 0.9));
        // The bound is on the remainder, not on the project: the size that
        // never fired as a threshold is fully indexed here.
        assert!(!vector_index_lags(1_000_000, 1.0));
    }

    /// A small index is compacted after a few flushes, not after the file
    /// budget's forty; a large one keeps a larger allowance of blocks.
    #[test]
    fn unmerged_blocks_are_bounded_by_the_index_size() {
        assert!(!wastes_disk(8, 1), "eight blocks is the floor");
        assert!(wastes_disk(9, 43), "a ninth on a small index is waste");
        assert!(
            !is_fragmented(9 * 6),
            "and the file budget would not have asked yet"
        );
        assert!(!wastes_disk(100, 13_014), "a large index is allowed more");
        assert!(wastes_disk(102, 13_014));
    }

    /// A completeness outside 0.0..=1.0 must not read as a negative remainder.
    #[test]
    fn a_nonsense_completeness_does_not_wrap() {
        assert!(vector_index_lags(30_000, -1.0));
        assert!(!vector_index_lags(30_000, 2.0));
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
    use super::{Marker, ProjectionIndex, VectorIndex};
    use crate::Profile;

    /// What an index was built with reads back as the profile that built it.
    ///
    /// The server opens a project nobody asked for under this answer, and
    /// the answer being wrong has no quiet failure: the open is refused, or
    /// -- for a directory with no index -- an empty one is created under the
    /// guess. So every profile is written and read back, and an empty
    /// directory has to answer that there is nothing to open.
    #[test]
    fn an_index_reads_back_the_profile_it_was_built_with() {
        for profile in [Profile::Speed, Profile::Balanced, Profile::Accuracy] {
            for index in VectorIndex::ALL {
                let dir = tempfile::tempdir().expect("a directory");
                Marker::current(profile, index)
                    .write(dir.path())
                    .expect("marking");
                assert_eq!(
                    ProjectionIndex::built_for(dir.path()).expect("reading"),
                    Some((profile, index))
                );
            }
        }

        // An index from before the vector indexes names no profile to open it
        // under: only a rebuild can.
        let legacy = tempfile::tempdir().expect("a directory");
        std::fs::write(
            legacy.path().join("profile"),
            format!("{}\ntopic\nfp32\nnamed", Profile::Accuracy.model_id()),
        )
        .expect("marking");
        assert_eq!(
            ProjectionIndex::built_for(legacy.path()).expect("reading"),
            None
        );

        let empty = tempfile::tempdir().expect("a directory");
        assert_eq!(
            ProjectionIndex::built_for(empty.path()).expect("reading"),
            None,
            "a directory with no index named a profile to open it under"
        );
    }
}
