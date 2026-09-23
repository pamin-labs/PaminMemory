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

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use pamin_core::{Scored, TopicId};

use crate::embedding::Profile;
use zvec_rust::{
    Collection, CollectionOptions, CollectionSchema, DataType, Doc, FieldSchema, Fts,
    FtsQueryParams, HnswQueryParams, IndexParams, MetricType, QuantizeType, SearchQuery,
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

    /// How this index is segmented, against what the policy would choose.
    fn segmentation(&self) -> Result<Segmentation>;

    /// What text this index's vectors are embedded from, which every write to
    /// it has to follow.
    fn passage(&self) -> Passage;
}

/// One document as an index holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Stored {
    /// The memory's text, exactly as written. The segmented field is derived
    /// from it, so it is all a copy needs to rebuild both lexical fields.
    pub content: String,
    /// The vector, as it was embedded under the index's [`Passage`].
    pub embedding: Vec<f32>,
}

/// A lexical or vector index over topics.
pub struct ProjectionIndex {
    collection: Collection,
    segmenter: Arc<Segmenter>,
    dir: std::path::PathBuf,
    /// How this collection's vectors are stored, so a query's refiner flag
    /// follows the index rather than the environment: reading the variable
    /// again at query time would let a process that changed it mid-flight ask
    /// for a refiner that is not there.
    storage: VectorStorage,
    /// What this index's vectors were embedded from. See [`Passage`].
    passage: Passage,
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

/// What one document in this index stands for.
///
/// Recorded beside the model because an index keyed by something else is not
/// stale, it is silently empty: the old scheme's identifiers are read as the
/// new scheme's, match nothing, and every search comes back with no results
/// and no error anywhere. Changing what a document is keyed by means changing
/// this, which turns that silence into a message naming `pamin reindex`.
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
/// by orders of magnitude keeps the size it was created with until
/// `pamin reindex` rebuilds it.
pub fn segment_documents(documents: u64) -> u64 {
    (documents / TARGET_SEGMENTS).clamp(SMALLEST_SEGMENT, LARGEST_SEGMENT)
}

/// How an index is segmented, against what the policy would choose now.
///
/// Reported because a workspace has no other way to find out. The size is
/// recorded when the collection is created and a workspace is created empty,
/// so every grown project records [`SMALLEST_SEGMENT`] and holds one segment
/// per ten thousand documents rather than the four the policy aims at -- 14
/// over 131,924. Measured over fifty thousand, 25 segments answer a query in
/// 39.8 ms where four answer in 16.9. `pamin reindex` reshapes an index without
/// embedding anything again; see [`Previous`].
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
/// for the cases that still reach it -- an import, and a workspace written to
/// with no server behind it -- and because a bound that is not being
/// approached is the one worth having.
const MAX_FILES: u64 = 256;

/// Whether an index is spread across more files than it should be.
pub fn is_fragmented(files: u64) -> bool {
    files > MAX_FILES
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
/// How the stored vectors are kept.
///
/// The vector field is the largest thing on disk: 64.2 MB of a 116 MB index
/// over 13,014 documents, 55% of it, against 44.7 MB for both full-text fields
/// and 7.4 MB for the identifier column. A 1024-dimensional fp32 vector is
/// 4 KB a document and that is most of the 64.
///
/// `Fp32` -- no quantization -- until a sweep says otherwise, and the sweep is
/// the point of this being a setting: [`PAMIN_VECTOR_STORAGE`] lets
/// `crates/pamin-index/tests/recall.rs` measure a cell without a rebuild of
/// the world, because the one thing reading the binding cannot answer is
/// whether the refiner keeps a full-precision copy beside the quantized one --
/// in which case quantizing costs disk rather than saving it.
///
/// ADR 0001 records a previous attempt at this returning recall@10 of 0.000
/// with no error and no visible symptom, under the only configuration that
/// existed then (`Int8`, before the binding exposed rotation). That is why the
/// storage is recorded in the profile marker: an index built one way and read
/// another is the silent-wrong-answer shape, and the marker turns it into a
/// message naming `pamin reindex`.
const VECTOR_STORAGE: VectorStorage = VectorStorage::Fp32;

/// Overrides [`VECTOR_STORAGE`], for the sweep that settles it.
///
/// Deliberately undocumented: a caller has no way to evaluate it, and reading
/// an index built under one value with another is exactly what the marker
/// exists to refuse.
const PAMIN_VECTOR_STORAGE: &str = "PAMIN_VECTOR_STORAGE";

/// How a stored vector is kept, and what the marker records.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VectorStorage {
    /// Four bytes a dimension, exactly what the model produced.
    Fp32,
    /// Two bytes a dimension.
    Fp16,
    /// One byte a dimension.
    Int8,
    /// Half a byte a dimension.
    Int4,
    /// A bit a dimension.
    ///
    /// Listed and not reachable: the engine refuses to train a RaBitQ
    /// quantizer without a `raw_vector_provider`, which this binding does not
    /// expose, so asking for it fails when the graph is built rather than
    /// returning a worse index. Kept as a name so the refusal is recorded
    /// where someone would look for it, and because it is the one storage
    /// whose codes are small enough to change the disk answer -- see
    /// `index_params`.
    Rabitq,
}

impl VectorStorage {
    /// The label the profile marker carries.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Rabitq => "rabitq",
        }
    }

    fn parse(label: &str) -> Option<Self> {
        match label.trim() {
            "fp32" => Some(Self::Fp32),
            "fp16" => Some(Self::Fp16),
            "int8" => Some(Self::Int8),
            "int4" => Some(Self::Int4),
            "rabitq" => Some(Self::Rabitq),
            _ => None,
        }
    }

    fn quantize(self) -> Option<QuantizeType> {
        match self {
            Self::Fp32 => None,
            Self::Fp16 => Some(QuantizeType::Fp16),
            Self::Int8 => Some(QuantizeType::Int8),
            Self::Int4 => Some(QuantizeType::Int4),
            Self::Rabitq => Some(QuantizeType::Rabitq),
        }
    }

    /// Whether a query should ask for the refiner.
    ///
    /// It rescores against a full-precision copy that exists only where the
    /// stored vectors were quantized, and asking for one otherwise fails
    /// outright rather than being ignored -- so this follows the storage rather
    /// than being a setting of its own.
    fn refines(self) -> bool {
        self.quantize().is_some()
    }

    /// The index parameters for this storage.
    fn index_params(self) -> Result<IndexParams> {
        let Some(quantize) = self.quantize() else {
            return Ok(IndexParams::hnsw(
                MetricType::Cosine,
                GRAPH_DEGREE,
                GRAPH_EFFORT,
            )?);
        };

        // Rotation is left off, and that is a measurement rather than the
        // binding's default carried through.
        //
        // It was on here for every quantized storage, on the reasoning that
        // spreading the bits across dimensions that carry comparable
        // information must help the coarse storages and could not hurt the
        // others. Both halves of that were wrong. The engine accepts it only
        // for int8 and int4 -- for anything else it refuses when the *segment*
        // opens its vector field rather than when the parameters are built, so
        // fp16 presented as a segment that would not take writes. And on the
        // two storages that do accept it, it is ruinous: recall@10 over 50,000
        // clustered vectors is **0.0530 with rotation and 0.9980 without** for
        // int8, 0.0580 against 0.9990 for int4. Everything else about the two
        // runs is equal, including the bytes on disk, and the failure is
        // silent -- an index that returns plausible neighbours that are not
        // the nearest ones, which is the exact shape ADR 0001 records from the
        // last quantization attempt.
        //
        // Rotation needs a fitted transform, and nothing here fits one; the
        // binding's RaBitQ path says as much out loud, refusing to train
        // without a `raw_vector_provider`. So this stays off until something
        // supplies that, and `scratch_quantize.rs` is what would notice.
        Ok(IndexParams::hnsw_with_quantize(
            MetricType::Cosine,
            GRAPH_DEGREE,
            GRAPH_EFFORT,
            quantize,
        )?)
    }
}

/// The storage this process will build and read with.
fn vector_storage() -> VectorStorage {
    std::env::var(PAMIN_VECTOR_STORAGE)
        .ok()
        .and_then(|value| VectorStorage::parse(&value))
        .unwrap_or(VECTOR_STORAGE)
}

const GRAPH_DEGREE: i32 = 32;

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
const SEARCH_EFFORT: i32 = 700;

impl ProjectionIndex {
    /// Opens the index at `dir`, creating it if absent.
    ///
    /// What the index was built for is recorded on creation and checked on
    /// every reopen: the embedding model, because mixing embedding spaces
    /// produces distances that mean nothing, and what a document stands for,
    /// because reading one scheme's keys as another's matches nothing at all.
    /// Neither failure looks like a failure -- one returns plausible rankings
    /// from meaningless distances and the other returns no results and no
    /// error -- so both are enforced rather than documented.
    pub fn open(
        dir: &Path,
        legacy_dir: &Path,
        profile: Profile,
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

        Self::open_sized(dir, profile, access, segment_documents(documents))
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
        access: Access,
        segment: u64,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let storage = vector_storage();
        let passage = match Marker::read(dir)? {
            Some(recorded) => {
                if recorded.storage != storage {
                    return Err(IndexError::VectorStorageMismatch {
                        indexed: recorded.storage.label().to_string(),
                        requested: storage.label().to_string(),
                    });
                }
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
                recorded.passage
            }
            None => {
                // What was actually built, not what was asked for. The two are
                // the same today; they stop being the same the moment a
                // storage needs a capability the machine may not have, and a
                // marker recording the request would then be read as a
                // description of the index -- ADR 0001's silent wrong answer.
                Marker::current(profile).write(dir)?;
                PASSAGE
            }
        };

        let mut index = Self::open_with_dimensions(dir, profile.dimensions(), access, segment)?;
        index.passage = passage;
        Ok(index)
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
    /// Whatever is at `dir` already is discarded first: it can only be a copy
    /// that did not finish.
    pub(crate) fn create_beside(
        source: &Path,
        dir: &Path,
        profile: Profile,
        documents: u64,
    ) -> Result<Self> {
        Self::discard(dir)?;
        std::fs::create_dir_all(dir)?;
        std::fs::copy(source.join(Marker::FILE), dir.join(Marker::FILE))?;
        Self::open_sized(
            dir,
            profile,
            Access::ReadWrite,
            segment_documents(documents),
        )
    }

    /// Opens the index at `dir` for writing, refusing to create one.
    ///
    /// For reopening an index after its directory was moved into place, where
    /// finding nothing there is a fault: an open that created an empty index
    /// would hand the caller a project with no memories and no error.
    pub(crate) fn reopen(dir: &Path, profile: Profile) -> Result<Self> {
        if !std::fs::exists(dir.join(COLLECTION))? {
            return Err(IndexError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no index at {}", dir.display()),
            )));
        }
        Self::open_sized(dir, profile, Access::ReadWrite, segment_documents(0))
    }

    fn open_with_dimensions(
        dir: &Path,
        dimensions: u32,
        access: Access,
        segment: u64,
    ) -> Result<Self> {
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
            .add_vector_field(
                FIELD_VECTOR,
                DataType::VectorFp32,
                dimensions,
                vector_storage().index_params()?,
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
            storage: vector_storage(),
            passage: PASSAGE,
        })
    }

    fn document(&self, topic: TopicId, content: &str, embedding: &[f32]) -> Result<Doc> {
        let mut doc = Doc::new()?;
        let key = topic.to_string();
        doc.set_pk(&key);
        doc.add_string(FIELD_ID, &key)?;
        doc.add_string(FIELD_SEGMENTED, &self.segmenter.segment_for_index(content))?;
        doc.add_string(FIELD_NGRAM, content)?;
        doc.add_vector_f32(FIELD_VECTOR, embedding)?;
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
        Ok(collect_scored(self.collection.query(&search)?, |score| {
            score
        }))
    }

    /// The documents stored under these topics, keyed by primary key.
    ///
    /// The one way anything reads a document back, so a rebuild lending its
    /// vectors and a reshape copying whole documents cannot come to disagree
    /// about what a stored document is. The text comes from the n-gram field,
    /// which holds the content verbatim; the segmented one is derived from it.
    fn fetch(&self, topics: &[TopicId], vectors: bool) -> Result<HashMap<String, Doc>> {
        let keys: Vec<String> = topics.iter().map(ToString::to_string).collect();
        let mut stored: HashMap<String, Doc> = HashMap::with_capacity(keys.len());
        for chunk in keys.chunks(WRITE_BATCH) {
            let chunk: Vec<&str> = chunk.iter().map(String::as_str).collect();
            for doc in self
                .collection
                .fetch_with_options(&chunk, Some(&[FIELD_NGRAM]), vectors)?
            {
                if let Some(key) = doc.get_pk() {
                    stored.insert(key.to_string(), doc);
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
/// Four lines: the embedding model, what a document stands for, how vectors
/// are stored, and what text they were embedded from. A line an older index
/// does not have reads as what that index was built with, so an existing
/// workspace opens unchanged: no storage line is `fp32`, no passage line is
/// content alone.
struct Marker {
    model: String,
    grain: String,
    storage: VectorStorage,
    passage: Passage,
}

impl Marker {
    const FILE: &str = "profile";

    /// What an index built now, for this profile, is.
    fn current(profile: Profile) -> Self {
        Self {
            model: profile.model_id().to_string(),
            grain: DOCUMENT_GRAIN.to_string(),
            storage: vector_storage(),
            passage: PASSAGE,
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
            storage: lines
                .next()
                .and_then(VectorStorage::parse)
                .unwrap_or(VectorStorage::Fp32),
            passage: lines
                .next()
                .and_then(Passage::parse)
                .unwrap_or(Passage::Content),
        }))
    }

    fn write(&self, dir: &Path) -> Result<()> {
        std::fs::write(
            dir.join(Self::FILE),
            format!(
                "{}\n{}\n{}\n{}",
                self.model,
                self.grain,
                self.storage.label(),
                self.passage.label()
            ),
        )?;
        Ok(())
    }

    /// Whether a vector this index holds is the vector an index built now
    /// would compute for the same text: same model, same storage, same
    /// encoding, same keys.
    fn matches(&self, other: &Self) -> bool {
        self.model == other.model
            && self.grain == other.grain
            && self.storage == other.storage
            && self.passage == other.passage
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
        let lends =
            Marker::read(dir)?.is_some_and(|recorded| recorded.matches(&Marker::current(profile)));
        if !lends {
            ProjectionIndex::discard(dir)?;
            return Ok(None);
        }
        std::fs::rename(dir, &aside)?;
        let mut index = ProjectionIndex::open_with_dimensions(
            &aside,
            profile.dimensions(),
            Access::ReadOnly,
            segment_documents(0),
        )?;
        index.passage = PASSAGE;
        Ok(Some(Self { index, dir: aside }))
    }

    /// For each topic, its stored vector if the text stored with it is
    /// exactly `content`, and `None` otherwise.
    pub fn vectors(&self, wanted: &[(TopicId, &str)]) -> Result<Vec<Option<Vec<f32>>>> {
        self.lend(wanted, true)
    }

    /// How many of these topics [`vectors`](Self::vectors) would supply,
    /// without reading a vector.
    pub fn lends(&self, wanted: &[(TopicId, &str)]) -> Result<usize> {
        Ok(self.lend(wanted, false)?.iter().flatten().count())
    }

    fn lend(&self, wanted: &[(TopicId, &str)], vectors: bool) -> Result<Vec<Option<Vec<f32>>>> {
        let topics: Vec<TopicId> = wanted.iter().map(|(topic, _)| *topic).collect();
        let stored = self.index.fetch(&topics, vectors)?;
        wanted
            .iter()
            .map(|(topic, content)| {
                let Some(doc) = stored.get(&topic.to_string()) else {
                    return Ok(None);
                };
                if doc.get_string(FIELD_NGRAM)?.as_deref() != Some(*content) {
                    return Ok(None);
                }
                if !vectors {
                    return Ok(Some(Vec::new()));
                }
                Ok(doc.get_vector_f32(FIELD_VECTOR)?)
            })
            .collect()
    }

    /// Deletes the set-aside index, once the rebuild no longer needs it.
    pub fn discard(self) -> Result<()> {
        let Self { index, dir } = self;
        drop(index);
        ProjectionIndex::discard(&dir)
    }
}

impl Projection for ProjectionIndex {
    fn passage(&self) -> Passage {
        self.passage
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
        let stored = self.fetch(topics, true)?;
        topics
            .iter()
            .map(|topic| {
                let Some(doc) = stored.get(&topic.to_string()) else {
                    return Ok(None);
                };
                match (
                    doc.get_string(FIELD_NGRAM)?,
                    doc.get_vector_f32(FIELD_VECTOR)?,
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
    fn recall_vector(&self, embedding: &[f32], limit: u32) -> Result<Vec<Scored>> {
        let mut search = SearchQuery::new(FIELD_VECTOR, embedding, limit as i32)?;
        search.set_output_fields(&[FIELD_ID])?;
        search.set_include_vector(false)?;
        // No radius bound and the graph rather than a linear scan. The refiner
        // follows what the vectors were stored as, for the reason
        // `VectorStorage::refines` gives: asking for one over unquantized
        // vectors fails outright rather than being ignored.
        search.set_hnsw_params(HnswQueryParams::new(
            SEARCH_EFFORT,
            0.0,
            false,
            self.storage.refines(),
        ))?;
        // Cosine *distance*, which is what the engine reports for a cosine
        // index: nearest is zero. `Scored` requires larger to be better,
        // because everything above compares magnitudes -- summing a distance
        // would sum this channel backwards and reading its confidence would
        // read its worst candidate as its best. Cosine distance is
        // `1 - similarity`, so this is the exact inverse and not a rescaling.
        Ok(collect_scored(self.collection.query(&search)?, |score| {
            1.0 - score
        }))
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
fn collect_scored(docs: Vec<Doc>, orient: impl Fn(f32) -> f32) -> Vec<Scored> {
    docs.iter()
        .filter_map(|doc| {
            let pk = doc.get_pk()?;
            let topic = uuid::Uuid::parse_str(pk).ok()?;
            Some(Scored::new(TopicId::from(topic), orient(doc.get_score())))
        })
        .collect()
}

#[cfg(test)]
mod upkeep {
    use super::{Segmentation, is_fragmented, segment_documents, vector_index_lags};

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

    /// A completeness outside 0.0..=1.0 must not read as a negative remainder.
    #[test]
    fn a_nonsense_completeness_does_not_wrap() {
        assert!(vector_index_lags(30_000, -1.0));
        assert!(!vector_index_lags(30_000, 2.0));
    }
}
