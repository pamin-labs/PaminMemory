//! How far one search reaches, and the depths the product searches with.
//!
//! Here rather than beside the engine and the index that use them, because the
//! command line offers each as a flag of `pamin search` and shows its default,
//! and the client that parses those flags links neither. Each is still
//! defined once: `pamin_engine::Depths`, `pamin_index::reranking::DEPTH` and
//! `pamin_index::projection::SEARCH_EFFORT` are these.

/// How deep each channel reaches before fusion.
///
/// These are inputs rather than constants because they are provisional: the
/// evaluation harness exists to settle them, and it cannot sweep a value that
/// is compiled in. The defaults are the ones the architecture specifies, so
/// nothing changes for a caller that does not ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Depths {
    /// Candidates each channel contributes.
    ///
    /// Deep enough for rank fusion to find agreement between channels, shallow
    /// enough that reranking stays cheap. Never zero, which would leave
    /// every channel, and so the search, empty.
    pub channel: std::num::NonZeroU32,
    /// Edges the graph channel walks out from its seeds.
    ///
    /// Two hops reaches a topic's neighbours and their neighbours, which is
    /// where "related to something related to this" stops being informative
    /// and starts being most of the project.
    pub graph: u8,
    /// How many of the fused results the reranker is shown as its head; see
    /// [`RERANK_DEPTH`] for what settled the default.
    /// Never zero, which would rerank nothing while calling it reranked.
    pub rerank: std::num::NonZeroUsize,
    /// How wide the in-memory vector graph is searched; see
    /// [`SEARCH_EFFORT`]. Never zero, which is no width for a graph search to
    /// have.
    pub effort: std::num::NonZeroU32,
}

impl Depths {
    /// What the product searches with.
    pub const DEFAULT: Self = Self {
        channel: std::num::NonZeroU32::new(50).unwrap(),
        graph: 2,
        rerank: std::num::NonZeroUsize::new(RERANK_DEPTH).unwrap(),
        effort: std::num::NonZeroU32::new(SEARCH_EFFORT).unwrap(),
    };
}

impl Default for Depths {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// How many of the fused results the reranker looks at.
///
/// Thirty, and it was twenty until the default tier was measured at it. The
/// twenty was settled on the `fast` model, where thirty bought +0.0009 of
/// XQuAD-R cross-lingual -- "twenty is simply where it stops". The default is
/// now `accurate`, a model fourteen times the size, and it keeps finding
/// answers further down. Paired against twenty through `search_reranked`,
/// nDCG@10 (the `DEPTH_VARIANTS` arm of each harness):
///
/// | depth | XQuAD-R cross (1,190) | own cross (43) | MIRACL (482) | MuSiQue (1,000) |
/// |---|---|---|---|---|
/// | 10 | **−0.0365**, p = 0.0001 | −0.0144 | −0.0021 | −0.0031, p = 0.069 |
/// | 15 | **−0.0096**, p = 0.0001 | +0.0027 | +0.0015 | −0.0004 |
/// | **30** | **+0.0063**, 192W/134L, p = 0.0001 | +0.0117 | −0.0008 | +0.0011 |
/// | 40 | **+0.0076**, p = 0.0001 | +0.0235, p = 0.062 | +0.0002 | +0.0007 |
///
/// Unmarked cells are not significant, and XQuAD-R's same-language group
/// moves by under 0.0006 at every depth. So shallower is refused -- fifteen
/// looked free on MIRACL and MuSiQue and costs a hundredth where the
/// reranker matters most -- and deeper is a significant gain on the
/// cross-lingual group and a loss nowhere. Thirty takes five sixths of forty's
/// gain for half its extra pairs: accuracy decides the direction and latency
/// the distance, and forty's last 0.0013 is not worth another third of the
/// reranker's time. The pass costs half again what it did at twenty in model
/// pairs. Its wall time at thirty is under
/// `pamin_index::reranking::BATCH_TOKENS`, taken on a machine shared with
/// other measurements rather than a quiet one, so it is the ratios there that
/// carry.
///
/// The default of a search's own depth ([`Depths::rerank`]), so a sweep sets
/// it per search rather than per process. `DEPTH`, `BATCH` and
/// `MAX_TOKENS` were each settled by measurement, and two of the three were
/// settled on the scratch harness whose figures turned out to be about half
/// again too high -- so they have to be re-settleable by the harness rather
/// than by editing a constant and rebuilding: this one through the search,
/// the other two through `pamin_index::RerankSettings`.
pub const RERANK_DEPTH: usize = 30;

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
/// [`VectorIndex::Memory`](crate::settings::VectorIndex::Memory) searches,
/// against the bar that fp32 at 700 sets -- within 0.002 of its recall on
/// 50,000 clustered vectors, 0.001 on MIRACL:
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
pub const SEARCH_EFFORT: u32 = 700;
