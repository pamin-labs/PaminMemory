//! Projection index: multilingual segmentation, lexical and vector recall channels.
//!
//! Everything here is derived data. Losing it costs a reindex, not a migration,
//! which is what makes a pre-1.0 index engine an acceptable dependency.

mod descriptors;
pub mod embedding;
pub mod error;
mod inference;
pub mod projection;
pub mod reranking;
pub mod segmentation;

pub use descriptors::raise_open_file_limit;
pub use embedding::{Embedder, Profile};
pub use error::{IndexError, Result};
pub use inference::Device;
pub use projection::{
    Access, Projection, ProjectionIndex, Segmentation, VectorStorage, is_fragmented,
    segment_documents, vector_index_lags,
};
pub use reranking::{Licence, Ranked, Rerank, Reranked, Reranker};
pub use segmentation::{Segmenter, detect_language};
