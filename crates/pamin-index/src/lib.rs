//! Projection index: multilingual segmentation, lexical and vector recall channels.
//!
//! Everything here is derived data. Losing it costs a reindex, not a migration,
//! which is what makes a pre-1.0 index engine an acceptable dependency.

mod attention;
mod descriptors;
pub mod embedding;
mod encoder;
pub mod error;
mod half;
mod hub;
mod inference;
mod prepared;
pub mod projection;
pub mod reranking;
mod reshape;
pub mod segmentation;
mod tokenizer;

pub use descriptors::raise_open_file_limit;
pub use embedding::{Embedder, Profile};
pub use error::{IndexError, Result};
pub use half::as_stored;
pub use inference::Device;
pub use projection::{
    Access, Passage, Previous, Projection, ProjectionIndex, Segmentation, Stored, VectorIndex,
    is_fragmented, segment_documents, vector_index_lags, wastes_disk,
};
pub use reranking::{Ranked, Rerank, Reranked, Reranker};
pub use reshape::{Held, Reshape, Reshaped};
pub use segmentation::{Segmenter, detect_language};
