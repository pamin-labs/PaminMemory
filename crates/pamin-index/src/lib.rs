//! Projection index: multilingual segmentation, lexical and vector recall channels.
//!
//! Everything here is derived data. Losing it costs a reindex, not a migration,
//! which is what makes a pre-1.0 index engine an acceptable dependency.

pub mod embedding;
pub mod error;
pub mod projection;
pub mod segmentation;

pub use embedding::{Embedder, Profile};
pub use error::{IndexError, Result};
pub use projection::ProjectionIndex;
pub use segmentation::{Segmenter, detect_language};
