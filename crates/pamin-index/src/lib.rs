//! Projection index: multilingual segmentation, lexical and vector recall channels.
//!
//! Everything here is derived data. Losing it costs a reindex, not a migration,
//! which is what makes a pre-1.0 index engine an acceptable dependency.

mod activity;
mod attention;
#[cfg(test)]
mod build_profile;
mod descriptors;
pub mod embedding;
mod encoder;
pub mod error;
mod half;
mod hub;
mod inference;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native;
mod onnx;
mod prepared;
pub mod projection;
pub mod reranking;
mod reshape;
pub mod segmentation;
mod tokenizer;
mod vector_math;
#[cfg(target_os = "windows")]
mod winml;

pub use activity::LoadActivity;
pub use descriptors::raise_open_file_limit;
pub use embedding::{DIMENSIONS, Embedder, MODEL_ID};
pub use error::{IndexError, Result};
pub use half::as_stored;
pub use hub::cache_root as model_cache_root;
pub use inference::{
    CUDA_SELECTION, Device, EXECUTION_SETTINGS, PlanOrigin, PlanUse, cuda_devices,
};

/// What this crate was built from and how: a sha256 over its sources, the
/// workspace manifests and Cargo config, the target, profile, opt-level,
/// debug setting, rustflags, target features, every Cargo profile override
/// and rustc's version. Execution plans are keyed by it, and an evaluation
/// record names its build by it.
pub const BUILD_FINGERPRINT: &str = env!("PAMIN_INFERENCE_BUILD");

/// Part of the same build, readable: each Cargo profile override set and
/// each `cfg` the target resolved to (`CARGO_CFG_*`, the target features
/// among them), as `NAME=value` lines separated by 0x1f.
pub const BUILD_CARGO: &str = env!("PAMIN_INFERENCE_CARGO");
pub use projection::{
    Access, Passage, Previous, Projection, ProjectionIndex, Segmentation, Stored, VectorDocument,
    VectorIndex, is_fragmented, segment_documents, vector_index_lags, wastes_disk,
};
pub use reranking::{Ranked, Reranked, Reranker};
pub use reshape::{Held, Reshape, Reshaped};
pub use segmentation::{Segmenter, detect_language};
