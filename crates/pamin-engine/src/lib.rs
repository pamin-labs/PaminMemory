//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection live in separate crates everywhere else.
//! This crate is the one place that holds both, so it is also the only place
//! where the two can drift out of step. It sits above `pamin-core` rather than
//! beside it, which is what keeps `zvec` types from reaching the domain layer.

mod cascade;
mod engine;
mod reshape;
mod settings;

pub use cascade::{Drained, Owed};
pub use engine::{
    Depths, Engine, Models, RERANK_FUSION, Rebuilt, Recorded, Retired, SearchHit, Write,
    file_locator, place, rerankable, score_blend_order,
};
pub use settings::Settings;
