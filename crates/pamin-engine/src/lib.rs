//! Composing the store, the index, and the embedder.
//!
//! The authority and the projection live in separate crates everywhere else.
//! This crate is the one place that holds both, so it is also the only place
//! where the two can drift out of step. It sits above `pamin-core` rather than
//! beside it, which is what keeps `zvec` types from reaching the domain layer.

mod engine;

pub use engine::{Depths, Engine, SearchHit};
