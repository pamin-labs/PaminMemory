//! PostgreSQL authority store: evidence, version ledger, and the cascade outbox.
//!
//! PostgreSQL is the sole authority. The projection index is derived data and
//! can be rebuilt from what lives here, which is what makes swapping the index
//! a reindex rather than a migration.

pub mod database;
pub mod error;
pub mod graph;
pub mod migrate;
pub mod repository;
mod sql;
pub mod workspace;

pub use database::Database;
pub use error::{Result, StoreError};
pub use workspace::{LocalServer, Workspace};
