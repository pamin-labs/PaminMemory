//! PostgreSQL authority store: evidence, version ledger, and the cascade outbox.
//!
//! PostgreSQL is the sole authority. The projection index is derived data and
//! can be rebuilt from what lives here, which is what makes swapping the index
//! a reindex rather than a migration.

pub mod database;
pub mod error;
pub mod graph;
pub mod jobs;
pub mod migrate;
pub mod repository;
mod sql;
pub mod workspace;

pub use database::{Connections, Database};
pub use error::{Result, StoreError};
/// What every `repository` and `graph` function takes as its first argument.
///
/// Re-exported because those signatures are generic over it, so a caller
/// outside this crate cannot write one down without naming it. A pool, a
/// connection, and a transaction borrowed as a connection all satisfy it,
/// which is what lets one function serve a query and a step of a transaction.
pub use sqlx::PgExecutor;
pub use workspace::{LocalServer, Workspace};
