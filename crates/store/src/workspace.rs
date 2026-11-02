//! On-disk layout and connection settings for a local install.
//!
//! Defined in `memory_core::workspace`, which the command-line client shares
//! without linking this crate, and named here where the store's callers have
//! always found it.

pub use memory_core::workspace::{LocalServer, StartedWith, Workspace};
