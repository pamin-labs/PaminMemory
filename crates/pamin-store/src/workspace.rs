//! On-disk layout and connection settings for a local install.
//!
//! Defined in `pamin_core::workspace`, which the command-line client shares
//! without linking this crate, and named here where the store's callers have
//! always found it.

pub use pamin_core::workspace::{LocalServer, StartedWith, Workspace};
