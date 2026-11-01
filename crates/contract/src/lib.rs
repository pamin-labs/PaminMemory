//! PaminMemory's operation contract: each operation's arguments, the result
//! it answers with, and how that result reads as text.
//!
//! The one definition every surface uses. The CLI parses these arguments,
//! the Unix socket carries them to the server and its results back, and MCP
//! serves them as tools. Computing a result is the server's, in `memory-cli`.
//!
//! Every argument and result type also describes itself as JSON Schema
//! 2020-12, which is what MCP tools are declared in. A field only a person
//! tuning retrieval should set is left out of the schema, and keeps the
//! default it has on the command line.
//!
//! Kept light for the reason `memory-client` is: the `pamin-memory` executable is
//! built from it, and what it links every command pays for at start.

/// The contract's version, which MCP clients see as the server's. Within a
/// major version the contract only grows: a removed field, or one whose type or
/// meaning changed, starts the next one.
pub const VERSION: &str = "1.0.0";

pub mod cascade;
pub mod grep;
pub mod import;
pub mod ingest;
pub mod init;
pub mod link;
pub mod neighbors;
pub mod read;
pub mod reindex;
pub mod search;
pub mod stop;
pub mod sync;
pub mod topics;
pub mod unlink;
pub mod unsync;
pub mod validity;
pub mod write;
