//! Each command's arguments, the result it answers with, and how that result
//! reads as text.
//!
//! The client half of every command: what is parsed here crosses the socket as
//! the request, and what the server computes crosses back as one of these
//! results. Computing a result is the server's, in `pamin-cli`.

pub mod cascade;
pub mod grep;
pub mod import;
pub mod init;
pub mod link;
pub mod neighbors;
pub mod read;
pub mod reindex;
pub mod search;
pub mod stop;
pub mod topics;
pub mod unlink;
pub mod validity;
pub mod write;
