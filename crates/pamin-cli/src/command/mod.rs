//! What each command does, run against the server's session.
//!
//! Each command's arguments, its result and how that result reads as text are
//! the client's, in `pamin_client::command`; this is the half that computes
//! the result.

pub mod cascade;
pub mod grep;
pub mod import;
pub mod init;
pub mod link;
pub mod neighbors;
pub mod read;
pub mod reindex;
pub mod resolve;
pub mod search;
pub mod stop;
pub mod topics;
pub mod unlink;
pub mod write;
