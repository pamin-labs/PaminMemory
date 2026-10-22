//! The client half of the `pamin` command line: the arguments every command
//! takes, the wire protocol to the workspace's server, reaching or starting
//! that server, and rendering what it answers.
//!
//! None of it needs the inference runtime, the vector engine, TLS or the
//! database driver, and this crate reaches none of them, so a client built
//! from it alone pays for parsing, one socket round trip and printing. The
//! server half -- running each command against the engine, and answering with
//! the result types defined here -- is `pamin-cli`'s.

pub mod cli;
pub mod client;
pub mod command;
pub mod output;
pub mod protocol;
pub mod settings;
