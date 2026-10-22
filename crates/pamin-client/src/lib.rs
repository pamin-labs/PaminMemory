//! The client half of the `pamin` command line: the arguments every command
//! takes, the wire protocol to the workspace's server, reaching or starting
//! that server, and rendering what it answers.
//!
//! The `pamin` executable is built from this crate alone. None of it needs the
//! inference runtime, the vector engine, TLS or the database driver, and this
//! crate reaches none of them, so a command an agent runs pays for parsing,
//! one socket round trip and printing. The server half -- `pamin-server`, in
//! `pamin-cli` -- runs each command against the engine and answers with the
//! result types defined here.

pub mod cli;
pub mod client;
pub mod command;
pub mod output;
pub mod protocol;
pub mod settings;
