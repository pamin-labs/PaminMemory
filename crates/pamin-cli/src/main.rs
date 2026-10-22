//! `pamin`, the command line an agent runs.
//!
//! The CLI is the primary surface: a shell command plus a skill file is a
//! zero-integration path for any agent that can run a process, with no client
//! library to adopt and no service to stand up.
//!
//! This executable is the client and nothing else: it is built from
//! `pamin_client` alone, which reaches none of the inference runtime, the
//! vector engine, TLS or the database driver. Every command starts it, and
//! what a process links is what it pays for at every start, so a command pays
//! for parsing, one socket round trip and printing. Being the server, and
//! stopping a database that no server is running to stop, take the engine;
//! this hands those to `pamin-server`, the executable installed beside it.
//!
//! Nothing here may name `pamin_engine`, `pamin_index` or `pamin_store`:
//! rustc links only the crates a binary uses, and naming one would put it all
//! back. `tests/install.rs` checks what the installed binary loads.

use anyhow::{Context, Result};
use pamin_client::cli;

fn main() -> Result<()> {
    let (cli, matches) = cli::parse()?;
    // One request and one answer, which one thread does without starting a
    // worker per core first.
    let unserved = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(cli::run(cli, matches))?;
    match unserved {
        Some(_) => hand_over(),
        None => Ok(()),
    }
}

/// Runs this command line again as `pamin-server`, which does what a client
/// cannot.
///
/// The same arguments, so the server parses, resolves and refuses exactly
/// what this did, and on Unix in this process's place: `pamin serve` in the
/// foreground is then the server itself, with its signals, its streams and
/// its exit status.
fn hand_over() -> Result<()> {
    let server = pamin_client::client::server_exe()?;
    let mut command = std::process::Command::new(&server);
    command.args(std::env::args_os().skip(1));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let error = command.exec();
        Err(error).with_context(|| format!("running {}", server.display()))
    }
    #[cfg(not(unix))]
    {
        let status = command
            .status()
            .with_context(|| format!("running {}", server.display()))?;
        std::process::exit(status.code().unwrap_or(1));
    }
}
