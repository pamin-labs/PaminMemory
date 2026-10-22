//! `pamin-server`, the resident server behind the `pamin` command line.
//!
//! It holds the database, the index and the models, and answers the commands
//! `pamin` sends it. `pamin` starts it as `pamin-server serve`, and hands it
//! `pamin serve` and a `pamin stop` that finds no server to ask.
//!
//! It takes the same command line as `pamin`, parsed by the same
//! `pamin_client`, and runs any command the same way; what it adds is what a
//! client cannot do -- be the server, and stop a database that no server is
//! running to stop -- and with it everything that links: the inference
//! runtime, the vector engine, TLS and the database driver.

mod command;
mod registry;
mod server;
mod session;
mod settings;

use anyhow::Result;
use pamin_client::cli::{self, Command, Unserved};

fn main() -> Result<()> {
    // Before the runtime starts its worker threads: glibc fixes how many heap
    // arenas it will make the first time a thread needs one of its own.
    server::tune_heap();
    let (cli, matches) = cli::parse()?;
    // Only the server runs work on the runtime. Every other command makes one
    // request and waits for the answer, which one thread does without
    // starting a worker per core first.
    let mut runtime = match cli.command {
        Command::Serve => tokio::runtime::Builder::new_multi_thread(),
        _ => tokio::runtime::Builder::new_current_thread(),
    };
    runtime.enable_all().build()?.block_on(async {
        match cli::run(cli, matches).await? {
            Some(unserved) => answer(unserved).await,
            None => Ok(()),
        }
    })
}

/// Does what the client left to this binary.
async fn answer(unserved: Unserved) -> Result<()> {
    match unserved {
        Unserved::Serve { workspace, server } => {
            // Before anything opens an index. The server is the only process
            // the CLI opens one in, and a process holding the descriptors a
            // large index needs under the 1,024 a Linux process starts with
            // does not degrade, it fails.
            match pamin_index::raise_open_file_limit() {
                Ok((before, after)) if after > before => {
                    tracing::info!(before, after, "raised the open-file limit")
                }
                Ok(_) => {}
                Err(error) => tracing::warn!(%error, "could not raise the open-file limit"),
            }
            server::run(&workspace, server).await
        }
        // There was no server, so there is nothing to ask, and the database
        // still needs stopping. Without opening a session, because opening
        // one starts the database.
        Unserved::Stop { workspace, format } => {
            let result = command::stop::execute(&workspace).await?;
            format.emit(&result, || pamin_client::command::stop::render(&result));
            Ok(())
        }
    }
}
