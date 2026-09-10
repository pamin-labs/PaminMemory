//! The `pamin` command line interface.
//!
//! The CLI is the primary surface: a shell command plus a skill file is a
//! zero-integration path for any agent that can run a process, with no client
//! library to adopt and no service to stand up.

mod command;
mod output;

use anyhow::Result;
use clap::{Parser, Subcommand};
use pamin_index::Profile;
use pamin_store::Workspace;

#[derive(Parser)]
#[command(
    name = "pamin",
    version,
    about = "Universal memory for AI agents",
    long_about = None
)]
struct Cli {
    /// Where PaminMemory keeps its database, indexes, and models.
    #[arg(long, env = "PAMIN_HOME", global = true)]
    home: Option<std::path::PathBuf>,

    /// The memory namespace to operate on.
    #[arg(long, env = "PAMIN_PROJECT", global = true, default_value = "default")]
    project: String,

    /// Which embedding profile to use: speed, balanced, or accuracy.
    ///
    /// The index records the profile it was built with, so changing this
    /// requires `pamin reindex` rather than silently mixing vector spaces.
    #[arg(long, env = "PAMIN_PROFILE", global = true, default_value = "balanced")]
    profile: String,

    /// Emit machine-readable JSON instead of text.
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Provision the local database and apply migrations.
    Init,

    /// Record a memory.
    Write(command::write::Args),

    /// Read a topic's current or historical state.
    Read(command::read::Args),

    /// Search memories across every recall channel.
    Search(command::search::Args),

    /// Find an exact string in the evidence, with nothing ranking.
    Grep(command::grep::Args),

    /// Assert a relationship between two topics.
    Link(command::link::Args),

    /// Retract a relationship, leaving the record that it was claimed.
    Unlink(command::unlink::Args),

    /// List the topics connected to one, without ranking them.
    Neighbors(command::neighbors::Args),

    /// Rebuild the projection index from PostgreSQL.
    Reindex(command::reindex::Args),

    /// Run and inspect the work a write left for the projection.
    Cascade(command::cascade::Args),

    /// Stop the local database server.
    Stop,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("PAMIN_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let workspace = match &cli.home {
        Some(path) => Workspace::at(path),
        None => Workspace::discover()?,
    };
    let format = output::Format::from_json_flag(cli.json);
    let profile = Profile::parse(&cli.profile)
        .ok_or_else(|| anyhow::anyhow!("unknown profile {:?}", cli.profile))?;

    match cli.command {
        Command::Init => {
            let result = command::init::execute(&workspace, &cli.project).await?;
            format.emit(&result, || command::init::render(&result));
            Ok(())
        }
        Command::Write(args) => {
            let result = command::write::execute(&workspace, &cli.project, profile, args).await?;
            format.emit(&result, || command::write::render(&result));
            Ok(())
        }
        Command::Read(args) => {
            let result = command::read::execute(&workspace, &cli.project, args).await?;
            format.emit(&result, || command::read::render(&result));
            Ok(())
        }
        Command::Search(args) => {
            let results = command::search::execute(&workspace, &cli.project, profile, args).await?;
            format.emit(&results, || command::search::render(&results));
            Ok(())
        }
        Command::Grep(args) => {
            let result = command::grep::execute(&workspace, &cli.project, args).await?;
            format.emit(&result, || command::grep::render(&result));
            Ok(())
        }
        Command::Link(args) => {
            let result = command::link::execute(&workspace, &cli.project, args).await?;
            format.emit(&result, || command::link::render(&result));
            Ok(())
        }
        Command::Unlink(args) => {
            let result = command::unlink::execute(&workspace, &cli.project, args).await?;
            format.emit(&result, || command::unlink::render(&result));
            Ok(())
        }
        Command::Neighbors(args) => {
            let result = command::neighbors::execute(&workspace, &cli.project, args).await?;
            format.emit(&result, || command::neighbors::render(&result));
            Ok(())
        }
        Command::Reindex(args) => {
            let result = command::reindex::execute(&workspace, &cli.project, profile, args).await?;
            format.emit(&result, || command::reindex::render(&result));
            Ok(())
        }
        Command::Cascade(args) => {
            command::cascade::execute(&workspace, &cli.project, profile, format, args).await
        }
        Command::Stop => {
            let result = command::stop::execute(&workspace).await?;
            format.emit(&result, || command::stop::render(&result));
            Ok(())
        }
    }
}
