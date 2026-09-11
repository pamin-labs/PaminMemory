//! The `pamin` command line interface.
//!
//! The CLI is the primary surface: a shell command plus a skill file is a
//! zero-integration path for any agent that can run a process, with no client
//! library to adopt and no service to stand up.

mod client;
mod command;
mod output;
mod protocol;
mod server;
mod session;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pamin_index::Profile;
use pamin_store::{Connections, Workspace};

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
    #[arg(long, env = "PAMIN_PROFILE", global = true, default_value = "accuracy")]
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

    /// Record many memories from a file, in one call.
    Import(command::import::Args),

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

    /// Stop the local database server, and the resident server if one is up.
    Stop,

    /// Hold the database, the index and the model, and answer commands.
    ///
    /// Started automatically by any command that needs one, so this is for
    /// running it in the foreground and watching it.
    Serve,
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

    // `serve` is the server, so it never goes through one.
    if let Command::Serve = cli.command {
        return server::run(&workspace).await;
    }

    // Taken before the command is moved into the call below.
    let project = cli.project.clone();

    let call = match cli.command {
        Command::Serve => unreachable!("handled above"),
        Command::Init => protocol::Call::Init,
        Command::Write(args) => protocol::Call::Write(args),
        Command::Import(args) => protocol::Call::Import(args),
        Command::Read(args) => protocol::Call::Read(args),
        Command::Search(args) => protocol::Call::Search(args),
        Command::Grep(args) => protocol::Call::Grep(args),
        Command::Link(args) => protocol::Call::Link(args),
        Command::Unlink(args) => protocol::Call::Unlink(args),
        Command::Neighbors(args) => protocol::Call::Neighbors(args),
        Command::Reindex(args) => protocol::Call::Reindex(args),
        Command::Cascade(args) => protocol::Call::Cascade(args),
        Command::Stop => protocol::Call::Stop,
    };

    // Content on standard input is read here rather than server-side: the
    // server has no standard input, and `git log | pamin write` is in the
    // reference.
    let call = fill_from_stdin(call)?;

    if client::wanted() {
        let request = protocol::Request {
            version: protocol::version(),
            project: project.clone(),
            profile: cli.profile.clone(),
            call,
        };
        if let Some(value) = client::ask(&workspace, &request).await? {
            return render(&request.call, &value, format);
        }
        // Only `stop` reaches here: there was no server, so there is nothing to
        // ask, and the database still needs stopping.
        return run_here(&workspace, &project, profile, request.call, format).await;
    }

    run_here(&workspace, &project, profile, call, format).await
}

/// Reads standard input into the one argument that takes it.
fn fill_from_stdin(call: protocol::Call) -> Result<protocol::Call> {
    let protocol::Call::Write(mut args) = call else {
        return Ok(call);
    };

    if args.content.is_none() {
        args.content =
            Some(std::io::read_to_string(std::io::stdin()).context("reading content from stdin")?);
    }
    Ok(protocol::Call::Write(args))
}

/// Runs a call in this process, which is what happens without a server.
async fn run_here(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    call: protocol::Call,
    format: output::Format,
) -> Result<()> {
    // Before the session, because opening one starts the database, and this is
    // the command that stops it. It is also the one call that reaches here
    // after a client has already looked for a server and found none.
    if let protocol::Call::Stop = call {
        let result = command::stop::execute(workspace).await?;
        format.emit(&result, || command::stop::render(&result));
        return Ok(());
    }

    let session = session::Session::open(workspace, Connections::PerCommand).await?;

    match call {
        protocol::Call::Stop => unreachable!("handled above"),
        protocol::Call::Init => {
            let result = command::init::execute(&session, project).await?;
            format.emit(&result, || command::init::render(&result));
        }
        protocol::Call::Write(args) => {
            let result = command::write::execute(&session, project, profile, args).await?;
            format.emit(&result, || command::write::render(&result));
        }
        protocol::Call::Import(args) => {
            let result = command::import::execute(&session, project, profile, args).await?;
            format.emit(&result, || command::import::render(&result));
        }
        protocol::Call::Read(args) => {
            let result = command::read::execute(&session, project, args).await?;
            format.emit(&result, || command::read::render(&result));
        }
        protocol::Call::Search(args) => {
            let results = command::search::execute(&session, project, profile, args).await?;
            format.emit(&results, || command::search::render(&results));
        }
        protocol::Call::Grep(args) => {
            let result = command::grep::execute(&session, project, args).await?;
            format.emit(&result, || command::grep::render(&result));
        }
        protocol::Call::Link(args) => {
            let result = command::link::execute(&session, project, args).await?;
            format.emit(&result, || command::link::render(&result));
        }
        protocol::Call::Unlink(args) => {
            let result = command::unlink::execute(&session, project, args).await?;
            format.emit(&result, || command::unlink::render(&result));
        }
        protocol::Call::Neighbors(args) => {
            let result = command::neighbors::execute(&session, project, args).await?;
            format.emit(&result, || command::neighbors::render(&result));
        }
        protocol::Call::Reindex(args) => {
            let result = command::reindex::execute(&session, project, profile, args).await?;
            format.emit(&result, || command::reindex::render(&result));
        }
        protocol::Call::Cascade(args) => {
            command::cascade::execute(&session, project, profile, format, args).await?;
        }
    }

    Ok(())
}

/// Prints what the server sent, in whichever form was asked for.
///
/// The JSON goes out as it arrived. The text form needs the value back as the
/// type it was, because rendering is the client's half of the split and each
/// command's `render` takes its own struct. Naming each type here rather than
/// hiding it behind a macro keeps the compiler checking that the type the
/// server serialized is the type the client renders.
fn render(call: &protocol::Call, value: &serde_json::Value, format: output::Format) -> Result<()> {
    fn parse<T: serde::de::DeserializeOwned>(value: &serde_json::Value, what: &str) -> Result<T> {
        serde_json::from_value(value.clone())
            .with_context(|| format!("reading the {what} the server sent"))
    }

    match call {
        protocol::Call::Init => {
            let result: command::init::Initialized = parse(value, "init")?;
            format.emit(&result, || command::init::render(&result));
        }
        protocol::Call::Import(_) => {
            let result: command::import::Imported = parse(value, "import")?;
            format.emit(&result, || command::import::render(&result));
        }
        protocol::Call::Write(_) => {
            let result: command::write::Written = parse(value, "write")?;
            format.emit(&result, || command::write::render(&result));
        }
        protocol::Call::Read(_) => {
            let result: command::read::Read = parse(value, "read")?;
            format.emit(&result, || command::read::render(&result));
        }
        protocol::Call::Search(_) => {
            let results: command::search::Results = parse(value, "search")?;
            format.emit(&results, || command::search::render(&results));
        }
        protocol::Call::Grep(_) => {
            let result: command::grep::Matches = parse(value, "grep")?;
            format.emit(&result, || command::grep::render(&result));
        }
        protocol::Call::Link(_) => {
            let result: command::link::Linked = parse(value, "link")?;
            format.emit(&result, || command::link::render(&result));
        }
        protocol::Call::Unlink(_) => {
            let result: command::unlink::Unlinked = parse(value, "unlink")?;
            format.emit(&result, || command::unlink::render(&result));
        }
        protocol::Call::Neighbors(_) => {
            let result: command::neighbors::Neighborhood = parse(value, "neighbors")?;
            format.emit(&result, || command::neighbors::render(&result));
        }
        protocol::Call::Reindex(_) => {
            let result: command::reindex::Reindexed = parse(value, "reindex")?;
            format.emit(&result, || command::reindex::render(&result));
        }
        protocol::Call::Stop => {
            let result: command::stop::Stopped = parse(value, "stop")?;
            format.emit(&result, || command::stop::render(&result));
        }
        // The subcommands answer with different types, so which one to parse
        // into is a question about the request rather than the response.
        protocol::Call::Cascade(args) => command::cascade::render_value(args, value, format)?,
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The profile a command gets when nobody names one.
    ///
    /// Two defaults, and nothing else makes them agree: `clap` parses a string
    /// and the library has its own `Default`. They are used in different
    /// places -- the string here, the impl by anything constructing a profile
    /// directly -- so a change to one and not the other is two defaults, both
    /// shipping, differing in which vector space a project ends up in.
    #[test]
    fn the_documented_default_profile_is_the_library_default() {
        // Read off the declared argument rather than by parsing a command
        // line, because parsing consults `PAMIN_PROFILE` and would then be
        // measuring whatever the environment running the test happens to say.
        let command = <Cli as clap::CommandFactory>::command();
        let declared = command
            .get_arguments()
            .find(|argument| argument.get_id() == "profile")
            .and_then(|argument| argument.get_default_values().first().cloned())
            .expect("the profile argument declares a default");

        assert_eq!(
            Profile::parse(&declared.to_string_lossy()),
            Some(Profile::default())
        );
    }
}
