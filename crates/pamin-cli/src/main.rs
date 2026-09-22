//! The `pamin` command line interface.
//!
//! The CLI is the primary surface: a shell command plus a skill file is a
//! zero-integration path for any agent that can run a process, with no client
//! library to adopt and no service to stand up.

mod client;
mod command;
mod output;
mod protocol;
mod registry;
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

    /// Indent the JSON, for a person reading it rather than a parser.
    ///
    /// Off by default: the usual caller is an agent paying for every token of
    /// whitespace, and indenting a ten-hit search costs it about a thousand.
    #[arg(long, global = true, requires = "json")]
    pretty: bool,

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

    /// Find the topics already here, by what they are called or what they hold.
    Topics(command::topics::Args),

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

    // Before anything opens an index, and for every command rather than for
    // `serve` alone -- which is where this used to be, and the reason it moved.
    // `PAMIN_NO_SERVER` runs the whole of a command in this process, and a
    // command holding the descriptors a large index needs under the 1,024 a
    // Linux process starts with does not degrade, it fails.
    match pamin_index::raise_open_file_limit() {
        Ok((before, after)) if after > before => {
            tracing::info!(before, after, "raised the open-file limit")
        }
        Ok(_) => {}
        Err(error) => tracing::warn!(%error, "could not raise the open-file limit"),
    }

    let cli = Cli::parse();
    let workspace = match &cli.home {
        Some(path) => Workspace::at(path),
        None => Workspace::discover()?,
    };
    let format = output::Format::from_flags(cli.json, cli.pretty);
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
        Command::Topics(args) => protocol::Call::Topics(args),
        Command::Reindex(args) => protocol::Call::Reindex(args),
        Command::Cascade(args) => protocol::Call::Cascade(args),
        Command::Stop => protocol::Call::Stop,
    };

    // Content on standard input is read here rather than server-side: the
    // server has no standard input, and `git log | pamin write` is in the
    // reference.
    let call = fill_from_stdin(call)?;

    // Before a server is started or a database provisioned, for two different
    // reasons. A misspelled tier is an error the caller can act on at once,
    // and one that arrived after a PostgreSQL install -- leaving a workspace
    // behind for a command that never ran -- would be a worse answer to the
    // same question. And a licence notice is worth printing only where a
    // person is reading, which is here and not in a resident server whose
    // stderr is a log file.
    if let protocol::Call::Search(args) = &call {
        let tier = pamin_index::Rerank::parse(&args.rerank)
            .ok_or_else(|| anyhow::anyhow!("unknown rerank tier {:?}", args.rerank))?;
        if let Some(warning) = command::search::caution(tier) {
            eprintln!("{warning}");
        }
    }

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
        protocol::Call::Topics(args) => {
            let result = command::topics::execute(&session, project, profile, args).await?;
            format.emit(&result, || command::topics::render(&result));
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
fn render(
    call: &protocol::Call,
    value: &serde_json::value::RawValue,
    format: output::Format,
) -> Result<()> {
    fn parse<T: serde::de::DeserializeOwned>(
        value: &serde_json::value::RawValue,
        what: &str,
    ) -> Result<T> {
        serde_json::from_str(value.get())
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
        protocol::Call::Topics(_) => {
            let result: command::topics::Topics = parse(value, "topics")?;
            format.emit(&result, || command::topics::render(&result));
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

    /// Variables the product reads and `docs/cli.md` deliberately omits.
    ///
    /// Each one shortens a window or a budget so that a test can reach a case
    /// in seconds that the shipped value reaches in minutes or at twenty-five
    /// thousand documents. A caller has no way to evaluate them, and a
    /// documented knob is a knob somebody will turn -- so they stay out of the
    /// table and in this list, where leaving one out is a failing test rather
    /// than a silent omission.
    const UNDOCUMENTED: &[&str] = &[
        "PAMIN_EVAL_HOME",
        "PAMIN_RERANK_BATCH",
        "PAMIN_RERANK_DEPTH",
        "PAMIN_RERANK_MAX_TOKENS",
        "PAMIN_UNINDEXED_BUDGET",
        "PAMIN_VECTOR_STORAGE",
    ];

    /// Every setting the product reads is documented, or listed as not.
    ///
    /// This repository has written a setting and not documented it more than
    /// once, and the reader cannot tell an omission from a decision. The
    /// source is the authority here: whatever `PAMIN_*` the crates read has to
    /// appear in the options table or in [`UNDOCUMENTED`], and adding one
    /// without doing either fails.
    ///
    /// Reads the tree rather than a list, because a list would be the thing
    /// that goes stale. Names are taken from string literals, which is how all
    /// of them are written -- a variable assembled at runtime would slip past
    /// this, and nothing here does that.
    #[test]
    fn every_setting_is_documented_or_deliberately_not() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|crates| crates.parent())
            .expect("the workspace root is two levels above this crate");
        let documented =
            std::fs::read_to_string(root.join("docs/cli.md")).expect("read docs/cli.md");

        let mut found: Vec<String> = Vec::new();
        collect_settings(&root.join("crates"), &mut found);
        found.sort();
        found.dedup();
        assert!(
            found.len() > 10,
            "only found {} settings, so this test is not reading the source",
            found.len()
        );

        let missing: Vec<&String> = found
            .iter()
            .filter(|name| !UNDOCUMENTED.contains(&name.as_str()))
            .filter(|name| !documented.contains(name.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "read by the product and in neither docs/cli.md nor UNDOCUMENTED: {missing:?}"
        );

        let gone: Vec<&&str> = UNDOCUMENTED
            .iter()
            .filter(|name| !found.contains(&(*name).to_string()))
            .collect();
        assert!(
            gone.is_empty(),
            "listed as deliberately undocumented but nothing reads them any more: {gone:?}"
        );
    }

    /// Every `"PAMIN_..."` literal under a directory, recursively.
    fn collect_settings(dir: &std::path::Path, into: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                // `tests/` holds evaluation harnesses, which are not the
                // product and document their own variables in their own module
                // documentation.
                if path.file_name().is_some_and(|name| name == "tests") {
                    continue;
                }
                collect_settings(&path, into);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let Ok(source) = std::fs::read_to_string(&path) else {
                    continue;
                };
                for (before, _) in source.match_indices("\"PAMIN_") {
                    let rest = &source[before + 1..];
                    let Some(end) = rest.find('"') else { continue };
                    let name = &rest[..end];
                    // Upper case and underscores only, which rules out the
                    // `"PAMIN_..."` this very comment would otherwise
                    // contribute -- a scanner that finds its own prose is a
                    // scanner that fails for the wrong reason.
                    if name
                        .bytes()
                        .all(|byte| byte.is_ascii_uppercase() || byte == b'_')
                    {
                        into.push(name.to_string());
                    }
                }
            }
        }
    }

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
