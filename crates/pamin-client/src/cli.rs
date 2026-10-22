//! The `pamin` command line: what it takes, and running a command through the
//! workspace's server.
//!
//! The CLI is the primary surface: a shell command plus a skill file is a
//! zero-integration path for any agent that can run a process, with no client
//! library to adopt and no service to stand up. Everything a command does
//! besides parsing its arguments and printing its answer happens in the
//! server, so this is all a client needs -- and what it leaves to the binary
//! running it is only what [`Unserved`] names.

use anyhow::{Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use pamin_core::{VectorIndex, Workspace};

use crate::{client, command, output, protocol, settings};

#[derive(Parser)]
#[command(
    name = "pamin",
    version,
    about = "Universal memory for AI agents",
    long_about = None
)]
pub struct Cli {
    /// Where PaminMemory keeps its database, indexes, and models.
    #[arg(long, env = "PAMIN_HOME", global = true)]
    pub home: Option<std::path::PathBuf>,

    /// The memory namespace to operate on.
    #[arg(long, env = "PAMIN_PROJECT", global = true, default_value = "default")]
    pub project: String,

    /// Which vector index to build: disk or memory.
    ///
    /// `disk` holds almost nothing resident and is slower to query and much
    /// slower to build; `memory` is the fastest and the smallest on disk, and
    /// holds its graph and vectors resident. The index records which it was
    /// built with, so changing this requires `pamin reindex`.
    #[arg(
        long,
        env = "PAMIN_VECTOR_INDEX",
        global = true,
        default_value = "memory"
    )]
    pub vector_index: String,

    /// Emit machine-readable JSON instead of text.
    #[arg(long, global = true)]
    pub json: bool,

    /// Indent the JSON, for a person reading it rather than a parser.
    ///
    /// Off by default: the usual caller is an agent paying for every token of
    /// whitespace, and indenting a ten-hit search costs it about a thousand.
    #[arg(long, global = true, requires = "json")]
    pub pretty: bool,

    #[command(flatten)]
    pub server: settings::ServerArgs,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
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

    /// Show the server settings a command would run with, and where each
    /// came from: a flag, `PAMIN_DEVICE`, `config.toml`, or the default.
    Config,
}

/// What a command leaves to the binary running it, because doing it takes
/// what a client does not link: the database, the index and the models.
pub enum Unserved {
    /// `serve`: be the server, with these settings.
    Serve {
        workspace: Workspace,
        server: settings::Server,
    },
    /// `stop`, with no server running to answer it: the database may still
    /// be up, so stop it directly -- without opening a session, because
    /// opening one starts the database -- and print the result as `format`.
    Stop {
        workspace: Workspace,
        format: output::Format,
    },
}

/// Logging, the removed-settings check and the arguments, which neither needs
/// the runtime: `--help` and `--version` exit here without starting one.
pub fn parse() -> Result<(Cli, clap::ArgMatches)> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("PAMIN_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    // Before anything else, so a caller still setting it is told at once
    // rather than finding its commands going through the server it set this
    // to avoid.
    refuse_removed_settings()?;

    let matches = Cli::command().get_matches();
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    Ok((cli, matches))
}

/// Runs one command through the workspace's server, starting one if it needs
/// to, and prints the answer -- or says what is left for the binary to do.
pub async fn run(cli: Cli, matches: clap::ArgMatches) -> Result<Option<Unserved>> {
    let workspace = match &cli.home {
        Some(path) => Workspace::at(path),
        None => Workspace::discover()?,
    };
    let format = output::Format::from_flags(cli.json, cli.pretty);

    // Before the settings are resolved: `stop` runs with none of them, and
    // it is what a workspace whose `config.toml` no longer parses, or whose
    // PostgreSQL prefix is gone, still needs to get its server down.
    if let Command::Stop = cli.command {
        let request = protocol::Request {
            version: protocol::version(),
            project: cli.project.clone(),
            profile: String::new(),
            vector_index: cli.vector_index.clone(),
            server: Default::default(),
            call: protocol::Call::Stop,
        };
        if send(&workspace, &request, &[], format).await? {
            return Ok(None);
        }
        return Ok(Some(Unserved::Stop { workspace, format }));
    }

    // Resolved by every other command, server or not: a client asks the running
    // server for these, and a bad `config.toml` is refused here, before a
    // server is started that would refuse it in a log nobody reads.
    let server = settings::Server::resolve(workspace.root(), &cli.server, &matches)?;
    // Checked here although the server parses it again, so that a misspelled
    // vector index is refused before a server is started for it.
    VectorIndex::parse(&cli.vector_index)
        .ok_or_else(|| anyhow::anyhow!("unknown vector index {:?}", cli.vector_index))?;

    // `serve` is the server, so it never goes through one.
    if let Command::Serve = cli.command {
        return Ok(Some(Unserved::Serve { workspace, server }));
    }

    if let Command::Config = cli.command {
        let shown = settings::Shown::new(&workspace, &server);
        format.emit(&shown, || shown.render());
        return Ok(None);
    }

    // Taken before the command is moved into the call below.
    let project = cli.project.clone();

    let call = match cli.command {
        Command::Serve | Command::Config | Command::Stop => unreachable!("handled above"),
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
    };

    // Content on standard input is read here rather than server-side: the
    // server has no standard input, and `git log | pamin write` is in the
    // reference.
    let call = fill_from_stdin(call)?;

    let request = protocol::Request {
        version: protocol::version(),
        project,
        profile: String::new(),
        vector_index: cli.vector_index.clone(),
        server: server.asked(),
        call,
    };
    let serve = settings::forwarded(&cli.server);
    send(&workspace, &request, &serve, format).await?;
    Ok(None)
}

/// Sends `request` to the workspace's server, starting one with `serve` if
/// the call wants one, and prints the answer. `false` when there was no
/// server and none was started, which only `stop` asks for.
async fn send(
    workspace: &Workspace,
    request: &protocol::Request,
    serve: &[std::ffi::OsString],
    format: output::Format,
) -> Result<bool> {
    match client::ask(workspace, request, serve).await? {
        Some(value) => render(&request.call, &value, format).map(|()| true),
        None => Ok(false),
    }
}

/// Refuses to run with a setting that is no longer read.
///
/// Ignoring one would be silent in exactly the case it was set for. A shell
/// that still exports `PAMIN_INFERENCE_THREADS=2` would get a server running
/// with settings its operator believes it does not have; `PAMIN_RERANK=off`
/// set to buy back search time would pay for a reranker its setter believes
/// is off; `PAMIN_NO_SERVER`, set by somebody debugging the server, would
/// have them reasoning about a process that is not the one they asked for. A
/// warning would go to stderr, or to `serve.log`, where an agent never
/// looks. So each is refused, by name, with what replaced it -- whatever it
/// is set to, since none can be honoured and the fix is the same.
fn refuse_removed_settings() -> Result<()> {
    for (name, what) in REMOVED {
        if std::env::var_os(name).is_some() {
            anyhow::bail!("{name} {what} Unset it to continue.");
        }
    }
    Ok(())
}

/// The settings [`refuse_removed_settings`] turns away, and what became of
/// each. Since only `PAMIN_HOME`, `PAMIN_PROJECT`, `PAMIN_VECTOR_INDEX`,
/// `PAMIN_DEVICE` and `PAMIN_LOG` are read, every other setting that was a
/// variable is a flag, a `config.toml` key, or gone.
const REMOVED: &[(&str, &str)] = &[
    (
        "PAMIN_NO_SERVER",
        "was removed: every command but `serve`, `stop` and `config` now goes through \
         the workspace's server, and none can run in the calling process.",
    ),
    (
        "PAMIN_PROFILE",
        "was removed: every project uses the one embedding model, BGE-M3, so there is no \
         profile to choose.",
    ),
    (
        "PAMIN_RERANK",
        "was removed: `pamin search` always reranks with its one reranker, so there is no \
         tier to choose.",
    ),
    (
        "PAMIN_INFERENCE_THREADS",
        "was replaced by `--inference-threads`, or `inference_threads` in config.toml.",
    ),
    (
        "PAMIN_EP_LIBRARIES",
        "was replaced by `--ep-library` (once per library), or `ep_libraries` in \
         config.toml.",
    ),
    (
        "PAMIN_PREPARED",
        "was replaced by `--prepared <true|false>`, or `prepared` in config.toml.",
    ),
    (
        "PAMIN_FUSED_ATTENTION",
        "was replaced by `--fused-attention <true|false>`.",
    ),
    (
        "PAMIN_MODEL_IDLE",
        "was replaced by `--model-idle <SECONDS>`, or `model_idle` in config.toml.",
    ),
    (
        "PAMIN_OPEN_INDEXES",
        "was replaced by `--open-indexes`, or `open_indexes` in config.toml.",
    ),
    (
        "PAMIN_JIT",
        "was replaced by `--jit <true|false>`, or `jit` in config.toml.",
    ),
    (
        "PAMIN_POSTGRES_DIR",
        "was replaced by `--postgres-dir`, or `postgres_dir` in config.toml.",
    ),
    (
        "PAMIN_CHANNEL_DEPTH",
        "was replaced by `pamin search --channel-depth`.",
    ),
    (
        "PAMIN_GRAPH_DEPTH",
        "was replaced by `pamin search --graph-depth`.",
    ),
    (
        "PAMIN_RERANK_DEPTH",
        "was replaced by `pamin search --rerank-depth`.",
    ),
    (
        "PAMIN_SEARCH_EFFORT",
        "was replaced by `pamin search --search-effort`.",
    ),
    (
        "PAMIN_RERANK_BATCH",
        "was removed: the reranker's batch is fixed at its measured value.",
    ),
    (
        "PAMIN_RERANK_BATCH_TOKENS",
        "was removed: the reranker's batch is fixed at its measured value.",
    ),
    (
        "PAMIN_RERANK_MAX_TOKENS",
        "was removed: the reranker reads candidates to its measured length.",
    ),
    (
        "PAMIN_UNINDEXED_BUDGET",
        "was removed: the vector index is built at its measured threshold.",
    ),
    (
        "PAMIN_CATCH_UP_BATCH",
        "was removed: the server catches up in rounds of its measured size.",
    ),
];

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

    /// The variables the binary still reads: where its state is, which
    /// project and vector index, which device, and what to log.
    const READ: &[&str] = &[
        "PAMIN_HOME",
        "PAMIN_PROJECT",
        "PAMIN_VECTOR_INDEX",
        "PAMIN_DEVICE",
        "PAMIN_LOG",
    ];

    fn documented() -> String {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|crates| crates.parent())
            .expect("the workspace root is two levels above this crate");
        std::fs::read_to_string(root.join("docs/cli.md")).expect("read docs/cli.md")
    }

    /// Every variable the binary reads is one of [`READ`], and documented;
    /// every removed one is named in the docs, so a reader refused one can
    /// look up what replaced it, and none is read any more.
    ///
    /// The variables clap reads are taken from the parser, not from a list:
    /// a list would be the thing that goes stale. The binary's sources are
    /// scanned too, for a variable read some other way.
    #[test]
    fn every_variable_is_documented_and_every_removed_one_refused() {
        let documented = documented();

        let mut read = Vec::new();
        fn envs(command: &clap::Command, into: &mut Vec<String>) {
            for argument in command.get_arguments() {
                if let Some(name) = argument.get_env() {
                    into.push(name.to_string_lossy().into_owned());
                }
            }
            for subcommand in command.get_subcommands() {
                envs(subcommand, into);
            }
        }
        envs(&Cli::command(), &mut read);
        read.push("PAMIN_LOG".to_string());
        read.sort();
        read.dedup();
        let mut expected: Vec<String> = READ.iter().map(|name| name.to_string()).collect();
        expected.sort();
        assert_eq!(read, expected, "the variables the binary reads changed");

        for name in READ {
            assert!(
                documented.contains(name),
                "{name} is read but not in docs/cli.md"
            );
        }
        for (name, _) in REMOVED {
            assert!(
                documented.contains(name),
                "{name} is refused but docs/cli.md does not say what replaced it"
            );
            assert!(!READ.contains(name), "{name} is both read and refused");
        }

        // Both binaries' sources: the client's, and the server's, which
        // parses the same command line.
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates directory");
        let mut sources = Vec::new();
        for binary in ["pamin-client", "pamin-cli"] {
            rust_files(&crates.join(binary).join("src"), &mut sources);
        }
        for path in sources {
            let source = std::fs::read_to_string(&path).expect("read a source file");
            for (at, _) in source.match_indices("\"PAMIN_") {
                let rest = &source[at + 1..];
                let Some(end) = rest.find('"') else { continue };
                let name = &rest[..end];
                // A name, upper case and underscores, which rules out prose
                // and the bare prefix this search is written with.
                if name.len() == "PAMIN_".len()
                    || !name
                        .bytes()
                        .all(|byte| byte.is_ascii_uppercase() || byte == b'_')
                {
                    continue;
                }
                assert!(
                    READ.contains(&name) || REMOVED.iter().any(|(removed, _)| *removed == name),
                    "{} names {name}, which is neither read nor refused",
                    path.display()
                );
            }
        }
    }

    /// Every server setting is documented: each flag a reader can see, and
    /// each `config.toml` key.
    #[test]
    fn every_server_setting_is_documented() {
        let documented = documented();
        let command = Cli::command();
        let mut flags = 0;
        for argument in command.get_arguments() {
            if argument.get_help_heading() != Some("Server settings") || argument.is_hide_set() {
                continue;
            }
            let flag = format!("--{}", argument.get_long().expect("a long flag"));
            assert!(documented.contains(&flag), "{flag} is not in docs/cli.md");
            flags += 1;
        }
        assert!(flags >= 8, "only {flags} server flags found");

        let root = tempfile::tempdir().unwrap();
        let matches = Cli::command()
            .try_get_matches_from(["pamin", "config"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();
        let server = settings::Server::resolve(root.path(), &cli.server, &matches).unwrap();
        for entry in &server.entries {
            // Hidden, like their flags: measurement switches, not settings.
            if matches!(entry.key, "fused_attention" | "split_embeddings") {
                continue;
            }
            let key = format!("`{}`", entry.key);
            assert!(
                documented.contains(&key),
                "config key {key} is not in docs/cli.md"
            );
        }
    }

    /// The libraries read no `PAMIN_*` variable: the binary turns its flags,
    /// its `config.toml` and the few variables it still reads into the values
    /// the libraries take, so what a server runs with is what it was started
    /// with, and a test varies a setting by building a value rather than
    /// writing the environment every other thread shares.
    ///
    /// Any `"PAMIN_` literal in a library's product code -- outside comments,
    /// outside its test module, and other than a compile-time `env!` stamp --
    /// fails, which also catches a name read through a constant.
    #[test]
    fn the_libraries_read_no_pamin_variable() {
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("the crates directory");
        let mut found = Vec::new();
        let mut files = 0;
        for library in ["pamin-core", "pamin-index", "pamin-store", "pamin-engine"] {
            let mut sources = Vec::new();
            rust_files(&crates.join(library).join("src"), &mut sources);
            for path in sources {
                files += 1;
                let source = std::fs::read_to_string(&path).expect("read a source file");
                // Up to the test module, however it is declared.
                let product = source
                    .match_indices("#[cfg(test)]\n")
                    .find(|(at, marker)| {
                        source[at + marker.len()..]
                            .lines()
                            .next()
                            .is_some_and(|line| line.contains("mod tests"))
                    })
                    .map_or(source.as_str(), |(at, _)| &source[..at]);
                for (number, line) in product.lines().enumerate() {
                    let code = line.trim_start();
                    if code.starts_with("//") {
                        continue;
                    }
                    for (at, _) in code.match_indices("\"PAMIN_") {
                        if !code[..at].ends_with("env!(") {
                            found.push(format!("{}:{}: {code}", path.display(), number + 1));
                        }
                    }
                }
            }
        }
        assert!(
            files > 20,
            "only read {files} files, so this is not reading the source"
        );
        assert!(
            found.is_empty(),
            "a library names a PAMIN_ variable: {found:#?}"
        );
    }

    fn rust_files(dir: &std::path::Path, into: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read a source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                rust_files(&path, into);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                into.push(path);
            }
        }
    }

    /// A server flag after the subcommand counts as given, so it is both
    /// forwarded to the server a command starts and asked of a running one.
    #[test]
    fn a_server_flag_after_the_subcommand_is_a_flag() {
        let matches = Cli::command()
            .try_get_matches_from(["pamin", "search", "q", "--model-idle", "5"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();
        let root = tempfile::tempdir().unwrap();
        let server = settings::Server::resolve(root.path(), &cli.server, &matches).unwrap();
        let asked = server.asked();
        assert_eq!(asked["model_idle"].value, "5");
        assert!(asked["model_idle"].chosen);
        assert_eq!(
            settings::forwarded(&cli.server),
            ["--model-idle", "5"].map(std::ffi::OsString::from)
        );
    }

    /// `stop` gets a workspace's server down without its settings, so a
    /// `config.toml` that no longer parses does not keep it running: with no
    /// server to ask, it is left to the binary to stop the database.
    #[tokio::test]
    async fn stop_needs_no_server_settings() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join(settings::CONFIG), "[serve\n").unwrap();
        let home = root.path().to_str().unwrap();
        let matches = Cli::command()
            .try_get_matches_from(["pamin", "--home", home, "stop"])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();
        assert!(settings::Server::resolve(root.path(), &cli.server, &matches).is_err());

        let unserved = run(cli, matches).await.unwrap();
        assert!(
            matches!(&unserved, Some(Unserved::Stop { workspace, .. }) if workspace.root() == root.path()),
            "stop with no server is left to the binary"
        );
    }

    /// A search depth of zero is refused, except the graph's, which turns the
    /// walk off.
    #[test]
    fn a_zero_search_depth_is_refused() {
        let parse = |flag: &str, value: &str| {
            Cli::command().try_get_matches_from(["pamin", "search", "q", flag, value])
        };
        for flag in ["--channel-depth", "--rerank-depth", "--search-effort"] {
            assert!(parse(flag, "0").is_err(), "{flag} 0 was accepted");
            assert!(parse(flag, "1").is_ok(), "{flag} 1 was refused");
        }
        assert!(parse("--graph-depth", "0").is_ok());
    }

    /// The vector index a command gets when nobody names one.
    ///
    /// Two defaults, and nothing else makes them agree: `clap` parses a string
    /// and the library has its own `Default`. A change to one and not the
    /// other is two defaults, both shipping.
    #[test]
    fn the_documented_default_vector_index_is_the_library_default() {
        // Read off the declared argument rather than by parsing a command
        // line, because parsing consults `PAMIN_VECTOR_INDEX` and would then
        // be measuring whatever the environment running the test says.
        let command = <Cli as clap::CommandFactory>::command();
        let declared = command
            .get_arguments()
            .find(|argument| argument.get_id() == "vector_index")
            .and_then(|argument| argument.get_default_values().first().cloned())
            .expect("the vector index argument declares a default");

        assert_eq!(
            VectorIndex::parse(&declared.to_string_lossy()),
            Some(VectorIndex::default())
        );
    }
}
