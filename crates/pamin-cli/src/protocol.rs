//! What a thin client and a resident server say to each other.
//!
//! Line-delimited JSON over a Unix domain socket. Nothing here is a new
//! interface: the request carries the same `clap::Args` the CLI already parses,
//! and the response carries the same struct each command already serialized for
//! `--json`. That is deliberate, and it is why there is no schema to keep in
//! step with anything -- `docs/cli.md` calls the JSON output the contract, so
//! the wire format is that contract and a command name.
//!
//! One command is missing from [`Call`] on purpose: `serve` is the server, so it
//! cannot be a request to one. `stop` is a request -- the server answers it and
//! then exits -- but it is the one call a client never starts a server to make.

use serde::{Deserialize, Serialize};

use crate::command;

/// The socket, relative to the workspace root.
///
/// Beside `server.json`, which records the database the same workspace started,
/// because they are the same kind of thing: a durable process that outlives the
/// command that needed it, and the file that says how to reach it.
pub const SOCKET: &str = "pamin.sock";

/// Where a server started in the background writes what it has to say.
///
/// A server outlives the command that started it, so it cannot inherit that
/// command's streams: holding the write end of a pipe open for ever turns
/// `pamin write ... | head` into a hang after the write has already succeeded.
/// A file solves that and is also the only way to see what a background server
/// is doing.
pub const SERVER_LOG: &str = "serve.log";

/// Held while a client starts the server, so that twenty commands racing at a
/// cold workspace start one server rather than twenty.
///
/// Created with `O_EXCL` rather than an advisory lock in the database: the
/// database may be the thing that is not up yet, and `migrate.rs` rules out
/// `pg_advisory_lock` for portability anyway.
pub const SPAWN_LOCK: &str = "pamin.spawning";

/// One request, and everything the server needs to serve it.
#[derive(Serialize, Deserialize)]
pub struct Request {
    /// The client binary's version.
    ///
    /// A server from another build answers with the same field names and
    /// different meanings, which is the one way this can be silently wrong
    /// rather than loudly broken. The client kills a mismatched server and
    /// starts its own.
    pub version: String,
    /// The namespace to operate in. Per request rather than per connection,
    /// because the server holds many and a client speaks for one at a time.
    pub project: String,
    /// The embedding profile, by name. Parsed server-side so an unknown one
    /// fails the same way it does in-process.
    pub profile: String,
    pub call: Call,
}

/// A command and its arguments, as parsed.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Call {
    Init,
    Write(command::write::Args),
    Import(command::import::Args),
    Read(command::read::Args),
    Search(command::search::Args),
    Grep(command::grep::Args),
    Link(command::link::Args),
    Unlink(command::unlink::Args),
    Neighbors(command::neighbors::Args),
    Reindex(command::reindex::Args),
    Cascade(command::cascade::Args),
    /// Stop the database, and the server answering this. The one call a client
    /// will not start a server in order to make.
    Stop,
}

impl Call {
    /// What to call this in a log line or an error.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Write(_) => "write",
            Self::Import(_) => "import",
            Self::Read(_) => "read",
            Self::Search(_) => "search",
            Self::Grep(_) => "grep",
            Self::Link(_) => "link",
            Self::Unlink(_) => "unlink",
            Self::Neighbors(_) => "neighbors",
            Self::Reindex(_) => "reindex",
            Self::Cascade(_) => "cascade",
            Self::Stop => "stop",
        }
    }
}

/// What came back.
///
/// A failure is a message rather than a structured error because that is
/// exactly what the in-process path produces: `anyhow` context, rendered once,
/// printed to stderr. Reconstructing an error type across the socket would give
/// the client something it has never had and does not use.
/// The success payload is carried as the bytes it already is.
///
/// A command's result is serialized once, by whoever produced it, and copied
/// from here into the socket. Holding a `serde_json::Value` instead meant
/// building a tree, walking it again to write it out, and -- on the other end
/// -- cloning the whole tree before reading a type out of it. None of those
/// three passes decided anything.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    Ok(Box<serde_json::value::RawValue>),
    Err(String),
    /// The server is from a different build. Its own identity comes back so the
    /// client can say what it replaced, and the client kills it and starts one
    /// of its own rather than trusting field names that may have moved.
    Mismatch {
        server: String,
    },
}

/// This binary's identity, for the version handshake.
///
/// The crate version is not enough. It stays `0.0.1` across every change, so
/// during development -- which is exactly when a stale server is left listening
/// after a rebuild -- two binaries that share nothing would agree. The
/// executable's size and modification time distinguish them, and a rebuild
/// always moves at least one.
pub fn version() -> String {
    let crate_version = env!("CARGO_PKG_VERSION");

    let built = std::env::current_exe()
        .and_then(|path| path.metadata())
        .ok()
        .map(|meta| {
            let stamp = meta
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |since| since.as_secs());
            format!("{}-{stamp}", meta.len())
        });

    match built {
        Some(built) => format!("{crate_version}+{built}"),
        // Nothing to distinguish builds with, which is worse than the version
        // above but never wrong about the version itself.
        None => crate_version.to_string(),
    }
}
