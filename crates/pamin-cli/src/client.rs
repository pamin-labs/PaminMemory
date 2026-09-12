//! Reaching the resident server, and starting one when there is none.
//!
//! The CLI is the primary surface and a shell command is meant to just work, so
//! nothing here asks the caller to run a server first. A command connects; if
//! there is nothing listening it starts one and connects again. That is the
//! same bargain the embedded database already makes -- `pamin init` leaves
//! PostgreSQL running so the next command does not pay for it -- applied to the
//! index and the model as well.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use futures::{SinkExt, StreamExt};
use pamin_store::Workspace;
use tokio::net::UnixStream;
use tokio_util::codec::{Framed, LinesCodec};

use crate::protocol::{Call, Request, Response, SERVER_LOG, SPAWN_LOCK, version};
use crate::server::socket_path;

/// How long to wait for a server we started to begin listening.
///
/// It has to open the database before it accepts, and a cold workspace runs
/// `initdb` first, which is why this is seconds rather than milliseconds.
const STARTUP: Duration = Duration::from_secs(90);

/// How long to wait between attempts to connect to a server that is starting.
const RETRY: Duration = Duration::from_millis(50);

/// Whether this process should talk to a server at all.
///
/// `PAMIN_NO_SERVER=1` runs everything in this process instead. It exists for
/// tests that want a single process to reason about, and for the case where the
/// server is the thing being debugged.
pub fn wanted() -> bool {
    !matches!(
        std::env::var("PAMIN_NO_SERVER").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Sends one request, starting a server if there is none.
///
/// `Ok(None)` means there was no server and none was wanted -- only `stop` asks
/// for that, since starting a server to stop it is a way of doing nothing
/// slowly. Every other call gets one started.
pub async fn ask(
    workspace: &Workspace,
    request: &Request,
) -> Result<Option<Box<serde_json::value::RawValue>>> {
    let path = socket_path(workspace);
    let start_one = !matches!(request.call, Call::Stop);

    let mut stream = match UnixStream::connect(&path).await {
        Ok(stream) => stream,
        Err(_) if !start_one => return Ok(None),
        Err(_) => {
            spawn(workspace, &path).await?;
            UnixStream::connect(&path)
                .await
                .with_context(|| format!("connecting to the server at {}", path.display()))?
        }
    };

    match exchange(&mut stream, request).await? {
        Response::Ok(value) => Ok(Some(value)),
        Response::Err(message) => bail!(message),
        Response::Mismatch { server } => {
            // A server from another build answers with the same field names and
            // possibly different meanings. Replace it rather than reason about
            // which fields still line up.
            tracing::info!(%server, ours = %version(), "replacing a server from another build");
            stop_and_wait(&path).await?;
            spawn(workspace, &path).await?;

            let mut fresh = UnixStream::connect(&path)
                .await
                .with_context(|| format!("connecting to the server at {}", path.display()))?;
            match exchange(&mut fresh, request).await? {
                Response::Ok(value) => Ok(Some(value)),
                Response::Err(message) => bail!(message),
                Response::Mismatch { server } => bail!(
                    "the server at {} reports build {server}, and this one is {}; \
                     a replacement reported the same, so something else is starting it",
                    path.display(),
                    version()
                ),
            }
        }
    }
}

/// One request, one response, on a connection this owns.
async fn exchange(stream: &mut UnixStream, request: &Request) -> Result<Response> {
    let mut framed = Framed::new(stream, LinesCodec::new_with_max_length(MAX_RESPONSE));

    framed
        .send(serde_json::to_string(request)?)
        .await
        .context("sending the request")?;

    let line = framed
        .next()
        .await
        .transpose()
        .context("reading the response")?
        .context("the server closed without answering")?;

    serde_json::from_str(&line).context("the server sent something unreadable")
}

/// The largest response line the client will read.
///
/// A search carries every hit's content, so the bound is on a result set rather
/// than an envelope.
const MAX_RESPONSE: usize = 16 * 1024 * 1024;

/// Starts a server and waits for it to listen.
///
/// Only one process starts one. The rest wait, which is the same outcome and
/// costs nothing: twenty agents running `pamin search` against a cold workspace
/// should provision it once, not twenty times.
async fn spawn(workspace: &Workspace, path: &Path) -> Result<()> {
    std::fs::create_dir_all(workspace.root())?;
    let lock = workspace.root().join(SPAWN_LOCK);

    // `create_new` is `O_EXCL`: exactly one caller wins, and the losers fall
    // through to waiting. Not an advisory lock in the database, because the
    // database is one of the things that may not be up yet.
    let won = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock);

    if won.is_ok() {
        // Both streams go to a file rather than being inherited. A server
        // outlives the command that started it, so an inherited stderr is a
        // pipe it holds open for ever: `pamin write ... | head` would hang
        // after the write succeeded, waiting on an end-of-file the server was
        // still holding. Redirecting also gives it somewhere to speak, which
        // matters more for a process nobody is watching than for one in the
        // foreground.
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(workspace.root().join(SERVER_LOG))
            .context("opening the server log")?;

        let started = std::process::Command::new(std::env::current_exe()?)
            .arg("--home")
            .arg(workspace.root())
            .arg("serve")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::from(log.try_clone()?))
            .stderr(std::process::Stdio::from(log))
            .spawn();

        if let Err(error) = started {
            // Release the lock, or every later command in this workspace waits
            // out the timeout for a server nobody is starting.
            let _ = std::fs::remove_file(&lock);
            return Err(error).context("starting the server");
        }
    }

    let deadline = Instant::now() + STARTUP;
    let listening = loop {
        if UnixStream::connect(path).await.is_ok() {
            break true;
        }
        if Instant::now() >= deadline {
            break false;
        }
        tokio::time::sleep(RETRY).await;
    };

    if won.is_ok() {
        let _ = std::fs::remove_file(&lock);
    }

    if !listening {
        bail!(
            "waited {}s for a server to listen at {}",
            STARTUP.as_secs(),
            path.display()
        );
    }

    Ok(())
}

/// Asks a server to stop, and waits for its socket to go.
async fn stop_and_wait(path: &Path) -> Result<()> {
    let farewell = Request {
        version: String::new(),
        project: String::new(),
        profile: String::new(),
        call: Call::Stop,
    };

    // The version is deliberately empty, and the server does not look at it for
    // `stop`: stopping means the same thing in every build, and a server that
    // refused it on a mismatch would be one nothing could ever replace.
    if let Ok(mut stream) = UnixStream::connect(path).await {
        let _ = exchange(&mut stream, &farewell).await;
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    while path.exists() && Instant::now() < deadline {
        tokio::time::sleep(RETRY).await;
    }

    let _ = std::fs::remove_file(path);
    Ok(())
}
