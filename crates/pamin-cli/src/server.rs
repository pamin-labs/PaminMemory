//! `pamin serve` — the process that holds what every command used to rebuild.
//!
//! One [`Session`] for the process: one pool, migrated once, and one engine per
//! project and profile, each holding its index open and its model loaded. A
//! command arriving here finds them already there. Before, and still without a
//! server, every command built all of it, used it once, and dropped it.
//!
//! A Unix domain socket rather than a port: it is filesystem-scoped, so it
//! inherits the workspace's permissions and cannot be reached from off the
//! machine by accident. No authentication, for the same reason.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use futures::{SinkExt, StreamExt};
use pamin_index::Profile;
use pamin_store::{Connections, Workspace};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::{Framed, LinesCodec};

use crate::command;
use crate::protocol::{Call, Request, Response, SOCKET};
use crate::session::Session;

/// Serves until stopped, and cleans up the socket on the way out.
pub async fn run(workspace: &Workspace) -> Result<()> {
    let path = socket_path(workspace);
    std::fs::create_dir_all(workspace.root())?;

    // Before the socket exists, so a client that connects finds a server that
    // can answer rather than one still starting the database.
    let session = Arc::new(Session::open(workspace, Connections::Resident).await?);

    // A socket file left by a process that died is not a listener, and binding
    // over it is the only way to find out. Removing it first is safe because a
    // live server would have failed the client's connect attempt, not this one.
    let _ = std::fs::remove_file(&path);
    let listener =
        UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;

    tokio::spawn(maintain(Arc::clone(&session)));

    tracing::info!(socket = %path.display(), "serving");

    loop {
        let (stream, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) => {
                tracing::warn!(%error, "accept failed");
                continue;
            }
        };

        let session = Arc::clone(&session);
        let path = path.clone();
        tokio::spawn(async move {
            match serve_connection(&session, stream).await {
                Ok(Shutdown::Requested) => {
                    // Answered first, then gone: the client is waiting on the
                    // reply that says the database stopped.
                    let _ = std::fs::remove_file(&path);
                    tracing::info!("stopping");
                    std::process::exit(0);
                }
                Ok(Shutdown::No) => {}
                Err(error) => tracing::warn!(%error, "connection failed"),
            }
        });
    }
}

/// How often the server looks for index upkeep to do.
///
/// Not a pace for the work -- the work is scheduled by whoever caused it, and
/// a tick that finds nothing owed costs one query per open project. It is how
/// long an index may stay untidy after a burst of writes, and seconds of that
/// changes nothing a caller can see.
const UPKEEP: std::time::Duration = std::time::Duration::from_secs(5);

/// Does the index's housekeeping, away from whoever caused it.
///
/// This is the half of the cascade a write no longer waits for. Compacting a
/// few hundred index files takes a third of a second and makes nothing more
/// correct, so paying for it in front of an agent was the wrong place; the
/// server is still here afterwards, which is the whole qualification for the
/// job.
///
/// It claims like any other worker, so two servers against one workspace share
/// the work rather than repeat it, and a failure is logged and retried on its
/// own schedule rather than taken out on a request.
async fn maintain(session: Arc<Session>) {
    loop {
        tokio::time::sleep(UPKEEP).await;

        for engine in session.open_engines().await {
            match engine.maintain().await {
                Ok(true) => tracing::debug!("ran index upkeep"),
                Ok(false) => {}
                Err(error) => tracing::warn!(%error, "index upkeep failed"),
            }
        }
    }
}

/// Whether the connection asked the server to stop.
enum Shutdown {
    Requested,
    No,
}

/// Reads requests from one client until it hangs up.
async fn serve_connection(session: &Session, stream: UnixStream) -> Result<Shutdown> {
    let mut framed = Framed::new(stream, LinesCodec::new_with_max_length(MAX_REQUEST));

    while let Some(line) = framed.next().await {
        let line = line.context("reading a request")?;

        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => {
                let response = Response::Err(format!("unreadable request: {error}"));
                framed.send(serde_json::to_string(&response)?).await?;
                continue;
            }
        };

        let stopping = matches!(request.call, Call::Stop);

        // Checked before anything is served, and after `stop` is recognised.
        // Stopping means the same thing in every build, and it is how a client
        // replaces a server it has just rejected -- refusing it on a version
        // mismatch would leave the mismatched server listening for ever, which
        // is the one situation this check exists to get out of.
        let ours = crate::protocol::version();
        if !stopping && request.version != ours {
            let response = Response::Mismatch { server: ours };
            framed.send(serde_json::to_string(&response)?).await?;
            // No point carrying on: the client is about to replace this
            // process rather than trust field names that may have moved.
            continue;
        }
        let name = request.call.name();
        let response = match answer(session, request).await {
            Ok(value) => Response::Ok(value),
            // Rendered here rather than shipped as a type: this is the same
            // string the in-process path prints, chain and all.
            Err(error) => {
                tracing::warn!(call = name, %error, "request failed");
                Response::Err(format!("{error:#}"))
            }
        };

        framed.send(serde_json::to_string(&response)?).await?;

        // Stopping happens whether or not the database could be stopped. The
        // reply carries that failure, and the caller sees it -- but a server
        // that stayed up because the cluster refused to would be one nothing
        // could ever shut down, and `stop` is also the way a client replaces a
        // server it cannot talk to.
        if stopping {
            return Ok(Shutdown::Requested);
        }
    }

    Ok(Shutdown::No)
}

/// The largest request line the server will read.
///
/// A memory arrives inline in a `write`, so this is a bound on one memory
/// rather than on a protocol envelope. Generous, but not unbounded: without a
/// limit a client that never sends a newline holds the buffer open for ever.
const MAX_REQUEST: usize = 16 * 1024 * 1024;

/// Runs one request against the in-process path.
///
/// The dispatch is a match rather than a trait because there is exactly one
/// implementation of each arm and the compiler checking that every command has
/// one is worth more than the indirection would be.
async fn answer(session: &Session, request: Request) -> Result<serde_json::Value> {
    let Request {
        project,
        profile,
        call,
        ..
    } = request;

    let profile =
        Profile::parse(&profile).ok_or_else(|| anyhow::anyhow!("unknown profile {profile:?}"))?;

    let value = match call {
        Call::Init => json(command::init::execute(session, &project).await?)?,
        Call::Write(args) => {
            json(command::write::execute(session, &project, profile, args).await?)?
        }
        Call::Import(args) => {
            json(command::import::execute(session, &project, profile, args).await?)?
        }
        Call::Read(args) => json(command::read::execute(session, &project, args).await?)?,
        Call::Search(args) => {
            json(command::search::execute(session, &project, profile, args).await?)?
        }
        Call::Grep(args) => json(command::grep::execute(session, &project, args).await?)?,
        Call::Link(args) => json(command::link::execute(session, &project, args).await?)?,
        Call::Unlink(args) => json(command::unlink::execute(session, &project, args).await?)?,
        Call::Neighbors(args) => json(command::neighbors::execute(session, &project, args).await?)?,
        Call::Reindex(args) => {
            json(command::reindex::execute(session, &project, profile, args).await?)?
        }
        Call::Cascade(args) => command::cascade::answer(session, &project, profile, args).await?,
        Call::Stop => json(command::stop::execute(session.workspace()).await?)?,
    };

    Ok(value)
}

fn json<T: serde::Serialize>(value: T) -> Result<serde_json::Value> {
    Ok(serde_json::to_value(value)?)
}

/// Where this workspace's socket lives.
pub fn socket_path(workspace: &Workspace) -> PathBuf {
    workspace.root().join(SOCKET)
}
