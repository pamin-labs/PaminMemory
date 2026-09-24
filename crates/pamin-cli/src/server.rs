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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use futures::{SinkExt, StreamExt};
use pamin_engine::Engine;
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
    map_large_blocks();

    // Before the socket exists, so a client that connects finds a server that
    // can answer rather than one still starting the database.
    let session = Arc::new(Session::open(workspace, Connections::Resident).await?);

    // A socket file left by a process that died is not a listener, so the stale
    // one has to go before binding. Connecting to it first is what tells the
    // two apart: a live server accepts, a dead one's leftover file does not.
    //
    // Removing unconditionally was the bug. A second server would take the
    // socket from a working one and then fail every request, because the index
    // lock it also needs is still held by the server it displaced -- and the
    // workspace looked broken rather than busy. It happens whenever a client
    // runs as a different user from the server, since that client cannot see
    // the running one as its own and tries to start its own.
    if tokio::net::UnixStream::connect(&path).await.is_ok() {
        bail!(
            "a server is already listening at {}; run `pamin stop` first if you \
             mean to replace it",
            path.display()
        );
    }
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
///
/// Flushing comes first and is the more important half. A write leaves its
/// document in the projection's buffer, where queries can already see it, and
/// leaves the job claimed until a flush puts it on disk -- so this loop is what
/// turns those claims into completions. Nothing is lost if it never runs: the
/// claims lapse and the ledger replays them. What is lost is the amortization,
/// which is the entire reason the write did not flush for itself.
async fn maintain(session: Arc<Session>) {
    let mut reshapes = Reshapes::default();
    loop {
        tokio::time::sleep(UPKEEP).await;

        // After the per-project work rather than before: flushing is what
        // turns a write's claim into a completion, and closing an index that
        // still owes one would leave the claim to lapse and be replayed. The
        // idle window is minutes and this loop runs every few seconds, so
        // anything owed has been drained long before a project looks idle.

        // One engine at a time, and taken by key. Holding all of them for the
        // length of a sweep makes every one of them look busy to eviction,
        // which then finds nothing to close and lets the registry grow past
        // its bound whenever a cold project arrives during a tick.
        let opened = session.opened_projects();
        reshapes.forget_all_but(&opened);
        for key in opened {
            let Some(engine) = session.opened_engine(&key) else {
                // Being opened, or being rebuilt. Its upkeep waits for the
                // next tick rather than this loop waiting for it.
                continue;
            };
            match engine.flush_what_is_applied().await {
                Ok(0) => {}
                Ok(durable) => tracing::debug!(durable, "made applied writes durable"),
                Err(error) => tracing::warn!(%error, "flushing applied writes failed"),
            }
            match engine.maintain().await {
                Ok(true) => tracing::debug!("ran index upkeep"),
                Ok(false) => {}
                Err(error) => tracing::warn!(%error, "index upkeep failed"),
            }
            // Beside the loop rather than in it: a reshape takes minutes, and
            // every other project's flushes wait on this loop -- a claim held
            // past its lease is replayed.
            reshapes.consider(&key, engine);
        }

        // The engines above are dropped by now, so a project that has gone
        // quiet can be closed and the weights it was pinning given back.
        let (engines, embedders, rerankers) = session.close_what_is_idle();
        // And an index the open-index bound closed since the last tick. It was
        // closed inside a request, which is no place to wait on the allocator.
        let evicted = session.take_evicted();
        if !engines.is_empty() || !embedders.is_empty() || !rerankers.is_empty() || evicted > 0 {
            tracing::debug!(
                ?engines,
                ?embedders,
                ?rerankers,
                evicted,
                "gave back what nothing had asked for"
            );
            trim_heap();
        }
    }
}

/// How long after one reshape of a project the server will consider another.
///
/// A reshape is only started when the index is spread over more than twice
/// the segments the policy wants, and one that finished leaves it at what the
/// policy wants -- so the next is not due until the project has roughly
/// doubled, and the interval is not what paces them. It is what paces a
/// failure: a reshape that failed on something that has not changed would
/// otherwise copy the whole index again every tick. An hour is a choice, not
/// a measurement. Outside it a project is checked on every tick, which costs
/// one read of the index's statistics under its lock.
const RESHAPE_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60 * 60);

/// The reshapes this server has started, one per project at most.
///
/// Held by the upkeep loop rather than the session, because nothing else
/// starts one: `pamin reindex` is a rebuild, and waits for a reshape of its
/// project to finish rather than being one.
#[derive(Default)]
struct Reshapes {
    started: HashMap<(String, Profile), (Instant, tokio::task::JoinHandle<()>)>,
}

impl Reshapes {
    /// Starts reshaping this project's index in the background, if its shape
    /// is worth it and no reshape of it has started within the interval.
    ///
    /// Never two at once for one project: one still running is never
    /// replaced, and the engine refuses a second on the same directory
    /// besides.
    fn consider(&mut self, key: &(String, Profile), engine: Arc<Engine>) {
        if let Some((at, running)) = self.started.get(key)
            && (!running.is_finished() || at.elapsed() < RESHAPE_INTERVAL)
        {
            return;
        }
        let shape = match engine.segmentation() {
            Ok(shape) => shape,
            Err(error) => {
                tracing::warn!(%error, "reading the index's shape failed");
                return;
            }
        };
        if !shape.is_worth_rebuilding() {
            return;
        }

        let project = key.0.clone();
        tracing::info!(
            %project,
            documents = shape.documents,
            segments = shape.segments(),
            wanted = shape.wanted(),
            "reshaping the index"
        );
        let running = tokio::spawn(async move {
            let started = Instant::now();
            match engine.reshape().await {
                Ok(Some(reshaped)) => tracing::info!(
                    %project,
                    before = reshaped.before.segments(),
                    after = reshaped.after.segments(),
                    copied = reshaped.copied,
                    caught_up = reshaped.caught_up,
                    seconds = started.elapsed().as_secs_f64(),
                    "reshaped the index"
                ),
                Ok(None) => tracing::info!(
                    %project,
                    "the index needed no reshaping, or something else was restructuring it"
                ),
                Err(error) => tracing::warn!(
                    %project,
                    %error,
                    seconds = started.elapsed().as_secs_f64(),
                    "reshaping the index failed; the index it was copying is still served"
                ),
            }
        });
        self.started.insert(key.clone(), (Instant::now(), running));
    }

    /// Forgets projects that are no longer open and have nothing running.
    ///
    /// A project closed and reopened is considered again at once, which is
    /// what opening a project that grew without a server should do.
    fn forget_all_but(&mut self, opened: &[(String, Profile)]) {
        self.started
            .retain(|key, (_, running)| !running.is_finished() || opened.contains(key));
    }
}

/// Asks the allocator to hand back what the sweep just freed.
///
/// Dropping a model returns its pages to the C heap and not to the kernel, and
/// the difference is a gigabyte. Measured on a server over the
/// 13,014-document project: resident 2,263 MB after one `fast` search, 1,000
/// MB once the idle sweep had released both models -- so 1,263 MB came back on
/// its own and roughly a gigabyte of free heap stayed mapped. `malloc_trim`
/// is what asks for that gigabyte.
///
/// Called only when the sweep released something, so a quiet server does not
/// walk its arenas every five seconds for nothing. It is advisory: the
/// allocator returns what it can and keeps what it cannot, and a return value
/// of zero means it found nothing to give, which is not an error.
///
/// glibc only. Other libcs either do this themselves or do not offer it, and
/// the fallback is the behaviour this had before -- which is why it is an
/// empty function rather than a compile error.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn trim_heap() {
    unsafe extern "C" {
        /// glibc's own, declared here rather than through a crate: it is one
        /// symbol, and `ci/budget.py` counts dependencies.
        fn malloc_trim(pad: usize) -> i32;
    }

    // SAFETY: the call takes a byte count by value, returns a flag, and has no
    // preconditions. What it touches is memory the allocator already considers
    // free, so nothing safe Rust can observe changes.
    let released = unsafe { malloc_trim(0) };
    tracing::trace!(released, "asked the allocator for its free pages back");
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn trim_heap() {}

/// The size from which the allocator maps a block of its own, fixed here
/// rather than left to move.
///
/// glibc starts at 128 KiB and raises the threshold to the size of any mapped
/// block that is freed, up to 32 MiB, so after the first large block goes
/// every later one is carved from the heap -- and a freed heap block stays
/// resident until [`trim_heap`] asks for it. The index makes that the common
/// case. Its full-text store keeps twelve RocksDB memtables per segment, each
/// of which allocates and zeroes an 8,000,000-byte hash-bucket array when it
/// is created: 92 MiB an open segment whatever it holds, one document or
/// forty. Closing an index therefore handed 92 MiB to the heap and none of it
/// to the operating system. Measured on the harness that opens six one-document
/// indexes and closes them again:
///
/// ```text
///                            resident after closing   malloc_trim then
///   glibc's moving threshold             561 MiB          57 to 108 ms
///   fixed at 1 MiB                        27 MiB           1.3 to 1.9 ms
/// ```
///
/// Through a server, the hundred-project end-to-end test with sixteen indexes
/// open reached 4,765 and 4,793 MiB anonymous at the last project with the
/// moving threshold, against 3,135 and 3,351 MiB with the threshold fixed at
/// 4 MiB, and grew by 1,124 and 1,137 MiB between the sixteenth project and
/// the hundredth against 148 and 80.
///
/// Four MiB rather than one: half the bucket arrays' size, so they are always
/// mapped, and high enough that the smaller blocks a forward pass allocates
/// and frees on every search stay on the heap and are reused rather than
/// mapped and faulted in again each time.
const MMAP_THRESHOLD: i32 = 4 * 1024 * 1024;

/// Fixes the allocator's mmap threshold at [`MMAP_THRESHOLD`].
///
/// Called once, before anything is opened. glibc only, like [`trim_heap`], and
/// for the same reason an empty function elsewhere.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn map_large_blocks() {
    unsafe extern "C" {
        /// glibc's own, declared here for the reason `malloc_trim` is.
        fn mallopt(param: i32, value: i32) -> i32;
    }
    /// `M_MMAP_THRESHOLD` in glibc's `malloc.h`.
    const M_MMAP_THRESHOLD: i32 = -3;

    // SAFETY: takes two integers by value and changes a tunable the allocator
    // reads on its next large allocation; no memory is touched. Setting it
    // also stops glibc adjusting it, which is the point.
    let set = unsafe { mallopt(M_MMAP_THRESHOLD, MMAP_THRESHOLD) };
    if set != 1 {
        tracing::warn!("the allocator refused a fixed mmap threshold");
    }
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
fn map_large_blocks() {}

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

        framed.send(within_bounds(&response)?).await?;

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
async fn answer(session: &Session, request: Request) -> Result<Payload> {
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
        Call::Topics(args) => {
            json(command::topics::execute(session, &project, profile, args).await?)?
        }
        Call::Reindex(args) => {
            json(command::reindex::execute(session, &project, profile, args).await?)?
        }
        Call::Cascade(args) => {
            json(command::cascade::answer(session, &project, profile, args).await?)?
        }
        Call::Stop => json(command::stop::execute(session.workspace()).await?)?,
    };

    Ok(value)
}

/// A command's result, serialized once and never re-walked.
type Payload = Box<serde_json::value::RawValue>;

/// The response as a line, refused here if it is one the client cannot read.
///
/// The codec's limit is a decoder's: it bounds what this reads and says nothing
/// about what it writes. So an oversized answer used to be computed in full,
/// written in full, and rejected at the other end by the client's decoder --
/// which reports it as an unreadable response rather than as a result too large
/// to send. Saying so here costs a length check and makes the message name the
/// problem.
fn within_bounds(response: &Response) -> Result<String> {
    let line = serde_json::to_string(response)?;
    if line.len() > MAX_REQUEST {
        let too_big = Response::Err(format!(
            "the result is {} bytes and the most that can be sent is {MAX_REQUEST}; \
             ask for less of it -- a smaller --limit, or a narrower grep",
            line.len()
        ));
        return Ok(serde_json::to_string(&too_big)?);
    }
    Ok(line)
}

fn json<T: serde::Serialize>(value: T) -> Result<Payload> {
    Ok(serde_json::value::to_raw_value(&value)?)
}

/// Where this workspace's socket lives.
pub fn socket_path(workspace: &Workspace) -> PathBuf {
    workspace.root().join(SOCKET)
}
