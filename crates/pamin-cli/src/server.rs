//! `pamin serve` — the process that holds what every command used to rebuild.
//!
//! One [`Session`] for the process: one pool, migrated once, and one engine per
//! project, each holding its index open, with the models loaded once. A
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
use pamin_core::JobKind;
use pamin_engine::{Engine, Owed};
use pamin_index::{ProjectionIndex, VectorIndex};
use pamin_store::Workspace;
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::{Framed, LinesCodec};

use crate::command;
use crate::protocol::{Call, Request, Response, SOCKET};
use crate::session::Session;

/// Serves until stopped, and cleans up the socket on the way out.
///
/// `server` is what it runs with, fixed for its lifetime: a request that asks
/// for other settings is refused, never served with these.
pub async fn run(workspace: &Workspace, server: crate::settings::Server) -> Result<()> {
    let path = socket_path(workspace);
    std::fs::create_dir_all(workspace.root())?;

    // Before the socket exists, so a client that connects finds a server that
    // can answer rather than one still starting the database.
    for entry in &server.entries {
        tracing::info!(
            setting = entry.key,
            value = %entry.value,
            source = entry.source.name(),
            "server setting"
        );
    }
    let session =
        Arc::new(Session::open(workspace, server.settings.clone(), server.open_indexes).await?);
    let server = Arc::new(server);

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
        let server = Arc::clone(&server);
        let path = path.clone();
        tokio::spawn(async move {
            match serve_connection(&session, &server, stream).await {
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
/// First it catches up on work that decides what a search finds and that
/// nobody is coming back for -- see [`CatchingUp`]. The rest is the half of
/// the cascade a write no longer waits for. Compacting a
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
    let mut catching_up = CatchingUp::default();
    loop {
        tokio::time::sleep(UPKEEP).await;
        let tick = Instant::now();

        // Before the flushes, which make what it applies durable in the same
        // tick rather than the next.
        catching_up.run(&session).await;
        tracing::debug!(
            seconds = tick.elapsed().as_secs_f64(),
            "catch-up stage finished"
        );

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
            let flush = Instant::now();
            match engine.flush_what_is_applied().await {
                Ok(0) => {}
                Ok(durable) => tracing::debug!(durable, "made applied writes durable"),
                Err(error) => tracing::warn!(%error, "flushing applied writes failed"),
            }
            tracing::debug!(
                ?key,
                seconds = flush.elapsed().as_secs_f64(),
                "flush stage finished"
            );
            // Only between requests. An `optimize` seals the segment being
            // written and lexical scores are per segment (BM25 statistics), so
            // compacting mid-import split the same corpus wherever the tick
            // landed and gave different top-5s from identical vectors. The
            // first quiet tick after an import compacts all it wrote.
            let upkeep = Instant::now();
            if session.is_quiet() {
                match engine.maintain().await {
                    Ok(true) => tracing::debug!("ran index upkeep"),
                    Ok(false) => {}
                    Err(error) => tracing::warn!(%error, "index upkeep failed"),
                }
            }
            tracing::debug!(
                ?key,
                seconds = upkeep.elapsed().as_secs_f64(),
                "index upkeep stage finished"
            );
            // Beside the loop rather than in it: a reshape takes minutes, and
            // every other project's flushes wait on this loop -- a claim held
            // past its lease is replayed. Only between requests too, for the
            // reason upkeep is: a reshape sizes its sibling from the documents
            // held when it starts and optimizes it before applying what was
            // written meanwhile, so one started mid-import would put segment
            // boundaries wherever the tick landed. One skipped here records
            // nothing, so the next quiet tick considers it afresh.
            if session.is_quiet() {
                reshapes.consider(&key, engine);
            }
        }

        // The engines above are dropped by now, so a project that has gone
        // quiet can be closed and the weights it was pinning given back.
        let sweep = Instant::now();
        let (engines, embedder, reranker) = session.close_what_is_idle();
        tracing::debug!(
            seconds = sweep.elapsed().as_secs_f64(),
            "idle sweep finished"
        );
        // And an index the open-index bound closed since the last tick. It was
        // closed inside a request, which is no place to wait on the allocator.
        let evicted = session.take_evicted();
        if !engines.is_empty() || embedder || reranker || evicted > 0 {
            tracing::debug!(
                ?engines,
                embedder,
                reranker,
                evicted,
                "gave back what nothing had asked for"
            );
            trim_heap();
        }
        tracing::debug!(
            seconds = tick.elapsed().as_secs_f64(),
            "upkeep tick finished"
        );
    }
}

/// How many jobs a round of catching up takes.
///
/// The round is what a request that arrives during one waits for, so this is
/// sized against that wait rather than against throughput: a round's forward
/// pass holds the embedding model, and a search on any project
/// queues behind the pass. Throughput gives up little for it. A write's own
/// drain takes sixty-four jobs a round because a round is what one flush
/// covers, and catching up does not flush per round -- it leaves the writes
/// applied, for the flush that follows it in the same tick, exactly as a write
/// does.
///
/// Smaller did not buy anything measurable. With rounds of four, eight and
/// sixteen jobs, two runs each of `catching_up_does_not_hold_a_search_up` on
/// a debug build on four busy cores, searches during catching up took 111,
/// 122 and 131 ms at the median and 2.0, 2.4 and 1.6 s at worst -- the same,
/// within noise that large. A round of sixteen index writes took 0.3 s at the
/// median and 1.2 s at worst on that machine. So this stays at sixteen, where
/// it was set before measuring.
const CATCH_UP_BATCH: i32 = 16;

/// How long one tick may spend catching up, across every project.
///
/// Not the wait a request can see -- a request stops the catching up between
/// rounds -- but how long the rest of the tick waits for it: the flushes that
/// complete other projects' claims, which lapse after a minute, and the idle
/// sweep. It is spent whether or not requests leave room for rounds in it,
/// and the round in progress when it runs out is finished, so a tick can
/// overrun it by one round.
///
/// Equal to the tick, so a server with nothing else to do spends about half
/// its time catching up and the other half idle, rather than every core it
/// has on work nobody is waiting for.
const CATCH_UP_BUDGET: std::time::Duration = UPKEEP;

/// How long a project the server could not open for catching up is left alone.
///
/// Without it a project whose index will not open -- held by another process,
/// or needing a rebuild -- is tried, and warned about, every tick.
const REOPEN_AFTER: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// Runs the work that decides what a search finds, when nobody is asking.
///
/// A write drains what its own memory needs before it returns, so ordinarily
/// nothing is owed. Two things leave work owed with nobody coming back for it:
/// `pamin write --defer`, and a server that died holding claims. Before this,
/// both waited for the next write or import into the same project or for
/// `pamin cascade drain`, and until then the memories were recorded and could
/// not be found. Now they are found a tick or two later.
///
/// Three rules keep it out of the way of the requests it is doing work for:
///
/// - **It runs only while no request is being answered.** It stops between
///   rounds when one arrives and starts again in the next gap, so a request
///   that arrives during a round waits for at most that round -- see
///   [`CATCH_UP_BATCH`] -- and an agent searching every second does not
///   stop the backlog shrinking, only slows it.
/// - **It opens at most one project per tick** that is not already open.
///   That is how a queue left by a server that died is reached: the project
///   it belongs to may be one nobody asks about again for hours, and the
///   first search that does ask should find the memories rather than open
///   the index and miss them. One at a time because opening one may load a
///   model, and only for work that has not already failed -- a job that
///   keeps failing is retried when something opens its project anyway, not
///   every hour on its own account.
/// - **Work owed counts as a use, and nothing else it does.** A project it
///   catches up is stamped as used, like one a request touched, so the idle
///   sweep gives it back one idle window after the work is done rather than
///   half way through a backlog. A server with nothing owed stamps nothing,
///   and gives back what it did before.
///
/// Through the drain every other caller uses, so a round claims, runs and
/// records its jobs exactly as a write's does.
struct CatchingUp {
    /// Projects that would not open, and when they last refused.
    refused: HashMap<String, Instant>,
    batch: i32,
}

impl Default for CatchingUp {
    fn default() -> Self {
        Self {
            refused: HashMap::new(),
            batch: CATCH_UP_BATCH,
        }
    }
}

impl CatchingUp {
    async fn run(&mut self, session: &Session) {
        let owing =
            match pamin_store::jobs::owing(session.database().pool(), &JobKind::URGENT).await {
                Ok(owing) => owing,
                Err(error) => {
                    tracing::warn!(%error, "asking which projects are owed work failed");
                    return;
                }
            };
        self.refused.retain(|_, at| at.elapsed() < REOPEN_AFTER);

        let started = Instant::now();
        let within_budget = || started.elapsed() < CATCH_UP_BUDGET;
        let go_on = || session.is_quiet() && within_budget();
        let mut opened_one = false;
        for owing in owing {
            if !within_budget() {
                return;
            }
            // Under the vector index the index was built with: nothing else
            // will open it, and no index means nothing to catch up.
            let dir = session.workspace().index_dir(owing.project);
            let vector_index = match ProjectionIndex::built_for(&dir) {
                Ok(Some(built)) => built,
                Ok(None) => continue,
                // An index built with a retired model is owed work only a
                // `pamin reindex` can do. Said once a refusal window rather
                // than every tick, and not passed over in silence: its
                // searches go on missing what was written until then.
                Err(error) => {
                    if let std::collections::hash_map::Entry::Vacant(refused) =
                        self.refused.entry(owing.name)
                    {
                        tracing::warn!(
                            project = %refused.key(),
                            %error,
                            "not catching up a project whose index cannot be opened"
                        );
                        refused.insert(Instant::now());
                    }
                    continue;
                }
            };
            let key = owing.name;

            let engine = if session.holds(&key) {
                // Being opened, or rebuilt: the next tick.
                let Some(engine) = session.opened_engine_for_work(&key) else {
                    continue;
                };
                engine
            } else {
                if opened_one || !owing.never_failed || self.refused.contains_key(&key) {
                    continue;
                }
                opened_one = true;
                match session.engine(&key, vector_index).await {
                    Ok(engine) => engine,
                    Err(error) => {
                        tracing::warn!(project = %key, %error, "opening a project to catch up failed");
                        self.refused.insert(key, Instant::now());
                        continue;
                    }
                }
            };

            // Rounds in every gap between requests until the project is
            // caught up or the budget is spent. A request stops the drain
            // between rounds, and waiting here for the next gap holds nothing
            // the request needs.
            let mut caught_up = CaughtUp::default();
            while within_budget() {
                if !session.is_quiet() {
                    tokio::time::sleep(QUIET_POLL).await;
                    continue;
                }
                let draining = Instant::now();
                match engine
                    .drain_cascade_while(Owed::WhatAMemoryNeeds, self.batch, go_on)
                    .await
                {
                    Ok(drained) => caught_up.add(drained, draining.elapsed()),
                    Err(error) => {
                        tracing::warn!(project = %key, %error, "catching up on owed work failed");
                        break;
                    }
                }
                // Stopped with nothing standing in its way: nothing is due.
                if go_on() {
                    break;
                }
            }
            caught_up.log(&key, self.batch);
        }
    }
}

/// How often catching up asks whether the request it stopped for has gone.
///
/// Short against a request and long against the question, which reads one
/// counter: the gap between two searches an agent makes is hundreds of
/// milliseconds, and a round started late into it is a round a later search
/// may wait for.
const QUIET_POLL: std::time::Duration = std::time::Duration::from_millis(20);

/// What one tick's catching up did in one project, for the log.
#[derive(Default)]
struct CaughtUp {
    drains: usize,
    completed: usize,
    failed: usize,
    seconds: f64,
    last: pamin_engine::Drained,
}

impl CaughtUp {
    fn add(&mut self, drained: pamin_engine::Drained, took: std::time::Duration) {
        self.drains += 1;
        self.completed += drained.completed;
        self.failed += drained.failed;
        self.seconds += took.as_secs_f64();
        self.last = drained;
    }

    fn log(&self, project: &str, batch: i32) {
        if self.completed + self.failed + self.last.applied == 0 {
            return;
        }
        tracing::debug!(
            project,
            batch,
            drains = self.drains,
            seconds = self.seconds,
            completed = self.completed,
            failed = self.failed,
            applied = self.last.applied,
            pending = self.last.pending,
            "caught up on owed work"
        );
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
    started: HashMap<String, (Instant, tokio::task::JoinHandle<()>)>,
}

impl Reshapes {
    /// Starts reshaping this project's index in the background, if its shape
    /// is worth it and no reshape of it has started within the interval.
    ///
    /// Never two at once for one project: one still running is never
    /// replaced, and the engine refuses a second on the same directory
    /// besides.
    fn consider(&mut self, project: &str, engine: Arc<Engine>) {
        if let Some((at, running)) = self.started.get(project)
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

        let project = project.to_string();
        let key = project.clone();
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
        self.started.insert(key, (Instant::now(), running));
    }

    /// Forgets projects that are no longer open and have nothing running.
    ///
    /// A project closed and reopened is considered again at once, which is
    /// what opening a project that grew without a server should do.
    fn forget_all_but(&mut self, opened: &[String]) {
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

/// How many heap arenas glibc may keep: [`cap_heap_arenas`].
#[cfg(all(target_os = "linux", target_env = "gnu"))]
const HEAP_ARENAS: i32 = 2;

/// Caps how many heap arenas glibc keeps, at [`HEAP_ARENAS`].
///
/// glibc gives a thread that finds the main heap contended an arena of its
/// own, up to eight per core, and an arena keeps what it has freed. A search
/// runs its reranker passes on threads of their own, so a four-core server
/// ended a search holding thirty-two arenas and about a gigabyte of free heap
/// in them. Measured at the entry point (q51: `pamin serve` at d0b6b89, four
/// Xeon cores, 3,230 texts, 120 timed searches, four rotated runs an arm,
/// the same cap set through `GLIBC_TUNABLES`): search peak RSS 2,354 to
/// 1,306 MB, its anonymous part 1,425 to 398 MB, ingest peak RSS 1,797 to
/// 1,124 MB, search p50 0.958x [0.899, 1.020] and p95 0.967x [0.899, 1.035]
/// of before, the same top five on every query.
///
/// Must run before a second thread allocates: glibc reads the cap when it
/// first needs an arena beyond the main one, and keeps that figure. So
/// `main` calls this before it starts the runtime.
///
/// An arena limit the process was started with -- `MALLOC_ARENA_MAX`, or
/// `arena_max` in `GLIBC_TUNABLES` -- is left alone, whether it is lower or
/// higher: `mallopt` replaces the limit rather than tightening it, and the
/// operator who set one chose it.
///
/// glibc only, like [`trim_heap`]; elsewhere it does nothing.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) fn cap_heap_arenas() {
    unsafe extern "C" {
        /// glibc's own, declared here rather than through a crate for the
        /// reason [`trim_heap`] gives.
        fn mallopt(param: i32, value: i32) -> i32;
    }
    /// `M_ARENA_MAX` in glibc's `malloc.h`.
    const M_ARENA_MAX: i32 = -8;

    if arena_limit_given(
        std::env::var_os("MALLOC_ARENA_MAX")
            .as_deref()
            .map(std::os::unix::ffi::OsStrExt::as_bytes),
        std::env::var_os("GLIBC_TUNABLES")
            .as_deref()
            .map(std::os::unix::ffi::OsStrExt::as_bytes),
    ) {
        return;
    }

    // SAFETY: `mallopt` takes two integers by value and only changes a limit
    // the allocator consults when it next makes an arena. Its flag is not
    // read: this runs before logging starts, and a refused cap leaves glibc's
    // default, which is the behaviour this had before.
    unsafe { mallopt(M_ARENA_MAX, HEAP_ARENAS) };
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn cap_heap_arenas() {}

/// Whether the process was started with its own glibc arena limit: a
/// non-empty `MALLOC_ARENA_MAX`, or `glibc.malloc.arena_max` set to anything
/// among the colon-separated `GLIBC_TUNABLES`, read as bytes as glibc reads
/// them.
///
/// The value is not parsed. glibc has no call that reports the limit it
/// took, and its parser's corners (signs, blanks, octal, overflow) are its
/// own; a copy would drift from it. Any value an operator wrote is left to
/// glibc: one it applies is never loosened, and one it ignores, which glibc
/// warns about at start-up, costs only this cap.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn arena_limit_given(malloc_arena_max: Option<&[u8]>, tunables: Option<&[u8]>) -> bool {
    malloc_arena_max.is_some_and(|value| !value.is_empty())
        || tunables.is_some_and(|tunables| {
            tunables.split(|byte| *byte == b':').any(|tunable| {
                tunable
                    .strip_prefix(b"glibc.malloc.arena_max=")
                    .is_some_and(|value| !value.is_empty())
            })
        })
}

/// Whether the connection asked the server to stop.
enum Shutdown {
    Requested,
    No,
}

/// Reads requests from one client until it hangs up.
async fn serve_connection(
    session: &Arc<Session>,
    server: &crate::settings::Server,
    stream: UnixStream,
) -> Result<Shutdown> {
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
        // Then the settings. Never served with other settings than asked
        // for, and never replaced from here: a server another agent is using
        // is not this client's to restart, so the caller is told how.
        if !stopping && let Some(message) = server.mismatch(&request.server) {
            framed
                .send(serde_json::to_string(&Response::Err(message))?)
                .await?;
            continue;
        }
        let name = request.call.name();
        let serving = session.serving();
        let response = match answer(session, request).await {
            Ok(value) => Response::Ok(value),
            // Rendered here rather than shipped as a type: the client fails
            // with this string as it stands, chain and all.
            Err(error) => {
                tracing::warn!(call = name, %error, "request failed");
                Response::Err(format!("{error:#}"))
            }
        };
        drop(serving);

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

/// Runs one request against the session.
///
/// The dispatch is a match rather than a trait because there is exactly one
/// implementation of each arm and the compiler checking that every command has
/// one is worth more than the indirection would be.
async fn answer(session: &Arc<Session>, request: Request) -> Result<Payload> {
    let Request {
        project,
        vector_index,
        call,
        ..
    } = request;

    // `stop` before the vector index is read: it opens no index, and a stop
    // that was refused for the index it named would end this server with
    // the database still running.
    if let Call::Stop = call {
        return json(command::stop::execute(session.workspace()).await?);
    }

    let vector_index = VectorIndex::parse(&vector_index)
        .ok_or_else(|| anyhow::anyhow!("unknown vector index {vector_index:?}"))?;

    // Before the request, not after it: the loads are what the next search
    // would wait for, and this request is the earliest sign one is coming.
    // Not for `reindex`, which discards the index a warm-up would be holding
    // open.
    if !matches!(call, Call::Reindex(_)) {
        session.warm(&project, vector_index);
    }

    let value = match call {
        Call::Init => json(command::init::execute(session, &project).await?)?,
        Call::Write(args) => {
            json(command::write::execute(session, &project, vector_index, args).await?)?
        }
        Call::Import(args) => {
            json(command::import::execute(session, &project, vector_index, args).await?)?
        }
        Call::Read(args) => json(command::read::execute(session, &project, args).await?)?,
        Call::Search(args) => {
            json(command::search::execute(session, &project, vector_index, args).await?)?
        }
        Call::Grep(args) => json(command::grep::execute(session, &project, args).await?)?,
        Call::Link(args) => json(command::link::execute(session, &project, args).await?)?,
        Call::Unlink(args) => json(command::unlink::execute(session, &project, args).await?)?,
        Call::Neighbors(args) => json(command::neighbors::execute(session, &project, args).await?)?,
        Call::Topics(args) => {
            json(command::topics::execute(session, &project, vector_index, args).await?)?
        }
        Call::Reindex(args) => {
            json(command::reindex::execute(session, &project, vector_index, args).await?)?
        }
        Call::Cascade(args) => {
            json(command::cascade::answer(session, &project, vector_index, args).await?)?
        }
        Call::Stop => unreachable!("answered above"),
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

#[cfg(all(test, target_os = "linux", target_env = "gnu"))]
mod tests {
    use super::arena_limit_given;

    #[test]
    fn an_arena_limit_the_process_was_started_with_is_left_alone() {
        let given = |env: Option<&str>, tunables: Option<&[u8]>| {
            arena_limit_given(env.map(str::as_bytes), tunables)
        };
        assert!(given(Some("1"), None));
        assert!(given(None, Some(b"glibc.malloc.arena_max=1")));
        assert!(given(
            None,
            Some(b"glibc.malloc.trim_threshold=0:glibc.malloc.arena_max=8")
        ));
        assert!(given(None, Some(b"glibc.malloc.arena_max=1:glibc.\xff=0")));
        // Whatever the operator wrote is glibc's to read.
        for written in ["-1", "+1", "0x2", "010", "0", "two"] {
            assert!(given(Some(written), None), "{written:?}");
            let tunables = format!("glibc.malloc.arena_max={written}");
            assert!(given(None, Some(tunables.as_bytes())), "{written:?}");
        }

        assert!(!given(None, None));
        assert!(!given(Some(""), None));
        assert!(!given(None, Some(b"glibc.malloc.arena_max=")));
        assert!(!given(None, Some(b"glibc.malloc.trim_threshold=0")));
        assert!(!given(None, Some(b"glibc.malloc.arena_test=2")));
    }
}
