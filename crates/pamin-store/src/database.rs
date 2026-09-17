//! Bringing up and connecting to the local PostgreSQL server.
//!
//! PostgreSQL is bundled rather than brought by the user. The CLI is meant to
//! be a zero-integration path for any agent with a shell, and that promise does
//! not survive a prerequisite that starts with installing a database.

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::time::Duration;

use postgresql_embedded::{PostgreSQL, Settings, VersionReq};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

use crate::error::Result;
use crate::workspace::{LocalServer, Workspace};

/// The database name created inside the embedded cluster.
const DATABASE: &str = "pamin";

/// How many connections a process may hold against the cluster.
///
/// Not a property of the cluster but of how many processes are pointing at it.
/// A pool sized for one process is the wrong size when there are thirty, and
/// the size that survives thirty processes is a bottleneck when there is one.
#[derive(Clone, Copy, Debug)]
pub enum Connections {
    /// One command among many, each with a pool of its own.
    PerCommand,
    /// The only process talking to this cluster, holding one pool for it all.
    Resident,
}

impl Connections {
    /// Whether this process is the one that stays.
    ///
    /// Asked about work that has to outlive the request that scheduled it.
    /// A command that exits after one write cannot hand anything on, so it
    /// finishes what it started; a server can, and should, because the
    /// alternative is an agent waiting out the index's housekeeping.
    pub fn resident(self) -> bool {
        matches!(self, Self::Resident)
    }

    /// The pool size this calls for.
    fn limit(self) -> u32 {
        match self {
            // Enough for the recall channels of one command to run at once,
            // and small enough to multiply by however many commands there are.
            Self::PerCommand => 4,
            // Four per core, because a connection spends most of its life
            // waiting on the server rather than on this process, capped
            // because the cluster allows 300 and one process should not be
            // able to take all of them.
            Self::Resident => (4 * available_cores()).clamp(4, 64),
        }
    }
}

/// How many cores this machine will actually give us.
fn available_cores() -> u32 {
    std::thread::available_parallelism()
        .map(|cores| cores.get() as u32)
        .unwrap_or(1)
}

/// A connection pool against this workspace's cluster.
///
/// Cloning shares the pool rather than opening a second one: `PgPool` is a
/// handle, and a resident server holding one for the machine is the whole
/// reason the pool replaced a single client.
#[derive(Clone)]
pub struct Database {
    pool: PgPool,
    connections: Connections,
}

impl Database {
    /// Ensures a server is running for this workspace, connects, and migrates.
    ///
    /// Safe to call repeatedly. The first call installs and initializes the
    /// cluster; later calls reuse the running server.
    pub async fn open(workspace: &Workspace, connections: Connections) -> Result<Self> {
        let server = match workspace.read_server()? {
            Some(existing) if can_connect(&existing).await => existing,
            _ => start_server(workspace).await?,
        };

        let database = Self::connect(&server, connections).await?;
        crate::migrate::run(&database.pool).await?;
        Ok(database)
    }

    /// Connects to an already running server without touching its lifecycle.
    pub async fn connect(server: &LocalServer, connections: Connections) -> Result<Self> {
        Ok(Self {
            pool: pool(&server.url(), connections).await?,
            connections,
        })
    }

    /// The connection pool. Every query goes through this.
    ///
    /// Handed out rather than wrapped. A pool is already the right thing to
    /// hand a query -- it lends a connection per statement and takes it back --
    /// whereas the single client this replaced was one connection that every
    /// query in the process queued behind, however many were ready to run.
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Whether the process holding this is the one that stays.
    ///
    /// The same fact that sizes the pool, asked for a different reason: work
    /// that can be handed on needs somebody to hand it to.
    pub fn is_resident(&self) -> bool {
        self.connections.resident()
    }
}

/// Opens a pool sized for one short-lived command.
///
/// Every `pamin` invocation is its own process with its own pool, all pointing
/// at one cluster, so the pool's size is multiplied by however many agents are
/// running. The default of ten connections each means thirty agents ask for
/// three hundred, against a server that allows a hundred, and what they get is
/// `too many clients` after a thirty-second wait. A resident server is the
/// other case entirely -- one pool for the machine, nothing to multiply by --
/// so [`Connections`] is the caller's to state.
///
/// `test_before_acquire` is off. It costs a full round trip on every acquire to
/// detect connections dropped by a proxy or an idle timer, and there is neither
/// between here and a cluster on this machine that this process just started.
async fn pool(url: &str, connections: Connections) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(connections.limit())
        .min_connections(1)
        .test_before_acquire(false)
        // A command outlives neither, so recycling connections underneath it
        // only adds reconnects.
        .idle_timeout(None)
        .max_lifetime(None)
        // Long enough to outlast a busy moment, short enough that a caller
        // learns the server is unreachable rather than appearing to hang.
        .acquire_timeout(Duration::from_secs(10))
        .connect(url)
        .await?;

    Ok(pool)
}

/// Returns true when a server is already listening with these credentials.
///
/// The pool is closed again rather than kept: this answers whether to start a
/// cluster, and the caller opens its own once it knows.
async fn can_connect(server: &LocalServer) -> bool {
    match PgPool::connect(&server.url()).await {
        Ok(pool) => {
            pool.close().await;
            true
        }
        Err(_) => false,
    }
}

/// What this cluster is started with.
///
/// Defaults chosen for a general-purpose server that somebody administers.
/// Nobody administers this one -- it is started by a memory tool on a
/// developer's machine and never configured -- so the few that are plainly
/// wrong for that are set here, and the rest are left alone.
///
/// Nothing here reserves memory up front. `shared_buffers` and
/// `maintenance_work_mem` are the two that usually head a tuning list and both
/// are deliberately untouched: the usual advice sizes them as a fraction of a
/// machine dedicated to the database, and this one is a laptop doing something
/// else. Their defaults cost the operating system's page cache doing the work
/// instead, which is the right trade for a background process.
fn settings() -> HashMap<String, String> {
    HashMap::from([
        // PostgreSQL allows a hundred clients, and every `pamin` command
        // without a server is a process with its own pool pointing at this one
        // cluster. A few dozen agents working at once exhaust that, and what
        // they see is a connection timeout rather than anything naming the
        // limit.
        ("max_connections".to_string(), "300".to_string()),
        // Four megabytes is enough for a sort of a few thousand rows and is
        // reached by a project long before it is large. Past it the sort goes
        // to a temporary file, which is the same answer arriving after a disk
        // round trip. Thirty-two is a bound on one sort of one connection, so
        // the worst case is the pool's size times this and not this times
        // `max_connections`.
        ("work_mem".to_string(), "32MB".to_string()),
        // The planner's default assumes a seek costs four times a sequential
        // page. That was a spinning disk. Local storage makes the two nearly
        // the same, and the default is what talks the planner out of index
        // scans it should be choosing.
        ("random_page_cost".to_string(), "1.1".to_string()),
        // Compiling a query pays off over seconds of execution. Every query
        // here is a lookup by key or a bounded scan, so the compilation is the
        // slow part and there is nothing for it to pay back against.
        ("jit".to_string(), "off".to_string()),
        // Autovacuum waits for a fifth of a table to be dead rows. On a table
        // of a hundred million states that is twenty million, and until then
        // every scan reads them. Two per cent is still rare enough to be
        // cheap and often enough to keep up.
        (
            "autovacuum_vacuum_scale_factor".to_string(),
            "0.02".to_string(),
        ),
    ])
}

/// Installs if needed, starts the server, and leaves it running.
async fn start_server(workspace: &Workspace) -> Result<LocalServer> {
    std::fs::create_dir_all(workspace.root())?;

    let mut settings = Settings {
        version: VersionReq::parse("=17.6.0").expect("valid version requirement"),
        installation_dir: workspace.postgres_install_dir(),
        data_dir: workspace.postgres_data_dir(),
        password_file: workspace.postgres_password_file(),
        // Not temporary: the cluster outlives the process that created it, so a
        // workspace survives between commands.
        temporary: false,
        // A hard limit on `initdb` and `pg_ctl start`, not a poll interval. The
        // default is five seconds, which a first `initdb` on a cold filesystem
        // exceeds routinely -- and the failure lands on whoever is setting the
        // project up for the first time, which is the worst audience for it.
        timeout: Some(Duration::from_secs(60)),
        // Passed to the server as it starts, so an already-initialised
        // workspace keeps whatever it was started with until `pamin stop`.
        configuration: settings(),
        ..Settings::default()
    };

    // Reuse the credentials and port from a previous run when the cluster was
    // already initialized, since initdb fixed the superuser password then.
    if let Some(existing) = workspace.read_server()? {
        settings.username = existing.username.clone();
        settings.password = existing.password.clone();
        settings.port = existing.port;
    }

    let mut postgres = PostgreSQL::new(settings);
    postgres.setup().await?;
    postgres.start().await?;

    if !postgres.database_exists(DATABASE).await? {
        postgres.create_database(DATABASE).await?;
    }

    let settings = postgres.settings();
    let server = LocalServer {
        host: settings.host.clone(),
        port: settings.port,
        username: settings.username.clone(),
        password: settings.password.clone(),
        database: DATABASE.to_string(),
        // setup() qualifies the install directory with the resolved version, so
        // record what it settled on rather than what we asked for.
        installation_dir: settings.installation_dir.clone(),
    };
    workspace.write_server(&server)?;

    // Suppress the handle's Drop, which would stop the server we just started.
    // Leaving it running is the point: an agent invoking the CLI repeatedly
    // should not pay cluster startup on every call.
    let _ = ManuallyDrop::new(postgres);

    Ok(server)
}

/// Stops the server for this workspace, if one is running.
///
/// "If one is running" is the whole of the difficulty. A cluster that was
/// killed rather than stopped -- the machine lost power, the container was
/// reclaimed -- leaves its `postmaster.pid` behind, and `pg_ctl` believes it:
/// it reads the recorded process id and signals it. On a machine that has
/// rebooted since, that number belongs to whatever started early enough to
/// claim it, so stopping a cluster that is already gone either fails loudly
/// against a process we are not allowed to signal, or succeeds against one we
/// had no business signalling. Both were reachable here.
///
/// So the file is checked against something that cannot be recycled: whether
/// anything is listening on the port the file itself records. A postmaster
/// binds its port before it writes that line and holds it until it exits, and
/// it holds it while starting up and while refusing connections, so the
/// listener answers the question a connection attempt would answer less
/// reliably. Nothing listening means the file outlived its cluster.
pub async fn stop(workspace: &Workspace) -> Result<()> {
    let Some(server) = workspace.read_server()? else {
        return Ok(());
    };

    if let Some(stale) = stale_pid_file(&workspace.postgres_data_dir()) {
        std::fs::remove_file(&stale)?;
        return Ok(());
    }

    let settings = Settings {
        installation_dir: server.installation_dir,
        data_dir: workspace.postgres_data_dir(),
        password_file: workspace.postgres_password_file(),
        username: server.username,
        password: server.password,
        port: server.port,
        temporary: false,
        // Same reason as starting: this is a hard limit on `pg_ctl stop`, and
        // a shutdown that waits for a long checkpoint is not a hung one.
        timeout: Some(Duration::from_secs(60)),
        ..Settings::default()
    };

    PostgreSQL::new(settings).stop().await?;
    Ok(())
}

/// The cluster's lock file, if it is one no cluster is behind.
///
/// Returns the path to remove; `None` means either that there is no lock file
/// -- nothing to stop, and `pg_ctl` will say so -- or that something is
/// listening on the port it names, which is the case where the file is telling
/// the truth and must be left alone.
///
/// Unreadable or malformed files are left alone too. A file we cannot parse is
/// not evidence that a cluster is gone, and deleting the lock of a cluster that
/// is running is how two postmasters end up on one data directory.
fn stale_pid_file(data_dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let path = data_dir.join("postmaster.pid");
    let contents = std::fs::read_to_string(&path).ok()?;
    // Line four, counting from one: the port. The format is PostgreSQL's and
    // has carried the port in that position since 9.1.
    let port: u16 = contents.lines().nth(3)?.trim().parse().ok()?;
    let listening = std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        PROBE,
    )
    .is_ok();
    (!listening).then_some(path)
}

/// How long to wait for the port in a lock file to answer.
///
/// The connection is to the loopback interface of this machine, where a
/// listener answers immediately and an unused port is refused immediately.
/// The timeout is for the case neither happens, and a second of it is already
/// far longer than either.
const PROBE: Duration = Duration::from_secs(1);

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::stale_pid_file;

    /// Writes a lock file shaped the way PostgreSQL writes one.
    ///
    /// The process id is deliberately one that is always running and that this
    /// test could never legitimately signal, because that is the case the
    /// check exists to stop reaching `pg_ctl` at all.
    fn lock_file(dir: &std::path::Path, port: &str) {
        let contents = format!(
            "1\n{}\n1789118516\n{port}\n/tmp\nlocalhost\n  2024230         4\nready\n",
            dir.display()
        );
        std::fs::write(dir.join("postmaster.pid"), contents).expect("write the lock file");
    }

    /// A cluster that is running keeps its lock file.
    ///
    /// Deleting this one is the failure that matters: a second postmaster on
    /// one data directory is corruption, where the other direction is only a
    /// command that reports an error.
    #[test]
    fn a_lock_file_whose_port_answers_is_left_alone() {
        let dir = tempfile::tempdir().expect("temp data dir");
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind a port");
        let port = listener.local_addr().expect("the bound address").port();

        lock_file(dir.path(), &port.to_string());

        assert_eq!(stale_pid_file(dir.path()), None);
    }

    /// A lock file whose port answers nothing outlived its cluster.
    #[test]
    fn a_lock_file_no_cluster_is_behind_is_named_for_removal() {
        let dir = tempfile::tempdir().expect("temp data dir");
        // Bound and released, so the number is one the operating system was
        // willing to give out and nothing is behind it now.
        let port = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind a port");
            listener.local_addr().expect("the bound address").port()
        };

        lock_file(dir.path(), &port.to_string());

        assert_eq!(
            stale_pid_file(dir.path()),
            Some(dir.path().join("postmaster.pid"))
        );
    }

    /// A lock file that cannot be read for a port says nothing either way.
    ///
    /// Which is the reading that leaves it alone: a file we cannot parse is
    /// not evidence that a cluster has gone.
    #[test]
    fn a_lock_file_that_does_not_parse_is_left_alone() {
        let dir = tempfile::tempdir().expect("temp data dir");
        lock_file(dir.path(), "not a port");

        assert_eq!(stale_pid_file(dir.path()), None);

        std::fs::write(dir.path().join("postmaster.pid"), "1\n").expect("a truncated lock file");
        assert_eq!(stale_pid_file(dir.path()), None);
    }

    /// No lock file is not a stale one.
    #[test]
    fn a_data_directory_without_a_lock_file_has_nothing_to_remove() {
        let dir = tempfile::tempdir().expect("temp data dir");
        assert_eq!(stale_pid_file(dir.path()), None);
    }
}
