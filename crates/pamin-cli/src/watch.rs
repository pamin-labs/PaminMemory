//! Keeping ingested files in sync with what pamin holds of them.
//!
//! Every path `pamin ingest` is given is kept in sync unless it was told
//! `--no-sync`: the server watches it and ingests what changes under it, one
//! second after the last change, and when it starts it catches up on what
//! changed while it was not running. Both go through the ingest a person runs,
//! so what a watched change records is what `pamin ingest` would have.
//!
//! What is watched is exactly what an ingest walks: each directory the walk
//! goes into gets a watch of its own, and a directory the walk skips -- hidden,
//! ignored by git, behind a link -- gets none. A change under one is never
//! heard of, and costs nothing. On macOS, whose file events come from one
//! stream a volume rather than one watch a directory, the synced directory is
//! watched whole and a change is kept only if its directory is one the walk
//! went into.
//!
//! The operating system's event says only that something under a name moved.
//! What moved is decided from the directory as a walk would list it: a file
//! that lists is changed, one that does not is gone, and a directory that
//! lists anew is walked. A file that changed is then compared with its record
//! like any other, so an event that changed nothing costs a stat.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::Watcher as _;
use pamin_index::{ProjectionIndex, VectorIndex};
use pamin_store::repository::SyncedPath;
use time::OffsetDateTime;
use tokio::sync::mpsc;

use crate::command::ingest::{self, File, Found};
use crate::session::Session;

/// How long after the last change to a path the server ingests what changed:
/// long enough that an editor's save, or a checkout, is one ingest rather than
/// one a file, and short enough to be found by the next search.
const QUIET: Duration = Duration::from_millis(1000);

/// The longest a change waits for the changes after it to stop. A file
/// rewritten every half second would otherwise never be ingested at all.
const LONGEST: Duration = Duration::from_secs(10);

/// How often a path the operating system would not watch is looked at instead:
/// a stat a file, against its record.
const POLL: Duration = Duration::from_secs(30);

/// How often a synced path that is not there is looked for.
const MISSING: Duration = Duration::from_secs(60);

/// The longest a search with a stale hit waits for an ingest of its project
/// already running before it answers without ingesting the hit's file.
const WAIT: Duration = Duration::from_secs(1);

/// What the server is doing about each synced path, and how the command
/// handlers reach it.
pub struct Syncing {
    changes: mpsc::UnboundedSender<Change>,
    status: Mutex<HashMap<(String, String), Status>>,
}

/// How a synced path stands, for `pamin sync`.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub state: State,
    pub synced_at: Option<OffsetDateTime>,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum State {
    /// Not caught up on yet.
    #[default]
    Starting,
    Watching,
    /// The operating system would not watch it, so it is looked at every
    /// [`POLL`].
    Polling,
    /// Not there; looked for every [`MISSING`].
    Missing,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Watching => "watching",
            Self::Polling => "polling",
            Self::Missing => "missing",
        }
    }
}

/// A synced path added or removed by a command.
enum Change {
    Add(SyncedPath),
    Remove { project: String, path: String },
}

impl Syncing {
    /// Starts keeping the paths synced in this workspace in sync, in the
    /// background: the server answers while it catches up.
    pub fn start(session: Arc<Session>) -> Arc<Self> {
        let (changes, receive) = mpsc::unbounded_channel();
        let syncing = Arc::new(Self {
            changes,
            status: Mutex::default(),
        });
        let watcher = Watcher::new(Arc::clone(&session), Arc::clone(&syncing));
        tokio::spawn(watcher.run(receive));
        syncing
    }

    /// Keeps `synced` in sync from now on, catching up on it first.
    pub fn add(&self, synced: SyncedPath) {
        let _ = self.changes.send(Change::Add(synced));
    }

    /// Stops keeping `path` of `project` in sync.
    pub fn remove(&self, project: &str, path: &str) {
        let _ = self.changes.send(Change::Remove {
            project: project.to_string(),
            path: path.to_string(),
        });
    }

    /// How `path` of `project` stands.
    pub fn status(&self, project: &str, path: &str) -> Status {
        self.status
            .lock()
            .expect("the sync status is poisoned")
            .get(&(project.to_string(), path.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn update(&self, synced: &SyncedPath, update: impl FnOnce(&mut Status)) {
        let mut status = self.status.lock().expect("the sync status is poisoned");
        update(
            status
                .entry((synced.project.clone(), synced.path.clone()))
                .or_default(),
        );
    }

    fn forget(&self, project: &str, path: &str) {
        self.status
            .lock()
            .expect("the sync status is poisoned")
            .remove(&(project.to_string(), path.to_string()));
    }
}

/// A synced path, as the watcher follows it.
struct Followed {
    synced: SyncedPath,
    path: PathBuf,
    /// What its files are named from, worked out once: a file under it is
    /// named from this however the change that found it was spelled, as the
    /// ingest that synced it named it.
    root: PathBuf,
    /// Whether it is one file rather than a directory.
    file: bool,
    /// The directories a walk of it goes into: what is watched, and the only
    /// directories a change is kept from.
    dirs: HashSet<PathBuf>,
    state: State,
    /// When it was last looked at by polling, or looked for while missing.
    looked: Instant,
}

impl Followed {
    /// Reads the file system, to name the path's root: off the runtime.
    fn new(synced: SyncedPath) -> Self {
        let path = PathBuf::from(&synced.path);
        Self {
            root: ingest::root(&path),
            file: !path.is_dir(),
            path,
            synced,
            dirs: HashSet::new(),
            state: State::Starting,
            looked: Instant::now(),
        }
    }

    /// Whether a change at `path` is one this follows.
    fn covers(&self, path: &Path) -> bool {
        if self.file {
            path == self.path
        } else {
            path.starts_with(&self.path)
        }
    }

    fn named(&self, path: &Path) -> Option<File> {
        ingest::file(path, &self.root)
    }

    /// When polling or looking for it is next due.
    fn due(&self) -> Option<Instant> {
        match self.state {
            State::Polling => Some(self.looked + POLL),
            State::Missing => Some(self.looked + MISSING),
            State::Starting | State::Watching => None,
        }
    }
}

/// The watcher's state, owned by its one task.
struct Watcher {
    session: Arc<Session>,
    syncing: Arc<Syncing>,
    events: mpsc::UnboundedReceiver<notify::Result<notify::Event>>,
    os: Option<notify::RecommendedWatcher>,
    followed: Vec<Followed>,
    /// How many followed paths want each directory watched.
    watches: HashMap<PathBuf, usize>,
    /// Paths changed since the last ingest, and when the first and the last of
    /// them changed.
    pending: BTreeSet<PathBuf>,
    first: Option<Instant>,
    last: Option<Instant>,
    /// Files found gone and not yet ingested, held for one more quiet period in
    /// case the other half of a move arrives: by followed project and path.
    held: Vec<(String, String, File)>,
    held_until: Option<Instant>,
    /// Everything is caught up on at the next ingest, because the operating
    /// system lost events.
    rescan: bool,
    /// The followed directories with a recursive stream of their own, on
    /// macOS: one each, so ending one never silences another inside it.
    streams: HashSet<PathBuf>,
    /// Never watched: the workspace's own files change with every write.
    /// Canonical, as the paths a walk and an event carry are.
    workspace: PathBuf,
}

impl Watcher {
    fn new(session: Arc<Session>, syncing: Arc<Syncing>) -> Self {
        let (send, events) = mpsc::unbounded_channel();
        let os = notify::recommended_watcher(move |event| {
            let _ = send.send(event);
        });
        let os = match os {
            Ok(os) => Some(os),
            Err(error) => {
                tracing::warn!(%error, "no file watcher: synced paths are polled");
                None
            }
        };
        let workspace = session.workspace().root();
        let workspace =
            std::fs::canonicalize(workspace).unwrap_or_else(|_| workspace.to_path_buf());
        Self {
            session,
            syncing,
            events,
            os,
            followed: Vec::new(),
            watches: HashMap::new(),
            pending: BTreeSet::new(),
            first: None,
            last: None,
            held: Vec::new(),
            held_until: None,
            rescan: false,
            streams: HashSet::new(),
            workspace,
        }
    }

    async fn run(mut self, mut changes: mpsc::UnboundedReceiver<Change>) {
        match pamin_store::repository::synced_paths(self.session.database().pool()).await {
            Ok(synced) => {
                for synced in synced {
                    self.follow(synced).await;
                }
            }
            Err(error) => tracing::warn!(%error, "reading the synced paths failed"),
        }

        // Asleep until something is due: an idle server with synced paths
        // wakes for an event, a command, or a poll, and for nothing else.
        loop {
            let due = self
                .next_due()
                .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600));
            tokio::select! {
                Some(event) = self.events.recv() => self.heard(event),
                Some(change) = changes.recv() => match change {
                    Change::Add(synced) => self.follow(synced).await,
                    Change::Remove { project, path } => self.unfollow(&project, &path),
                },
                () = tokio::time::sleep_until(due.into()) => self.due().await,
            }
        }
    }

    /// The soonest of: the ingest of what changed, the held deletions, a poll,
    /// a look for a missing path.
    fn next_due(&self) -> Option<Instant> {
        let changed = match (self.first, self.last) {
            (Some(first), Some(last)) => Some((last + QUIET).min(first + LONGEST)),
            _ => None,
        };
        changed
            .into_iter()
            .chain(self.held_until)
            .chain(self.followed.iter().filter_map(Followed::due))
            .min()
    }

    /// An event from the operating system: what it names is pending.
    fn heard(&mut self, event: notify::Result<notify::Event>) {
        let event = match event {
            Ok(event) => event,
            Err(error) => {
                if matches!(error.kind, notify::ErrorKind::MaxFilesWatch) {
                    tracing::warn!("the file watch limit was reached: synced paths are polled");
                    self.poll_all_unwatched();
                } else {
                    tracing::debug!(%error, "file watcher error");
                }
                return;
            }
        };
        // Reading a file is an event too, and an ingest reads them.
        if matches!(event.kind, notify::EventKind::Access(_)) {
            return;
        }
        if event.need_rescan() {
            self.rescan = true;
        }
        let now = Instant::now();
        for path in event.paths {
            if path.starts_with(&self.workspace) {
                continue;
            }
            self.pending.insert(path);
        }
        if !self.pending.is_empty() || self.rescan {
            self.first.get_or_insert(now);
            self.last = Some(now);
        }
    }

    /// Whatever is due: an ingest of what changed, of what was held, of what
    /// is polled, or a look for what is missing.
    async fn due(&mut self) {
        let now = Instant::now();
        let changed = match (self.first, self.last) {
            (Some(first), Some(last)) => {
                now.duration_since(last) >= QUIET || now.duration_since(first) >= LONGEST
            }
            _ => false,
        };
        let held = self.held_until.is_some_and(|until| now >= until);
        if changed || held {
            self.flush().await;
        }
        for at in 0..self.followed.len() {
            if self.followed[at].due().is_some_and(|due| now >= due) {
                self.followed[at].looked = Instant::now();
                self.catch_up(at).await;
            }
        }
    }

    /// Follows a synced path: watches it, then catches up on it. In that
    /// order, so a change made while the catch-up walks is heard of. A path
    /// already followed only takes the new settings: the ingest that synced it
    /// again has just read it.
    async fn follow(&mut self, synced: SyncedPath) {
        if let Some(followed) = self.followed.iter_mut().find(|followed| {
            followed.synced.project == synced.project && followed.synced.path == synced.path
        }) {
            followed.synced = synced;
            return;
        }
        self.syncing
            .update(&synced, |status| status.state = State::Starting);
        let followed = match tokio::task::spawn_blocking(move || Followed::new(synced)).await {
            Ok(followed) => followed,
            Err(error) => {
                tracing::warn!(%error, "following a synced path failed");
                return;
            }
        };
        self.followed.push(followed);
        let at = self.followed.len() - 1;
        self.catch_up(at).await;
    }

    fn unfollow(&mut self, project: &str, path: &str) {
        let Some(at) = self.followed.iter().position(|followed| {
            followed.synced.project == project && followed.synced.path == path
        }) else {
            return;
        };
        let followed = self.followed.remove(at);
        for dir in &followed.dirs {
            self.unwatch(dir);
        }
        self.end_stream(&followed.path);
        self.held.retain(|(p, s, _)| !(p == project && s == path));
        self.syncing.forget(project, path);
    }

    /// Re-walks a followed path, watches what the walk goes into and stops
    /// watching what it no longer does, and ingests it: what changed while
    /// nobody was listening.
    async fn catch_up(&mut self, at: usize) {
        let path = self.followed[at].path.clone();
        match tokio::fs::metadata(&path).await {
            Ok(meta) => self.followed[at].file = !meta.is_dir(),
            Err(_) => {
                let followed = &mut self.followed[at];
                followed.state = State::Missing;
                followed.looked = Instant::now();
                let dirs: Vec<PathBuf> = followed.dirs.drain().collect();
                self.syncing.update(&self.followed[at].synced, |status| {
                    status.state = State::Missing
                });
                for dir in dirs {
                    self.unwatch(&dir);
                }
                self.end_stream(&path);
                return;
            }
        }
        let found = match walked(&path).await {
            Ok(found) => found,
            Err(error) => {
                self.failed(at, &error);
                return;
            }
        };
        let dirs: HashSet<PathBuf> = if self.followed[at].file {
            path.parent().map(Path::to_path_buf).into_iter().collect()
        } else {
            found.dirs.into_iter().collect()
        };
        self.watch_dirs(at, dirs);
        // Walked again by the ingest, after the watches: a file created
        // between the walk above and its directory's watch is found here.
        let found = match walked(&path).await {
            Ok(found) => found,
            Err(error) => {
                self.failed(at, &error);
                return;
            }
        };
        self.ingest(at, found, Vec::new()).await;
    }

    /// Makes `dirs` the directories watched for a followed path.
    fn watch_dirs(&mut self, at: usize, dirs: HashSet<PathBuf>) {
        let dirs: HashSet<PathBuf> = dirs
            .into_iter()
            .filter(|dir| !dir.starts_with(&self.workspace))
            .collect();
        let before = std::mem::take(&mut self.followed[at].dirs);
        for dir in before.difference(&dirs) {
            self.unwatch(dir);
        }
        let synced = self.followed[at].path.clone();
        let mut refused = false;
        let mut watched = HashSet::new();
        for dir in dirs {
            if before.contains(&dir) || self.watch(&dir, &synced) {
                watched.insert(dir);
            } else {
                refused = true;
            }
        }
        self.followed[at].dirs = watched;
        self.settle(at, refused);
    }

    /// Records whether every directory of a followed path is watched.
    fn settle(&mut self, at: usize, refused: bool) {
        let state = if refused || self.os.is_none() {
            State::Polling
        } else {
            State::Watching
        };
        self.followed[at].state = state;
        self.syncing
            .update(&self.followed[at].synced, |status| status.state = state);
    }

    /// Watches `dir` for the followed path `synced`, or counts one more path
    /// wanting it. False when the operating system refused, for any reason: a
    /// directory it would not watch is one whose changes are never heard of,
    /// and is polled instead.
    fn watch(&mut self, dir: &Path, synced: &Path) -> bool {
        let Some(os) = self.os.as_mut() else {
            return false;
        };
        // On macOS one recursive stream a followed directory, filtered by the
        // walked directories; elsewhere one watch a directory.
        if cfg!(target_os = "macos") {
            if !self.streams.contains(synced) {
                if let Err(error) = os.watch(synced, notify::RecursiveMode::Recursive) {
                    tracing::debug!(dir = %synced.display(), %error, "not watched");
                    return false;
                }
                self.streams.insert(synced.to_path_buf());
            }
            *self.watches.entry(dir.to_path_buf()).or_default() += 1;
            return true;
        }
        if let Some(count) = self.watches.get_mut(dir) {
            *count += 1;
            return true;
        }
        match os.watch(dir, notify::RecursiveMode::NonRecursive) {
            Ok(()) => {
                self.watches.insert(dir.to_path_buf(), 1);
                true
            }
            Err(error) => {
                tracing::debug!(dir = %dir.display(), %error, "not watched");
                false
            }
        }
    }

    /// Watches `dir` afresh, keeping its count: a directory deleted and made
    /// again, or replaced by a move, is a new inode its old watch never sees.
    fn rewatch(&mut self, dir: &Path) -> bool {
        if cfg!(target_os = "macos") {
            return true;
        }
        let Some(os) = self.os.as_mut() else {
            return false;
        };
        let _ = os.unwatch(dir);
        os.watch(dir, notify::RecursiveMode::NonRecursive).is_ok()
    }

    fn unwatch(&mut self, dir: &Path) {
        let Some(count) = self.watches.get_mut(dir) else {
            return;
        };
        *count -= 1;
        if *count > 0 {
            return;
        }
        self.watches.remove(dir);
        if cfg!(target_os = "macos") {
            return;
        }
        if let Some(os) = self.os.as_mut() {
            let _ = os.unwatch(dir);
        }
    }

    /// Ends the recursive stream of a followed directory nothing follows any
    /// more, on macOS.
    fn end_stream(&mut self, path: &Path) {
        let followed = self.followed.iter().any(|followed| followed.path == path);
        if followed || !self.streams.remove(path) {
            return;
        }
        if let Some(os) = self.os.as_mut() {
            let _ = os.unwatch(path);
        }
    }

    fn poll_all_unwatched(&mut self) {
        for followed in &mut self.followed {
            if followed.state == State::Watching
                && followed
                    .dirs
                    .iter()
                    .any(|dir| !self.watches.contains_key(dir))
            {
                followed.state = State::Polling;
                let synced = followed.synced.clone();
                self.syncing
                    .update(&synced, |status| status.state = State::Polling);
            }
        }
    }

    /// Ingests what changed since the last ingest, and what was held.
    async fn flush(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        self.first = None;
        self.last = None;
        let rescan = std::mem::take(&mut self.rescan);
        // Held at the last flush: applied now whatever comes with them.
        let mut held = std::mem::take(&mut self.held);
        self.held_until = None;

        for at in 0..self.followed.len() {
            if self.followed[at].state == State::Missing {
                continue;
            }
            let key = (
                self.followed[at].synced.project.clone(),
                self.followed[at].synced.path.clone(),
            );
            let mut deleted: Vec<File> = Vec::new();
            held.retain(|(project, path, file)| {
                let ours = (project, path) == (&key.0, &key.1);
                if ours {
                    deleted.push(file.clone());
                }
                !ours
            });
            let paths: Vec<PathBuf> = pending
                .iter()
                .filter(|path| self.followed[at].covers(path))
                .cloned()
                .collect();
            if rescan {
                self.catch_up(at).await;
                continue;
            }
            if paths.is_empty() && deleted.is_empty() {
                continue;
            }
            // The synced path itself gone, or a `.gitignore` changing what a
            // walk finds: walked afresh rather than sorted.
            let gone = tokio::fs::metadata(&self.followed[at].path).await.is_err();
            let ignores = paths
                .iter()
                .any(|path| path.file_name().is_some_and(|name| name == ".gitignore"));
            if gone || ignores {
                self.catch_up(at).await;
                continue;
            }
            let sorted = match self.sort(at, &paths).await {
                Ok(sorted) => sorted,
                Err(error) => {
                    self.failed(at, &error);
                    continue;
                }
            };
            // A file gone with nothing new beside it may be half a move whose
            // other half is still to come: held for one more quiet period, so
            // both reach one ingest and the section keeps its history. What
            // was held already is applied.
            if sorted.changed.is_empty() && !sorted.deleted.is_empty() {
                self.held.extend(
                    sorted
                        .deleted
                        .into_iter()
                        .map(|file| (key.0.clone(), key.1.clone(), file)),
                );
                self.held_until = Some(Instant::now() + QUIET);
                if deleted.is_empty() {
                    continue;
                }
            } else {
                deleted.extend(sorted.deleted);
            }
            let found = Found {
                files: sorted.changed,
                skipped: Vec::new(),
                trees: Vec::new(),
                dirs: Vec::new(),
            };
            self.ingest(at, found, deleted).await;
        }
    }

    /// What the changed `paths` under a followed directory came to, as a walk
    /// would list each of their directories now.
    async fn sort(&mut self, at: usize, paths: &[PathBuf]) -> anyhow::Result<Sorted> {
        let followed = &self.followed[at];
        if followed.file {
            let path = followed.path.clone();
            let mut sorted = Sorted::default();
            if let Some(file) = followed.named(&path) {
                if tokio::fs::metadata(&path).await.is_ok() {
                    sorted.changed.push(file);
                } else {
                    sorted.deleted.push(file);
                }
            }
            return Ok(sorted);
        }
        let dirs = followed.dirs.clone();
        let root = followed.root.clone();
        let paths = paths.to_vec();
        let (mut sorted, vanished) =
            tokio::task::spawn_blocking(move || sort(&paths, &dirs, &root)).await?;

        // A directory that is gone takes its files with it, and the operating
        // system says so of the directory alone: its files are what its
        // records name, and what its sections say they were cut from, for a
        // file whose record was lost.
        for dir in &vanished {
            self.forget_dirs_under(at, dir);
            sorted.deleted.extend(self.files_under(at, dir).await?);
        }
        // A directory that lists now -- new, moved in, or deleted and made
        // again -- is watched afresh and then walked, so a file written into
        // it meanwhile is found by the walk or heard of by the watch.
        for dir in std::mem::take(&mut sorted.new_dirs) {
            self.forget_dirs_under(at, &dir);
            let files = self.adopt(at, &dir).await?;
            sorted.changed.extend(files);
        }
        sorted.changed.sort();
        sorted.changed.dedup();
        sorted.deleted.sort();
        sorted.deleted.dedup();
        Ok(sorted)
    }

    /// Stops watching `dir` and every directory under it for a followed path.
    fn forget_dirs_under(&mut self, at: usize, dir: &Path) {
        let under: Vec<PathBuf> = self.followed[at]
            .dirs
            .iter()
            .filter(|watched| watched.starts_with(dir))
            .cloned()
            .collect();
        for watched in under {
            self.followed[at].dirs.remove(&watched);
            self.unwatch(&watched);
        }
    }

    /// The files of a followed path that were under the directory `dir`, now
    /// gone: those its records name, and those its sections were cut from.
    async fn files_under(&self, at: usize, dir: &Path) -> anyhow::Result<Vec<File>> {
        let Some(named) = self.followed[at].named(dir) else {
            return Ok(Vec::new());
        };
        let project = self
            .session
            .project(&self.followed[at].synced.project)
            .await?;
        let pool = self.session.database().pool();
        let prefix = format!("{}/", named.name);
        let records =
            pamin_store::repository::file_records_under(pool, project, &named.root, &prefix)
                .await?;
        let sections = pamin_store::repository::file_topics_under(
            pool,
            project,
            std::slice::from_ref(&prefix),
        )
        .await?;
        let mut files: BTreeSet<String> = records.into_iter().map(|record| record.file).collect();
        files.extend(
            sections
                .into_iter()
                .filter(|topic| topic.root.as_deref() == Some(named.root.as_str()))
                .map(|topic| ingest::file_of(&topic.name).to_string()),
        );
        Ok(files
            .into_iter()
            .map(|file| File {
                path: ingest::on_disk(&named.root, &file),
                name: file,
                root: named.root.clone(),
            })
            .collect())
    }

    /// Watches the directory `dir` under a followed path, walks it, watches
    /// what the walk went into, and walks it again: the files under it, named
    /// from the followed path's root.
    async fn adopt(&mut self, at: usize, dir: &Path) -> anyhow::Result<Vec<File>> {
        let synced = self.followed[at].path.clone();
        let mut refused = !self.watch_one(at, dir, &synced);
        let found = walked(dir).await?;
        for walked in &found.dirs {
            if !walked.starts_with(&self.workspace) && !self.followed[at].dirs.contains(walked) {
                refused |= !self.watch_one(at, walked, &synced);
            }
        }
        let found = walked(dir).await?;
        if refused {
            self.settle(at, true);
        }
        let root = self.followed[at].root.clone();
        Ok(found
            .files
            .iter()
            .filter_map(|file| ingest::file(&file.path, &root))
            .collect())
    }

    /// Watches one directory of a followed path, afresh if another followed
    /// path already counts it.
    fn watch_one(&mut self, at: usize, dir: &Path, synced: &Path) -> bool {
        let watched = if let Some(count) = self.watches.get_mut(dir) {
            *count += 1;
            self.rewatch(dir)
        } else {
            self.watch(dir, synced)
        };
        if self.watches.contains_key(dir) {
            self.followed[at].dirs.insert(dir.to_path_buf());
        }
        watched
    }

    /// Ingests `found` and `deleted` for a followed path, the way `pamin
    /// ingest` does: under the project's lock, counted as serving so index
    /// upkeep waits for it.
    async fn ingest(&mut self, at: usize, found: Found, deleted: Vec<File>) {
        let synced = self.followed[at].synced.clone();
        match ingest_for(&self.session, &synced, found, deleted).await {
            Ok(ingested) => {
                if ingested.written + ingested.removed.len() + ingested.renamed.len() > 0 {
                    tracing::info!(
                        path = %synced.path,
                        written = ingested.written,
                        removed = ingested.removed.len(),
                        renamed = ingested.renamed.len(),
                        "synced"
                    );
                }
                self.syncing.update(&synced, |status| {
                    status.synced_at = Some(OffsetDateTime::now_utc());
                    status.error = None;
                });
            }
            Err(error) => self.failed(at, &error),
        }
    }

    fn failed(&self, at: usize, error: &anyhow::Error) {
        let synced = &self.followed[at].synced;
        tracing::warn!(path = %synced.path, error = format!("{error:#}"), "syncing failed");
        self.syncing
            .update(synced, |status| status.error = Some(format!("{error:#}")));
    }
}

/// What the changes under a followed directory came to.
#[derive(Debug, Default, PartialEq, Eq)]
struct Sorted {
    /// Files to ingest: changed, created, or moved in.
    changed: Vec<File>,
    /// Files gone: deleted, moved out, or renamed away.
    deleted: Vec<File>,
    /// Directories that list now: new, moved in, or made again.
    new_dirs: Vec<PathBuf>,
}

/// Sorts changed `paths` by listing each one's directory as a walk would: a
/// file that lists is changed, a walked kind of file that no longer appears in
/// its directory at all is gone, and a directory that lists is to be walked --
/// one already walked may have been deleted and made again. A directory among `dirs` that no longer appears is returned apart,
/// for its files to be found from their records. A path in a directory the
/// walk does not go into is dropped: hidden, ignored, or outside.
///
/// Gone is decided from the directory's own listing of names rather than from
/// asking whether the path exists, which on a case-insensitive filesystem a
/// file renamed `Notes.md` to `notes.md` would answer yes to.
fn sort(paths: &[PathBuf], dirs: &HashSet<PathBuf>, root: &Path) -> (Sorted, Vec<PathBuf>) {
    let mut sorted = Sorted::default();
    let mut vanished = Vec::new();
    let mut listings: HashMap<PathBuf, (BTreeSet<PathBuf>, HashSet<std::ffi::OsString>)> =
        HashMap::new();
    for path in paths {
        let Some(parent) = path.parent() else {
            continue;
        };
        if !dirs.contains(parent) {
            continue;
        }
        let (listed, names) = listings.entry(parent.to_path_buf()).or_insert_with(|| {
            let names = std::fs::read_dir(parent)
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .map(|entry| entry.file_name())
                        .collect()
                })
                .unwrap_or_default();
            (ingest::listed(parent), names)
        });
        let present = path.file_name().is_some_and(|name| names.contains(name));
        if listed.contains(path) {
            if path.is_dir() {
                sorted.new_dirs.push(path.clone());
            } else if let Some(file) = ingest::file(path, root) {
                sorted.changed.push(file);
            }
        } else if !present {
            if dirs.contains(path) {
                vanished.push(path.clone());
            } else if let Some(file) =
                ingest::file(path, root).filter(|_| ingest::walked_kind(path))
            {
                sorted.deleted.push(file);
            }
        }
    }
    (sorted, vanished)
}

/// Ingests the files behind these search hits that changed since they were
/// last ingested, if nothing has yet: what a search is about to answer with is
/// what the file says now, even while the watcher is still catching up or has
/// not heard. What it changes counts in [`Session::ingested`], which is how the
/// search knows to ask again.
///
/// Only a file under a path kept in sync, as that path's ingest would, and
/// not one whose synced path is not there, whose sections the watcher keeps.
/// Behind another ingest of the project for at most [`WAIT`].
pub async fn freshen(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    topics: &[String],
) -> anyhow::Result<()> {
    let project_id = session.project(project).await?;
    let pool = session.database().pool();
    let synced = pamin_store::repository::synced_paths_of(pool, project_id).await?;
    if synced.is_empty() {
        return Ok(());
    }
    let roots = pamin_store::repository::file_roots_of(pool, project_id, topics).await?;
    if roots.is_empty() {
        return Ok(());
    }
    let candidates: BTreeSet<(String, String)> = roots
        .iter()
        .flat_map(|(topic, root)| {
            ingest::files_of(topic).map(move |file| (root.clone(), file.to_string()))
        })
        .collect();
    let (roots, files): (Vec<String>, Vec<String>) = candidates.into_iter().unzip();
    let records =
        pamin_store::repository::file_records_of(pool, project_id, &roots, &files).await?;
    if records.is_empty() {
        return Ok(());
    }

    // What changed, by the synced path it is under. A file gone is left to
    // the watcher, which holds a deletion for the other half of a move: a
    // search retiring it here would cost the moved section its history.
    let stale = tokio::task::spawn_blocking(move || {
        let mut stale: Vec<(SyncedPath, Vec<File>)> = Vec::new();
        for record in records {
            let path = ingest::on_disk(&record.root, &record.file);
            let Some(synced) = synced
                .iter()
                .find(|synced| path.starts_with(&synced.path) && Path::new(&synced.path).exists())
            else {
                continue;
            };
            let file = File {
                name: record.file.clone(),
                path: path.clone(),
                root: record.root.clone(),
            };
            match ingest::stat(&path) {
                Ok(stat) if !ingest::trusted(&record, &stat) => {
                    match stale.iter_mut().find(|(s, _)| s.path == synced.path) {
                        Some((_, changed)) => changed.push(file),
                        None => stale.push((synced.clone(), vec![file])),
                    }
                }
                _ => {}
            }
        }
        stale
    })
    .await?;
    if stale.is_empty() {
        return Ok(());
    }

    // An ingest already running is most often the watcher on one file, which
    // is over in well under a second; one that is not -- a catch-up of a
    // large tree -- is not waited out.
    let ingesting = session.ingesting(project);
    let Ok(_ingesting) = tokio::time::timeout(WAIT, ingesting.lock()).await else {
        return Ok(());
    };
    for (synced, changed) in stale {
        let found = Found {
            files: changed,
            skipped: Vec::new(),
            trees: Vec::new(),
            dirs: Vec::new(),
        };
        ingest::ingest(
            session,
            project,
            vector_index,
            found,
            Vec::new(),
            synced.keep_missing,
        )
        .await?;
    }
    Ok(())
}

/// What a walk of `path` finds, walked off the runtime.
async fn walked(path: &Path) -> anyhow::Result<Found> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || ingest::files(std::slice::from_ref(&path))).await?
}

/// Ingests for a synced path, under the vector index its project's index was
/// built with, or the one the ingest that synced it asked for.
async fn ingest_for(
    session: &Session,
    synced: &SyncedPath,
    found: Found,
    deleted: Vec<File>,
) -> anyhow::Result<pamin_client::command::ingest::Ingested> {
    let project = session.project(&synced.project).await?;
    let dir = session.workspace().index_dir(project);
    let built = tokio::task::spawn_blocking(move || ProjectionIndex::built_for(&dir)).await??;
    let vector_index = match built {
        Some(built) => built,
        None => VectorIndex::parse(&synced.vector_index)
            .ok_or_else(|| anyhow::anyhow!("unknown vector index {:?}", synced.vector_index))?,
    };
    let ingesting = session.ingesting(&synced.project);
    let _ingesting = ingesting.lock().await;
    let _serving = session.serving();
    ingest::ingest(
        session,
        &synced.project,
        vector_index,
        found,
        deleted,
        synced.keep_missing,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    fn names(files: &[File]) -> Vec<&str> {
        files.iter().map(|file| file.name.as_str()).collect()
    }

    /// A tree with a git root, a doc, an ignored directory and a hidden one.
    fn tree() -> (tempfile::TempDir, PathBuf, HashSet<PathBuf>) {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("repo");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        put(&root.join(".gitignore"), "target/\n");
        put(&root.join("docs/a.md"), "# A");
        put(&root.join("target/doc.md"), "# Built");
        put(&root.join(".hidden/b.md"), "# B");
        let found = ingest::files(std::slice::from_ref(&root)).unwrap();
        let dirs = found.dirs.into_iter().collect();
        (tree, root, dirs)
    }

    /// What is watched is what the walk goes into: an ignored or a hidden
    /// directory is not, nor is `.git`.
    #[test]
    fn only_the_directories_a_walk_goes_into_are_watched() {
        let (_tree, root, dirs) = tree();
        let mut dirs: Vec<&Path> = dirs.iter().map(PathBuf::as_path).collect();
        dirs.sort();
        assert_eq!(dirs, [root.as_path(), root.join("docs").as_path()]);
    }

    /// A changed file that lists is changed; one that is gone is gone; a new
    /// directory is walked; a change in an ignored or hidden directory, or to
    /// a file of a kind a walk does not read, is dropped.
    #[test]
    fn a_change_is_sorted_by_listing_its_directory() {
        let (_tree, root, dirs) = tree();
        put(&root.join("docs/a.md"), "# A again");
        put(&root.join("docs/new.md"), "# New");
        put(&root.join("docs/image.png"), "png");
        put(&root.join("guides/c.md"), "# C");
        put(&root.join("target/doc.md"), "# Built again");
        put(&root.join(".hidden/b.md"), "# B again");
        let paths = [
            "docs/a.md",
            "docs/new.md",
            "docs/image.png",
            "docs/deleted.md",
            "guides",
            "target/doc.md",
            ".hidden/b.md",
        ]
        .map(|path| root.join(path));
        let (sorted, vanished) = sort(&paths, &dirs, &root);
        assert_eq!(
            names(&sorted.changed),
            ["repo/docs/a.md", "repo/docs/new.md"]
        );
        assert_eq!(names(&sorted.deleted), ["repo/docs/deleted.md"]);
        assert_eq!(sorted.new_dirs, [root.join("guides")]);
        assert!(vanished.is_empty());
    }

    /// A walked directory that is gone is returned apart, for its files to be
    /// found from their records; a file renamed by case alone is the old name
    /// gone and the new one changed.
    #[test]
    fn a_directory_gone_is_apart_and_a_renamed_file_is_gone_and_new() {
        let (_tree, root, dirs) = tree();
        std::fs::rename(root.join("docs/a.md"), root.join("docs/A.md")).unwrap();
        let paths = [root.join("docs/a.md"), root.join("docs/A.md")];
        let (sorted, _) = sort(&paths, &dirs, &root);
        assert_eq!(names(&sorted.changed), ["repo/docs/A.md"]);
        assert_eq!(names(&sorted.deleted), ["repo/docs/a.md"]);

        std::fs::remove_dir_all(root.join("docs")).unwrap();
        let (sorted, vanished) = sort(&[root.join("docs")], &dirs, &root);
        assert_eq!(vanished, [root.join("docs")]);
        assert_eq!(sorted, Sorted::default());
    }

    /// A directory already walked that an event names again -- deleted and
    /// made again, or replaced by a move -- is walked afresh, since its old
    /// watch was on an inode that is gone.
    #[test]
    fn a_directory_made_again_is_walked_again() {
        let (_tree, root, dirs) = tree();
        std::fs::remove_dir_all(root.join("docs")).unwrap();
        put(&root.join("docs/a.md"), "# A made again");
        let (sorted, vanished) = sort(&[root.join("docs")], &dirs, &root);
        assert_eq!(sorted.new_dirs, [root.join("docs")]);
        assert!(vanished.is_empty());
    }

    /// A followed path covers what is under it by whole components, and a
    /// file followed alone covers itself only.
    #[test]
    fn a_followed_path_covers_what_is_under_it() {
        let synced = |path: &str| SyncedPath {
            project: "p".into(),
            path: path.into(),
            keep_missing: false,
            vector_index: "memory".into(),
        };
        let dir = tempfile::tempdir().unwrap();
        let docs = dir.path().join("docs");
        put(&docs.join("a.md"), "# A");
        let followed = Followed::new(synced(docs.to_str().unwrap()));
        assert!(followed.covers(&docs.join("a.md")));
        assert!(followed.covers(&docs.join("deep/b.md")));
        assert!(!followed.covers(&dir.path().join("docsier/a.md")));
        let file = Followed::new(synced(docs.join("a.md").to_str().unwrap()));
        assert!(file.file);
        assert!(file.covers(&docs.join("a.md")));
        assert!(!file.covers(&docs.join("b.md")));
    }
}
