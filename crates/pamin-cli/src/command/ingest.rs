//! `pamin ingest` — record files as memories, one per section.
//!
//! What `pamin import` is for memories somebody has already written, this is
//! for the files they were written in. A file is cut into sections by
//! [`pamin_index::sections`], and each section is recorded the way an import
//! records a memory: against one open engine, several topics at once, with the
//! projection catching up in rounds.
//!
//! Re-ingesting is the ordinary case. A file whose metadata is what its record
//! says it was when it was last read is not read at all, and in a file that is,
//! a section that says what its memory already does is left alone -- no
//! evidence, no version, no work for the projection -- so ingesting an
//! unchanged tree costs a stat per file and a changed one costs what changed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pamin_client::command::ingest::{Args, Ingested, Renamed};
use pamin_engine::{Owed, Retired};
use pamin_index::VectorIndex;
use pamin_index::sections::{SEPARATOR, Section, Unnamed, named, number, pairs, parents, unnamed};
use pamin_index::spans::TokenCount;
use pamin_store::repository::{FileRecord, FileTopic};

use crate::session::Session;

/// The extensions read from a directory. A file named on the command line is
/// read whatever it is called.
const EXTENSIONS: [&str; 3] = ["md", "markdown", "txt"];

/// How many sections to record between checks on the queue. See
/// `import::BETWEEN_CHECKS`, which this is for the same reason.
const BETWEEN_CHECKS: usize = 1_000;

/// The floor on how many sections are recorded at once. See
/// `import::LEAST_AT_ONCE`.
const LEAST_AT_ONCE: usize = 2;

/// A file to ingest, and the name its sections go by.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct File {
    /// Its path from the root of the repository it is in, or from what was
    /// named on the command line when it is in none, with `/` between
    /// components -- and the root's own name first, so that two repositories'
    /// `README.md` are two files and not one.
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    /// The directory it is named from, as a path.
    pub(crate) root: String,
}

/// A directory an ingest walked.
#[derive(Debug)]
pub(crate) struct Tree {
    /// Named as a file under it is named, up to it.
    name: String,
    /// The directory its files are named from, as a path.
    root: String,
    /// Whether the walk read all of it. One that met an error cannot say a
    /// file is gone from it rather than unreachable, and retires nothing.
    whole: bool,
}

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Ingested> {
    let found = files(&args.paths)?;
    let ingested = {
        let ingesting = session.ingesting(project);
        let _ingesting = ingesting.lock().await;
        ingest(
            session,
            project,
            vector_index,
            found,
            Vec::new(),
            args.keep_missing,
        )
        .await?
    };
    keep_in_sync(session, project, vector_index, &args).await?;
    Ok(ingested)
}

/// Keeps the paths an ingest was given in sync from now on, or stops keeping
/// them under `--no-sync`. Each as it resolves, which is how the server watches
/// it and how `pamin sync` lists it.
async fn keep_in_sync(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: &Args,
) -> Result<()> {
    let mut paths = Vec::new();
    for path in &args.paths {
        if let Some(path) = tokio::fs::canonicalize(path)
            .await
            .ok()
            .and_then(|path| path.to_str().map(str::to_string))
        {
            paths.push(path);
        }
    }
    let project_id = session.project(project).await?;
    let pool = session.database().pool();
    if args.no_sync {
        for path in pamin_store::repository::unsync_paths(pool, project_id, &paths).await? {
            if let Some(syncing) = session.syncing() {
                syncing.remove(project, &path);
            }
        }
        return Ok(());
    }
    let label = vector_index.label();
    pamin_store::repository::sync_paths(pool, project_id, &paths, args.keep_missing, label).await?;
    if let Some(syncing) = session.syncing() {
        for path in paths {
            syncing.add(pamin_store::repository::SyncedPath {
                project: project.to_string(),
                path,
                keep_missing: args.keep_missing,
                vector_index: label.to_string(),
            });
        }
    }
    Ok(())
}

/// Ingests what was found, and takes `deleted` -- files known to be gone, by
/// whoever noticed -- as files read and found empty, so their sections are
/// retired, or renamed to a section found under a new name elsewhere in the
/// same ingest.
///
/// The caller holds the project's [`Session::ingesting`] lock: two ingests of
/// one project interleaving their writes could leave a record trusting text
/// the other one replaced.
pub(crate) async fn ingest(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    found: Found,
    deleted: Vec<File>,
    keep_missing: bool,
) -> Result<Ingested> {
    let project_id = session.project(project).await?;
    let database = session.database();
    let pool = database.pool();
    let started = time::OffsetDateTime::now_utc();

    // What each file looked like when it was last read, so one that has not
    // changed since is left unread.
    let (roots, names): (Vec<String>, Vec<String>) = found
        .files
        .iter()
        .map(|file| (file.root.clone(), file.name.clone()))
        .unzip();
    let mut records: std::collections::HashMap<String, Vec<FileRecord>> = Default::default();
    for record in pamin_store::repository::file_records_of(pool, project_id, &roots, &names).await?
    {
        records.entry(record.file.clone()).or_default().push(record);
    }

    // Stat, read, hash and cut off the runtime: every one of them is blocking
    // file work or CPU, and a tree is thousands of them. Cut by BGE-M3's own
    // tokenizer, which is the reranker's vocabulary too: both are XLM-R's. The
    // one the server's models keep, so it is read from disk once a process
    // rather than once an ingest, and not at all for an ingest that read
    // nothing.
    let models = session.models().clone();
    let files = found.files;
    let Looking {
        read: cut,
        unchanged: unchanged_files,
        refreshed,
        unread,
        unrecorded,
        mut skipped,
    } = tokio::task::spawn_blocking(move || look(files, &records, &models))
        .await
        .context("reading files")??;
    let gone_files: Vec<(String, String)> = deleted
        .iter()
        .map(|file| (file.root.clone(), file.name.clone()))
        .collect();
    let files_read = cut.len();
    let mut cut = cut;
    cut.extend(deleted.into_iter().map(|file| (file, None, Vec::new())));
    skipped.extend(found.skipped);
    let trees = found.trees;

    // The sections of a file are shared by every tree that names a file alike,
    // so the records other trees keep of these files stop being true the
    // moment this ingest writes: they go first, before anything is written,
    // so no failure part way leaves one trusting text it no longer holds.
    let (roots, names): (Vec<String>, Vec<String>) = cut
        .iter()
        .map(|(file, _, _)| (file.root.clone(), file.name.clone()))
        .unzip();
    pamin_store::repository::forget_file_records(pool, project_id, &roots, &names, true).await?;

    // Only the sections of the files in play: those read, and those of files
    // under a directory walked that the walk no longer found. Those are the
    // files its records name and the walk did not find -- unless a file under
    // it had no record, which an ingest from before records existed may have
    // left with sections in files since deleted, and then every section under
    // the directory is looked at.
    let found_files: std::collections::HashSet<(String, String)> = cut
        .iter()
        .map(|(file, _, _)| (file.root.clone(), file.name.clone()))
        .chain(
            unchanged_files
                .iter()
                .map(|(name, root, _)| (root.clone(), name.clone())),
        )
        .chain(
            unread
                .iter()
                .map(|(name, root)| (root.clone(), name.clone())),
        )
        .collect();
    let left = left_trees(database, project_id, &trees, &found_files).await?;
    let mut prefixes: Vec<String> = cut.iter().map(|(file, _, _)| file.name.clone()).collect();
    for (tree, (left, accounted)) in trees.iter().zip(&left) {
        let prefix = format!("{}/", tree.name);
        let unknown = unrecorded
            .iter()
            .any(|(name, root)| root == &tree.root && under(name, &tree.name))
            || pamin_store::repository::count_file_topics_under(pool, project_id, &prefix).await?
                > *accounted;
        if unknown {
            prefixes.push(prefix);
        } else {
            prefixes.extend(left.iter().map(|(_, file)| file.clone()));
        }
    }
    let recorded = in_play(
        pamin_store::repository::file_topics_under(pool, project_id, &prefixes).await?,
        &cut.iter()
            .map(|(file, _, _)| file.name.as_str())
            .collect::<Vec<_>>(),
        &trees,
    );

    let unchanged_sections: usize = unchanged_files.iter().map(|(_, _, n)| n).sum();
    let unchanged_files: Vec<(String, String)> = unchanged_files
        .into_iter()
        .map(|(name, root, _)| (name, root))
        .collect();
    let (cut, parents) = numbered(database, project_id, cut, &recorded).await?;

    let read: Vec<(String, String)> = cut
        .iter()
        .map(|read| (read.file.name.clone(), read.file.root.clone()))
        .collect();
    let sections_of: Vec<(&Section, &str)> = cut
        .iter()
        .flat_map(|read| {
            read.sections
                .iter()
                .map(move |section| (section, read.file.root.as_str()))
        })
        .collect();
    let known: std::collections::HashSet<String> =
        recorded.iter().map(|topic| topic.name.clone()).collect();
    // What each section on the surface said when this ingest looked, for one
    // the filter now holds: see the retiring below.
    let surfaced: std::collections::HashMap<String, pamin_core::TopicStateId> = recorded
        .iter()
        .map(|topic| (topic.name.clone(), topic.current))
        .collect();
    let mut gone = gone(
        &Looked {
            read: &read,
            unchanged: &unchanged_files,
            unread: &unread,
            trees: &trees,
            exists: &|path: &Path| path.exists(),
        },
        &sections_of
            .iter()
            .map(|(section, _)| *section)
            .collect::<Vec<_>>(),
        recorded,
    );

    let mut outcome = Outcome::default();
    // Nothing read and nothing to retire: no engine, no index, no models. A
    // tree whose every file is unchanged ends here, having read none of them.
    let retiring = !keep_missing && gone.iter().any(|gone| gone.retirable);
    if sections_of.is_empty() && !retiring {
        outcome.missing = gone.into_iter().map(|gone| gone.name).collect();
    } else {
        let engine = session.engine(project, vector_index).await?;
        outcome = record(
            &engine,
            &sections_of,
            &parents,
            &known,
            &surfaced,
            &mut gone,
            keep_missing,
        )
        .await?;
    }
    outcome.removed.sort();
    outcome.missing.sort();
    let cut_sections = sections_of.len();
    drop(sections_of);

    // Last, once everything above has succeeded: a record says a file's
    // sections say what it does, which is only true now. Not for a file one of
    // whose sections changed under this ingest, which is read again next time.
    // Records of files that left a tree walked whole go with them; a walk that
    // met an error cannot say a file left rather than went out of reach.
    // Nor for a file with a section this ingest left on the surface although
    // the file no longer has it -- under --keep-missing, or written by hand --
    // which an ingest without the flag must find again: a file read keeps no
    // record, and a file gone keeps the one it has.
    let unsettled = |file: &str| {
        outcome
            .changed
            .iter()
            .chain(&outcome.missing)
            .any(|name| belongs(name, file))
    };
    let mut left: Vec<(String, String)> = trees
        .iter()
        .zip(left)
        .filter(|(tree, _)| tree.whole)
        .flat_map(|(_, (left, _))| left)
        .chain(gone_files)
        .filter(|(_, file)| !unsettled(file))
        .collect();
    left.sort();
    left.dedup();
    let settled: Vec<Read> = cut
        .into_iter()
        .filter(|read| !unsettled(&read.file.name))
        .collect();
    // Each settled file by its sections' names, for the check below.
    let (mut roots, mut names, mut topics) = (Vec::new(), Vec::new(), Vec::new());
    for read in &settled {
        for section in &read.sections {
            roots.push(read.file.root.clone());
            names.push(read.file.name.clone());
            topics.push(section.name.clone());
        }
    }
    let mut records = tokio::task::spawn_blocking(move || still_as_read(settled))
        .await
        .context("checking files after recording them")?;
    records.extend(refreshed);
    write_records(database, project_id, &records, &left).await?;
    // A section written by hand while this ingest recorded its file forgot a
    // record that did not exist yet, and the one just written would trust the
    // file over it. Asked of every hand write since a minute before this
    // ingest began, the margin covering a write whose clock started first.
    pamin_store::repository::forget_file_records_written_since(
        pool,
        project_id,
        &roots,
        &names,
        &topics,
        started - time::Duration::minutes(1),
    )
    .await?;
    if outcome.written + outcome.removed.len() + outcome.renamed.len() > 0 {
        session.changed_by_ingest(project);
    }

    Ok(Ingested {
        files: files_read + unchanged_files.len(),
        sections: cut_sections + unchanged_sections,
        written: outcome.written,
        unchanged: outcome.unchanged + unchanged_sections,
        held: outcome.held,
        removed: outcome.removed,
        missing: outcome.missing,
        renamed: outcome.renamed,
        skipped,
        cascade: outcome.cascade.unwrap_or("applied").to_string(),
        cascade_lagging: outcome.lagging,
    })
}

/// How far a timestamp may trail the read that took it and still be trusted.
///
/// A file modified within its filesystem's timestamp granularity of being read
/// may change again with no change to its stat, so a record taken that close
/// is checked by its text next time, the way git treats a "racily clean"
/// entry. Two seconds covers the coarsest common granularity, FAT's.
const RACY_NS: i64 = 2_000_000_000;

/// A file read and cut, with what its record will say once its sections are
/// recorded -- or a file known to be gone, which has no sections and no
/// record to write.
struct Read {
    file: File,
    record: Option<FileRecord>,
    sections: Vec<Section>,
}

/// What looking at the found files came to.
struct Looking {
    /// Read and cut into sections, unnamed until [`numbered`] names them.
    read: Vec<(File, Option<FileRecord>, Vec<Unnamed>)>,
    /// Left unread, their record trusted: name, root, and how many sections
    /// they held.
    unchanged: Vec<(String, String, usize)>,
    /// Read, and found to say what their record does: the record with the
    /// file's new stat, nothing else to do.
    refreshed: Vec<FileRecord>,
    /// Found and not read: their sections are not missing, only unread.
    unread: Vec<(String, String)>,
    /// Read for want of a record, by name and root.
    unrecorded: Vec<(String, String)>,
    skipped: Vec<String>,
}

/// Each file's stat, against its record: trusted, read and found the same,
/// or read and cut.
fn look(
    files: Vec<File>,
    records: &std::collections::HashMap<String, Vec<FileRecord>>,
    models: &pamin_engine::Models,
) -> Result<Looking> {
    let mut looking = Looking {
        read: Vec::new(),
        unchanged: Vec::new(),
        refreshed: Vec::new(),
        unread: Vec::new(),
        unrecorded: Vec::new(),
        skipped: Vec::new(),
    };
    let mut counter = None;
    for file in files {
        let record = records
            .get(&file.name)
            .and_then(|records| records.iter().find(|record| record.root == file.root));
        let (stat, text) = match stat(&file.path).and_then(|stat| {
            if record.is_some_and(|record| trusted(record, &stat)) {
                return Ok((stat, None));
            }
            let text = String::from_utf8(std::fs::read(&file.path)?)
                .map_err(|_| std::io::Error::other("not UTF-8"))?;
            Ok((stat, Some(text)))
        }) {
            Ok(looked) => looked,
            Err(error) => {
                looking
                    .skipped
                    .push(format!("{}: {error}", file.path.display()));
                looking.unread.push((file.name, file.root));
                continue;
            }
        };
        let Some(text) = text else {
            let sections = record.map_or(0, |record| record.sections as usize);
            looking.unchanged.push((file.name, file.root, sections));
            continue;
        };
        if record.is_none() {
            looking
                .unrecorded
                .push((file.name.clone(), file.root.clone()));
        }
        let hash = pamin_engine::content_hash(&text);
        let mut fresh = stat.recorded(&file, hash);
        if let Some(record) = record
            .filter(|record| record.content_hash == fresh.content_hash && record.build == build())
        {
            fresh.sections = record.sections;
            looking
                .unchanged
                .push((file.name, file.root, record.sections as usize));
            looking.refreshed.push(fresh);
            continue;
        }
        let counter = match &counter {
            Some(counter) => counter,
            None => counter.insert(models.span_tokens()?),
        };
        let counter: &dyn TokenCount = counter.as_ref();
        let sections = unnamed(&file.name, &text, counter);
        looking.read.push((file, Some(fresh), sections));
    }
    Ok(looking)
}

/// The server build cutting files, as a record keeps it: a file is cut by its
/// build's rules, and one left unread is left on the cut it was given, so a
/// new build reads every file once, whether or not it changed how files are
/// cut.
fn build() -> &'static str {
    static BUILD: std::sync::LazyLock<String> =
        std::sync::LazyLock::new(pamin_client::protocol::version);
    &BUILD
}

/// A file's stat as a record keeps it, and when it was taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stat {
    size: u64,
    modified_ns: i64,
    changed_ns: i64,
    inode: u64,
    read_ns: i64,
}

impl Stat {
    fn recorded(&self, file: &File, content_hash: String) -> FileRecord {
        FileRecord {
            root: file.root.clone(),
            file: file.name.clone(),
            size: self.size,
            modified_ns: self.modified_ns,
            changed_ns: self.changed_ns,
            inode: self.inode,
            read_ns: self.read_ns,
            content_hash,
            sections: 0,
            build: build().to_string(),
        }
    }

    fn same_file(&self, record: &FileRecord) -> bool {
        self.size == record.size
            && self.modified_ns == record.modified_ns
            && self.changed_ns == record.changed_ns
            && self.inode == record.inode
    }
}

/// The stat of `path`, following a link as reading it does, taken after the
/// clock is read: a record must never claim a read later than its stat.
pub(crate) fn stat(path: &Path) -> std::io::Result<Stat> {
    use std::os::unix::fs::MetadataExt as _;
    let read_ns = now_ns();
    let meta = std::fs::metadata(path)?;
    Ok(Stat {
        size: meta.size(),
        modified_ns: meta.mtime() * 1_000_000_000 + meta.mtime_nsec(),
        changed_ns: meta.ctime() * 1_000_000_000 + meta.ctime_nsec(),
        inode: meta.ino(),
        read_ns,
    })
}

fn now_ns() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_nanos()).unwrap_or(i64::MAX)
        })
}

/// Whether `record` can be believed about a file whose stat is `stat` without
/// reading it: the same stat, taken long enough after the file last changed
/// that a change within the same timestamp would have shown.
pub(crate) fn trusted(record: &FileRecord, stat: &Stat) -> bool {
    record.build == build()
        && stat.same_file(record)
        && record.modified_ns.max(record.changed_ns) < record.read_ns.saturating_sub(RACY_NS)
}

/// What recording the read files' sections and retiring the gone ones did.
#[derive(Default)]
struct Outcome {
    written: usize,
    unchanged: usize,
    held: usize,
    removed: Vec<String>,
    missing: Vec<String>,
    renamed: Vec<Renamed>,
    /// Sections whose topic changed under this ingest, so it left them be.
    changed: Vec<String>,
    lagging: bool,
    cascade: Option<&'static str>,
}

/// Records the read files' sections against `engine`, renames the gone ones
/// found under a new name, and retires the rest.
async fn record(
    engine: &pamin_engine::Engine,
    cut: &[(&Section, &str)],
    parents: &[(String, String)],
    known: &std::collections::HashSet<String>,
    surfaced: &std::collections::HashMap<String, pamin_core::TopicStateId>,
    gone: &mut Vec<Gone>,
    keep_missing: bool,
) -> Result<Outcome> {
    // Before the sections are recorded, so a section found under a new name
    // is written to the topic it already was rather than to a new one.
    let mut outcome = Outcome {
        renamed: renamed(engine, cut, known, gone).await?,
        ..Outcome::default()
    };

    // Every section is a topic of its own -- its name holds its file's path,
    // and one file's names are numbered apart -- so they all run at once, in
    // the bound `import` derives from the pool for the same reason.
    let at_once = usize::try_from(engine.database.pool().options().get_max_connections())
        .unwrap_or(LEAST_AT_ONCE)
        .max(LEAST_AT_ONCE);

    // The sections on the surface that the filter now holds, cut down to next
    // to nothing in their file.
    let mut emptied = Vec::new();
    // The sections found unchanged, by the root they were read from: each
    // records that root, in one statement a root, after the writes.
    let mut kept_at: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
    let mut since_check = 0;
    for chunk in cut.chunks(at_once) {
        let recorded = futures::future::try_join_all(chunk.iter().map(|(section, root)| {
            engine.remember_section(
                &section.name,
                &section.content,
                pamin_core::Validity::ALWAYS,
                root,
            )
        }))
        .await?;

        for (recorded, (section, root)) in recorded.into_iter().zip(chunk) {
            match recorded {
                None => {
                    outcome.unchanged += 1;
                    kept_at
                        .entry(*root)
                        .or_default()
                        .push(pamin_engine::file_locator(&section.name));
                }
                Some((verdict, _)) if verdict.is_promoted() => outcome.written += 1,
                Some(_) => {
                    outcome.held += 1;
                    if let Some(&current) = surfaced.get(&section.name) {
                        emptied.push((section.name.clone(), current));
                    }
                }
            }
        }

        since_check += chunk.len();
        if since_check >= BETWEEN_CHECKS {
            since_check = 0;
            let behind = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
            if !pamin_core::may_defer(behind) {
                outcome.lagging = true;
                engine.drain_cascade(Owed::WhatAMemoryNeeds).await?;
            }
        }
    }

    for (root, locators) in &kept_at {
        pamin_store::repository::set_roots_at(
            engine.database.pool(),
            engine.project,
            locators,
            root,
        )
        .await?;
    }

    // After the sections are recorded, so each has the state its edge cites,
    // and before retiring, whose sections no section sits under any more.
    let sections: Vec<String> = cut
        .iter()
        .map(|(section, _)| section.name.clone())
        .collect();
    engine.restate_parents(&sections, parents).await?;

    // After the sections are recorded, so a section that moved within a file
    // is written under its new name before its old one is retired, and before
    // the drain, which removes what retiring left for the projection.
    for gone in gone.iter() {
        if keep_missing || !gone.retirable {
            outcome.missing.push(gone.name.clone());
            continue;
        }
        match engine.retire_section(&gone.name, gone.current).await? {
            Retired::Retired => outcome.removed.push(gone.name.clone()),
            Retired::Kept => outcome.missing.push(gone.name.clone()),
            Retired::Changed => {
                outcome.missing.push(gone.name.clone());
                outcome.changed.push(gone.name.clone());
            }
            Retired::Absent => {}
        }
    }
    // A held write leaves the topic's state as it was, so a section its file
    // emptied would still be found by the text the file no longer has. Taken
    // off the surface as a section the file lost is, and back on it once the
    // file says something again -- under --keep-missing too, since the file
    // still has the section and only says next to nothing in it. One somebody
    // also wrote by hand stays, the held count saying so.
    for (name, current) in emptied {
        match engine.retire_section(&name, current).await? {
            Retired::Retired => outcome.removed.push(name),
            Retired::Changed => outcome.changed.push(name),
            Retired::Kept | Retired::Absent => {}
        }
    }

    engine.drain_cascade(Owed::WhatAMemoryNeeds).await?;
    engine.flush_what_is_applied().await?;
    let owed = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
    outcome.cascade = Some(if owed == 0 { "applied" } else { "queued" });
    Ok(outcome)
}

/// The records of the read files, for those whose stat is still what it was
/// before they were read: one that changed while it was being recorded is
/// left without one, so the next ingest reads it again.
fn still_as_read(read: Vec<Read>) -> Vec<FileRecord> {
    read.into_iter()
        .filter_map(|read| {
            let mut record = read.record?;
            let now = stat(&read.file.path).ok()?;
            record.sections = u32::try_from(read.sections.len()).unwrap_or(u32::MAX);
            now.same_file(&record).then_some(record)
        })
        .collect()
}

/// For each tree, the files its records name that the walk did not find --
/// files that left it, by root and name -- and how many sections its records
/// account for.
async fn left_trees(
    database: &pamin_store::Database,
    project: pamin_core::ProjectId,
    trees: &[Tree],
    found: &std::collections::HashSet<(String, String)>,
) -> Result<Vec<(Vec<(String, String)>, i64)>> {
    let mut left = Vec::with_capacity(trees.len());
    for tree in trees {
        let prefix = format!("{}/", tree.name);
        let records = pamin_store::repository::file_records_under(
            database.pool(),
            project,
            &tree.root,
            &prefix,
        )
        .await?;
        let accounted = records
            .iter()
            .map(|record| i64::from(record.sections))
            .sum();
        let gone = records
            .into_iter()
            .map(|record| (record.root, record.file))
            .filter(|key| !found.contains(key))
            .collect();
        left.push((gone, accounted));
    }
    Ok(left)
}

/// Whether the file named `name` is under the directory named `tree`.
fn under(name: &str, tree: &str) -> bool {
    name.strip_prefix(tree)
        .is_some_and(|rest| rest.starts_with('/'))
}

/// Writes the records of the files read and forgets those of files gone.
async fn write_records(
    database: &pamin_store::Database,
    project: pamin_core::ProjectId,
    records: &[FileRecord],
    left: &[(String, String)],
) -> Result<()> {
    let (roots, files): (Vec<String>, Vec<String>) = left.iter().cloned().unzip();
    pamin_store::repository::forget_file_records(database.pool(), project, &roots, &files, false)
        .await?;
    pamin_store::repository::put_file_records(database.pool(), project, records).await?;
    Ok(())
}

/// The topics of `recorded` that belong to one of `files` or sit under one of
/// `trees`: a prefix finds a file's sections and also those of any file whose
/// name starts with its own.
fn in_play(recorded: Vec<FileTopic>, files: &[&str], trees: &[Tree]) -> Vec<FileTopic> {
    let files: std::collections::HashSet<&str> = files.iter().copied().collect();
    recorded
        .into_iter()
        .filter(|topic| {
            files_of(&topic.name).any(|file| files.contains(file))
                || trees.iter().any(|tree| under(&topic.name, &tree.name))
        })
        .collect()
}

/// Every name a file could have for `name` to be one of its sections' names:
/// the whole of it, or what comes before a heading or a number. Each one
/// [`belongs`] confirms.
pub(crate) fn files_of(name: &str) -> impl Iterator<Item = &str> {
    name.match_indices(SEPARATOR)
        .chain(name.match_indices(" #"))
        .map(|(at, _)| &name[..at])
        .chain(std::iter::once(name))
        .filter(move |file| belongs(name, file))
}

/// The sections of `cut` that are ones in `gone` under a new name -- a file
/// moved, or a heading above them renamed -- renamed to it, keeping their
/// topics, and taken out of `gone`.
///
/// A section is the one it was by the rule that keeps a twin's number: the
/// same text first, then most of its words in order, one to one. Only a new
/// name, which no earlier ingest recorded, and only a section this ingest
/// could retire: one it cannot say is gone is not one it can say moved.
async fn renamed(
    engine: &pamin_engine::Engine,
    cut: &[(&Section, &str)],
    known: &std::collections::HashSet<String>,
    gone: &mut Vec<Gone>,
) -> Result<Vec<Renamed>> {
    let new: Vec<&Section> = cut
        .iter()
        .map(|(section, _)| *section)
        .filter(|section| !known.contains(&section.name))
        .collect();
    let candidates: Vec<&Gone> = gone.iter().filter(|gone| gone.retirable).collect();
    if new.is_empty() || candidates.is_empty() {
        return Ok(Vec::new());
    }
    let names: Vec<String> = candidates.iter().map(|gone| gone.name.clone()).collect();
    let said: std::collections::HashMap<String, String> =
        pamin_store::repository::current_named(engine.database.pool(), engine.project, &names)
            .await?
            .into_iter()
            .map(|named| (named.name, named.content))
            .collect();
    let candidates: Vec<(&Gone, &str)> = candidates
        .into_iter()
        .filter_map(|gone| Some((gone, said.get(&gone.name)?.as_str())))
        .collect();
    let now: Vec<&str> = new.iter().map(|section| section.content.as_str()).collect();
    let before: Vec<&str> = candidates.iter().map(|(_, content)| *content).collect();

    let mut renamed = Vec::new();
    for (at, from) in pairs(&now, &before) {
        let (gone, _) = candidates[from];
        if engine
            .rename_section(&gone.name, &new[at].name, gone.current)
            .await?
        {
            renamed.push(Renamed {
                from: gone.name.clone(),
                to: new[at].name.clone(),
            });
        }
    }
    gone.retain(|gone| !renamed.iter().any(|renamed| renamed.from == gone.name));
    Ok(renamed)
}

/// Each file's sections named, numbered against what its sections said when
/// it was last ingested wherever a number is at stake: in a file where two
/// sections share headings, or where an earlier ingest numbered one. Every
/// other file's sections are named as they are cut, with no read.
///
/// And the section each sits under, by those names: the `part_of` edges
/// [`Engine::restate_parents`](pamin_engine::Engine::restate_parents) states.
async fn numbered(
    database: &pamin_store::Database,
    project: pamin_core::ProjectId,
    cut: Vec<(File, Option<FileRecord>, Vec<Unnamed>)>,
    recorded: &[FileTopic],
) -> Result<(Vec<Read>, Vec<(String, String)>)> {
    // Per file, the earlier sections a number could be kept from.
    let wanted: Vec<Vec<String>> = cut
        .iter()
        .map(|(file, _, sections)| {
            let bases: BTreeSet<&str> = sections.iter().map(|s| s.base.as_str()).collect();
            let earlier: Vec<String> = recorded
                .iter()
                .filter(|topic| belongs(&topic.name, &file.name))
                .filter(|topic| bases.iter().any(|base| number(&topic.name, base).is_some()))
                .map(|topic| topic.name.clone())
                .collect();
            let repeated = bases.len() < sections.len();
            let renumbered = earlier.iter().any(|name| {
                bases
                    .iter()
                    .any(|base| number(name, base).is_some_and(|n| n > 1))
            });
            if repeated || renumbered {
                earlier
            } else {
                Vec::new()
            }
        })
        .collect();
    let names: Vec<String> = wanted.iter().flatten().cloned().collect();
    let contents: std::collections::HashMap<String, String> =
        pamin_store::repository::current_named(database.pool(), project, &names)
            .await?
            .into_iter()
            .map(|named| (named.name, named.content))
            .collect();
    let mut edges = Vec::new();
    let files = cut
        .into_iter()
        .zip(wanted)
        .map(|((file, record, sections), wanted)| {
            let previous: Vec<Section> = wanted
                .into_iter()
                .filter_map(|name| {
                    let content = contents.get(&name)?.clone();
                    Some(Section { name, content })
                })
                .collect();
            let bases: Vec<String> = sections.iter().map(|s| s.base.clone()).collect();
            let sections = named(sections, &previous);
            let bases: Vec<&str> = bases.iter().map(String::as_str).collect();
            for (from, to) in parents(&bases) {
                edges.push((sections[from].name.clone(), sections[to].name.clone()));
            }
            Read {
                file,
                record,
                sections,
            }
        })
        .collect();
    Ok((files, edges))
}

/// What a command's paths came to.
#[derive(Debug)]
pub(crate) struct Found {
    /// Every file, once each, in name order.
    pub(crate) files: Vec<File>,
    /// What was skipped, with why.
    pub(crate) skipped: Vec<String>,
    /// Every directory walked.
    pub(crate) trees: Vec<Tree>,
    /// Every directory the walks went into, the trees themselves among them:
    /// what a watcher watches, and nothing a walk skips.
    pub(crate) dirs: Vec<PathBuf>,
}

/// Every file `paths` names, once each, in name order, and the ones skipped
/// with why.
///
/// A path that cannot be read at all is a refusal, because it is what the
/// command was asked to do; a file or directory found under one and unreadable
/// is skipped and reported, as one that is not UTF-8 is. Two files that would
/// go by one name are a refusal too: they would be one topic, which each
/// ingest would hand from one to the other.
pub(crate) fn files(paths: &[PathBuf]) -> Result<Found> {
    let mut files = BTreeSet::new();
    let mut skipped = Vec::new();
    let mut trees = Vec::new();
    let mut dirs = Vec::new();
    for path in paths {
        // Resolved here rather than by the client, which made it absolute but
        // left `..` and links in it: two spellings of one file must give its
        // sections one name.
        let path =
            std::fs::canonicalize(path).with_context(|| format!("reading {}", path.display()))?;
        let root = root(&path);
        if path.is_dir() {
            let whole = walk(&path, &root, &mut files, &mut dirs, &mut skipped);
            match file(&path, &root) {
                Some(tree) => trees.push(Tree {
                    name: tree.name,
                    root: tree.root,
                    whole,
                }),
                None => skipped.push(format!("{}: name is not UTF-8", path.display())),
            }
        } else {
            match file(&path, &root) {
                Some(file) => {
                    files.insert(file);
                }
                None => skipped.push(format!("{}: name is not UTF-8", path.display())),
            }
        }
    }

    let mut by_name: std::collections::BTreeMap<&str, Vec<&Path>> = Default::default();
    for file in &files {
        by_name.entry(&file.name).or_default().push(&file.path);
    }
    let clashes: Vec<String> = by_name
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(name, paths)| {
            let paths: Vec<String> = paths
                .iter()
                .map(|path| path.display().to_string())
                .collect();
            format!("{name}: {}", paths.join(", "))
        })
        .collect();
    if !clashes.is_empty() {
        anyhow::bail!(
            "these files would be recorded under one name; ingest them separately:\n{}",
            clashes.join("\n")
        );
    }

    Ok(Found {
        files: files.into_iter().collect(),
        skipped,
        trees,
        dirs,
    })
}

/// What the files under `path` are named relative to: the repository it is
/// in, so that inside a repository, ingesting a tree and ingesting one file of
/// it name that file alike; or else `path` itself when it is a directory and
/// its parent when it is a file, so that outside one, a file named on its own
/// is named from its own directory rather than from a tree it was once
/// ingested with.
///
/// A submodule's `.git` is a file pointing into its superproject's
/// `.git/modules`, and is passed over, so a file in a submodule is named alike
/// whether the superproject or the submodule was ingested. A worktree's is a
/// file too, pointing anywhere else, and is a root.
pub(crate) fn root(path: &Path) -> PathBuf {
    let start = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    let is_root = |ancestor: &Path| {
        let git = ancestor.join(".git");
        if git.is_dir() {
            return true;
        }
        std::fs::read_to_string(&git).is_ok_and(|pointer| !pointer.contains(".git/modules/"))
    };
    start
        .ancestors()
        .find(|ancestor| is_root(ancestor))
        .unwrap_or(start)
        .to_path_buf()
}

/// The file at `path`, named from `root`; `None` when a component of its name
/// is not UTF-8, which a topic's name has to be and which a lossy conversion
/// would let two files share.
pub(crate) fn file(path: &Path, root: &Path) -> Option<File> {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let mut components = Vec::new();
    if let Some(root) = root.file_name() {
        components.push(root.to_str()?);
    }
    for component in relative.components() {
        components.push(component.as_os_str().to_str()?);
    }
    Some(File {
        name: components.join("/"),
        path: path.to_path_buf(),
        root: root.to_str()?.to_string(),
    })
}

/// The files under `directory` with one of [`EXTENSIONS`], and the
/// directories the walk went into -- and whether the walk met no error, so
/// that a file it did not find is one that is not there. By the rules of
/// [`walker`].
fn walk(
    directory: &Path,
    root: &Path,
    files: &mut BTreeSet<File>,
    dirs: &mut Vec<PathBuf>,
    skipped: &mut Vec<String>,
) -> bool {
    let mut whole = true;
    for entry in walker(directory).build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                skipped.push(error.to_string());
                whole = false;
                continue;
            }
        };
        let path = entry.path();
        if entry.file_type().is_some_and(|kind| kind.is_dir()) {
            dirs.push(path.to_path_buf());
            continue;
        }
        let is_text = entry.file_type().is_some_and(|kind| kind.is_file()) && walked_kind(path);
        if !is_text {
            continue;
        }
        match file(path, root) {
            Some(file) => {
                files.insert(file);
            }
            None => skipped.push(format!("{}: name is not UTF-8", path.display())),
        }
    }
    whole
}

/// The walk of `directory` an ingest reads a tree with, skipping hidden
/// entries -- `.git` among them -- symbolic links, which could lead out of the
/// tree or round in a circle, and whatever git ignores there. A repository's
/// `node_modules/` or `target/` is full of Markdown nobody wrote for it.
fn walker(directory: &Path) -> ignore::WalkBuilder {
    let mut walker = ignore::WalkBuilder::new(directory);
    walker
        .hidden(true)
        .follow_links(false)
        // Git's rules only: `.gitignore` in the tree and above it,
        // `.git/info/exclude` and the user's global excludes, inside a
        // repository. Not ripgrep's own `.ignore` and `.rgignore`, which a
        // user writing them meant for a search tool.
        .ignore(false)
        .parents(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true);
    walker
}

/// What a walk would find directly inside `directory`, by the walk's own
/// rules: its directories, and its files of a kind a walk reads. Empty when it
/// cannot be read -- gone, or never there.
pub(crate) fn listed(directory: &Path) -> BTreeSet<PathBuf> {
    walker(directory)
        .max_depth(Some(1))
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.depth() == 1)
        .filter(|entry| {
            entry
                .file_type()
                .is_some_and(|kind| kind.is_dir() || (kind.is_file() && walked_kind(entry.path())))
        })
        .map(|entry| entry.into_path())
        .collect()
}

/// Where an ingest looked.
struct Looked<'a> {
    /// The files it read, by name and root.
    read: &'a [(String, String)],
    /// The files it found and left unread because their record says they have
    /// not changed: their sections are all still there.
    unchanged: &'a [(String, String)],
    /// The files it found and could not read.
    unread: &'a [(String, String)],
    /// The directories it walked.
    trees: &'a [Tree],
    /// Whether a path is there at all: a file a walk skipped -- hidden, or
    /// ignored by git, and named on its own some earlier time -- is not gone.
    exists: &'a dyn Fn(&Path) -> bool,
}

/// A section an earlier ingest recorded that this one did not cut again.
#[derive(Debug, PartialEq, Eq)]
struct Gone {
    name: String,
    /// What its topic said when this ingest looked.
    current: pamin_core::TopicStateId,
    /// Whether this ingest can say it is gone rather than out of its reach: it
    /// was cut from a file the ingest read, or from a file of a kind a walk
    /// reads under a directory it walked whole.
    retirable: bool,
}

/// The sections in `recorded` -- every section an ingest has put on the
/// retrieval surface -- that are not among `cut` and that this ingest would
/// have cut again: from a file it read, or from a file under a directory it
/// walked that is no longer there -- deleted, moved, or ignored now.
///
/// Each from the root it was recorded from, so that another tree whose files
/// are named alike -- another clone, another `notes` -- is not taken for this
/// one. A file it found and could not read is not where a section went
/// missing, and a file of another kind named on its own an earlier time is not
/// one a walk would have found; neither is listed. Under a directory whose
/// walk met an error, a file not found may only have been out of reach, so its
/// sections are listed but not retirable.
fn gone(looked: &Looked, cut: &[&Section], recorded: Vec<FileTopic>) -> Vec<Gone> {
    let present: BTreeSet<&str> = cut.iter().map(|section| section.name.as_str()).collect();
    // By root and name, asked of every name a section's file could have, so a
    // catch-up of thousands of files is not thousands of comparisons a section.
    fn by_root(files: &[(String, String)]) -> std::collections::HashSet<(&str, &str)> {
        files
            .iter()
            .map(|(file, root)| (root.as_str(), file.as_str()))
            .collect()
    }
    let (read, unchanged) = (by_root(looked.read), by_root(looked.unchanged));
    let mut gone = Vec::new();
    for topic in recorded {
        if present.contains(topic.name.as_str()) {
            continue;
        }
        let Some(root) = topic.root.as_deref() else {
            continue;
        };
        let name = topic.name.as_str();
        if files_of(name).any(|file| unchanged.contains(&(root, file))) {
            continue;
        }
        let retirable = if files_of(name).any(|file| read.contains(&(root, file))) {
            Some(true)
        } else {
            let file = file_of(name);
            let unread = looked
                .unread
                .iter()
                .any(|(unread, at)| at == root && unread == file);
            if unread || !walked_kind(Path::new(file)) {
                None
            } else {
                // Under every walked tree that holds it, and retirable if any
                // of them was walked whole and the file is not on disk after
                // all, where the walk skipped it.
                looked
                    .trees
                    .iter()
                    .filter(|tree| tree.root == root && under(file, &tree.name))
                    .map(|tree| tree.whole)
                    .reduce(|left, right| left || right)
                    .map(|whole| whole && !(looked.exists)(&on_disk(root, file)))
            }
        };
        if let Some(retirable) = retirable {
            gone.push(Gone {
                name: topic.name,
                current: topic.current,
                retirable,
            });
        }
    }
    gone.sort_by(|left, right| left.name.cmp(&right.name));
    gone
}

/// Where the file named `file`, named from `root`, would be: its name is the
/// root's own name and then its path from the root.
pub(crate) fn on_disk(root: &str, file: &str) -> PathBuf {
    let root = Path::new(root);
    let relative = match root.file_name() {
        Some(_) => file.split_once('/').map_or("", |(_, rest)| rest),
        None => file,
    };
    root.join(relative)
}

/// The name of the file the section named `name` was cut out of: its name up
/// to its headings, without the number it was given.
pub(crate) fn file_of(name: &str) -> &str {
    let mut file = name.split(SEPARATOR).next().unwrap_or(name);
    while let Some((before, _)) = file
        .rsplit_once(" #")
        .filter(|(_, number)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()))
    {
        file = before;
    }
    file
}

/// Whether a walk reads a file at `path`: one with an extension of
/// [`EXTENSIONS`].
pub(crate) fn walked_kind(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// Whether the section named `name` was cut out of the file named `file`: its
/// name is the file's, then any numbers ` #n` it was given, then nothing or
/// its headings.
///
/// Read exactly, because `a.md #1.md` begins with `a.md #1` and is another
/// file.
fn belongs(name: &str, file: &str) -> bool {
    let Some(mut rest) = name.strip_prefix(file) else {
        return false;
    };
    loop {
        if rest.is_empty() || rest.starts_with(SEPARATOR) {
            return true;
        }
        match rest.strip_prefix(" #").and_then(digits) {
            Some(after) => rest = after,
            None => return false,
        }
    }
}

/// `text` after the run of ASCII digits it opens with, if it opens with one.
fn digits(text: &str) -> Option<&str> {
    let after = text.trim_start_matches(|c: char| c.is_ascii_digit());
    (after.len() < text.len()).then_some(after)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(name: &str) -> Section {
        Section {
            name: name.to_string(),
            content: String::new(),
        }
    }

    fn names(paths: &[PathBuf]) -> Vec<String> {
        let Found { files, skipped, .. } = files(paths).expect("files");
        assert!(skipped.is_empty(), "{skipped:?}");
        files.into_iter().map(|file| file.name).collect()
    }

    /// The name of the temporary directory `path` is, which every name under
    /// it starts with.
    fn base(path: &Path) -> String {
        path.file_name().unwrap().to_str().unwrap().to_string()
    }

    #[test]
    fn a_section_belongs_to_its_file_and_no_other() {
        for name in [
            "docs/a.md",
            "docs/a.md › Usage",
            "docs/a.md #2",
            "docs/a.md #2 #3",
            "docs/a.md #2 › Usage",
        ] {
            assert!(belongs(name, "docs/a.md"), "{name}");
        }
        for name in [
            "docs/a.md.bak › Usage",
            "docs/ab.md",
            "docs",
            "docs/a.md (2)",
            "docs/a.md #1.md",
            "docs/a.md #",
        ] {
            assert!(!belongs(name, "docs/a.md"), "{name}");
        }
    }

    const ROOT: &str = "/home/a/repo";

    fn recorded(name: &str, root: &str) -> FileTopic {
        FileTopic {
            name: name.to_string(),
            current: pamin_core::TopicStateId::new(),
            root: Some(root.to_string()),
        }
    }

    fn read(name: &str) -> (String, String) {
        (name.to_string(), ROOT.to_string())
    }

    fn tree(name: &str, whole: bool) -> Tree {
        Tree {
            name: name.to_string(),
            root: ROOT.to_string(),
            whole,
        }
    }

    fn listed(gone: &[Gone]) -> Vec<(&str, bool)> {
        gone.iter()
            .map(|gone| (gone.name.as_str(), gone.retirable))
            .collect()
    }

    /// Only the files this ingest read are compared: a section of a file left
    /// out of the command is not gone, it is just not being looked at.
    #[test]
    fn gone_is_what_an_earlier_ingest_left_under_the_files_read() {
        let kept = section("repo/a.md › Kept");
        let gone = gone(
            &Looked {
                read: &[read("repo/a.md")],
                unchanged: &[],
                unread: &[],
                trees: &[],
                exists: &|_: &Path| false,
            },
            &[&kept],
            vec![
                recorded("repo/a.md › Kept", ROOT),
                recorded("repo/a.md › Gone", ROOT),
                recorded("repo/b.md › Elsewhere", ROOT),
            ],
        );
        assert_eq!(listed(&gone), [("repo/a.md › Gone", true)]);
    }

    /// Under a directory it walked, an ingest also finds the sections of a
    /// file that is no longer there -- but not of one it found and could not
    /// read, nor of a file of a kind a walk does not read, which was named on
    /// its own before.
    #[test]
    fn gone_includes_files_missing_from_a_walked_tree() {
        let kept = section("repo/docs/a.md › Kept");
        let gone = gone(
            &Looked {
                read: &[read("repo/docs/a.md")],
                unchanged: &[],
                unread: &[read("repo/docs/unreadable.md")],
                trees: &[tree("repo/docs", true)],
                exists: &|_: &Path| false,
            },
            &[&kept],
            [
                "repo/docs/a.md › Kept",
                "repo/docs/deleted.md › Usage",
                "repo/docs/deleted.md #2",
                "repo/docs/deep/moved.txt",
                "repo/docs/unreadable.md › Kept",
                "repo/docs/script.py › Comment",
                "repo/docsier/other.md",
                "repo/README.md › Top",
            ]
            .iter()
            .map(|name| recorded(name, ROOT))
            .collect(),
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/docs/deep/moved.txt", true),
                ("repo/docs/deleted.md #2", true),
                ("repo/docs/deleted.md › Usage", true)
            ]
        );
    }

    /// Another tree whose files are named alike -- another clone of `repo`
    /// somewhere else -- is not this one: its sections are not gone.
    #[test]
    fn another_tree_of_the_same_name_is_not_looked_at() {
        let gone = gone(
            &Looked {
                read: &[read("repo/a.md")],
                unchanged: &[],
                unread: &[],
                trees: &[tree("repo", true)],
                exists: &|_: &Path| false,
            },
            &[],
            vec![
                recorded("repo/a.md › Here", ROOT),
                recorded("repo/a.md › There", "/home/b/repo"),
                recorded("repo/b.md › There", "/home/b/repo"),
            ],
        );
        assert_eq!(listed(&gone), [("repo/a.md › Here", true)]);
    }

    /// A walk that met an error cannot tell a file that is gone from one it
    /// could not reach: its sections are listed and not retirable, unless a
    /// file the ingest read says otherwise.
    #[test]
    fn a_walk_that_met_an_error_retires_nothing_it_did_not_read() {
        let gone = gone(
            &Looked {
                read: &[read("repo/docs/a.md")],
                unchanged: &[],
                unread: &[],
                trees: &[tree("repo/docs", false)],
                exists: &|_: &Path| false,
            },
            &[],
            vec![
                recorded("repo/docs/a.md › Old heading", ROOT),
                recorded("repo/docs/locked/b.md › Usage", ROOT),
            ],
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/docs/a.md › Old heading", true),
                ("repo/docs/locked/b.md › Usage", false)
            ]
        );
    }

    /// A file the walk skipped but that is still on disk -- hidden, or ignored
    /// by git, and named on its own before -- is listed, not retired.
    #[test]
    fn a_file_the_walk_skipped_is_not_gone() {
        let gone = gone(
            &Looked {
                read: &[],
                unchanged: &[],
                unread: &[],
                trees: &[tree("repo", true)],
                exists: &|path: &Path| path == Path::new("/home/a/repo/.github/CONTRIBUTING.md"),
            },
            &[],
            vec![
                recorded("repo/.github/CONTRIBUTING.md › Style", ROOT),
                recorded("repo/deleted.md › Usage", ROOT),
            ],
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/.github/CONTRIBUTING.md › Style", false),
                ("repo/deleted.md › Usage", true)
            ]
        );
        assert_eq!(
            on_disk("/home/a/repo", "repo/docs/a.md"),
            Path::new("/home/a/repo/docs/a.md")
        );
    }

    /// A file left unread because its record says it has not changed still
    /// has every section it had: none is gone, under a walked tree or not.
    #[test]
    fn a_file_left_unread_as_unchanged_is_not_gone() {
        let gone = gone(
            &Looked {
                read: &[read("repo/docs/a.md")],
                unchanged: &[read("repo/docs/same.md"), read("repo/b › c.md")],
                unread: &[],
                trees: &[tree("repo", true)],
                exists: &|_: &Path| true,
            },
            &[],
            vec![
                recorded("repo/docs/same.md › Usage", ROOT),
                recorded("repo/docs/same.md #2", ROOT),
                recorded("repo/b › c.md › Heading", ROOT),
                recorded("repo/docs/a.md › Old", ROOT),
            ],
        );
        assert_eq!(listed(&gone), [("repo/docs/a.md › Old", true)]);
    }

    /// Every name a section's file could have is tried, the whole of it among
    /// them, and only those [`belongs`] confirms: a file whose own name holds
    /// a heading's separator is found as itself.
    #[test]
    fn a_sections_file_is_found_among_every_name_it_could_have() {
        let files = |name| files_of(name).collect::<Vec<_>>();
        assert_eq!(files("r/a.md"), ["r/a.md"]);
        assert_eq!(
            files("r/a.md › H › I"),
            ["r/a.md", "r/a.md › H", "r/a.md › H › I"]
        );
        assert_eq!(
            files("r/a.md #2 › H"),
            ["r/a.md #2", "r/a.md", "r/a.md #2 › H"]
        );
        assert_eq!(files("r/a #x.md"), ["r/a #x.md"]);
    }

    /// The prefix the store matches by finds a file's sections and every file
    /// whose name starts with its own; only the file's own are in play, and
    /// whatever sits under a walked tree.
    #[test]
    fn only_the_sections_of_the_files_and_trees_looked_at_are_in_play() {
        let kept = in_play(
            vec![
                recorded("repo/a.md › H", ROOT),
                recorded("repo/a.md #2", ROOT),
                recorded("repo/a.md.bak › H", ROOT),
                recorded("repo/docs/b.md", ROOT),
                recorded("repo/docsier/c.md", ROOT),
            ],
            &["repo/a.md"],
            &[tree("repo/docs", true)],
        );
        let names: Vec<&str> = kept.iter().map(|topic| topic.name.as_str()).collect();
        assert_eq!(names, ["repo/a.md › H", "repo/a.md #2", "repo/docs/b.md"]);
    }

    fn record_of(stat: &Stat) -> FileRecord {
        let file = File {
            name: "repo/a.md".into(),
            path: PathBuf::from("/home/a/repo/a.md"),
            root: ROOT.into(),
        };
        stat.recorded(&file, "hash".into())
    }

    /// A record is believed only for the very stat it was taken with, and only
    /// when that stat was taken well after the file last changed: a file
    /// modified within a timestamp's granularity of being read may change
    /// again without its stat showing it.
    #[test]
    fn a_record_is_trusted_for_its_own_stat_taken_after_the_file_settled() {
        let second = 1_000_000_000;
        let settled = Stat {
            size: 10,
            modified_ns: 100 * second,
            changed_ns: 100 * second,
            inode: 7,
            read_ns: 200 * second,
        };
        let record = record_of(&settled);
        assert!(trusted(&record, &settled));
        // Cut by another build: read again, whatever its stat says.
        let other = FileRecord {
            build: "another build".into(),
            ..record.clone()
        };
        assert!(!trusted(&other, &settled));
        for moved in [
            Stat {
                size: 11,
                ..settled
            },
            Stat {
                modified_ns: 101 * second,
                ..settled
            },
            Stat {
                changed_ns: 101 * second,
                ..settled
            },
            Stat {
                inode: 8,
                ..settled
            },
        ] {
            assert!(!trusted(&record, &moved), "{moved:?}");
        }
        // Read within two seconds of a change, to the file or its inode.
        for racy in [
            Stat {
                read_ns: 101 * second,
                ..settled
            },
            Stat {
                changed_ns: 199 * second,
                ..settled
            },
        ] {
            assert!(!trusted(&record_of(&racy), &racy), "{racy:?}");
        }
    }

    /// A file's stat is what a record keeps: its size, both times to the
    /// nanosecond and its inode, with the clock read before them.
    #[test]
    fn a_stat_is_taken_after_the_clock_and_moves_with_a_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.md");
        std::fs::write(&path, "one").unwrap();
        let before = now_ns();
        let first = stat(&path).unwrap();
        assert!(first.read_ns >= before);
        assert_eq!(first.size, 3);
        assert!(first.modified_ns <= first.read_ns + RACY_NS);
        assert_eq!(stat(&path).unwrap().inode, first.inode);
        // An atomic save: a new file renamed over the old one.
        let next = dir.path().join("a.md.tmp");
        std::fs::write(&next, "two!").unwrap();
        std::fs::rename(&next, &path).unwrap();
        let second = stat(&path).unwrap();
        assert!(!second.same_file(&record_of(&first)));
    }

    #[test]
    fn a_sections_file_is_its_name_before_its_headings_and_numbers() {
        assert_eq!(file_of("r/a.md"), "r/a.md");
        assert_eq!(file_of("r/a.md › H › I"), "r/a.md");
        assert_eq!(file_of("r/a.md #2"), "r/a.md");
        assert_eq!(file_of("r/a.md #2 › H"), "r/a.md");
        assert_eq!(file_of("r/a (old).md"), "r/a (old).md");
    }

    #[test]
    fn a_directory_is_read_for_text_files_skipping_hidden_entries_and_links() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("docs/deep")).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        for name in [
            "README.md",
            "docs/guide.markdown",
            "docs/deep/notes.TXT",
            "docs/image.png",
            ".hidden/secret.md",
            "docs/.draft.md",
        ] {
            std::fs::write(root.join(name), "text").unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("README.md"), root.join("docs/link.md")).unwrap();

        let base = base(root);
        assert_eq!(
            names(&[root.to_path_buf()]),
            [
                format!("{base}/README.md"),
                format!("{base}/docs/deep/notes.TXT"),
                format!("{base}/docs/guide.markdown"),
            ]
        );
    }

    /// Inside a repository, a file is named from the repository's root
    /// whether the tree or the file itself was named, so the two ingests
    /// record the same sections.
    #[test]
    fn a_file_is_named_alike_however_it_was_reached() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs/cli.md"), "text").unwrap();

        let expected = format!("{}/docs/cli.md", base(root));
        assert_eq!(names(&[root.join("docs")]), std::slice::from_ref(&expected));
        assert_eq!(names(&[root.join("docs/../docs/cli.md")]), [expected]);

        // Named twice, read once.
        assert_eq!(
            names(&[root.join("docs"), root.join("docs/cli.md")]).len(),
            1
        );
    }

    /// A submodule's `.git` file points into the superproject, so a file in
    /// it is named from the superproject's root however it is reached; a
    /// worktree's points elsewhere and is a root of its own.
    #[test]
    fn a_submodule_is_named_from_its_superproject_and_a_worktree_from_itself() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git/modules/sub")).unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/.git"), "gitdir: ../.git/modules/sub\n").unwrap();
        std::fs::write(root.join("sub/x.md"), "text").unwrap();
        let expected = format!("{}/sub/x.md", base(root));
        assert_eq!(
            names(&[root.to_path_buf()]),
            std::slice::from_ref(&expected)
        );
        assert_eq!(names(&[root.join("sub/x.md")]), [expected]);

        std::fs::create_dir_all(root.join("wt")).unwrap();
        std::fs::write(
            root.join("wt/.git"),
            "gitdir: /elsewhere/.git/worktrees/wt\n",
        )
        .unwrap();
        std::fs::write(root.join("wt/y.md"), "text").unwrap();
        assert_eq!(names(&[root.join("wt/y.md")]), ["wt/y.md"]);
    }

    /// Inside a repository, what git ignores is not read: by a `.gitignore`
    /// in the tree, in a directory under it, or above the directory named,
    /// and by `.git/info/exclude`.
    #[test]
    fn what_git_ignores_is_not_read() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git/info")).unwrap();
        std::fs::write(
            root.join(".gitignore"),
            "build/\n*.draft.md\ndocs/private/\n",
        )
        .unwrap();
        std::fs::write(root.join(".git/info/exclude"), "scratch.md\n").unwrap();
        for name in [
            "keep.md",
            "a.draft.md",
            "scratch.md",
            "build/out.md",
            "docs/guide.md",
            "docs/private/secret.md",
            "docs/notes/local.md",
            "docs/notes/kept.md",
        ] {
            std::fs::create_dir_all(root.join(name).parent().unwrap()).unwrap();
            std::fs::write(root.join(name), "text").unwrap();
        }
        std::fs::write(root.join("docs/notes/.gitignore"), "local.md\n").unwrap();

        let base = base(root);
        assert_eq!(
            names(&[root.to_path_buf()]),
            [
                format!("{base}/docs/guide.md"),
                format!("{base}/docs/notes/kept.md"),
                format!("{base}/keep.md"),
            ]
        );
        // Named below the `.gitignore` that excludes part of it.
        assert_eq!(
            names(&[root.join("docs")]),
            [
                format!("{base}/docs/guide.md"),
                format!("{base}/docs/notes/kept.md"),
            ]
        );
    }

    /// Outside a repository a `.gitignore` is a file like any other, as it is
    /// to git.
    #[test]
    fn outside_a_repository_nothing_is_ignored() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".gitignore"), "*.md\n").unwrap();
        std::fs::write(root.join("a.md"), "text").unwrap();
        assert_eq!(names(&[root]), ["notes/a.md"]);
    }

    #[test]
    fn outside_a_repository_a_tree_is_named_from_itself() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("notes")).unwrap();
        std::fs::write(root.join("notes/a.md"), "text").unwrap();
        assert_eq!(names(&[root.join("notes")]), ["notes/a.md"]);
    }

    /// Two trees whose files would share names are refused rather than
    /// recorded under one topic each.
    #[test]
    fn two_files_under_one_name_are_refused() {
        let (left, right) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for tree in [&left, &right] {
            std::fs::create_dir_all(tree.path().join("notes")).unwrap();
            std::fs::write(tree.path().join("notes/a.md"), "text").unwrap();
        }
        // Different roots with different names: two names.
        assert_eq!(
            names(&[left.path().to_path_buf(), right.path().to_path_buf()]).len(),
            2
        );
        // The same root name twice: one name, refused.
        let refused = files(&[left.path().join("notes"), right.path().join("notes")])
            .expect_err("two files under one name");
        assert!(refused.to_string().contains("notes/a.md"), "{refused}");
    }

    #[test]
    #[cfg(unix)]
    fn an_unreadable_directory_is_skipped_rather_than_ending_the_ingest() {
        use std::os::unix::fs::PermissionsExt;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("locked")).unwrap();
        std::fs::write(root.join("a.md"), "text").unwrap();
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o000))
            .unwrap();
        // Root reads every directory whatever its mode, so there is nothing
        // to observe there.
        let readable = std::fs::read_dir(root.join("locked")).is_ok();
        let Found {
            files: found,
            skipped,
            trees,
            ..
        } = files(&[root.to_path_buf()]).expect("files");
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert_eq!(found.len(), 1);
        if !readable {
            assert_eq!(skipped.len(), 1, "{skipped:?}");
            // And the tree is not taken to be whole, so nothing under the
            // directory that could not be read is retired as gone.
            assert!(!trees[0].whole);
        }
    }
}
