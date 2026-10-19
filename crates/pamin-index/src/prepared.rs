//! A copy of a model that ONNX Runtime maps from disk instead of copying.
//!
//! Loaded from the file the hub serves, a model's weights end up on the heap.
//! ONNX Runtime copies every initializer embedded in the `.onnx` into memory
//! it owns, and the CPU's matrix kernels then pack the int8 weights of every
//! `MatMulInteger` into a layout of their own, which is another allocation;
//! the file itself is read and discarded. Measured on the `accurate`
//! reranker's int8 export, 570 MB on disk, in a bare ONNX Runtime 1.28 session
//! with the memory arena off and four threads: **+664 MB anonymous after load,
//! +673 MB after four passes of sixteen 128-token pairs.**
//!
//! ONNX Runtime can write the optimized graph back out with every initializer
//! -- the packed ones included -- in an external data file, and a session
//! loaded from that copy maps the file rather than reading it. The same
//! session over such a copy: **+11 MB anonymous after load, +21 MB after the
//! same passes**, 344 MB resident because only the pages a pass touches are
//! read in, loaded in 0.3 s against 2 to 3, and the sixteen logits summing to
//! the same `f32` bits (-125.484772). The passes were not slower, on a machine
//! busy with other work, so no speed-up is claimed. Mapped pages are the
//! file's: the kernel can drop them under pressure and read them back, which it
//! cannot do with anonymous memory when there is no swap.
//!
//! Through the loads the product makes -- `Reranker::load` and
//! `Embedder::load`, each holding its tokenizer as well -- and counting only
//! what is live once the allocator has returned what it freed, the copy takes
//! the `accurate` reranker from 822 MB anonymous to 271, BGE-M3 from 824 to
//! 272, and the `fast` reranker from 385 to 268, with every score and vector
//! bit-identical. `tests/prepared.rs` is what measures that and fails if it
//! stops being true.
//!
//! What it costs is disk and one slow first load. The copy is written once per
//! model from the downloaded weights, and is larger than the file it came
//! from -- 874 MB of data for the 570 MB reranker, because the packed weights
//! are stored alongside the originals. Writing it took 5.1 s on that model,
//! during which the source is loaded onto the heap exactly as it was before,
//! and then dropped.
//!
//! A copy belongs to the machine that wrote it, which is what the cache key
//! is for. A graph optimized with layout transformations is specific to the
//! instruction set it was optimized on, and ONNX Runtime documents that it
//! may need that instruction set to run. And the runtime finds a packed
//! weight on disk by a hash of the packed bytes, so a copy written on a CPU
//! that packs differently -- AMX against AVX-512 VNNI against AVX2 -- loads
//! and scores correctly but finds none of them and packs onto the heap again,
//! silently: a copy of the `accurate` reranker written without its packed
//! weights held 304 MB in a bare session, against 667 for the source at the
//! same optimization level. So the key covers the runtime build, the
//! architecture and its matrix-relevant features, and the optimization level,
//! and a machine that differs in any of them writes its own copy.
//!
//! Only the CPU. An accelerator copies weights into its own memory whatever
//! the file looks like, and loads the file the hub serves.
//!
//! Once a copy has loaded, the download it was written from is removed
//! ([`release`]): nothing reads it again on this runtime and this CPU, and at
//! the defaults it was 1,141 MB of the 2,923 MB the two models took, measured
//! on a model directory before and after one load of each. What that
//! gives up is the source for the next copy. A copy that no longer fits -- a
//! runtime upgrade, a model directory moved to another CPU -- is written from
//! a fresh download, so that load needs the network, and fails saying so when
//! there is none ([`load_path`]). A small record of what the download was
//! (`<repository>--<file>.<revision>.source`, beside the copies, one per
//! revision) is what finds the copy without the file it was keyed by, and
//! what lets a re-download key exactly as the removed one did.
//!
//! A copy's key changes with the runtime and the CPU, so every upgrade of ONNX
//! Runtime writes new copies and leaves the old ones -- 0.8 to 1.2 GB a model
//! -- where nothing would otherwise remove them. Every load therefore
//! collects, by one rule: **a copy is removed once it is not the copy this
//! runtime and CPU load for a model recorded beside it, no process holds it,
//! and none has loaded it for [`UNUSED_FOR`]**.
//!
//! The first part is what keeps a copy whose download is gone. Every model
//! this build has loaded has a record, and the key this build derives from a
//! record is the copy its next load of that model looks for. Removing that
//! copy would turn the load into a download -- offline, into an error --
//! however long the model had gone unused, so it stays for as long as this
//! build finds that record. The price is that a model nothing loads any more
//! -- a reranker tier switched off -- keeps its copy until `prepared/` is
//! deleted by hand. What goes is every other copy: an upgrade's
//! leftovers, one written on a CPU the directory was moved from, one of a file
//! the repository has since replaced, and one an older build wrote beside a
//! download it kept, whose loss costs only a rewrite from that download.
//!
//! The rest of the rule is not by key: a key is a hash nobody can read an
//! owner from, and two binaries built on different runtimes can share one
//! model directory, each loading its own copy and each entitled to find it
//! there. Each keeps its own copies from its own collections by the first
//! part, and from the other's by these two:
//!
//! - *Holds.* A process that loads a copy holds its key's lock shared until
//!   it exits, and removal takes that lock exclusively and gives up if it
//!   cannot. So no process of this build or a later one has a copy removed
//!   while it runs, however long ago it loaded the model: not the mapping it
//!   is reading, which Linux and macOS would keep alive anyway and Windows
//!   would refuse to delete, and not the copy it loads again after releasing
//!   an idle model, which it would otherwise have to write again.
//! - *When it was last loaded* is the latest of three times: the stamp a load
//!   rewrites in the copy (`used`); the graph file's modification time, which
//!   is when the copy was written; and the data file's access time, which the
//!   kernel moves when a process maps it -- at most daily under `relatime`,
//!   never under `noatime`. The last two are for binaries built before stamps
//!   and holds existed, which do neither: while one of them keeps loading its
//!   copy, the access time keeps that copy wherever the filesystem records it.
//!
//! A load checks for its copy and stamps it under the same lock, held shared,
//! so a removal either sees the fresh stamp or has already happened and the
//! load writes the copy again. A removal renames the copy to its key's partial
//! directory first, so one that stops part way -- Windows will not delete a
//! file a process has mapped -- leaves what the next writer clears rather than
//! a copy that looks complete; and a partial directory no writer holds is
//! removed with the rest. Lock files stay. They are empty, and deleting one
//! that another process is about to open would let two writers each hold
//! "the" lock.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use ort::session::builder::GraphOptimizationLevel;
use sha2::{Digest, Sha256};

use crate::error::{IndexError, Result};

/// The optimization level the copy is written at.
///
/// It has to be the level the copy is loaded at, so it is the one
/// `crate::inference::session` loads every model at: `Level3`, ONNX Runtime's
/// layout level, which is what `fastembed` builds every session at with no way
/// to ask for another. Loading runs the optimizers again over the optimized
/// graph.
/// That the result scores bit-identically and still finds its packed weights
/// on disk is not argued here but checked, by `tests/prepared.rs`; a change of
/// level is exactly what that check would have to be re-run for, so the level
/// is part of the key.
pub(crate) const LEVEL: GraphOptimizationLevel = GraphOptimizationLevel::Level3;

/// What a prepared copy's two files are called.
///
/// The data file is named inside the graph, relative to it, so the pair moves
/// together and can be renamed into place as one directory.
const MODEL: &str = "model.onnx";
const DATA: &str = "model.onnx.data";

/// The same graph with its attention fused (see `crate::attention`), beside
/// the one ONNX Runtime wrote and reading the same data file. Present only
/// once it has scored a probe bit-identically to [`MODEL`].
const FUSED: &str = "attention.onnx";

/// Written instead of [`FUSED`] when there is nothing to fuse or the fused
/// graph did not score identically, saying which, so no later load asks
/// again. The copy then loads [`MODEL`].
const UNFUSED: &str = "attention.unfused";

/// What a graph of a copy -- [`MODEL`] or [`FUSED`] -- is called with its
/// word embeddings split out (see `crate::word_embeddings`), beside it and
/// reading the same data file: `model.split1.onnx`, with
/// `model.split1.table` saying where the table it no longer maps is, or
/// `model.split1.unsplit` saying why there is no such graph. Present only
/// once it has scored a probe bit-identically to the graph it came from.
///
/// Derived from a settled copy rather than settled with it, so a copy written
/// before the split existed gains one on its next load without being written
/// again, and the copy's key is unchanged. The number is the rewrite's: one
/// that changed what it writes would take the next, and leave this one's
/// files to go with their copy.
const SPLIT: &str = "split1";

/// The stamp a load leaves in a copy: an empty file whose modification time is
/// when the copy was last handed out.
const USED: &str = "used";

/// How long a copy nothing holds may go without a load before a load of any
/// model removes it.
///
/// A choice, not a measurement. Long enough that a binary run once a week
/// keeps its copy between runs -- one running now is covered by its hold
/// however long ago it loaded -- and short enough that an upgrade's leftovers
/// are gone within a month. Removing a copy too early costs the next load of
/// it one rewrite -- a few seconds, with the source on the heap while it is
/// written -- and a download first when its own was removed. The copy this
/// build loads for a recorded model is not judged by this at all (see the
/// module's documentation).
const UNUSED_FOR: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// How long a record's partial file is left before a collection removes it:
/// a release writes and renames one within moments, so one this old was
/// left by a release that stopped between the two.
const RECORD_PARTIAL_FOR: Duration = Duration::from_secs(60 * 60);

/// The lock of every copy this process has loaded, held shared until it exits
/// so that no other process removes the copy (see the module's documentation).
///
/// Keyed by the copy's directory, one handle each however often it is loaded.
static HELD: Mutex<BTreeMap<PathBuf, File>> = Mutex::new(BTreeMap::new());

/// A model file as [`load_path`] asks for it: on disk, from the hub, or
/// removed once a copy has replaced it. `crate::hub::File` in the product; a
/// stand-in in the tests, which cannot reach a hub.
pub(crate) trait Download {
    /// `repository/file`, which names the file's record and its log lines.
    fn label(&self) -> String;
    /// `repository@commit`: the revision the repository's ref resolves to
    /// now, which a record names so that a copy is found only for it.
    fn revision(&self) -> String;
    /// The file, if it is on disk, without asking the hub.
    fn on_disk(&self) -> Option<PathBuf>;
    /// The file, downloaded first if it is not on disk.
    fn fetch(&self) -> Result<PathBuf>;
    /// Removes the file from disk, returning whether it did: `false`, having
    /// touched nothing, when the file is not this model directory's to remove.
    fn remove(&self) -> Result<bool>;
}

/// The path to load `download` from on the CPU: a mapped copy of it under
/// `cache_dir`, written first if there is none yet.
///
/// A copy that exists is found without the download, which [`release`] has
/// usually removed. When there is none for this runtime and CPU the download
/// is fetched again, and the copy written from it; with no network that is an
/// error saying why the model is not on disk, rather than the hub client's.
///
/// Falls back to the download itself, with a warning, when the copy cannot be
/// written -- a read-only or full disk, a file lock the filesystem does not
/// support, a graph the runtime will not re-serialize. The model then loads
/// the way it did before copies existed, which costs memory and nothing else.
/// Turning [`prepared`](crate::InferenceSettings::prepared) off asks for that
/// on purpose, for a measurement that needs the unmapped load, and fetches the
/// download if it was removed.
///
/// The copy's graph with its attention fused, where that scored identically,
/// and turning [`fused_attention`](crate::InferenceSettings::fused_attention)
/// off asks for the graph as ONNX Runtime wrote it -- the other arm of a
/// measurement of what the fusion is worth. Either with its word embeddings
/// split out, where that scored identically ([`split`]), unless
/// [`split_embeddings`](crate::InferenceSettings::split_embeddings) is off.
pub(crate) fn load_path(
    download: &impl Download,
    cache_dir: &Path,
    settings: &crate::InferenceSettings,
) -> Result<PathBuf> {
    let root = cache_dir.join("prepared");
    let fused = settings.fused_attention;
    if !settings.prepared {
        let fetched = download.fetch()?;
        if let Some(recorded) = record(download, &root) {
            recorded
                .source
                .restore(&fetched, recorded.digest.as_deref());
        }
        return Ok(fetched);
    }
    let source = match download.on_disk() {
        Some(source) => source,
        None => {
            let recorded = recorded(download, &root);
            if let Some(copy) = recorded
                .as_ref()
                .and_then(|source| found(source, &root, fused))
            {
                collect(&root, SystemTime::now());
                return Ok(split(copy, settings));
            }
            let fetched = download.fetch().map_err(|error| match &recorded {
                Some(_) => IndexError::Engine(format!(
                    "no mapped copy of {} fits this runtime and CPU, and its download was \
                     removed once a copy had replaced it; writing one needs the download \
                     again, and fetching it failed. Connect to the network for this one \
                     load, or copy the model directory from a machine that has the file. \
                     ({error})",
                    download.label()
                )),
                None => error,
            })?;
            // This revision's, or else the model's latest of any revision: a
            // fetch whose bytes are the record's keys as the removed file did.
            if let Some(recorded) = record(download, &root).or_else(|| latest(download, &root)) {
                recorded
                    .source
                    .restore(&fetched, recorded.digest.as_deref());
            }
            fetched
        }
    };
    match prepare(&source, &root, settings.budget()?) {
        Ok(copy) => {
            collect(&root, SystemTime::now());
            Ok(split(graph(copy, fused), settings))
        }
        Err(error) => {
            // Another process released the download between this one finding
            // it and keying it, which it does only once the copy is complete.
            if !source.exists()
                && let Some(copy) =
                    recorded(download, &root).and_then(|source| found(&source, &root, fused))
            {
                collect(&root, SystemTime::now());
                return Ok(split(copy, settings));
            }
            tracing::warn!(
                source = %source.display(),
                %error,
                "could not write a mapped copy of the model; loading it onto the heap instead"
            );
            Ok(source)
        }
    }
}

/// Whether `path`, from [`load_path`], is a mapped copy rather than the
/// download it fell back to, whose weights a session copies onto the heap.
pub(crate) fn is_copy(path: &Path, cache_dir: &Path) -> bool {
    path.starts_with(cache_dir.join("prepared"))
}

/// Removes `download` from disk once a copy of it has loaded, keeping a
/// record of what it was.
///
/// Called by a load that has just succeeded from [`load_path`]'s answer, so a
/// copy that will not load never costs the file it could be written again
/// from. Under the copy's lock, so a writer of the same copy is never left
/// without its source; the record is written before the file goes, so an
/// interruption between the two costs a download, never the copy.
///
/// Best effort, and a no-op where there is nothing to do: the download is
/// already gone, there is no complete copy for it, copies are not
/// [`prepared`](crate::InferenceSettings::prepared), or the file is not the
/// model directory's own ([`Download::remove`]).
pub(crate) fn release(
    download: &impl Download,
    cache_dir: &Path,
    settings: &crate::InferenceSettings,
) {
    if !settings.prepared {
        return;
    }
    let Some(source) = download.on_disk() else {
        return;
    };
    let root = cache_dir.join("prepared");
    let released = (|| -> Result<Option<u64>> {
        let identity = Source::of(&source)?;
        let (key, _) = identity.key();
        if settled(&root.join(&key)).is_none() {
            return Ok(None);
        }
        // Shared: enough to wait out a writer, which holds it exclusively, and
        // what this process already holds on the copy it has just loaded --
        // an exclusive lock would wait on that hold for ever.
        let lock = open(&root.join(format!("{key}.lock")))?;
        lock.lock_shared()?;
        // Again under the lock, which is what a writer of this copy holds.
        if settled(&root.join(&key)).is_none() {
            return Ok(None);
        }
        let record = root.join(record_name(download));
        // Written beside the record under a name of this writer's own, so
        // neither another revision's release nor another release of this
        // one can write into it before it is renamed into place.
        static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let partial = root.join(format!(
            "{}.{key}.{}-{}.partial",
            record_name(download),
            std::process::id(),
            WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::write(
            &partial,
            format!(
                "{}{REVISION}{}\n",
                identity.record_with_digest(&source)?,
                download.revision()
            ),
        )?;
        if let Err(error) = std::fs::rename(&partial, &record) {
            let _ = std::fs::remove_file(&partial);
            return Err(error.into());
        }
        // The model's other records -- other revisions', and one written
        // before records were per revision -- name copies nothing loads for
        // the latest revision. Each is removed now unless a process holds
        // it, rather than after [`UNUSED_FOR`] (two weeks of a second copy
        // per moved model), and its record with it; a held copy keeps its
        // record, so the revision holding it still finds it.
        //
        // Except a record of this copy: another revision whose weights are
        // these -- only its tokenizer or configuration changed, or it is
        // pinned beside a moving ref -- loads this copy too, and finds it
        // offline only through its own record. Only one naming no revision
        // is replaced by this one, since no load finds a copy through it.
        for other in records(download, &root) {
            if other == record {
                continue;
            }
            let Some(old) = read_record(&other) else {
                continue;
            };
            let (revision, old) = (old.revision, old.source.key().0);
            let copy = root.join(&old);
            if old != key
                && let Err(error) =
                    collect_one(&root, &copy, &current(&root), SystemTime::now(), true)
            {
                tracing::warn!(copy = old, %error, "could not remove a replaced revision's mapped copy");
            }
            if !copy.exists() || (old == key && revision.is_none()) {
                let _ = std::fs::remove_file(&other);
            }
        }
        Ok(download.remove()?.then_some(identity.length))
    })();
    match released {
        Ok(Some(bytes)) => tracing::info!(
            model = %download.label(),
            megabytes = bytes / 1_000_000,
            "removed a model's download, which its mapped copy has replaced"
        ),
        Ok(None) => {}
        Err(error) => tracing::warn!(
            model = %download.label(),
            %error,
            "could not remove a model's download beside its mapped copy; keeping it"
        ),
    }
}

/// Whether loading `download` reads only what is on disk: the download, or a
/// copy that fits this runtime and CPU.
///
/// Asking is not loading: the copy is neither stamped nor held.
pub(crate) fn is_ready(
    download: &impl Download,
    cache_dir: &Path,
    settings: &crate::InferenceSettings,
) -> bool {
    let root = cache_dir.join("prepared");
    download.on_disk().is_some()
        || (settings.prepared
            && recorded(download, &root)
                .is_some_and(|source| settled(&root.join(source.key().0)).is_some()))
}

/// The graph a caller loads from a settled copy: the fused one unless `fused`
/// is false, which asks for the one ONNX Runtime wrote.
fn graph(copy: PathBuf, fused: bool) -> PathBuf {
    if copy.ends_with(FUSED) && !fused {
        copy.with_file_name(MODEL)
    } else {
        copy
    }
}

/// The settled copy of the download `source` described, if this runtime and
/// CPU have one, stamped and held as [`prepare`] does with a copy it finds.
fn found(source: &Source, root: &Path, fused: bool) -> Option<PathBuf> {
    find(root, &source.key().0).map(|copy| graph(copy, fused))
}

/// What a removed download was, as [`release`] recorded it, if the record
/// names the revision the repository's ref resolves to now.
///
/// Once `refs/main` moves, the copy of the old revision holds the old
/// weights while everything else the model reads -- its tokenizer --
/// follows the new ref. A record of another revision is therefore not this
/// load's, and neither is one written before records named a revision,
/// which cannot say which it was: the download is
/// fetched again, and [`load_path`] keys it as the record's own source when
/// the two are the same file, so an unchanged model finds its copy without
/// writing it again. Otherwise a new copy is written, and its record
/// replaces this one, which removes the old copy (see [`release`]).
fn recorded(download: &impl Download, root: &Path) -> Option<Source> {
    let record = record(download, root)?;
    (record.revision.as_deref() == Some(download.revision().as_str())).then_some(record.source)
}

/// What [`release`] recorded of a removed download.
struct Record {
    source: Source,
    /// The revision it was read at, if the record names one.
    revision: Option<String>,
    /// The SHA-256 of its bytes, if the record names one.
    digest: Option<String>,
}

/// The record [`release`] wrote for `download`'s revision, or, read in its
/// place, the one an older build wrote for the model under a name of no
/// revision, which [`recorded`] checks the revision of.
fn record(download: &impl Download, root: &Path) -> Option<Record> {
    read_record(&root.join(record_name(download)))
        .or_else(|| read_record(&root.join(legacy_record_name(download))))
}

/// The record of `download`'s model written last, of any revision.
fn latest(download: &impl Download, root: &Path) -> Option<Record> {
    records(download, root)
        .into_iter()
        .filter_map(|path| Some((std::fs::metadata(&path).ok()?.modified().ok()?, path)))
        .max()
        .and_then(|(_, path)| read_record(&path))
}

/// What the record at `path` says, if it is one.
fn read_record(path: &Path) -> Option<Record> {
    let record = std::fs::read_to_string(path).ok()?;
    let line = |prefix: &str| {
        record
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .map(str::to_string)
    };
    Some(Record {
        source: Source::parse(&record)?,
        revision: line(REVISION),
        digest: line(DIGEST),
    })
}

/// The record line naming the revision a copy was written from.
const REVISION: &str = "revision ";

/// The record line naming the SHA-256 of the download a copy was written
/// from.
const DIGEST: &str = "sha256 ";

/// The record's file name: the label, readable, with nothing a path would
/// split on, and a short hash of the revision, so each revision of a model
/// has its own record.
fn record_name(download: &impl Download) -> String {
    let revision = Sha256::digest(download.revision().as_bytes());
    let tag: String = revision[..REVISION_TAG / 2]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{}.{tag}.source", model_name(download))
}

/// How many hex digits of the revision's hash a record's name carries.
const REVISION_TAG: usize = 16;

/// The model's records' common name: its label, with nothing a path would
/// split on.
fn model_name(download: &impl Download) -> String {
    download.label().replace(['/', '\\'], "--")
}

/// The name an older build gave the model's one record, of whichever
/// revision.
fn legacy_record_name(download: &impl Download) -> String {
    format!("{}.source", model_name(download))
}

/// The model a record's file name is of: its name without the revision,
/// which a record an older build wrote does not carry.
fn record_model(file: &str) -> Option<&str> {
    let name = file.strip_suffix(".source")?;
    Some(match name.rsplit_once('.') {
        Some((model, tag))
            if tag.len() == REVISION_TAG && tag.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
        {
            model
        }
        _ => name,
    })
}

/// Every record of `download`'s model, of any revision.
fn records(download: &impl Download, root: &Path) -> Vec<PathBuf> {
    let model = model_name(download);
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .and_then(record_model)
                == Some(model.as_str())
        })
        .collect()
}

/// Finds or writes the copy of `source` under `root`, stamps it as used, and
/// holds it for the rest of the process.
///
/// A copy is written into `<key>.partial` and renamed to `<key>` only once both
/// files are on disk, so a directory named by its key is always complete: a
/// crash leaves a partial directory, never a half-written copy that loads.
/// Writing happens under an exclusive lock on `<key>.lock`, so two processes
/// loading the same model at once write it once -- the second waits and then
/// finds the first's copy -- and a partial directory found while holding the
/// lock is a crashed writer's and safe to remove. A copy that is already there
/// is found by [`find`], under the same lock held shared.
///
/// The attention fusion is settled under the exclusive lock, once per copy --
/// for a copy written before the fusion existed, on the first load that finds
/// it -- and the path returned is the graph to load: [`FUSED`] if it was
/// kept, [`MODEL`] if not.
fn prepare(source: &Path, root: &Path, threads: usize) -> Result<PathBuf> {
    let (key, described) = Source::of(source)?.key();
    let done = root.join(&key);
    let copy = done.join(MODEL);
    if let Some(found) = find(root, &key) {
        return Ok(found);
    }
    // Not holding it shared while waiting to write it, or this process would
    // wait on itself: a copy it loaded before and somebody has since deleted.
    drop(held().remove(&done));

    std::fs::create_dir_all(root)?;
    let exclusive = open(&root.join(format!("{key}.lock")))?;
    exclusive.lock()?;
    let graph = match settled(&done) {
        Some(settled) => settled,
        None if copy.exists() => {
            crate::activity::prepared();
            fuse(&done, threads)
        }
        None => {
            crate::activity::prepared();
            let partial = root.join(format!("{key}.partial"));
            if partial.exists() {
                std::fs::remove_dir_all(&partial)?;
            }
            std::fs::create_dir(&partial)?;
            let started = std::time::Instant::now();
            let written = write(source, &partial)
                .and_then(|()| std::fs::rename(&partial, &done).map_err(IndexError::from));
            if written.is_err() {
                // Best effort: whatever is left is removed by the next writer anyway.
                let _ = std::fs::remove_dir_all(&partial);
            }
            written?;

            tracing::info!(
                source = %source.display(),
                copy = %copy.display(),
                seconds = started.elapsed().as_secs_f64(),
                key = %described,
                "wrote a mapped copy of the model"
            );
            fuse(&done, threads)
        }
    };
    // Stamped before the lock is let go of, so a removal that takes it in
    // between finds the copy in use.
    stamp(&done);
    exclusive.unlock()?;
    exclusive.lock_shared()?;
    hold(&done, exclusive);
    Ok(graph)
}

/// The settled copy under `key` in `root`, if there is one, stamped as used
/// and held for the rest of the process.
///
/// Looked for under the key's lock held shared, which is what a removal
/// waits on, so a removal either sees the stamp or has already happened. A
/// model directory this process cannot write -- a lock it cannot open or
/// take -- still finds a copy that is already there. Nothing can remove it
/// from such a directory either, so it needs no hold.
fn find(root: &Path, key: &str) -> Option<PathBuf> {
    let done = root.join(key);
    let Ok(shared) = open(&root.join(format!("{key}.lock"))) else {
        return settled(&done);
    };
    if shared.lock_shared().is_err() {
        return settled(&done);
    }
    let found = settled(&done)?;
    stamp(&done);
    hold(&done, shared);
    Some(found)
}

/// Opens a key's lock file, creating it if it is not there.
fn open(lock: &Path) -> std::io::Result<File> {
    std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock)
}

fn held() -> std::sync::MutexGuard<'static, BTreeMap<PathBuf, File>> {
    HELD.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Keeps `lock`, held shared on the copy at `done`, until the process exits.
fn hold(done: &Path, lock: File) {
    held().entry(done.to_path_buf()).or_insert(lock);
}

/// Records that the copy at `done` was just loaded.
///
/// Best effort: a stamp that cannot be written leaves the copy to be judged
/// by when it was written and last mapped, and the hold covers this process.
fn stamp(done: &Path) {
    let stamped =
        File::create(done.join(USED)).and_then(|used| used.set_modified(SystemTime::now()));
    if let Err(error) = stamped {
        tracing::debug!(copy = %done.display(), %error, "could not stamp a mapped copy as used");
    }
}

/// When the copy at `done` was last loaded, as far as its files can say: the
/// latest of its stamp, its graph's modification time and its data's access
/// time. `None` if none of them can be read, which is never taken for old.
fn last_used(done: &Path) -> Option<SystemTime> {
    let time = |file: &str, read: fn(&std::fs::Metadata) -> std::io::Result<SystemTime>| {
        std::fs::metadata(done.join(file))
            .and_then(|metadata| read(&metadata))
            .ok()
    };
    [
        time(USED, std::fs::Metadata::modified),
        time(MODEL, std::fs::Metadata::modified),
        time(DATA, std::fs::Metadata::accessed),
    ]
    .into_iter()
    .flatten()
    .max()
}

/// Whether the copy at `done` has gone [`UNUSED_FOR`] without a load, at `now`.
fn unused(done: &Path, now: SystemTime) -> bool {
    last_used(done)
        .and_then(|last| now.duration_since(last).ok())
        .is_some_and(|idle| idle >= UNUSED_FOR)
}

/// Removes, from `root`, every copy that is not the one this runtime and CPU
/// load for a model recorded there, that has gone [`UNUSED_FOR`] without a
/// load and that no process holds, every partial directory no writer
/// holds, and every record partial file [`RECORD_PARTIAL_FOR`] old. See the
/// module's documentation for the rule and why it is safe.
///
/// Best effort, entry by entry: what cannot be removed now -- a copy Windows
/// will not let go of, say -- is warned about and tried again on the next
/// load, and stops nothing else from being removed.
fn collect(root: &Path, now: SystemTime) {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            tracing::warn!(root = %root.display(), %error, "could not list the mapped copies");
            return;
        }
    };
    let current = current(root);
    for entry in entries.flatten() {
        let path = entry.path();
        if let Err(error) = collect_one(root, &path, &current, now, false) {
            tracing::warn!(copy = %path.display(), %error, "could not remove an unused mapped copy");
        }
    }
}

/// The keys of the copies this runtime and CPU load for the models recorded
/// in `root`, each model's by the record written last -- its latest
/// revision: the ones a collection never removes. Another revision's copy
/// is kept by the rest of the rule, while it is held or used.
fn current(root: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return BTreeSet::new();
    };
    let mut latest: BTreeMap<String, (SystemTime, PathBuf)> = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(model) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(record_model)
        else {
            continue;
        };
        let Some(written) = entry.metadata().ok().and_then(|meta| meta.modified().ok()) else {
            continue;
        };
        if latest
            .get(model)
            .is_none_or(|(newest, _)| written > *newest)
        {
            latest.insert(model.to_string(), (written, path));
        }
    }
    latest
        .into_values()
        .filter_map(|(_, path)| Source::parse(&std::fs::read_to_string(path).ok()?))
        .map(|source| source.key().0)
        .collect()
}

/// `superseded`: the copy's record now names another revision, so it goes
/// as soon as no process holds it, without waiting [`UNUSED_FOR`].
fn collect_one(
    root: &Path,
    path: &Path,
    current: &BTreeSet<String>,
    now: SystemTime,
    superseded: bool,
) -> std::io::Result<()> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(());
    };
    // A record's partial file (`<record>.<key>.<writer>.partial`), not a
    // copy's: each writer's own, so it is never resumed, only left behind.
    if name
        .strip_suffix(".partial")
        .is_some_and(|stem| stem.contains(".source."))
    {
        let stale = std::fs::metadata(path)
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|written| now.duration_since(written).ok())
            .is_some_and(|age| age >= RECORD_PARTIAL_FOR);
        return if stale && path.is_file() {
            std::fs::remove_file(path)
        } else {
            Ok(())
        };
    }
    let (key, partial) = match name.strip_suffix(".partial") {
        Some(key) => (key, true),
        None => (name, false),
    };
    let is_key = key.len() == 32
        && key
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    let idle = |path: &Path| superseded || unused(path, now);
    if !is_key || !path.is_dir() || (!partial && (current.contains(key) || !idle(path))) {
        return Ok(());
    }

    let lock = open(&root.join(format!("{key}.lock")))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Ok(()),
        Err(std::fs::TryLockError::Error(error)) => return Err(error),
    }
    if partial {
        return std::fs::remove_dir_all(path);
    }
    // Again under the lock: a load may have stamped it since.
    if !idle(path) {
        return Ok(());
    }
    let aside = root.join(format!("{key}.partial"));
    if aside.exists() {
        std::fs::remove_dir_all(&aside)?;
    }
    std::fs::rename(path, &aside)?;
    std::fs::remove_dir_all(&aside)?;
    tracing::info!(
        copy = %path.display(),
        superseded,
        "removed a mapped copy this runtime and CPU do not load, which nothing holds"
    );
    Ok(())
}

/// The graph to load from a complete copy whose fusion has been settled, or
/// `None` if either is still to do.
fn settled(copy: &Path) -> Option<PathBuf> {
    let fused = copy.join(FUSED);
    if fused.exists() {
        return Some(fused);
    }
    (copy.join(UNFUSED).exists() && copy.join(MODEL).exists()).then(|| copy.join(MODEL))
}

/// Fuses the attention of the complete copy in `copy`, keeps the result only
/// if it scores a probe exactly as the unfused graph does, and returns the
/// graph to load. Called under the copy's lock.
///
/// Never fails the copy. Whatever goes wrong -- nothing to fuse, a probe
/// that differs, a graph the runtime refuses -- is written to [`UNFUSED`]
/// and logged, and the copy loads the graph ONNX Runtime wrote, which is what
/// it loaded before this existed.
fn fuse(copy: &Path, threads: usize) -> PathBuf {
    let model = copy.join(MODEL);
    let started = std::time::Instant::now();
    match try_fuse(copy, threads) {
        Ok((layers, (before, after))) => {
            tracing::info!(
                copy = %copy.display(),
                layers,
                nodes_before = before,
                nodes_after = after,
                seconds = started.elapsed().as_secs_f64(),
                "fused the attention of a mapped copy; it scored a probe bit-identically"
            );
            copy.join(FUSED)
        }
        Err(reason) => {
            tracing::info!(copy = %copy.display(), %reason, "left the attention of a mapped copy unfused");
            // Best effort: without it the next load asks again, which costs
            // time and changes nothing it loads.
            let _ = std::fs::write(copy.join(UNFUSED), format!("{reason}\n"));
            model
        }
    }
}

/// The fusion itself: the rewrite, written beside the graph it came from,
/// checked, and renamed into place only once it has passed.
fn try_fuse(copy: &Path, threads: usize) -> std::result::Result<(usize, (usize, usize)), String> {
    let model = copy.join(MODEL);
    let original = std::fs::read(&model).map_err(|error| format!("reading the graph: {error}"))?;
    let fused = crate::attention::fuse(&original)?;
    let candidate = copy.join(format!("{FUSED}.partial"));
    let written = std::fs::write(&candidate, &fused.model)
        .and_then(|()| std::fs::File::open(&candidate)?.sync_all())
        .map_err(|error| format!("writing the fused graph: {error}"))
        .and_then(|()| same_on_a_probe(&model, &candidate, threads))
        .and_then(|()| {
            std::fs::rename(&candidate, copy.join(FUSED))
                .map_err(|error| format!("renaming the fused graph into place: {error}"))
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&candidate);
    }
    written.map(|()| (fused.layers, fused.nodes))
}

/// `graph`, a graph of a copy this process holds, with its word embeddings
/// split out where that has scored a probe identically; otherwise `graph`.
///
/// Settled on the first load that asks, as the fusion is, but not under the
/// copy's lock: the load already holds it shared, as every process that loads
/// the copy does, and taking it exclusively would wait on those holds. Nothing
/// needs it. The split is written under names of this writer's own and renamed
/// into place, its description before its graph, so a split graph that exists
/// has its description; two processes splitting one copy at once write the
/// same bytes, and the second either replaces the first's or -- where Windows
/// will not replace a file another process has open -- finds it there and
/// keeps it. The copy cannot be collected meanwhile, since this process holds
/// it, and what a writer that stopped part way leaves goes with the copy.
///
/// Never fails the load. Whatever goes wrong -- nothing to split, a probe that
/// differs, a directory that cannot be written -- is written to the `unsplit`
/// file where it can be, and logged, and the copy loads `graph`, which is what
/// it loaded before this existed. Turning
/// [`split_embeddings`](crate::InferenceSettings::split_embeddings) off asks
/// for `graph` without looking.
fn split(graph: PathBuf, settings: &crate::InferenceSettings) -> PathBuf {
    if !settings.split_embeddings {
        return graph;
    }
    let split = named(&graph, "onnx");
    if split.exists() {
        return split;
    }
    let unsplit = named(&graph, "unsplit");
    if unsplit.exists() {
        return graph;
    }
    let Ok(threads) = settings.budget() else {
        return graph;
    };
    let started = std::time::Instant::now();
    match try_split(&graph, threads) {
        Ok(table) => {
            tracing::info!(
                graph = %graph.display(),
                rows = table.rows,
                width = table.width,
                seconds = started.elapsed().as_secs_f64(),
                "split the word embeddings out of a mapped copy; it scored a probe bit-identically"
            );
            split
        }
        Err(reason) => {
            tracing::info!(graph = %graph.display(), %reason, "left the word embeddings of a mapped copy in its graph");
            // Best effort: without it the next load asks again, which costs
            // time and changes nothing it loads.
            let _ = std::fs::write(unsplit, format!("{reason}\n"));
            graph
        }
    }
}

/// `graph`'s split file with `extension`: `model.onnx` has
/// `model.split1.onnx`, `model.split1.unsplit`.
fn named(graph: &Path, extension: &str) -> PathBuf {
    graph.with_extension(format!("{SPLIT}.{extension}"))
}

/// The split itself: the rewrite and its description, written beside the
/// graph they came from under this writer's own names, checked, and renamed
/// into place only once they have passed. Returns where the table is.
fn try_split(
    graph: &Path,
    threads: usize,
) -> std::result::Result<crate::word_embeddings::Table, String> {
    let original = std::fs::read(graph).map_err(|error| format!("reading the graph: {error}"))?;
    let split = crate::word_embeddings::split(&original)?;
    // `model.split1.<writer>.onnx`, whose description the probe finds beside
    // it as a load finds the final one's.
    static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let candidate = graph.with_extension(format!(
        "{SPLIT}.{}-{}.onnx",
        std::process::id(),
        WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let described = crate::word_embeddings::description(&candidate);
    let final_graph = named(graph, "onnx");
    let written = [
        (&described, split.table.encode().into_bytes()),
        (&candidate, split.model),
    ]
    .into_iter()
    .try_for_each(|(path, bytes)| {
        std::fs::write(path, bytes).and_then(|()| std::fs::File::open(path)?.sync_all())
    })
    .map_err(|error| format!("writing the split graph: {error}"))
    .and_then(|()| same_on_a_probe(graph, &candidate, threads))
    .and_then(|()| {
        place(
            &described,
            &crate::word_embeddings::description(&final_graph),
        )
    })
    .and_then(|()| place(&candidate, &final_graph));
    // Gone already once renamed.
    let _ = std::fs::remove_file(&described);
    let _ = std::fs::remove_file(&candidate);
    written.map(|()| split.table)
}

/// Renames `from` to `to`, or keeps the `to` another writer of the same bytes
/// put there first if it cannot be replaced.
fn place(from: &Path, to: &Path) -> std::result::Result<(), String> {
    match std::fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(_) if to.exists() => Ok(()),
        Err(error) => Err(format!("renaming {} into place: {error}", to.display())),
    }
}

/// Whether two graphs return the same bits on a probe, as the product runs
/// them: a CPU session at [`LEVEL`], and every output compared.
///
/// Two rows, the second padded, because the padding is what the fused
/// kernel reads differently -- a key padding mask where the graph it
/// replaces added a bias -- and a probe with none would not exercise it.
/// A split graph's word embeddings are read for its ids by
/// `crate::word_embeddings::WordTable`, the reader the encoder feeds it
/// from, so the probe checks that reader as well as the rewrite; its ids
/// include the first rows and repeats, as padding does.
/// The token ids are arbitrary: any ids below a thousand are in every
/// vocabulary these models have, and the question is whether the two graphs
/// agree, not what either one thinks of the text.
fn same_on_a_probe(
    expected: &Path,
    candidate: &Path,
    threads: usize,
) -> std::result::Result<(), String> {
    const LENGTH: usize = 24;
    const PADDED_FROM: usize = 13;
    let ids: Vec<i64> = (0..2 * LENGTH)
        .map(|at| match at % LENGTH {
            0 => 0,
            position if at >= LENGTH && position >= PADDED_FROM => 1,
            position if position == LENGTH - 1 || (at >= LENGTH && position == PADDED_FROM - 1) => {
                2
            }
            position => 5 + (at as i64 * 37 + position as i64 * 11) % 900,
        })
        .collect();
    let mask: Vec<i64> = (0..2 * LENGTH)
        .map(|at| i64::from(at < LENGTH || at % LENGTH < PADDED_FROM))
        .collect();

    let outputs = |path: &Path| -> std::result::Result<Vec<(String, Vec<u32>)>, String> {
        let failed = |error: &dyn std::fmt::Display| format!("{}: {error}", path.display());
        let mut session = crate::inference::probe(threads, path).map_err(|error| failed(&error))?;
        // A split graph's word embeddings, read as the encoder reads them.
        let table =
            crate::word_embeddings::WordTable::beside(path).map_err(|error| failed(&error))?;
        type Input = ort::session::SessionInputValue<'static>;
        let mut feed: Vec<(std::borrow::Cow<'static, str>, Input)> = Vec::new();
        for input in session.inputs() {
            let tokens = |values: Vec<i64>| -> std::result::Result<Input, ort::Error> {
                ort::value::Tensor::from_array(([2, LENGTH], values)).map(Into::into)
            };
            let value = match (input.name(), &table) {
                ("input_ids", _) => tokens(ids.clone()),
                ("attention_mask", _) => tokens(mask.clone()),
                ("token_type_ids", _) => tokens(vec![0; ids.len()]),
                (name, Some(table)) if name == table.input() => {
                    let rows = table.embed(&ids).map_err(|error| failed(&error))?;
                    ort::value::Tensor::from_array(([2, LENGTH, table.width()], rows))
                        .map(Into::into)
                }
                (other, _) => return Err(failed(&format!("no probe for an input named {other}"))),
            }
            .map_err(|error| failed(&error))?;
            feed.push((input.name().to_string().into(), value));
        }
        let names: Vec<String> = session
            .outputs()
            .iter()
            .map(|output| output.name().to_string())
            .collect();
        let ran = session.run(feed).map_err(|error| failed(&error))?;
        names
            .into_iter()
            .map(|name| {
                let (_, values) = ran[name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|error| failed(&format!("output {name}: {error}")))?;
                Ok((name, values.iter().map(|value| value.to_bits()).collect()))
            })
            .collect()
    };

    let expected = outputs(expected)?;
    let candidate = outputs(candidate)?;
    if expected != candidate {
        let differing = expected
            .iter()
            .zip(&candidate)
            .map(|((name, one), (_, other))| {
                let count = one
                    .iter()
                    .zip(other)
                    .filter(|(one, other)| one != other)
                    .count();
                format!("{name}: {count} of {} values", one.len())
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "the rewritten graph scored the probe differently ({differing})"
        ));
    }
    Ok(())
}

/// Writes the optimized graph and its external data into `into`.
///
/// A CPU session over `source` built the way a loaded model's is -- the same
/// execution provider and optimization level -- and told to save what it
/// optimized, with every initializer of a kilobyte or more, packed ones
/// included, in the data file. Committing the session is what writes; it is
/// dropped straight after. Both files are flushed before the caller renames
/// the directory, so a power cut cannot leave a complete-looking copy whose
/// data never reached the disk.
fn write(source: &Path, into: &Path) -> Result<()> {
    fn failed(error: impl std::fmt::Display) -> IndexError {
        IndexError::Engine(format!("writing a mapped copy: {error}"))
    }
    let session = ort::session::Session::builder()
        .map_err(failed)?
        .with_execution_providers([crate::inference::cpu()])
        .map_err(failed)?
        .with_optimization_level(LEVEL)
        .map_err(failed)?
        .with_optimized_model_path(into.join(MODEL))
        .map_err(failed)?
        .with_config_entry(
            "session.optimized_model_external_initializers_file_name",
            DATA,
        )
        .map_err(failed)?
        .with_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes",
            "1024",
        )
        .map_err(failed)?
        .with_config_entry("session.save_external_prepacked_constant_initializers", "1")
        .map_err(failed)?
        .commit_from_file(source)
        .map_err(failed)?;
    drop(session);

    // A graph with nothing large enough to externalize writes no data file.
    // That copy would load, but onto the heap like the source, so it is
    // refused here rather than kept for nothing.
    for file in [MODEL, DATA] {
        std::fs::OpenOptions::new()
            .write(true)
            .open(into.join(file))
            .and_then(|written| written.sync_all())
            .map_err(|error| failed(format!("{file}: {error}")))?;
    }
    Ok(())
}

/// What identifies a copy's source: the name of the file it resolves to --
/// for a hub download that is the blob, named by the hash of its content --
/// with its length and modification time, so a file replaced in place under
/// the same name is a different source.
///
/// A file whose name is not a content hash -- where symlinks are
/// unavailable the hub client moves the blob into the snapshot under the
/// file's own name -- is named by its SHA-256 as well: a moved ref's weights
/// can have the old file's name, length and time on a filesystem with coarse
/// times, and would key as the old copy. That costs one read of the file
/// per load on that path only; a blob's name already is its digest.
#[derive(Debug, PartialEq, Eq)]
struct Source {
    name: String,
    length: u64,
    modified: u128,
    /// The content's SHA-256, for a source not named by it.
    digest: Option<String>,
}

/// Whether a file named `name` is named by its content, as a hub blob is:
/// a git object id (40 hex digits) or an LFS SHA-256 (64).
fn content_named(name: &str) -> bool {
    matches!(name.len(), 40 | 64) && name.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl Source {
    fn of(source: &Path) -> Result<Self> {
        let resolved = std::fs::canonicalize(source)?;
        let metadata = std::fs::metadata(&resolved)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |since| since.as_nanos());
        let name = resolved
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        let digest = if content_named(&name) {
            None
        } else {
            Some(source_digest(&resolved)?)
        };
        Ok(Self {
            name,
            length: metadata.len(),
            modified,
            digest,
        })
    }

    /// The directory name a copy of this source is kept under, and what it was
    /// derived from, for the log.
    ///
    /// The source, and everything that decides whether a copy written here is
    /// valid there: the runtime's build, the architecture and its matrix
    /// features, and the optimization level. Hashed because the build string
    /// alone is longer than some filesystems allow a name to be.
    fn key(&self) -> (String, String) {
        let described = format!(
            "source {} {} bytes modified {}{}; runtime 1.{} {}; {} {}; {LEVEL:?}",
            self.name,
            self.length,
            self.modified,
            self.digest
                .as_ref()
                .map_or(String::new(), |digest| format!(" sha256 {digest}")),
            ort::MINOR_VERSION,
            ort::info(),
            std::env::consts::ARCH,
            features().join(","),
        );
        let digest = Sha256::digest(described.as_bytes());
        let key = digest[..16]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        (key, described)
    }

    /// One line each, as [`release`] writes it.
    fn record(&self) -> String {
        format!("{}\n{}\n{}\n", self.name, self.length, self.modified)
    }

    /// Preserve byte provenance before release deletes the original file.
    /// The first three metadata lines and cache key stay compatible with old records.
    fn record_with_digest(&self, source: &Path) -> Result<String> {
        let digest = match &self.digest {
            Some(digest) => digest.clone(),
            None => source_digest(source)?,
        };
        Ok(format!("{}{DIGEST}{digest}\n", self.record()))
    }

    fn parse(record: &str) -> Option<Self> {
        let mut lines = record.lines();
        let name = lines.next()?.to_string();
        let length = lines.next()?.parse().ok()?;
        let modified = lines.next()?.parse().ok()?;
        // A source not named by its content is keyed by the digest its
        // record names; a record without one cannot find its copy.
        let digest = if content_named(&name) {
            None
        } else {
            Some(
                lines
                    .find_map(|line| line.strip_prefix(DIGEST))?
                    .to_string(),
            )
        };
        (!name.is_empty()).then_some(Self {
            name,
            length,
            modified,
            digest,
        })
    }

    /// Gives a fresh download of this source this source's modification time,
    /// so it keys exactly as the removed file did -- only when its bytes are
    /// the ones the record's `digest` names.
    ///
    /// Left changed, every copy of the model would stop fitting -- including
    /// those of another runtime or CPU sharing the directory, which would
    /// then fetch and write again in turn, each undoing the other. The name
    /// and length alone do not say it is the same file: a hub blob is named
    /// by its content, but where symlinks are unavailable the hub client
    /// moves the blob into the snapshot under the file's own name, so a
    /// moved ref's weights of the same length look like the old ones. Their
    /// old time would key them as the old copy. A different file, or a
    /// record with no digest to check against, is left as it is and keys as
    /// new. That costs one read of the download per fetch.
    fn restore(&self, fetched: &Path, digest: Option<&str>) {
        let restored = (|| -> Result<bool> {
            let now = Self::of(fetched)?;
            if now.name != self.name || now.length != self.length || now.modified == self.modified {
                return Ok(false);
            }
            let fetched_digest = match &now.digest {
                Some(digest) => digest.clone(),
                None => source_digest(fetched)?,
            };
            if digest != Some(fetched_digest.as_str()) {
                return Ok(false);
            }
            let at = std::time::UNIX_EPOCH
                + std::time::Duration::new(
                    (self.modified / 1_000_000_000) as u64,
                    (self.modified % 1_000_000_000) as u32,
                );
            std::fs::OpenOptions::new()
                .write(true)
                .open(std::fs::canonicalize(fetched)?)?
                .set_modified(at)?;
            Ok(true)
        })();
        if let Err(error) = restored {
            tracing::warn!(
                source = %fetched.display(),
                %error,
                "could not give a fresh download its old time; its copies will be written again"
            );
        }
    }
}

/// Content identity without keeping the complete model on the heap.
pub(crate) fn source_digest(source: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    std::io::copy(&mut File::open(source)?, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

/// The CPU features that decide how the matrix kernels pack a weight.
///
/// Named rather than read as raw CPUID, because raw leaves also carry
/// microcode mitigation bits that change with an update and would orphan every
/// copy for nothing. AMX is read from CPUID directly: `std` does not yet
/// detect it on a stable toolchain. That reads what the CPU has, not whether
/// the kernel lets this process use it, which is the one case where two
/// machines could share a key and pack differently -- and the cost of that is
/// the heap packing described above, not a wrong score.
#[cfg(target_arch = "x86_64")]
pub(crate) fn features() -> Vec<&'static str> {
    use std::arch::is_x86_feature_detected as has;
    let mut found: Vec<&'static str> = [
        ("avx", has!("avx")),
        ("avx2", has!("avx2")),
        ("fma", has!("fma")),
        ("f16c", has!("f16c")),
        ("avx512f", has!("avx512f")),
        ("avx512bw", has!("avx512bw")),
        ("avx512dq", has!("avx512dq")),
        ("avx512vl", has!("avx512vl")),
        ("avx512vnni", has!("avx512vnni")),
        ("avx512bf16", has!("avx512bf16")),
        ("avx512fp16", has!("avx512fp16")),
        ("avxvnni", has!("avxvnni")),
        ("avxvnniint8", has!("avxvnniint8")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect();
    if std::arch::x86_64::__cpuid(0).eax >= 7 {
        let extended = std::arch::x86_64::__cpuid_count(7, 0).edx;
        for (bit, name) in [(22, "amx-bf16"), (24, "amx-tile"), (25, "amx-int8")] {
            if extended >> bit & 1 == 1 {
                found.push(name);
            }
        }
    }
    found
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn features() -> Vec<&'static str> {
    use std::arch::is_aarch64_feature_detected as has;
    [
        ("neon", has!("neon")),
        ("fp16", has!("fp16")),
        ("dotprod", has!("dotprod")),
        ("i8mm", has!("i8mm")),
        ("bf16", has!("bf16")),
        ("sve", has!("sve")),
        ("sve2", has!("sve2")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) fn features() -> Vec<&'static str> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// A hub that is a file on this disk: `fetch` copies `origin` into place
    /// as a download would, and fails while `online` is false.
    struct Hub {
        origin: PathBuf,
        blob: std::cell::RefCell<PathBuf>,
        online: Cell<bool>,
        fetches: Cell<usize>,
        revision: std::cell::RefCell<String>,
    }

    impl Hub {
        fn new(dir: &Path, model: &[u8]) -> Self {
            let origin = dir.join("origin.onnx");
            std::fs::write(&origin, model).expect("write the hub's copy");
            Self {
                origin,
                blob: std::cell::RefCell::new(dir.join("models").join("blobs").join("0123abcd")),
                online: Cell::new(true),
                fetches: Cell::new(0),
                revision: std::cell::RefCell::new("someone/model@c0ffee".into()),
            }
        }
    }

    impl Hub {
        /// Moves the ref to `revision`, whose weights are another file: a
        /// hub blob is named by its content.
        fn moved(&self, revision: &str, blob: &str) {
            *self.revision.borrow_mut() = revision.into();
            let path = self.blob.borrow().with_file_name(blob);
            *self.blob.borrow_mut() = path;
        }
    }

    impl Download for Hub {
        fn label(&self) -> String {
            "someone/model/onnx/model.onnx".into()
        }

        fn revision(&self) -> String {
            self.revision.borrow().clone()
        }

        fn on_disk(&self) -> Option<PathBuf> {
            let blob = self.blob.borrow();
            blob.exists().then(|| blob.clone())
        }

        fn fetch(&self) -> Result<PathBuf> {
            if let Some(blob) = self.on_disk() {
                return Ok(blob);
            }
            if !self.online.get() {
                return Err(IndexError::Engine("the hub cannot be reached".into()));
            }
            let blob = self.blob.borrow().clone();
            std::fs::create_dir_all(blob.parent().expect("a parent"))?;
            std::fs::copy(&self.origin, &blob)?;
            self.fetches.set(self.fetches.get() + 1);
            Ok(blob)
        }

        fn remove(&self) -> Result<bool> {
            std::fs::remove_file(&*self.blob.borrow())?;
            Ok(true)
        }
    }

    /// The smallest model a copy can be written of: one `MatMul` by a 32x32
    /// weight, which at 4 kB is large enough to go to the data file.
    fn tiny_model() -> Vec<u8> {
        fn varint(out: &mut Vec<u8>, mut value: u64) {
            while value >= 0x80 {
                out.push(value as u8 | 0x80);
                value >>= 7;
            }
            out.push(value as u8);
        }
        fn number(out: &mut Vec<u8>, field: u64, value: u64) {
            varint(out, field << 3);
            varint(out, value);
        }
        fn bytes(out: &mut Vec<u8>, field: u64, value: &[u8]) {
            varint(out, field << 3 | 2);
            varint(out, value.len() as u64);
            out.extend_from_slice(value);
        }
        fn tensor(name: &str, dims: &[u64]) -> Vec<u8> {
            let mut shape = Vec::new();
            for dim in dims {
                let mut value = Vec::new();
                number(&mut value, 1, *dim);
                bytes(&mut shape, 1, &value);
            }
            let mut tensor_type = Vec::new();
            number(&mut tensor_type, 1, 1); // float
            bytes(&mut tensor_type, 2, &shape);
            let mut ty = Vec::new();
            bytes(&mut ty, 1, &tensor_type);
            let mut info = Vec::new();
            bytes(&mut info, 1, name.as_bytes());
            bytes(&mut info, 2, &ty);
            info
        }

        let mut node = Vec::new();
        bytes(&mut node, 1, b"x");
        bytes(&mut node, 1, b"w");
        bytes(&mut node, 2, b"y");
        bytes(&mut node, 3, b"project");
        bytes(&mut node, 4, b"MatMul");
        let mut weight = Vec::new();
        number(&mut weight, 1, 32);
        number(&mut weight, 1, 32);
        number(&mut weight, 2, 1); // float
        bytes(&mut weight, 8, b"w");
        let raw: Vec<u8> = (0..32 * 32)
            .flat_map(|at| (at as f32 / 1024.0).to_le_bytes())
            .collect();
        bytes(&mut weight, 9, &raw);
        let mut graph = Vec::new();
        bytes(&mut graph, 1, &node);
        bytes(&mut graph, 2, b"tiny");
        bytes(&mut graph, 5, &weight);
        bytes(&mut graph, 11, &tensor("x", &[1, 32]));
        bytes(&mut graph, 12, &tensor("y", &[1, 32]));
        let mut opset = Vec::new();
        bytes(&mut opset, 1, b"");
        number(&mut opset, 2, 13);
        let mut model = Vec::new();
        number(&mut model, 1, 7);
        bytes(&mut model, 8, &opset);
        bytes(&mut model, 7, &graph);
        model
    }

    /// Loads `path` the way a model is loaded, which is what [`release`]
    /// waits for.
    fn load(path: &Path) {
        crate::inference::session(1, vec![crate::inference::cpu()], || Ok(path.to_path_buf()))
            .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
    }

    /// A copy is found by its record only for the revision it was written
    /// from: once the ref moves, the new revision is fetched and written
    /// rather than the old weights loaded under the new ref's name.
    #[test]
    fn a_moved_ref_does_not_load_the_old_revision_s_copy() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let copy =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("the first load");
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(hub.on_disk().is_none());
        let record = std::fs::read_to_string(cache.join("prepared").join(record_name(&hub)))
            .expect("the record");
        assert!(
            record.ends_with("revision someone/model@c0ffee\n"),
            "{record}"
        );

        hub.online.set(false);
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default())
                .expect("the same revision"),
            copy
        );
        hub.moved("someone/model@decaf", "4567cdef");
        assert!(
            !is_ready(&hub, &cache, &crate::InferenceSettings::default()),
            "the old revision's copy is ready"
        );
        assert!(
            load_path(&hub, &cache, &crate::InferenceSettings::default()).is_err(),
            "the old revision's copy loaded under the new ref"
        );
        hub.online.set(true);
        let new = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("a load of the new revision");
        assert_eq!(hub.fetches.get(), 2, "the new revision was not fetched");

        // A replaced revision's copy goes as soon as its record names
        // another revision -- unless a process holds it, as this one holds
        // the first copy, which then waits for the usual sweep.
        load(&new);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(copy.exists(), "a held copy was removed");
        held().remove(new.parent().expect("the copy's directory"));
        hub.moved("someone/model@beef", "89abef01");
        let newest = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("a load of the next revision");
        assert_eq!(hub.fetches.get(), 3, "the next revision was not fetched");
        load(&newest);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(!new.exists(), "the replaced, unheld copy is still on disk");
        assert!(newest.exists() && copy.exists());
    }

    /// A source named by its content keys by its name as before, without
    /// reading it; one that is not -- a hub file moved into its snapshot
    /// where symlinks are unavailable -- keys by its SHA-256 too, so two
    /// files with one name, length and time but other bytes are two copies.
    #[test]
    fn a_source_not_named_by_its_content_is_keyed_by_it() {
        let dir = tempfile::tempdir().expect("temp dir");
        let at = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let file = |path: &Path, bytes: &[u8]| {
            std::fs::create_dir_all(path.parent().expect("a parent")).unwrap();
            std::fs::write(path, bytes).unwrap();
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(at)
                .unwrap();
            Source::of(path).unwrap()
        };

        let blob = file(&dir.path().join("blobs").join("a".repeat(64)), b"weights");
        assert_eq!(blob.digest, None, "a blob was read to key it");
        assert!(!blob.key().1.contains("sha256"), "{}", blob.key().1);

        let old = file(&dir.path().join("old/model.onnx"), b"weights");
        let new = file(&dir.path().join("new/model.onnx"), b"wEights");
        assert_eq!(
            (&old.name, old.length, old.modified),
            (&new.name, new.length, new.modified)
        );
        assert_ne!(old.key().0, new.key().0, "other bytes keyed as one copy");

        // Its record keys as it does, and one without the digest finds no
        // copy rather than one of other bytes.
        let path = dir.path().join("old/model.onnx");
        let recorded = Source::parse(&old.record_with_digest(&path).unwrap()).unwrap();
        assert_eq!(recorded.key().0, old.key().0);
        assert_eq!(Source::parse(&old.record()), None);
    }

    /// Where symlinks are unavailable, the hub client moves a blob into the
    /// snapshot under the file's own name, so a moved ref's weights can have
    /// the old file's name and length. They are not given the old file's
    /// time, which would key them as its copy, unless their bytes are the
    /// ones the record names.
    #[test]
    fn same_named_same_length_weights_are_not_keyed_as_the_old_copy() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = tiny_model();
        let hub = Hub::new(dir.path(), &model);
        let cache = dir.path().join("models");
        let copy =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("the first load");
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());

        // Another weight, the same length.
        let raw: Vec<u8> = (0..32 * 32)
            .flat_map(|at| (at as f32 / 1024.0).to_le_bytes())
            .collect();
        let weights = model
            .windows(raw.len())
            .position(|window| window == raw.as_slice())
            .expect("the weight's bytes");
        let mut other = model.clone();
        other[weights + 4] ^= 0x01;
        assert_eq!(other.len(), model.len());
        std::fs::write(&hub.origin, &other).expect("the new revision's weights");
        let blob = hub.blob.borrow().file_name().expect("a name").to_owned();
        hub.moved("someone/model@decaf", &blob.to_string_lossy());

        let new = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("a load of the new revision");
        assert_eq!(hub.fetches.get(), 2);
        assert_ne!(new, copy, "the new weights were keyed as the old copy");
        load(&new);
    }

    /// A record written before records named a revision cannot say which
    /// revision its copy is of, so it is not trusted to find the copy: the
    /// download is fetched again, and the same file keys to the same copy,
    /// which is not written again.
    #[test]
    fn a_record_without_a_revision_is_verified_by_a_fetch() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let copy =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("the first load");
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        let path = cache.join("prepared").join(record_name(&hub));
        let record = std::fs::read_to_string(&path).expect("the record");
        let legacy: String = record
            .lines()
            .filter(|line| !line.starts_with(REVISION))
            .map(|line| format!("{line}\n"))
            .collect();
        // As an older build wrote it: one record of no revision, under the
        // model's name alone.
        let old = cache.join("prepared").join(legacy_record_name(&hub));
        std::fs::write(&old, legacy).expect("an old record");
        std::fs::remove_file(&path).expect("no record of this revision");

        hub.online.set(false);
        assert!(
            !is_ready(&hub, &cache, &crate::InferenceSettings::default()),
            "an unverified copy is ready"
        );
        assert!(
            load_path(&hub, &cache, &crate::InferenceSettings::default()).is_err(),
            "an unverified copy loaded"
        );
        hub.online.set(true);
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("a verified load"),
            copy
        );
        assert_eq!(hub.fetches.get(), 2);
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert_eq!(std::fs::read_to_string(&path).expect("the record"), record);
        assert!(
            !old.exists(),
            "the old record outlived the one replacing it"
        );
    }

    /// Two revisions of one model released in turn -- as two processes on
    /// either side of a ref move would -- each keep a record of their own:
    /// neither is written over by the other, and each finds its own copy
    /// offline while that copy is held.
    #[test]
    fn each_revision_keeps_its_own_record() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let first = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("the first revision");
        load(&first);
        hub.moved("someone/model@decaf", "4567cdef");
        let second = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("the second revision");
        load(&second);
        // Both downloads are on disk and both copies are held, as in two
        // processes loading at once; the second revision releases first.
        release(&hub, &cache, &crate::InferenceSettings::default());
        hub.moved("someone/model@c0ffee", "0123abcd");
        release(&hub, &cache, &crate::InferenceSettings::default());
        let records = records(&hub, &cache.join("prepared"));
        assert_eq!(records.len(), 2, "{records:?}");

        hub.online.set(false);
        let fetches = hub.fetches.get();
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default())
                .expect("the first, offline"),
            first
        );
        hub.moved("someone/model@decaf", "4567cdef");
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default())
                .expect("the second, offline"),
            second
        );
        assert_eq!(hub.fetches.get(), fetches);
    }

    /// A revision whose weights are another's -- only its tokenizer or
    /// configuration changed, so both name one blob and one copy -- does
    /// not take the other's record with it when it releases: the other
    /// revision, loaded pinned and offline, still finds the copy.
    #[test]
    fn revisions_sharing_one_copy_keep_their_records() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let copy = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("the first revision");
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(hub.on_disk().is_none());

        // The ref moves to a commit with the same weights: the same blob.
        hub.moved("someone/model@decaf", "0123abcd");
        let same = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("the second revision");
        assert_eq!(same, copy, "the same weights were written again");
        load(&same);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(hub.on_disk().is_none());
        let records = records(&hub, &cache.join("prepared"));
        assert_eq!(records.len(), 2, "{records:?}");

        hub.online.set(false);
        let fetches = hub.fetches.get();
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default())
                .expect("the second, offline"),
            copy
        );
        hub.moved("someone/model@c0ffee", "0123abcd");
        assert!(
            is_ready(&hub, &cache, &crate::InferenceSettings::default()),
            "the first revision lost its record"
        );
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default())
                .expect("the first, offline"),
            copy
        );
        assert_eq!(hub.fetches.get(), fetches);
    }

    /// The download's whole life: written into a copy and removed, the copy
    /// found without it, a copy that no longer fits refused offline with the
    /// reason, and the download fetched again and keyed as before once the
    /// network is back.
    #[test]
    fn a_download_goes_once_its_copy_loads_and_comes_back_when_a_copy_must_be_written() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let root = cache.join("prepared");

        // First load: fetched, written into a copy, loaded, then removed.
        let copy =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("the first load");
        assert_eq!(hub.fetches.get(), 1);
        assert!(copy.starts_with(&root), "loaded {copy:?}, not a copy");
        assert!(is_copy(&copy, &cache));
        assert!(
            hub.on_disk().is_some(),
            "the download went before the copy had loaded"
        );
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(
            hub.on_disk().is_none(),
            "the download is still beside its copy"
        );
        let key = copy
            .parent()
            .and_then(Path::file_name)
            .expect("the copy's directory")
            .to_owned();

        // Offline, the copy is found by its record alone.
        hub.online.set(false);
        assert!(is_ready(&hub, &cache, &crate::InferenceSettings::default()));
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("a load offline"),
            copy
        );
        assert_eq!(hub.fetches.get(), 1, "a load with a fitting copy fetched");

        // What a runtime upgrade or another CPU looks like from here: the
        // only copy on disk is keyed for something else.
        std::fs::rename(root.join(&key), root.join("0".repeat(32))).expect("move the copy");
        assert!(!is_ready(
            &hub,
            &cache,
            &crate::InferenceSettings::default()
        ));
        let refused = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect_err("a load with no fitting copy and no network must fail")
            .to_string();
        assert!(
            refused.contains("someone/model/onnx/model.onnx")
                && refused.contains("download was removed")
                && refused.contains("the hub cannot be reached"),
            "the offline error does not say why: {refused}"
        );

        // Back online: fetched again, and keyed as the removed file was, so
        // a copy another runtime keeps for it would still fit.
        hub.online.set(true);
        let rewritten = load_path(&hub, &cache, &crate::InferenceSettings::default())
            .expect("a load once online");
        assert_eq!(hub.fetches.get(), 2);
        assert_eq!(
            rewritten, copy,
            "the fresh download keyed differently from the removed one"
        );
        load(&rewritten);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(hub.on_disk().is_none());
    }

    /// A copy that cannot be written falls back to the download, leaves
    /// nothing behind that a later load could mistake for a copy, and keeps
    /// the download: it is the only thing that loads.
    #[test]
    fn a_failed_copy_loads_the_download_and_keeps_it() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), b"this is not a protobuf");
        let cache = dir.path().join("models");

        let loaded =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("fall back");
        assert!(
            !is_copy(&loaded, &cache),
            "the download it fell back to is taken for a copy"
        );
        assert_eq!(Some(loaded), hub.on_disk());
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(
            hub.on_disk().is_some(),
            "the only loadable file was removed"
        );

        let (key, _) = Source::of(&hub.blob.borrow())
            .expect("key the source")
            .key();
        let root = cache.join("prepared");
        assert!(!root.join(&key).exists(), "a failed write left a copy");
        assert!(
            !root.join(format!("{key}.partial")).exists(),
            "a failed write left its partial directory"
        );
    }

    /// A file the runtime cannot read as a model.
    fn not_a_model(dir: &Path) -> PathBuf {
        let source = dir.join("broken.onnx");
        std::fs::write(&source, b"this is not a protobuf").expect("write the fake model");
        source
    }

    /// A partial directory found under the lock is a crashed writer's, and is
    /// cleared rather than written into or trusted.
    #[test]
    fn a_crashed_writers_partial_directory_is_cleared() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = not_a_model(dir.path());
        let root = dir.path().join("prepared");
        let (key, _) = Source::of(&source).expect("key the source").key();
        let partial = root.join(format!("{key}.partial"));
        std::fs::create_dir_all(&partial).expect("make a partial directory");
        std::fs::write(partial.join(MODEL), b"half a model").expect("half-write it");

        let before = crate::LoadActivity::now();
        assert!(prepare(&source, &root, 1).is_err());
        assert!(!partial.exists(), "the crashed writer's files survived");
        // Counted as preparing: the load did the work, if not to the end.
        assert!(crate::LoadActivity::now().prepared > before.prepared);
    }

    /// A complete copy is used as found, without writing another.
    ///
    /// The source here is not a model at all, so reaching the runtime would
    /// fail: returning the copy proves the lookup never got that far.
    #[test]
    fn a_complete_copy_is_reused() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = not_a_model(dir.path());
        let root = dir.path().join("prepared");
        let (key, _) = Source::of(&source).expect("key the source").key();
        std::fs::create_dir_all(root.join(&key)).expect("make the copy's directory");
        std::fs::write(root.join(&key).join(MODEL), b"a copy").expect("write the copy");

        assert_eq!(
            prepare(&source, &root, 1).expect("find the copy"),
            root.join(key).join(MODEL)
        );
    }

    #[test]
    fn a_named_source_record_keeps_a_digest_before_the_download_goes() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model_int8.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let source = Source::of(&path).unwrap();
        let record = source.record_with_digest(&path).unwrap();
        assert_eq!(record.lines().next(), Some("model_int8.onnx"));
        assert_eq!(
            record.lines().nth(3),
            Some("sha256 ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        std::fs::remove_file(path).unwrap();
        assert_eq!(Source::parse(&record), Some(source));
    }

    #[test]
    fn digest_provenance_tracks_equal_length_timestamp_preserving_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("source.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
        let identity = Source::of(&path).unwrap();
        let before = identity.record_with_digest(&path).unwrap();
        std::fs::write(&path, b"def").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        let replaced = Source::of(&path).unwrap();
        assert_ne!(replaced.key().0, identity.key().0);
        let after = replaced.record_with_digest(&path).unwrap();
        assert_ne!(before.lines().nth(3), after.lines().nth(3));
        assert_eq!(
            after.lines().nth(3).unwrap(),
            format!("sha256 {}", source_digest(&path).unwrap())
        );
    }

    /// The same source keys the same way twice, and a different one does not
    /// share its key; and a record reads back as the source it describes.
    #[test]
    fn a_key_is_the_sources_own() {
        let dir = tempfile::tempdir().expect("temp dir");
        let one = not_a_model(dir.path());
        let other = dir.path().join("other.onnx");
        std::fs::write(&other, b"a different file entirely").expect("write another");

        let key = |path: &Path| Source::of(path).expect("key").key().0;
        assert_eq!(key(&one), key(&one));
        assert_ne!(key(&one), key(&other));

        let source = Source::of(&one).expect("describe");
        let record = source.record_with_digest(&one).expect("record");
        assert_eq!(Source::parse(&record), Some(source));
        assert_eq!(Source::parse("name\nnot a length\n1\n"), None);
    }

    const DAY: Duration = Duration::from_secs(24 * 60 * 60);

    /// A fake copy under `root` whose times say it was written at `written`,
    /// last mapped at `mapped`, and -- if `stamped` -- last loaded then.
    fn fake_copy(
        root: &Path,
        key: &str,
        written: SystemTime,
        mapped: SystemTime,
        stamped: Option<SystemTime>,
    ) -> PathBuf {
        let done = root.join(key);
        std::fs::create_dir_all(&done).expect("make the copy");
        let at = |file: &str, times: std::fs::FileTimes| {
            let file = File::create(done.join(file)).expect("write a file of the copy");
            file.set_times(times).expect("set its times");
        };
        at(MODEL, std::fs::FileTimes::new().set_modified(written));
        at(
            DATA,
            std::fs::FileTimes::new()
                .set_modified(written)
                .set_accessed(mapped),
        );
        if let Some(stamped) = stamped {
            at(USED, std::fs::FileTimes::new().set_modified(stamped));
        }
        open(&root.join(format!("{key}.lock"))).expect("its lock");
        done
    }

    fn key_of(n: u8) -> String {
        format!("{n:02x}").repeat(16)
    }

    /// Copies go by when they were last used, and only when nothing holds
    /// them: the stamp a load leaves, and for a copy written before stamps
    /// existed, when it was written and when it was last mapped. A partial
    /// directory goes unless a writer holds its lock, and nothing that is not
    /// a copy is touched.
    ///
    /// Each copy is the case one part of the rule exists for, so taking any
    /// part out fails here: without the lock check the held copy goes, without
    /// the access time the one an older binary still maps, and without the
    /// stamp the one loaded yesterday.
    #[test]
    fn a_copy_goes_once_nothing_holds_it_or_has_used_it_for_two_weeks() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path();
        let now = SystemTime::now();
        let long_ago = now - 60 * DAY;

        let loaded_yesterday = fake_copy(root, &key_of(1), long_ago, long_ago, Some(now - DAY));
        let loaded_long_ago = fake_copy(root, &key_of(2), long_ago, long_ago, Some(now - 20 * DAY));
        let written_long_ago = fake_copy(root, &key_of(3), now - 20 * DAY, now - 20 * DAY, None);
        let mapped_recently = fake_copy(root, &key_of(4), long_ago, now - 2 * DAY, None);
        let held = fake_copy(root, &key_of(5), long_ago, long_ago, Some(now - 20 * DAY));
        let holder = open(&root.join(format!("{}.lock", key_of(5)))).expect("lock");
        holder
            .lock_shared()
            .expect("hold it, as a process that loaded it does");

        let crashed = root.join(format!("{}.partial", key_of(6)));
        std::fs::create_dir_all(&crashed).expect("a crashed writer's partial directory");
        let writing = root.join(format!("{}.partial", key_of(7)));
        std::fs::create_dir_all(&writing).expect("a writer's partial directory");
        let writer = open(&root.join(format!("{}.lock", key_of(7)))).expect("lock");
        writer.lock().expect("hold it, as a writer does");
        let other = root.join("not-a-copy");
        std::fs::create_dir_all(&other).expect("something else in the directory");

        collect(root, now);

        assert!(
            loaded_yesterday.exists(),
            "a copy loaded yesterday was removed"
        );
        assert!(
            !loaded_long_ago.exists(),
            "a copy last loaded twenty days ago was kept"
        );
        assert!(
            !written_long_ago.exists(),
            "an unstamped copy written twenty days ago was kept"
        );
        assert!(
            mapped_recently.exists(),
            "a copy an older binary mapped two days ago was removed"
        );
        assert!(held.exists(), "a copy another process holds was removed");
        assert!(
            !crashed.exists(),
            "a partial directory nothing is writing was kept"
        );
        assert!(
            writing.exists(),
            "a partial directory being written was removed"
        );
        assert!(other.exists(), "something that is not a copy was removed");
        assert!(
            !root.join(format!("{}.partial", key_of(2))).exists(),
            "a removed copy was left aside"
        );
        assert!(
            root.join(format!("{}.lock", key_of(2))).exists(),
            "a removed copy's lock file was deleted"
        );
    }

    /// Makes every time the collection reads say the copy at `done` was last
    /// used at `at`.
    fn age(done: &Path, at: SystemTime) {
        for (file, times) in [
            (USED, std::fs::FileTimes::new().set_modified(at)),
            (MODEL, std::fs::FileTimes::new().set_modified(at)),
            (
                DATA,
                std::fs::FileTimes::new().set_modified(at).set_accessed(at),
            ),
        ] {
            File::options()
                .create(true)
                .truncate(false)
                .write(true)
                .open(done.join(file))
                .and_then(|file| file.set_times(times))
                .expect("age a file of the copy");
        }
    }

    /// A copy found through its record, with the download gone, is stamped
    /// and held just as one found beside its download is. Once downloads are
    /// removed that lookup is how nearly every load finds its copy, so without
    /// this a copy in use would look unused and could be removed under the
    /// process loading it.
    #[test]
    fn a_copy_found_through_its_record_is_stamped_and_held() {
        let dir = tempfile::tempdir().expect("temp dir");
        let hub = Hub::new(dir.path(), &tiny_model());
        let cache = dir.path().join("models");
        let copy =
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("the first load");
        load(&copy);
        release(&hub, &cache, &crate::InferenceSettings::default());
        assert!(hub.on_disk().is_none(), "the download was kept");
        let done = copy.parent().expect("the copy's directory").to_path_buf();
        let key = done
            .file_name()
            .and_then(|name| name.to_str())
            .expect("key");

        // What a later process finds: a copy nothing holds, last used long ago.
        drop(held().remove(&done));
        let now = SystemTime::now();
        age(&done, now - 60 * DAY);
        assert!(unused(&done, now), "the fixture's copy is not old enough");

        hub.online.set(false);
        assert_eq!(
            load_path(&hub, &cache, &crate::InferenceSettings::default()).expect("a load offline"),
            copy
        );
        assert!(
            !unused(&done, now),
            "a copy found through its record was not stamped"
        );
        let other = open(&cache.join("prepared").join(format!("{key}.lock"))).expect("lock");
        assert!(
            matches!(other.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
            "a copy found through its record was not held"
        );
    }

    /// The copy a record points at for this runtime and CPU is never
    /// collected, however long it goes unused: its download is gone, so
    /// removing it would make the next load of that model a download. A copy
    /// under any other key -- an upgrade's leftover, say -- still goes by the
    /// rest of the rule, and the record itself is left alone.
    #[test]
    fn the_copy_a_record_points_at_stays_and_other_keys_go() {
        let dir = tempfile::tempdir().expect("temp dir");
        let model = not_a_model(dir.path());
        let source = Source::of(&model).expect("describe the source");
        let root = dir.path().join("prepared");
        std::fs::create_dir_all(&root).expect("the copies' directory");
        let record = root.join("someone--model--onnx--model.onnx.source");
        std::fs::write(&record, source.record_with_digest(&model).unwrap()).expect("the record");
        let now = SystemTime::now();
        let long_ago = now - 60 * DAY;

        let (key, _) = source.key();
        let recorded = fake_copy(&root, &key, long_ago, long_ago, Some(long_ago));
        let stale = fake_copy(&root, &key_of(1), long_ago, long_ago, Some(long_ago));
        assert!(unused(&recorded, now) && unused(&stale, now));

        collect(&root, now);

        assert!(
            recorded.join(MODEL).exists(),
            "the copy a record points at for this runtime and CPU was removed"
        );
        assert!(
            !stale.exists(),
            "a copy under a key this build does not load was kept"
        );
        assert!(record.exists(), "the record was removed");
    }

    /// A record's partial file a release left behind goes once it is an hour
    /// old; a fresh one, which a release may be about to rename, stays.
    #[test]
    fn a_left_record_partial_goes_after_an_hour() {
        let dir = tempfile::tempdir().expect("temp dir");
        let root = dir.path().join("prepared");
        std::fs::create_dir_all(&root).expect("the copies' directory");
        let name = |n: u8| {
            format!(
                "someone--model--onnx--model.onnx.0123456789abcdef.source.{}.42-{n}.partial",
                key_of(n)
            )
        };
        let (old, fresh) = (root.join(name(1)), root.join(name(2)));
        let now = SystemTime::now();
        for (path, at) in [(&old, now - 2 * RECORD_PARTIAL_FOR), (&fresh, now)] {
            std::fs::write(path, "half a record").expect("a partial record");
            File::options()
                .write(true)
                .open(path)
                .and_then(|file| file.set_modified(at))
                .expect("its time");
        }

        collect(&root, now);

        assert!(!old.exists(), "an hour-old record partial was kept");
        assert!(fresh.exists(), "a fresh record partial was removed");
    }

    /// A load stamps the copy it hands out and holds it, so a collection in
    /// another process leaves it however long this one goes on using it.
    ///
    /// The collection runs in this process, on handles of its own, which the
    /// lock treats as another process's: a lock is held per open file, not
    /// per process.
    #[test]
    fn a_loaded_copy_is_stamped_and_held() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = not_a_model(dir.path());
        let root = dir.path().join("prepared");
        let (key, _) = Source::of(&source).expect("key the source").key();
        let now = SystemTime::now();
        let done = fake_copy(
            &root,
            &key,
            now - 60 * DAY,
            now - 60 * DAY,
            Some(now - 60 * DAY),
        );
        assert!(
            unused(&done, now),
            "the fixture's copy is not old enough to be collected"
        );

        prepare(&source, &root, 1).expect("find the copy");
        assert!(!unused(&done, now), "loading the copy did not stamp it");

        collect(&root, now + 30 * DAY);
        assert!(
            done.join(MODEL).exists(),
            "a copy this process holds was removed"
        );
    }
}
