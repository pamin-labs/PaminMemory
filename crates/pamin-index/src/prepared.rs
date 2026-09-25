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
//! (`<repository>--<file>.source`, beside the copies) is what finds the copy
//! without the file it was keyed by, and what lets a re-download key exactly
//! as the removed one did.
//!
//! A copy's key changes with the runtime and the CPU, so every upgrade of ONNX
//! Runtime writes new copies and leaves the old ones -- 0.8 to 1.2 GB a model
//! -- where nothing would otherwise remove them. Every load therefore
//! collects, by one rule: **a copy is removed once no process holds it and
//! none has loaded it for [`UNUSED_FOR`]**, whichever binary wrote it. Not by
//! key: a key is a hash nobody can read an owner from, and two binaries built
//! on different runtimes can share one model directory, each loading its own
//! copy and each entitled to find it there.
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

use std::collections::BTreeMap;
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

/// The stamp a load leaves in a copy: an empty file whose modification time is
/// when the copy was last handed out.
const USED: &str = "used";

/// How long a copy nothing holds may go without a load before a load of any
/// model removes it.
///
/// A choice, not a measurement. Long enough that a binary run once a week
/// keeps its copy between runs -- one running now is covered by its hold
/// however long ago it loaded -- and short enough that an upgrade's leftovers
/// are gone within a month. Removing a copy too early costs one rewrite the
/// next time something loads it: a few seconds, with the source on the heap
/// while it is written.
const UNUSED_FOR: Duration = Duration::from_secs(14 * 24 * 60 * 60);

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
/// `PAMIN_PREPARED=off` asks for that on purpose, for a measurement that needs
/// the unmapped load, and fetches the download if it was removed.
///
/// The copy's graph with its attention fused, where that scored identically,
/// and `PAMIN_FUSED_ATTENTION=off` asks for the graph as ONNX Runtime wrote
/// it -- the other arm of a measurement of what the fusion is worth.
pub(crate) fn load_path(download: &impl Download, cache_dir: &Path) -> Result<PathBuf> {
    let root = cache_dir.join("prepared");
    if !wanted() {
        let fetched = download.fetch()?;
        if let Some(recorded) = recorded(download, &root) {
            recorded.restore(&fetched);
        }
        return Ok(fetched);
    }
    let source = match download.on_disk() {
        Some(source) => source,
        None => {
            let recorded = recorded(download, &root);
            if let Some(copy) = recorded.as_ref().and_then(|source| found(source, &root)) {
                return Ok(copy);
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
            if let Some(recorded) = &recorded {
                recorded.restore(&fetched);
            }
            fetched
        }
    };
    match prepare(&source, &root) {
        Ok(copy) => {
            collect(&root, SystemTime::now());
            Ok(graph(copy))
        }
        Err(error) => {
            // Another process released the download between this one finding
            // it and keying it, which it does only once the copy is complete.
            if !source.exists()
                && let Some(copy) =
                    recorded(download, &root).and_then(|source| found(&source, &root))
            {
                return Ok(copy);
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
/// already gone, there is no complete copy for it, `PAMIN_PREPARED=off`, or
/// the file is not the model directory's own ([`Download::remove`]).
pub(crate) fn release(download: &impl Download, cache_dir: &Path) {
    if !wanted() {
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
        let partial = record.with_extension("source.partial");
        std::fs::write(&partial, identity.record())?;
        std::fs::rename(&partial, &record)?;
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
pub(crate) fn is_ready(download: &impl Download, cache_dir: &Path) -> bool {
    download.on_disk().is_some()
        || (wanted()
            && recorded(download, &cache_dir.join("prepared"))
                .and_then(|source| found(&source, &cache_dir.join("prepared")))
                .is_some())
}

/// Whether copies are wanted at all: unless `PAMIN_PREPARED=off`.
fn wanted() -> bool {
    !std::env::var("PAMIN_PREPARED").is_ok_and(|value| value.eq_ignore_ascii_case("off"))
}

/// The graph a caller loads from a settled copy: the fused one unless
/// `PAMIN_FUSED_ATTENTION=off` asks for the one ONNX Runtime wrote.
fn graph(copy: PathBuf) -> PathBuf {
    if copy.ends_with(FUSED) && !fusion_wanted() {
        copy.with_file_name(MODEL)
    } else {
        copy
    }
}

/// The settled copy of the download `source` described, if this runtime and
/// CPU have one.
fn found(source: &Source, root: &Path) -> Option<PathBuf> {
    settled(&root.join(source.key().0)).map(graph)
}

/// What a removed download was, as [`release`] recorded it.
fn recorded(download: &impl Download, root: &Path) -> Option<Source> {
    Source::parse(&std::fs::read_to_string(root.join(record_name(download))).ok()?)
}

/// The record's file name: the label, readable, with nothing a path would
/// split on.
fn record_name(download: &impl Download) -> String {
    format!("{}.source", download.label().replace(['/', '\\'], "--"))
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
/// lock is a crashed writer's and safe to remove. Finding a copy happens under
/// the same lock held shared, which is what a removal waits on.
///
/// A model directory this process cannot write -- a lock it cannot open --
/// still loads a copy that is already there. Nothing can remove it from such
/// a directory either, so it needs no hold.
///
/// The attention fusion is settled under the exclusive lock, once per copy --
/// for a copy written before the fusion existed, on the first load that finds
/// it -- and the path returned is the graph to load: [`FUSED`] if it was
/// kept, [`MODEL`] if not.
fn prepare(source: &Path, root: &Path) -> Result<PathBuf> {
    let (key, described) = Source::of(source)?.key();
    let done = root.join(&key);
    let copy = done.join(MODEL);
    let lock = root.join(format!("{key}.lock"));

    let shared = match std::fs::create_dir_all(root).and_then(|()| open(&lock)) {
        Ok(shared) => shared,
        Err(error) => return settled(&done).ok_or_else(|| error.into()),
    };
    shared.lock_shared()?;
    if let Some(settled) = settled(&done) {
        stamp(&done);
        hold(&done, shared);
        return Ok(settled);
    }
    // Not holding it shared while waiting to write it, or this process would
    // wait on itself: a copy it loaded before and somebody has since deleted.
    drop(shared);
    drop(held().remove(&done));

    let exclusive = open(&lock)?;
    exclusive.lock()?;
    let graph = match settled(&done) {
        Some(settled) => settled,
        None if copy.exists() => fuse(&done),
        None => {
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
            fuse(&done)
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

/// Removes, from `root`, every copy that has gone [`UNUSED_FOR`] without a
/// load and that no process holds, and every partial directory no writer
/// holds. See the module's documentation for the rule and why it is safe.
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
    for entry in entries.flatten() {
        let path = entry.path();
        if let Err(error) = collect_one(root, &path, now) {
            tracing::warn!(copy = %path.display(), %error, "could not remove an unused mapped copy");
        }
    }
}

fn collect_one(root: &Path, path: &Path, now: SystemTime) -> std::io::Result<()> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(());
    };
    let (key, partial) = match name.strip_suffix(".partial") {
        Some(key) => (key, true),
        None => (name, false),
    };
    let is_key = key.len() == 32
        && key
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    if !is_key || !path.is_dir() || (!partial && !unused(path, now)) {
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
    if !unused(path, now) {
        return Ok(());
    }
    let aside = root.join(format!("{key}.partial"));
    if aside.exists() {
        std::fs::remove_dir_all(&aside)?;
    }
    std::fs::rename(path, &aside)?;
    std::fs::remove_dir_all(&aside)?;
    tracing::info!(copy = %path.display(), "removed a mapped copy nothing had loaded for two weeks");
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

/// Whether a caller wants the fused graph where there is one: unless
/// `PAMIN_FUSED_ATTENTION=off`, which exists so the two can be measured
/// against each other through the product's own load.
fn fusion_wanted() -> bool {
    !std::env::var("PAMIN_FUSED_ATTENTION").is_ok_and(|value| value.eq_ignore_ascii_case("off"))
}

/// Fuses the attention of the complete copy in `copy`, keeps the result only
/// if it scores a probe exactly as the unfused graph does, and returns the
/// graph to load. Called under the copy's lock.
///
/// Never fails the copy. Whatever goes wrong -- nothing to fuse, a probe
/// that differs, a graph the runtime refuses -- is written to [`UNFUSED`]
/// and logged, and the copy loads the graph ONNX Runtime wrote, which is what
/// it loaded before this existed.
fn fuse(copy: &Path) -> PathBuf {
    let model = copy.join(MODEL);
    let started = std::time::Instant::now();
    match try_fuse(copy) {
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
fn try_fuse(copy: &Path) -> std::result::Result<(usize, (usize, usize)), String> {
    let model = copy.join(MODEL);
    let original = std::fs::read(&model).map_err(|error| format!("reading the graph: {error}"))?;
    let fused = crate::attention::fuse(&original)?;
    let candidate = copy.join(format!("{FUSED}.partial"));
    let written = std::fs::write(&candidate, &fused.model)
        .and_then(|()| std::fs::File::open(&candidate)?.sync_all())
        .map_err(|error| format!("writing the fused graph: {error}"))
        .and_then(|()| same_on_a_probe(&model, &candidate))
        .and_then(|()| {
            std::fs::rename(&candidate, copy.join(FUSED))
                .map_err(|error| format!("renaming the fused graph into place: {error}"))
        });
    if written.is_err() {
        let _ = std::fs::remove_file(&candidate);
    }
    written.map(|()| (fused.layers, fused.nodes))
}

/// Whether two graphs return the same bits on a probe, as the product runs
/// them: a CPU session at [`LEVEL`], and every output compared.
///
/// Two rows, the second padded, because the padding is what the fused
/// kernel reads differently -- a key padding mask where the graph it
/// replaces added a bias -- and a probe with none would not exercise it.
/// The token ids are arbitrary: any ids below a thousand are in every
/// vocabulary these models have, and the question is whether the two graphs
/// agree, not what either one thinks of the text.
fn same_on_a_probe(expected: &Path, candidate: &Path) -> std::result::Result<(), String> {
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
        let mut session =
            crate::inference::session(vec![crate::inference::cpu()], || Ok(path.to_path_buf()))
                .map_err(|error| failed(&error))?;
        let mut feed = Vec::new();
        for input in session.inputs() {
            let values = match input.name() {
                "input_ids" => ids.clone(),
                "attention_mask" => mask.clone(),
                "token_type_ids" => vec![0; ids.len()],
                other => return Err(failed(&format!("no probe for an input named {other}"))),
            };
            let tensor = ort::value::Tensor::from_array(([2, LENGTH], values))
                .map_err(|error| failed(&error))?;
            feed.push((input.name().to_string(), tensor));
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
            "the fused graph scored the probe differently ({differing})"
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
#[derive(Debug, PartialEq, Eq)]
struct Source {
    name: String,
    length: u64,
    modified: u128,
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
        Ok(Self {
            name: resolved
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            length: metadata.len(),
            modified,
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
            "source {} {} bytes modified {}; runtime 1.{} {}; {} {}; {LEVEL:?}",
            self.name,
            self.length,
            self.modified,
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

    fn parse(record: &str) -> Option<Self> {
        let mut lines = record.lines();
        let source = Self {
            name: lines.next()?.to_string(),
            length: lines.next()?.parse().ok()?,
            modified: lines.next()?.parse().ok()?,
        };
        (!source.name.is_empty()).then_some(source)
    }

    /// Gives a fresh download of this source this source's modification time,
    /// so it keys exactly as the removed file did.
    ///
    /// A hub blob is named by its content, so the same name and length is the
    /// same file, and the time is the only thing the download changed. Left
    /// changed, every copy of the model would stop fitting -- including those
    /// of another runtime or CPU sharing the directory, which would then fetch
    /// and write again in turn, each undoing the other. A different file --
    /// the repository moved on -- is left as it is and keys as new.
    fn restore(&self, fetched: &Path) {
        let restored = (|| -> Result<bool> {
            let now = Self::of(fetched)?;
            if now.name != self.name || now.length != self.length || now.modified == self.modified {
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
fn features() -> Vec<&'static str> {
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
fn features() -> Vec<&'static str> {
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
fn features() -> Vec<&'static str> {
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
        blob: PathBuf,
        online: Cell<bool>,
        fetches: Cell<usize>,
    }

    impl Hub {
        fn new(dir: &Path, model: &[u8]) -> Self {
            let origin = dir.join("origin.onnx");
            std::fs::write(&origin, model).expect("write the hub's copy");
            Self {
                origin,
                blob: dir.join("models").join("blobs").join("0123abcd"),
                online: Cell::new(true),
                fetches: Cell::new(0),
            }
        }
    }

    impl Download for Hub {
        fn label(&self) -> String {
            "someone/model/onnx/model.onnx".into()
        }

        fn on_disk(&self) -> Option<PathBuf> {
            self.blob.exists().then(|| self.blob.clone())
        }

        fn fetch(&self) -> Result<PathBuf> {
            if let Some(blob) = self.on_disk() {
                return Ok(blob);
            }
            if !self.online.get() {
                return Err(IndexError::Engine("the hub cannot be reached".into()));
            }
            std::fs::create_dir_all(self.blob.parent().expect("a parent"))?;
            std::fs::copy(&self.origin, &self.blob)?;
            self.fetches.set(self.fetches.get() + 1);
            Ok(self.blob.clone())
        }

        fn remove(&self) -> Result<bool> {
            std::fs::remove_file(&self.blob)?;
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
        crate::inference::session(vec![crate::inference::cpu()], || Ok(path.to_path_buf()))
            .unwrap_or_else(|error| panic!("load {}: {error}", path.display()));
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
        let copy = load_path(&hub, &cache).expect("the first load");
        assert_eq!(hub.fetches.get(), 1);
        assert!(copy.starts_with(&root), "loaded {copy:?}, not a copy");
        assert!(
            hub.on_disk().is_some(),
            "the download went before the copy had loaded"
        );
        load(&copy);
        release(&hub, &cache);
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
        assert!(is_ready(&hub, &cache));
        assert_eq!(load_path(&hub, &cache).expect("a load offline"), copy);
        assert_eq!(hub.fetches.get(), 1, "a load with a fitting copy fetched");

        // What a runtime upgrade or another CPU looks like from here: the
        // only copy on disk is keyed for something else.
        std::fs::rename(root.join(&key), root.join("0".repeat(32))).expect("move the copy");
        assert!(!is_ready(&hub, &cache));
        let refused = load_path(&hub, &cache)
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
        let rewritten = load_path(&hub, &cache).expect("a load once online");
        assert_eq!(hub.fetches.get(), 2);
        assert_eq!(
            rewritten, copy,
            "the fresh download keyed differently from the removed one"
        );
        load(&rewritten);
        release(&hub, &cache);
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

        let loaded = load_path(&hub, &cache).expect("fall back");
        assert_eq!(Some(loaded), hub.on_disk());
        release(&hub, &cache);
        assert!(
            hub.on_disk().is_some(),
            "the only loadable file was removed"
        );

        let (key, _) = Source::of(&hub.blob).expect("key the source").key();
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

        assert!(prepare(&source, &root).is_err());
        assert!(!partial.exists(), "the crashed writer's files survived");
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
            prepare(&source, &root).expect("find the copy"),
            root.join(key).join(MODEL)
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
        assert_eq!(Source::parse(&source.record()), Some(source));
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

        prepare(&source, &root).expect("find the copy");
        assert!(!unused(&done, now), "loading the copy did not stamp it");

        collect(&root, now + 30 * DAY);
        assert!(
            done.join(MODEL).exists(),
            "a copy this process holds was removed"
        );
    }
}
