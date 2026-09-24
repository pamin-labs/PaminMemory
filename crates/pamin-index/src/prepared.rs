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
//! model, beside the downloaded weights, and is larger than the file it came
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

/// The path to load `source` from on the CPU: a mapped copy of it under
/// `cache_dir`, written first if there is none yet.
///
/// Falls back to `source` itself, with a warning, when the copy cannot be
/// written -- a read-only or full disk, a file lock the filesystem does not
/// support, a graph the runtime will not re-serialize. The model then loads
/// the way it did before copies existed, which costs memory and nothing else.
/// `PAMIN_PREPARED=off` asks for that on purpose, for a measurement that needs
/// the unmapped load or a disk that cannot spare the second copy.
pub(crate) fn prepared(source: &Path, cache_dir: &Path) -> PathBuf {
    if std::env::var("PAMIN_PREPARED").is_ok_and(|value| value.eq_ignore_ascii_case("off")) {
        return source.to_path_buf();
    }
    let root = cache_dir.join("prepared");
    match prepare(source, &root) {
        Ok(copy) => {
            collect(&root, SystemTime::now());
            copy
        }
        Err(error) => {
            tracing::warn!(
                source = %source.display(),
                %error,
                "could not write a mapped copy of the model; loading it onto the heap instead"
            );
            source.to_path_buf()
        }
    }
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
fn prepare(source: &Path, root: &Path) -> Result<PathBuf> {
    let (key, described) = key(source)?;
    let done = root.join(&key);
    let copy = done.join(MODEL);
    let lock = root.join(format!("{key}.lock"));

    let shared = match std::fs::create_dir_all(root).and_then(|()| open(&lock)) {
        Ok(shared) => shared,
        Err(_) if copy.exists() => return Ok(copy),
        Err(error) => return Err(error.into()),
    };
    shared.lock_shared()?;
    if copy.exists() {
        stamp(&done);
        hold(&done, shared);
        return Ok(copy);
    }
    // Not holding it shared while waiting to write it, or this process would
    // wait on itself: a copy it loaded before and somebody has since deleted.
    drop(shared);
    drop(held().remove(&done));

    let exclusive = open(&lock)?;
    exclusive.lock()?;
    if !copy.exists() {
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
    }
    // Stamped before the lock is let go of, so a removal that takes it in
    // between finds the copy in use.
    stamp(&done);
    exclusive.unlock()?;
    exclusive.lock_shared()?;
    hold(&done, exclusive);
    Ok(copy)
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

/// The directory name a copy of `source` is kept under, and what it was
/// derived from, for the log.
///
/// The source is identified by the name of the file it resolves to -- for a
/// hub download that is the blob, named by the hash of its content -- with its
/// length and modification time, so a file replaced in place under the same
/// name is a different source. The rest is everything that decides whether a
/// copy written here is valid there: the runtime's build, the architecture and
/// its matrix features, and the optimization level. Hashed because the build
/// string alone is longer than some filesystems allow a name to be.
fn key(source: &Path) -> Result<(String, String)> {
    let resolved = std::fs::canonicalize(source)?;
    let metadata = std::fs::metadata(&resolved)?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |since| since.as_nanos());
    let described = format!(
        "source {} {} bytes modified {modified}; runtime 1.{} {}; {} {}; {LEVEL:?}",
        resolved
            .file_name()
            .map(|name| name.to_string_lossy())
            .unwrap_or_default(),
        metadata.len(),
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
    Ok((key, described))
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
    use super::*;

    /// A file the runtime cannot read as a model, standing in for any failure
    /// to write a copy.
    fn not_a_model(dir: &Path) -> PathBuf {
        let source = dir.join("broken.onnx");
        std::fs::write(&source, b"this is not a protobuf").expect("write the fake model");
        source
    }

    /// A copy that cannot be written falls back to the source and leaves
    /// nothing behind that a later load could mistake for a copy.
    #[test]
    fn a_failed_copy_loads_the_source_and_leaves_nothing() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = not_a_model(dir.path());
        let cache = dir.path().join("models");

        assert_eq!(prepared(&source, &cache), source);

        let (key, _) = key(&source).expect("key the source");
        let root = cache.join("prepared");
        assert!(!root.join(&key).exists(), "a failed write left a copy");
        assert!(
            !root.join(format!("{key}.partial")).exists(),
            "a failed write left its partial directory"
        );
    }

    /// A partial directory found under the lock is a crashed writer's, and is
    /// cleared rather than written into or trusted.
    #[test]
    fn a_crashed_writers_partial_directory_is_cleared() {
        let dir = tempfile::tempdir().expect("temp dir");
        let source = not_a_model(dir.path());
        let root = dir.path().join("prepared");
        let (key, _) = key(&source).expect("key the source");
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
        let (key, _) = key(&source).expect("key the source");
        std::fs::create_dir_all(root.join(&key)).expect("make the copy's directory");
        std::fs::write(root.join(&key).join(MODEL), b"a copy").expect("write the copy");

        assert_eq!(
            prepare(&source, &root).expect("find the copy"),
            root.join(key).join(MODEL)
        );
    }

    /// The same source keys the same way twice, and a different one does not
    /// share its key.
    #[test]
    fn a_key_is_the_sources_own() {
        let dir = tempfile::tempdir().expect("temp dir");
        let one = not_a_model(dir.path());
        let other = dir.path().join("other.onnx");
        std::fs::write(&other, b"a different file entirely").expect("write another");

        assert_eq!(key(&one).expect("key").0, key(&one).expect("key again").0);
        assert_ne!(key(&one).expect("key").0, key(&other).expect("key").0);
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
        let (key, _) = key(&source).expect("key the source");
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
