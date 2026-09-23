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

use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use sha2::{Digest, Sha256};

use crate::error::{IndexError, Result};

/// The optimization level the copy is written at.
///
/// It has to be the level the copy is loaded at, and `fastembed` builds every
/// session at `Level3` -- ONNX Runtime's layout level -- with no way to ask
/// for another. Loading runs the optimizers again over the optimized graph.
/// That the result scores bit-identically and still finds its packed weights
/// on disk is not argued here but checked, by `tests/prepared.rs`; a change of
/// level is exactly what that check would have to be re-run for, so the level
/// is part of the key.
const LEVEL: GraphOptimizationLevel = GraphOptimizationLevel::Level3;

/// What a prepared copy's two files are called.
///
/// `model.onnx` because that is the name `fastembed`'s path-based loaders
/// expect in a directory. The data file is named inside the graph, relative to
/// it, so the pair moves together and can be renamed into place as one
/// directory.
const MODEL: &str = "model.onnx";
const DATA: &str = "model.onnx.data";

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
    match prepare(source, &cache_dir.join("prepared")) {
        Ok(copy) => copy,
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

/// Finds or writes the copy of `source` under `root`.
///
/// A copy is written into `<key>.partial` and renamed to `<key>` only once both
/// files are on disk, so a directory named by its key is always complete: a
/// crash leaves a partial directory, never a half-written copy that loads.
/// Writing happens under a lock on `<key>.lock`, so two processes loading the
/// same model at once write it once -- the second waits and then finds the
/// first's copy -- and a partial directory found while holding the lock is a
/// crashed writer's and safe to remove.
fn prepare(source: &Path, root: &Path) -> Result<PathBuf> {
    let (key, described) = key(source)?;
    let done = root.join(&key);
    let copy = done.join(MODEL);
    if copy.exists() {
        return Ok(copy);
    }

    std::fs::create_dir_all(root)?;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(root.join(format!("{key}.lock")))?;
    lock.lock()?;
    if copy.exists() {
        return Ok(copy);
    }

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
    Ok(copy)
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
}
