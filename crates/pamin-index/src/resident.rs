//! Handing a model's mapped weights back to the page cache between uses.
//!
//! A CPU session reads its weights from a mapped copy (see `crate::prepared`),
//! and every page a pass touches stays in the process's resident set until the
//! mapping goes. A search runs the embedder and then the reranker, so its peak
//! counts both models' weights although only one is in use at a time. The
//! pages are clean copies of the file: dropping them from this process leaves
//! them in the page cache, and the next pass maps them back with a minor fault
//! rather than a read.

use std::path::Path;

use crate::error::Result;

/// Held across every session load from a file, so the mappings that appear
/// while one loads are that session's own. See [`loading`].
#[cfg(target_os = "linux")]
static LOADS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// What `load` returns, and the `[start, end)` address ranges of the clean
/// mappings of files under `directory` that it created, for [`release`].
///
/// Every session load from a file goes through here, one at a time, so the
/// mappings that appear during `load` are the loaded session's and no other's.
/// They are unmapped only when that session is dropped, so whoever holds the
/// session may release its ranges for as long as it holds it. Ranges found by
/// looking at a directory instead could belong to another session over the
/// same copy, which may be dropped and its addresses reused while they are
/// still held.
///
/// A mapping is left out if any of its pages has been written: the mappings
/// are private, so a written page holds the only copy of what was written, and
/// dropping it would bring the file's bytes back in its place. The weights are
/// never written, so in practice nothing is left out, but the check is what
/// makes releasing safe rather than the assumption.
///
/// Advisory: no ranges if the mappings cannot be read.
#[cfg(target_os = "linux")]
pub(crate) fn loading<T>(
    directory: &Path,
    load: impl FnOnce() -> Result<T>,
) -> Result<(T, Vec<(usize, usize)>)> {
    let _loading = LOADS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // smaps names a mapping by its resolved path, so a directory reached
    // through a symbolic link would match nothing.
    let directory = std::fs::canonicalize(directory).ok();
    let before = directory.as_deref().and_then(|directory| {
        Some(mapped(
            &std::fs::read_to_string("/proc/self/maps").ok()?,
            directory,
        ))
    });
    let value = load()?;
    let ranges = match (directory, before) {
        (Some(directory), Some(before)) => std::fs::read_to_string("/proc/self/smaps")
            .map(|smaps| new_since(&clean_mappings(&smaps, &directory), &before))
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    Ok((value, ranges))
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn loading<T>(
    _directory: &Path,
    load: impl FnOnce() -> Result<T>,
) -> Result<(T, Vec<(usize, usize)>)> {
    Ok((load()?, Vec::new()))
}

/// Drops from this process every resident page of `ranges`, which
/// [`loading`] found for a session the caller holds.
///
/// Advisory: a range that cannot be released is left as it is.
#[cfg(target_os = "linux")]
pub(crate) fn release(ranges: &[(usize, usize)]) {
    let mut released = 0usize;
    for &(start, end) in ranges {
        // SAFETY: the range is a whole mapping, with no written page, of a
        // session the caller holds, so it is still that mapping.
        // `MADV_DONTNEED` on a private file mapping with no written page only
        // unmaps pages the next access maps back from the file. Nothing in safe
        // Rust observes the difference.
        let done =
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
        if done == 0 {
            released += end - start;
        }
    }
    tracing::trace!(
        released,
        "handed a model's mapped weights back to the page cache"
    );
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn release(_ranges: &[(usize, usize)]) {}

/// The ranges of the mappings in `maps` (`/proc/self/maps`, or the header
/// lines of `smaps`) whose file is under `directory`.
#[cfg(any(target_os = "linux", test))]
fn mapped(maps: &str, directory: &Path) -> Vec<(usize, usize)> {
    maps.lines()
        .filter_map(|line| header(line, directory).flatten())
        .collect()
}

/// The parts of `after` that no range of `before` covers.
///
/// Not just the ranges of `after` missing from `before`: the kernel merges a
/// new mapping into an adjacent one of the same file at the next offset, so a
/// mapping found after a load can hold one that was there before it, which
/// belongs to another session and may be unmapped while this one lives.
#[cfg(any(target_os = "linux", test))]
fn new_since(after: &[(usize, usize)], before: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut before = before.to_vec();
    before.sort_unstable();
    let mut new = Vec::new();
    for &(start, end) in after {
        let mut from = start;
        for &(old_start, old_end) in &before {
            if old_end <= from || old_start >= end {
                continue;
            }
            if old_start > from {
                new.push((from, old_start));
            }
            from = from.max(old_end);
        }
        if from < end {
            new.push((from, end));
        }
    }
    new
}

/// The `[start, end)` address ranges of the mappings in `smaps` whose file is
/// under `directory` and none of whose pages is anonymous, that is written.
#[cfg(any(target_os = "linux", test))]
fn clean_mappings(smaps: &str, directory: &Path) -> Vec<(usize, usize)> {
    let mut clean = Vec::new();
    // The mapping the lines below the current header describe, if it is one
    // of `directory`'s files.
    let mut current: Option<(usize, usize)> = None;
    for line in smaps.lines() {
        if let Some(range) = header(line, directory) {
            current = range;
        } else if let Some(kb) = line.strip_prefix("Anonymous:")
            && let Some(range) = current.take()
            && kb.trim().strip_suffix("kB").map(str::trim) == Some("0")
        {
            clean.push(range);
        }
    }
    clean
}

/// For a mapping's header line, `Some` of its range if its file is under
/// `directory` and `Some(None)` otherwise; `None` for any other line.
#[cfg(any(target_os = "linux", test))]
fn header(line: &str, directory: &Path) -> Option<Option<(usize, usize)>> {
    let mut fields = line.splitn(6, ' ');
    let (start, end) = fields.next()?.split_once('-')?;
    let start = usize::from_str_radix(start, 16).ok()?;
    let end = usize::from_str_radix(end, 16).ok()?;
    // perms, offset, device, inode, then the path after padding.
    let path = fields.nth(4).map(str::trim_start).unwrap_or("");
    Some((Path::new(path).starts_with(directory) && !path.is_empty()).then_some((start, end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMAPS: &str = "\
7f0000000000-7f0000004000 rw-p 00100000 fd:01 42                         /models/prepared/a/model.onnx.data
Size:                 16 kB
Rss:                  16 kB
Anonymous:             0 kB
7f0000004000-7f0000008000 rw-p 00200000 fd:01 42                         /models/prepared/a/model.onnx.data
Size:                 16 kB
Anonymous:             4 kB
7f0000008000-7f000000c000 r--p 00000000 fd:01 43                         /models/prepared/b/model.onnx.data
Anonymous:             0 kB
7f000000c000-7f0000010000 rw-p 00000000 00:00 0
Anonymous:            16 kB
";

    #[test]
    fn only_the_directorys_unwritten_mappings_are_released() {
        assert_eq!(
            clean_mappings(SMAPS, Path::new("/models/prepared/a")),
            vec![(0x7f00_0000_0000, 0x7f00_0000_4000)]
        );
        assert_eq!(
            clean_mappings(SMAPS, Path::new("/models/prepared/b")),
            vec![(0x7f00_0000_8000, 0x7f00_0000_c000)]
        );
    }

    #[test]
    fn a_directorys_mappings_are_read_from_maps_and_smaps_alike() {
        let a = Path::new("/models/prepared/a");
        assert_eq!(
            mapped(SMAPS, a),
            vec![
                (0x7f00_0000_0000, 0x7f00_0000_4000),
                (0x7f00_0000_4000, 0x7f00_0000_8000)
            ]
        );
    }

    #[test]
    fn a_load_owns_only_what_was_not_mapped_before_it() {
        // Unchanged, new, and new but merged around two older mappings.
        let before = [(0x5000, 0x6000), (0x1000, 0x2000), (0x3000, 0x3800)];
        let after = [(0x1000, 0x2000), (0x8000, 0x9000), (0x0, 0x4000)];
        assert_eq!(
            new_since(&after, &before),
            vec![
                (0x8000, 0x9000),
                (0x0, 0x1000),
                (0x2000, 0x3000),
                (0x3800, 0x4000)
            ]
        );
    }

    #[test]
    fn an_anonymous_mapping_is_never_under_a_directory() {
        assert!(clean_mappings(SMAPS, Path::new("/")).len() == 2);
    }
}
