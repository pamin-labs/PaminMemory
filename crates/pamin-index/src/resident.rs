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

/// Drops from this process every resident page of the clean mappings of files
/// under `directory`.
///
/// A mapping is skipped if any of its pages has been written: the mappings are
/// private, so a written page holds the only copy of what was written, and
/// dropping it would bring the file's bytes back in its place. The weights are
/// never written, so in practice nothing is skipped, but the check is what
/// makes the call safe rather than the assumption.
///
/// Advisory: a mapping that cannot be read or released is left as it is.
#[cfg(target_os = "linux")]
pub(crate) fn release(directory: &Path) {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return;
    };
    // smaps names a mapping by its resolved path, so a directory reached
    // through a symbolic link would match nothing.
    let Ok(directory) = std::fs::canonicalize(directory) else {
        return;
    };
    let directory = directory.as_path();
    let mut released = 0usize;
    for (start, end) in clean_mappings(&smaps, directory) {
        // SAFETY: the range is a whole mapping this process holds, read from
        // its own smaps, and `MADV_DONTNEED` on a private file mapping with no
        // written page only unmaps pages the next access maps back from the
        // file. Nothing in safe Rust observes the difference.
        let done =
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
        if done == 0 {
            released += end - start;
        }
    }
    tracing::trace!(
        directory = %directory.display(),
        released,
        "handed a model's mapped weights back to the page cache"
    );
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn release(_directory: &Path) {}

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
    fn an_anonymous_mapping_is_never_under_a_directory() {
        assert!(clean_mappings(SMAPS, Path::new("/")).len() == 2);
    }
}
