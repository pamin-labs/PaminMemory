//! Handing a model's mapped weights back to the page cache once it is idle.
//!
//! A CPU session reads its weights from a mapped copy (see `crate::prepared`),
//! and every page a pass touches stays in the process's resident set until the
//! mapping goes. A search runs the embedder and then the reranker, so the
//! process holds both models' weights although only one is in use at a time.
//! The pages are clean copies of the file: dropping them from this process
//! leaves them in the page cache, and the next pass maps them back with a
//! minor fault rather than a read.
//!
//! They are dropped only after a model has gone `IDLE` without a pass, not
//! after every search: releasing after every search cost latency (see
//! `docs/measured.md`), and latency outranks memory.

use std::path::Path;
#[cfg(any(target_os = "linux", test))]
use std::time::{Duration, Instant};

use crate::error::Result;

/// The directories sessions are loading from now, so the mappings that appear
/// under one while a session loads are that session's own. See [`loading`].
#[cfg(target_os = "linux")]
static LOADS: Loads = Loads {
    directories: std::sync::Mutex::new(Vec::new()),
    finished: std::sync::Condvar::new(),
};

#[cfg(target_os = "linux")]
struct Loads {
    directories: std::sync::Mutex<Vec<std::path::PathBuf>>,
    finished: std::sync::Condvar,
}

/// A load from one directory, from when no other load overlaps it until it
/// is dropped.
#[cfg(target_os = "linux")]
struct Load(std::path::PathBuf);

#[cfg(target_os = "linux")]
impl Load {
    /// Waits until no load is running from `directory`, or from a directory
    /// inside it or around it, whose new mappings `directory`'s would include
    /// or fall in; loads from other directories run alongside.
    fn start(directory: &Path) -> Self {
        let mut loading = LOADS
            .directories
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while loading
            .iter()
            .any(|other| other.starts_with(directory) || directory.starts_with(other))
        {
            loading = LOADS
                .finished
                .wait(loading)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        loading.push(directory.to_path_buf());
        Self(directory.to_path_buf())
    }
}

#[cfg(target_os = "linux")]
impl Drop for Load {
    fn drop(&mut self) {
        let mut loading = LOADS
            .directories
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(at) = loading.iter().position(|other| *other == self.0) {
            loading.swap_remove(at);
        }
        LOADS.finished.notify_all();
    }
}

/// What `load` returns, and the `[start, end)` address ranges of the clean
/// mappings of files under `directory` that it created, for
/// [`Resident::extend`].
///
/// Every session load from a file goes through here, and no two from the same
/// directory, or from one inside the other, run at once, so the mappings of
/// `directory` that appear during `load` are the loaded session's and no
/// other's. Loads from other directories run alongside it: a mapping is
/// counted only under the directory of its file, so theirs are never counted
/// here (q106).
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
    // smaps names a mapping by its resolved path, so a directory reached
    // through a symbolic link would match nothing.
    let resolved = std::fs::canonicalize(directory).ok();
    let _loading = Load::start(resolved.as_deref().unwrap_or(directory));
    let directory = resolved;
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

/// How long a model goes without a pass before its weights are released.
///
/// An agent's queries come in bursts a few seconds apart: a burst keeps the
/// weights resident and pays no refaults, while a memory tool spends most of
/// its life idle, and that is when the weights are handed back. Not tuned:
/// any delay longer than the gap between a burst's queries keeps a burst's
/// latency, and 30 s is well beyond it.
#[cfg(any(target_os = "linux", test))]
const IDLE: Duration = Duration::from_secs(30);

/// How often the releasing thread looks for idle models.
#[cfg(target_os = "linux")]
const TICK: Duration = Duration::from_secs(1);

/// The models whose weights may be released, by the slot of their [`Resident`].
#[cfg(target_os = "linux")]
static IDLE_MODELS: std::sync::Mutex<Vec<Option<Entry>>> = std::sync::Mutex::new(Vec::new());

/// One model's releasable weights, and whether it is in use.
#[cfg(any(target_os = "linux", test))]
struct Entry {
    /// What [`loading`] found for the sessions of the [`Resident`]'s owner.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    ranges: Vec<(usize, usize)>,
    /// How many [`Resident::busy`] calls have no [`Resident::idle`] yet.
    busy: usize,
    last_used: Instant,
    /// Whether `ranges` were released since the model was last used.
    released: bool,
}

#[cfg(target_os = "linux")]
impl Entry {
    fn new(now: Instant) -> Self {
        Self {
            ranges: Vec::new(),
            busy: 0,
            last_used: now,
            released: false,
        }
    }
}

/// The slots of `entries` whose weights should be released at `now`: those
/// not in use, not yet released, and unused for at least `idle`.
#[cfg(any(target_os = "linux", test))]
fn due(entries: &[Option<Entry>], now: Instant, idle: Duration) -> Vec<usize> {
    entries
        .iter()
        .enumerate()
        .filter_map(|(slot, entry)| {
            let entry = entry.as_ref()?;
            (entry.busy == 0
                && !entry.released
                && now.saturating_duration_since(entry.last_used) >= idle)
                .then_some(slot)
        })
        .collect()
}

/// The handle through which a model's owner registers its sessions' weights
/// for release once idle, and marks their use.
///
/// Its owner must drop it before the sessions whose ranges it holds: dropping
/// it unregisters them under the lock every release is made under, so no
/// range is released once its session may have been dropped and its
/// addresses reused.
#[derive(Default)]
pub(crate) struct Resident {
    #[cfg(target_os = "linux")]
    slot: Option<usize>,
}

#[cfg(target_os = "linux")]
fn idle_models() -> std::sync::MutexGuard<'static, Vec<Option<Entry>>> {
    IDLE_MODELS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(target_os = "linux")]
impl Resident {
    /// This handle's entry, registered if it has none.
    fn entry<'a>(&mut self, entries: &'a mut Vec<Option<Entry>>) -> &'a mut Entry {
        let slot = *self.slot.get_or_insert_with(|| {
            let entry = Some(Entry::new(Instant::now()));
            match entries.iter().position(Option::is_none) {
                Some(free) => {
                    entries[free] = entry;
                    free
                }
                None => {
                    entries.push(entry);
                    entries.len() - 1
                }
            }
        });
        entries[slot]
            .as_mut()
            .expect("a registered slot holds its entry")
    }

    /// Adds `ranges`, which [`loading`] found for a session the owner holds
    /// until it drops this handle.
    pub(crate) fn extend(&mut self, ranges: Vec<(usize, usize)>) {
        if ranges.is_empty() {
            return;
        }
        start_releasing();
        let mut entries = idle_models();
        let entry = self.entry(&mut entries);
        entry.ranges.extend(ranges);
        entry.last_used = Instant::now();
        entry.released = false;
    }

    /// Marks the model in use, so its weights are not released until a
    /// matching [`idle`](Self::idle).
    pub(crate) fn busy(&mut self) {
        self.entry(&mut idle_models()).busy += 1;
    }

    /// Marks the end of a use begun by [`busy`](Self::busy): the weights are
    /// released once the model has gone [`IDLE`] without another.
    pub(crate) fn idle(&mut self) {
        let mut entries = idle_models();
        let entry = self.entry(&mut entries);
        entry.busy = entry.busy.saturating_sub(1);
        entry.last_used = Instant::now();
        entry.released = false;
    }
}

#[cfg(target_os = "linux")]
impl Drop for Resident {
    fn drop(&mut self) {
        if let Some(slot) = self.slot {
            idle_models()[slot] = None;
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl Resident {
    pub(crate) fn extend(&mut self, _ranges: Vec<(usize, usize)>) {}
    pub(crate) fn busy(&mut self) {}
    pub(crate) fn idle(&mut self) {}
}

/// Starts, once, the thread that releases idle models' weights.
///
/// Advisory: if it cannot start, nothing is released.
#[cfg(target_os = "linux")]
fn start_releasing() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let started = std::thread::Builder::new()
            .name(String::from("pamin-idle-release"))
            .spawn(|| {
                loop {
                    std::thread::sleep(TICK);
                    let mut entries = idle_models();
                    for slot in due(&entries, Instant::now(), IDLE) {
                        let entry = entries[slot].as_mut().expect("due slots are registered");
                        // Under the lock a `Resident` takes to unregister, so
                        // its owner still holds the sessions.
                        release(&entry.ranges);
                        entry.released = true;
                    }
                }
            });
        if let Err(error) = started {
            tracing::warn!(%error, "could not start releasing idle models' weights");
        }
    });
}

/// Drops from this process every resident page of `ranges`, which
/// [`loading`] found for sessions whose owner still holds them.
///
/// Advisory: a range that cannot be released is left as it is.
#[cfg(target_os = "linux")]
fn release(ranges: &[(usize, usize)]) {
    let mut released = 0usize;
    for &(start, end) in ranges {
        // SAFETY: the range is a whole mapping, with no written page, of a
        // session [`loading`] loaded. It is released only under the lock of
        // `IDLE_MODELS` while its `Resident` is registered, and the owner
        // drops the `Resident` -- unregistering under that lock -- before the
        // session, so the session still holds it and it is still that
        // mapping. `MADV_DONTNEED` on a private file mapping with no written
        // page only unmaps pages the next access maps back from the file,
        // even during a pass on another thread. Nothing in safe Rust observes
        // the difference.
        let done =
            unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_DONTNEED) };
        if done == 0 {
            released += end - start;
        }
    }
    tracing::trace!(
        released,
        "handed an idle model's mapped weights back to the page cache"
    );
}

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

    fn entry(busy: usize, last_used: Instant, released: bool) -> Option<Entry> {
        Some(Entry {
            ranges: vec![(0x1000, 0x2000)],
            busy,
            last_used,
            released,
        })
    }

    #[test]
    fn only_an_unreleased_model_idle_long_enough_is_due() {
        let start = Instant::now();
        let now = start + IDLE;
        // In use, used recently, idle, already released, and unregistered.
        let entries = [
            entry(1, start, false),
            entry(0, now - IDLE / 2, false),
            entry(0, start, false),
            entry(0, start, true),
            None,
        ];
        assert_eq!(due(&entries, now, IDLE), vec![2]);
    }

    #[test]
    fn an_idle_model_is_released_once_and_again_after_its_next_use() {
        let start = Instant::now();
        let mut entries = vec![entry(0, start, false)];
        assert_eq!(due(&entries, start + IDLE, IDLE), vec![0]);
        entries[0].as_mut().unwrap().released = true;
        assert!(due(&entries, start + 2 * IDLE, IDLE).is_empty());
        // What `Resident::idle` does after a use.
        let used = start + 2 * IDLE;
        let model = entries[0].as_mut().unwrap();
        (model.last_used, model.released) = (used, false);
        assert!(due(&entries, used + IDLE / 2, IDLE).is_empty());
        assert_eq!(due(&entries, used + IDLE, IDLE), vec![0]);
    }

    #[test]
    fn an_anonymous_mapping_is_never_under_a_directory() {
        assert!(clean_mappings(SMAPS, Path::new("/")).len() == 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn loads_wait_only_for_their_own_directory() {
        use std::sync::mpsc::channel;
        let root = std::env::temp_dir().join(format!("pamin-loads-{}", std::process::id()));
        let (outer, inner, other) = (root.join("a"), root.join("a/b"), root.join("c"));
        let (outer, inner) = (&outer, &inner);
        let (entered, inside) = channel();
        let (finish, finished) = channel::<()>();
        std::thread::scope(|scope| {
            let holder = scope.spawn(move || {
                loading(outer, || {
                    entered.send(()).unwrap();
                    finished.recv().unwrap();
                    Ok(())
                })
            });
            inside.recv().unwrap();
            // Another directory loads while `outer`'s load is still running.
            assert!(loading(&other, || Ok(())).is_ok());
            // One inside it waits for that load to finish.
            let (started, start) = channel();
            let nested = scope.spawn(move || {
                loading(inner, || {
                    started.send(()).unwrap();
                    Ok(())
                })
            });
            assert!(start.recv_timeout(Duration::from_millis(200)).is_err());
            finish.send(()).unwrap();
            start.recv_timeout(Duration::from_secs(10)).unwrap();
            holder.join().unwrap().unwrap();
            nested.join().unwrap().unwrap();
        });
    }
}
