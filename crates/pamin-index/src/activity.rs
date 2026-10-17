//! What model loads in this process have written to the model directory:
//! files fetched from the hub, and copies prepared from them (a mapped copy,
//! a rewritten graph, a compiled accelerator package). An evaluation reads
//! it before and after its models load, so a run that paid for a download
//! or a preparation in its timings is told from one that found everything
//! in place -- by what the loads did, not by predicting what they would do.

use std::sync::atomic::{AtomicU64, Ordering};

static FETCHED: AtomicU64 = AtomicU64::new(0);
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// How many files model loads in this process fetched and how many copies
/// they prepared, counted from process start; [`since`](Self::since) gives
/// what a stretch of the process did. Process-wide: loads on other threads
/// count too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadActivity {
    pub fetched: u64,
    pub prepared: u64,
}

impl LoadActivity {
    /// The counts so far.
    pub fn now() -> Self {
        Self {
            fetched: FETCHED.load(Ordering::Relaxed),
            prepared: PREPARED.load(Ordering::Relaxed),
        }
    }

    /// What happened between `earlier` and `self`.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            fetched: self.fetched - earlier.fetched,
            prepared: self.prepared - earlier.prepared,
        }
    }
}

/// A file fetched from the hub.
pub(crate) fn fetched() {
    FETCHED.fetch_add(1, Ordering::Relaxed);
}

/// A copy prepared from a download.
pub(crate) fn prepared() {
    PREPARED.fetch_add(1, Ordering::Relaxed);
}
