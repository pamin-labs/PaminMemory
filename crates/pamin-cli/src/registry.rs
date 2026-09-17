//! A keyed set of things that are expensive to open, opened at most once each.
//!
//! Opening one of these is not a detail: it reads an index off disk and, the
//! first time a profile is wanted, downloads hundreds of megabytes of model
//! weights. Two callers arriving at the same cold key must not both do that,
//! and a caller arriving at a *different* key must not wait while they do.
//!
//! Those two requirements pull against each other, and the shape here is what
//! satisfies both: one lock over the map, held only long enough to find or
//! create a slot, and one lock per slot, held across the open. Before this the
//! map's lock was held across the open, so a cold project stalled every project
//! the process was serving.
//!
//! Generic over what it holds because that is what makes the concurrency
//! testable. Exercising it against the real thing means a database and half a
//! gigabyte of weights; against a loader the test supplies, it is microseconds.

use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::Arc;

use anyhow::Result;

/// One key's value, and the exclusion that keeps it opened once.
///
/// `None` means nobody has opened it yet, or the last attempt failed. The
/// `Arc` is what eviction counts: cloned out before the map's lock is
/// released, so a slot somebody is inside always has more than one holder.
type Slot<T> = Arc<tokio::sync::Mutex<Option<Arc<T>>>>;

pub struct Registry<K, T> {
    /// A `std::sync::Mutex` rather than tokio's, deliberately. Its guard is not
    /// `Send`, and the futures here are spawned, so holding it across an
    /// `.await` does not compile -- which is the bug this type exists to
    /// prevent, turned from a convention into a compiler error.
    open: std::sync::Mutex<Open<K, T>>,
    capacity: usize,
}

struct Open<K, T> {
    slots: HashMap<K, Entry<T>>,
    /// Counts uses rather than reading a clock: what matters is the order they
    /// were last wanted in, and a counter cannot go backwards.
    uses: u64,
}

struct Entry<T> {
    slot: Slot<T>,
    used: u64,
}

impl<K: Eq + Hash + Clone, T> Registry<K, T> {
    /// Holds at most `capacity` open, evicting the least recently used.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            open: std::sync::Mutex::new(Open {
                slots: HashMap::new(),
                uses: 0,
            }),
            capacity,
        }
    }

    /// The value for this key, opening it if nobody has.
    ///
    /// Callers for one key queue behind each other and exactly one of them
    /// runs `open`; callers for other keys are not delayed at all.
    pub async fn get_or_open<F, Fut>(&self, key: K, open: F) -> Result<Arc<T>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        self.fill(key, open, false).await
    }

    /// The value for this key, discarding whatever was there and opening again.
    ///
    /// The registry's handle is dropped *before* `open` runs, because a rebuild
    /// throws away the directory the old value points at. A concurrent
    /// `get_or_open` for this key waits on the same slot and gets the new one,
    /// which is the guarantee holding one lock over everything used to give.
    pub async fn reopen<F, Fut>(&self, key: K, open: F) -> Result<Arc<T>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        self.fill(key, open, true).await
    }

    async fn fill<F, Fut>(&self, key: K, open: F, discard: bool) -> Result<Arc<T>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        // Everything touching the map happens here, with no `.await` in it, so
        // the map is never locked while something is being opened.
        let slot = {
            let mut registry = self.open.lock().expect("the registry lock is poisoned");
            registry.uses += 1;
            let now = registry.uses;

            if let Some(entry) = registry.slots.get_mut(&key) {
                entry.used = now;
                Arc::clone(&entry.slot)
            } else {
                registry.make_room(self.capacity);
                let slot = Slot::default();
                registry.slots.insert(
                    key,
                    Entry {
                        // Cloned before the lock goes, so there is no moment
                        // when a slot being opened looks idle to `make_room`.
                        slot: Arc::clone(&slot),
                        used: now,
                    },
                );
                slot
            }
        };

        let mut held = slot.lock().await;
        if discard {
            *held = None;
        } else if let Some(value) = held.as_ref() {
            return Ok(Arc::clone(value));
        }

        // A failure leaves the slot empty and writes nothing, so the next
        // caller tries again rather than inheriting a poisoned entry. The
        // empty slot is evictable and costs one place until then.
        let value = Arc::new(open().await?);
        *held = Some(Arc::clone(&value));
        Ok(value)
        // The guard drops here, before the caller uses the value: two requests
        // for one key share it rather than queueing on the registry.
    }

    /// The keys currently held, for a caller that works through all of them.
    ///
    /// Keys rather than values, so that walking them does not pin every one
    /// against eviction for the length of the walk.
    pub fn keys(&self) -> Vec<K> {
        self.open
            .lock()
            .expect("the registry lock is poisoned")
            .slots
            .keys()
            .cloned()
            .collect()
    }

    /// The value for this key if it is open and idle, without opening one.
    ///
    /// `None` while another caller holds the slot -- opening it, or rebuilding
    /// it. A caller walking the keys skips those rather than waiting.
    pub fn opened(&self, key: &K) -> Option<Arc<T>> {
        let slot = {
            let registry = self.open.lock().expect("the registry lock is poisoned");
            // Deliberately does not count as a use. A sweep touches everything
            // and would flatten the order the eviction depends on.
            Arc::clone(&registry.slots.get(key)?.slot)
        };
        let held = slot.try_lock().ok()?;
        held.as_ref().map(Arc::clone)
    }
}

impl<K: Eq + Hash + Clone, T> Open<K, T> {
    /// Closes least-recently-used entries until there is room for one more.
    ///
    /// Only ones nothing is using: a value a request still holds stays alive
    /// whether or not this drops the registry's handle, so evicting it would
    /// buy nothing and cost the next caller for that key a second open. When
    /// everything is busy the bound gives way rather than the request -- it
    /// exists to stop idle values accumulating, not to cap concurrency.
    fn make_room(&mut self, capacity: usize) {
        while self.slots.len() >= capacity {
            let idle = self
                .slots
                .iter()
                .filter(|(_, entry)| Arc::strong_count(&entry.slot) == 1)
                .filter(|(_, entry)| match entry.slot.try_lock() {
                    // Empty means a load failed or has not run; there is
                    // nothing to lose by dropping the place it holds.
                    Ok(held) => held
                        .as_ref()
                        .is_none_or(|value| Arc::strong_count(value) == 1),
                    // Somebody is inside it. Never wait for them here: this
                    // runs under the map's lock, and the one lock order this
                    // design must not invert is map-then-slot.
                    Err(_) => false,
                })
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());

            let Some(idle) = idle else { return };
            self.slots.remove(&idle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Registry;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// A cold key does not hold up a different one.
    ///
    /// This is the whole reason the type exists. The exclusion used to be one
    /// lock over the registry held across the open, and an open downloads
    /// hundreds of megabytes the first time a profile is wanted -- so one cold
    /// project stopped every project the process was serving.
    ///
    /// Reverting it takes two changes to `fill`: put a `tokio::sync::Mutex`
    /// back and keep its guard alive across the open. Checked -- the run then
    /// does not fail, it hangs, because the task holding the map is waiting on
    /// a release this task never reaches. That is the failure the timeout here
    /// is meant to turn into a message; it only manages it when the blocked
    /// caller is a task rather than the whole test.
    ///
    /// With a `std::sync::Mutex` the same revert does not compile at all: the
    /// guard is not `Send` and these futures are spawned. That is the stronger
    /// half of the fix and the reason for the choice of mutex.
    #[tokio::test]
    async fn a_cold_key_does_not_hold_up_another_one() {
        let registry: Arc<Registry<&str, u32>> = Arc::new(Registry::with_capacity(16));
        let (started, opening) = tokio::sync::oneshot::channel();
        let (release, released) = tokio::sync::oneshot::channel::<()>();

        let slow = {
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                registry
                    .get_or_open("cold", || async move {
                        started.send(()).expect("nobody waiting for the open");
                        released.await.expect("never released");
                        Ok(1)
                    })
                    .await
            })
        };

        // Only meaningful once the slow open is actually inside the registry.
        opening.await.expect("the open never started");

        let other = tokio::time::timeout(
            Duration::from_secs(5),
            registry.get_or_open("warm", || async { Ok(2) }),
        )
        .await
        .expect("a different key waited for the cold one")
        .expect("opening the other key");
        assert_eq!(*other, 2);

        release.send(()).expect("the slow open went away");
        assert_eq!(*slow.await.expect("joining").expect("the cold open"), 1);
    }

    /// Twenty callers at one cold key open it once.
    ///
    /// The property the single lock was protecting, and the reason the slot is
    /// a lock rather than a bare `Option`. Without it twenty first-searches
    /// against one project start twenty downloads of the same weights.
    #[tokio::test]
    async fn twenty_callers_at_one_cold_key_open_it_once() {
        let registry: Arc<Registry<&str, u32>> = Arc::new(Registry::with_capacity(16));
        let opens = Arc::new(AtomicUsize::new(0));

        let mut waiting = Vec::new();
        for _ in 0..20 {
            let registry = Arc::clone(&registry);
            let opens = Arc::clone(&opens);
            waiting.push(tokio::spawn(async move {
                registry
                    .get_or_open("cold", || async move {
                        opens.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok(7)
                    })
                    .await
            }));
        }

        let mut got = Vec::new();
        for one in waiting {
            got.push(one.await.expect("joining").expect("opening"));
        }

        assert_eq!(
            opens.load(Ordering::SeqCst),
            1,
            "it was opened more than once"
        );
        assert!(
            got.windows(2).all(|pair| Arc::ptr_eq(&pair[0], &pair[1])),
            "twenty callers did not get the same one"
        );
    }

    /// An open that failed is tried again, and leaves nothing behind.
    #[tokio::test]
    async fn an_open_that_failed_is_tried_again() {
        let registry: Registry<&str, u32> = Registry::with_capacity(16);

        assert!(
            registry
                .get_or_open("flaky", || async { Err(anyhow::anyhow!("no")) })
                .await
                .is_err()
        );
        assert!(
            registry.opened(&"flaky").is_none(),
            "a failed open left something behind"
        );
        assert_eq!(
            *registry
                .get_or_open("flaky", || async { Ok(3) })
                .await
                .expect("the second attempt"),
            3,
            "the key was poisoned by the first attempt"
        );
    }

    /// A key being opened is not evicted out from under its opener.
    #[tokio::test]
    async fn a_key_being_opened_is_not_evicted() {
        const CAPACITY: usize = 4;
        let registry: Arc<Registry<String, u32>> = Arc::new(Registry::with_capacity(CAPACITY));
        let (started, opening) = tokio::sync::oneshot::channel();
        let (release, released) = tokio::sync::oneshot::channel::<()>();

        let slow = {
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                registry
                    .get_or_open("held".to_string(), || async move {
                        started.send(()).expect("nobody waiting");
                        released.await.expect("never released");
                        Ok(99)
                    })
                    .await
            })
        };
        opening.await.expect("the open never started");

        // Enough other keys to drive eviction well past the bound.
        for i in 0..CAPACITY * 3 {
            registry
                .get_or_open(format!("other {i}"), || async move { Ok(i as u32) })
                .await
                .expect("opening another key");
        }

        release.send(()).expect("the slow open went away");
        assert_eq!(*slow.await.expect("joining").expect("the held open"), 99);
        assert_eq!(
            registry.opened(&"held".to_string()).map(|held| *held),
            Some(99),
            "the key was evicted while it was being opened"
        );
    }
}
