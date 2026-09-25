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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

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
    /// How many entries the capacity bound has closed since
    /// [`take_evicted`](Self::take_evicted) last asked.
    evicted: AtomicUsize,
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
    /// When it was last wanted, for the caller that closes idle entries.
    ///
    /// Beside the counter rather than replacing it: `used` orders eviction and
    /// must not be able to go backwards, and this answers a different question
    /// -- *how long* since anyone wanted this -- which a counter cannot.
    at: Instant,
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
            evicted: AtomicUsize::new(0),
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
                entry.at = Instant::now();
                Arc::clone(&entry.slot)
            } else {
                let evicted = registry.make_room(self.capacity);
                self.evicted.fetch_add(evicted, Ordering::Relaxed);
                let slot = Slot::default();
                registry.slots.insert(
                    key,
                    Entry {
                        // Cloned before the lock goes, so there is no moment
                        // when a slot being opened looks idle to `make_room`.
                        slot: Arc::clone(&slot),
                        used: now,
                        at: Instant::now(),
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

    /// Whether this key has a place here, open or being opened.
    ///
    /// Does not count as a use, for the same reason [`Registry::opened`]
    /// does not.
    pub fn holds(&self, key: &K) -> bool {
        self.open
            .lock()
            .expect("the registry lock is poisoned")
            .slots
            .contains_key(key)
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

    /// How many entries the capacity bound has closed since the last call.
    ///
    /// For the caller that gives freed memory back to the operating system,
    /// which a request should not wait for: the bound closes an entry on the
    /// way to opening another, and that request has somewhere to be.
    pub fn take_evicted(&self) -> usize {
        self.evicted.swap(0, Ordering::Relaxed)
    }

    /// Closes everything nothing has wanted for `idle`, and says what it closed.
    ///
    /// The capacity bound stops idle entries *accumulating*; it does nothing
    /// about the ones already there. Sixteen indexes a server opened this
    /// morning and has not been asked for since cost what sixteen busy ones
    /// cost -- the module above this says so in as many words, *an open index
    /// is not free and does not become free by being idle* -- and on the
    /// shipping profile that is about 205 MB each. This is the half of the
    /// policy that gives them back.
    ///
    /// Only entries nothing is using, by exactly the test [`Open::make_room`]
    /// uses and for exactly the same reason: dropping the registry's handle
    /// while a request holds its own frees nothing and costs the next caller
    /// for that key a second open.
    pub fn close_idle(&self, idle: Duration) -> Vec<K> {
        let mut registry = self.open.lock().expect("the registry lock is poisoned");
        let now = Instant::now();

        let closing: Vec<K> = registry
            .slots
            .iter()
            .filter(|(_, entry)| now.saturating_duration_since(entry.at) >= idle)
            .filter(|(_, entry)| Open::<K, T>::is_unused(entry))
            .map(|(key, _)| key.clone())
            .collect();

        for key in &closing {
            registry.slots.remove(key);
        }
        closing
    }

    /// The value for this key if it is open and idle, without opening one.
    ///
    /// `None` while another caller holds the slot -- opening it, or rebuilding
    /// it. A caller walking the keys skips those rather than waiting.
    pub fn opened(&self, key: &K) -> Option<Arc<T>> {
        // Deliberately does not count as a use. A sweep touches everything
        // and would flatten the order the eviction depends on.
        self.opened_counting(key, false)
    }

    /// [`Registry::opened`], counted as a use.
    ///
    /// For a caller about to do work on the value that is a reason to keep it
    /// open: stamped now for the idle sweep and moved to the back of the
    /// eviction order, as a caller of [`Registry::get_or_open`] would be.
    pub fn opened_for_work(&self, key: &K) -> Option<Arc<T>> {
        self.opened_counting(key, true)
    }

    fn opened_counting(&self, key: &K, counts: bool) -> Option<Arc<T>> {
        let slot = {
            let mut registry = self.open.lock().expect("the registry lock is poisoned");
            let now = registry.uses + 1;
            let entry = registry.slots.get_mut(key)?;
            let slot = Arc::clone(&entry.slot);
            if counts {
                entry.used = now;
                entry.at = Instant::now();
                registry.uses = now;
            }
            slot
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
    ///
    /// Returns how many it closed.
    fn make_room(&mut self, capacity: usize) -> usize {
        let mut closed = 0;
        while self.slots.len() >= capacity {
            let idle = self
                .slots
                .iter()
                .filter(|(_, entry)| Self::is_unused(entry))
                .min_by_key(|(_, entry)| entry.used)
                .map(|(key, _)| key.clone());

            let Some(idle) = idle else { return closed };
            self.slots.remove(&idle);
            closed += 1;
        }
        closed
    }

    /// Whether dropping this entry would actually free what it holds.
    ///
    /// Shared by the two things that close entries, because getting it wrong
    /// in either place is the same bug: a value a request still holds stays
    /// alive whether or not the registry drops its handle, so closing it buys
    /// nothing and costs the next caller for that key a second open.
    fn is_unused(entry: &Entry<T>) -> bool {
        if Arc::strong_count(&entry.slot) != 1 {
            return false;
        }
        match entry.slot.try_lock() {
            // Empty means a load failed or has not run; there is nothing to
            // lose by dropping the place it holds.
            Ok(held) => held
                .as_ref()
                .is_none_or(|value| Arc::strong_count(value) == 1),
            // Somebody is inside it. Never wait for them here: this may run
            // under the map's lock, and the one lock order this design must not
            // invert is map-then-slot.
            Err(_) => false,
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

    /// An entry nothing has wanted for the window is closed; a fresh one is not.
    ///
    /// The capacity bound only stops idle entries accumulating. Sixteen
    /// indexes a server opened this morning cost what sixteen busy ones cost,
    /// about 205 MB each on the shipping profile, and nothing was giving them
    /// back.
    #[tokio::test]
    async fn an_entry_nothing_has_wanted_is_closed_and_a_fresh_one_is_not() {
        let registry: Registry<&str, u32> = Registry::with_capacity(16);
        registry
            .get_or_open("stale", || async { Ok(1) })
            .await
            .expect("opening");
        // Long enough to be past any window this asks about, without sleeping.
        tokio::time::sleep(Duration::from_millis(40)).await;
        registry
            .get_or_open("fresh", || async { Ok(2) })
            .await
            .expect("opening");

        let closed = registry.close_idle(Duration::from_millis(30));

        assert_eq!(closed, vec!["stale"], "the wrong set of keys was closed");
        assert!(
            registry.opened(&"stale").is_none(),
            "the stale entry is still held"
        );
        assert_eq!(
            registry.opened(&"fresh").map(|held| *held),
            Some(2),
            "an entry wanted a moment ago was closed"
        );
    }

    /// An entry a request is holding is not closed, however old it looks.
    ///
    /// The half that is not about time. Dropping the registry's handle while a
    /// caller holds its own frees nothing and makes the next caller for that
    /// key open a second copy alongside the first, which is the opposite of
    /// the point -- so `close_idle` applies the same in-use test the capacity
    /// bound does, and this is what fails if it stops.
    #[tokio::test]
    async fn an_entry_a_caller_is_holding_is_not_closed() {
        let registry: Registry<&str, u32> = Registry::with_capacity(16);
        let held = registry
            .get_or_open("busy", || async { Ok(5) })
            .await
            .expect("opening");

        tokio::time::sleep(Duration::from_millis(40)).await;
        let closed = registry.close_idle(Duration::from_millis(1));

        assert!(
            closed.is_empty(),
            "an entry a caller still holds was closed: {closed:?}"
        );
        assert_eq!(*held, 5);
        drop(held);

        assert_eq!(
            registry.close_idle(Duration::from_millis(1)),
            vec!["busy"],
            "the entry was not closed once nothing held it"
        );
    }

    /// What the capacity bound closes is counted, once.
    ///
    /// The server gives the heap back after an eviction rather than during
    /// one, so the request that caused it does not wait for the allocator; the
    /// count is how it learns there was one. Counting idle closes here too
    /// would trim twice for one close, and never counting would leave an
    /// evicted index's heap mapped until the idle sweep, which is the growth
    /// this exists to stop.
    #[tokio::test]
    async fn what_the_bound_closes_is_counted_once() {
        let registry: Registry<u32, u32> = Registry::with_capacity(2);
        for key in 0..3 {
            registry
                .get_or_open(key, || async move { Ok(key) })
                .await
                .expect("opening");
        }
        assert_eq!(registry.take_evicted(), 1, "the third key closed one");
        assert_eq!(registry.take_evicted(), 0, "and it is not counted again");

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(registry.close_idle(Duration::from_millis(1)).len(), 2);
        assert_eq!(
            registry.take_evicted(),
            0,
            "closing idle entries is not eviction"
        );
    }

    /// Work on an entry keeps it open; looking at it does not.
    ///
    /// The server's upkeep does both. Flushing and compacting only look, and
    /// counting them as uses would keep every index the server ever opened
    /// out of the idle sweep. Catching up on owed work is a reason to keep
    /// the index, and not counting it would let the sweep close a project
    /// half way through a backlog and the next tick open it again.
    #[tokio::test]
    async fn work_on_an_entry_keeps_it_open_and_looking_does_not() {
        let registry: Registry<&str, u32> = Registry::with_capacity(16);
        for key in ["looked_at", "worked_on"] {
            registry
                .get_or_open(key, || async { Ok(1) })
                .await
                .expect("opening");
        }
        tokio::time::sleep(Duration::from_millis(40)).await;

        assert!(registry.opened(&"looked_at").is_some());
        assert!(registry.opened_for_work(&"worked_on").is_some());

        assert_eq!(
            registry.close_idle(Duration::from_millis(30)),
            vec!["looked_at"],
            "the entry worked on a moment ago was closed, or the one only looked at was kept"
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
