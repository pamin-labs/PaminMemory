//! A bounded cache that forgets in the order it learned.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// Values already computed, oldest first, holding at most `capacity`.
///
/// Insertion order rather than use order. Keeping a true LRU means writing to
/// the queue on every hit, and what this protects is milliseconds of inference;
/// a query asked twice is asked twice close together. The embedder's query
/// vectors and the reranker's scores both remember this way.
pub(crate) struct Remembered<K, V> {
    known: HashMap<K, V>,
    order: VecDeque<K>,
    capacity: usize,
}

impl<K: Eq + Hash + Clone, V> Remembered<K, V> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            known: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: std::borrow::Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.known.get(key)
    }

    /// Remembers a value, forgetting the oldest once full.
    ///
    /// A key already held has its value replaced and keeps its place in the
    /// queue, so rewriting one entry cannot push the others out.
    pub(crate) fn put(&mut self, key: K, value: V) {
        if self.known.insert(key.clone(), value).is_some() {
            return;
        }
        self.order.push_back(key);
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.known.remove(&oldest);
            }
        }
    }

    /// How many values are held.
    pub(crate) fn len(&self) -> usize {
        self.known.len()
    }
}

#[cfg(test)]
mod tests {
    use super::Remembered;

    /// The oldest is forgotten, and the cache stays the size it says.
    #[test]
    fn the_oldest_is_forgotten_once_it_is_full() {
        let mut remembered = Remembered::with_capacity(4);
        for n in 0..14u32 {
            remembered.put(n, n * 10);
        }
        assert_eq!(remembered.len(), 4);
        assert!(remembered.get(&9).is_none(), "the oldest survived");
        assert_eq!(remembered.get(&13), Some(&130), "the newest was lost");
        assert_eq!(remembered.get(&10), Some(&100), "a kept one was lost");
    }

    /// Rewriting a key must not queue it a second time.
    ///
    /// It would evict an entry per rewrite while leaving the rewritten one in
    /// the map, so the cache would hold fewer and fewer live values while
    /// reporting itself full.
    #[test]
    fn remembering_a_key_twice_does_not_shorten_the_cache() {
        let mut remembered = Remembered::with_capacity(4);
        for n in 0..8u32 {
            remembered.put("the same question", n);
        }
        assert_eq!(remembered.order.len(), 1);
        assert_eq!(remembered.get("the same question"), Some(&7));
    }
}
