//! The spans memories were cut into, kept so that a search does not cut them
//! again.
//!
//! A search over an index cut into spans shows the reranker a window of each
//! candidate it scores and gives the caller a window of each hit, and finding
//! a window means cutting the memory: [`pamin_index::spans::pack`] tokenizes
//! all of it, then counts every block and the cuts inside the long ones. That
//! is the work the index already did when it wrote the memory, done again for
//! every candidate of every search, and it grows with the memory's length, not
//! the window's.
//!
//! Kept by the state that was cut, which never changes, and the name its spans
//! were headed with, which can, so an entry is never stale: a memory rewritten
//! is a new state and a memory renamed is a new key.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pamin_core::TopicStateId;
use pamin_index::spans::Span;

/// How many bytes of spans one generation holds before it is made the previous
/// one. What is held at most is twice this: a full previous generation and a
/// current one about to be retired.
const GENERATION_BYTES: usize = 32 << 20;

type Key = (TopicStateId, String);

/// Cuts, bounded by the bytes they hold.
///
/// Two generations rather than an ordered list: the current one fills until it
/// holds [`GENERATION_BYTES`], then becomes the previous one and a new one
/// starts. A cut found in the previous generation moves to the current one, so
/// what searches keep reaching stays, and what they stopped reaching is dropped
/// one generation later -- least-recently-used, at the grain of a generation,
/// for two map operations a hit.
pub(crate) struct Cuts {
    held: Mutex<Generations>,
    budget: usize,
}

#[derive(Default)]
struct Generations {
    current: HashMap<Key, Arc<[Span]>>,
    bytes: usize,
    previous: HashMap<Key, Arc<[Span]>>,
}

impl Default for Cuts {
    fn default() -> Self {
        Self::with_budget(GENERATION_BYTES)
    }
}

impl Cuts {
    fn with_budget(budget: usize) -> Self {
        Self {
            held: Mutex::new(Generations::default()),
            budget,
        }
    }

    /// The spans `state` was cut into under `name`, cutting it with `cut` only
    /// when no search or write has yet.
    ///
    /// The cut runs outside the lock, so two searches reaching the same memory
    /// at once may both cut it; they cut it identically, and the second keeps
    /// what it cut.
    pub(crate) fn get_or_cut(
        &self,
        state: TopicStateId,
        name: &str,
        cut: impl FnOnce() -> Vec<Span>,
    ) -> Arc<[Span]> {
        let key = (state, name.to_string());
        if let Some(spans) = self.lock().get(&key, self.budget) {
            return spans;
        }
        let spans: Arc<[Span]> = cut().into();
        self.lock().insert(key, Arc::clone(&spans), self.budget);
        spans
    }

    /// Records what a write cut `state` into, which the searches that follow
    /// it would otherwise cut again.
    pub(crate) fn put(&self, state: TopicStateId, name: &str, spans: Vec<Span>) {
        self.lock()
            .insert((state, name.to_string()), spans.into(), self.budget);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Generations> {
        self.held.lock().expect("the span cut lock is poisoned")
    }
}

impl Generations {
    fn get(&mut self, key: &Key, budget: usize) -> Option<Arc<[Span]>> {
        if let Some(spans) = self.current.get(key) {
            return Some(Arc::clone(spans));
        }
        let spans = self.previous.remove(key)?;
        self.insert(key.clone(), Arc::clone(&spans), budget);
        Some(spans)
    }

    fn insert(&mut self, key: Key, spans: Arc<[Span]>, budget: usize) {
        let size = bytes(&spans);
        if self.bytes + size > budget && !self.current.is_empty() {
            self.previous = std::mem::take(&mut self.current);
            self.bytes = 0;
        }
        self.bytes += size;
        if let Some(replaced) = self.current.insert(key, spans) {
            self.bytes -= bytes(&replaced);
        }
    }
}

/// What a cut holds: each span's text and header, and the span itself.
fn bytes(spans: &[Span]) -> usize {
    spans
        .iter()
        .map(|span| std::mem::size_of::<Span>() + span.text.len() + span.header.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(text: &str) -> Span {
        Span {
            ordinal: 0,
            range: 0..text.len(),
            header: String::new(),
            text: text.to_string(),
            tokens: 1,
        }
    }

    #[test]
    fn a_held_cut_is_not_cut_again() {
        let cuts = Cuts::default();
        let state = TopicStateId::new();
        let first = cuts.get_or_cut(state, "name", || vec![span("one")]);
        let again = cuts.get_or_cut(state, "name", || panic!("cut again"));
        assert!(Arc::ptr_eq(&first, &again));
    }

    #[test]
    fn a_new_name_is_cut_again() {
        let cuts = Cuts::default();
        let state = TopicStateId::new();
        cuts.put(state, "old", vec![span("one")]);
        let renamed = cuts.get_or_cut(state, "new", || vec![span("two")]);
        assert_eq!(renamed[0].text, "two");
    }

    #[test]
    fn what_searches_keep_reaching_survives_a_generation() {
        let size = bytes(&[span("0123456789")]);
        let cuts = Cuts::with_budget(2 * size);
        let (kept, dropped) = (TopicStateId::new(), TopicStateId::new());
        cuts.put(kept, "", vec![span("0123456789")]);
        cuts.put(dropped, "", vec![span("0123456789")]);
        // Fills the first generation: both move to the previous one.
        cuts.put(TopicStateId::new(), "", vec![span("0123456789")]);
        // Reached, so moved to the current generation.
        cuts.get_or_cut(kept, "", || panic!("dropped too soon"));
        // Retires the generation holding `dropped`.
        cuts.put(TopicStateId::new(), "", vec![span("0123456789")]);
        cuts.get_or_cut(kept, "", || panic!("dropped what was reached"));
        let mut cut = false;
        cuts.get_or_cut(dropped, "", || {
            cut = true;
            vec![span("0123456789")]
        });
        assert!(cut, "what was not reached is dropped a generation later");
    }
}
