//! The plumbing every evaluation harness needs and none of them measures:
//! where its data lives, which profile it runs, and how a corpus gets into an
//! engine.
//!
//! Each harness had its own copy of each of these. They agreed, which is the
//! reason to keep one: a harness whose corpus went in differently from
//! another's would be comparing two write paths rather than two corpora.

// Included by harnesses that use only some of it, like the other shared
// modules here.
#![allow(dead_code)]

pub mod compiled;
pub mod shape;
pub mod vectors;

use std::path::PathBuf;

use pamin_core::Fusion;
use pamin_engine::{Depths, Drained, Engine, Owed, SearchHit, Write};

/// What records and projects call the one embedding model.
///
/// The name of the profile that shipped it, kept from when there were
/// profiles so that records and cached corpora written then are the ones
/// read and compared with now.
pub const PROFILE: &str = "accuracy";

/// This process's turn at loading models: held by an evaluation from
/// before it reads its settings ([`shape::Shape::publish`]) until its record
/// is written, and by every other test here that loads a model
/// ([`load_models`]), so no two of them load at once.
///
/// It is also the turn at the process environment: an experimental arm
/// that sets a variable the product reads (`PAMIN_RERANK_DEPTH`,
/// `PAMIN_SEARCH_EFFORT`) does so only while its thread holds the turn,
/// taking it first if it does not ([`environment_turn`]), so no
/// evaluation's settings snapshot reads another test's transient arm.
///
/// What the loads fetched and prepared is counted process wide
/// (`pamin_index::LoadActivity`), and libtest runs the tests of one binary
/// side by side: without turns, another test's download would land in an
/// evaluation's interval and name a warm run cold. The holder's thread
/// behind a condition variable rather than a held `MutexGuard`, so the turn
/// is `Send`, is let go of however the test ends, a panic included, and a
/// test can tell that it, not another, holds it.
///
/// And across processes: evaluations sharing an evaluation home share its
/// model cache, plan records and caches, so another process's download or
/// calibration would land in this one's interval just the same -- waited
/// for here, read back as warm or as calibrated before the run. The turn
/// therefore also holds an exclusive lock on `eval.lock` under
/// [`eval_home`], taken once the process's own turn is, so one process's
/// threads never wait on each other for it, and given up when the turn is.
/// Two evaluations sharing a home take turns whole: their loads,
/// calibrations and fetches never overlap.
pub struct ModelsTurn {
    across: Option<std::fs::File>,
}

static TURN: std::sync::Mutex<Option<std::thread::ThreadId>> = std::sync::Mutex::new(None);
static TURN_FREED: std::sync::Condvar = std::sync::Condvar::new();

/// Waits for this process's turn at loading models, and then for the
/// evaluation home's (see [`ModelsTurn`]), for this thread.
pub fn load_models() -> ModelsTurn {
    load_models_at(&eval_home().join("eval"))
}

/// [`load_models`], with the turn across processes taken on `<lock>.lock`.
fn load_models_at(lock: &std::path::Path) -> ModelsTurn {
    let mut holder = TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    while holder.is_some() {
        holder = TURN_FREED
            .wait(holder)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
    *holder = Some(std::thread::current().id());
    drop(holder);
    // Built before the lock is waited for, so a lock that cannot be taken
    // gives the process's turn back as it unwinds.
    let mut turn = ModelsTurn { across: None };
    turn.across = Some(vectors::lock(lock));
    turn
}

/// Whether this thread holds the turn (see [`ModelsTurn`]).
pub fn holds_turn() -> bool {
    *TURN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        == Some(std::thread::current().id())
}

/// The turn for code about to set the process environment: none if this
/// thread holds it already -- an evaluation, or a model-loading test --
/// and otherwise this thread's own, waited for, to hold while it sets it.
pub fn environment_turn() -> Option<ModelsTurn> {
    (!holds_turn()).then(load_models)
}

impl Drop for ModelsTurn {
    fn drop(&mut self) {
        // The home's turn first: a thread of this process waiting for the
        // process's turn would otherwise wait for this one's lock as well.
        drop(self.across.take());
        *TURN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        TURN_FREED.notify_one();
    }
}

/// A `PAMIN_` knob an arm sweeps through the process environment --
/// `PAMIN_RERANK_DEPTH` for depth variants, `PAMIN_SEARCH_EFFORT` for the
/// effort sweep -- claimed holding the turn.
///
/// A value the caller set is refused: it would name the record while no
/// pass ran under it. Dropped, panicking or not, the guard unsets the
/// knob, and it is dropped before the turn it was claimed under, so no
/// later evaluation reads a pass's value as its own setting. Every sweep
/// sets its knob through one of these; none sets the environment itself.
pub struct KnobSweep {
    name: &'static str,
}

impl KnobSweep {
    /// Claims `name` for a sweep. This thread must hold the turn.
    pub fn claim(name: &'static str) -> Self {
        assert!(holds_turn(), "a sweep sets {name} only holding the turn");
        assert!(
            std::env::var_os(name).is_none(),
            "this arm sets {name} itself: unset it for this run"
        );
        Self { name }
    }

    /// Sets the knob to `value` for the passes that follow.
    pub fn set(&self, value: &str) {
        // SAFETY: claimed holding the turn, which this guard is dropped
        // before; every test that reads the environment for its settings
        // waits for the turn.
        unsafe { std::env::set_var(self.name, value) };
    }

    /// Unsets the knob, for a pass under the shipped value.
    pub fn unset(&self) {
        // SAFETY: as for `set`.
        unsafe { std::env::remove_var(self.name) };
    }
}

impl Drop for KnobSweep {
    fn drop(&mut self) {
        self.unset();
    }
}

/// Where the harnesses keep datasets, models and workspaces between runs:
/// `PAMIN_EVAL_HOME`, or `pamin-eval` under the system's temporary directory.
pub fn eval_home() -> PathBuf {
    std::env::var("PAMIN_EVAL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("pamin-eval"))
}

/// Where one dataset lives: the directory `var` names, or `name` under
/// [`eval_home`].
pub fn dataset_dir(var: &str, name: &str) -> PathBuf {
    match std::env::var(var) {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => eval_home().join(name),
    }
}

/// [`PROFILE`], once nothing asks for another.
///
/// `PAMIN_PROFILE` chose the embedding model a harness measured. It chooses
/// nothing now, and a run that set it would be recorded under the name of a
/// model it did not use, so it is refused rather than ignored.
pub fn profile() -> String {
    if let Some(named) = std::env::var_os("PAMIN_PROFILE") {
        panic!(
            "PAMIN_PROFILE={} was removed: there is one embedding model. Unset it to continue.",
            named.to_string_lossy()
        );
    }
    PROFILE.to_string()
}

/// Whether a measured search reranks: the shipped search, or fusion alone.
///
/// The product has one search, and it reranks. What an arm compares it with
/// is fusion alone, [`Engine::search_fused`] -- the list the reranker is
/// handed, cut at the caller's limit. Named as the reranking tiers were, `off`
/// and `accurate`, so records written while there were tiers compare with
/// the ones written since.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Rerank {
    Off,
    #[default]
    Accurate,
}

impl Rerank {
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Accurate => "accurate",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "accurate" => Some(Self::Accurate),
            _ => None,
        }
    }

    /// The search this arm runs.
    pub async fn search(
        self,
        engine: &Engine,
        query: &str,
        limit: u32,
        depths: Depths,
    ) -> anyhow::Result<Vec<SearchHit>> {
        match self {
            Self::Off => {
                engine
                    .search_fused(query, limit, depths, Fusion::default())
                    .await
            }
            Self::Accurate => engine.search_reranked(query, limit, depths).await,
        }
    }
}

/// Writes one memory, promoted, unless its topic is already in the project.
///
/// Whether it wrote, so a harness resuming over a workspace it filled on an
/// earlier run can say how much of the corpus was new.
pub async fn write_absent(
    engine: &Engine,
    topic: &str,
    content: &str,
    language: &str,
    reason: &str,
) -> bool {
    let existing =
        pamin_store::repository::find_topic(engine.database.pool(), engine.project, topic)
            .await
            .expect("look for the topic");
    if existing.is_some() {
        return false;
    }
    engine
        .write(&Write {
            topic,
            content,
            content_hash: &content.len().to_string(),
            verdict: pamin_core::FilterDecision::Promoted,
            reason,
            promoted: true,
            language: Some(language),
            language_confidence: None,
            observed_at: time::OffsetDateTime::now_utc(),
            validity: pamin_core::Validity::ALWAYS,
        })
        .await
        .unwrap_or_else(|error| panic!("writing {topic}: {error}"));
    true
}

/// Hands `engine`'s index to `shape` under `label` before the run writes
/// any of its `total` documents (see `Shape::indexed`): loads the embedder,
/// so the producer that will embed them is known; the shape takes the
/// index's build lock and counts what it already holds.
pub async fn index(
    shape: &mut shape::Shape,
    label: &str,
    engine: &Engine,
    workspace: &std::path::Path,
    total: usize,
) {
    let loading = engine.clone();
    tokio::task::spawn_blocking(move || loading.load_embedder())
        .await
        .expect("the embedder load panicked")
        .expect("load the embedding model");
    shape.indexed(label, engine, workspace, total);
}

/// Runs the cascade until nothing is owed, and fails if anything still is.
pub async fn drain(engine: &Engine) -> Drained {
    let drained = engine
        .drain_cascade(Owed::Everything)
        .await
        .expect("drain the cascade");
    assert_eq!(
        drained.pending, 0,
        "the corpus is not fully indexed: {} jobs still owed",
        drained.pending
    );
    drained
}

#[cfg(test)]
mod tests {
    /// Each sweep's knob is refused when the caller set it, and a sweep that
    /// panics mid-pass leaves no value behind.
    #[test]
    fn a_knob_sweep_refuses_a_set_value_and_unsets_its_own() {
        let _turn = super::environment_turn();
        for name in ["PAMIN_RERANK_DEPTH", "PAMIN_SEARCH_EFFORT"] {
            let before = std::env::var_os(name);
            // SAFETY: this thread holds the turn.
            unsafe { std::env::set_var(name, "30") };
            assert!(std::panic::catch_unwind(|| super::KnobSweep::claim(name)).is_err());
            unsafe { std::env::remove_var(name) };

            let swept = std::panic::catch_unwind(|| {
                let sweep = super::KnobSweep::claim(name);
                sweep.set("15");
                panic!("a search failed mid-pass");
            });
            assert!(swept.is_err());
            assert_eq!(std::env::var_os(name), None, "{name}");
            if let Some(before) = before {
                // SAFETY: as above.
                unsafe { std::env::set_var(name, before) };
            }
        }
    }

    /// The turn waits for another process holding the evaluation home's --
    /// simulated by a second lock on the same file, which conflicts just as
    /// another process's would -- and is taken once that one lets go.
    #[test]
    fn the_turn_waits_for_another_process_s() {
        let dir = tempfile::tempdir().expect("temp dir");
        let lock = dir.path().join("eval");
        let other = super::vectors::lock(&lock);
        let (taken, waiting) = std::sync::mpsc::channel();
        let ours = std::thread::spawn({
            let lock = lock.clone();
            move || {
                let turn = super::load_models_at(&lock);
                taken.send(()).expect("say it took the turn");
                drop(turn);
            }
        });
        assert!(
            waiting
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err(),
            "the turn was taken while another process held it"
        );
        drop(other);
        waiting
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("the turn once the other process let go");
        ours.join().expect("the thread that took the turn");
    }
}
