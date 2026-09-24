//! Running the work a write left in the outbox.
//!
//! The handlers are the other half of the write path. A write commits what the
//! ledger owes the projection and stops there; these are what pay it.
//!
//! Every one of them is written to be safe to run twice, because at-least-once
//! delivery means it will be: a lease expires, a worker is killed between doing
//! the work and recording that it did, a job is requested again while it runs.
//! None of that needs handling as long as running a handler a second time
//! leaves the same result -- which is why jobs name a subject rather than an
//! event, and why "sync topic T" is the shape and "T changed" is not.

use anyhow::{Result, anyhow};

use pamin_core::{JobKind, TopicId};
use pamin_store::jobs::{self, Job};

use crate::engine::Engine;

/// How many jobs one round takes.
///
/// A round holds its jobs for the length of the lease, and every job it took
/// but has not reached yet is work nothing else will do in the meantime, which
/// argues for a small number. Flushing argues the other way, and louder: a
/// round is what one flush covers, and flushing per document rather than per
/// batch of thirty-two measured 13.6 documents a second against 328, with
/// 1,161 index files against 35. Sixty-four leaves a comfortable margin under
/// a sixty-second lease -- a round is a couple of seconds -- and is large
/// enough that the three jobs a new topic queues still leave twenty documents
/// under one flush.
const BATCH: i32 = 64;

/// How many applied writes may wait for one flush.
///
/// The bound is on the writes a flush covers, not on time, because the time
/// bound is the upkeep tick and that is already short. What this decides is
/// how much replay a power loss costs and how long a claim is held: a job in
/// here still holds its claim, and a claim is good for [`jobs::LEASE`], so the
/// list has to be flushed well inside that whatever the write rate.
///
/// A hundred and twenty-eight, from what a flush costs per document at each
/// batch: one is 57.7 ms and leaves 1,287 files, eight is 6.3 ms and 141,
/// thirty-two is 1.8 ms and 69, and a hundred and twenty-eight is 0.65 ms and
/// 51. The curve is flat past that and the cost of being wrong is not, so this
/// is the knee rather than the floor.
const AWAITING_DURABILITY: usize = 128;

/// What a drain did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Drained {
    /// Jobs that ran and were recorded as done.
    pub completed: usize,
    /// Jobs still owed when the drain stopped, counted no further than
    /// [`LAGGING_AT`](pamin_core::LAGGING_AT) past [`Drained::applied`].
    ///
    /// Counted from the queue, so it includes [`Drained::applied`] -- work this
    /// process has already done and is holding a claim on until a flush makes
    /// it durable. A caller asking "is this memory findable" wants the
    /// difference; a caller asking "what would replay after a power cut" wants
    /// this.
    ///
    /// Capped because every write drains, and the two questions a write asks
    /// of the difference -- is it zero, is it past the lag bound -- are both
    /// answered exactly below the cap. Counting a backlog in full costs a read
    /// of all of it, on every write, exactly when the writer is behind. A
    /// caller that reports the number, as `pamin cascade` does, counts it
    /// with `jobs::pending`.
    pub pending: i64,
    /// Jobs that failed and will be tried again, or have run out of attempts.
    pub failed: usize,
    /// Jobs whose writes the index has, waiting only for a flush.
    pub applied: usize,
}

/// How much of what is owed a drain is willing to pay for.
///
/// Compacting the index makes it faster and never makes it more correct, which
/// is the property that lets somebody other than the caller who caused it do
/// the work. A resident server has such a somebody; a command that exits after
/// one write does not, so it pays for its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owed {
    /// Everything. What `pamin cascade` means, and what a process with nothing
    /// behind it has to do for itself.
    Everything,
    /// What a memory needs before it can be found, and nothing that only makes
    /// finding it quicker.
    WhatAMemoryNeeds,
}

impl Owed {
    /// The kinds this claims.
    fn kinds(self) -> &'static [JobKind] {
        match self {
            Self::Everything => &JobKind::ALL,
            Self::WhatAMemoryNeeds => &JobKind::URGENT,
        }
    }
}

impl Engine {
    /// Runs queued work until there is none left that is due.
    ///
    /// Bounded by what is due rather than by a round count: a handler that
    /// schedules more work -- creating a topic schedules a backfill -- would
    /// otherwise leave it for whoever came next, and "drain" would mean
    /// something different each time it was called.
    pub async fn drain_cascade(&self, owed: Owed) -> Result<Drained> {
        let mut drained = Drained::default();
        // Once, at the end, and never again in this drain: the tidy-up is
        // itself a job, so queueing another after running one would spin.
        let mut tidied = false;

        loop {
            let claimed = jobs::claim(
                self.database.pool(),
                self.project,
                &self.worker,
                BATCH,
                owed.kinds(),
            )
            .await?;
            if claimed.is_empty() {
                // A drain that wrote anything ends by asking whether the index
                // has spread across more files than it should have. Queued
                // rather than called: maintenance is per-project work, and the
                // outbox is what makes one worker run it rather than every
                // worker racing to. The next round claims it.
                if drained.completed > 0
                    && !tidied
                    && (self.index_is_fragmented()? || self.vector_index_lags()?)
                {
                    tidied = true;
                    jobs::enqueue(
                        self.database.pool(),
                        self.project,
                        JobKind::OptimizeIndex,
                        None,
                    )
                    .await?;
                    // Scheduled either way, run here only when nobody else
                    // will. Leaving it queued is the whole of the difference
                    // between a write that pays for maintenance and one that
                    // does not.
                    if owed == Owed::Everything {
                        continue;
                    }
                }
                break;
            }

            // The jobs that write the index first, then one flush, then the
            // jobs that read it back. Two things follow from the order. A
            // backfill searches the projection for memories naming a new
            // topic, so what this round wrote has to be visible before it
            // runs -- previously it was, by accident, because every write
            // flushed itself, and only if the writing job happened to be
            // claimed first. And the flush lands once per round instead of
            // once per document, which is the difference measured in `BATCH`.
            let (writes, reads): (Vec<&Job>, Vec<&Job>) = claimed
                .iter()
                .partition(|job| job.kind == JobKind::SyncTopicIndex);

            let mut written: Vec<(&Job, Result<()>)> = Vec::with_capacity(claimed.len());
            self.sync_indexes(&writes, &mut written).await;

            // What happens to the writes now is the difference between a
            // caller with somebody behind it and one without. Either way they
            // are applied, and applied is findable: the projection buffers a
            // write in memory and a query reads that buffer. What is left is
            // durability, and a job stays claimed until a flush provides it.
            //
            // Without a server there is nobody to flush later, so this pays
            // for it -- one flush for the round, before anything is recorded
            // as done, so that a process dying here leaves the jobs owed
            // rather than marked complete against an index that never received
            // them.
            let mut outcomes: Vec<(&Job, Result<()>)> = Vec::with_capacity(claimed.len());
            match owed {
                Owed::Everything => {
                    if written.iter().any(|(_, result)| result.is_ok()) {
                        crate::engine::off_the_runtime(|| self.index().flush())?;
                    }
                    outcomes.append(&mut written);
                }
                Owed::WhatAMemoryNeeds => {
                    for (job, result) in written {
                        match result {
                            Ok(()) => self.await_durability(job),
                            // A failure put nothing in the buffer, so there is
                            // nothing for a flush to cover.
                            Err(error) => outcomes.push((job, Err(error))),
                        }
                    }
                }
            }

            // Whatever a read job writes goes to the ledger, which commits it,
            // so nothing it does is waiting on a flush. It reads the index,
            // and what this round wrote is already there to be read.
            self.run_reads(&reads, &mut outcomes).await;

            // The round's completions go in one statement. Separately they
            // cost more than the work they record -- a thousand of them is 165
            // ms one at a time and 26 batched -- and the queue cannot tell the
            // difference, because a job is addressed by subject and a crash
            // before this leaves every one of them owed either way.
            let mut done: Vec<&Job> = Vec::with_capacity(outcomes.len());
            for (job, outcome) in outcomes {
                match outcome {
                    Ok(()) => done.push(job),
                    Err(error) => {
                        tracing::warn!(
                            job = %job.kind,
                            attempts = job.attempts,
                            %error,
                            "cascade job failed"
                        );
                        jobs::fail(self.database.pool(), job, &self.worker, &error.to_string())
                            .await?;
                        drained.failed += 1;
                    }
                }
            }

            let completed = jobs::complete(self.database.pool(), &done, &self.worker).await?;
            drained.completed += completed.len();
            // Past the bound the flush is this caller's after all: the list
            // holds claims, and a claim is only good for so long.
            if self.awaiting_durability() >= AWAITING_DURABILITY {
                drained.completed += self.flush_what_is_applied().await?;
            }
            // Requested again while it ran, or the lease expired. Either way it
            // stays owed, and counting it complete here would be the lie the
            // claim guard exists to prevent.
            drained.failed += done.len() - completed.len();
        }

        // Read before the count, not after. Both move -- the server's flusher
        // is completing these rows as this runs -- and the order decides which
        // way a job caught between the two reads is wrong. Counted as applied
        // and then not as pending, it is reported findable, which it is.
        // Counted as pending and then not as applied, it is reported owed,
        // which it is not, and every write says so.
        drained.applied = self.awaiting_durability();
        drained.pending = jobs::pending_up_to(
            self.database.pool(),
            self.project,
            pamin_core::LAGGING_AT + drained.applied as i64,
        )
        .await?;
        Ok(drained)
    }

    /// Holds a claim on a write the index has but the disk does not.
    fn await_durability(&self, job: &Job) {
        self.unflushed()
            .expect("the durability queue lock is poisoned")
            .push(job.clone());
    }

    /// How many applied writes are waiting for a flush.
    pub fn awaiting_durability(&self) -> usize {
        self.unflushed()
            .expect("the durability queue lock is poisoned")
            .len()
    }

    /// Makes the applied writes durable, and completes the jobs they came from.
    ///
    /// One flush for all of them, then one statement to record them done. The
    /// order is the whole point and is the same order a round takes without a
    /// server: nothing is recorded as done until a flush has covered it, so a
    /// power cut here leaves every one of them owed and the ledger replays
    /// them.
    ///
    /// A job whose claim lapsed while it waited is not completed -- the
    /// completion names the claim it belongs to -- so it runs again. That
    /// costs an embedding and produces the same document, which is what every
    /// handler here is written to survive.
    ///
    /// Returns how many were recorded done.
    pub async fn flush_what_is_applied(&self) -> Result<usize> {
        let waiting: Vec<Job> = std::mem::take(
            &mut *self
                .unflushed()
                .expect("the durability queue lock is poisoned"),
        );
        if waiting.is_empty() {
            return Ok(0);
        }

        crate::engine::off_the_runtime(|| self.index().flush())?;

        let held: Vec<&Job> = waiting.iter().collect();
        let completed = jobs::complete(self.database.pool(), &held, &self.worker).await?;
        Ok(completed.len())
    }

    /// Runs one job.
    async fn run(&self, job: &Job) -> Result<()> {
        match job.kind {
            JobKind::SyncTopicIndex => self.sync_topic_index(subject(job)?.into()).await,
            JobKind::DeriveMentions => self.derive_topic_mentions(subject(job)?.into()).await,
            JobKind::BackfillMentions => self.backfill_topic(subject(job)?.into()).await,
            JobKind::OptimizeIndex => self.optimize_projection().await,
        }
    }

    /// Indexes every write job of a round in one forward pass.
    ///
    /// A round claims up to sixty-four jobs and this used to run them one at a
    /// time, so a round that indexed sixty-four topics paid sixty-four forward
    /// passes where the model can do one. The flush in this same round was
    /// already batched -- see `BATCH` -- so the pass was the half that got
    /// left, and `Embedder::embed_passages` had been sitting next to it all
    /// along, used only by the rebuild.
    ///
    /// **Each job still gets its own outcome, and that is the part worth being
    /// careful about.** The outbox retries per job, so a batch that failed as a
    /// unit would turn one unindexable document into sixty-four jobs owed --
    /// and the next round would batch the same sixty-four and fail again. So a
    /// failed batch falls back to indexing one at a time, which costs the
    /// passes only on the rounds that actually hit a problem and gives every
    /// job the verdict it would have had before.
    async fn sync_indexes<'j>(&self, jobs: &[&'j Job], into: &mut Vec<(&'j Job, Result<()>)>) {
        // A job whose subject will not parse is its own failure and must not
        // take the batch with it.
        let mut topics = Vec::with_capacity(jobs.len());
        for job in jobs {
            match subject(job) {
                Ok(subject) => topics.push((*job, TopicId::from(subject))),
                Err(error) => into.push((*job, Err(error))),
            }
        }
        if topics.is_empty() {
            return;
        }

        let wanted: Vec<TopicId> = topics.iter().map(|(_, topic)| *topic).collect();
        let states = match pamin_store::repository::current_states_of(
            self.database.pool(),
            self.project,
            &wanted,
        )
        .await
        {
            Ok(states) => states,
            // The read failed for all of them, so it failed for all of them.
            // Nothing was attempted, so nothing is half done. The message is
            // formatted once and each job gets its own error, because the
            // store's error is not `Clone` and a job's outcome has to be its
            // own value.
            Err(error) => {
                let message = error.to_string();
                for (job, _) in topics {
                    into.push((
                        job,
                        Err(anyhow!("reading the states this round owes: {message}")),
                    ));
                }
                return;
            }
        };

        // A topic that resolves to nothing has its document removed instead,
        // which is the same job because the projection holds one document per
        // topic. Both halves are batched; both are all-or-nothing, which the
        // fallback below is for.
        let found: std::collections::HashMap<TopicId, &pamin_core::TopicState> =
            states.iter().map(|state| (state.topic_id, state)).collect();
        let (present, absent): (Vec<TopicId>, Vec<TopicId>) = wanted
            .iter()
            .copied()
            .partition(|topic| found.contains_key(topic));
        let indexing: Vec<pamin_core::TopicState> = present
            .iter()
            .filter_map(|topic| found.get(topic).map(|state| (*state).clone()))
            .collect();

        let batched = async {
            self.index_states(&indexing).await?;
            if !absent.is_empty() {
                crate::engine::off_the_runtime(|| self.index().delete(&absent))?;
            }
            Ok::<(), anyhow::Error>(())
        }
        .await;

        match batched {
            Ok(()) => {
                for (job, _) in topics {
                    into.push((job, Ok(())));
                }
            }
            // One at a time, so the job that cannot be indexed is the only one
            // recorded as failing. Costs the passes only on a round that hit a
            // problem.
            Err(_) => {
                for (job, topic) in topics {
                    let one = match found.get(&topic) {
                        Some(state) => self.index_state(state).await,
                        None => crate::engine::off_the_runtime(|| self.index().delete(&[topic]))
                            .map_err(Into::into),
                    };
                    into.push((job, one));
                }
            }
        }
    }

    /// Runs the jobs of a round that read the index back, restating every
    /// memory's mentions together and backfilling every new topic together.
    ///
    /// Batched the way [`Self::sync_indexes`] batches the writes, and for the
    /// same reason: each job asked the same few statements with different
    /// arguments, sixty-four times a round. One connection for all of them,
    /// the states to restate in one lookup and the names to backfill in
    /// another, and the rest in [`Engine::restate_mentions`] and
    /// [`Engine::backfill_all`]. Maintenance runs one job at a time as before.
    ///
    /// **Each job still gets its own outcome.** A batch that fails is run again
    /// one job at a time, so a memory that cannot be restated fails alone
    /// rather than holding the rest of the round's jobs owed with it.
    async fn run_reads<'j>(&self, jobs: &[&'j Job], into: &mut Vec<(&'j Job, Result<()>)>) {
        let mut batch = Vec::new();
        for job in jobs {
            if !matches!(
                job.kind,
                JobKind::DeriveMentions | JobKind::BackfillMentions
            ) {
                into.push((*job, self.run(job).await));
                continue;
            }
            match subject(job) {
                Ok(subject) => batch.push((*job, TopicId::from(subject))),
                Err(error) => into.push((*job, Err(error))),
            }
        }
        if batch.is_empty() {
            return;
        }

        let of_kind = |kind: JobKind| -> Vec<TopicId> {
            batch
                .iter()
                .filter(|(job, _)| job.kind == kind)
                .map(|(_, topic)| *topic)
                .collect()
        };
        let batched = async {
            let mut connection = self.database.pool().acquire().await?;
            // A topic that resolves to nothing has nothing to restate, and one
            // that does not exist has no name to backfill, as one at a time.
            let states = pamin_store::repository::current_states_of(
                &mut *connection,
                self.project,
                &of_kind(JobKind::DeriveMentions),
            )
            .await?;
            self.restate_mentions(&mut connection, &states).await?;
            let named: Vec<(TopicId, String)> = pamin_store::repository::topics_by_id(
                &mut *connection,
                self.project,
                &of_kind(JobKind::BackfillMentions),
            )
            .await?
            .into_iter()
            .map(|(topic, name, _)| (topic, name))
            .collect();
            self.backfill_all(&mut connection, &named).await?;
            Ok::<(), anyhow::Error>(())
        }
        .await;

        match batched {
            Ok(()) => {
                for (job, _) in batch {
                    into.push((job, Ok(())));
                }
            }
            Err(_) => {
                for (job, _) in batch {
                    into.push((job, self.run(job).await));
                }
            }
        }
    }

    /// Writes a topic's current state into the projection.
    ///
    /// Reads the state at execution rather than taking one from the job, which
    /// is what makes the job coalesce: fourteen edits leave one row and this
    /// runs once, against the fourteenth.
    ///
    /// A topic that now resolves to nothing -- every state soft deleted --
    /// has its document removed rather than left behind. That is the same job
    /// because the projection holds one document per topic: there is no state
    /// to unindex separately from the topic it belonged to.
    async fn sync_topic_index(&self, topic: TopicId) -> Result<()> {
        let states = pamin_store::repository::current_states_of(
            self.database.pool(),
            self.project,
            &[topic],
        )
        .await?;

        let Some(state) = states.first() else {
            return crate::engine::off_the_runtime(|| self.index().delete(&[topic]))
                .map_err(Into::into);
        };

        self.index_state(state).await
    }

    /// Recomputes the edges a topic's current content implies.
    async fn derive_topic_mentions(&self, topic: TopicId) -> Result<()> {
        let states = pamin_store::repository::current_states_of(
            self.database.pool(),
            self.project,
            &[topic],
        )
        .await?;

        let Some(state) = states.first().cloned() else {
            return Ok(());
        };

        self.derive_mentions(&state).await?;
        Ok(())
    }

    /// Links a topic to memories written before it existed that already name it.
    async fn backfill_topic(&self, topic: TopicId) -> Result<()> {
        let topics =
            pamin_store::repository::topics_by_id(self.database.pool(), self.project, &[topic])
                .await?;

        let Some((_, name, _)) = topics.first() else {
            return Ok(());
        };

        self.backfill_mentions(topic, name).await?;
        Ok(())
    }

    /// Runs the maintenance a write left for somebody else, if any is owed.
    ///
    /// The other half of [`Owed::WhatAMemoryNeeds`]. A write schedules this and
    /// returns; a process that is going to be here afterwards runs it.
    ///
    /// **It does not make writes quicker, and it was measured rather than
    /// assumed.** Two hundred and fifty writes with the model already warm:
    ///
    /// ```text
    ///                      median   p90   p95   p99    max   total
    ///     the writer runs it  124    148   177   564   1112   34.8 s
    ///     this runs it        127    156   166   662   1530   35.2 s
    /// ```
    ///
    /// Indistinguishable, and the index's own lock is why: one caller at a time
    /// means a write waits out a compaction whether or not it is the one doing
    /// it. Moving the work off the writer does not move the wait, and nothing
    /// on this side of the engine can -- that needs maintenance able to run
    /// alongside a query, which is the same thing the mutex is there for.
    ///
    /// What it does buy is that the wait no longer lands on whichever caller
    /// happened to cross the budget, and that the shape is the one this can be
    /// fixed from: when the engine can serve a query during its own upkeep,
    /// the upkeep is already somewhere else.
    ///
    /// It claims like any other worker, so several servers against one
    /// workspace do not duplicate the work, and a tick that finds nothing owed
    /// costs one query.
    ///
    /// Returns whether it did anything, which is what a caller waiting for a
    /// quiet moment wants to know.
    pub async fn maintain(&self) -> Result<bool> {
        let claimed = jobs::claim(
            self.database.pool(),
            self.project,
            &self.worker,
            BATCH,
            &JobKind::MAINTENANCE,
        )
        .await?;
        if claimed.is_empty() {
            return Ok(false);
        }

        let mut done: Vec<&Job> = Vec::with_capacity(claimed.len());
        for job in &claimed {
            match self.run(job).await {
                Ok(()) => done.push(job),
                Err(error) => {
                    tracing::warn!(job = %job.kind, %error, "maintenance failed");
                    jobs::fail(self.database.pool(), job, &self.worker, &error.to_string()).await?;
                }
            }
        }
        jobs::complete(self.database.pool(), &done, &self.worker).await?;

        Ok(true)
    }

    /// Whether the index is spread across more files than it should be.
    ///
    /// What this replaced looked at documents, because it was written for the
    /// graph: building one is superlinear in documents, so gating it on
    /// documents is right. Compaction is not the same cost with the same
    /// argument -- files accumulate with *writes*, about two per write, whatever
    /// the document count -- and gating both on one condition meant the second
    /// never ran. Ten documents rewritten two hundred times left four hundred
    /// and twenty files, growing without bound, and a workspace used normally
    /// for a week died of `Too many open files`.
    ///
    /// This condition is compaction's alone. It used to carry the graph as
    /// well, on the argument that "a segment seals after thousands of writes
    /// and this fires every sixty or so, so by the time a segment is due a
    /// graph the index has already been asked many times over". That was true
    /// when it was written and stopped being true in the same release that
    /// moved the flush to the server: files then accumulate at about 0.045 a
    /// write rather than two, so this fires roughly every five and a half
    /// thousand writes instead of every sixty, and a project below that never
    /// built a graph at all. See `vector_index_lags`.
    fn index_is_fragmented(&self) -> Result<bool> {
        let files = crate::engine::off_the_runtime(|| self.index().file_count())?;
        Ok(pamin_index::is_fragmented(files))
    }

    /// Whether enough documents sit outside the vector graph to build one.
    ///
    /// The graph's own condition, which it went a release without. Both
    /// conditions queue the same job -- `optimize` compacts *and* builds -- so
    /// this adds a reason to run it, not a second kind of maintenance.
    fn vector_index_lags(&self) -> Result<bool> {
        let (documents, completeness) =
            crate::engine::off_the_runtime(|| -> Result<(u64, f32)> {
                let index = self.index();
                Ok((index.document_count()?, index.vector_index_completeness()?))
            })?;
        Ok(pamin_index::vector_index_lags(documents, completeness))
    }

    /// Compacts the index, and builds a graph over any segment that sealed.
    ///
    /// Both happen in one call because the engine does them in one call, and
    /// it skips whichever is already done -- which is what makes asking cheap
    /// enough to ask often.
    ///
    /// Runs on a handle rather than under the index lock, so the searches this
    /// is speeding up are still answered while it runs. The engine supports
    /// that and has since before the version this depends on; holding the lock
    /// across it was our own doing. [`Engine::index_for_upkeep`] carries the
    /// argument and the one measurement that has to keep passing for it.
    ///
    /// [`Engine::index_for_upkeep`]: crate::engine::Engine::index_for_upkeep
    async fn optimize_projection(&self) -> Result<()> {
        let index = self.index_for_upkeep();
        crate::engine::off_the_runtime(|| index.optimize())?;
        Ok(())
    }
}

/// The identifier a job is about.
fn subject(job: &Job) -> Result<uuid::Uuid> {
    job.subject
        .ok_or_else(|| anyhow!("a {} job was queued without a subject", job.kind))
}
