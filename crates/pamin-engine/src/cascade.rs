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

/// How many unindexed documents are worth a rebuild of the vector graph.
///
/// Written documents land in a flat buffer and only join the graph when the
/// index is optimized, so until then every vector query scans them. Optimizing
/// after each write would rebuild the graph for one document; never optimizing
/// leaves the graph the index was configured for unbuilt, which is what was
/// happening -- completeness sat at zero and the vector channel had been
/// brute-forcing since the index was created.
const UNINDEXED_BEFORE_OPTIMIZE: f32 = 100_000.0;

/// What a drain did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Drained {
    /// Jobs that ran and were recorded as done.
    pub completed: usize,
    /// Jobs that failed and will be tried again, or have run out of attempts.
    pub failed: usize,
    /// Jobs still owed when the drain stopped.
    pub pending: i64,
}

impl Engine {
    /// Runs queued work until there is none left that is due.
    ///
    /// Bounded by what is due rather than by a round count: a handler that
    /// schedules more work -- creating a topic schedules a backfill -- would
    /// otherwise leave it for whoever came next, and "drain" would mean
    /// something different each time it was called.
    pub async fn drain_cascade(&self) -> Result<Drained> {
        let mut drained = Drained::default();
        // Considered once, at the end, and never again in this drain. Queueing
        // a rebuild after running one would spin here if the rebuild ever
        // failed to reduce what is unindexed.
        let mut weighed_a_rebuild = false;

        loop {
            let claimed = jobs::claim(self.database.pool(), &self.worker, BATCH).await?;
            if claimed.is_empty() {
                // Queued at the end rather than by whoever wrote the hundred
                // thousandth document: a rebuild is per-project work, and the
                // outbox is what makes one worker run it rather than every
                // worker racing to. The next round claims it.
                if drained.completed > 0 && !weighed_a_rebuild {
                    weighed_a_rebuild = true;
                    if self.needs_optimizing()? {
                        jobs::enqueue(
                            self.database.pool(),
                            self.project,
                            JobKind::OptimizeIndex,
                            None,
                        )
                        .await?;
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

            let mut outcomes: Vec<(&Job, Result<()>)> = Vec::with_capacity(claimed.len());
            for job in writes {
                outcomes.push((job, self.run(job).await));
            }
            // Before anything is recorded as done, so that a process that dies
            // here leaves the jobs owed rather than marked complete against an
            // index that never received them.
            if outcomes.iter().any(|(_, result)| result.is_ok()) {
                crate::engine::off_the_runtime(|| self.writing().flush())?;
            }
            for job in reads {
                outcomes.push((job, self.run(job).await));
            }

            for (job, outcome) in outcomes {
                match outcome {
                    Ok(()) => {
                        if jobs::complete(self.database.pool(), job, &self.worker).await? {
                            drained.completed += 1;
                        } else {
                            // Requested again while it ran, or the lease
                            // expired. Either way it stays owed, and saying it
                            // completed here would be the lie the guard exists
                            // to prevent.
                            drained.failed += 1;
                        }
                    }
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
        }

        drained.pending = jobs::pending(self.database.pool(), self.project).await?;
        Ok(drained)
    }

    /// Whether enough has been written to be worth rebuilding the vector graph.
    fn needs_optimizing(&self) -> Result<bool> {
        let (documents, complete) = crate::engine::off_the_runtime(|| {
            let index = self.reading();
            Ok::<_, anyhow::Error>((index.document_count()?, index.vector_index_completeness()?))
        })?;

        Ok(documents as f32 * (1.0 - complete) >= UNINDEXED_BEFORE_OPTIMIZE)
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
            return crate::engine::off_the_runtime(|| self.writing().delete(&[topic]))
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

    /// Builds the vector index over everything written since the last build.
    async fn optimize_projection(&self) -> Result<()> {
        crate::engine::off_the_runtime(|| self.writing().optimize())?;
        Ok(())
    }
}

/// The identifier a job is about.
fn subject(job: &Job) -> Result<uuid::Uuid> {
    job.subject
        .ok_or_else(|| anyhow!("a {} job was queued without a subject", job.kind))
}
