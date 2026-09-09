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

use pamin_core::{JobKind, TopicId, TopicStateId};
use pamin_store::jobs::{self, Job};

use crate::engine::Engine;

/// How many jobs one round takes.
///
/// Small: a round holds its jobs for the length of the lease, and every job it
/// took but has not reached yet is work nothing else will do in the meantime.
const BATCH: i32 = 8;

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
    pub async fn drain_cascade(&mut self) -> Result<Drained> {
        let mut drained = Drained::default();

        loop {
            let claimed = jobs::claim(self.database.pool(), &self.worker, BATCH).await?;
            if claimed.is_empty() {
                break;
            }

            for job in &claimed {
                match self.run(job).await {
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

    /// Runs one job.
    async fn run(&mut self, job: &Job) -> Result<()> {
        match job.kind {
            JobKind::SyncTopicIndex => self.sync_topic_index(subject(job)?.into()).await,
            JobKind::UnindexState => self.unindex_state(subject(job)?.into()).await,
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
    /// leaves nothing to write, and the states themselves are removed by
    /// `unindex_state`.
    async fn sync_topic_index(&mut self, topic: TopicId) -> Result<()> {
        let states = pamin_store::repository::current_states_of(
            self.database.pool(),
            self.project,
            &[topic],
        )
        .await?;

        let Some(state) = states.first() else {
            return Ok(());
        };

        self.index_state(state).await
    }

    /// Removes a state from the projection.
    async fn unindex_state(&mut self, state: TopicStateId) -> Result<()> {
        let index = &self.index;
        crate::engine::off_the_runtime(|| {
            index.delete(&[state])?;
            index.flush()
        })?;

        Ok(())
    }

    /// Recomputes the edges a topic's current content implies.
    async fn derive_topic_mentions(&mut self, topic: TopicId) -> Result<()> {
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
    async fn backfill_topic(&mut self, topic: TopicId) -> Result<()> {
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
    async fn optimize_projection(&mut self) -> Result<()> {
        let index = &self.index;
        crate::engine::off_the_runtime(|| index.optimize())?;
        Ok(())
    }
}

/// The identifier a job is about.
fn subject(job: &Job) -> Result<uuid::Uuid> {
    job.subject
        .ok_or_else(|| anyhow!("a {} job was queued without a subject", job.kind))
}
