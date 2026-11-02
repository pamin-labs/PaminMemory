-- The outbox becomes usable.
--
-- V1 created this table and predicted how it would be claimed, and no Rust
-- code has touched it since: `IndexJobId` was declared and never referenced.
-- What it was missing is everything a worker needs to be honest about a job it
-- picked up -- who has it, since when, why it failed last time, and what order
-- to take them in.
--
-- `job_type` is the only enumerated column in this schema with no CHECK, which
-- is what let it stay unused: nothing could write a wrong value because nothing
-- wrote at all. The constraint lists what `JobKind` produces, and a test holds
-- the two lists to each other.

ALTER TABLE index_jobs
    -- Who holds it, and since when. Together they are what distinguishes a job
    -- being worked on from one whose worker died holding it: the claim carries
    -- a lease in `available_at`, and when the lease expires the job is
    -- claimable again whatever `claimed_by` still says.
    ADD COLUMN claimed_at TIMESTAMPTZ,
    ADD COLUMN claimed_by TEXT,
    -- Why the last attempt failed. Without it a job that has exhausted its
    -- attempts says only that it did.
    ADD COLUMN last_error TEXT,
    -- Lower runs first. Deriving edges for a memory somebody just wrote is
    -- worth more than rebuilding a vector index in the background, and without
    -- an ordering the background work is in front of it as often as not.
    ADD COLUMN priority INTEGER NOT NULL DEFAULT 100;

ALTER TABLE index_jobs
    ADD CONSTRAINT index_jobs_job_type_known CHECK (job_type IN (
        'sync_topic_index', 'unindex_state', 'derive_mentions',
        'backfill_mentions', 'optimize_index'
    ));

-- The claim reads pending work in priority order and takes the first rows it
-- can lock. `index_jobs_claimable` leads with project_id, which answers "what
-- is pending for this project" and not "what should run next"; this one is
-- ordered the way the claim is.
--
-- Partial on the same predicate the claim uses, so completed jobs leave the
-- index as they are completed rather than accumulating in it. That matters
-- more here than elsewhere: every write adds a row, and all of them end up
-- completed.
CREATE INDEX index_jobs_by_priority
    ON index_jobs (priority, available_at, project_id)
    WHERE completed_at IS NULL;

-- Finding what failed, without scanning everything that succeeded.
CREATE INDEX index_jobs_exhausted
    ON index_jobs (project_id, attempts)
    WHERE completed_at IS NULL AND last_error IS NOT NULL;
