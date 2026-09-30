-- The projection holds one document per topic rather than one per state, so
-- there is no longer a job that removes a state from it.
--
-- A topic that stands for nothing after a soft delete has its document removed
-- by the same job that would have rewritten it, because with one document per
-- topic those are the same operation on the same key. Nothing ever enqueued the
-- removed kind -- there is no delete command yet -- so no row can carry it.
ALTER TABLE index_jobs DROP CONSTRAINT index_jobs_job_type_known;

ALTER TABLE index_jobs ADD CONSTRAINT index_jobs_job_type_known CHECK (
    job_type IN (
        'sync_topic_index', 'derive_mentions',
        'backfill_mentions', 'optimize_index'
    )
);
