-- A job is deleted when it is done, rather than marked done and kept.
--
-- Nothing reads a settled row. Every statement over this table asks about work
-- still owed -- `completed_at IS NULL` was in every predicate but the one that
-- deleted settled rows after an hour -- and `enqueue`'s upsert inserts a row in
-- exactly the state its revival of a settled one would have produced. So the
-- rows were kept for nobody, and on the evaluation workspace they were 651,128
-- of 1,054,646, in a table that was 38.7% of the database.
--
-- The rows go while `completed_at` still says which they are, and after the
-- three indexes below are dropped, so a delete of hundreds of thousands of rows
-- maintains the two indexes that stay rather than five. Nothing in flight is
-- touched: a claimed job has no completion. The new index is built last, over
-- only the rows still owed.

-- Three indexes answered two questions, and one answered none.
--
--   * `index_jobs_by_priority` (priority, available_at, project_id) was built
--     when a claim took the best job from any project. A claim is now one
--     project's, so it leads with the wrong column: the scan walks every
--     project's pending work in priority order to find this one's.
--   * `index_jobs_claimable` (project_id, available_at) served the per-project
--     reads -- counting what is pending, listing and replaying what failed --
--     through its first column alone. `available_at` is never a condition on
--     any of them.
--   * `index_jobs_exhausted` was partial on `last_error IS NOT NULL`, which no
--     statement says, so the planner could never prove a query fit inside it.
--     EXPLAIN on every statement in `jobs.rs` shows it unused.
--
-- One index ordered the way a claim reads -- project, then priority, then age
-- -- answers the claim directly and the per-project reads by its first column,
-- which is everything the three were used for. With settled rows gone none of
-- it needs to be partial.
DROP INDEX index_jobs_by_priority;
DROP INDEX index_jobs_claimable;
DROP INDEX index_jobs_exhausted;

DELETE FROM index_jobs WHERE completed_at IS NOT NULL;

ALTER TABLE index_jobs DROP COLUMN completed_at;

CREATE INDEX index_jobs_claim_order
    ON index_jobs (project_id, priority, available_at);
