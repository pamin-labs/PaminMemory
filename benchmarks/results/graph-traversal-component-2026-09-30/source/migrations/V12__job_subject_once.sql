-- A queued job says what it is about once.
--
-- A job is one project's kind of work on one subject, and the row said the
-- subject three times over: in `payload` as JSON, and inside
-- `idempotency_key` as `<kind>:<subject>` text, where the kind is also
-- `job_type`. The key existed to make the three one row -- the enqueue
-- conflicts on it -- and its unique index was the queue's largest, 126 MB on
-- a synthetic queue of 1,054,646 rows, beside 59 MB of payloads and 57 MB of
-- keys. The subject is now a column of its own, and the uniqueness the key
-- carried is stated on the columns it was spelled from.
--
-- `NULLS NOT DISTINCT`, because project-wide work has no subject and must
-- still coalesce: two `optimize_index` rows for one project are one job owed,
-- as the key `optimize_index:` made them.
--
-- A subject that is not a UUID becomes no subject, which is what the reader
-- this replaces made of it. If that, or anything else, would leave two rows
-- naming the same work, adding the constraint fails, the transaction the
-- runner wraps this migration in rolls back, and the queue is left as it was
-- rather than merged.
--
-- As with V9 and V11, rows queued before this keep their bytes until something
-- rewrites the table; the queue turns over as its work is done, and every row
-- queued after it is smaller.
ALTER TABLE index_jobs ADD COLUMN subject UUID;

UPDATE index_jobs
   SET subject = CASE
           WHEN payload->>'subject'
                ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
           THEN CAST(payload->>'subject' AS UUID)
       END;

ALTER TABLE index_jobs
    DROP CONSTRAINT index_jobs_project_idempotency_key_key,
    DROP COLUMN idempotency_key,
    DROP COLUMN payload,
    ADD CONSTRAINT index_jobs_project_kind_subject_key
        UNIQUE NULLS NOT DISTINCT (project_id, job_type, subject);
