-- The five retrieval-signal columns V1 gave every state go.
--
-- `importance`, `worth_positive` and `worth_negative` were meant for
-- post-fusion modifiers, and `access_count` and `last_accessed_at` for access
-- tracking. No build of this project has ever written any of them: every row
-- holds the default it was inserted with. The modifiers that read the first
-- three multiplied every result by exactly 1.0 and were removed, and with them
-- the last reader. A feature that wants them back needs a write path first,
-- and can add the column it writes in the same change.
--
-- The check comes first, the way V9's does: if a single state holds anything
-- but the default -- written by hand, or by a build this project never
-- shipped -- the statement below raises an error naming that state, the
-- transaction the runner wraps this migration in rolls back, and the columns
-- stay. A workspace migrates only if dropping them drops no information.
SELECT CAST(
           'topic_states has a retrieval signal set; not dropping it: '
           || CAST(id AS TEXT) AS INTEGER
       )
  FROM topic_states
 WHERE importance <> 0
    OR worth_positive <> 0
    OR worth_negative <> 0
    OR access_count <> 0
    OR last_accessed_at IS NOT NULL
 LIMIT 1;

-- As with V9, rows written before this keep their bytes until something
-- rewrites the table; rows written after it are smaller.
ALTER TABLE topic_states
    DROP COLUMN importance,
    DROP COLUMN worth_positive,
    DROP COLUMN worth_negative,
    DROP COLUMN access_count,
    DROP COLUMN last_accessed_at;
