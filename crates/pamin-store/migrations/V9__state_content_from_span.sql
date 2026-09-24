-- A topic state's content is the text of its span, so it is read from the
-- evidence the span points into rather than stored a second time.
--
-- V1 gave every state a `content` column beside `source_span_id`, and the only
-- writer has always filled it with exactly the bytes that span covers. Checked
-- on the evaluation workspace, all 425,916 states equalled their span's slice
-- of `source_versions.content`, with no exceptions -- the same text held twice,
-- once as evidence and once as the state derived from it.
--
-- The check comes first, and it is what keeps this from losing anything: if a
-- single state says something other than its span, the statement below raises
-- an error naming that state, the transaction the runner wraps this migration
-- in rolls back, and the column stays where it is. A workspace migrates only
-- if dropping the column drops no information.
--
-- Offsets are bytes of UTF-8, which is what the application writes and what
-- `SourceSpan` documents, so the comparison is on bytes: SQL's `substring` on
-- text counts characters, and would disagree at the first character outside
-- ASCII. The cast is how a portable statement fails on purpose -- the value
-- depends on the row, so nothing can fold it away before a row is found, and
-- the error message carries the reason.
--
-- An aggregate rather than `LIMIT 1`, because the limit is what the planner
-- plans for: expecting to stop at the first mismatch, it chose primary-key
-- probes -- 850,000 of them through 3.4M buffers on a 425,916-state
-- workspace, 3.5-4.0 s of a first start that then looked hung. Every row has
-- to be read when none differ, which is the case that matters, and asked for
-- every row the planner hash-joins in parallel: 0.83-0.92 s over the same
-- data, and still one error naming a state when any differs.
SELECT CAST(
           'topic_states.content differs from its span; not dropping it: '
           || MIN(CAST(ts.id AS TEXT)) AS INTEGER
       )
  FROM topic_states ts
  JOIN source_spans sp ON sp.id = ts.source_span_id
  JOIN source_versions sv ON sv.id = sp.source_version_id
 WHERE convert_to(ts.content, 'UTF8')
       <> substring(convert_to(sv.content, 'UTF8')
                    FROM sp.byte_start + 1 FOR sp.byte_end - sp.byte_start)
HAVING COUNT(*) > 0;

-- Dropping a column marks it dropped rather than rewriting the table, so rows
-- written before this keep their old bytes until something rewrites them; rows
-- written after it are smaller. Nothing here rewrites the table, because the
-- one statement that does, `VACUUM FULL`, cannot run inside the transaction a
-- migration runs in.
ALTER TABLE topic_states DROP COLUMN content;
