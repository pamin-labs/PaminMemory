-- Answers "what has this project been writing about lately" without reading
-- every topic to find out.
--
-- `pamin topics` with no query lists the most recent, which is a bounded page
-- of an ordered scan -- and an ordered scan with no index behind it reads the
-- whole table and throws away all but the first twenty. At the size this store
-- is built for that is the difference between a lookup and a minute.
CREATE INDEX topics_by_recency ON topics (project_id, created_at DESC);
