-- Every span vector an index job or a rebuild embedded, so a rebuild, or a
-- node catching up on an index it lost, writes the vectors back rather than
-- running the model over every memory again.
--
-- Keyed by the hash of the text the vector was embedded from, not by topic:
-- a renamed topic embeds a new text and so gets a new row, and a span that
-- did not change is embedded once however often its topic is rewritten. And by
-- the model's revision and the plan that embedded it (`Embedder::encoding`),
-- since another model's vectors, or another plan's, of one text are not
-- interchangeable. A reader always names the revision, and the plan follows
-- the hash so one lookup finds a text's vectors under every plan, which is
-- what a rebuild that loads no model has to choose from.
--
-- `vector` is the fp16 codes, little-endian: what the index stores anyway. Its
-- width is the model's, so only its shape is checked here and a reader logs
-- and skips a vector of another width.
--
-- ponytail: rows only grow, with every edit under every plan, until the
-- project is purged; pruning hashes no current span has is the follow-up.
CREATE TABLE span_vectors (
    project_id   UUID  NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    revision     TEXT  NOT NULL,
    passage_hash BYTEA NOT NULL CHECK (octet_length(passage_hash) = 32),
    encoding     TEXT  NOT NULL,
    vector       BYTEA NOT NULL
                       CHECK (octet_length(vector) > 0 AND octet_length(vector) % 2 = 0),
    PRIMARY KEY (project_id, revision, passage_hash, encoding)
);

-- Granted here rather than left to V20's default privileges, which cover only
-- tables the role that ran V20 creates.
GRANT SELECT, INSERT, UPDATE, DELETE ON span_vectors TO pamin_memory_app;

-- Guarded as V20 and V21 guard what a project knows.
ALTER TABLE span_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE span_vectors FORCE ROW LEVEL SECURITY;
CREATE POLICY one_project ON span_vectors TO pamin_memory_app
    USING (project_id = NULLIF(current_setting('app.project_id', true), '')::uuid)
    WITH CHECK (project_id = NULLIF(current_setting('app.project_id', true), '')::uuid);
