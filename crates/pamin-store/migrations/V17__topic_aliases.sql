-- The names a topic went by before it was renamed.
--
-- `pamin ingest` names a section by its file's path and the headings above it,
-- so moving a file or renaming a heading changes the name and not the section.
-- Recorded as a new topic, the section would lose its history and every edge
-- to it; renamed instead, it keeps them, and the name it had is kept here so
-- that a caller who learned it -- a link, a note, an agent's memory of an
-- earlier answer -- still reaches the topic. A name resolves here only when
-- no topic holds it: a new topic created under an old name takes it back.
CREATE TABLE topic_aliases (
    project_id UUID        NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    name       TEXT        NOT NULL,
    topic_id   UUID        NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    -- When the topic stopped going by this name: the record of the move, which
    -- the evidence log, keyed by content, does not show.
    renamed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (project_id, name)
);

CREATE INDEX topic_aliases_by_topic ON topic_aliases (topic_id);
