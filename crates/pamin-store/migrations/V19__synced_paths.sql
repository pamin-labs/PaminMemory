-- The paths `pamin ingest` keeps in sync: every path an ingest was given,
-- unless it was told `--no-sync`. The server watches each one and ingests what
-- changes under it, and catches up on what changed while it was not running.
CREATE TABLE synced_paths (
    project_id   UUID        NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    -- Canonical, as the ingest resolved it: a file or a directory.
    path         TEXT        NOT NULL,
    -- The vector index the ingest asked for, for an ingest nobody asks for.
    vector_index TEXT        NOT NULL,
    added_at     TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (project_id, path)
);
