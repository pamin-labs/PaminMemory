-- Two corrections that only get cheaper the earlier they land.
--
-- The first is the shard key. Every table carries project_id and is meant to
-- be sharded on it, but four uniqueness constraints were written without it.
-- They enforce exactly the same thing with the prefix added -- topic_id
-- already determines project_id through the topics foreign key -- so this
-- costs nothing today and is a rewrite of four large tables later. The
-- exception is index_jobs, where the prefix is not merely equivalent but
-- better: a globally unique idempotency key across tenants is precisely the
-- cross-shard coordination point the shard rule exists to prevent.
--
-- The second is a set of indexes for queries that already exist. PostgreSQL
-- does not index foreign keys automatically, so several joins and every
-- project-scoped scan had nothing to use.

ALTER TABLE topic_states
    DROP CONSTRAINT topic_states_topic_id_version_key,
    ADD CONSTRAINT topic_states_project_topic_version_key
        UNIQUE (project_id, topic_id, version);

ALTER TABLE source_versions
    DROP CONSTRAINT source_versions_source_id_version_key,
    ADD CONSTRAINT source_versions_project_source_version_key
        UNIQUE (project_id, source_id, version);

ALTER TABLE relationship_versions
    DROP CONSTRAINT relationship_versions_relationship_id_version_key,
    ADD CONSTRAINT relationship_versions_project_relationship_version_key
        UNIQUE (project_id, relationship_id, version);

ALTER TABLE index_jobs
    DROP CONSTRAINT index_jobs_idempotency_key_key,
    ADD CONSTRAINT index_jobs_project_idempotency_key_key
        UNIQUE (project_id, idempotency_key);

-- Two of the indexes this migration was going to add are already above.
-- A UNIQUE constraint is backed by a btree over exactly its columns, so
-- (project_id, topic_id, version) on topic_states and
-- (project_id, relationship_id, version) on relationship_versions arrive with
-- the constraints. Scanning a project's live states and reading an edge's
-- history are both served by them; naming them again would have bought two
-- duplicate btrees maintained on every write forever.

-- grep sorts by recorded_at before it limits, so without this the whole
-- project's evidence is sorted to return twenty rows.
CREATE INDEX source_versions_by_recency
    ON source_versions (project_id, recorded_at DESC, id);

-- An unindexed foreign key. Deleting a source cascades through source_spans
-- into topic_states, and every step of that walk was a sequential scan.
CREATE INDEX topic_states_by_span
    ON topic_states (source_span_id);

-- Closing the edges a soft-deleted state derived needs to find them by cause.
CREATE INDEX relationship_versions_by_cause
    ON relationship_versions (project_id, caused_by_topic_state);
