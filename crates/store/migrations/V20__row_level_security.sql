-- Row-level security on project_id: a second guard behind the filter every
-- query already applies, so a bug that names the wrong project, or none, reads
-- and writes nothing of another.
--
-- The guard holds for one role, pamin_memory_app, which can neither log in nor
-- bypass row-level security. A per-project transaction sets it locally along
-- with the project it works on (`Database::scoped`), whoever the connection
-- belongs to, so the same path is guarded locally, where the connection is a
-- superuser, and hosted. Outside such a transaction nothing changes: the
-- migrations, the resident server's scan of which projects owe work, and the
-- store's own tests run as the connection's role.
--
-- The tables that hold what a project knows are guarded. Three are not:
-- `projects` (names and ids, looked up by name before a project is known),
-- `index_jobs` (the queue, ids and kinds only, scanned across projects to
-- schedule work) and `synced_paths` (local file paths, listed across projects
-- by the watcher).

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'pamin_memory_app') THEN
        CREATE ROLE pamin_memory_app NOLOGIN NOSUPERUSER NOBYPASSRLS;
    END IF;
END
$$;

-- Whoever runs the migrations is who the application connects as, and it has
-- to be able to take this role on.
GRANT pamin_memory_app TO CURRENT_USER;

DO $$
BEGIN
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO pamin_memory_app', current_schema());
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA %I TO pamin_memory_app',
        current_schema()
    );
    EXECUTE format(
        'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA %I TO pamin_memory_app',
        current_schema()
    );
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES IN SCHEMA %I
             GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO pamin_memory_app',
        current_schema()
    );
END
$$;

-- `app.project_id` reads as NULL before it is first set and as '' after a
-- transaction that set it ends; either way the role sees no rows.
DO $$
DECLARE
    guarded TEXT;
BEGIN
    FOREACH guarded IN ARRAY ARRAY[
        'sources', 'source_versions', 'source_spans', 'topics', 'topic_states',
        'relationships', 'relationship_versions', 'topic_aliases',
        'topic_name_tokens', 'file_records'
    ] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', guarded);
        EXECUTE format(
            'CREATE POLICY one_project ON %I TO pamin_memory_app
                 USING (project_id = NULLIF(current_setting(''app.project_id'', true), '''')::uuid)
                 WITH CHECK (project_id = NULLIF(current_setting(''app.project_id'', true), '''')::uuid)',
            guarded
        );
    END LOOP;
END
$$;
