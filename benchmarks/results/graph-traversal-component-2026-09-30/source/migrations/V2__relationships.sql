-- The relationship graph: stable edge identities and immutable edge facts.
--
-- The graph lives here rather than in the projection index, and that is the
-- reason fusion has to happen in our own layer: the index cannot see these
-- rows, so it cannot fuse the channel they feed.
--
-- Portability rules from V1 continue to hold. Every table carries project_id,
-- primary keys are application-generated UUIDs, and nothing here needs an
-- extension.

-- Stable logical edge identity.
--
-- Endpoints are topic identities rather than topic states. An edge asserted
-- between two topics survives both of them changing; resolving each endpoint to
-- a version is retrieval's job, not the edge's.
CREATE TABLE relationships (
    id         UUID        PRIMARY KEY,
    project_id UUID        NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    from_topic UUID        NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    to_topic   UUID        NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    kind       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    -- One identity per (pair, kind). Two topics can be related several ways at
    -- once, and each way has its own history.
    UNIQUE (project_id, from_topic, to_topic, kind),
    CONSTRAINT relationships_kind_known CHECK (kind IN (
        'mentions', 'supports', 'contradicts', 'supersedes', 'related_to',
        'part_of', 'derived_from', 'same_as', 'depends_on'
    )),
    -- A topic related to itself carries no information and would make every
    -- traversal revisit its own start.
    CONSTRAINT relationships_no_self_edge CHECK (from_topic <> to_topic)
);

CREATE INDEX relationships_by_from ON relationships (project_id, from_topic);
CREATE INDEX relationships_by_to   ON relationships (project_id, to_topic);

-- Immutable edge facts.
--
-- Changing an edge closes the current version and appends a new one. Nothing is
-- overwritten, so "when did we start believing this, and what did we believe
-- before" stays answerable.
CREATE TABLE relationship_versions (
    id                    UUID        PRIMARY KEY,
    project_id            UUID        NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    relationship_id       UUID        NOT NULL REFERENCES relationships (id) ON DELETE CASCADE,
    version               INTEGER     NOT NULL,
    -- Truth validity: when the edge is asserted to hold. Independent of the
    -- endpoints' own validity, because a relationship can outlive or predate
    -- the versions of the topics it connects.
    valid_from            TIMESTAMPTZ,
    valid_to              TIMESTAMPTZ,
    -- System validity: when we recorded the claim, and when we stopped
    -- standing behind it.
    created_at            TIMESTAMPTZ NOT NULL,
    invalidated_at        TIMESTAMPTZ,
    supersedes            UUID        REFERENCES relationship_versions (id),
    -- Provenance. NULL for an edge a caller asserted directly, since no topic
    -- state caused it.
    caused_by_topic_state UUID        REFERENCES topic_states (id),
    -- Orders same-hop neighbours during graph recall. Without it, neighbours at
    -- equal distance would be returned in arbitrary order.
    confidence            REAL        NOT NULL,
    derivation            TEXT        NOT NULL,
    -- Why the edge was closed, set together with invalidated_at.
    tombstone_reason      TEXT,
    UNIQUE (relationship_id, version),
    CONSTRAINT relationship_versions_derivation_known CHECK (derivation IN (
        'explicit', 'deterministic', 'model', 'imported'
    )),
    CONSTRAINT relationship_versions_tombstone_reason_known CHECK (
        tombstone_reason IS NULL OR tombstone_reason IN (
            'closed', 'superseded', 'deleted'
        )
    ),
    -- A closed version says why, and an open one has nothing to say.
    CONSTRAINT relationship_versions_tombstone_matches_invalidation CHECK (
        (invalidated_at IS NULL) = (tombstone_reason IS NULL)
    ),
    CONSTRAINT relationship_versions_confidence_in_range CHECK (
        confidence > 0 AND confidence <= 1
    ),
    CONSTRAINT relationship_versions_interval_ordered CHECK (
        valid_to IS NULL OR valid_from IS NULL OR valid_to > valid_from
    )
);

-- The live version is resolved from this index rather than stored in a flag,
-- for the same reason as topic_states_latest: a flag needs transactional
-- maintenance on every write, and one missed path leaves two rows claiming to
-- be current.
CREATE INDEX relationship_versions_live
    ON relationship_versions (relationship_id, version DESC)
    WHERE invalidated_at IS NULL;
