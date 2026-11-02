-- A graph walk reads a topic's strongest live edges without reading the rest.
--
-- The walk ranks what it reaches by the confidence of the edge it crossed and
-- keeps the head, but it read a hub's edges to the end to find that head: a
-- five-thousand-degree topic was five thousand rows fetched, mapped twice and
-- sorted, for fifty kept. Reading only the head needs an index that holds a
-- topic's live edges in that order -- endpoint, then confidence, then the
-- topic on the other end -- and no table held all three. The endpoints are on
-- `relationships` and the confidence is on the version, because a version is
-- what changes.
--
-- So each version carries its identity's endpoints and kind as well. They are
-- copies of columns that never change -- an identity is one (pair, kind) for
-- ever, and a changed claim is a new version of the same identity -- so there
-- is nothing to keep in step: they are written once, with the version, from
-- the row the version belongs to.
--
-- `kind` comes too so that a walk restricted to some kinds is answered from
-- the same rows. The indexes are partial on the live predicate the walk
-- states, which keeps closed history out of them.
ALTER TABLE relationship_versions
    ADD COLUMN from_topic UUID,
    ADD COLUMN to_topic   UUID,
    ADD COLUMN kind       TEXT;

UPDATE relationship_versions v
   SET from_topic = r.from_topic,
       to_topic   = r.to_topic,
       kind       = r.kind
  FROM relationships r
 WHERE r.id = v.relationship_id;

ALTER TABLE relationship_versions
    ALTER COLUMN from_topic SET NOT NULL,
    ALTER COLUMN to_topic   SET NOT NULL,
    ALTER COLUMN kind       SET NOT NULL;

CREATE INDEX relationship_versions_live_from
    ON relationship_versions (project_id, from_topic, confidence DESC, to_topic)
    WHERE invalidated_at IS NULL;

CREATE INDEX relationship_versions_live_to
    ON relationship_versions (project_id, to_topic, confidence DESC, from_topic)
    WHERE invalidated_at IS NULL;
