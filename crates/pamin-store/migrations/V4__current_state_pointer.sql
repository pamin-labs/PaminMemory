-- Which state a topic currently resolves to, stored on the topic.
--
-- V1 argued against storing this, and against the shape it had in mind it was
-- right: a boolean flag on topic_states needs every write path to clear the old
-- row and set the new one, and one missed path leaves two rows both claiming to
-- be current. A partial unique index is the usual patch, and it is a
-- single-engine construct this schema does not want.
--
-- A pointer on the parent has neither problem. One column on one row cannot
-- disagree with itself, so there is nothing for a constraint to enforce and
-- nothing to go half-applied. Maintenance is already paid for: both writers --
-- appending a state and soft deleting one -- run in a transaction that holds a
-- lock on the topic row.
--
-- What it buys is that resolving a topic to its content stops being a scan.
-- `topic_states_latest` answers it for one topic; the search path asks it for
-- every candidate at once, and had no way to do that but to read the project.
--
-- Nullable: a topic whose every state has been soft deleted resolves to
-- nothing, and so does one that exists before its first state is appended.

ALTER TABLE topics
    ADD COLUMN current_state_id UUID REFERENCES topic_states (id),
    ADD COLUMN current_version  INTEGER;

-- Backfill from what the ledger already says, so this migration is complete
-- rather than a starting point. The two columns move together from here on.
UPDATE topics t
   SET current_state_id = latest.id,
       current_version  = latest.version
  FROM (
      SELECT DISTINCT ON (topic_id) topic_id, id, version
      FROM topic_states
      WHERE deleted_at IS NULL
      ORDER BY topic_id, version DESC
  ) AS latest
 WHERE latest.topic_id = t.id;

-- Both or neither. A pointer without a version, or a version pointing nowhere,
-- is the disagreement this design exists to make impossible -- so it is stated
-- rather than assumed.
ALTER TABLE topics
    ADD CONSTRAINT topics_current_state_is_whole
        CHECK ((current_state_id IS NULL) = (current_version IS NULL));
