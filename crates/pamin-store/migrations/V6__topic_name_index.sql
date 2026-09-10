-- An inverted index from a topic's name, as the segmenter tokenizes it, to the
-- topic.
--
-- Deriving the edges a memory implies means asking "does this text name that
-- topic?" of every topic in the project, and the answer for a given topic never
-- changes unless the topic is renamed. Asking it by loading every topic makes
-- the cost of one write follow the size of the project; asking it here makes it
-- follow the length of what was written.
--
-- The key is the whole name as a space-joined token sequence rather than one
-- row per token, because a name matches only as a contiguous run. Storing
-- tokens separately would turn an exact question into a candidate set that then
-- has to be re-checked.
CREATE TABLE topic_name_tokens (
    project_id  UUID     NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    topic_id    UUID     NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    -- The topic's name after the same normalisation the matcher applies:
    -- separators opened up, segmented, lowercased, joined by single spaces.
    name_key    TEXT     NOT NULL,
    -- How many tokens that is. The lookup takes every window of every width up
    -- to the widest name in the project, so the widest name is what bounds the
    -- work.
    token_count SMALLINT NOT NULL,
    PRIMARY KEY (project_id, topic_id)
);

-- The lookup itself.
CREATE INDEX topic_name_tokens_by_key ON topic_name_tokens (project_id, name_key);

-- Answers "how wide is the widest name here" without reading the table.
CREATE INDEX topic_name_tokens_by_width ON topic_name_tokens (project_id, token_count);
