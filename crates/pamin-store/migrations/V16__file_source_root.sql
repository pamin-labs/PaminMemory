-- The directory a file's sections were last ingested from: the root of the
-- repository the file is in, or the directory named. A section's name starts
-- with that root's own name rather than its path, so two trees of the same
-- name -- two clones, two `notes` directories -- name their files alike, and
-- an ingest of one must not take the other's sections off the retrieval
-- surface as missing. Null for every source that is not a file's.
ALTER TABLE sources ADD COLUMN root TEXT;
