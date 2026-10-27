-- What each file `pamin ingest` read looked like when it read it, so that the
-- next ingest of it -- a re-run, a server catching up after being stopped, a
-- file watcher -- can tell an unchanged file by its metadata alone and leave
-- it unread, the way git's index tells a clean file from a stat.
--
-- A record says that the sections of `file` under `root` say what the file
-- said when it was read. Anything that makes that untrue deletes the record:
-- another tree's ingest of a file of the same name rewriting those sections,
-- or a manual write to one of them. A file with no record is read.
--
-- The numbers are a stat's, and only ever compared for equality: `inode` is a
-- u64 and the times are nanoseconds since the epoch, each stored in a BIGINT's
-- bits.
CREATE TABLE file_records (
    project_id   UUID    NOT NULL REFERENCES projects (id) ON DELETE CASCADE,
    -- The directory the file is named from: a section's `sources.root`.
    root         TEXT    NOT NULL,
    -- The file's name, which every name of its sections starts with.
    file         TEXT    NOT NULL,
    size         BIGINT  NOT NULL,
    modified_ns  BIGINT  NOT NULL,
    -- The inode's change time, which a write that preserves the modification
    -- time (`touch -m`, `rsync -t`) still moves.
    changed_ns   BIGINT  NOT NULL,
    inode        BIGINT  NOT NULL,
    -- When the stat above was taken, before the file was read. A file modified
    -- within a timestamp's granularity of it may have changed again unseen, so
    -- its record is not trusted on its stat alone.
    read_ns      BIGINT  NOT NULL,
    -- The text that was read, hashed as evidence is: a file whose stat moved
    -- and whose text did not is recognized without being cut again.
    content_hash TEXT    NOT NULL,
    -- How many sections it was cut into, which an ingest that leaves it unread
    -- still counts.
    sections     INTEGER NOT NULL,
    PRIMARY KEY (project_id, root, file)
);

-- The sections of a file, or of every file under a directory, found by their
-- locator's prefix. In the "C" collation a prefix is a range of the index; in
-- the database's own collation it need not be.
CREATE INDEX sources_file_locators
    ON sources (project_id, (locator COLLATE "C"))
    WHERE kind = 'file';
