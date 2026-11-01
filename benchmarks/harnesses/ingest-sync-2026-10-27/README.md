# File sync cost, 2026-10-27

`measure.sh.in` drives the release `pamin` client and the resident
`pamin-server` it talks to -- the product entry point -- for two builds side by
side: `base`, the top of the ingest stack this work builds on (`0539965b`), and
`head`, this branch (`9505cf8`). It produces numbers only and asserts nothing;
the invariants it relies on (an unchanged file is left unread, an edit reaches
search with no ingest) are asserted by the ignored end-to-end tests in
`crates/pamin-cli/tests/end_to_end.rs`.

It runs as root and runs `pamin` as the unprivileged user `pamintest`, because
PostgreSQL refuses to run as root. Each build gets its own workspace and its
own copy of the corpus, PostgreSQL 17.6 named by `postgres_dir` in each
workspace's `config.toml`, and a copy of an already downloaded model cache.
Both servers stay up for the whole run, and a warm-up search loads each one's
models before any arm.

- **Ingest arms**, after one cold ingest each: an unchanged re-ingest, then a
  one-word edit to the first line of prose of a different file, then a
  re-ingest, for `rounds` rounds with the build order alternated each round.
  Each arm waits 3 s first, so `head` measures a file whose record is trusted
  rather than one read within the 2 s racy window. `head` ingests with
  `--no-sync`, so its watcher does not ingest an edit before the arm does.
  Per arm: client wall time, and the server process's user+system CPU from
  `/proc` -- **PostgreSQL's backends are separate processes and not included**.
- **Search arms**: the 20 queries in `queries.txt`, client wall time each,
  three rounds of `base`, `head` with its corpus kept in sync (`synced`, the
  hit re-check runs) and `head` with it unsynced (`unsynced`, the re-check
  returns after one query).

Freeze the binaries and their native libraries into one directory per build
before running, and build nothing while an arm runs.

```sh
git ls-files -- '*.md' '*.markdown' '*.txt' | grep -v '^internal-docs\|^\.' \
  | while read -r f; do mkdir -p "$CORPUS/$(dirname "$f")"; cp "$f" "$CORPUS/$f"; done
mkdir -p "$CORPUS/.git"
bash measure.sh.in "$BASE_BIN" "$HEAD_BIN" "$CORPUS" "$OUT" 5
```

Results and their conditions are in
[`benchmarks/results/ingest/`](../../results/ingest/ingest-sync-2026-10-27.md).
