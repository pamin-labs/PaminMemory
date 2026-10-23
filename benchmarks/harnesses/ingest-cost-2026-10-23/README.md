# `pamin ingest` cost, 2026-10-23

Two bash harnesses that drive the release `pamin` client and the resident
`pamin-server` it talks to — the product entry point — over a copy of this
repository's own tracked `.md`, `.markdown` and `.txt` files. They produce
numbers only and assert nothing, so they live here rather than as tests; the
invariants they rely on (an unchanged section writes nothing, a section's name
stays out of the name index) are asserted by
`crates/pamin-engine/tests/ingest.rs` and the last test in
`crates/pamin-cli/tests/end_to_end.rs`.

They run as root and run `pamin` as an unprivileged user, `tester` in the
scripts, because PostgreSQL refuses to run as root: root creates each
workspace under `/home/tester`, hands it to that user, and reads the server's
`/proc` entries and the database. PostgreSQL is an installed PostgreSQL 16,
named by `postgres_dir` in each workspace's `config.toml`, and `HF_HOME` holds
the models already downloaded. Every arm reads the server process's CPU time
from `/proc` before and after: **PostgreSQL's backends are separate processes
and are not included**, nor is the client, while any index upkeep the server
runs during the arm is. `server_hwm_mb` is the server's `VmHWM`, its peak
resident set over its whole life so far, so it can only rise across arms.
`db_mb` is `pg_database_size` in MiB; `index_mb` is the workspace's `index/`
directory as `du` allocates it, in MiB. Neither includes the model cache.

- `measure_ingest.sh.in <bin dir> <corpus> <out.csv> [repeats]` — one fresh
  workspace: a cold ingest, then `repeats` rounds of an unchanged re-ingest and
  a one-word edit to the first line of prose of a different file, then one
  `##` heading of the README renamed.
- `measure_import.sh.in <bin dir> <corpus> <out.csv>` — a cold ingest and an
  unchanged re-ingest in one fresh workspace, then `pamin import` of exactly
  the sections that ingest recorded, exported from its database, and the same
  import again, in a second fresh workspace.

Freeze the binaries and their native libraries into one directory before
running, and build nothing while an arm runs.

```sh
cargo build --release -p pamin-cli
mkdir -p "$BIN" && cp target/release/pamin target/release/pamin-server target/release/*.so* "$BIN"/
git ls-files -- '*.md' '*.markdown' '*.txt' | grep -v '^internal-docs\|^\.' \
  | while read -r f; do mkdir -p "$CORPUS/$(dirname "$f")"; cp "$f" "$CORPUS/$f"; done
mkdir -p "$CORPUS/.git"   # so the copy is a repository root of its own
bash measure_ingest.sh.in "$BIN" "$CORPUS" ingest-arms.csv 5
bash measure_import.sh.in "$BIN" "$CORPUS" ingest-vs-import.csv
```

Results and their conditions are in
[`benchmarks/results/ingest/`](../../results/ingest/ingest-cost-2026-10-23.md).
