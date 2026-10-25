# Qasper: heading sections against sized sections, 2026-10-25

The harness behind
[`results/ingest/qasper-sections-2026-10-25.md`](../../results/ingest/qasper-sections-2026-10-25.md).
It produces numbers and asserts its own premises only, so it is kept here as
appended blocks rather than as a test in the tree, following `pamin-dev`.

- `dump.rs.append` -- appended to `crates/pamin-engine/tests/sectioning.rs` at
  `a2b6fb3`, the last commit whose `sections` sized a section to the reranker's
  window. Writes that cut of all 416 test-split papers to JSON, so a later
  commit can index it as an arm.
- `qasper.rs.append` -- appended to the same file at `cf6ebe0`. Indexes three
  arms -- `whole`, `sized` (the JSON) and `headings` (`cf6ebe0`'s `sections`) --
  into a fresh project each, through `Engine::remember_section`, and asks every
  question through `Engine::search_reranked` at the product's depths. Rounds
  rotate the arm order.
- `summarise.py` -- per-paper paired table with a bootstrap CI, from the rows.
  The p values published are the harness's own (`statistics::compare`).

Each arm asserts:

- the project is fresh: every memory it records is written;
- the index holds exactly the spans `Engine::cut` predicts for the arm;
- one new index directory appeared, which is the one measured;
- every question returns hits.

The run asserts that the sized dump carries size parts and the heading cut
carries none, and prints the largest per-question drift across rounds.

## Commands

```sh
REPO=$PWD
H=$REPO/benchmarks/harnesses/qasper-sections-2026-10-25
export PAMIN_EVAL_HOME=/path/to/eval   # datasets, models, projects

# 1. The sized cut, at a2b6fb3.
git worktree add --detach /tmp/sized a2b6fb3
cat /tmp/sized/crates/pamin-engine/tests/sectioning.rs $H/dump.rs.append \
  > /tmp/sized/crates/pamin-engine/tests/scratch_dump.rs
(cd /tmp/sized && QASPER_PAPERS=416 SIZED_OUT=$PAMIN_EVAL_HOME/sized-a2b6fb3.json \
  cargo test -p pamin-engine --test scratch_dump -- --ignored --exact scratch_dump_sized)

# 2. The three arms, at cf6ebe0, release build, three rounds.
git checkout cf6ebe0
rm -f crates/*/tests/scratch_*.rs
cat crates/pamin-engine/tests/sectioning.rs $H/qasper.rs.append \
  > crates/pamin-engine/tests/scratch_qasper.rs
SIZED_IN=$PAMIN_EVAL_HOME/sized-a2b6fb3.json SCRATCH_ROUNDS=3 SCRATCH_OUT=rows.json \
  cargo test --release -p pamin-engine --test scratch_qasper -- \
  --ignored --exact scratch_qasper_arms --nocapture
rm -f crates/*/tests/scratch_*.rs

# 3. The table.
python3 $H/summarise.py rows.json
```

In the container this ran in, the bundled PostgreSQL could not be installed
(the GitHub releases API was unreachable) and `initdb` refuses root. The
published build of the same 17.6.0 was supplied with
`PAMIN_EVAL_POSTGRES_DIR` and the harness run as an unprivileged user.
