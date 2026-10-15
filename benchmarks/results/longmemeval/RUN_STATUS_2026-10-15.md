# LongMemEval dual-metric run — progress checkpoint (not a final result)

Run in progress on this branch (`bench/mem0-mempalace-miracl-head-to-head`),
started as part of task #5: dual-metric LongMemEval (retrieval-only to 1880
rows/arm + answer-accuracy to 2500 rows/arm) across `bm25`, `pamin`,
`pamin-wide`, `mem0`, `mempalace-raw`, using Haiku via a local shim.

**This is a checkpoint, not a completed run.** It exists because the run has
been repeatedly stalled by disk exhaustion in its execution container, and
most recently (2026-10-15, ~01:38 UTC) the container's shell tool itself
stopped functioning (`ENOSPC`), so no commands — including the restart
script, `df`, or `summarise.py` — can currently be run to verify or progress
further. Rather than let that state go unrecorded, this captures the last
known checkpoint row counts, taken via the GitHub API (no local shell
available at push time).

## Row counts as of last successful checkpoint write (~01:38 UTC, 2026-10-15)

### Retrieval-only metric (target: 1880 rows per arm)

| arm | rows | % of target |
|---|---|---|
| bm25 | 470 | 25% |
| pamin | 454 | 24% |
| pamin-wide | 69 | 4% |
| mempalace-raw | 5 | 0.3% |
| mem0 | 0 | 0% |

### Answer-accuracy metric (target: 2500 rows per arm)

| arm | rows | % of target |
|---|---|---|
| bm25 | 500 | 20% |
| pamin | 421 | 17% |
| pamin-wide | 0 | 0% |
| mem0 | 1 | 0.04% |
| mempalace-raw | 0 | 0% |

Counts are exact (from matching `"arm": "<name>"` occurrences in the raw
checkpoint files), not estimates.

## Why no accuracy/recall figures are reported here

Per `benchmarks/results/.gitignore`, raw per-row checkpoints
(`longmemeval/retrieval.jsonl`, `longmemeval_qa/quality.jsonl`) are
deliberately not committed — only `summary-*.json` tables, recomputed from
the raw rows by this repo's own tooling, are. With the shell tool
non-functional, that tooling cannot currently be run against this run's raw
rows, and this checkpoint will not hand-compute recall/accuracy aggregates
in its place: a figure this file cannot verify is worse than no figure.
Those aggregate tables will be generated from the raw rows, via the run's
own summarisation path, once the run can resume and either complete or
reach a stable stopping point.

## Known blockers at the time of this checkpoint

- Both `pamin-home` and `pamin-home-2` server instances failed to restart
  after 3 attempts each; their index/WAL writer cannot write
  (`[errno 28] No space left on device`).
- The execution container's shell tool is itself non-functional
  (`ENOSPC` on its own temp filesystem), independent of the above.
- No data has been lost — checkpoints are intact on the container's disk
  pending recovery.
- A likely fix (pruning ~6.4G of pre-split historical data specific to
  `pamin-home`'s index) has been identified but not applied, pending
  the shell tool recovering enough to execute it.
