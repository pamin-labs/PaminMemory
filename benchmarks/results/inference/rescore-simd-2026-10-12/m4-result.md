# Apple M4 rescore screen: per-run figures

The raw output is retained in [`m4-raw/`](m4-raw/). [`SHA256SUMS`](m4-raw/SHA256SUMS)
lists the operator's hashes, and they match the files.

| File | What it is |
| --- | --- |
| [`rescore-run1.txt`](m4-raw/rescore-run1.txt), [`rescore-run2.txt`](m4-raw/rescore-run2.txt), [`rescore-run3.txt`](m4-raw/rescore-run3.txt) | Each run's premises, per-pass figures and pooled figures |
| [`step1-conditions.txt`](m4-raw/step1-conditions.txt) | `uptime`, `pmset -g therm` and the power source before and after each run |
| [`REPORT.md`](m4-raw/REPORT.md) | The operator's report for runbook revision 2 |

Only steps 0 and 1 of `REPORT.md` back this screen. Its step 2, the reranker,
was aborted, and every timing in that step is contaminated by swapping. Later
reranker runs used revisions 3 and 4 and are not part of this pull request.

The table below is read directly from the run files. The per-run table in
`REPORT.md` labels the runs in a different order: its "run 1" column is
`rescore-run3.txt`, its "run 2" is `rescore-run1.txt`, and its "run 3" is
`rescore-run2.txt`. The medians over the three runs do not depend on the order.

The three runs finished back to back between 20:28:19 and 20:28:24 local time,
on AC power, with no thermal warning before or after any run.

| Run | 1-minute load, before → after |
| --- | --- |
| 1 | 2.59 → 2.59 |
| 2 | 2.59 → 2.55 |
| 3 | 2.55 → 2.55 |

Each cell is the pooled p50 / p95 in µs for one run, with n = 6000 per arm per
run.

| Arm | Run 1 | Run 2 | Run 3 | Median over runs |
| --- | ---: | ---: | ---: | ---: |
| `mac-main` | 39.71 / 50.38 | 40.38 / 46.33 | 43.33 / 47.08 | 40.38 / 47.08 |
| `mac-pr` | 21.75 / 25.58 | 22.50 / 23.92 | 22.42 / 24.17 | 22.42 / 24.17 |
| `half-neon` | 19.38 / 23.92 | 21.46 / 22.38 | 21.50 / 22.62 | 21.46 / 22.62 |
| `half-seq` | 102.67 / 120.00 | 103.42 / 117.12 | 103.46 / 119.00 | 103.42 / 119.00 |
| `dot-accelerate` | 5.92 / 7.42 | 6.00 / 6.75 | 6.08 / 6.75 | 6.00 / 6.75 |
| `dot-neon` | 5.21 / 6.75 | 5.25 / 5.92 | 5.29 / 5.88 | 5.25 / 5.92 |
| `dot-seq` | 42.46 / 52.96 | 42.46 / 47.79 | 42.46 / 48.04 | 42.46 / 48.04 |

Every run passed the same premises:

- the `half` conversion equals `decode` for all 65536 codes;
- every arm's top 10 matches the oracle's, 10/10;
- the maximum |cos − oracle| per arm is:

| Arm | Max \|cos − oracle\| |
| --- | ---: |
| `mac-main` | 1.176e-8 |
| `mac-pr` | 1.176e-8 |
| `half-neon` | 1.428e-8 |
| `half-seq` | 7.749e-8 |

The harness is kept out of this tree. It is in
`pamin-memory/research/2026-10-12/raw/rescore-simd-m4/` in the
`internal-docs` submodule.
