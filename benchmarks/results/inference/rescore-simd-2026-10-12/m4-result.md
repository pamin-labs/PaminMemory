# Apple M4 rescore screen: per-run figures

These figures were transcribed from the M4 operator's report. The raw files
stayed on that machine: `REPORT.md`, `step1-conditions.txt`, and
`rescore-run1.txt` to `rescore-run3.txt`. The three runs finished back to back
between 20:28:19 and 20:28:24 local time, each in about 2 seconds, on AC power,
with no thermal warning before or after any run.

| Run | 1-minute load, before → after |
| --- | --- |
| 1 | 2.59 → 2.59 |
| 2 | 2.59 → 2.55 |
| 3 | 2.55 → 2.55 |

Each cell is the pooled p50 / p95 in µs for one run, with n = 6000 per arm per
run.

| Arm | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| `mac-main` | 43.33 / 47.08 | 39.71 / 50.38 | 40.38 / 46.33 |
| `mac-pr` | 22.42 / 24.17 | 21.75 / 25.58 | 22.50 / 23.92 |
| `half-neon` | 21.50 / 22.62 | 19.38 / 23.92 | 21.46 / 22.38 |
| `half-seq` | 103.46 / 119.00 | 102.67 / 120.00 | 103.42 / 117.12 |
| `dot-accelerate` | 6.08 / 6.75 | 5.92 / 7.42 | 6.00 / 6.75 |
| `dot-neon` | 5.29 / 5.88 | 5.21 / 6.75 | 5.25 / 5.92 |
| `dot-seq` | 42.46 / 48.04 | 42.46 / 52.96 | 42.46 / 47.79 |

Every arm passed the premise: the maximum |cos − oracle| was at most 7.749e-8,
and every arm's top 10 matched the oracle's 10/10.

The harness is kept out of this tree. It is in
`pamin-memory/research/2026-10-12/raw/rescore-simd-m4/` in the
`internal-docs` submodule.
