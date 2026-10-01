# Exact int8 pooled-vector normalization

This is a standalone 1024-i8 normalization **throughput diagnostic**, not model inference or whole-search latency. The previous six-block single-process result (614.38 → 98.78 ns, 6.22x) remains in `summary.json` as a provisional historical observation. `independent-processes.json` and `process-*.jsonl` supersede it for repeatability.

## Three independent invocations

Shared Apple M4, Mac16,12; Rust 1.98.1 (`48a229cea`, 2026-09-01), `--edition=2024 -O`. Each invocation warms both arms with 1,000 calls, then alternates arm order across six blocks, each timing 100,000 calls per arm. The table takes the median of the three process medians. Baseline code: `034fe12b9562c52d2d8bb3bfcdd014bab7c73e02`; candidate code: `cf1b7b4c8b2c6d511f903b72e33771b2954cd8a6`. The frozen program implements just those normalization kernels, not their enclosing model calls.

| Metric | Before | After | Difference | Change |
|---|---:|---:|---:|---:|
| Mean throughput cost / vector, median process | 601.63 ns | 92.61 ns | -509.02 ns | -84.61% |
| Process-median range | 601.02–602.21 ns | 91.84–95.29 ns | N/A | N/A |
| Per-call p50 / p95 | N/A | N/A | N/A | N/A |
| Vector payload / call | 4,096 B | 4,096 B | 0 B | 0% |
| Product accuracy / search p50/p95 / RSS / disk | N/A | N/A | N/A | N/A |

The ratio is 6.50x **for this kernel throughput**. Aggregate intervals do not retain individual-call latency samples, so p50/p95 cannot be recovered. This diagnostic cannot establish search/model speed or process memory gains. Both outputs are asserted bit-identical in each invocation; ordinary product regressions also cover signed extrema, mixed values and zero rejection. There is no model/index encoding change.

The full squared sum is at most 1024 * 128^2 = 2^24, exactly representable in i32 and f32. The archived compiler output in `normalization-functions.s.txt` shows `addv.4s` in the integer reduction and `fdiv.4s` in both output loops. Old conversion/division already used SIMD.

## Reproduce

The numbers-only program lives in scratch space, not the current tracked tree. Its immutable source (SHA-256 in both summaries) is recoverable from the historical measured revision:

```sh
git fetch origin refs/tags/bench/i8-normalization-source-2026-10-01:refs/tags/bench/i8-normalization-source-2026-10-01
git show bench/i8-normalization-source-2026-10-01:benchmarks/results/compute/i8-normalization-2026-10-01/frozen-program.rs.txt > /tmp/normalization.rs
python3 - /tmp/normalization.rs <<'PY'
import hashlib, sys
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == "12e11819938fcf062382a651dd7f520de68854e38d4857aec78d404c80db4b08"
PY
rustc --edition=2024 -O /tmp/normalization.rs -o /tmp/normalization
rustc --edition=2024 -O --emit=asm /tmp/normalization.rs -o /tmp/normalization.s
for run in 0 1 2; do /tmp/normalization > "/tmp/normalization-$run.jsonl"; done
rg -n 'addv|fdiv' /tmp/normalization.s
```

Run with our own builds/experiments stopped; the host remains shared. Retain all independent rows and their spread rather than choosing the fastest run.
