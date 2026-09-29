# Full XQuAD-R search: main CPU control and accelerated stack

The actual `Engine::search_reranked` entry completed 1,190 queries per arm over
13,014 sentences in 11 languages, on the same complete revision-bound memory
index. The machine was shared (Mac16,12, 10 logical CPUs, 32 GiB RAM).

This compares main `a249554` forced to CPU with the frozen `0a222d1` accelerated
runtime. Its restacked runtime tree is identical; the harness logged the
AGENTS-only `c9ab024` checkout. **This is not yet a comparison with main's default
`auto` path**, which is being measured separately.

Both arms used the accuracy profile, accurate tier, head depth 30 and result
limit 60. The embedder remained CPU int8; the candidate reranker used the
FP16 encoder/FP32 head on CoreML ALL. Runtime logs confirmed `coreml`; exact
per-operation CPU/GPU/ANE placement was not captured for this full run.

## Observations

| Metric | Main CPU | Accelerated stack | Absolute difference | Percentage change |
|---|---:|---:|---:|---:|
| Cross-language nDCG@10 | 0.726808 | 0.727194 | +0.000386 | +0.05% |
| Same-language nDCG@10 | 0.868201 | 0.869446 | +0.001245 | +0.14% |
| Cross-language recall@50 | 0.903193 | 0.903193 | 0 | 0% |
| Same-language recall@50 | 0.964706 | 0.964706 | 0 | 0% |
| Whole-search p50 | 2.243566 s | 0.483738 s | -1.759828 s | -78.44% |
| Whole-search p95 | 4.343058 s | 0.814224 s | -3.528834 s | -81.25% |
| Wall time summed over 1,190 search calls | 2,976.234 s | 723.069 s | -2,253.165 s | -75.71% |
| Sampled whole-harness peak process RSS | 1,500,790,784 B | 3,068,149,760 B | +1,567,358,976 B | +104.44% |
| Benchmark-process CPU user/system time | Not captured | 316.351 / 86.276 s | N/A | N/A |
| Three-bucket compiled cache, logical bytes | No native bucket cache | 6,836,411,689 B | +6,836,411,689 B | N/A; zero baseline |
| Total isolated models/index/temporary disk | Not measured | Not measured | N/A | N/A |

The candidate's entire accurate-plus-off harness took 777.250 s; its process
CPU total was 402.627 s, averaging 0.518 CPU cores. CPU time excludes CoreML
services and device execution and cannot establish total system energy/cost.
The shared reused index was 83,252,390 logical bytes; it was held constant,
not rebuilt as an optimization. Its PostgreSQL/WAL footprint was not measured.

## Precision and timing interpretation

Recall is unchanged on every query. nDCG changes are not statistically
significant: paragraph-cluster sign flips across the four predefined metrics
give adjusted p=0.8537 cross-language and p=0.5782 same-language. Cross-language
wins/losses/ties are 214/210/766; same-language 32/20/1,138. This does not prove
identical rankings or zero precision drift.

These are **single, sequential shared-host timing observations**, including
first-query model and shape setup. The large observed latency reduction is a
completed product-path result, but repeated rotated arms are still needed for
a causal, reproducible speed estimate. It is the combined stack's observation;
it is not the isolated contribution of the cache-recovery fix or short bucket.
RSS includes compilation and model loading and excludes accelerator service
memory; it is not steady model-only residency.

All 64/128/256 bucket completion markers were published and the full candidate
exited zero without native or resource rejection. The compiled cache was
inventoried and then retired to recover headroom for the main default-auto
control, preserving original ONNX files and all results. Logical cache bytes
are not unique APFS allocated bytes and retirement is not a product disk gain.

## Evidence and reproduction

[Summary and provenance](coreml-search-full-2026-09-30.json),
[main rankings/timings](coreml-search-full-2026-09-30-main-rows.json),
[candidate rankings/timings](coreml-search-full-2026-09-30-candidate-rows.json),
and [judgments](coreml-search-full-2026-09-30-judgments.json) retain all queries.
The canonical source is `crates/pamin-engine/tests/crosslingual.rs`; the
measurement-only wrapper retained its corpus and scoring helpers and called
the real product entry, adding per-query timers and row serialization.
Offline scoring and paragraph-cluster tests reuse the repository's scoring and
statistics owners. Raw result hashes are in the summary.

Remaining acceptance: main default-auto comparison, rotated timing repeats,
other corpora, service/steady memory scope, isolated total disk and the full
ignored/e2e suite. Keep runtime adoption/merge readiness separate from this
single-corpus result.
