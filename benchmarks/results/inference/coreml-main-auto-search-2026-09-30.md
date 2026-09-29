# Full default-main auto comparison

Both arms complete all 1,190 XQuAD-R queries at actual `Engine::search_reranked`
with the same complete 13,014-document revision-bound memory index, accuracy
profile, accurate tier, head depth 30 and returned limit 60. The host was shared
Mac16,12 (10 logical CPUs, 32 GiB RAM).

Main `a249554` requested `PAMIN_DEVICE=auto` and logged `coreml`. It uses the
legacy NeuralNetwork provider defaults. The frozen candidate `0a222d1` uses
MLProgram ALL with native standard operators, FP16 encoder/FP32 classifier and
static 64/128/256 buckets; its restacked runtime tree is identical. Neither arm
captured exact per-operation CoreML internal CPU/GPU/ANE placement. Embedding
remains the same CPU int8 model and the index did not change.

| Metric | Main default auto | Accelerated stack | Absolute difference | Percentage change |
|---|---:|---:|---:|---:|
| Cross-language nDCG@10 | 0.727362 | 0.727194 | -0.000168 | -0.02% |
| Same-language nDCG@10 | 0.869148 | 0.869446 | +0.000298 | +0.03% |
| Cross-language recall@50 | 0.903193 | 0.903193 | 0 | 0% |
| Same-language recall@50 | 0.964706 | 0.964706 | 0 | 0% |
| Whole-search p50 | 5.085653 s | 0.483738 s | -4.601915 s | -90.49% |
| Whole-search p95 | 7.374466 s | 0.814224 s | -6.560242 s | -88.96% |
| 1,190 accurate search-call wall total | 6,177.679 s | 723.069 s | -5,454.610 s | -88.30% |
| Accurate+off harness wall total | 6,232.904 s | 777.250 s | -5,455.655 s | -87.53% |
| Harness process CPU user time | 24,491.185 s | 316.351 s | -24,174.834 s | -98.71% |
| Harness process CPU system time | 1,588.832 s | 86.276 s | -1,502.556 s | -94.57% |
| Harness process CPU total | 26,080.018 s | 402.627 s | -25,677.390 s | -98.46% |
| Sampled peak process RSS | 4,436,525,056 B | 3,068,149,760 B | -1,368,375,296 B | -30.84% |
| Candidate three-bucket persistent cache logical bytes | No equivalent native bucket cache | 6,836,411,689 B | +6,836,411,689 B | N/A |
| Total isolated model/index/temp disk | Not measured | Not measured | N/A | N/A |

The observed p50 ratio is 10.51×, p95 9.06× and search-call wall total 8.54×.
These are **one sequential shared-host pass per arm**, not yet repeated rotated
speed acceptance. Initial model and shape setup is included. The main run
included one recorded one-second CPU stack sample; future timing repeats must
exclude that diagnostic. CPU totals describe the benchmark process/threads,
not CoreML services, GPU/ANE time or total system energy. Mean occupied CPU
cores were 4.184 versus 0.518 over each whole harness.

Recall is identical per query. Cross-language nDCG has 13 wins/23 losses/1,154
ties, mean -0.000168, four-metric paragraph-cluster adjusted p=0.1532.
Same-language has 1 win/1 loss/1,188 ties, mean +0.000298, adjusted p=1.0.
Neither difference is significant. This is not a claim of identical rankings,
zero precision drift or statistical equivalence. Both product accuracy gates
and reranker-contribution assertions pass. Other corpora remain necessary.

The lower process peak is relative to **default-main auto**. In the separate
[main CPU control](coreml-search-full-2026-09-30.md), the candidate peak is
higher than CPU: 3.068 GB versus 1.501 GB. Neither is steady model-only memory;
startup/compilation is included and accelerator services are excluded.
The persistent candidate cache is logical file size, not unique APFS allocation;
it was inventoried and retired for main-auto headroom after the complete run.
That housekeeping is not a product disk optimization.

## Evidence

[Summary, phases, deltas and precision tests](coreml-main-auto-search-2026-09-30.json),
[all main auto rankings/timings](coreml-main-auto-search-2026-09-30-rows.json),
[candidate rankings/timings](coreml-search-full-2026-09-30-candidate-rows.json),
and [judgments](coreml-search-full-2026-09-30-judgments.json).
Both use canonical corpus/scoring helpers and the actual product entry, with
measurement-only row timers. Offline precision uses the repository's scoring
and paragraph-cluster/family sign-flip owners.

Remaining: clean rotated timing, other corpora, full ignored/e2e suite, steady/
service memory, isolated total disk, shared embedding/NPU integration and native
hardware/platform checks. Overall goal and whole-stack merge readiness are open.
