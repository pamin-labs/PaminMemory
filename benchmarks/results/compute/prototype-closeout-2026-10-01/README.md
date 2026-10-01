# Dual-vector and shared compute prototype evidence — 2026-10-01

## Precision and scope

The product boundary is `Engine::search_reranked`: memory index, named passages, accurate reranker, limit 60, default semantic depth 50 / graph depth 2. The paired CPU-only XQuAD-R comparison has 13,014 documents and the same 1,190 language-rotated queries. Corpus files are fingerprinted in `corpus-files.json`; raw rankings, relevance/exclusions, the predeclared rule, summaries and immutable execution-provenance correction are retained here. No fusion weights were fitted on this corpus.

Baseline CPU executable: `555846c`; dual executable: `45a3bcc601b2f9ca7d23d0bf49c6c95b3284f73e`. The candidate raw manifest sampled a later git HEAD after ingestion; the correction sidecar identifies the frozen executed binary and source. Do not treat that late field as execution identity. A 66-query true-main `315c102` control matched the baseline rankings exactly.

| Metric | BGE | Dual | Difference | Change |
|---|---:|---:|---:|---:|
| Cross-language nDCG@10 | 0.726706 | 0.734442 | +0.007736 | +1.06% |
| Cross-language recall@50 | 0.903193 | 0.883109 | -0.020084 | -2.22% |
| Same-language nDCG@10 | 0.868201 | 0.872163 | +0.003962 | +0.46% |
| Same-language recall@50 | 0.964706 | 0.966387 | +0.001681 | +0.17% |

Cross nDCG improves (paired randomization, Holm p≈0.00020); same-language gain is not significant. Cross recall declines. This is a ranking/coverage tradeoff, remains **opt-in**, and does not justify default promotion from one corpus. BGE revision `2b34e84df040034d4b9eabb62383a87c18955822` is combined with PPLX `2c4d510dd4a732063c31a0f70193e35067b51fd8`, with all 196 MatMulNBits at accuracy_level=4. Encoding `pool-int8-single-v2-level4` requires reindex; both dense fields are FP16. Semantic depth and total vector vote budget are unchanged.

## Product costs

Three independent sequential process blocks per arm, 66 unique queries per process (six per language), actual search boundary. Every arm asserts corpus count, vector completeness, nonempty results and model work. Shared Apple M4/Mac16,12; warmed model calls and fresh process query/score caches. The median table is descriptive: CPU block ranges overlap and no per-query cost significance is asserted. Initial and cached automatic arms are different regimes.

Each per-process p50 is the sample median; p95 is nearest rank `sorted[ceil(0.95*N)-1]`. The table takes medians of process-block cells. `cost-summary-historical.json` preserves an earlier p95 calculation one rank too low; `cost-summary-correction.json` records the correction. Raw rows are unchanged.

| Comparison / metric | Before | After | Difference | Change |
|---|---:|---:|---:|---:|
| Default CPU stack: p50 | 1244.01 ms | 1272.31 ms | 28.30 ms | +2.28% |
| Default CPU stack: p95 | 1635.81 ms | 1686.48 ms | 50.67 ms | +3.10% |
| Default CPU stack: sampled warm process RSS | 1,724,874,752 B | 1,706,983,424 B | -17,891,328 B | -1.04% |
| Dual CPU vs main: p50 | 1244.01 ms | 1125.23 ms | -118.77 ms | -9.55% |
| Dual CPU vs main: p95 | 1635.81 ms | 1623.66 ms | -12.15 ms | -0.74% |
| Dual CPU vs main: sampled warm process RSS | 1,724,874,752 B | 2,465,054,720 B | 740,179,968 B | +42.91% |
| Cached automatic vs main: p50 | 399.81 ms | 400.01 ms | 0.20 ms | +0.05% |
| Cached automatic vs main: p95 | 524.61 ms | 522.38 ms | -2.22 ms | -0.42% |
| Cached automatic vs main: sampled warm process RSS | 2,707,390,464 B | 2,705,063,936 B | -2,326,528 B | -0.09% |
| Initial prototype vs corrected cached regime: p50 | 398.54 ms | 400.01 ms | 1.47 ms | +0.37% |
| Initial prototype vs corrected cached regime: p95 | 531.06 ms | 522.38 ms | -8.68 ms | -1.63% |
| Initial prototype vs corrected cached regime: sampled warm process RSS | 6,322,061,312 B | 2,705,063,936 B | -3,616,997,376 B | -57.21% |

Do not infer a CPU latency gain from the lower dual median: candidate lists/reranker work differ and the host is shared. Default automatic search is at main parity in these retained blocks. The initial-to-cached RSS comparison is the **combined** resource correction (early rejection plus reuse), not an isolated persistence effect. RSS is sampled after warm outside query timers, includes the native process and excludes accelerator services/VRAM. Compiled CoreML cache warmth was not independently controlled: causal cold-start and whole-harness peak comparisons are N/A.

These costs describe frozen `503bd9e` and `0f023be6a8d8d070a971f7e590ccccff2c3292bb` candidates, with true main `315c10242ddf7a1cec3bccbf550a942320e09557`. Later calibration-lock/workload changes have **not** been retimed end to end. CPU-only arms use optimized ORT CPU/SIMD; automatic main/candidate arms use CPU embedding and the CoreML accurate-reranker EP. `device-and-cache-proof.json` retains the loaded-device records and proves all three persisted-hit arms contain no recalibration or rejection lines. EP selection does not prove ANE/GPU physical placement. Raw BGE CoreML still fails CPU-space conformance; this stack does not claim that numerical bug is repaired. Native WinML/NPU speed/quality/RSS remain N/A: no vendor NPU hardware is available.

## Disk

| Metric | Before | Dual | Difference | Change |
|---|---:|---:|---:|---:|
| Projection logical bytes | 83,303,573 B | 117,151,847 B | +33,848,274 B | +40.63% |
| Filesystem allocated blocks | 83,877,888 B | 117,755,904 B | +33,878,016 B | +40.39% |

Projection only; PostgreSQL and model caches excluded. APFS unique shared extents are not resolved. PPLX external data is 706,117,632 bytes, with another owned logical copy required beside the derived graph for ORT path validation; no physical COW saving is claimed. Observed plan metadata totals 64,646 bytes. Native accelerator service memory and causal model-cache disk differences are unmeasured.

## Verification / reproduction

```sh
python3 benchmarks/verify_compute_evidence.py
```

This independently recomputes paired nDCG/recall and all 21 cost blocks from raw rows, asserts paired IDs/corpus sizes and checks each reported percentile/RSS value. Timing program sources remain scratch-only; source/binary hashes and execution identities are retained. Re-running the live workload requires a provisioned unprivileged PostgreSQL workspace and the fingerprinted models/data, 66 fixed query indices `18*i` (i=0..65), warm query 1189, and the product settings above. Freeze executables before launch and keep builds out of timing runs. Historical total ingest time was measured on a shared host with intervening builds and is not a causal speed comparison.
