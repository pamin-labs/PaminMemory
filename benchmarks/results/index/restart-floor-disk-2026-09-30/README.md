# Restarted Engine upkeep on a legacy DiskANN index

This retained experiment measures actual product restart/write/flush/upkeep and Accurate search on a native DiskANN index. It uses three rotated independent processes per arm and the same stopped 18,000-document snapshot. Figures below are recomputed from the complete nine-process archive; they are descriptive observations, not a general search-speedup or multilingual-quality claim.

## Immediate predecessor comparison

Before: `f57f9c218d03d88666b3cc89fae9ae7e9eed2e50`; after: `11493c1388f74b087db23136a94e4b8feed1efe3`.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-topic recall@10 (synthetic) | 0.875000 | 0.875000 | +0.000000 | +0.000% |
| Known-topic MRR@10 (synthetic) | 0.826389 | 0.826389 | +0.000000 | +0.000% |
| Optimize jobs per first upkeep tick | 1.000000 | 0.000000 | -1.000000 | -100.000% |
| Full-process elapsed median, includes diagnostics | 88.680526 s | 80.926449 s | -7.754077 s | -8.744% |
| Engine open wall median | 514.131634 ms | 477.707146 ms | -36.424488 ms | -7.085% |
| Write + urgent drain wall median | 1421.857852 ms | 1388.009180 ms | -33.848672 ms | -2.381% |
| Durability flush wall median | 71.154003 ms | 64.986683 ms | -6.167320 ms | -8.668% |
| Maintenance wall median | 8463.866316 ms | 0.388897 ms | -8463.477419 ms | -99.995% |
| Maintenance process CPU median, 10ms tick counters | 27.320000 s | 0.000000 s | -27.320000 s | -100.000% |
| First-search wall median, model load included | 3286.225253 ms | 2846.251162 ms | -439.974091 ms | -13.388% |
| Warm search p50, median across processes | 2743.028312 ms | 2860.084011 ms | +117.055699 ms | +4.267% |
| Warm search p95, median across processes | 2999.154590 ms | 3070.581105 ms | +71.426515 ms | +2.382% |
| Peak process RSS | 1782.003906 MiB | 1778.898438 MiB | -3.105469 MiB | -0.174% |
| RSS after maintenance | 1351.167969 MiB | 1272.832031 MiB | -78.335938 MiB | -5.798% |
| Open index apparent bytes | 152.999107 MiB | 158.007785 MiB | +5.008677 MiB | +3.274% |
| Open index allocated bytes | 154.695312 MiB | 155.773438 MiB | +1.078125 MiB | +0.697% |
| Closed index apparent bytes | 152.999107 MiB | 158.007785 MiB | +5.008677 MiB | +3.274% |
| Closed index allocated bytes | 153.703125 MiB | 154.781250 MiB | +1.078125 MiB | +0.701% |
| Vector graph completeness, hybrid visibility checked | 1.000000 | 0.999944 | -0.000056 | -0.006% |

## Combined stack against main

Before: `315c10242ddf7a1cec3bccbf550a942320e09557`; after: `11493c1388f74b087db23136a94e4b8feed1efe3`.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-topic recall@10 (synthetic) | 0.875000 | 0.875000 | +0.000000 | +0.000% |
| Known-topic MRR@10 (synthetic) | 0.826389 | 0.826389 | +0.000000 | +0.000% |
| Optimize jobs per first upkeep tick | 1.000000 | 0.000000 | -1.000000 | -100.000% |
| Full-process elapsed median, includes diagnostics | 89.830001 s | 80.926449 s | -8.903552 s | -9.912% |
| Engine open wall median | 510.071716 ms | 477.707146 ms | -32.364570 ms | -6.345% |
| Write + urgent drain wall median | 1475.758326 ms | 1388.009180 ms | -87.749146 ms | -5.946% |
| Durability flush wall median | 65.174956 ms | 64.986683 ms | -0.188273 ms | -0.289% |
| Maintenance wall median | 8725.241690 ms | 0.388897 ms | -8724.852793 ms | -99.996% |
| Maintenance process CPU median, 10ms tick counters | 27.990000 s | 0.000000 s | -27.990000 s | -100.000% |
| First-search wall median, model load included | 3088.230447 ms | 2846.251162 ms | -241.979285 ms | -7.836% |
| Warm search p50, median across processes | 2746.017698 ms | 2860.084011 ms | +114.066313 ms | +4.154% |
| Warm search p95, median across processes | 3000.415286 ms | 3070.581105 ms | +70.165818 ms | +2.339% |
| Peak process RSS | 1781.714844 MiB | 1778.898438 MiB | -2.816406 MiB | -0.158% |
| RSS after maintenance | 1351.890625 MiB | 1272.832031 MiB | -79.058594 MiB | -5.848% |
| Open index apparent bytes | 152.999107 MiB | 158.007785 MiB | +5.008677 MiB | +3.274% |
| Open index allocated bytes | 154.695312 MiB | 155.773438 MiB | +1.078125 MiB | +0.697% |
| Closed index apparent bytes | 152.999107 MiB | 158.007785 MiB | +5.008677 MiB | +3.274% |
| Closed index allocated bytes | 153.703125 MiB | 154.781250 MiB | +1.078125 MiB | +0.701% |
| Vector graph completeness, hybrid visibility checked | 1.000000 | 0.999944 | -0.000056 | -0.006% |

[Full-process elapsed](episode-elapsed.json) wraps spawn, open, write, flush, upkeep,28 search calls, shutdown, log output and in-process diagnostics. It excludes snapshot copying and parent log parsing/provider checks. This is a measured restart episode, not pure-search latency or proof of a stable general speedup.

## Evidence and limits

- All 72 warm ordered top-ten lists match the candidate per reference; known-topic recall/MRR are reported from actual ranks, including misses. All 9 processes retrieve the new restart-proof memory. Near-duplicate synthetic identifier queries do not establish general multilingual accuracy.
- Candidate graph completeness remains 18000/18001. The implementation includes an exact flat-buffer path, but this experiment asserts only that the new memory is visible in full hybrid search. Lexical channels can also retrieve it; no vector-channel-specific result or attribution was retained. These rows therefore cannot prove that this new vector was returned by the vector channel, or rule out vector-channel loss. Coverage is retained in raw rows and tables and must not be rounded to1. Both RSS and disk increases or decreases are included.
- Legacy 2,000-document segments preserve the original schema and stored 1024-dimensional fp16 vectors. Source BGE-M3 and Accurate reranker are int8 models; this does not mean the vector field is int8. Native DiskANN/Cosine uses degree 64, build list 100, PQ chunks 0, quantize 0 and no rotation. Conversion preserved all 18,000 IDs/text/vector bit patterns, logical digest eb2483ff691e5e245079745e4745934620caf7ae692deea3766c84511c061d6b. The helper 125.91 s is total conversion plus before/after digest validation, not isolated graph-build time. Setup is excluded from timing.
- Persisted native-file floor is 273. Total diagnostic files are 274 because the marker itself is included by disk inspection but excluded from the product budget. Product Optimize generated the floor; no inert files or handwritten floors establish the premise.
- Native Linux DiskANN uses synchronous pread because io_uring is unavailable and libaio cannot load. No async-I/O performance claim is supported. One workload ran at a time; the host remains shared. Kernel file-cache reclaim pressure occurred. Cgroup memory.events includes intervening fusion diagnostics plus this matrix, so its post-run values are not an isolated DiskANN delta; OOM and oom_kill observations are retained without attributing max events to this experiment. Warm model/OS caches do not guarantee unpressured residency. [Post-run resource snapshot](post-trial-resources.json) retains exact counters and available disk. Hardware/runtime/model identities, exact commits and actual CPU provider assignments are retained in provenance and logs.
- Product order is Engine::open → remember → WhatAMemoryNeeds → flush_what_is_applied → one maintain → search_reranked(Accurate). Only Optimize may remain at the maintenance boundary; predecessor/main complete 1 and candidate 0. Maintenance counters stop before diagnostic SQL/disk reads. First search includes model load; two distinct warmups precede 24 distinct queries with uncached reranker-score assertions.
- Process CPU includes all Engine threads but excludes PostgreSQL, child/services and devices. RSS is Engine VmRSS/VmHWM, including mapped pages. CPU tick resolution is 10 ms; zero observed candidate ticks are not zero cost. Total service CPU/RSS and accelerator cost are N/A. Index apparent/allocated bytes exclude shared model cache, PostgreSQL, runner copies and build/setup artifacts.
- Every source ONNX/tokenizer/config and prepared graph/external weight file was hashed before and after timing. Frozen binary identities were checked after timing; dated hashes are retained in binaries.json. Private credential records are excluded; public paths are placeholders and raw values/list ordering remain unchanged.
- p95 is linear interpolation at (n−1)×0.95 within 24 samples, then the median of three process p95s. All other table values are process medians. Three repetitions do not establish patch causality for small search/RSS movements; all tradeoffs remain visible.

Run `python3 benchmarks/results/index/restart-floor-disk-2026-09-30/verify.py` to check hashes, all 9 processes, 72 paired lists per reference, actual providers, job counts, new-write visibility, native schema/profile and recomputed summary. Raw rows are in [raw.jsonl](raw.jsonl); full process/setup logs are gzip files in [logs](logs/); [DiskANN reproduction instructions](../../../harnesses/restart-floor-2026-09-30/disk-README.md) describe setup and the exact runner; the conversion source is retained as [an inert helper](../../../harnesses/restart-floor-2026-09-30/disk_schema.rs.in).

A dated review-time post-trial rehash in `post-trial-libraries.json` matches the pre-trial ORT and native zvec library hashes and sizes. These endpoint checks cannot exclude an intervening library replacement; they are not per-process attestation. The future conversion helper refuses a seed whose full stored-document digest/count differs from this retained reference before mutating the index.

Historical conversion source and the future guarded reproducer are retained separately. The verifier parses both actual `DISK_SETUP_JSON` observations from the setup log and compares them exactly with provenance; it also verifies all five profile values against the retained profile-file hash.

Historical build provenance and tuning limit: the builder recorded nominal checkout commits and frozen executable hashes but did not reject dirty tracked inputs or capture build-time source contents. These records alone do not certify that the measured executables came from pristine revisions. Inherited `PAMIN_*` values, including inference-thread tuning, were not recorded; shipped-default configuration is not established. No retrospective byte-identical rebuild binding is claimed in this archive. Separate [prospective guarded wrappers](../../../harnesses/restart-floor-2026-09-30/README.md#prospective-sourceconfiguration-guards) reject changed/extra source inputs, attest fresh builds and explicitly use four inference threads for both setup and trials; they add no historical configuration claim.
