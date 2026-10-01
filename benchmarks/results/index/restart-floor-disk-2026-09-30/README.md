# Restarted Engine upkeep on a legacy DiskANN index

This retained experiment measures actual product restart/write/flush/upkeep and Accurate search on a native DiskANN index. It uses three rotated independent processes per arm and a nominal shared stopped 18,000-document seed. The original runner did not attest every copied snapshot, so identical corpus contents across all repetitions are not certified. Figures below are recomputed from the complete nine-process archive; they are descriptive observations, not a general search-speedup or multilingual-quality claim.

## Immediate predecessor comparison

Before: `f57f9c218d03d88666b3cc89fae9ae7e9eed2e50`; after: `11493c1388f74b087db23136a94e4b8feed1efe3`.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-topic recall@10 (synthetic) | 0.875000 | 0.875000 | +0.000000 | +0.000% |
| Known-topic MRR@10 (synthetic) | 0.826389 | 0.826389 | +0.000000 | +0.000% |
| Optimize jobs per first upkeep tick | 1.000000 | 0.000000 | -1.000000 | -100.000% |
| Full-process elapsed median, includes diagnostics | 88.680526 s | 80.926449 s | -7.754077 s | -8.744% |
| Engine open wall median | 514.131634 ms | 477.707146 ms | -36.424488 ms | -7.085% |
| Write + urgent drain wall median | 1421.857852 ms | 1388.009180 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Durability flush wall median | 71.154003 ms | 64.986683 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Maintenance wall median | 8463.866316 ms | 0.388897 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Maintenance process CPU median, 10ms tick counters | 27.320000 s | <0.020 s at combined counter resolution (0 observed ticks) | Withheld: censored counter observation | Withheld: censored counter observation |
| First-search wall median, model load included | 3286.225253 ms | 2846.251162 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p50, median across processes | 2743.028312 ms | 2860.084011 ms | +117.055699 ms | +4.267% |
| Warm search p95, median across processes | 2999.154590 ms | 3070.581105 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Peak process RSS | N/A: historical HWM chronology uncertified | N/A: historical HWM chronology uncertified | N/A | N/A |
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
| Engine open wall median | 510.071716 ms | 477.707146 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Write + urgent drain wall median | 1475.758326 ms | 1388.009180 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Durability flush wall median | 65.174956 ms | 64.986683 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Maintenance wall median | 8725.241690 ms | 0.388897 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Maintenance process CPU median, 10ms tick counters | 27.990000 s | <0.020 s at combined counter resolution (0 observed ticks) | Withheld: censored counter observation | Withheld: censored counter observation |
| First-search wall median, model load included | 3088.230447 ms | 2846.251162 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p50, median across processes | 2746.017698 ms | 2860.084011 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p95, median across processes | 3000.415286 ms | 3070.581105 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Peak process RSS | N/A: historical HWM chronology uncertified | N/A: historical HWM chronology uncertified | N/A | N/A |
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
- Legacy 2,000-document segments preserve the original schema and stored 1024-dimensional fp16 vectors. Source BGE-M3 and Accurate reranker are int8 models; this does not mean the vector field is int8. Native DiskANN/Cosine requests degree 64, build list 100, PQ chunks 0, quantize 0 and no rotation. These are schema getter values; effective PQ chunks/centroids were not captured, so this does not establish disabled PQ. Conversion preserved all 18,000 IDs/text/vector bit patterns, logical digest eb2483ff691e5e245079745e4745934620caf7ae692deea3766c84511c061d6b. The helper 125.91 s is total conversion plus before/after digest validation, not isolated graph-build time. Setup is excluded from timing.
- Persisted native-file floor is 273. Total diagnostic files are 274 because the marker itself is included by disk inspection but excluded from the product budget. Product Optimize generated the floor; no inert files or handwritten floors establish the premise.
- Native Linux DiskANN uses synchronous pread because io_uring is unavailable and libaio cannot load. No async-I/O performance claim is supported. One workload ran at a time; the host remains shared. Kernel file-cache reclaim pressure occurred. Cgroup memory.events includes intervening fusion diagnostics plus this matrix, so its post-run values are not an isolated DiskANN delta; OOM and oom_kill observations are retained without attributing max events to this experiment. Warm model/OS caches do not guarantee unpressured residency. [Post-run resource snapshot](post-trial-resources.json) retains exact counters and available disk. Hardware/runtime/model identities, exact commits and actual CPU provider assignments are retained in provenance and logs.
- Product order is Engine::open → remember → WhatAMemoryNeeds → flush_what_is_applied → one maintain → search_reranked(Accurate). Only Optimize may remain at the maintenance boundary; predecessor/main complete 1 and candidate 0. Maintenance counters stop before diagnostic SQL/disk reads. First search includes model load; two distinct warmups precede 24 distinct queries with uncached reranker-score assertions.
- Process CPU includes all Engine threads but excludes PostgreSQL, child/services and devices. RSS is Engine VmRSS/VmHWM, including mapped pages. CPU tick resolution is 10 ms; zero observed candidate ticks are not zero cost. Total service CPU/RSS and accelerator cost are N/A. Index apparent/allocated bytes exclude shared model cache, PostgreSQL, runner copies and build/setup artifacts.
- Every source ONNX/tokenizer/config and prepared graph/external weight file was hashed before and after timing. Frozen binary identities were checked after timing; dated hashes are retained in binaries.json. Private credential records are excluded; public paths are placeholders and raw values/list ordering remain unchanged.
- p95 is linear interpolation at (n−1)×0.95 within 24 samples, then the median of three process p95s. All other table values are process medians. Three repetitions do not establish patch causality for small search/RSS movements; all tradeoffs remain visible.

Run `python3 benchmarks/results/index/restart-floor-disk-2026-09-30/verify.py` to check hashes, all 9 processes, 72 paired lists per reference, actual providers, job counts, new-write visibility, native schema/profile and recomputed summary. Raw rows are in [raw.jsonl](raw.jsonl); full process/setup logs are gzip files in [logs](logs/); [DiskANN reproduction instructions](../../../harnesses/restart-floor-2026-09-30/disk-README.md) describe setup and the exact runner; the conversion source is retained as [an inert helper](../../../harnesses/restart-floor-2026-09-30/disk_schema.rs.in).

A dated review-time post-trial rehash in `post-trial-libraries.json` matches the pre-trial ORT and native zvec library hashes and sizes. These endpoint checks cannot exclude an intervening library replacement; they are not per-process attestation. The future conversion helper refuses a seed whose full stored-document digest/count differs from this retained reference before mutating the index.

Historical conversion source and the future guarded reproducer are retained separately. The verifier parses both actual `DISK_SETUP_JSON` observations from the setup log and compares them exactly with provenance; it also verifies all five profile values against the retained profile-file hash.

Historical build provenance and tuning limit: the builder recorded nominal checkout commits and frozen executable hashes but did not reject dirty tracked inputs or capture build-time source contents. These records alone do not certify that the measured executables came from pristine revisions. Inherited `PAMIN_*` values, including inference-thread tuning, were not recorded; shipped-default configuration is not established. The separate [retrospective rebuild](rebuild/README.md) verifies byte equality for all three frozen executables after fresh compilation of the four product libraries and helper from their stated revisions. The shared Cargo target reused all 538 third-party compiler artifacts per arm; their source-to-artifact provenance is unknown, so this is not full source-to-artifact attestation. Original build-time source state and all inherited `PAMIN_*` tuning remain unknown, including search effort, reranker depth, batch/token controls and inference threads. Separate [prospective guarded wrappers](../../../harnesses/restart-floor-2026-09-30/README.md#prospective-sourceconfiguration-guards) reject changed/extra source inputs, attest fresh builds and explicitly use four inference threads for both setup and trials; they add no historical configuration claim.

First-search comparisons are withheld: the predecessor samples span about 18.2% of their median, and repetition-matched candidate differences reverse direction (approximately −15.2%, +8.6%, −17.1%). The table retains observed arm medians for inspection but makes no latency improvement claim. Three rotated processes do not satisfy the additional-round requirement for this spread. Raw arithmetic remains in the archive for audit; further controlled rounds must fix the historically unrecorded `PAMIN_*` tuning before supporting a comparison.

The main read-only verifier also checks retrospective source/binary binding. It requires local Git objects for all three recorded revisions; a shallow checkout must obtain those objects before verification. No build, model or database is run by this check.

Timing comparison policy now scans every retained wall/CPU timing metric and full-process elapsed for both references. It withholds difference/percentage cells when either arm spans more than 10% of its process median or matched repetition deltas reverse sign. Before/after medians and raw arithmetic remain descriptive observations; this screen is not a significance test. [Timing review](timing-review.json) retains all process values and reasons, plus every actual timed phase/query row including warmups and new-memory search. The saved work counts still establish skipped optimize work, including when small candidate upkeep timings are unstable.

[Provider bindings](provider-bindings.json) normalize each actual process-copy model path through the recorded models symlink to its exact embedding/reranker prepared graph and external-data inventory. Source metadata contents were captured in a new dated read-only observation and match the historical metadata digest; source size/SHA/revision are verified. This is not a retrospective preparation attestation. The verifier additionally binds the complete unique arm-keyed binary pretrial inventory and exact historical runner bytes, and requires successful seed/conversion final markers.

A [dated review-time seed endpoint](post-review-seed.json) rehashes all 2,952 retained regular-file inputs (312,706,264 bytes) from the surviving stopped seed and matches the complete pretrial inventory. The model symlink target is also unchanged. This is a late endpoint check after all nine historical copies; it cannot exclude intervening source changes or replace missing per-copy attestations. Regular-file scope excludes symlinks and credential/control records as listed in the record. Cross-repetition identical-corpus attribution remains unknown. The initial endpoint scope check included eight PostgreSQL-library symlinks and failed before hashing; the successful capture uses the retained regular-file scope.

Maintenance CPU is censored: all candidate observations have zero user and system ticks. At the stated 100 Hz accounting resolution, each component is below one 10 ms tick, with a nominal combined bound below 20 ms; this is not zero CPU cost. Exact CPU differences and percentages are withheld. Raw tick arithmetic remains retained for recomputation, without treating zero ticks as a measured zero. PostgreSQL/service/device CPU is excluded.

Historical `TOKIO_WORKER_THREADS` and the effective Tokio worker count were not recorded and remain unknown. The future guarded runner rejects inherited Tokio tuning, explicitly sets four Tokio workers and records that setting; it does not retrospectively certify historical async concurrency.

Historical memory chronology qualification: the [recomputed receipt](hwm-review.json) binds the exact retained before/after HWM decreases and sampled VmRSS observations. No tolerance is applied: any added, removed or changed anomaly fails verification. The HWM-derived peak process RSS comparisons above are uncertified/N/A; sampled maintenance RSS remains an observation, not a lifetime peak. Original raw/log rows and calculator remain unchanged. The machine-readable summary is newly derived and supersedes its numeric predecessor: all peak-RSS process/median/comparison values are null with enforced N/A certification. Its derivation records the raw, calculator and superseded numeric-summary hashes; other metrics retain the same arithmetic.

CPU-counter verification checks nonnegative integer user/system cumulative ticks across every consecutive non-null before/after snapshot, including phase boundaries and missing-snapshot gaps. It does not add full-lifetime CPU coverage beyond the retained measurement scope.

Offline verification requires each maintenance optimize-job observation to be
an actual nonnegative JSON integer, excluding booleans, before comparing the
arm’s expected count. Matching raw/log values and refreshed derived summaries
or manifests do not replace this type check. Historical raw values are unchanged.

Verification binds the exact retained warm-cache/isolated-stopped-copy row
annotation and requires query ranks to be null or actual positive integers,
excluding booleans, before comparing them with returned positions. These checks
validate declared historical evidence, not independent runtime cache isolation.

Stopped-seed endpoint verification requires inventory paths within the recorded
`<SCRATCH>/seed-disk` root and the exact retained regular-file/credential exclusion
policy. The late endpoint remains a surviving-file observation, not per-copy proof.
