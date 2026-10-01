# Restarted Engine upkeep on a legacy HNSW index

Persisting the successful optimize floor avoids a repeated optimize after reopening an index whose compacted layout exceeds the existing 256-file budget. In three rotated processes per arm, the predecessor and current main each completed one optimize; the candidate completed none. This establishes the upkeep cost in this workload. It does not establish a whole-search speedup.

Accuracy has first priority: every one of the 24 fixed warm queries returned exactly the same ordered top ten across all three arms in each repetition (72/72 per reference). Known-topic recall was 21/24, not perfect, and MRR was 0.826389. All nine processes also retrieved the newly written `restart-proof` topic after upkeep. These near-duplicate synthetic identifier queries do not establish general multilingual retrieval accuracy.

Candidate graph completeness is 18000/18001, rather than 1.0. The product includes an exact flat-buffer path, but this experiment only confirms visibility in full hybrid search; lexical channels can retrieve the new memory. No vector-channel-specific attribution was retained, so these results do not rule out vector-channel loss. This costs 1,138,688 extra allocated index bytes after close (1.086 MiB) and 5,260,170 extra apparent bytes. Graph completeness alone does not establish vector loss or full vector recall; a skipped allocation does not establish RSS savings.

## Immediate predecessor comparison

Before: `f57f9c218d03d88666b3cc89fae9ae7e9eed2e50`; after: `11493c1388f74b087db23136a94e4b8feed1efe3`.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-topic recall@10 (synthetic) | 0.875000 | 0.875000 | +0.000000 | +0.000% |
| Known-topic MRR@10 (synthetic) | 0.826389 | 0.826389 | +0.000000 | +0.000% |
| Optimize jobs per first upkeep tick | 1.000 | 0.000 | -1.000 | -100.000% |
| Maintenance wall median | 500.255 ms | 0.479 ms | -499.776 ms | -99.904% |
| First-search wall median (model load included) | 2618.299 ms | 2621.711 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p50: median across processes | 2126.304 ms | 2139.206 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p95: median across processes | 2289.523 ms | 2368.295 ms | +78.771 ms | +3.441% |
| Peak process RSS | N/A: historical HWM chronology uncertified | N/A: historical HWM chronology uncertified | N/A | N/A |
| RSS after maintenance | 1304.809 MiB | 1230.020 MiB | -74.789 MiB | -5.732% |
| Closed index allocated bytes | 127.336 MiB | 128.422 MiB | +1.086 MiB | +0.853% |
| Closed index apparent bytes | 126.662 MiB | 131.678 MiB | +5.016 MiB | +3.961% |
| Vector graph completeness (hybrid visibility checked) | 1.000000 | 0.999944 | -0.000056 | -0.006% |

## Combined stack against current main

Before: `315c10242ddf7a1cec3bccbf550a942320e09557`; after: `11493c1388f74b087db23136a94e4b8feed1efe3`. Remote main was checked unchanged after timing. The earlier custom-fusion-k correction uses the unchanged default k=10 here.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-topic recall@10 (synthetic) | 0.875000 | 0.875000 | +0.000000 | +0.000% |
| Known-topic MRR@10 (synthetic) | 0.826389 | 0.826389 | +0.000000 | +0.000% |
| Optimize jobs per first upkeep tick | 1.000 | 0.000 | -1.000 | -100.000% |
| Maintenance wall median | 478.813 ms | 0.479 ms | -478.334 ms | -99.900% |
| First-search wall median (model load included) | 2731.516 ms | 2621.711 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |
| Warm search p50: median across processes | 2134.241 ms | 2139.206 ms | +4.965 ms | +0.233% |
| Warm search p95: median across processes | 2248.118 ms | 2368.295 ms | +120.176 ms | +5.346% |
| Peak process RSS | N/A: historical HWM chronology uncertified | N/A: historical HWM chronology uncertified | N/A | N/A |
| RSS after maintenance | 1305.109 MiB | 1230.020 MiB | -75.090 MiB | -5.754% |
| Closed index allocated bytes | 127.336 MiB | 128.422 MiB | +1.086 MiB | +0.853% |
| Closed index apparent bytes | 126.662 MiB | 131.678 MiB | +5.016 MiB | +3.961% |
| Vector graph completeness (hybrid visibility checked) | 1.000000 | 0.999944 | -0.000056 | -0.006% |

Warm p50/p95 are descriptive observations; the warm timing comparison screen is retained separately. Historical HWM-derived peak comparisons are uncertified/N/A because retained process observations decrease. Three independent process repetitions are insufficient to assign small timing or sampled-RSS movements to the patch. Individual raw process observations remain retained; the derived machine-readable summary explicitly nulls uncertified HWM peaks. No query-level significance test is reported.

## Conditions and measurement limits

- Linux 6.18.44, AMD EPYC 9V74, cgroup 4 CPU cores and 16 GiB RAM. One workload process at a time; no builds or other model experiments during timing. The host is shared, so external interference is not controlled.
- Actual product calls: `Engine::open`, `remember`, urgent cascade, `flush_what_is_applied`, one `Engine::maintain`, then `search_reranked(Accurate)`. The actual server's flush-before-maintain order is retained. Open, write, durability flush, upkeep, first search, two warmups, 24 unique warm queries and shutdown are separated; diagnostic SQL/disk inspections are outside upkeep timing. A preliminary missing-flush harness failed on the held Sync claim and was excluded, rather than weakening the zero-pending check.
- The runner copies a nominal shared stopped 18,000-document database/index seed outside timing. Historical per-copy content attestations were not captured; identical corpus contents across all repetitions are not certified. All records were written through real Engine embedding/cascade operations. Only the empty initial collection's supported legacy segment schema was set to 2,000; no inert files simulate fragmentation. Setup advanced 8,000→12000→16000→18000 with a real optimize and clean shutdown each time. Its successful final floor is 271. New collections use the current 10,000 policy; this result applies to the supported legacy layout, not every index.
- Release test binaries are frozen and hashed before timing. Setup used a debug Rust harness with the same release ORT/native libraries; embedding kernels and stored vectors are shared by every arm. The retained HNSW post-trial receipt asserts the complete seven prehashed source/tokenizer/ONNX paths unchanged; it does not contain individual post-trial byte counts or digests for those seven files, so independent endpoint rehash verification is unavailable. Both original CPU exports are listed unchanged; no released-source transition was recorded or is accepted retrospectively. The verifier checks the two actual external-weight endpoint sizes, digests, mtimes and scope against the separate Disk inventory and original receipt. Prepared external weight files were identified and hashed after timing, not independently before; that limitation is explicit in [post-trial asset identities](post-trial-assets.json). Exact commits, compiler and library identities, model/tokenizer/graph hashes, index snapshot file hashes and binary hashes are retained in [provenance](provenance.json) and [binary identities](binaries.json).
- BGE-M3 int8 revision `2b34e84df040034d4b9eabb62383a87c18955822`; Accurate BGE reranker int8 revision `6f5ff65298512715a1e669753bc754d2bc8f367b`; fp16 stored 1,024-dimensional vectors, HNSW/Cosine, explicit accuracy profile; inherited search effort, reranker depth, batch/token controls and other `PAMIN_*` tuning were not recorded. Both actual scoring graphs are asserted to have only nonzero CPUExecutionProvider assignment. This is a CPU measurement, not a CoreML/ANE comparison. Loaded ORT distribution is 1.28.0; the historical prepared-key label 1.24 identifies its Rust API compatibility target, and is not the loaded binary version.
- Model download/prepared-graph and OS file caches are warm; every arm starts a fresh Engine process. The first search includes reranker load. The verifier checks the exact ordered schedule in every process: cold document `0`, warmups `n//2` then `n-1`, and warm documents `j*(n-2)//25+1` for `j=1..24`, with `n=18000`. Every warm query asserts actual fresh reranker scores for all offered pairs. Retained pretrial identities and observed CPU provider graph paths agree across arms; these records do not establish unchanged shared model-cache bytes between launches or rule out model substitution.
- Per-process p95 uses linear interpolation at (n−1)×0.95, followed by the median of three process p95s. Other table values are medians of process metrics. Wall time is separate from all-thread process user/system CPU; PostgreSQL child/service CPU and RSS are excluded. CPU tick resolution is 10 ms: median measured upkeep CPU is 0.47 s for predecessor, 0.44 s for main and 0 observed ticks for candidate. This does not mean zero CPU cost. Device/service memory and total server-resource comparisons are N/A.
- RSS observations cover this Engine process, including mapped model/index pages; retained VmHWM observations do not certify a lifetime peak. Index disk is reported while open and after Engine/PG close, with apparent and allocated bytes separate. Common model-cache disk, PostgreSQL data, runner copies and build/scratch artifacts are excluded. DiskANN, full corpus multilingual quality and service concurrency are N/A in this HNSW experiment.
- The saved-floor marker is present in the identical seed for all arms. Before-patch binaries ignore it and count its one file; the candidate excludes it from its file budget. The genuine floor is already 271 without this file, so it does not create the >256 premise.

## Recompute and reproduce

Run `python3 benchmarks/results/index/restart-floor-2026-09-30/verify.py`. This verifies archive hashes, process counts, actual providers, work premise, paired outputs, new-write visibility and recomputes [summary.json](summary.json) from [raw.jsonl](raw.jsonl). Raw per-phase CPU and RSS observations are retained.

Source and setup/build/runner instructions are in [the inert harness archive](../../../harnesses/restart-floor-2026-09-30/README.md). Corpus content and query IDs are defined in `harness.rs.in`; order is predecessor/candidate/main, candidate/main/predecessor, main/predecessor/candidate. [Process logs](logs/) retain provider assignment and all search rows. Public logs and metadata replace local path prefixes with placeholders and trim trailing blank lines; model hashes, metrics and result lists are unchanged. No database connection or credential files are published.

A [retrospective rebuild audit](../restart-floor-disk-2026-09-30/rebuild/README.md) reproduces all three frozen executables byte for byte after freshly compiling the four product libraries and retained helper from their stated revisions. All 538 third-party compiler artifacts per arm were reused from the shared Cargo target; their source-to-artifact provenance is unknown, so this is not full source-to-artifact attestation. The read-only verifier checks this binding against local Git objects for all three revisions; shallow checkouts must obtain them first. Historical build-time source state and all inherited `PAMIN_*` tuning remain unknown, so these are not certified shipped-default cost comparisons.

First-search change comparisons are withheld because the three predecessor observations span 20.5% of their median and candidate observations span 25.6%. The observed arm medians and raw arithmetic remain inspectable; further controlled rounds are needed for a latency comparison.

Timing comparison policy now scans every retained wall/CPU timing metric and full-process elapsed for both references. It withholds difference/percentage cells when either arm spans more than 10% of its process median or matched repetition deltas reverse sign. Before/after medians and raw arithmetic remain descriptive observations; this screen is not a significance test. [Timing review](timing-review.json) retains all process values and reasons, plus every actual timed phase/query row including warmups and new-memory search. The saved work counts still establish skipped optimize work, including when small candidate upkeep timings are unstable.

The verifier binds each process/role graph path and node count to the HNSW source/prepared-graph inventory, whose retained byte counts and hashes agree with the Disk archive role binding. Prepared source metadata and external-weight inventories are retained in that separate Disk archive; they were not independently inventoried for each HNSW process. This cross-archive consistency check does not recover historical per-launch shared-cache immutability.

Historical `TOKIO_WORKER_THREADS` and the effective Tokio worker count were not recorded and remain unknown. The future guarded runner rejects inherited Tokio tuning, explicitly sets four Tokio workers and records that setting; it does not retrospectively certify historical async concurrency.

Historical memory chronology qualification: the [recomputed receipt](hwm-review.json) binds the exact retained before/after HWM decreases and sampled VmRSS observations. No tolerance is applied: any added, removed or changed anomaly fails verification. The HWM-derived peak process RSS comparisons above are uncertified/N/A; sampled maintenance RSS remains an observation, not a lifetime peak. Original raw/log rows and calculator remain unchanged. The machine-readable summary is newly derived and supersedes its numeric predecessor: all peak-RSS process/median/comparison values are null with enforced N/A certification. Its derivation records the raw, calculator and superseded numeric-summary hashes; other metrics retain the same arithmetic.

CPU-counter verification checks nonnegative integer user/system cumulative ticks across every consecutive non-null before/after snapshot, including phase boundaries and missing-snapshot gaps. It does not add full-lifetime CPU coverage beyond the retained measurement scope.
