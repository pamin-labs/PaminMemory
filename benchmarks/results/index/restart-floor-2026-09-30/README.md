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
| Peak process RSS | 1768.152 MiB | 1780.238 MiB | +12.086 MiB | +0.684% |
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
| Peak process RSS | 1769.750 MiB | 1780.238 MiB | +10.488 MiB | +0.593% |
| RSS after maintenance | 1305.109 MiB | 1230.020 MiB | -75.090 MiB | -5.754% |
| Closed index allocated bytes | 127.336 MiB | 128.422 MiB | +1.086 MiB | +0.853% |
| Closed index apparent bytes | 126.662 MiB | 131.678 MiB | +5.016 MiB | +3.961% |
| Vector graph completeness (hybrid visibility checked) | 1.000000 | 0.999944 | -0.000056 | -0.006% |

Warm p50/p95 and peak RSS are descriptive observations. Peak process RSS increases 0.684%; the warm timing comparison screen is retained separately. Three independent process repetitions are insufficient to assign these changes to the patch. Both the increases and individual process values are retained. No query-level significance test is reported.

## Conditions and measurement limits

- Linux 6.18.44, AMD EPYC 9V74, cgroup 4 CPU cores and 16 GiB RAM. One workload process at a time; no builds or other model experiments during timing. The host is shared, so external interference is not controlled.
- Actual product calls: `Engine::open`, `remember`, urgent cascade, `flush_what_is_applied`, one `Engine::maintain`, then `search_reranked(Accurate)`. The actual server's flush-before-maintain order is retained. Open, write, durability flush, upkeep, first search, two warmups, 24 unique warm queries and shutdown are separated; diagnostic SQL/disk inspections are outside upkeep timing. A preliminary missing-flush harness failed on the held Sync claim and was excluded, rather than weakening the zero-pending check.
- The runner copies a nominal shared stopped 18,000-document database/index seed outside timing. Historical per-copy content attestations were not captured; identical corpus contents across all repetitions are not certified. All records were written through real Engine embedding/cascade operations. Only the empty initial collection's supported legacy segment schema was set to 2,000; no inert files simulate fragmentation. Setup advanced 8,000→12000→16000→18000 with a real optimize and clean shutdown each time. Its successful final floor is 271. New collections use the current 10,000 policy; this result applies to the supported legacy layout, not every index.
- Release test binaries are frozen and hashed before timing. Setup used a debug Rust harness with the same release ORT/native libraries; embedding kernels and stored vectors are shared by every arm. Post-trial rehashing confirms the prehashed model source/tokenizer/ONNX assets are unchanged. Prepared external weight files were identified and hashed after timing, not independently before; that limitation is explicit in [post-trial asset identities](post-trial-assets.json). Exact commits, compiler and library identities, model/tokenizer/graph hashes, index snapshot file hashes and binary hashes are retained in [provenance](provenance.json) and [binary identities](binaries.json).
- BGE-M3 int8 revision `2b34e84df040034d4b9eabb62383a87c18955822`; Accurate BGE reranker int8 revision `6f5ff65298512715a1e669753bc754d2bc8f367b`; fp16 stored 1,024-dimensional vectors, HNSW/Cosine, default accuracy profile and depths. Both actual scoring graphs are asserted to have only nonzero CPUExecutionProvider assignment. This is a CPU measurement, not a CoreML/ANE comparison. Loaded ORT distribution is 1.28.0; the historical prepared-key label 1.24 identifies its Rust API compatibility target, and is not the loaded binary version.
- Model download/prepared-graph and OS file caches are warm; every arm starts a fresh Engine process. The first search includes reranker load. Two disjoint warmups precede 24 distinct fixed queries. Every warm query asserts actual fresh reranker scores for all offered pairs. Model-cache state and hashes are held fixed; no backend/model substitution occurs.
- Per-process p95 uses linear interpolation at (n−1)×0.95, followed by the median of three process p95s. Other table values are medians of process metrics. Wall time is separate from all-thread process user/system CPU; PostgreSQL child/service CPU and RSS are excluded. CPU tick resolution is 10 ms: median measured upkeep CPU is 0.47 s for predecessor, 0.44 s for main and 0 observed ticks for candidate. This does not mean zero CPU cost. Device/service memory and total server-resource comparisons are N/A.
- RSS is this Engine process, including mapped model/index pages; peak is kernel VmHWM. Index disk is reported while open and after Engine/PG close, with apparent and allocated bytes separate. Common model-cache disk, PostgreSQL data, runner copies and build/scratch artifacts are excluded. DiskANN, full corpus multilingual quality and service concurrency are N/A in this HNSW experiment.
- The saved-floor marker is present in the identical seed for all arms. Before-patch binaries ignore it and count its one file; the candidate excludes it from its file budget. The genuine floor is already 271 without this file, so it does not create the >256 premise.

## Recompute and reproduce

Run `python3 benchmarks/results/index/restart-floor-2026-09-30/verify.py`. This verifies archive hashes, process counts, actual providers, work premise, paired outputs, new-write visibility and recomputes [summary.json](summary.json) from [raw.jsonl](raw.jsonl). Raw per-phase CPU and RSS observations are retained.

Source and setup/build/runner instructions are in [the inert harness archive](../../../harnesses/restart-floor-2026-09-30/README.md). Corpus content and query IDs are defined in `harness.rs.in`; order is predecessor/candidate/main, candidate/main/predecessor, main/predecessor/candidate. [Process logs](logs/) retain provider assignment and all search rows. Public logs and metadata replace local path prefixes with placeholders and trim trailing blank lines; model hashes, metrics and result lists are unchanged. No database connection or credential files are published.

A [retrospective rebuild audit](../restart-floor-disk-2026-09-30/rebuild/README.md) reproduces all three frozen executables byte for byte from their stated revisions and the retained harness. The read-only verifier checks this binding against local Git objects for all three revisions; shallow checkouts must obtain them first. Historical build-time source state and inherited inference-thread tuning remain unknown, so these are not certified shipped-default cost comparisons.

First-search change comparisons are withheld because the three predecessor observations span 20.5% of their median and candidate observations span 25.6%. The observed arm medians and raw arithmetic remain inspectable; further controlled rounds are needed for a latency comparison.

Timing comparison policy now scans every retained wall/CPU timing metric and full-process elapsed for both references. It withholds difference/percentage cells when either arm spans more than 10% of its process median or matched repetition deltas reverse sign. Before/after medians and raw arithmetic remain descriptive observations; this screen is not a significance test. [Timing review](timing-review.json) retains all process values and reasons, plus every actual timed phase/query row including warmups and new-memory search. The saved work counts still establish skipped optimize work, including when small candidate upkeep timings are unstable.
