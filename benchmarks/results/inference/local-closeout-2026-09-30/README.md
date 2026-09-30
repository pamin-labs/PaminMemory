# Local backend validation, 2026-09-30

Completed product-entry timing probe: 24 fixed XQuAD queries, three rotated rounds, 432 searches on a shared Apple workstation. Eight disjoint warmup queries; measured reranker calls asserted newly scored == offered > 0. Complete 13,014-document accuracy memory index asserted. These are alternative reranker plans (embedding fixed on CPU), not a new optimization against main.

CoreML ALL allows CPU, GPU and ANE. The GPU arm is CoreML CPUAndGPU. CPU uses ONNX Runtime optimized kernels; its model export differs from CoreML. Accurate stays ALL, Fast stays CPU.

## Accurate

| Alternative / metric | Current path | Alternative | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| gpu: p50 search ms | 619.722 | 3067.060 | +2447.338 | +394.91% |
| gpu: p95 search ms | 880.858 | 3722.879 | +2842.020 | +322.64% |
| gpu: 24-search wall total s | 16.908 | 75.434 | +58.526 | +346.13% |
| gpu: whole-process wall s (load + warmup + searches) (uncontrolled cache warmth) | 32.440 | 120.230 | N/A: not comparable | N/A |
| gpu: process CPU s (same process scope) (uncontrolled cache warmth) | 13.650 | 23.650 | N/A: not comparable | N/A |
| gpu: peak process RSS MiB (uncontrolled cache warmth) | 2498.469 | 4771.656 | N/A: not comparable | N/A |
| gpu: sample same_language nDCG@10 | 0.9097186 | 0.9097186 | +0.0000000 | +0.00% |
| gpu: sample cross_lingual nDCG@10 | 0.8613851 | 0.8613851 | +0.0000000 | +0.00% |
| cpu: p50 search ms | 619.722 | 2917.843 | +2298.121 | +370.83% |
| cpu: p95 search ms | 880.858 | 4262.603 | +3381.745 | +383.91% |
| cpu: 24-search wall total s | 16.908 | 73.956 | +57.048 | +337.39% |
| cpu: whole-process wall s (load + warmup + searches) (uncontrolled cache warmth) | 32.440 | 99.390 | N/A: not comparable | N/A |
| cpu: process CPU s (same process scope) (uncontrolled cache warmth) | 13.650 | 328.070 | N/A: not comparable | N/A |
| cpu: peak process RSS MiB (uncontrolled cache warmth) | 2498.469 | 1503.391 | N/A: not comparable | N/A |
| cpu: sample same_language nDCG@10 | 0.9097186 | 0.9001513 | -0.0095673 | -1.05% |
| cpu: sample cross_lingual nDCG@10 | 0.8613851 | 0.8624608 | +0.0010757 | +0.12% |

## Fast

| Alternative / metric | Current path | Alternative | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| all: p50 search ms | 448.874 | 4533.056 | +4084.182 | +909.87% |
| all: p95 search ms | 657.363 | 6790.455 | +6133.092 | +932.98% |
| all: 24-search wall total s | 12.577 | 135.486 | +122.909 | +977.25% |
| all: whole-process wall s (load + warmup + searches) (uncontrolled cache warmth) | 17.970 | 171.570 | N/A: not comparable | N/A |
| all: process CPU s (same process scope) (uncontrolled cache warmth) | 46.430 | 144.470 | N/A: not comparable | N/A |
| all: peak process RSS MiB (uncontrolled cache warmth) | 1159.422 | 2728.922 | N/A: not comparable | N/A |
| all: sample same_language nDCG@10 | 0.8812674 | 0.8812674 | +0.0000000 | +0.00% |
| all: sample cross_lingual nDCG@10 | 0.7580525 | 0.7620027 | +0.0039502 | +0.52% |
| gpu: p50 search ms | 448.874 | 4409.418 | +3960.545 | +882.33% |
| gpu: p95 search ms | 657.363 | 6326.876 | +5669.513 | +862.46% |
| gpu: 24-search wall total s | 12.577 | 137.022 | +124.445 | +989.47% |
| gpu: whole-process wall s (load + warmup + searches) (uncontrolled cache warmth) | 17.970 | 175.050 | N/A: not comparable | N/A |
| gpu: process CPU s (same process scope) (uncontrolled cache warmth) | 46.430 | 144.910 | N/A: not comparable | N/A |
| gpu: peak process RSS MiB (uncontrolled cache warmth) | 1159.422 | 2833.750 | N/A: not comparable | N/A |
| gpu: sample same_language nDCG@10 | 0.8812674 | 0.8812674 | +0.0000000 | +0.00% |
| gpu: sample cross_lingual nDCG@10 | 0.7580525 | 0.7620027 | +0.0039502 | +0.52% |

## Process-round diagnostic statistics

| Contrast | Process pairs | Descriptive time ratio | Two-sided block p | Family p |
| --- | ---: | ---: | ---: | ---: |
| accurate-gpu-vs-all | 3 | 4.4393 | 0.25 | 1.00 |
| accurate-cpu-vs-all | 3 | 4.2682 | 0.25 | 1.00 |
| fast-all-vs-cpu | 3 | 10.2557 | 0.25 | 1.00 |
| fast-gpu-vs-cpu | 3 | 10.0592 | 0.25 | 1.00 |

## Scope and decision

The prior query-level p=0.0002 claim is withdrawn: queries share a process-level backend assignment and host-load conditions. Four within-tier diagnostic contrasts now preserve three process-round pairs and enumerate all eight round-block sign flips. These three deterministic rotated rounds are not a randomized confirmatory trial; they cannot establish statistical significance. Reported latency differences are descriptive and do not change the existing per-tier policy. Sample nDCG remains diagnostic, not precision acceptance.

All process costs include loading and warmups. CoreML compilation-cache warmth was not recorded or controlled in any process. These whole-process wall/CPU/RSS magnitudes are observational only, not like-for-like backend cost comparisons; their deltas are N/A. Search-only durations begin after explicit model warmups and remain a separate scope. Process CPU excludes CoreML services/GPU/ANE work; RSS excludes their allocations. Thus these are neither energy nor model resident memory measurements. Model/index persistent disk deltas and device memory: N/A, not measured. Backend runs did not attest source-weight/tokenizer hashes before and after every process. The compiled test program was frozen; experimental changes are removed after measurement.

## FP64 scorer validation

| Metric | Current FP32 | ORT CPU FP64 | Native SIMD FP64 | Absolute delta (both) | Relative delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50k recall@10, original f32 oracle | 0.9960 | 0.9960 | 0.9960 | 0 | 0% |
| 50k recall@10, stored FP16 oracle | 0.9980 | 0.9980 | 0.9980 | 0 | 0% |
| Full XQuAD same_language nDCG@10 | 0.8694455 | 0.8694455 | 0.8694455 | 0 | 0% |
| Full XQuAD cross_lingual nDCG@10 | 0.7271936 | 0.7271936 | 0.7271936 | 0 | 0% |
| Full MIRACL-Swahili nDCG@10 | 0.8204219 | 0.8204219 | 0.8204219 | 0 | 0% |

### 50k vector scorer component latency

| Alternative / metric | Current FP32 | Alternative | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| ORT CPU FP64: p50 ms | 8.137 | 8.307 | +0.170 | +2.09% |
| ORT CPU FP64: p95 ms | 11.848 | 12.264 | +0.417 | +3.52% |
| Native SIMD FP64: p50 ms | 8.137 | 7.979 | -0.157 | -1.93% |
| Native SIMD FP64: p95 ms | 11.848 | 11.275 | -0.573 | -4.84% |

This times ProjectionIndex::recall_vector on one shared 50k memory index, 100 queries, three per-query rotations. Percentiles are nearest-rank over each query's three-round median. It includes the native candidate query and vector rescore, not whole search or model inference. These descriptive timings do not establish a speed gain on a shared machine; the real-corpus cache warning has a different scope.

One shared 50k/1024-dimension HNSW memory index, 100 queries and three rotated rounds; independent original-f32 and stored-FP16 cosine oracles. Full XQuAD has 1,190 queries and 13,014 documents; MIRACL has 482 queries and 131,924 documents, both at accuracy profile through search_reranked. Same native ANN candidate pools asserted for every FP64 arm. No returned-list changes; do not adopt an FP64 scorer based on the earlier synthetic near-tie probe. Real-corpus query embedding/reranker scores were shared to isolate vector math, so their call timings cannot compare inference backends. FP64 process CPU, memory and disk deltas: N/A, not measured.

## Device copies and synchronization

The retained profiling records contain only node names, providers and durations; no private paths, tensor contents or thread identifiers. profiling.py regenerates the kernel-duration fractions after removing the first eight calls per node/bucket. Most recorded time is in fused CoreML partitions. CoreML durations include prediction, wrapper handling and waits; they cannot isolate actual GPU/ANE execution or physical transfer cost. Declared intermediate tensor volumes are not measured bus bytes. Keep partition-collapse and mask-broadcast changes as future experiments with product precision and latency gates.

[ORT I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html), [pinned CoreML wrapper](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/providers/coreml/model/model.mm), [Apple compute plans](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw).

summary.json retains the aggregate values. python3 tables.py regenerates this text to stdout without modifying the evidence. Sanitized per-query rows, process resource files, provider attestations, exact experimental patch and run template are retained. python3 verify.py recomputes all backend aggregates and process-block diagnostics without modifying evidence. See manifest.json and REPRODUCE.md for identities, order, provenance limits and commands. Full device traces are excluded; minimal sanitized profiling node timings and derivation are retained.

### Steady recorded kernel duration

| Bucket | CPU kernel us | All kernel us | CPU fraction |
| --- | ---: | ---: | ---: |
| 4x128 | 49572 | 27235635 | 0.1820% |
| 4x64 | 16285 | 6043254 | 0.2695% |
| 2x256 | 10157 | 5964002 | 0.1703% |
