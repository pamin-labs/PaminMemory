# CoreML compiled-cache controls and BGE reload pilot

The direct ORT load/compile boundary is measured here, not complete product
startup, search or corpus accuracy. [Evidence](coreml-compile-cache-2026-09-29.json)
contains the source, seven completed BGE processes, tiny-model failures, raw
hashes and resource inventory. Both arms load the same typed FP16 BGE graph,
CoreML ALL, Level3 with MatMulAddFusion disabled, and 4×64 static shapes.
Both use two blocking CPU threads. The product keeps its existing configured
thread count; the pilot is not a timing of the complete product loader.

## Before / after

Three rotated rounds: before/after, after/before, before/after. The cache is
populated once before those rounds; the priming process costs 58.692 seconds.

| Metric | Recompile each load | Reuse compiled cache | Change |
| --- | ---: | ---: | ---: |
| Returned synthetic-input logits | Same four logits | Bit-identical across all seven processes | 0 changed outputs |
| ORT load time, median | 59.343 s | 2.092 s | -57.251 s (-96.5%); 28.37× |
| Sampled peak process RSS, median | 2,101,821,440 B | 1,591,590,912 B | -510,230,528 B (-24.3%) |
| Persistent compiled-cache logical bytes | 0 B | 2,278,103,629 B | +2.278 GB for this one bucket |
| Prepared ONNX graph / source bytes | Existing files | Same files | 0 changed bytes |
| Corpus accuracy / whole-search latency | Not measured | Not measured | N/A |

| Round | Before load seconds | After load seconds |
| --- | ---: | ---: |
| 1 | 63.991 | 2.092 |
| 2 | 59.343 | 0.435 |
| 3 | 55.543 | 2.112 |

Every profile confirms real CoreML kernel execution. Each process checks finite
outputs and exact logit bits against the first process. Inputs are synthetic,
identical token IDs and masks, so this does not establish retrieval quality.
RSS sampling is process-only and excludes system accelerator services. The
shared host and only three rounds limit how broadly the speed ratio transfers.

## Cache correctness

A tiny MatMul/Add/Relu model demonstrates why a single cache directory is unsafe:
loading 128 tokens after caching 64 fails with a compiled shape mismatch, and
replacing weights at the same URL returns stale values (absolute error 0.25).
Separate shape/content namespaces pass. The production regression goes through
`fixed_coreml`, loading 64/128/64, then replacing weights and repeating it.

The implementation keys the cache under the prepared model's content-addressed
parent, then by graph digest, runtime info, policy version, rows and tokens. It
serializes package construction across processes. The policy version must be
updated when compile options change. No new user configuration is introduced.

## Tradeoff and remaining gates

This reduces repeated load/compile waiting; it does not claim a 28× search
improvement or reduce the number of retained bucket sessions. First population
still compiles. Persistent disk grows; the pilot's graph/metadata inventory was
saved and its completed 2.278 GB cache retired to preserve build headroom.
The other BGE bucket sizes, natural-language fixed heads, full product reload
latency, corpus acceptance and full ignored suite remain unmeasured here.
