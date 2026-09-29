# Apple Fast reranker: CPU INT8 versus CoreML NeuralNetwork

At public code commit `bd523771551874fb47f38c70c2cacb6b01bc7b06`,
frozen release binaries called `Engine::search_reranked` on the same complete
13,014-document, revision-bound XQuAD-R memory index. Both used the default
`accuracy` embedding profile, memory vector index, Fast rerank head 30,
return limit 60, and the same 1,190 query order. The host was shared Apple
Mac16,12 (10 logical CPUs, 32 GiB RAM, macOS 26.6.2). The model was the pinned
[`mmarco-mMiniLMv2-L12-H384-v1`](../retrieval/xquad-r-fast-model-artifacts.json):
ARM INT8 on ONNX Runtime's optimized CPU EP against FP32 on CoreML's legacy
NeuralNetwork format with `ALL` compute units. The Accurate tier remains the
default and was not changed by this choice.

The CPU arm and its Off control completed in one process. The CoreML arm was
stopped by the free-disk guard after query 770. A second frozen CoreML process
ran queries 760–1,189; all ten overlapping query IDs, complete rankings, and
both nDCG values matched exactly. The merged file therefore has complete
**ranking and accuracy coverage**. Its per-query timings come from two
processes, so they are descriptive, not a clean single-process speed ratio.
The disk-volume swing has no established CoreML cause; the 99 temporary
directories owned by the stopped process allocated about 389 MB in total and
were all born at startup.

| Measure | Before: Fast CoreML FP32 | After: Fast CPU INT8 | Change |
| --- | ---: | ---: | ---: |
| Cross-language nDCG@10 | 0.656096 | 0.657172 | +0.001077 (+0.16%) |
| Same-language nDCG@10 | 0.841400 | 0.841533 | +0.000133 (+0.02%) |
| Cross-language recall@50 | 0.903193 | 0.903193 | 0 |
| Same-language recall@50 | 0.964706 | 0.964706 | 0 |
| Whole-search per-query p50 | 1.629954 s, stitched | 0.330035 s | −1.299919 s (−79.75%), descriptive |
| Whole-search per-query p95 | 2.833080 s, stitched | 0.769942 s | −2.063139 s (−72.82%), descriptive |
| Sampled peak benchmark-process RSS | 4,452,089,856 B before interruption | 1,304,920,064 B, full CPU+Off arm | Different process scopes; no accepted percentage |
| Selected model source logical bytes | 470,883,696 B | 118,620,017 B | −352,263,679 B (−74.81%) |
| Total model/index/temporary disk | Not isolated | Not isolated | N/A |

The scorer's paragraph-cluster sign-flip comparison gives CPU minus CoreML
cross-language nDCG `+0.001077` with 167 query wins, 137 losses, 886 ties,
and four-metric family-adjusted `p=0.1558`. Same-language is `+0.000133`,
5 wins, 7 losses, 1,178 ties, adjusted `p=0.9609`. Neither difference is a
significant quality gain or loss; recall@50 is identical query by query. The
CPU arm's Off control scored cross-language `0.637232` and same-language
`0.843806`, so Fast itself still has a same-language cost. The Accurate tier
remains the precision-first default.

CPU search calls were faster on **all 1,190 paired queries** in this shared-host
run. The p50 ratio is 4.94× and p95 ratio 3.68× as a descriptive comparison;
they are not an isolated hardware-throughput or energy estimate. The CPU arm's
whole harness took 532.945 s including Fast and Off; the first CoreML process
spent 1,440.667 s on only 770 Fast queries before termination, and the
second spent 632.559 s on 430 Fast queries. Those totals are not compared as
equivalent work. CPU process time excludes CoreML services and device work;
CoreML EP assignment (574 nodes, with 246 on CPU EP) does not reveal its
internal GPU/ANE placement. ARM INT8's actual CPU graph had 528 CPU EP nodes
and ran with ONNX Runtime's `neon,fp16,dotprod,i8mm,bf16` build.

This result selects optimized CPU for **Apple Silicon Fast** under the shared
fastest-viable-backend policy. It does not turn off CoreML for Accurate, change
stored embedding vectors, or require reindexing. Other platforms' Fast backend
selection needs their own same-model measurement.

## Reproduction and limits

[Both complete ranking/timing row sets](fast-apple-backend-2026-09-30-cpu-rows.json)
and [CoreML rows](fast-apple-backend-2026-09-30-coreml-rows.json), the
[paired four-metric comparison](fast-apple-backend-2026-09-30-paired.json),
[rejected first CoreML prefix](fast-apple-backend-2026-09-30-coreml-rejected-prefix.json),
[complete resume rows](fast-apple-backend-2026-09-30-coreml-resume-rows.json),
[first-arm resource status](fast-apple-backend-2026-09-30-arm-status.json),
[resume status](fast-apple-backend-2026-09-30-resume-status.json), and
[source and provider fingerprints](../../harnesses/fast-apple-2026-09-30/sources.json)
retain the accepted and rejected boundaries. The [harness archive](../../harnesses/fast-apple-2026-09-30/README.md)
contains the exact source patches, lockfile, resource guards and overlap rule.
It reuses the repository's [scorer and comparison owners](../../harnesses/product-search-2026-09-30/README.md)
and the existing [XQuAD-R judgments](coreml-search-full-2026-09-30-judgments.json),
whose parsed content was checked equal to the input used here.

The real-model Apple auto-route test passed with the pinned ARM INT8 export.
`cargo fmt --all --check`, all-target/all-feature Clippy with warnings denied,
`cargo test --workspace`, and `python3 ci/budget.py` passed on this code tree.
Native Windows/Ubuntu CI, bot review, rotated performance repeats, other
corpora, service memory and total disk remain open.
