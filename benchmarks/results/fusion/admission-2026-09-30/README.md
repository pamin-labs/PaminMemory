# A fusion-to-reranker admission gap

Three identifier queries in the matched restart fixture find their known target in lexical recall but leave it outside Accurate's 30-candidate main shortlist. The complete fused union retains it at ranks34,35,37. This is an admission gap, not missing n-gram recall within the shipped window or a demonstrated model precision problem.

An experimental shortlist reserves the live rank-one candidate from each effective non-graph channel, deduplicates shared winners, and fills remaining slots from the original fused head. Graph-only extras, weights, score blend, placement and model are unchanged. The candidate is a scratch literal-true variant. Product source and defaults remain unchanged pending independent multilingual acceptance.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Known-target recall@10 | 0.875000 | 1.000000 | +0.125000 | +14.286% |
| Known-target MRR@10 | 0.826389 | 0.902778 | +0.076389 | +9.244% |
| Offered reranker candidates/search | 30 | 30 | 0 | 0% |
| Selected targets | 21/24 | 24/24 | +3 | +14.286% |
| Search p50/p95 | N/A | N/A | N/A | N/A |
| Process/device memory | N/A | N/A | N/A | N/A |
| Model/index/cache disk | N/A | N/A | N/A | N/A |

This is a single sequential baseline/pooled process pair on 24 near-duplicate synthetic identifier queries and18001 indexed live topics, using the actual `Engine::search_reranked` path. No general accuracy, significance or latency claim follows. Ten queries change scorer candidates; seven of those were already retrieved. Their incorrect lexical winners are retained in raw evidence, even though no known target falls out of the top ten. Other returned documents have no relevance judgments. The three recovered target ranks are35→3,34→2,37→1. Recall's absolute difference is0.125, or12.5 percentage points.

## Conditions and reproduction

Both source variants derive from `13ee710c9df865f1dac98dc77a8108e438ddc539` and differ only in a private shortlist boolean. The false path preserves predecessor selection; this is not a measured combined-stack comparison against current main. The fixture is a stopped copy of the actual main-reference HNSW restart trial, including its visible `restart-proof` write. Both arms use new copies, accuracy profile, channel depth50 and reranker depth30. All24 complete fused lists match exactly between arms; the Why::Reranked IDs match the selected30 inputs and confirm available scores, including cache reuse. The limit200 diagnostic prefix equals the product limit10 output.

Actual CPUExecutionProvider assignments are asserted for the BGE-M3 INT8 embedding and BGE-v2-m3 INT8 reranker, with identical prepared graph hashes. Runtime is dynamic ONNX Runtime1.28.0 and zvec-rust0.7.2; warmed downloaded/prepared model files are shared. [Pretrial asset hashes and hardware](pretrial-assets.json) supplement the runner provenance. Prepared graphs were hashed after each arm; external weight files were hashed before the pair and were not independently rehashed afterward. This Linux CPU comparison does not certify Apple's different Accurate CoreML export. It does not change Apple CoreML ALL or Fast ORT CPU policy.

Run `python3 benchmarks/results/fusion/admission-2026-09-30/verify.py` to validate archive hashes, matched full traces, selected-candidate budget, actual providers and independent metric recomputation. [Raw traces](baseline.jsonl.gz), [candidate traces](pooled.jsonl.gz), compressed logs and [per-query summary](paired-summary.json) preserve complete candidate choices, final ranks and scores. Only local path prefixes are replaced; ranking/score fields are unchanged. Originals remain in the scratch experiment. No database credentials, database files or private inputs are published.

[Inert source archive](../../../harnesses/fusion-admission-2026-09-30/README.md) describes rebuilding the frozen variants and fixture. Timing, CPU time, RSS, service/device memory and disk were not collected for this diagnostic. Each query is searched at limit10 and replayed at limit200; the latter may reuse cached scores. Selected/offered candidate count does not measure fresh model computations or padded batch work and does not establish equal latency or memory.
