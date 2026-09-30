# Reranker batch-context cache component evidence

On two public native-corpus queries, the predecessor's pair-only cache returns different logits/ranks for a final list depending on whether another list ran first. Caching complete ordered batches removes all recorded same-list history differences. Both fresh variants preserve the original implementation's score bits and raw ranks. This is numerical/cache mechanism evidence, not broad retrieval quality or product cost acceptance.

Before is the planner refactor at measured `d8536d0ce97170298d20e731109fe5cae583dc09`; after is the cache fix at measured `8aad14bbd98751f5da7fce385c0cb7cd0e933fe0`. Original, pre-refactor fresh results are retained in [original-oracle.json](original-oracle.json) and backed by the earlier [strict raw trace](original/raw.jsonl.gz). Historical commit identifiers document the source lineage; verification needs no Git objects. Exact product source and measured appended harness source are archived locally under sources/. Review commits named in [provenance.json](provenance.json) were rebased; the review fix adds three empty-string test assertions to the measured fix and leaves its production implementation unchanged. Later stack hashes do not replace measured identities.

| Component metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| q80 baseline->pooled / different score bits | 4 | 0 | -4 | -100% |
| q80 baseline->pooled / different raw ranks | 8 | 0 | -8 | -100% |
| q80 baseline->pooled / new forward logical pairs | 1 | 4 | 3 | 300% |
| q80 pooled->baseline / different score bits | 4 | 0 | -4 | -100% |
| q80 pooled->baseline / different raw ranks | 6 | 0 | -6 | -100% |
| q80 pooled->baseline / new forward logical pairs | 1 | 4 | 3 | 300% |
| q101 baseline->pooled / different score bits | 16 | 0 | -16 | -100% |
| q101 baseline->pooled / different raw ranks | 17 | 0 | -17 | -100% |
| q101 baseline->pooled / new forward logical pairs | 2 | 16 | 14 | 700% |
| q101 pooled->baseline / different score bits | 16 | 0 | -16 | -100% |
| q101 pooled->baseline / different raw ranks | 19 | 0 | -19 | -100% |
| q101 pooled->baseline / new forward logical pairs | 2 | 16 | 14 | 700% |
| All 16 hot/new_forward_pairs | 0 | 0 | 0 | N/A |
| Hot component median wall us (8 calls) | 11 | 1630.5 | 1619.5 | 14722.727273% |
| Component rank-call wall sum us (24 calls) | 13288817 | 15435146 | 2146329 | 16.151393% |
| Child component final HWM KiB | 696172 | 696048 | -124 | -0.017812% |
| Child aggregate CPU seconds (unequal fresh work) | 42.54 | 47.98 | 5.44 | 12.787964% |
| Engine p50/p95 latency | N/A | N/A | N/A | N/A |
| Broad nDCG/recall | N/A | N/A | N/A | N/A |
| Whole Engine-service-device memory | N/A | N/A | N/A | N/A |
| Model-index-cache-temp disk | N/A | N/A | N/A | N/A |

Differences are after−before; percentages are 100×difference/before, with N/A for a zero denominator or unmeasured quantity. Each list offers 30 logical pairs. A/B are saved baseline/pooled candidate lists from an experimental admission rule, used solely as context-sensitive model inputs. This proof does not enable or justify that rule. It changes no selection, model precision, execution-provider policy, score blending or fusion default.

The archive contains 86 actual JSON records across the two variants, including 48 rank calls:16 fresh calls,16 immediate hot repeats and16 history calls. [All 240 comparisons](complete-same-list-comparisons.json) join all 30 positions of each second history call to the fresh reference for that exact final list, for 2 variants×2 queries×2 directions. These are raw model ranks, not final Engine blend ranks. Different final lists may legitimately produce different scores; the checked invariant is the same final list scored fresh versus after history. All 16 hot repeats have zero new forward pairs/tokens/padding/batches/forward time and preserve fresh scores bitwise.

Correct changed-history work is larger: q80 reexecutes 4 logical pairs rather than 1, and q101 reexecutes 16 rather than 2. Exact hot lists now retokenize all 30 offered pairs. In this one sequential shared-host component pass, hot call times range 1.455–4.077 ms after versus 0.010–0.041 ms before. The descriptive median and aggregate component timings above are not Engine p50/p95 or a replicated latency comparison. The two processes perform different amounts of valid forward work; aggregate CPU includes child runtime threads, load and diagnostics, and cannot establish a throughput gain. HWM is child process peak resident memory, including model loading/inference and measurement allocations; it excludes parent/file cache, PostgreSQL and accelerator services. Its small observed difference does not isolate cache memory or demonstrate a memory gain. Whole Engine costs, device/service memory and model/index/cache/temp disk usage remain N/A.

## Model, configuration and source binding

Both variants execute the existing BGE-reranker-v2-m3 INT8 CPU model revision 6f5ff65298512715a1e669753bc754d2bc8f367b using dynamic ORT 1.28.0. Raw provider logs assert 295 CPUExecutionProvider nodes in attention.onnx, SHA256 24cc5ad23811a65c6a4648d8be0b79a7ad67a180e2db234b08c899c882ac5164. Raw loaded-library records tie the actual ORT/native library paths to retained hashes; prepared graph, external weights, tokenizer/config and runtime/native hashes match before/after both arms. Model source export was already absent because prepared::release had removed it previously; this pair asserts absent→absent and does not pretend to hash an unavailable model file.

Launch records capture an explicit safe environment, CPU device and offline model use. No inherited PAMIN tuning remains; the loaded model reports truncation 256, and recorded complete plans retain four-pair/512 padded-token defaults. Clean build records assert fresh compilation of all four path crates, freeze index and Engine test binaries independently for each variant, retain exact compiler artifacts and source inventories, and assert unchanged source/binary identities around execution. Four test binaries were built, but only the two index binaries ran this component proof; building Engine does not establish Engine accuracy or cost acceptance. The verifier checks the archived build/launch/source records and emitted artifact identities; it does not rebuild binaries, independently prove the build machine's behavior, or turn declarative historical records into a cryptographic build attestation.

The original oracle run's whole-asset guard failed because source export changed present→absent through the documented loader release policy. [Qualified postvalidation](original/qualified-postvalidation.json.gz) preserves that failure and separately records unchanged executed graph/weights/tokenizers/runtime plus the cleanup record. It is not retroactively marked as a successful whole-asset guard. The new refactor/fix pair begins with the source export absent and records unchanged executed assets throughout. Original oracle qualification, raw logs and launch/provider records remain inspectable.

[ONNX DynamicQuantizeLinear](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html) specifies scalar per-tensor quantization parameters; [ORT quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) describes dynamic activation parameters computed at runtime. These references explain why neighboring rows/padding can affect this existing INT8 execution. The causal tests establish the observed behavior for these inputs; they do not introduce a new quantization setting or require cross-input equal logits. Apple backend/precision and broad retrieval quality remain unmeasured.

## Inspect the standalone package

```sh
python3 verify.py
python3 negative-fixtures.py
```

Run these commands from this directory, or pass absolute script paths from elsewhere. The read-only verifier requires every declared package member, checks compressed/payload/original hashes, parses actual raw/provider logs, ties the public fixture to its exact input, reconstructs all 240 joined comparisons, validates f32 bits/rank order and all 16 hot repeats, derives fresh/changed-batch work from complete plans, checks fresh original/refactor/fix parity and recomputes every table delta/percentage. It uses no models, databases, compiled binaries, Git objects or compiler. Negative fixtures modify temporary copies to check semantic corruption, omitted members, archive hashes and optimized-Python refusal. Both scripts refuse -O/PYTHONOPTIMIZE.

The original eight-member component export is expanded here with raw metadata/provider logs, safe launch records, asset identities, original oracle evidence, source/build bindings and inert .in.gz measurement sources. [Sanitization](sanitization.json) preserves original and sanitized SHA256 identities through reversible literal local-path substitutions; score bits/ranks/query text are unchanged. README and manifest are newly authored package files. No credentials, binary/model payloads, private planning contents, database data or workspace homes are included. Historical plan hashes remain provenance references without publishing private plan contents. Inert measurement sources are archival evidence, not prospective safe runners: their original assertion-based scripts lack optimization guards. New execution requires coordinated resources and safe launch/build capture; verification authorizes no inference.
