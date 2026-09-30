# Native handwritten candidate-admission diagnostic

A CPU INT8 baseline/pooled pair on the repository's public synthetic handwritten corpus completed 157 queries over 230 ingested documents. Candidate sets change on 29 queries; no group gain is established. The observed cross-lingual nDCG change is−0.000870481, with 1 win/1 loss/41 ties and native raw/family p=1. Recall@10/@50 is unchanged in every group. This evidence does not warrant default adoption or establish general product quality.

The frozen variants derive from `13ee710c9df865f1dac98dc77a8108e438ddc539`, differ only in the private candidate-pool toggle, and measure actual `Engine::search_reranked` at Accuracy. This is not a combined-stack comparison with current main. Both arms retain the same full fused lists and per-query selected budget. CPUExecutionProvider assignments in raw logs are 1023 embedding nodes and 295 reranker nodes, with matching prepared graph, external weight and runtime hashes before/after. Models are BGE-M3 INT8 and BGE-reranker-v2-m3 INT8, dynamic ORT 1.28.0. Apple export/backend/precision is unverified by this CPU evidence; existing accelerator policies are unchanged.

| Group / metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| monolingual / nDCG@10 | 0.982141762 | 0.982141762 | +0.000000000 | +0.000000% |
| monolingual / recall@10 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| monolingual / recall@50 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| cross_lingual / nDCG@10 | 0.866870645 | 0.866000164 | -0.000870481 | -0.100416% |
| cross_lingual / recall@10 | 0.862790698 | 0.862790698 | +0.000000000 | +0.000000% |
| cross_lingual / recall@50 | 0.977906977 | 0.977906977 | +0.000000000 | +0.000000% |
| lexical / nDCG@10 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| lexical / recall@10 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| lexical / recall@50 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| relational / nDCG@10 | 0.704743803 | 0.704743803 | +0.000000000 | +0.000000% |
| relational / recall@10 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| relational / recall@50 | 1.000000000 | 1.000000000 | +0.000000000 | +0.000000% |
| Offered candidates/search | 30–31 | 30–31 | 0 | 0% |
| Search latency p50/p95 | N/A | N/A | N/A | N/A |
| Wall / CPU user-system time | N/A | N/A | N/A | N/A |
| Process RSS / service-device memory | N/A | N/A | N/A | N/A |
| Model-cache / index / temporary disk | N/A | N/A | N/A | N/A |

Absolute differences use after−before; percentages use100×difference/before. Group sizes are monolingual 62, cross_lingual 43, lexical 32 and relational 20. Candidate churn is 1/21/0/7, respectively. Monolingual/lexical/relational nDCG each has zero wins/losses; recall in the group/metric cells is retained in [metrics.json](metrics.json) and [summary.json.gz](summary.json.gz).

Both arms select 30 candidates on 155 queries and 31 on query IDs 1 and 62. The native main head remains 30; native graph-only extras retain threshold 0.5/cap 30. Query 1 adds schema migration window at fused rank 43; query 62 adds rollout control_en at rank 38; both have graph score 0.5. Pooled non-graph rank-one reserves replace main-head slots and preserve these extras. [Full raw traces](baseline.jsonl.gz) and [pooled traces](pooled.jsonl.gz) retain every selected position, topic ID, score and Why entry. The actual limit10 output equals each fullwide result prefix. The harness captures native rerankable selection and checks its ID set against available Why::Reranked entries. Available scores may come from cache; candidate count does not measure fresh inference or padded batch work.

[Changed cases](changed-ndcg-cases.json.gz) retain full relevant scores. Japanese feature-flag query 80 admits incorrect office_wifi_ja, while relevant feature_flag_ru moves 8→7 with a changed INT8 score. Korean rate-limit query 101 admits incorrect customer_support_sla_ko/laptop_replacement_ko, and relevant English drops 2→3 with an unchanged available score. Shared candidates can receive different INT8 scores under changed composition; this pair does not isolate a ranking-only mechanism.

Seed and both stopped-clone arms retain 230 live documents,11 mentions edges and vector completeness 0 before/after: the native small-corpus flat buffer, not HNSW coverage 1. No explicit optimize/reindex or 25,000 document threshold change occurred. Identity records link to raw seed/baseline/pooled logs. The controller records one native seed then sequential baseline/pooled clones, warmed shared OS/model caches and no parallel build/model experiment. Hardware topology and whole-search cost instrumentation were not captured for this own-corpus pair. Libtest elapsed times include diagnostics and are not product latency observations; p50/p95, CPU time, process/service/device memory, model-cache/index/temp disk remain N/A.

## Provenance limits

**Historical inherited tuning state is UNKNOWN.** The controller set CPU/profile/home and rejected HF_HOME/depth override but did not capture/reject every other tuning knob before launch. A later observer shell record cannot attest either child's historical environment; child /proc inspection was denied. Matching providers/fused lists/offered budgets is observed evidence, not certification of exact shipped defaults. See [configuration limit](historical-config-limit.md) and [observer record](config-audit-limit.json).

[Binary identities](binaries.json), original compiler/build logs, [source identities](sources.json) and [provenance](provenance.json) retain evidence of two clean release builds and their scratch test target source hashes. The identical test-source hash is 69f2f02d8fd3b66c50406899c99bddf425c5095b5e4ac1cb95bacdc6387e4119. These records do not prospectively bind every transitive build input, build environment or runtime knob to the measured binaries. Current retained source hashes cannot retro-certify historical build inputs. Future execution needs explicit safe environment capture/rejection and a prospective source/toolchain/lockfile/compiler-output/binary/config attestation.

The initial seed wrote 230 documents then failed an incorrect completeness 1 assertion; [failure log](initial-seed-failure.log.gz) and inert failed harness are retained. Its owned PostgreSQL was stopped. The corrected flat-buffer seed and baseline completed; an incorrect parser then missed a libtest-prefixed graph marker and aborted after baseline's 157 rows. [Parser failure](parser-failure.log.gz) and the prior controller are retained. The repaired controller revalidated original seed/baseline logs/assets and explicitly resumed only pooled. Successful raw logs were not overwritten. All owned PostgreSQL instances were stopped.

[Native statistics](native-statistics.json) reuses the repository compare/family implementation: deterministic 10,000 draw Monte Carlo paired sign flips, not exhaustive enumeration. Twelve fixed group×metric cells are aligned over all 157 query IDs with zero differences outside each group, preserving shared sign pairing. Raw compare uses group-specific rows. With only two opposing nonzero differences, p=1 also follows from the four possible signs. These sparse results establish neither a gain nor equivalence/noninferiority. Retained ordering instructions preceded the workflow, but archive creation does not independently prove preregistration timing.

## Inspect without running inference

```sh
python3 benchmarks/results/fusion/own-admission-2026-09-30/verify.py
python3 benchmarks/results/fusion/own-admission-2026-09-30/negative-fixtures.py
```

The portable read-only verifier checks archive and sanitized payload hashes, original raw-log→JSONL equality, native public fixture hashes, all 157 paired IDs/gold/group keys, complete fused equality, native main+graph selection, available Why IDs,30–31 budgets, actual result prefixes, coverage 0/11 mentions, raw provider graph/node identities, native Scores arithmetic, before/after deltas/percentages, changed-case details and 12 native Monte Carlo cells. Negative fixtures bypass only outer hash checks to exercise semantic checks; normal verification never bypasses hashes. Both programs refuse -O/PYTHONOPTIMIZE. No model, database, compiled binary or compiler is invoked.

[Sanitization map](sanitization.json) records SHA256 of each original payload, path-only sanitized payload and compressed archive bytes. Literal local path prefixes become aliases; ranking/score/query/gold fields are unchanged. The package contains only public repository fixtures/sources and derived diagnostic records. No database credentials/files, homes, binaries, model blobs, model symlinks or private planning notes are published. [Inert source package](../../../harnesses/fusion-own-admission-2026-09-30/README.md) retains the native loader/ingestion/scoring/statistics modules and variant sources. It is separate from the 24 query/18,001 topic [identifier fixture](../admission-2026-09-30/README.md), whose HNSW premise and 30-only budget do not apply here.
