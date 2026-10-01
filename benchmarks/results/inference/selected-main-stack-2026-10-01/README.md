# Selected main versus frozen reranker stack — CPU search evidence

The measured stack corrects history-dependent ordered results against an independent fresh-final-input oracle. This experiment supports no stable speed gain or universal retrieval-quality improvement. The [sanitized evidence](evidence.json) contains all 80 accepted processes and 880 actual product calls. The [complete recomputed tables](tables.md) retain every observed gain, regression and eligibility exclusion.

Before is selected main `315c10242ddf7a1cec3bccbf550a942320e09557`; after is measured stack `d0b6a14a4f317fea3c1e117f627289aad1b1c964`. These immutable measured revisions precede subsequent ordinary merges; “main” is a comparison label, not a claim that current main was retimed. This combined comparison does not attribute each change's contribution. Other experiments against immediate predecessors retain their own separate scopes.

| Metric and scope | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Accurate history-failure processes, 32 per arm | 16 | 0 | -16 | -100% |
| Scenario2 A→B changed nDCG, both limits | 0.5531464700081437 | 0.48522855511632257 | -0.06791791489182109 | -12.278468466195068% |
| Scenario2 B→A changed nDCG, both limits | 0.48522855511632257 | 0.5531464700081437 | +0.06791791489182109 | +13.997097692558365% |
| Accurate limit5 scenario1 initial-hot pooled p50, ms | 2.7025 | 2.8275 | +0.125 | +4.62535% descriptive |
| Same cell descriptive p95, ms | 3.56945 | 3.87755 | +0.3081 | +8.63158% descriptive |
| Native sampled peak RSS, four-process cell mean, KiB | 1,135,029 | 1,136,469 | +1,440 | +0.126869% descriptive |
| Index logical bytes | 22,028,682 | 22,028,682 | 0 | 0% |
| Shared bound model/runtime logical bytes | 2,415,072,587 | 2,415,072,587 | 0 | 0% |
| Executable logical bytes | 16,289,008 | 16,300,208 | +11,200 | +0.068758% |
| Total service/device memory; lifetime CPU; temporary peak allocation | N/A | N/A | N/A | N/A |

Actual boundary: CPU ReadOnly `Engine::search_reranked`, 230-document Flat index, accuracy profile, Accurate limits5/10 with two primary synthetic scenarios and two context histories, plus Off controls. Four independent rotated process blocks per cell; hot quantiles pool five dependent calls from each process. Do not treat 20 hot calls as 20 independent backend repetitions. All 46 p50 and 46 descriptive p95 wall groups withhold stable eligibility because of per-arm variability, paired sign changes or paired-percentage spread; incorrect-output/different-work comparisons are additionally ineligible.

Both sources already use INT8 prepared BGE-M3 and Accurate exports with FP32 outputs. Model revisions are `2b34e84df040034d4b9eabb62383a87c18955822` and `6f5ff65298512715a1e669753bc754d2bc8f367b`; actual ORT1.28.0 CPU node assignments are embedding1023 and reranker295. AMD EPYC9V74 shared Linux host, four-CPU quota and five eligible logical CPUs. Inference thread environment was unset; effective intra-op count and during-query frequency are unmeasured. First search uses a fresh process/session, while hash preflight warmed OS pages and prepared model caches already existed. This is not disk-cold loading or Apple device evidence.

The main failure affects 16/32 Accurate processes, versus 0/32 stack processes; 64 is the combined-source total. All 32 stack history groups match their own fresh oracle, including 192 dependent changed/hot calls. Recall/MRR are unchanged in observed cells. Scenario2's corrected nDCG moves in both directions, so exact-context correctness must not be presented as universal corpus accuracy improvement. Its changed-context timing also compares different successful recomputation work and cannot establish a like-for-like cost regression.

Native RSS/HWM includes startup and diagnostics and excludes PostgreSQL; roughly1Hz sampling can miss peaks. The owned PG main PID is reported separately, excluding its children. Per-call CPU uses 100Hz ticks, and 539 calls with both deltas zero are resolution-censored. Native launch-to-exit-detection wall includes startup/diagnostics and polling, excluding PG startup/stop. CPU after final search, total service/device memory and endpoint allocated blocks remain N/A. Equal index logical bytes do not imply zero metadata writes or peak allocation savings.

Recompute all published arithmetic with the inert standard-library [verifier](verify.py):

```bash
python3 -B verify.py evidence.json
```

It reads only the supplied JSON and prints tables; it does not execute inference, archived sources, external commands or network requests. It checks schema, matrix completeness, per-call work compatibility, quantiles, block medians, eligibility, resource arithmetic and disk deltas. Private correctness/quality, source/runtime/provider execution and full-payload attestations are explicitly marked: this public artifact cannot independently certify them. Raw private traces, controllers and original gold inputs are omitted. The omitted full index payloads also remain external to the private selective trace packet.

Accepted attempt B is distinct from a preserved failed attempt A: A made13 native calls but accepted none after its archive comparison failed; B ran a fresh complete matrix. The initial offline analyzer's historical-alias failure was repaired through exact retained link/role/hash binding without rerunning the product or weakening provider checks. Private references preserve both amendments and the independent review.

The submodule update depends on private InternalDocs PR22, published but unmerged at creation. It is a valid published commit dependency, not the private repository's merged main. This public evidence contains only checked non-sensitive numeric summaries and an inert verifier.
