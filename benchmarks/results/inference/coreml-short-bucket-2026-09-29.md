# CoreML short bucket: matched fixed-head comparison

This measures `Reranker::rank` with product score fusion/placement on the same
1,190 XQuAD-R heads of 30 candidates. It is **not complete search latency** or
other-corpus acceptance. [Machine-readable evidence](coreml-short-bucket-2026-09-29.json)
contains all query timing rows, run hashes, source, the control patch and limits.

Both arms use CoreML ALL, original FP16 encoder weights, the FP32 classifier,
cap 256 and identical release dependencies. Both have the ARM checksum change;
the control only reverts the additional 4×64 bucket. Three rounds run in orders
control/candidate, candidate/control, control/candidate. Each bucket gets eight
separate one-pair warmups with asserted physical-token counters. Timers start
after the resource-inventory barrier. All six runs exit 0 without guard rejection.

## Results, accuracy then latency then memory then disk

Latency/resource columns use the median of three round summaries, except paired
query latency, which uses each query's median across the three repetitions.

| Metric | 4×128 / 2×256 control | Additional lazy 4×64 | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Cross-language nDCG@10 | 0.727206402 | 0.727206402 | 0 | 0% |
| Same-language nDCG@10 | 0.869445521 | 0.869445521 | 0 | 0% |
| Cross-language recall@50 | 0.903193277 | 0.903193277 | 0 | 0% |
| Same-language recall@50 | 0.964705882 | 0.964705882 | 0 | 0% |
| Uncached rank p50 | 616.875 ms | 531.703 ms | -85.172 ms | -13.8% |
| Uncached rank p95 | 857.948 ms | 782.844 ms | -75.104 ms | -8.8% |
| Physical padded tokens per run | 5,037,568 | 4,233,472 | -804,096 | -16.0% |
| Sampled peak process RSS | 2.578 GiB | 2.859 GiB | +0.281 GiB | +10.9% |
| Sampled warm process RSS median | 0.307 GiB | 0.770 GiB | +0.463 GiB | +150.7% |
| PID-owned temporary-package logical bytes | 4.558 GB | 6.836 GB | +2.278 GB | +50.0% |
| Shared model-cache folder logical bytes | 9.652 GB | 9.652 GB | 0 | 0% |

Every returned logit and ranking matches across all six runs: zero changed
queries, not just cancellation in aggregate accuracy. Each arm has the same
1,189 fully uncached queries; the remaining query's cache condition also matches.

| Round | Control p50 / p95, ms | Candidate p50 / p95, ms | Candidate/control ratios |
| --- | ---: | ---: | ---: |
| 1 | 616.875 / 836.043 | 531.703 / 782.844 | 0.862 / 0.936 |
| 2 | 643.995 / 907.157 | 544.981 / 806.844 | 0.846 / 0.889 |
| 3 | 589.409 / 857.948 | 528.902 / 779.473 | 0.897 / 0.909 |

Per-query median latency ratio is 0.8600; geometric mean ratio is 0.85745.
The existing project randomization test, with all questions in each of 240
paragraph clusters flipped together, gives p=0.0001. Repetitions are averaged
within the query; they are not three times as many independent observations.
The direction agrees in all three rounds, but the effect size is host/workload
specific, and significance does not remove the limitations below.

## Limits and decision scope

- Shared machine; disk-preflight pauses occurred between arms in the first and
  final pair. Frozen binaries, input fingerprint and order were retained on
  resumption. Report all round values rather than hiding this variation.
- RSS is the sampled process, including startup for the peak, not all memory
  in CoreML/ANE system services or the entire machine.
- Package inventories include source and compiled ORT temporary packages,
  selected by the live PID and creation time. They exclude global ANE caches.
  Logical/stat-block totals are not unique APFS physical extent consumption.
  Normal process exit released the observed temporary directories.
- The shared cache folder includes other cached models. Its zero delta is not
  an isolated model-size claim. No index was rebuilt or measured by this replay.
- A metadata-only disk observer ran once per warmed arm; the first baseline's
  supplemental observation occurred after its query loop began.

The model-boundary tradeoff is measurable: preserved outputs and lower latency
at higher residency and temporary storage. Under the requested ordering this
supports continuing the implementation to complete search/corpus acceptance;
it does not declare all of #142/#145 or the broader optimization goal complete.

## Review correction (2026-09-30)

The original warmup only asserted padded-token totals. Both 4×128 and 2×256
produce 512, so that guard did not independently distinguish those shapes.
The original source remains unchanged in JSON; a revised source adds logical
token intervals (short 1–64, medium 65–128, long 129–256) for new runs. Existing
timing is conditional on the inferred warmup shape, not a recorded shape trace.

A later [real-model warmup probe](../../harnesses/coreml-short-bucket-2026-09-29/README.md)
checked the retained tokenizer bytes at SHA256
`8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3`
and ran the original short/medium/long strings with all eight warmup queries.
Every call scored a new pair on CoreML: short had 16–17 logical tokens and
256 padded tokens (4×64), medium 91–92 and 512 (4×128), long 256 and 512
(2×256). This distinguishes medium from long despite equal physical-token
totals, and a changed tokenizer snapshot fails the probe's hash assertion.
It is a later same-host observation, **not** a contemporaneous shape trace
from the six timed arms; the historical timing keeps that provenance limit.

The [review supplement](coreml-short-bucket-2026-09-29-review.json) records inspection of the retained model/tokenizer snapshot and
the same host (Apple M4, Mac16,12, 10 cores, 32 GiB; OS 26.6.2 inspected on
2026-09-30). These inspections are explicitly distinguished from missing
contemporaneous model/hash/OS logs. The
[actual resource collectors and summarizer](../../harnesses/coreml-short-bucket-2026-09-29/README.md)
are retained with source hashes. Later eager bucket validation changes startup
behavior; the old six-arm observation does not measure that new startup cost.
