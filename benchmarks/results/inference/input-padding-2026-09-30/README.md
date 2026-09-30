# Input preparation: fewer allocations, no demonstrated rank speed gain

Reference runtime: `1d7c1ab`; measured candidate: `935715d`, retained on
`codex/bench-input-padding-2026-09-30`. The restacked PR uses identical runtime
code. The refactor fills the two model input tensors directly rather than
cloning token text, offsets and word mappings into static padded Encodings.
The optional token-type column has the same differential input test.

## Measured allocation requests

The release observer loads the cached tokenizer, compares the actual new helper
with the retained legacy input-building code, and asserts identical tensor
values. Only the test thread's allocation/reallocation requests during input
preparation are counted. This is cumulative requested memory, not live memory,
allocator overhead or process/model RSS; no timing claim comes from the observer.

| Static batch | Before calls | After calls | Before requested bytes | After requested bytes |
| --- | ---: | ---: | ---: | ---: |
| Short, 1 logical row → 4×64 | 303 | 2 | 27,676 | 4,096 |
| Short, 4 logical rows → 4×64 | 323 | 2 | 30,684 | 4,096 |
| Medium, 3 logical rows → 4×128 | 573 | 2 | 81,956 | 8,192 |
| Long, 1 logical row → 2×256 | 1,064 | 2 | 86,980 | 8,192 |

Requested bytes fall 85.2%–90.6% in these four cases. [Raw counts](allocations.json)
and the [temporary observer](sources/allocation-observer.rs) are retained; the
observer is removed from production/test sources after capture.

## Actual scoring-stage control

Three sequential rotated rounds run the real `Reranker::rank` on 60 XQuAD-R
queries, selected at indices 0,20,...,1180 from the fixed 1,190-row input, with
30 candidates each. Order is before/after, after/before, before/after.
All 5,400 candidate scores per arm are exactly equal across every process;
batches and padded-token counts match. Each arm asserts 30 newly scored pairs
per query so a score-cache hit fails the experiment. Warmups reach all three
static buckets before timing. ORT reports 765 CoreML and 5 CPU nodes in each
session; this does not attest CoreML's internal GPU/ANE/CPU execution.

| Metric | Before | Candidate |
| --- | ---: | ---: |
| Rank p50, per-query median over three rounds | 0.739786 s | 0.729956 s |
| Rank p95, same scope | 0.951109 s | 0.968910 s |
| Median 60-query sum of rank-call wall time | 42.993843 s | 43.955873 s |
| Median whole-process CPU user+system | 9.94 s | 10.37 s |
| Median peak process RSS | 2,230,321,152 B | 2,231,648,256 B |
| Whole search / model-only steady RSS / isolated total disk | Not measured | Not measured |

Candidate/before paired geometric timing ratio is 1.016012, p=0.2275 from
19,999 query-paired log-ratio sign flips (seed 0), with 23/60 queries faster.
Percentiles use nearest rank on each query's median. This supports no stable
rank-speed improvement. The boundary is rank only, not `Engine::search_reranked`.
The host was a shared Apple M4, Mac16,12, 10 logical CPUs, 32 GiB RAM.
Process CPU/RSS include load and warmup but exclude CoreML service/device costs.
The first process round was substantially colder than later rounds, so do not
interpret whole-process differences as a model memory or energy improvement.
Persistent models/index formats and their weights are unchanged; total disk
was not isolated. The decision is allocation simplification, not search speed.

## Provenance and reproduction

[Manifest](manifest.json) records runtime trees, frozen executable hashes,
input hash, arm order, source/graph/native-library identities, and original versus
redacted artifact hashes. The four retained tokenizer files and FP16/native
graph bytes were checked before and after every process. The active hub ref was
not recorded in those six processes: this is retained-cache provenance, not a
per-load tokenizer identity log. All observed scores/shapes still agree exactly.

All six raw JSON row files, stdout and resource logs are retained beside this
page. Local paths become `${REPO}`, `${EVAL_HOME}`, `${RUN_ROOT}` and `${INPUT_FILE}`.
The actual generating [rank harness](sources/rank.rs), [controller](sources/run.py.in),
[summary](sources/summarize.py.in), and [runtime patch](sources/candidate.patch)
are retained. Templates change local paths only. To rerun, expand their path
placeholders, decompress the input, build the ignored rank harness in separate
reference/candidate checkouts, and copy their executables to `RUN_ROOT/baseline`
and `RUN_ROOT/candidate`. Record the fresh executable hashes in the controller's
`expected` map; historical executable hashes cannot attest a new build.
The same hashed zvec library must be loaded by both arms. The controller sets
its DYLD environment after macOS `/usr/bin/time` starts so SIP does not remove
it, records the actual loaded path, and guards ≥12 GiB disk and ≥25% free memory.

The rank harness is copied to `crates/pamin-engine/tests/scratch_padding_rank.rs`
and built with `cargo test --release -p pamin-engine --test scratch_padding_rank --no-run`.
Remove stale scratch copies first and use an EXIT cleanup trap when reproducing.
For allocation capture, temporarily append the observer to `encoder.rs` on the
candidate, run `cargo test --release -p pamin-index -p pamin-engine --lib
scratch_padding_allocations -- --ignored --nocapture` with `PAMIN_EVAL_HOME` and
`PAD_ALLOC_OUTPUT`, then restore the backed-up source via an EXIT trap.
Do not commit the observer or run it as ordinary CI instrumentation.

Run `python3 benchmarks/results/inference/input-padding-2026-09-30/verify.py`
to check artifact hashes, complete paired scores/shapes, the exact input and
allocation counts, and regenerate the reported statistics.

## Input data licence

`xquad-fixed-candidates.json.gz` is a retrieval adaptation of
[XQuAD by Artetxe, Ruder and Yogatama](https://github.com/google-deepmind/xquad),
with sentence extraction, candidate lists and model/fusion scores added by
PaminMemory. This data artifact is distributed under **CC BY-SA 4.0**,
not the repository's code licence; the [original licence](DATA-LICENSE.txt) is
included. Decompressed bytes SHA-256 is
`cd17aba46410010832ce681008bd22a93e11d0a2b2cc1325f4b7a2e6328d7ac5`.
