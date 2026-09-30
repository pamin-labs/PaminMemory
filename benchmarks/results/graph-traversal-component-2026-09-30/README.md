# Native graph traversal component reproduction

A synthetic fixture reproduced four graph traversal defects through
`Engine::search_fused`. The disabled experimental arm selected weaker arrivals
or stopped before a stronger two-hop answer. The scored experimental arm
returned the explicit edge-score oracle in every case.

This is a component invariant check. It does not establish retrieval quality,
latency, memory or disk improvements for the product. Both arms created their
own UUID projects, and some non-graph tie orders differed between them. The
within-arm candidate, rank and raw-score premises remained unchanged across
setup and graph-query phases.

| Controlled case | Disabled arm | Scored arm | Explicit oracle | Absolute score change | Score change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two origins reach one target | 0.33333334 | 0.80000001 | 0.8 | +0.46666667 | +139.999996% |
| A later, stronger arrival from one origin | 0.10000000 | 0.50000000 | 0.5 | +0.40000000 | +399.999993% |
| A stronger route at the same hop | 0.05000000 | 0.50000000 | 0.5 | +0.45000000 | +899.999985% |
| Two-hop answer after 60 one-hop decoys | Absent | 0.50000000 | 0.5 | N/A | N/A |

## Premises and score arithmetic

Each arm wrote 241 synthetic topics through the native write/drain path, then
explicitly queued `OptimizeIndex` and drained the product queue. Both asserted
241 indexed documents, vector completeness 1.0, and zero automatic graph edges.
The fixture selected 67 targets, intermediates and decoys absent from the union
of all three non-graph top-50 lists. The first phase asserted nine edges. The
second added 60 decoy edges and one certain final edge.

The graph score is final-edge confidence times `0.5^(hops-1)` times origin
relevance. The query explicitly names the strong origin, so its relevance is
1.0. The weak origin's best measured non-graph rank was **23 in the disabled
arm** and **22 in the scored arm**. At the supplied default `k = 10`, those
relevances are `11/33` and `11/32`. The disabled arm's first target therefore
scored `1.0 * 11/33 = 0.33333334`, while the better strong-origin arrival is
`0.8 * 1.0 = 0.8`. The other two path-selection oracles are
`1.0 * 0.5 * 1.0 = 0.5`; the weaker retained arrivals scored 0.1 and
`0.1 * 0.5 = 0.05`. Every target carried native graph-channel and path evidence.

The first-64-seed wording in the fixture describes its data-driven selection
premise. It selects a candidate from the graphless fused head; the native
path evidence confirms that the selected weak origin actually participated.
This check does not independently expose the engine's complete round-robin
seed list. No identical-input paired quality or timing conclusion follows.

## Source, build and actual inference

The source base is `13ee710c9df865f1dac98dc77a8108e438ddc539`. Both source copies
include the same experimental traversal patch and differ only in its scratch
compile-time constant, `false` versus `true`. `GraphRecallOptions`,
`options.depths`, `options.k` and the caller's `fusion.k()` binding are retained.
The recorded local main ref was `315c10242ddf7a1cec3bccbf550a942320e09557`;
that separate main arm was prepared but was not executed here.

The product's `accuracy` profile ran on the CPU for this controlled experiment.
The fixture calls fused search and does not load a reranker. Native assignment
reports show **1,023 CPUExecutionProvider nodes**, the same prepared graph hash
and external weight-data hash in both arms. The actual mapped ONNX Runtime
library was the pinned 1.28.0 build, reporting commit `da9b5e364c`. Native
library, prepared model, external-weight and tokenizer hashes remained stable
before and after both runs. Previously released CPU download sources are
identified separately from the prepared assets; no model was downloaded.

The build used Rust/Cargo 1.98.1, release mode, two build workers, offline locked
dependencies, no custom Rust flags/wrappers and an explicit dynamic runtime.
After a scoped release clean, Cargo reported `fresh: false` for **all four**
product path crates and both integration targets in each arm. Source hashes
were captured before compilation and checked afterward; each binary was
copied out of the build target before the next clean/build. The complete
relevant source hashes, binary digests, filtered compiler-artifact records,
runtime/provider evidence and asset hashes are in [provenance.json](provenance.json).

## Resources and limits

| Metric | Disabled arm | Scored arm | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| General retrieval accuracy | N/A | N/A | N/A | N/A |
| Product search latency p50/p95 | N/A | N/A | N/A | N/A |
| Graph-attributed memory | N/A | N/A | N/A | N/A |
| Graph-attributed disk | N/A | N/A | N/A | N/A |

No general corpus or repeated timing run was performed. The raw process usage
includes model loading, 241 writes, setup and query work. It excludes
independent PostgreSQL services from process CPU/RSS and is not an accelerator
cost measurement. Graphless probes warmed the query before graph work.

A shared file cache required reclaim. The runner recorded conservative
`max-current+inactive_file` headroom estimates, cgroup pressure and OOM counters,
and monitored memory while executing arms serially. That estimate does not
guarantee reclaim. It dropped no shared cache and changed no system limits.
OOM and OOM-kill counter deltas were zero. Both disposable PostgreSQL clones
were stopped after execution.

The unresolved 2,000-position frontier cap remains approximate; its stress test
was compiled but not run. No MuSiQue query was executed. A future matched
snapshot experiment must freeze authority/index UUIDs and exact non-graph
IDs/ranks/scores/seed relevance. Full 2,417-query MuSiQue acceptance, product
reranked cost and a separate current-main stack comparison remain necessary
before any default change. This evidence makes no default change.

Two prelaunch failures are retained in the original private evidence: a Rustup
component-download race before product compilation, then an unavailable
`/usr/bin/time` before native test launch. The successful build used explicit
installed toolchain paths; the runner used Linux `wait4` accounting. Neither
failure produced a retrieval measurement.

## Evidence and reproduction

The raw query/path rows are [baseline.jsonl](baseline.jsonl) and
[scored.jsonl](scored.jsonl); actual inference events are
[baseline.trace.jsonl](baseline.trace.jsonl) and
[scored.trace.jsonl](scored.trace.jsonl). Test logs, process-usage rows and
[comparison.json](comparison.json) accompany them. Local model/runtime paths
in the published copy use `${MODEL_CACHE}`, `${ORT_LIB}` and `${ZVEC_LIB}`.
JSON is canonicalized and empty trailing log lines are trimmed in the published copy. Original digests and published-copy
digests document the formatting and path redactions in
`provenance.json`; the original evidence was preserved.

The inert [source files](source) contain the exact native fixture, its
structured inference tracer, the experimental patch and a preparation script.
They are artifacts, not active product changes. The fixture also contains an
ignored hub test that was not selected by the recorded command.

To reproduce, create a scratch source copy at the recorded base, apply
`source/experimental-traversal.patch` with `git apply --unidiff-zero` there
(the artifact uses zero-context hunks to avoid trailing context whitespace),
and run:

```sh
python3 source/prepare.py --source "$SOURCE" --out "$ARMS"
```

Build the generated `scratch_scored_fixture` integration target. Use a prepared
workspace and pinned native runtime/model assets; clear inherited `PAMIN_*`
knobs, set `PAMIN_PROFILE=accuracy` and `PAMIN_DEVICE=cpu`, and run each copied
binary serially with `GRAPH_ARM=baseline` or `scored`, unique `GRAPH_OUT` and
`GRAPH_TRACE` paths, `GRAPH_SHARED_MACHINE` describing interference, and
`GRAPH_CLK_TCK` set to the platform clock-tick frequency:

```sh
"$BINARY" scratch_scored_graph_finite_fixture \
  --exact --ignored --nocapture --test-threads=1
```

Use separate output paths and retain every failure. Verify source/build hashes,
all four path-crate freshness records, actual provider assignment and model
asset stability before accepting the result. The read-only
[verifier](verify.py) checks the retained published evidence without compiling,
loading a model or opening a database.

The score changes above are arithmetic differences between controlled component outcomes, not product accuracy improvements. Percentages use the retained f32 channel values before rounding. An absent result has no numeric baseline or percentage.

The recorded launch projection binds the selected native test, arm, `GRAPH_OUT` and `GRAPH_TRACE` filenames; its original launch digest is retained. The historical fixture persisted its JSON directly to `GRAPH_OUT`; stdout contains test success but no same-log JSON marker. The verifier cannot establish a missing stdout linkage. Trace events record mapped library paths, not inode identities; inode equality was not captured. It independently parses the retained trace and checks exact provider, runtime and mapped-path correspondence with provenance. All published files except the provenance manifest itself have current digests in its explicit inventory; provenance is checked semantically rather than given a circular self-hash. `python3 -O verify.py` is rejected. Run `python3 test_verify.py` for mutation checks.
