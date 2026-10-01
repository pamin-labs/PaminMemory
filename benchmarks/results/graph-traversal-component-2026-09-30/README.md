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
library was the pinned 1.28.0 build, reporting commit `da9b5e364c`. The provisioned Zvec file was hashed before and after, but its actual loaded
mapping was not captured: historical loaded-Zvec identity is **N/A**. ONNX Runtime
mapping, prepared model, external-weight and tokenizer hashes remained stable
before and after both runs. Previously released CPU download sources are
identified separately from the prepared assets; no model was downloaded.

The build used Rust/Cargo 1.98.1, release mode, two build workers, offline locked
dependencies, no custom Rust flags/wrappers and an explicit dynamic runtime.
After a scoped release clean, Cargo reported `fresh: false` for **all four**
product path crates and both integration targets in each arm. Source hashes
were captured before compilation and checked afterward; each binary was
copied out of the build target before the next clean/build. The original **partial 52-entry**
source hashes, binary digests, filtered compiler-artifact records,
runtime/provider evidence and asset hashes are in [provenance.json](provenance.json).

The original source inventory omitted the second integration target's source and the 14 embedded migration SQL files; their
build-time pre/post-byte attestation is **N/A**. The exact preserved
`scratch_scored_multihop.rs` source is now retained as [multihop.rs.in](source/multihop.rs.in),
with a separate dated retrospective source binding in provenance. Its two frozen
arm copies were byte-identical before copying (SHA256
`70d5127e265a475edc3ededc871dc91cd79a0bfb56a4e6feca4d1501fd206c00`).
This binds the preserved later-inspected source, not its historical compilation
bytes; the original 52-entry maps remain unchanged and partial. A dated, read-only
[retrospective SQL audit](retrospective-sql-audit.json) now confirms that every
preserved frozen SQL file matches Git at the recorded base and occurs verbatim
in both preserved binaries whose SHA256 matches the original run records.
The [SQL text](source/migrations) is retained with hashes, sizes and binary
offsets. This later payload inspection does not retroactively make the original
source inventory complete. The public verifier checks the retained audit and
SQL pins; the original binaries are not published, so it cannot independently
replay the binary payload-membership inspection. Future builds must capture all embedded SQL before
and after compilation.

## Resources and limits

| Metric | Disabled arm | Scored arm | Difference | Change |
| --- | ---: | ---: | ---: | ---: |
| General retrieval accuracy | N/A | N/A | N/A | N/A |
| Product search latency p50/p95 | N/A | N/A | N/A | N/A |
| Graph-attributed memory | N/A | N/A | N/A | N/A |
| Graph-attributed disk | N/A | N/A | N/A | N/A |

Historical CPU model, kernel, CPU quota and affinity are **N/A: not captured**.
A dated [current platform observation](platform-observation.json) records those
fields separately. Both original cgroups recorded a 16 GiB memory limit, which
matches the current observation; equality of the other historical fields is
unknown. The current four-CPU quota does not establish historical execution
capacity.

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

The inert [source files](source) contain the exact native fixture, the separately compiled multihop target, its
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

Build both generated integration targets with the installed pinned Rust/Cargo
1.98.1 tools, two build workers, explicit runtime paths and no inherited flags or
wrappers. Keep build and model work out of the run slot. The six retained
freshness records cover four product path crates and these two integration
targets; other dependency freshness is not claimed.

`ORT_LIB` and `ZVEC_LIB` are directories containing the exact retained
`libonnxruntime.so.1.28.0` and `libzvec_c_api.so` assets; resolve the archive
placeholders to provisioned directories and verify their pinned digests before
building. `--offline` only constrains Cargo and does not disable a native build
script download fallback; `ZVEC_AUTO_BUILD=0` disables that fallback. The build
clears `ORT_LIB_PATH`, which otherwise takes precedence over `ORT_LIB_LOCATION`.
For each
scratch arm:

```sh
(
set -eu
unset RUSTUP_TOOLCHAIN RUSTC RUSTDOC
TOOLCHAIN=1.98.1-x86_64-unknown-linux-gnu
test "$(rustc +"$TOOLCHAIN" -Vv)" = "$(python3 -c 'import json; print(json.load(open("provenance.json"))["toolchain"]["rustc"], end="")')"
test "$(cargo +"$TOOLCHAIN" -Vv)" = "$(python3 -c 'import json; print(json.load(open("provenance.json"))["toolchain"]["cargo"], end="")')"
ARMS=$(cd "$ARMS" && pwd -P)
cd "$ARMS/$ARM"
env -u ORT_LIB_PATH -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS \
-u RUSTC_WRAPPER -u RUSTC_WORKSPACE_WRAPPER \
ORT_LIB_LOCATION="$ORT_LIB" ORT_PREFER_DYNAMIC_LINK=1 \
ZVEC_LIB_DIR="$ZVEC_LIB" ZVEC_AUTO_BUILD=0 \
LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB" \
CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 cargo +"$TOOLCHAIN" test -p pamin-engine \
  --manifest-path "$ARMS/$ARM/Cargo.toml" \
  --test scratch_scored_fixture --test scratch_scored_multihop \
  --no-run --release --offline --locked --message-format=json
)
```

The build and launch both use the pinned dynamic-library directories. These
variables reproduce the selected native link/runtime path; they are not a
complete transitive source-to-build certificate. Run this complete subshell from the evidence directory. It clears inherited
Rust toolchain/compiler selectors, explicitly selects `1.98.1-x86_64-unknown-linux-gnu`,
and checks both complete verbose versions against the pinned provenance before
canonicalizing `$ARMS` and changing to the generated arm workspace. Relative
`ARMS` values are resolved from the initial evidence directory. The explicit manifest and child working
directory both select `$ARMS/$ARM`. Set `ARM` to `baseline` or `scored`
before each build. The command selects the same two-target compilation scope. It does not select or run the multihop stress test. Copy the fixture executable before any later
clean/build. Use a prepared
workspace and pinned native runtime/model assets; clear inherited `PAMIN_*`
knobs, set `PAMIN_PROFILE=accuracy` and `PAMIN_DEVICE=cpu`, and run each copied
binary serially with `GRAPH_ARM=baseline` or `scored`, unique `GRAPH_OUT` and
`GRAPH_TRACE` paths, `GRAPH_SHARED_MACHINE` describing interference, and
`GRAPH_CLK_TCK` set to the platform clock-tick frequency:

```sh
LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB" \
PAMIN_EVAL_HOME="$WORKSPACE" PAMIN_PROFILE=accuracy PAMIN_DEVICE=cpu \
GRAPH_ARM="$ARM" GRAPH_OUT="$RESULTS/$ARM.jsonl" \
GRAPH_TRACE="$RESULTS/$ARM.trace.jsonl" GRAPH_CLK_TCK="$(getconf CLK_TCK)" \
GRAPH_SHARED_MACHINE="$INTERFERENCE" \
python3 -c 'import os, sys; retained = {key: os.environ[key] for key in ("PAMIN_EVAL_HOME", "PAMIN_PROFILE", "PAMIN_DEVICE")}; clean = {key: value for key, value in os.environ.items() if not key.startswith("PAMIN_")}; clean.update(retained); os.execvpe(sys.argv[1], sys.argv[1:], clean)' \
python3 source/future-accounting.py.in \
  --log "$RESULTS/$ARM.log" --usage "$RESULTS/$ARM.usage.json" \
  --mapped "$RESULTS/$ARM.mapped-libraries.json" \
  --ort "$ORT_LIB/libonnxruntime.so.1.28.0" --zvec "$ZVEC_LIB/libzvec_c_api.so" \
  -- "$BINARY" scratch_scored_graph_finite_fixture \
  --exact --ignored --nocapture --test-threads=1
```

The new [accounting runner](source/future-accounting.py.in) is a prospective reconstruction of the Linux `wait4` method, not the original measurement-time runner source. It captures native stdout/stderr into a fresh `.log` and writes whole-process wall/user/system/RSS/exit values into `.usage.json`. The wall interval includes process launch, model/fixture setup and wait; wait4 CPU includes the native process and its reaped children, and excludes independent PostgreSQL services. RSS is Linux native-process high water in KiB, not aggregate service memory. This supplies the missing reproduction outputs without changing historical numbers or certifying historical runner bytes.

It also samples actual ONNX Runtime and Zvec mapped paths in that same native process, hashes their files, and requires the supplied pinned paths and digests before accepting a new run. The extra `.mapped-libraries.json` is prospective local evidence; hashes bind files at capture, not mapped inode identity or continuous residency. The frozen historical `graph_trace.rs.in`, trace rows, source/binary identities and 52-entry inventory remain unchanged: their map collector recorded only ONNX Runtime, so historical loaded-Zvec identity cannot be reconstructed from the provisioned Zvec digest. Missing or conflicting future mappings fail while preserving logs and failed-capture accounting whenever the owned child is reaped; timeout has the same failure receipt. If an owned child cannot exit within the bounded kill wait, accounting is explicitly unavailable and the log remains. Failure receipts cannot count as accepted runs. New outputs require separate publication review.

The Python launch shim retains only the three explicitly assigned `PAMIN_*`
values and removes every other inherited product knob before executing the
fixture binary. It does not run a model itself.

Set `WORKSPACE` to that arm's prepared pinned workspace and `ARM` to `baseline` or
`scored`; retain the actual interference declaration in `INTERFERENCE`. The
archive declaration is `exclusive model/build slot; shared file cache and reclaim
pressure; non-timing component reproduction`. Future runs must record their own
conditions. Use separate output paths and retain every failure. Verify source/build hashes,
all six product/helper freshness records, actual provider assignment and model
asset stability before accepting the result. The read-only
[verifier](verify.py) checks the retained published evidence without compiling,
loading a model or opening a database.

The score changes above are arithmetic differences between controlled component outcomes, not product accuracy improvements. Percentages use the retained f32 channel values before rounding. An absent result has no numeric baseline or percentage.

The recorded launch projection binds the selected native test and the complete retained effective product-setting projection, including the arm, workspace, output/trace filenames and interference declaration; its original launch digest is retained. The historical fixture persisted its JSON directly to `GRAPH_OUT`; stdout contains test success but no same-log JSON marker. The verifier cannot establish a missing stdout linkage. Trace events record mapped library paths, not inode identities; inode equality was not captured. It independently parses the retained trace and checks exact provider, runtime and mapped-path correspondence with provenance. All published files except the provenance manifest itself have current digests in its explicit inventory; provenance is checked semantically rather than given a circular self-hash. `verify.py` and `source/prepare.py` reject `-O`, `-OO` and `PYTHONOPTIMIZE`; preparation rejects them before reading inputs or writing output. Run `python3 test_verify.py` for mutation checks.

The graph-only target verifier requires the complete Why list to contain only
one graph-channel record and one path record. The weak origin must be the first
qualifying non-strong result in the retained first 63 fused results, matching
the preserved fixture selection rather than an arbitrary equal-rank topic.
Targeted inert regressions can run with `python3 test_verify.py --round7-graph-only`,
`--round7-weak-only`, or `--round7-native-env-only`; these do not execute the native
fixture or load a model/database.

The verifier binds the complete approved build and launch shell blocks with
literal SHA256 values, including their final newline. Retaining an earlier
correct setting cannot hide a later overriding ORT, zvec or loader assignment.
These pins protect the documented recipe; they do not execute it or establish
historical effective settings beyond the retained receipts.

The unchanged preparation script has an independent verifier digest. Every
retained non-graph result must name one of the 241 fixture topics and carry
allowed channel evidence; controlled paths require `related_to` edges with
`deterministic` derivation. Launch scope and redaction-change flags are checked
against the retained projection and digest differences. Run the bounded new
counterexamples with `python3 test_verify.py --round8-only`.

A local read-only audit found the exact private original digests for all eight redacted artifacts: raw fixture/usage JSON values were unchanged under canonical formatting, trace JSON additionally replaced five local path prefixes per arm, and logs differed only by trailing newline formatting. The six JSON scope declarations record those allowed transformations. Private originals are not distributed; the public verifier checks the declarations and independent captured-value pins and cannot independently replay the original-to-public transformation audit.
