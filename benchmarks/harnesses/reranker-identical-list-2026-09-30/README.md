# Regenerate the last-identical-list memo measurement

This package can generate a **new PR216-method experiment**: 36 serial native
processes and 428 timed calls, comparing the complete-batch predecessor with
the last-identical-list memo. It complements the arithmetic verifier for the
[historical evidence](../../results/reranker/cache-identical-list-2026-09-30/README.md).
It does not claim PR228's separate 80-process/880-call experiment reproduced
these measurements. No benchmark was executed while preparing this package.

The original 8,383-byte native probe is retained exactly in `probe.rs.in`, SHA256
`7a16f16acd939b68a65d8ae4c9a8fcff53e44d76508d4812ef0b8438b5299d2f`.
Combining it with either original retrieval scaffold must produce helper SHA256
`1473c07b397282d3a5bc833b2e8d3a94e824703a4219012251d2cd80af027b50`.
It calls the actual product `Engine::search_reranked` or the supported explicit
Fusion context override; it does not reimplement retrieval or reranking.
Return limits are 5/10. Full-result limit300 diagnostics remain outside each
timed clock. The original public synthetic fixture contains 230 documents and
157 queries; only queries 80/81/101/102 enter this measurement.

## Explicit offline shared dependency

This package requires a separately supplied, hash-bound copy of
`benchmarks/harnesses/selected-main-stack-2026-10-01` from the same-stack [PR228](https://github.com/pamin-labs/PaminMemory/pull/228)
package, including its updated nested/ancestor cgroup guards and host-condition
journal. Exact supplied file hashes are authoritative; the recorded local source
revision is provenance, not a remote commit-fetch prerequisite.
The exact file hashes and sizes are in [shared-dependency.json](shared-dependency.json).
A bare PR216 checkout does **not** automatically contain this dependency.
Supply an existing offline checkout/export of that package; the materializer
does not fetch or download it. A current whole-stack checkout is usable only if
its dependency files match those pins. Later changes to shared guards require
an explicit dependency update and repeated source/mock checks.

Build, seed, owned PostgreSQL, native-process execution and cleanup controllers
are reused from that dependency, not maintained as a second independent fork.
Materialization copies them byte-for-byte to the scratch export and installs
this package's own schedule, row checks and analysis. The dependency's
80-process schedule is never invoked by this runner. These are sanitized
prospective controllers, **not byte-exact reconstructions of the historical
private controllers**. Original private controller hashes remain historical
identities rather than certificates for these new scripts.

The baseline archive contains only 124 selected compilation inputs from
`2f5bd087ceb4b65a25dc94932e416d8ac5c29740`, independently checked against the
original Git blobs in [baseline-source-manifest.json](baseline-source-manifest.json).
The candidate uses the shared dependency's corresponding 124-file archive of
`d0b6a14a4f317fea3c1e117f627289aad1b1c964`. Full historical commits/trees are
provenance identifiers, not public fetch prerequisites or whole-history
certificates. Both subsets include LICENSE/NOTICE, production sources,
dependency declarations and the selected public test scaffold/fixture. Private
submodule contents, planning, logs, databases, model weights, native libraries
and executables are omitted. No historical binary identity is reconstructed.

## Prospective commands

Use Linux x86-64, Python 3.12+, the locally installed Rust 1.98.1 toolchain,
offline Cargo dependencies, PostgreSQL 17.6.0, the prepared CPU model cache,
ORT 1.28.0 and the pinned zvec library. The shared `sources.json` enumerates
required model/runtime assets, sizes, hashes, revisions and actual CPU node
counts. Supply existing prerequisites; no downloader is provided. Build uses
`--offline --locked`, and model access uses offline settings. Third-party build
scripts still require an externally enforced network policy if strict offline
execution is required.

Use a dedicated disposable model-cache copy: product cache housekeeping may
write only there. Run as a non-root user capable of starting owned PostgreSQL.
Keep the original clone, shared dependency and historical evidence untouched.
Scratch output must be outside this source repository. A supplied dependency
at its standard `benchmarks/harnesses/selected-main-stack-2026-10-01` path protects
its entire enclosing repository; a generic offline package protects that package
directory itself.
Replace all example paths with existing local prerequisites:

```sh
python3 -B materialize.py \
  --shared-package /local/offline-package/selected-main-stack-2026-10-01 \
  --work /local/new-identical-list-experiment \
  --models /local/disposable-prepared-model-cache \
  --native /local/zvec-library-directory --ort /local/ort-1.28.0-library-directory \
  --postgres /local/pg-17.6.0 --cargo /local/rust-1.98.1/bin/cargo \
  --rustc /local/rust-1.98.1/bin/rustc --rustdoc /local/rust-1.98.1/bin/rustdoc
cd /local/new-identical-list-experiment
python3 -B build.py --execute --exclusive
python3 -B seed.py --execute --exclusive
python3 -B run.py --execute --exclusive
python3 -B analyze.py
```

Configure the local `CARGO_HOME`/`RUSTUP_HOME` offline caches first. The shared
build controller creates owned, config-free `HOME` and `CARGO_HOME` directories,
linking only existing registry/git caches. It refuses either Cargo configuration
filename in source-checkout ancestors and the owned Cargo home, and records
those checks; external cache contents are prerequisites, not independently
attested. PostgreSQL uses authenticated loopback TCP with Unix sockets disabled.
Reserve at
least 8 GiB free disk, additional build/clone growth and the shared controller's
2 GiB cgroup memory headroom. Claim `--exclusive` only after excluding competing
builds/model experiments; the cooperative host lock cannot detect unrelated
tools. The commands above are available for a later authorized reproduction;
running them is not a prerequisite for inspecting this source-only change.

The matrix preserves two Accurate rounds (baseline→memo, then memo→baseline),
eight histories per source per round, and four Off processes after those rounds.
Accurate runs six initial-context calls, six changed-context calls and one new
query. Off runs three calls. The generated scripts label baseline as `main` and
memo as `stack` internally; these labels do not mean current main or the entire
current stack. New reports include the exact frozen revisions and this mapping.

The source/asset guards check the exact probe/scaffold and runtime identities.
The shared guards locate cgroup2 membership through actual mountinfo and record
the host/affinity and visible nested/ancestor quota/memory conditions rather
than assuming the historical host.
The controllers authenticate and clean up only owned PostgreSQL/child process
groups. Each new native-process receipt captures its copied PostgreSQL
installation/build configuration, running executable and mapped-library file
hashes; historical PostgreSQL build identity remains `N/A`. These are capture
checks, not an attestation of library inode identity. This 36-process controller
does not invoke the shared 80-process controller’s checkpoint/resume logic.
The row checks require actual requested settings, corpus count, Flat
coverage, visible selected pairs, typed f32 results, matched gold, hot behavior,
equal successful work and fresh-final-context oracles. Baseline hot calls may
encode; memo hot calls must skip encoding. Missing rows, changed work premises
or failure markers reject completion. Failed clones/logs remain for inspection.

`metrics.json` reports correctness classifications, observed recall/MRR/nDCG,
wall median/interpolated within-sample p95, successful work, CPU ticks,
RSS/HWM and process observations, with before/after/absolute/percentage cells.
Hot calls remain dependent within two rounds/four relevant histories; no stable
population tail or causal speed claim is generated. CPU zero ticks are censored
by their nominal resolution, and CPU percentage claims are withheld. RSS/HWM
includes model initialization and outside-clock diagnostics; it is not memo
heap cost. PostgreSQL samples cover its main PID only. Full-lifetime CPU,
accelerator-service memory, allocator-specific memo heap and omitted-byte
immutability remain N/A. Equal logical index sizes do not mean zero writes.

A newly ingested seed preserves IDs across the two arms but does not recreate
historical UUIDs/database/index bytes. Different tied rankings can change
observed quality or fail workload assertions; do not force a passing result.
The recipe can regenerate the method's newly observed rows, **not promise the
historical timing values or binaries**. Historical complete-asset/seed equality
and private build/environment receipts are not reconstructed by this package.
Compilation, ingestion and native execution on another host remain unverified.

## Inert verification

Templates materialize as ignored tests/controllers only in a new scratch
export; no Cargo target is added to the tracked product. The source-only mock
checks do not need the shared dependency or native prerequisites:

```sh
python3 -B test_templates.py
```

They check the exact 36/428 matrix, original probe, selected baseline Git blobs,
dependency rejection, execution opt-in, predecessor/memo encoding distinctions,
analysis completeness and the descriptive percentile definition. Shared
controller lifetime/cleanup checks remain the dependency package's tests;
neither suite establishes a new benchmark or hardware result.
