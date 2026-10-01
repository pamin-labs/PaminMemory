# Regenerate the selected main/stack synthetic search experiment

These sources can generate a **new** 80-process/880-product-call experiment and every reported metric category in the [retained evidence](../../results/inference/selected-main-stack-2026-10-01/README.md). They complement its arithmetic verifier. They do not promise the historical numbers, UUIDs, database bytes, index bytes or executable bytes. No product experiment was rerun while preparing this package.

The before source is public main `315c10242ddf7a1cec3bccbf550a942320e09557`. The after source's provenance is `d0b6a14a4f317fea3c1e117f627289aad1b1c964`; that historical full commit is not a public fetch prerequisite. Its [archive](stack-source.tar.gz) contains only byte-exact compilation inputs for this selected target, checked against each original Git blob in the [manifest](stack-source-manifest.json). It includes LICENSE and NOTICE. The recorded full historical tree hash is provenance, not a whole-tree certificate. Private submodule contents, historical measurement runners, traces, grants, credentials, databases and model weights are omitted.

The measured Rust probe is preserved byte for byte. Materialization combines each source's public retrieval scaffold with that probe and requires the original helper hash. A separate ignored seed helper uses the real product `write_corpus` path to ingest the public 230-document/157-query fixture once. Both arms then use stopped copies of that same freshly generated seed, preserving identifiers across their comparisons. The fixture is already in public source; no private gold or raw trace packet is imported.

Use a dedicated disposable copy of the prepared model cache for this experiment. Product model-cache housekeeping is allowed only in that owned copy; pre/post asset guards reject unexpected changes. Use Linux x86-64, Python 3.12+, the pinned Rust 1.98.1 toolchain, a locally available Cargo dependency cache, PostgreSQL 17.6.0 and the source/model/runtime assets in [sources.json](sources.json). In particular, supply the prepared CPU BGE-M3 and Accurate model cache, its source descriptors and tokenizer/configuration files, ORT 1.28.0 and the bound zvec sidecar. Model revisions are `2b34e84df040034d4b9eabb62383a87c18955822` and `6f5ff65298512715a1e669753bc754d2bc8f367b`. Actual executed prepared graph hashes and CPU node counts must match; merely registering a provider does not pass.

No downloader or model provisioning command is supplied. Cargo uses `--offline --locked`, and model requests use offline cache settings. Locally buildable native dependencies and fully prepared artifacts are external prerequisites. Use network isolation if enforcing offline behavior of third-party build scripts. Missing artifacts fail the run; do not substitute revisions or report an incomplete matrix. Model licenses remain those in NOTICE; this archive does not distribute their weights or PostgreSQL/native binaries.

Obtain the public before revision in a clone, then materialize **outside that clone** into a new scratch directory. Replace the local paths below with existing prerequisites; nothing writes into the original clone.

```sh
git fetch origin 315c10242ddf7a1cec3bccbf550a942320e09557
python3 -B materialize.py --repo /local/PaminMemory --work /local/new-experiment \
  --models /local/prepared-model-cache --native /local/zvec-library-directory \
  --ort /local/ort-1.28.0-library-directory --postgres /local/pg-17.6.0 \
  --cargo /local/rust-1.98.1/bin/cargo --rustc /local/rust-1.98.1/bin/rustc \
  --rustdoc /local/rust-1.98.1/bin/rustdoc
cd /local/new-experiment
python3 -B build.py --execute --exclusive
python3 -B seed.py --execute --exclusive
python3 -B run.py --execute --exclusive
python3 -B analyze.py
```

Run as a non-root PostgreSQL-capable user. Provision `CARGO_HOME`/`RUSTUP_HOME` for the local offline toolchain cache. The controller admits only one cooperative experiment per host and requires at least 8 GiB free disk and 2 GiB cgroup memory headroom throughout monitored operations. Reserve additional disk for fresh builds and fixture copies. Claim `--exclusive` only after excluding competing builds/model experiments; the cooperative lock cannot detect unrelated tools. The historical host had a four-CPU quota/five eligible logical CPUs. Record a changed host/quota as new conditions, not comparable historical timing.

The schedule preserves four rotated process blocks, limits 5/10, queries 80/101, Accurate A/B context histories, new-query controls and Off controls. Accurate uses six initial calls, six changed-context calls and one new query; Off uses three calls. Each requested limited product call is timed through `Engine::search_reranked` or its explicit fusion-ablation counterpart. Full-result and retrieval diagnostics remain outside that timing interval. Indexed count, Flat coverage, relational edges, successful work, model/provider/library identities, hot behavior and independent fresh-final-context oracles must pass before completion is published. The PostgreSQL helper checks owned data-directory/PID/start-time/executable/UID/port identity and stops only the owned clone. Failed clones and logs remain for inspection; only successful stopped clones are deleted.

`metrics.json` and `tables.md` include correctness counts, recall/MRR/nDCG, wall p50/descriptive p95, per-call CPU/work counters, native RSS/HWM, process wall/open/cumulative CPU, owned PostgreSQL main-PID memory and disk sizes. Hot quantiles pool dependent calls within four independent blocks. Original variability, sign-change, work and accuracy eligibility rules are retained; every regression remains visible. CPU zero ticks are resolution-censored. Native launch-to-exit-detection wall includes startup/diagnostics/polling, excluding PG start/stop; cumulative CPU stops at the final search, so full lifetime CPU is N/A. Native sampling excludes PG; PG children/device/service totals remain N/A.

The new logical asset row covers the explicitly listed pins, not every historical source descriptor/loader/cache file. Historical full-asset inventory parity is therefore N/A. Baseline allocated index bytes can be collected; endpoint allocation and temporary peak allocation remain N/A. This smaller reproduction does not recreate the omitted full-payload metadata-write attestation: its disk-equality or immutability certificate is explicitly N/A. Equal logical bytes do not mean zero writes. Fresh generated index/UUID differences can change tied rankings and therefore observed quality; results support the reproduced method's observed cells, not universal retrieval quality or historical numeric identity.

Templates remain inert in this checkout. They materialize as scripts and ignored test targets only in scratch exports; they never add a Cargo target to the tracked project or modify tracked source. The local result logs contain regenerated synthetic text and local paths and are not automatically published. Reproducibility preparation can be checked without native execution:

```sh
python3 -B test_templates.py
```

These tests cover the complete schedule, quantile/eligibility arithmetic, mocked report orchestration, provider rejection, source/archive bindings and tracked-source boundaries. They do not establish compilation, model execution, fixture ingestion or product accuracy on another host.
