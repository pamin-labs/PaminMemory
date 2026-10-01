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

The schedule preserves four rotated process blocks, limits 5/10, queries 80/101, Accurate A/B context histories, new-query controls and Off controls. Accurate uses six initial calls, six changed-context calls and one new query; Off uses three calls. Each requested limited product call is timed through `Engine::search_reranked` for A/N contexts (496 calls) or `Engine::search_reranked_with` for B contexts with explicit fusion (384 Accurate calls). Full-result and retrieval diagnostics remain outside that timing interval. Indexed count, Flat coverage, relational edges, successful work, model/provider/library identities, hot behavior and independent fresh-final-context oracles must pass before completion is published. The PostgreSQL helper checks owned data-directory/PID/start-time/executable/UID/port identity and stops only the owned clone. Failed clones and logs remain for inspection; only successful stopped clones are deleted.

`metrics.json` and `tables.md` include correctness counts, recall/MRR/nDCG, wall p50/descriptive p95, per-call CPU/work counters, native RSS/HWM, process wall/open/cumulative CPU, owned PostgreSQL main-PID memory and disk sizes. Hot quantiles pool dependent calls within four independent blocks. Original variability, sign-change, work and accuracy eligibility rules are retained; every regression remains visible. CPU zero ticks are resolution-censored. Native launch-to-exit-detection wall includes startup/diagnostics/polling, excluding PG start/stop; cumulative CPU stops at the final search, so full lifetime CPU is N/A. Native sampling excludes PG; PG children/device/service totals remain N/A.

The new logical asset row covers the explicitly listed pins, not every historical source descriptor/loader/cache file. Historical full-asset inventory parity is therefore N/A. Baseline allocated index bytes can be collected; endpoint allocation and temporary peak allocation remain N/A. This smaller reproduction does not recreate the omitted full-payload metadata-write attestation: its disk-equality or immutability certificate is explicitly N/A. Equal logical bytes do not mean zero writes. Fresh generated index/UUID differences can change tied rankings and therefore observed quality; results support the reproduced method's observed cells, not universal retrieval quality or historical numeric identity.

Templates remain inert in this checkout. They materialize as scripts and ignored test targets only in scratch exports; they never add a Cargo target to the tracked project or modify tracked source. The local result logs contain regenerated synthetic text and local paths and are not automatically published. From this harness directory, reproducibility preparation discovers every retained `test_*.py` module without native execution:

```sh
python3 -B -m unittest discover -s . -p 'test_*.py' -v
```

These tests cover the complete schedule, quantile/eligibility arithmetic, mocked report orchestration, provider rejection, source/archive bindings and tracked-source boundaries. They do not establish compilation, model execution, fixture ingestion or product accuracy on another host.

The selected build recipe sets `ZVEC_LIB_DIR` to the pinned native-library directory and `ZVEC_AUTO_BUILD=0`, plus `ORT_LIB_LOCATION` and `ORT_PREFER_DYNAMIC_LINK=1`. The clean environment excludes inherited `ORT_LIB_PATH` overrides. The installation package directory is named `17.6.0`; both PostgreSQL executables must report `17.6`, and the owned server must report `server_version_num=170006`.

Build, seed, and native helper processes start in new owned process groups. Cleanup terminates live members of those groups even if the leader has already exited; a descendant that creates another session is outside this group check. PostgreSQL is stopped separately after ownership verification. PostgreSQL startup failures (including SQL/version/settings assertions) and seed failures attempt a bounded stop only after verifying the actual PID, executable, UID, data directory and port. Seed cleanup works before `server.json` exists. If ownership cannot be proved, cleanup refuses to signal an unknown process and reports the failure. Failed clones and logs remain available for inspection. These guards are covered by synthetic fault tests; native execution remains unverified.

Exit observation uses Linux `waitid(WNOWAIT)`: the owned leader remains unreaped until group cleanup, reserving its PID/group ID against reuse. Cleanup waits at most ten seconds for live group members to exit before reaping the leader; an already-reaped leader refuses any signal, and a lingering member causes explicit failure with retained evidence. Zombies count as exited, not live work. No subsequent job or successful completion receipt is produced after cleanup failure.

The controller resolves its cgroup v2 membership and checks every visible
ancestor's memory limit against that ancestor's own usage; effective CPU quota
is the minimum quota/period across the same hierarchy, with affinity retained
separately. Admission and monitoring append actual host hardware, kernel,
affinity, quota, memory scope/headroom and disk reserve observations to
`host-conditions.jsonl`. This mutable startup/monitoring journal remains ancillary
and is not imported into reports. New `metrics.json` and `tables.md` report only
each process's before/after observations from completion-bound packets, revalidated
before reporting. Missing endpoint fields remain `N/A` and withhold cost eligibility;
no current-host reconstruction supplies missing measured conditions. These are
new-run endpoint observations, not reconstructed historical hardware or proof of
continuous hardware equality.

Before deleting a successful stopped clone, the packet captures owned
PostgreSQL data-directory regular-file logical and allocated bytes, including
local `pg_wal`, before and after the trial. Allocated bytes use `st_blocks * 512`;
external symlink targets and directory metadata are excluded. Reports compare
mean clone endpoint sizes by source. Missing older packet measurements remain
`N/A`; these are endpoint file sizes, not peak allocation or write cost.

Cgroup accounting covers only the mounted Linux cgroup v2 hierarchy visible to
the controller. A membership path that cannot be mapped to an existing leaf
fails closed; limits of invisible ancestors are unknown and are not claimed
as included. Memory headroom is conservatively `memory.max - memory.current`
at every visible limited ancestor, with no `inactive_file` reclaim credit.
It is not an available-memory estimate and can reject a host under file-cache
reclaim pressure. These controller changes have only synthetic source-level
validation; no native reproduction or current-host admission was run.
The PostgreSQL before endpoint is the fresh seed clone before server startup;
the after endpoint is measured after the native process and owned PG stop,
before deletion.

Each Accurate history arm is classified by whether the ordered actual
query/document bytes passed to the reranker change; changed topic identifiers
alone are insufficient. Unchanged-input histories remain diagnostic controls
and are excluded from changed-input correctness and speed eligibility. Every
source/limit/initial-history comparison must include at least one genuinely
changed-input history, or the report fails rather than claiming such coverage.
Memory sampling skips unavailable `/proc` fields during process exit, without
substituting zero RSS/HWM. Builds require an exact tool version token and retain
the complete reported cargo/rustc/rustdoc versions in `build/toolchain.json`.

Owned PostgreSQL runs on loopback TCP with Unix sockets explicitly disabled,
and authenticated `SHOW unix_socket_directories` must confirm the empty setting.
The existing PID/data/executable/UID/port ownership and effective native resource
setting checks still apply. This avoids distribution-specific unwritable socket
directories; actual PostgreSQL startup remains unverified here.

Builds use fresh owned `build-home` and `build-cargo-home` directories. The latter
links only prerequisite `registry` and optional `git` caches from the supplied
Cargo cache; neither inherited home/configuration nor credentials are copied.
Both `config` and `config.toml` are rejected in the canonical working checkouts,
every searched ancestor and the owned Cargo home before/after each offline
build. `build/cargo-config.json` records absent search paths and cache bindings,
not credential values. Existing RUSTUP/toolchain assets remain prerequisites;
cache contents are not independently content-attested, and endpoint checks do
not prevent concurrent mutation. No Cargo build was performed for these tests.

Frozen binaries produced by this package are `scratch_cache_product_limits`
integration-test helpers. Their logical byte row includes the probe/scaffold and
source-root strings; it describes helper artifacts only. The shipped CLI/server
executable disk size is unmeasured and reported as N/A. No helper size delta
establishes a product executable disk change.

### Checkpoint and service identity scope

The 80-process controller now atomically publishes completed packets after owned
cleanup. Resuming revalidates the plan, retained logs, job chronology, row/hot
checks and PostgreSQL bindings, loads completed jobs and runs only incomplete
jobs. Unbound old result directories, changed plans and corrupt checkpoints fail
closed. Incomplete attempt directories remain available; their PostgreSQL state
must pass preflight before a new attempt. The cooperative lock and exclusive-slot
condition cannot certify unrelated surviving native descendants after a lost
controller. Resume fixtures are mocked; interruption/recovery with actual native
processes remains unverified.

The plan and each prospective process packet retain the PostgreSQL installation
file inventory/configuration digest, actual server executable digest and mapped
library file digests. The report keeps per-job build identities. All new and
resumed jobs must share the same sorted mapped-library location, byte count and
SHA-256 identities. A changed, added or removed mapped PostgreSQL dependency
refuses checkpoint loading, new packet publication and completion; the analyzer
reuses checkpoint loading before reporting. Native runtime mapping paths decode
procfs octal escapes before matching pinned zvec and ORT paths, including scratch
directories containing spaces, tabs, newlines or backslashes. Historical
PostgreSQL build identity is N/A; version text alone does not establish equality.
Mapped library inode identity is unmeasured. `pg_config` configuration flags may
contain local build paths; generated receipts stay local until publication review.

Off cold/new-query controls must agree across sources on limited, complete and
fused results before the comparison is accepted. Equal zero model work alone
does not establish like-for-like retrieval.

New completion receipts bind the exact bytes of all 80 checkpoint packets with
a per-job SHA256 map, checked against validated packets at completion and again
before reporting. This includes controller-only usage, host and PostgreSQL disk
fields even if an in-packet self-hash is refreshed. These are mutable local
endpoint receipts, not independent authenticity or continuous immutability
proof. Legacy completed receipts without the exact binding are rejected; there
is no in-place upgrade. Original historical evidence and reports are unchanged.

Report publication renders both payloads before exposing either final filename. Each file is published without overwrite; an interruption between files can resume only when every existing report exactly matches the regenerated bytes. Conflicting files are preserved and rejected. The two filenames are not published atomically together. Retained native maps must contain exactly the pinned ORT/Zvec library identity set; repeated mappings and unrelated system libraries are allowed, but additional ORT/Zvec identities are rejected.

New-query cost eligibility additionally requires equal recorded cumulative
successful-work and allocation-shape counters immediately before the new query
in each paired block. Equal new-query deltas alone do not qualify wall/CPU/RSS/HWM
comparisons after different preceding inference. Missing prior snapshots withhold
cost claims; quality observations remain independent. These counter endpoints do
not prove identical operation order or allocation lifetime.

Changed-input correctness and unchanged-control counts use only whole main/stack
pairs whose actual input-change scopes agree. A scope mismatch excludes both arms
from those differences and is retained in `unpaired_history_scopes`; each row shows
its actual paired-block denominator, which can be smaller than four. The original
classifications remain available as diagnostics. Frozen helper receipts require
exact main/stack/seed helper entries and fields, pinned revision and digest, and nonnegative integer sizes
matching actual regular-file sizes before run-plan publication. These prospective
eligibility and receipt checks do not rewrite historical raw data or numerical rows.
