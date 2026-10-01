# Dual-vector and shared compute prototype evidence — 2026-10-01

## Precision and scope

The product boundary is `Engine::search_reranked`: memory index, named passages, accurate reranker, limit 60, default semantic depth 50 / graph depth 2. The paired CPU-only XQuAD-R comparison has 13,014 documents and the same 1,190 language-rotated queries. Corpus files are fingerprinted in `corpus-files.json`; raw rankings, relevance/exclusions, the predeclared rule, summaries and immutable execution-provenance correction are retained here. No fusion weights were fitted on this corpus.

Baseline CPU executable: `555846c`; dual executable: `45a3bcc601b2f9ca7d23d0bf49c6c95b3284f73e`. The candidate raw manifest sampled a later git HEAD after ingestion; the correction sidecar identifies the frozen executed binary and source. Do not treat that late field as execution identity. The retained main-cpu and new-cpu 66-query blocks both match the full baseline ranking slice exactly, as independently asserted by the verifier.

| Metric | BGE | Dual | Difference | Change |
|---|---:|---:|---:|---:|
| Cross-language nDCG@10 | 0.726706 | 0.734442 | +0.007736 | +1.06% |
| Cross-language recall@50 | 0.903193 | 0.883109 | -0.020084 | -2.22% |
| Same-language nDCG@10 | 0.868201 | 0.872163 | +0.003962 | +0.46% |
| Same-language recall@50 | 0.964706 | 0.966387 | +0.001681 | +0.17% |

Cross nDCG improves in this exploratory, post-hoc corrected analysis (shared paragraph-cluster sign flips, Holm p≈0.000040); same-language gain is not significant. Cross recall declines. This is a ranking/coverage tradeoff, remains **opt-in**, and does not justify default promotion from one corpus. BGE revision `2b34e84df040034d4b9eabb62383a87c18955822` is combined with PPLX `2c4d510dd4a732063c31a0f70193e35067b51fd8`, with all 196 MatMulNBits at accuracy_level=4. Encoding `pool-int8-single-v2-level4` requires reindex; both dense fields are FP16. Semantic depth and total vector vote budget are unchanged.

## Product costs

Three independent sequential process blocks per arm, 66 unique queries per process (six per language), actual search boundary. Every arm asserts corpus count, vector completeness, nonempty results and model work. Shared Apple M4/Mac16,12; warmed model calls and fresh process query/score caches. The median table is descriptive: CPU block ranges overlap and no per-query cost significance is asserted. Initial and cached automatic arms are different regimes.

Each per-process p50 is the sample median; p95 is nearest rank `sorted[ceil(0.95*N)-1]`. The table takes medians of process-block cells. `cost-summary-historical.json` preserves an earlier p95 calculation one rank too low; `cost-summary-correction.json` records the correction. Raw rows are unchanged.

| Comparison / metric | Before | After | Difference | Change |
|---|---:|---:|---:|---:|
| Default CPU stack: p50 | 1244.01 ms | 1272.31 ms | 28.30 ms | +2.28% |
| Default CPU stack: p95 | 1635.81 ms | 1686.48 ms | 50.67 ms | +3.10% |
| Default CPU stack: sampled warm process RSS | 1,724,874,752 B | 1,706,983,424 B | -17,891,328 B | -1.04% |
| Dual CPU vs main: p50 | 1244.01 ms | 1125.23 ms | -118.77 ms | -9.55% |
| Dual CPU vs main: p95 | 1635.81 ms | 1623.66 ms | -12.15 ms | -0.74% |
| Dual CPU vs main: sampled warm process RSS | 1,724,874,752 B | 2,465,054,720 B | 740,179,968 B | +42.91% |
| Cached automatic vs main: p50 | 399.81 ms | 400.01 ms | 0.20 ms | +0.05% |
| Cached automatic vs main: p95 | 524.61 ms | 522.38 ms | -2.22 ms | -0.42% |
| Cached automatic vs main: sampled warm process RSS | 2,707,390,464 B | 2,705,063,936 B | -2,326,528 B | -0.09% |
| Initial prototype vs corrected cached regime: p50 | 398.54 ms | 400.01 ms | 1.47 ms | +0.37% |
| Initial prototype vs corrected cached regime: p95 | 531.06 ms | 522.38 ms | -8.68 ms | -1.63% |
| Initial prototype vs corrected cached regime: sampled warm process RSS | 6,322,061,312 B | 2,705,063,936 B | -3,616,997,376 B | -57.21% |

Do not infer a CPU latency gain from the lower dual median: candidate lists/reranker work differ and the host is shared. Default automatic search is at main parity in these retained blocks. The initial-to-cached RSS comparison is the **combined** resource correction (early rejection plus reuse), not an isolated persistence effect. RSS is sampled after warm outside query timers, includes the native process and excludes accelerator services/VRAM. Compiled CoreML cache warmth was not independently controlled: causal cold-start and whole-harness peak comparisons are N/A.

These costs describe frozen `503bd9e` and `0f023be6a8d8d070a971f7e590ccccff2c3292bb` candidates, with true main `315c10242ddf7a1cec3bccbf550a942320e09557`. Later calibration-lock/workload changes have **not** been retimed end to end. CPU-only arms use optimized ORT CPU/SIMD; automatic main/candidate arms use CPU embedding and the CoreML accurate-reranker EP. `device-and-cache-proof.json` retains the loaded-device records and proves all three persisted-hit arms contain no recalibration or rejection lines. EP selection does not prove ANE/GPU physical placement. Raw BGE CoreML still fails CPU-space conformance; this stack does not claim that numerical bug is repaired. Native WinML/NPU speed/quality/RSS remain N/A: no vendor NPU hardware is available.

## Disk

| Metric | Before | Dual | Difference | Change |
|---|---:|---:|---:|---:|
| Projection logical bytes | 83,303,573 B | 117,151,847 B | +33,848,274 B | +40.63% |
| Filesystem allocated blocks | 83,877,888 B | 117,755,904 B | +33,878,016 B | +40.39% |

Projection only; PostgreSQL and model caches excluded. APFS unique shared extents are not resolved. PPLX external data is 706,117,632 bytes, with another owned logical copy required beside the derived graph for ORT path validation; no physical COW saving is claimed. Observed plan metadata totals 64,646 bytes. Native accelerator service memory and causal model-cache disk differences are unmeasured.

## Verification / reproduction

```sh
python3 -m pip install numpy==2.5.1
python3 benchmarks/verify_compute_evidence.py
```

This independently recomputes paired nDCG/recall, 240-paragraph shared sign flips/cluster bootstrap (exploratory post-hoc correction) and all 21 cost blocks from raw rows, asserts paired IDs/corpus sizes and checks each reported percentile/RSS value. Timing program sources remain scratch-only; source/binary hashes and execution identities are retained. Frozen candidate commits are reachable as `bench/prototype-cost-2026-10-01` (503bd9e) and `bench/persisted-cost-2026-10-01` (0f023be); fetch these tags when inspecting those revisions. Re-running the live workload requires a provisioned unprivileged PostgreSQL workspace and the fingerprinted models/data, 66 fixed query indices `18*i` (i=0..65), warm query 1189, and the product settings above. Freeze executables before launch and keep builds out of timing runs. Historical total ingest time was measured on a shared host with intervening builds and is not a causal speed comparison.

The original query-level statistical summary is retained as historical, not a valid independent-question acceptance analysis. `cluster-inference.json` records 100,000 shared flips and cluster-bootstrap draws (PCG64 seed 20261001): cross Holm p≈0.000040; same-language p≈0.153978, lower 95% delta bound -0.001348, still above -0.005. Point metrics and raw rankings are unchanged.

## Fetch measured source refs in a shallow review checkout

A pull-request checkout does not automatically include unrelated benchmark tags.
The remote refs are verified at [prototype source](https://github.com/pamin-labs/PaminMemory/tree/bench/prototype-cost-2026-10-01) and [persisted source](https://github.com/pamin-labs/PaminMemory/tree/bench/persisted-cost-2026-10-01).

```sh
git fetch origin refs/tags/bench/prototype-cost-2026-10-01:refs/tags/bench/prototype-cost-2026-10-01 refs/tags/bench/persisted-cost-2026-10-01:refs/tags/bench/persisted-cost-2026-10-01
git cat-file -t 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27
git cat-file -t 0f023be6a8d8d070a971f7e590ccccff2c3292bb
```

## Executable scratch reproduction

The [frozen scratch sources](https://gist.github.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/eff4c8c55e69c341f00517046aa9e0d6e3aba751) remain outside the tracked tree, with SHA-256 in `persisted-cost-identity.json`. A provisioned, unprivileged `/private/tmp/pamin-dual-product-eval` must already contain the fingerprinted corpus, model cache, PostgreSQL and complete `dual-product-accuracy-24ad7f1862182925` / `dual-product-dual_accuracy-24ad7f1862182925` indexes. The complete guarded recipe below builds all three revisions and aborts rather than timing an empty corpus. It retains separate source/executable identities and uses the provisioned workspace described above.

The frozen process-worker source records whole-process wall/user/system time, binary SHA and RSS, with its macOS runtime-library path declared explicitly. Use its four arguments `case-name frozen-executable profile policy` when reproducing those process columns; warm search columns come from the Rust rows. Provisioning/downloading is not part of the timed search. Native device/service memory remains unmeasured. Strictly redacted event streams underlying cache/device proof are retained under `logs/` and hashed/recounted by the verifier. Every original line is mapped to a fixed enum; load events additionally retain only whitelisted public model identities, accurate tier and maximum_tokens=256. Paths, arbitrary free text and content are excluded. Main’s historical log lacks embedder-load events: its BGE identity comes from frozen source/model fingerprints, not an invented load event. Single-model logs identify the repository, not independently the loaded revision; dual logs do include both pinned revisions. Original source hashes are retained; original logs stay local.

The original `predeclared.json` is immutable. It specified independent question-level inference; the corrected cluster analysis was chosen after the results were observed. The new p-value supports an exploratory signal, not a preregistered acceptance claim. Confirm on an independent corpus before changing defaults.

### Seven-arm command matrix

Use a separate checkout of each listed revision. Download `pamin-main-cost-harness.rs` for main, `pamin-new-cost-harness-reconstructed.rs` for 503bd9e, or `pamin-persist-cost-harness.rs` for 0f023be from the [versioned scratch bundle](https://gist.github.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/72859f283b963c0294875dbef2a4e3a00209f9b5), into the ignored `crates/pamin-engine/tests/scratch_matched_costs.rs` location. The 503 harness is a disclosed reconstruction of the retained persisted harness with only its literal execution identity changed; it is not claimed as an exact original source-byte archive. Main's original retained source and the persisted source are separate.

| Arm | Revision | Harness | Profile | Policy |
|---|---|---|---|---|
| main-cpu | 315c102 | main | accuracy | cpu |
| main-auto | 315c102 | main | accuracy | auto (unset PAMIN_DEVICE) |
| main-auto-repeat | 315c102 | main | accuracy | auto (unset PAMIN_DEVICE) |
| new-cpu | 503bd9e | reconstructed 503 | accuracy | cpu |
| dual-cpu | 503bd9e | reconstructed 503 | dual_accuracy | cpu |
| new-auto | 503bd9e | reconstructed 503 | accuracy | auto (unset PAMIN_DEVICE) |
| new-auto-persist-hit | 0f023be | persisted | accuracy | auto (unset PAMIN_DEVICE), established nonexpired persisted plan |

Use the guarded three-build and seven-arm controller below. Each source checkout/profile/policy comes from this matrix; each process writes a distinct arm/block path. The controller automatically runs bounded unmeasured primers immediately before each measured persisted-hit block, requiring one clean hit probe before measurement.

### Whole-process resource reproduction

The worker is Python, so it needs no native build. Fetch the retained source and syntax-check it (this does not run a model):

```sh
curl -fsSL https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/a8327acdbaecfa651ddd78a293df7dae43b223da/pamin-cost-worker.py -o /private/tmp/pamin-cost-worker.py
python3 - /private/tmp/pamin-cost-worker.py <<'PYWORKER'
import hashlib, sys
from pathlib import Path
if not hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == '677f12a00f42563ada52368f69b7e088fa67d07aab7f8f99905328d5fb4e5d6c':
    raise RuntimeError('reproduction integrity check failed')
PYWORKER
python3 -m py_compile /private/tmp/pamin-cost-worker.py
```

Build all three sources explicitly in fresh detached checkouts. The installer checks HEAD and source SHA, removes stale ignored tests, and registers cleanup before downloading/building. All builds share one target directory to limit disk usage; each executable is copied before the next build. Each build session owns unique temporary source paths and installs outer teardown before adding the first worktree. It unregisters only worktrees it created, on success/failure/interruption; unrelated registrations are preserved.

```sh
# Run in a fresh Bash shell from the repository. The PostgreSQL/model/index
# provisioning described above must already exist, under an unprivileged user.
(
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
source_root=$(mktemp -d /private/tmp/pamin-cost-sources.XXXXXXXX)
created=()
cleanup_sources() {
  for ((i=${#created[@]}-1; i>=0; i--)); do
    path=${created[i]}
    git -C "$repo_root" worktree remove --force "$path" 2>/dev/null || true
  done
  # Successful remove unregisters each worktree; never prune unrelated entries.
  rmdir "$source_root" 2>/dev/null || true
}
trap cleanup_sources EXIT
trap 'exit 130' HUP INT TERM
add_source() {
  path="$source_root/$1"
  git -C "$repo_root" -c submodule.recurse=false worktree add --detach "$path" "$2"
  created+=("$path")
}
add_source main 315c10242ddf7a1cec3bccbf550a942320e09557
add_source prototype 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27
add_source persisted 0f023be6a8d8d070a971f7e590ccccff2c3292bb
build_cost_binary() (
  set -euo pipefail
  cd "$1"
  test "$(git rev-parse HEAD)" = "$2"
  git diff --exit-code --ignore-submodules
  git diff --cached --exit-code --ignore-submodules
  scratch=crates/pamin-engine/tests/scratch_matched_costs.rs
  rm -f "$scratch"
  trap 'rm -f "$scratch"' EXIT HUP INT TERM
  curl -fsSL "$3" -o "$scratch"
  python3 - "$scratch" "$4" <<'PYSHA'
import hashlib, sys
from pathlib import Path
if not hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == sys.argv[2]:
    raise RuntimeError('reproduction integrity check failed')
PYSHA
  export CARGO_TARGET_DIR=/private/tmp/pamin-cost-reproduction-target
  cargo test --locked --offline -p pamin-engine --test scratch_matched_costs --no-run --message-format=json > /private/tmp/cost-build.jsonl
  python3 - /private/tmp/cost-build.jsonl "$5" /private/tmp/pamin-cost-runtime <<'PYBUILD'
import hashlib, json, shutil, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
executables = [r['executable'] for r in rows if r.get('reason') == 'compiler-artifact' and r.get('target', {}).get('name') == 'scratch_matched_costs' and r.get('executable')]
if not len(executables) == 1:
    raise RuntimeError('reproduction integrity check failed')
paths = [p.removeprefix('native=') for r in rows if r.get('reason') == 'build-script-executed' and 'zvec-rust-sys' in r.get('package_id', '') for p in r.get('linked_paths', []) if p.startswith('native=')]
libraries = [Path(p) / 'libzvec_c_api.dylib' for p in paths if (Path(p) / 'libzvec_c_api.dylib').is_file()]
if not libraries:
    raise RuntimeError('no resolved macOS zvec library in Cargo build-script output')
hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}
if not len(hashes) == 1:
    raise RuntimeError('ambiguous zvec linked-library identity')
source = libraries[0]
sha = next(iter(hashes))
import subprocess
ort_paths = [p.removeprefix('native=') for r in rows if r.get('reason') == 'build-script-executed' and 'ort-sys' in r.get('package_id', '') for p in r.get('linked_paths', []) if p.startswith('native=')]
needed = []
for line in subprocess.check_output(['otool', '-L', executables[0]], text=True).splitlines()[1:]:
    dependency = line.strip().split(' (', 1)[0]
    if 'onnxruntime' not in dependency:
        continue
    absolute = Path(dependency)
    candidates = [absolute] if absolute.is_absolute() and absolute.is_file() else [Path(directory) / absolute.name for directory in ort_paths if (Path(directory) / absolute.name).is_file()]
    if not candidates:
        raise RuntimeError('dynamic ONNX Runtime dependency not resolved by Cargo')
    if len({hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}) != 1:
        raise RuntimeError('ambiguous ONNX Runtime dependency')
    needed.append(candidates[0])
for directory in {path.parent for path in needed}:
    needed.extend((path for path in directory.glob('libonnxruntime*.dylib') if path.is_file()))
runtime = Path(sys.argv[3])
runtime.mkdir(parents=True, exist_ok=True)
loaded = runtime / source.name
if loaded.exists():
    if not hashlib.sha256(loaded.read_bytes()).hexdigest() == sha:
        raise RuntimeError('runtime directory contains a different zvec library')
else:
    shutil.copy2(source, loaded)
executable_sha = hashlib.sha256(Path(executables[0]).read_bytes()).hexdigest()
shutil.copy2(executables[0], sys.argv[2])
if hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() != executable_sha:
    raise RuntimeError('executable changed during freezing')
receipt = {'sha256': sha, 'resolved_source': str(source), 'runtime_path': str(loaded), 'onnxruntime': [], 'executable_sha256': executable_sha}
for library in dict.fromkeys(needed):
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    destination = runtime / library.name
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
        raise RuntimeError('runtime directory contains a different ONNX Runtime/provider library')
    if not destination.exists():
        shutil.copy2(library, destination)
    receipt['onnxruntime'].append({'sha256': digest, 'runtime_path': str(destination), 'resolved_source': str(library)})
Path(sys.argv[2] + '.zvec.json').write_text(json.dumps(receipt, indent=2))
PYBUILD
)
build_cost_binary "$source_root/main" 315c10242ddf7a1cec3bccbf550a942320e09557 https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/b1054c02a237de3f8ecb8e7252fbc562526d6311/pamin-main-cost-harness.rs 253a1546bba55ff9bb3733c60a790d489565729b37edd259b8a6e6b8cfd373cc /private/tmp/pamin-cost-frozen-main
build_cost_binary "$source_root/prototype" 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27 https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/451f9cd328dbaa636be72909291b5a9dd97419c7/pamin-new-cost-harness-reconstructed.rs b2f170bd06e488311ec4c1b75d8ded4bb1a56e728ac12802fa47f7d346eccee8 /private/tmp/pamin-cost-frozen-prototype
build_cost_binary "$source_root/persisted" 0f023be6a8d8d070a971f7e590ccccff2c3292bb https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/2faf59eecb05cebd9376d24f157403e7057a3121/pamin-persist-cost-harness.rs 2f697a84df87277b65091dd7bf633bec179c167c4a4c50b66d50995e2ba6eaad /private/tmp/pamin-cost-frozen-persisted
)
```

With all builds stopped, these are the seven per-arm invocations; each runs three fresh processes. The outer environment also clears graph/search overrides missing from the historical worker. Do not run arms in parallel; the executable loop rotates their order across process blocks for a new comparison. This listing reproduces recorded arm settings, not controlled historic machine load or CoreML compilation warmth.

```sh
python3 - <<'PYRUN'
import hashlib, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
arms = [
    ("main-cpu", "main", "accuracy", "cpu"),
    ("main-auto", "main", "accuracy", "auto"),
    ("main-auto-repeat", "main", "accuracy", "auto"),
    ("new-cpu", "prototype", "accuracy", "cpu"),
    ("dual-cpu", "prototype", "dual_accuracy", "cpu"),
    ("new-auto", "prototype", "accuracy", "auto"),
    ("new-auto-persist-hit", "persisted", "accuracy", "auto"),
]
env = os.environ.copy()
for key in ("PAMIN_SEARCH_EFFORT", "PAMIN_PREPARED", "PAMIN_FUSED_ATTENTION", "PYTHONOPTIMIZE", "ORT_DYLIB_PATH"):
    env.pop(key, None)
def run_case(name, binary, profile, policy, check_persisted=False, expected_events=None):
    worker_source = Path("/private/tmp/pamin-cost-worker.py").read_bytes()
    if hashlib.sha256(worker_source).hexdigest() != "677f12a00f42563ada52368f69b7e088fa67d07aab7f8f99905328d5fb4e5d6c":
        raise RuntimeError("retained worker changed before invocation")
    receipt = json.loads(Path(f"/private/tmp/pamin-cost-frozen-{binary}.zvec.json").read_text())
    if hashlib.sha256(Path(f"/private/tmp/pamin-cost-frozen-{binary}").read_bytes()).hexdigest() != receipt["executable_sha256"]:
        raise RuntimeError("frozen executable differs from its build receipt")
    for library in [receipt] + receipt["onnxruntime"]:
        if hashlib.sha256(Path(library["runtime_path"]).read_bytes()).hexdigest() != library["sha256"]:
            raise RuntimeError("native runtime bytes changed after build")
    subprocess.run([sys.executable, "-c", worker_source.decode("utf-8"), name,
        f"/private/tmp/pamin-cost-frozen-{binary}", profile, policy], env=env, check=True)
    log = Path(f"/private/tmp/pamin-cost-{name}.log").read_text()
    rerankers = re.findall(r'reranker loaded tier="([^"\n]+)" device="([^"\n]+)" maximum_tokens=(\d+)', log)
    embedders = re.findall(r'embedder loaded model="([^"\n]+)" device="([^"\n]+)"', log)
    expected_device = "cpu" if policy == "cpu" else "coreml"
    dual = "bge-m3-int8@2b34e84df040034d4b9eabb62383a87c18955822+pplx-0.6b@2c4d510dd4a732063c31a0f70193e35067b51fd8:pool-int8-single-v2-level4"
    expected_embedders = [] if binary == "main" else [(dual if profile == "dual_accuracy" else "gpahal/bge-m3-onnx-int8", "cpu")]
    if rerankers != [("accurate", expected_device, "256")] or embedders != expected_embedders:
        raise RuntimeError("model/device premise differs from retained arm")
    observed_events = (sum("complete model-call calibration" in line for line in log.splitlines()), sum("compute candidate" in line for line in log.splitlines()))
    if expected_events is not None and observed_events != expected_events:
        raise RuntimeError("calibration/rejection regime differs from retained arm")
    if not check_persisted: return True
    forbidden = ("complete model-call calibration", "compute candidate", "calibrated winner failed", "cached compute plan failed")
    return not any(marker in log for marker in forbidden)
for block in range(3):
    offset = 2 * block
    for arm, binary, profile, policy in arms[offset:] + arms[:offset]:
        persisted = arm == "new-auto-persist-hit"
        if persisted:
            # Unmeasured primers run immediately before each hit block, so a
            # short negative-plan expiry cannot invalidate a whole-loop setup.
            for attempt in range(3):
                if run_case(f"setup-{run_id}-persisted-{block}-{attempt}", binary, profile, policy, True):
                    break
            else:
                raise RuntimeError("persisted plan did not stabilize; do not label a hit arm")
        name = f"reproduced-{run_id}-{arm}-{block}"
        expected_events = (1, 1) if arm == "new-auto" else (0, 0)
        if not run_case(name, binary, profile, policy, persisted, expected_events):
            raise RuntimeError("measured persisted-hit recalibrated/rejected; discard outputs")
PYRUN
```

Each invocation writes `/private/tmp/pamin-cost-reproduced-RUN-ARM-BLOCK.jsonl` (warm product rows), `.log` (local-only diagnostic log) and `-process.json` (whole-process wall/user/system, binary SHA and maximum RSS). For persisted-hit, first establish a nonexpired plan using one unmeasured persisted setup process and check measured logs have zero calibration/rejection events. Do not publish raw logs: only the strictly whitelisted structured load events are public evidence. These commands target the retained macOS environment; other hosts need their native runtime-library setup and produce new measurements.

Reproduction outputs use a `reproduced-` prefix to avoid replacing the original local cost logs/rows. The source checkouts and shared reproduction target can be removed after retaining the new outputs and frozen executable hashes; historical public rows stay unchanged.

Each frozen executable has a `.zvec.json` receipt from Cargo’s actual `zvec-rust-sys` native link paths, including override/sibling/vendor selections. The worker loop verifies the runtime-library SHA before each process. Conflicting runtime files or differing linked copies fail closed. These receipts belong to the new reproduction, not a retroactive assertion about unrecorded historical library bytes.

The single guarded recipe uses explicit Python exceptions for integrity and arm premises, so optimized Python cannot disable them. The child environment removes PYTHONOPTIMIZE and ORT_DYLIB_PATH. Dynamic ONNX Runtime/provider libraries are captured from actual linked dependencies and rehashed before each process; static ORT needs no external DLL receipt.

Historical builds require provisioned dependencies and `--locked --offline` with a clean tracked tree. Each process executes the freshly checksum-verified worker bytes rather than reopening its mutable pathname. All measured arms also match their retained calibration/rejection counts (initial new-auto: 1/1; other arms: 0/0); unmeasured primers permit calibration while establishing the hit regime.
