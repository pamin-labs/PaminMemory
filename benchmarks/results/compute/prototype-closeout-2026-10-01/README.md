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

The [frozen scratch sources](https://gist.github.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/eff4c8c55e69c341f00517046aa9e0d6e3aba751) remain outside the tracked tree, with SHA-256 in `persisted-cost-identity.json`. A provisioned, unprivileged `/private/tmp/pamin-dual-product-eval` must already contain the fingerprinted corpus, model cache, PostgreSQL and complete `dual-product-accuracy-24ad7f1862182925` / `dual-product-dual_accuracy-24ad7f1862182925` indexes. This command aborts rather than timing an empty corpus. In a separate checkout of the retained `bench/persisted-cost-2026-10-01` tag:

```sh
# Use a fresh shell for this block so cleanup also runs on failure/interruption.
set -eu
scratch=crates/pamin-engine/tests/scratch_matched_costs.rs
rm -f "$scratch"
trap 'rm -f "$scratch"' EXIT HUP INT TERM
curl -fsSL https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/2faf59eecb05cebd9376d24f157403e7057a3121/pamin-persist-cost-harness.rs -o "$scratch"
shasum -a 256 crates/pamin-engine/tests/scratch_matched_costs.rs
# Expected: 2f697a84df87277b65091dd7bf633bec179c167c4a4c50b66d50995e2ba6eaad
# Run one arm at a time, builds/other experiments stopped; rotate arms and use fresh processes.
env -u HF_HOME -u PAMIN_DEVICE -u PAMIN_SEARCH_EFFORT -u PAMIN_PREPARED -u PAMIN_FUSED_ATTENTION -u PAMIN_RERANK_DEPTH -u PAMIN_RERANK_MAX_TOKENS -u PAMIN_RERANK_BATCH -u PAMIN_RERANK_BATCH_TOKENS -u PAMIN_INFERENCE_THREADS PAMIN_EVAL_HOME=/private/tmp/pamin-dual-product-eval PAMIN_PROFILE=accuracy MATCHED_COST_ROWS=/private/tmp/reproduced-cost.jsonl cargo test -p pamin-engine --test scratch_matched_costs matched_product_costs -- --exact --ignored --nocapture
rm crates/pamin-engine/tests/scratch_matched_costs.rs
```

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

For every row, use its checkout/profile/policy and run three fresh processes, rotating arm order. Common command (auto):

```sh
env -u HF_HOME -u PAMIN_DEVICE -u PAMIN_SEARCH_EFFORT -u PAMIN_PREPARED -u PAMIN_FUSED_ATTENTION -u PAMIN_RERANK_DEPTH -u PAMIN_RERANK_MAX_TOKENS -u PAMIN_RERANK_BATCH -u PAMIN_RERANK_BATCH_TOKENS -u PAMIN_INFERENCE_THREADS PAMIN_EVAL_HOME=/private/tmp/pamin-dual-product-eval PAMIN_PROFILE=accuracy MATCHED_COST_ROWS=/private/tmp/cost-arm-block.jsonl cargo test -p pamin-engine --test scratch_matched_costs matched_product_costs -- --exact --ignored --nocapture
```

For CPU rows add `PAMIN_DEVICE=cpu` after the `env -u` options. For `dual-cpu` set `PAMIN_PROFILE=dual_accuracy` as well. Build first, freeze/copy the executable, and invoke the retained process-worker with `case-name frozen-executable profile policy` for process resource columns. The cache-hit arm must have zero calibration/rejection events; one setup process may establish the plan but is not a measured hit block. Do not label expired quarantine or cold CoreML compilation as a controlled hit.

### Whole-process resource reproduction

The worker is Python, so it needs no native build. Fetch the retained source and syntax-check it (this does not run a model):

```sh
curl -fsSL https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/a8327acdbaecfa651ddd78a293df7dae43b223da/pamin-cost-worker.py -o /private/tmp/pamin-cost-worker.py
python3 -m py_compile /private/tmp/pamin-cost-worker.py
```

Build all three sources explicitly in fresh detached checkouts. The installer checks HEAD and source SHA, removes stale ignored tests, and registers cleanup before downloading/building. All builds share one target directory to limit disk usage; each executable is copied before the next build. Existing checkout paths cause `git worktree add` to fail instead of overwriting them.

```sh
# Run in a fresh Bash shell from the repository. The PostgreSQL/model/index
# provisioning described above must already exist, under an unprivileged user.
set -euo pipefail
git worktree add --detach /private/tmp/pamin-cost-src-main 315c10242ddf7a1cec3bccbf550a942320e09557
git worktree add --detach /private/tmp/pamin-cost-src-prototype 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27
git worktree add --detach /private/tmp/pamin-cost-src-persisted 0f023be6a8d8d070a971f7e590ccccff2c3292bb
build_cost_binary() (
  set -euo pipefail
  cd "$1"
  test "$(git rev-parse HEAD)" = "$2"
  scratch=crates/pamin-engine/tests/scratch_matched_costs.rs
  rm -f "$scratch"
  trap 'rm -f "$scratch"' EXIT HUP INT TERM
  curl -fsSL "$3" -o "$scratch"
  python3 - "$scratch" "$4" <<'PYSHA'
import hashlib, sys
from pathlib import Path
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == sys.argv[2]
PYSHA
  export CARGO_TARGET_DIR=/private/tmp/pamin-cost-reproduction-target
  cargo test -p pamin-engine --test scratch_matched_costs --no-run --message-format=json > /private/tmp/cost-build.jsonl
  python3 - /private/tmp/cost-build.jsonl "$5" <<'PYBUILD'
import json, shutil, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
executables = [r["executable"] for r in rows if r.get("reason") == "compiler-artifact" and r.get("target", {}).get("name") == "scratch_matched_costs" and r.get("executable")]
assert len(executables) == 1
shutil.copy2(executables[0], sys.argv[2])
PYBUILD
  mkdir -p /private/tmp/pamin-cost-runtime
  find "$CARGO_TARGET_DIR/debug" -name libzvec_c_api.dylib -type f -exec cp {} /private/tmp/pamin-cost-runtime/ \;
)
build_cost_binary /private/tmp/pamin-cost-src-main 315c10242ddf7a1cec3bccbf550a942320e09557 https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/b1054c02a237de3f8ecb8e7252fbc562526d6311/pamin-main-cost-harness.rs 253a1546bba55ff9bb3733c60a790d489565729b37edd259b8a6e6b8cfd373cc /private/tmp/pamin-cost-frozen-main
build_cost_binary /private/tmp/pamin-cost-src-prototype 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27 https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/451f9cd328dbaa636be72909291b5a9dd97419c7/pamin-new-cost-harness-reconstructed.rs b2f170bd06e488311ec4c1b75d8ded4bb1a56e728ac12802fa47f7d346eccee8 /private/tmp/pamin-cost-frozen-prototype
build_cost_binary /private/tmp/pamin-cost-src-persisted 0f023be6a8d8d070a971f7e590ccccff2c3292bb https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/2faf59eecb05cebd9376d24f157403e7057a3121/pamin-persist-cost-harness.rs 2f697a84df87277b65091dd7bf633bec179c167c4a4c50b66d50995e2ba6eaad /private/tmp/pamin-cost-frozen-persisted
```

With all builds stopped, these are the seven per-arm invocations; each runs three fresh processes. The outer environment also clears graph/search overrides missing from the historical worker. Do not run arms in parallel; the executable loop rotates their order across process blocks for a new comparison. This listing reproduces recorded arm settings, not controlled historic machine load or CoreML compilation warmth.

```sh
python3 - <<'PYRUN'
import os, subprocess, sys
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
for key in ("PAMIN_SEARCH_EFFORT", "PAMIN_PREPARED", "PAMIN_FUSED_ATTENTION"):
    env.pop(key, None)
# The retained worker clears HF_HOME/device and all five reranker/thread overrides.
for block in range(3):
    offset = 2 * block
    for arm, binary, profile, policy in arms[offset:] + arms[:offset]:
        subprocess.run([
            sys.executable, "/private/tmp/pamin-cost-worker.py", f"reproduced-{arm}-{block}",
            f"/private/tmp/pamin-cost-frozen-{binary}", profile, policy,
        ], env=env, check=True)
PYRUN
```

Each invocation writes `/private/tmp/pamin-cost-reproduced-ARM-BLOCK.jsonl` (warm product rows), `.log` (local-only diagnostic log) and `-process.json` (whole-process wall/user/system, binary SHA and maximum RSS). For persisted-hit, first establish a nonexpired plan using one unmeasured persisted setup process and check measured logs have zero calibration/rejection events. Do not publish raw logs: only the strictly whitelisted structured load events are public evidence. These commands target the retained macOS environment; other hosts need their native runtime-library setup and produce new measurements.

Reproduction outputs use a `reproduced-` prefix to avoid replacing the original local cost logs/rows. The source checkouts and shared reproduction target can be removed after retaining the new outputs and frozen executable hashes; historical public rows stay unchanged.
