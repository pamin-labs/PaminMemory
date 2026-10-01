# Dual-vector and shared compute prototype evidence — 2026-10-01

## Precision and scope

The product boundary is `Engine::search_reranked`: memory index, named passages, accurate reranker, limit 60, default semantic depth 50 / graph depth 2. The paired CPU-only XQuAD-R comparison has 13,014 documents and the same 1,190 language-rotated queries. Corpus files are fingerprinted in `corpus-files.json`; raw rankings, relevance/exclusions, the predeclared rule, summaries and immutable execution-provenance correction are retained here. No fusion weights were fitted on this corpus.

Baseline CPU executable: `555846c`; dual executable: `45a3bcc601b2f9ca7d23d0bf49c6c95b3284f73e`. The candidate raw manifest sampled a later git HEAD after ingestion; the correction sidecar identifies the frozen executed binary and source. Do not treat that late field as execution identity. A 66-query true-main `315c102` control matched the baseline rankings exactly.

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

This independently recomputes paired nDCG/recall, 240-paragraph shared sign flips/cluster bootstrap (exploratory post-hoc correction) and all 21 cost blocks from raw rows, asserts paired IDs/corpus sizes and checks each reported percentile/RSS value. Timing program sources remain scratch-only; source/binary hashes and execution identities are retained. The historical candidate identifiers `503bd9e` and `0f023be` are provenance labels: their Git objects are unavailable in the reviewed checkout, and public tag availability is not established. Candidate source reproduction is currently unavailable from this package. Do not substitute the later published source and label it as either historical candidate. Re-running the live workload requires a provisioned unprivileged PostgreSQL workspace and the fingerprinted models/data, 66 fixed query indices `18*i` (i=0..65), warm query 1189, and the product settings above. Freeze executables before launch and keep builds out of timing runs. Historical total ingest time was measured on a shared host with intervening builds and is not a causal speed comparison.

The original query-level statistical summary is retained as historical, not a valid independent-question acceptance analysis. `cluster-inference.json` records 100,000 shared flips and cluster-bootstrap draws (PCG64 seed 20261001): cross Holm p≈0.000040; same-language p≈0.153978, lower 95% delta bound -0.001348, still above -0.005. Point metrics and raw rankings are unchanged.

## Fetch measured source refs in a shallow review checkout

A pull-request checkout does not automatically include unrelated benchmark tags.
The retained rows and verifier remain available independently of those missing compilation inputs. No candidate source archive or complete compilation-input manifest is retained here. Publishing private history is not required or authorized by this recipe; a future sanitized exact source subset would need per-file hashes and an explicit build scope before candidate commands can be called runnable.

## Conditional scratch reproduction

The [versioned public scratch bundle](https://gist.github.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/72859f283b963c0294875dbef2a4e3a00209f9b5) contains separately retained main and persisted harness sources. The 503bd9e harness is a disclosed reconstruction of the persisted source with only its literal execution identity changed; its original measurement-time source bytes are not archived here. Re-running that reconstruction cannot certify those missing original bytes or historical compiler inputs. Use the source matching the selected revision, rather than labeling the persisted harness as a main/503 execution.

A provisioned, unprivileged `/private/tmp/pamin-dual-product-eval` must already contain the fingerprinted corpus, model cache, PostgreSQL and complete `dual-product-accuracy-24ad7f1862182925` / `dual-product-dual_accuracy-24ad7f1862182925` indexes. The harness aborts on an empty/incomplete corpus. Each source revision needs exact available compilation inputs. The candidate recipes below are conditional and must fail if those inputs are unavailable; only main source reachability is established here. Run the following setup and subsequent helpers inside one disposable subshell for each available revision. This helper downloads only the selected public scratch source to its ignored test location inside the fresh external export and checks its frozen SHA-256; it does not provision models or PostgreSQL:

Before defining the helpers, create a fresh external export; never sweep or overwrite an existing checkout's ignored test files. Enclose **all** setup, downloads, builds and runs for that revision in this subshell. Its trap removes only the directory it successfully created, including on command failure, HUP, INT or TERM. SIGKILL/power loss cannot run traps; inspect and remove only your recorded owned directory after such an interruption. Outputs intended for retention remain outside that directory.

```sh
(
set -eu
cost_repository=$(git rev-parse --show-toplevel)
cost_checkout_revision=$(git rev-parse HEAD)
cost_parent=$(cd "${TMPDIR:-/tmp}" && pwd -P)
case "$cost_parent/" in "$cost_repository/"*) exit 1 ;; esac
cost_owned_root=$(mktemp -d "$cost_parent/pamin-cost.XXXXXXXX")
readonly cost_owned_root cost_repository cost_checkout_revision
printf '%s\n' "$cost_owned_root" > "$cost_owned_root/.owned-cost-export"
cleanup_cost_export() {
  test ! -L "$cost_owned_root" && test -f "$cost_owned_root/.owned-cost-export" || return 1
  test "$(cat "$cost_owned_root/.owned-cost-export")" = "$cost_owned_root" || return 1
  rm -rf -- "$cost_owned_root"
}
trap cleanup_cost_export EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$cost_owned_root/source"
git archive --format=tar --output="$cost_owned_root/source.tar" "$cost_checkout_revision"
tar -xf "$cost_owned_root/source.tar" -C "$cost_owned_root/source"
rm "$cost_owned_root/source.tar"
cd "$cost_owned_root/source"
# Define and use the helpers/selected-revision commands below here.
# Close with ')' only after that revision's downloads/builds/runs finish.
```

```sh
cost_harness_identity() {
  cost_revision=$1
  test "$cost_checkout_revision" = "$cost_revision" || return 1
  case "$cost_revision" in
    315c10242ddf7a1cec3bccbf550a942320e09557)
      cost_url=https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/b1054c02a237de3f8ecb8e7252fbc562526d6311/pamin-main-cost-harness.rs
      cost_sha=253a1546bba55ff9bb3733c60a790d489565729b37edd259b8a6e6b8cfd373cc ;;
    503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27)
      cost_url=https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/451f9cd328dbaa636be72909291b5a9dd97419c7/pamin-new-cost-harness-reconstructed.rs
      cost_sha=b2f170bd06e488311ec4c1b75d8ded4bb1a56e728ac12802fa47f7d346eccee8 ;;
    0f023be6a8d8d070a971f7e590ccccff2c3292bb)
      cost_url=https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/2faf59eecb05cebd9376d24f157403e7057a3121/pamin-persist-cost-harness.rs
      cost_sha=2f697a84df87277b65091dd7bf633bec179c167c4a4c50b66d50995e2ba6eaad ;;
    *) return 1 ;;
  esac
  cost_source=crates/pamin-engine/tests/scratch_matched_costs.rs
}
install_cost_harness() {
  cost_harness_identity "$1" || return 1
  test ! -e "$cost_source" && test ! -L "$cost_source" || return 1
  curl -fsSL "$cost_url" -o "$cost_source" || return 1
  printf '%s  %s\n' "$cost_sha" "$cost_source" | shasum -a 256 -c -
}
run_cost() {
  cost_arm=$1; cost_revision=$2; cost_profile=$3; cost_policy=$4; cost_block=$5
  test "$cost_checkout_revision" = "$cost_revision" || return 1
  case "$cost_policy" in cpu|auto) ;; *) return 1 ;; esac
  cost_harness_identity "$cost_revision" || return 1
  printf '%s  %s\n' "$cost_sha" "$cost_source" | shasum -a 256 -c - || return 1
  test "$cost_block" -ge 0 && test "$cost_block" -le 2 || return 1
  cost_rows=/private/tmp/reproduced-cost-${cost_arm}-${cost_block}.jsonl
  if test "$cost_policy" = cpu; then
    env -u HF_HOME -u PAMIN_DEVICE -u PAMIN_SEARCH_EFFORT -u PAMIN_PREPARED -u PAMIN_FUSED_ATTENTION -u PAMIN_RERANK_DEPTH -u PAMIN_RERANK_MAX_TOKENS -u PAMIN_RERANK_BATCH -u PAMIN_RERANK_BATCH_TOKENS -u PAMIN_INFERENCE_THREADS PAMIN_DEVICE=cpu PAMIN_EVAL_HOME=/private/tmp/pamin-dual-product-eval PAMIN_PROFILE="$cost_profile" MATCHED_COST_ROWS="$cost_rows" cargo test -p pamin-engine --test scratch_matched_costs matched_product_costs -- --exact --ignored --nocapture
  else
    env -u HF_HOME -u PAMIN_DEVICE -u PAMIN_SEARCH_EFFORT -u PAMIN_PREPARED -u PAMIN_FUSED_ATTENTION -u PAMIN_RERANK_DEPTH -u PAMIN_RERANK_MAX_TOKENS -u PAMIN_RERANK_BATCH -u PAMIN_RERANK_BATCH_TOKENS -u PAMIN_INFERENCE_THREADS PAMIN_EVAL_HOME=/private/tmp/pamin-dual-product-eval PAMIN_PROFILE="$cost_profile" MATCHED_COST_ROWS="$cost_rows" cargo test -p pamin-engine --test scratch_matched_costs matched_product_costs -- --exact --ignored --nocapture
  fi
}
# In each checkout install its matching source before using the corresponding rows.
# Repeat blocks 0, 1, 2 as fresh processes, rotating arm order; do not run these builds during timings.
# Example for main: install_cost_harness 315c10242ddf7a1cec3bccbf550a942320e09557
block=0
run_cost main-cpu 315c10242ddf7a1cec3bccbf550a942320e09557 accuracy cpu "$block"
run_cost new-cpu 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27 accuracy cpu "$block"
run_cost dual-cpu 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27 dual_accuracy cpu "$block"
run_cost main-auto 315c10242ddf7a1cec3bccbf550a942320e09557 accuracy auto "$block"
run_cost new-auto 503bd9ee61a4d9fc6e7a9e16ae4d9a0537494f27 accuracy auto "$block"
run_cost main-auto-repeat 315c10242ddf7a1cec3bccbf550a942320e09557 accuracy auto "$block"
run_cost new-auto-persist-hit 0f023be6a8d8d070a971f7e590ccccff2c3292bb accuracy auto "$block"
# The enclosing owned-export EXIT trap removes the scratch source on every catchable exit.
```

These are warm-search reproduction commands, not a new executed measurement or a complete build certificate. Establish a nonexpired persisted plan in a separate setup process before the persisted-hit blocks, then verify their retained events contain no calibration/rejection. Do not label expired quarantine or uncontrolled CoreML compilation as a controlled hit. Build first and freeze the executable before invoking the retained worker for process-resource columns; the commands above intentionally preserve the historical Cargo-based scratch entry point and exclude its compilation from reported query timers.

The frozen process-worker source records whole-process wall/user/system time, binary SHA and RSS, with its macOS runtime-library path declared explicitly. Use its four arguments `case-name frozen-executable profile policy` when reproducing those process columns; warm search columns come from the Rust rows. Provisioning/downloading is not part of the timed search. Native device/service memory remains unmeasured. Strictly redacted event streams underlying cache/device proof are retained under `logs/` and hashed/recounted by the verifier. The exact top-level schema is `source_sha256`, `redaction`, and `events`; hashes must be lowercase SHA-256, the redaction marker is fixed, and extra fields reject even after resealing. Every original line is mapped to a fixed enum; paths, free text and content are excluded. Loaded events require a device from `cpu`, `cuda`, `coreml`, `directml`, or `npu`; unrelated events reject device fields. Loaded event/device pairs are independently compared with the published loaded-device summary, including order and count. Loaded summaries must identify the correct single/dual model for the arm and the accurate reranker with maximum_tokens=256; these values are independently pinned rather than accepted because two projections agree. Single-model summaries name a repository, not a separately logged revision. They must also match independent frozen per-arm/process role/device sequences in the verifier, including the reversed load order in persisted-hit process 2. Historical main logs contain no embedder-loaded event; no missing event or physical ANE/GPU placement is inferred. This verifies the retained projection and its frozen premises, not completeness or correct classification against externally retained original logs. Each cost block's published manifest must equal its validated raw manifest, and all three `main-cpu` rankings must match the full baseline at every selected query `18*i`. Original source hashes are retained; original logs stay local.

The original `predeclared.json` is immutable. It specified independent question-level inference; the corrected cluster analysis was chosen after the results were observed. The new p-value supports an exploratory signal, not a preregistered acceptance claim. Confirm on an independent corpus before changing defaults.


### Whole-process resource reproduction

The original [public Python worker](https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/a8327acdbaecfa651ddd78a293df7dae43b223da/pamin-cost-worker.py) is separately pinned; it targets the retained macOS workspace and loader directory. Download and verify it before any workload:

```sh
curl -fsSL https://gist.githubusercontent.com/JasonXuDeveloper/24e8f310edc69ed9259c1f2ab658398f/raw/a8327acdbaecfa651ddd78a293df7dae43b223da/pamin-cost-worker.py -o "$cost_owned_root/pamin-cost-worker.py"
printf '%s  %s\n' 677f12a00f42563ada52368f69b7e088fa67d07aab7f8f99905328d5fb4e5d6c "$cost_owned_root/pamin-cost-worker.py" | shasum -a 256 -c -
```

Within the same owned-export subshell, install and verify its corresponding Rust source using the helper above. Build before timing and extract exactly that test executable from Cargo's structured output. The following example freezes the persisted checkout; use distinct `main` and `prototype` output names in their corresponding checkouts:

```sh
cost_harness_identity 0f023be6a8d8d070a971f7e590ccccff2c3292bb
printf '%s  %s\n' "$cost_sha" "$cost_source" | shasum -a 256 -c -
cargo test -p pamin-engine --test scratch_matched_costs --no-run --offline --locked --message-format=json > /private/tmp/cost-build.jsonl
python3 - /private/tmp/cost-build.jsonl /private/tmp/pamin-cost-frozen-persisted <<'PYBUILD'
import json, shutil, sys
from pathlib import Path
rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
executables = [r["executable"] for r in rows if r.get("reason") == "compiler-artifact" and r.get("target", {}).get("name") == "scratch_matched_costs" and r.get("executable")]
assert len(executables) == 1
shutil.copy2(executables[0], sys.argv[2])
PYBUILD
```

Retain the actual native build/runtime settings and hashes separately. The historical compiler flags and complete transitive build certificate are not reconstructed by this command. The worker sets `DYLD_LIBRARY_PATH=/private/tmp/pamin-cost-runtime`; provision that directory with the actual dynamic dependencies used by these frozen executables, including their selected Zvec dylib and any required ORT dylibs. Do not infer a library's identity from its filename or silently substitute a different runtime. Build, copying and provisioning finish before running the worker; the commands do not download models or initialize PostgreSQL.

With all builds stopped, use fresh case names and existing completed indexes. The original worker clears reranker/intra-op overrides; the outer environment additionally clears graph/search overrides absent from that worker. Each invocation is a fresh Python/native process. The controller rotates the seven source/profile/policy combinations across blocks:

```sh
python3 - "$cost_owned_root/pamin-cost-worker.py" <<'PYRUN'
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
for block in range(3):
    offset = 2 * block
    for arm, binary, profile, policy in arms[offset:] + arms[:offset]:
        subprocess.run([
            sys.executable, sys.argv[1], f"reproduced-{arm}-{block}",
            f"/private/tmp/pamin-cost-frozen-{binary}", profile, policy,
        ], env=env, check=True)
PYRUN
```

The worker writes `/private/tmp/pamin-cost-reproduced-ARM-BLOCK.jsonl`, a local diagnostic `.log` and `-process.json`. Existing row filenames abort through `create_new`; preserve historical files and choose a new case prefix for another rerun. Establish the nonexpired persisted plan in one unmeasured setup process before hit blocks and check the measured logs for zero calibration/rejection events. Keep raw diagnostic logs private.

These commands are a reconstruction recipe, not new executed evidence. Historical shared-machine load and CoreML compilation warmth cannot be recreated or claimed controlled. The original worker's wall interval includes log setup, native startup/model warm/search and its post-run executable hashing; child user/system time excludes the Python wrapper/hash CPU and includes reaped helper children such as `ps`. Its maximum RSS is the child-resource high-water measure, not an aggregate of PostgreSQL, concurrent processes or accelerator services. Product-call timings and sampled native RSS remain in Rust rows. Other hosts require appropriate native loader settings and produce new measurements; native device/service memory, whole-system cost and exact historical build reproduction remain N/A.

The focused verifier regression suite exercises retained cost rows and redacted events without invoking the 100,000-draw precision calculation, models or native helpers:

```sh
python3 -B -m unittest discover -s benchmarks -p test_verify_compute_evidence.py -v
```

This evidence-only correction changes no product behavior and introduces no new accuracy, latency, memory or disk measurement; new before/after performance values are N/A. Original raw rows, precision/cost values, proof summaries and redacted event artifacts remain unchanged.
