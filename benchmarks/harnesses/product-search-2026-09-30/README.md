# Generating sources retained from the complete product comparison

These are patches for the actual measurement wrappers and their controllers,
not a replacement
retrieval pipeline. They call the product's `Engine::search_reranked`. Templates
change only machine-specific path roots; sources.json retains original and
template SHA256 values. Main CPU/default-auto link the main runtime a249554;
candidate links the frozen runtime 0a222d1 (the equivalent restacked runtime is
fbbcc26). Both use the shared scoring/statistics owners from the evidence repo.
Do not attribute later runtime changes to these recorded arms.

Reconstruct each Rust wrapper from the immutable canonical source at
`c9ab0245940bb6c40059ad812ed061502e18810d`:

```sh
git show c9ab0245940bb6c40059ad812ed061502e18810d:crates/pamin-engine/tests/crosslingual.rs > crosslingual-base.rs
patch -o lib.rs crosslingual-base.rs < <arm>.patch
```

Each patch reconstructs the exact retained wrapper template; this was checked
byte-for-byte. Keeping patches reuses the canonical harness instead of adding
three maintained copies. Materialize the reconstructed wrapper and each other
`.in` file by replacing `${RUN_ROOT}` with a fresh scratch
root, `${MAIN_REPO}` with a clean a249554 checkout, and `${EVIDENCE_REPO}` with
the evidence checkout. For the candidate Cargo manifest, its runtime path must
point to a clean 0a222d1/equivalent runtime checkout, while included evaluation
modules retain the evidence checkout. Use a distinct Cargo target directory per
runtime; archive `Cargo.lock` and freeze each executable before any later build.
The original project directories under the scratch root are pamin-product-main,
pamin-product-main-auto, pamin-product-acceptance and pamin-product-score.
Rename each corresponding rs/Cargo template to lib.rs/Cargo.toml; score.rs and
compare.rs keep their names. The controllers expect the original scratch layout.

Build each wrapper with `cargo test --release --no-run --manifest-path
<scratch-project>/Cargo.toml`. Obtain its executable from Cargo's JSON compiler
artifact output (`--message-format=json`), then copy it to the frozen filename
in that controller. Build the two score/compare binaries with `cargo build
--release --manifest-path <scratch-root>/pamin-product-score/Cargo.toml` and
freeze them to pamin-score-frozen and pamin-compare-frozen.

Use a fresh corpus/model workspace at `<scratch-root>/pamin-perf58`, provision
models with the pinned preparation command in docs/measured.md, and install the
selected runtime's zvec sidecar in `<scratch-root>/pamin-gpu-research/runtime`.
The candidate controller runs first; the main-default controller consumes its
completed rankings/judgments for paired scoring. To invoke an arm directly:

```sh
env -u HF_HOME PAMIN_EVAL_HOME=<workspace> PAMIN_PROFILE=accuracy \
  PAMIN_DEVICE=auto PAMIN_VECTOR_INDEX=memory \
  PAMIN_PRODUCT_ROWS=<output-prefix> DYLD_LIBRARY_PATH=<zvec-directory> \
  <frozen-wrapper> --exact search_reaches_across_languages --ignored --nocapture
```

CPU uses PAMIN_DEVICE=cpu. Do not run arms concurrently. The retained controllers
collect whole accurate-plus-off process user/system time using wait4 and sample
process RSS once per second; these are not search-only CPU or service/device
costs. Timed query calls in emitted rows are the accurate search-only clocks.
Controllers require at least 10 GiB free disk and cap process RSS at 8 GiB.
Resolve the native library path and runtime/model identities for each rerun;
original machine paths, cache keys and timings are not reproducibility premises.
A rerun is fresh evidence, not a demand to reproduce shared-host seconds exactly.
