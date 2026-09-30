# Audit and rerun

Audit without models or writes:

```sh
python3 benchmarks/results/inference/local-closeout-2026-09-30/verify.py
python3 benchmarks/results/inference/local-closeout-2026-09-30/summarize.py
```

The verifier recomputes medians, nearest-rank p50/p95 of per-query three-round medians, process wall/CPU/RSS and four process-round-block sign-flip diagnostics from 432 rows and 18 resource records. It also checks provider/cache-miss attestations and file hashes. It writes no retained artifacts.

To rerun, use an isolated checkout at public commit `be329b00636987781689ec6ef0c4fc48611cdf1d` and apply `sources/experiment.patch`. This patch is an experimental appendix, never default code. Provision the complete pinned XQuAD-R accuracy index first; the harness asserts 13,014 indexed documents and completeness 1.0, and will not silently measure an empty index. Model cache identities, source exports, query IDs/text hashes/order, compiler, CPU configuration and requested device plans are in manifest.json. Cached revisions and OS were collected after execution; they are not historical pre/post attestations. Warmups are query positions 1..8, sample positions 0..1190 step 50. No model downloads or index rebuilds belong in measured search durations.

Build the integration executable:

```sh
cargo test --release -p pamin-engine --test crosslingual scratch_backend_repeat --no-run
```

Copy the printed executable to `$RUN_ROOT/benchmark`. Use a fresh output directory and retained `sources/run.py.in` (replace `${REPO}`, `${EVAL_HOME}`, `${RUN_ROOT}` with your checkout, populated evaluation cache and output directory). Update the native library path to your build output. Run the rendered Python program; it executes all 18 arms in the exact three rotations retained in manifest.json, one at a time, with 10 GiB disk and 25% memory headroom guards. It verifies a frozen executable, actual reranker device, 24 rows and per-query fresh scoring. Requested ALL is not proof every hardware unit executes the graph; Fast CoreML logged E5RT/ANE shape warnings.

Models use CPU INT8 vs Accurate CoreML FP16 / Fast CoreML FP32. These are different complete exports, not an isolated hardware arithmetic comparison. The BGE-M3 embedder remains CPU in all arms; only reranker provider/export changes. All results were collected on a shared workstation. Do not use the sample nDCG columns for full-corpus precision acceptance.

Result rows contain only public XQuAD topic IDs, scores, counters and timings. Local paths are replaced in source/attestation files. No private notes, credentials, model binaries or raw device traces are published.

## FP64 appendices

The same verifier also recomputes the full FP64 section: 900 generated-vector rows (recall from gold nearest IDs; unchanged candidates/lists), 3,570 XQuAD rows and 1,446 MIRACL rows (nDCG from retained public judgement keys and rankings). It compares every query score before aggregating; corpus sizes and mode coverage are asserted. The 50k generator uses seed 0x5eed and independent original-f32 and stored-FP16 cosine oracles.

Use a separate checkout at the manifest source commit, apply `fp64/sources/experiment.patch` (not the backend patch), and copy `fp64/sources/50k.rs` to `crates/pamin-engine/tests/scratch_fp64_50k.rs`. Set PAMIN_FP64_MODEL to the retained cosine-f64.onnx. Run each separately in release mode:

```sh
cargo test --release -p pamin-engine --test scratch_fp64_50k scratch_fp64_50k -- --exact --ignored --nocapture
cargo test --release -p pamin-engine --test crosslingual scratch_fp64_product -- --exact --ignored --nocapture
cargo test --release -p pamin-engine --test monolingual scratch_fp64_miracl -- --exact --ignored --nocapture
```

Set FP64_OUTPUT to a fresh per-test output file, FP64_50K_HOME to a fresh index directory, PAMIN_EVAL_HOME to the complete cached evaluation workspace, PAMIN_DEVICE=auto and PAMIN_VECTOR_INDEX=memory. MIRACL_DIR must hold the full Swahili corpus/topics/qrels; unset MIRACL_MAX_DOCS, HF_HOME, HF_ENDPOINT and PAMIN_RERANK_MAX_TOKENS. The real-corpus harnesses reject incomplete indices. The patch contains the exact test sources; the standalone 50k file retains the deterministic input generator.

## Statistical correction

The assignment unit is a complete process, not a query. Pair search-time totals within each of the three rotated rounds and enumerate all 2^3 sign patterns; keep the 24 queries within each process together. No query-independent significance claim is supported. Deterministic order/shared load also limits exchangeability, so these p-values are diagnostic, not a randomized confirmatory trial. Original p=0.0002 is withdrawn.

## Profiling and MIRACL identity checks

`profiling/*.json.gz` contains only public node names, provider names and microsecond durations, in recorded execution order. `profiling.py` groups by node/provider, discards the first eight warmups per group/bucket and recomputes steady CPU/all kernel fractions. `verify.py` checks these against the retained summary and generated table. Full traces, tensor contents and local paths are excluded. Kernel durations still include wrapper/prediction/waits, not isolated hardware or bus timing.

Before an FP64 MIRACL rerun, verify the exact complete cached input files:

```sh
python3 benchmarks/results/inference/local-closeout-2026-09-30/verify.py --miracl-dir "$MIRACL_DIR"
```

The manifest stores byte lengths/SHA256 for docs.jsonl, topics.tsv and qrels.tsv, and the measured order of all 482 query IDs and modes 0/1/2. These are post-run hashes of retained cache inputs, not claimed pre-run attestations. A future resolve/main change must fail the hash check; do not accept mere count equality.

`fp64/sources/measured-experiment.patch` is the exact historical test source. `experiment.patch` adds guarded rerun assertions: positive offered pairs and zero new reranker scores for modes 1/2. `fp64.py` also checks zero newly scored pairs and unchanged offered count in every retained comparison. Reference mode 0 is allowed to reuse a prior identical query; it is not used as an uncached inference timing arm.

## Fresh backend outputs

Fresh runs write raw files in a new RUN_ROOT. Summarize them directly, without copying over the retained archive or changing its manifest:

```sh
python3 benchmarks/results/inference/local-closeout-2026-09-30/summarize.py --run-root "$RUN_ROOT" > "$RUN_ROOT/summary.json"
```

This reads runner-produced JSON, log and resource files, enforces the same row/cache/provider checks and uses the same aggregation code as the archived path. The output is a new run's backend report, not a replacement for retained historical FP64/profiling evidence.

CoreML compile-cache warmth was neither recorded nor controlled in the historical runs. Whole-process wall/CPU/RSS values are observed magnitudes only; paired resource deltas are N/A/incomparable. The rerun template currently has the same limitation. Do not infer cold-start savings from these process totals.

The guarded MIRACL rerun patch opens only the accuracy-profile project. Missing or incomplete accuracy data fails the premise even if a complete speed index is available. The separately retained measured patch preserves the historical source; the verifier confirms every measured row used accuracy.

## Requested versus attested compute units

Historical backend runs did not emit or assert compute-unit policy. Their ALL/CPUAndGPU labels identify intended configurations from retained source/environment, not independently attested actual policy. Do not treat their timing ratio as a verified ALL-versus-GPU hardware advantage. Historical provider logs and frozen binary do not close that gap.

`sources/measured-experiment.patch` retains historical source. The guarded `sources/experiment.patch` now logs policy from the same units variable passed into the CoreML builder. Copy policy.py beside the rendered runner (or set PYTHONPATH to the archive directory). Runner and fresh-output summarizer require ALL for all and CPUAndGPU for gpu, rejecting both missing/stale markers and the wrong policy. The archived path retains historical uncertainty instead of inventing policy attestations.
