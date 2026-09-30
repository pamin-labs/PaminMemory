# Audit and rerun

Audit without models or writes:

```sh
python3 benchmarks/results/inference/local-closeout-2026-09-30/verify.py
python3 benchmarks/results/inference/local-closeout-2026-09-30/summarize.py
```

The verifier recomputes medians, nearest-rank p50/p95 of per-query three-round medians, process wall/CPU/RSS and four sign-flip tests from 432 rows and 18 resource records. It also checks provider/cache-miss attestations and file hashes. It writes no retained artifacts.

To rerun, use an isolated checkout at public commit `be329b00636987781689ec6ef0c4fc48611cdf1d` and apply `sources/experiment.patch`. This patch is an experimental appendix, never default code. Provision the complete pinned XQuAD-R accuracy index first; the harness asserts 13,014 indexed documents and completeness 1.0, and will not silently measure an empty index. Model cache identities, source exports, query IDs/text hashes/order, compiler, CPU configuration and requested device plans are in manifest.json. Cached revisions and OS were collected after execution; they are not historical pre/post attestations. Warmups are query positions 1..8, sample positions 0..1190 step 50. No model downloads or index rebuilds belong in measured search durations.

Build the integration executable:

```sh
cargo test --release -p pamin-engine --test crosslingual scratch_backend_repeat --no-run
```

Copy the printed executable to `$RUN_ROOT/benchmark`. Use a fresh output directory and retained `sources/run.py.in` (replace `${REPO}`, `${EVAL_HOME}`, `${RUN_ROOT}` with your checkout, populated evaluation cache and output directory). Update the native library path to your build output. Run the rendered Python program; it executes all 18 arms in the exact three rotations retained in manifest.json, one at a time, with 10 GiB disk and 25% memory headroom guards. It verifies a frozen executable, actual reranker device, 24 rows and per-query fresh scoring. Requested ALL is not proof every hardware unit executes the graph; Fast CoreML logged E5RT/ANE shape warnings.

Models use CPU INT8 vs Accurate CoreML FP16 / Fast CoreML FP32. These are different complete exports, not an isolated hardware arithmetic comparison. All results were collected on a shared workstation. Do not use the sample nDCG columns for full-corpus precision acceptance.

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
