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
