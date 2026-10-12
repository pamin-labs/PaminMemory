# PaminMemory M4 measurements — report (runbook revision 2)

Status: **step 0 done, step 1 done, step 2 aborted** (ORT 1.28 stopped at pass 2 round 4 of 3×6; ORT 1.31 never ran).
All files are in `~/pamin-m4`.

## Step 0 — machine and conditions (`step0.txt`)

- MacBook Air Mac16,12, Apple M4, 10 cores (4 P + 6 E), 32 GB; macOS 27.0.1 (26A434)
- FEAT_DotProd 1, FEAT_BF16 1, FEAT_I8MM 1, FEAT_SME 1, FEAT_SME2 1, FEAT_SME2p1 0, FEAT_SME_F16F16 0
- Low Power Mode 0. Spotlight indexing disabled. Free disk 74 GiB.
- 20:24 (step 0): **on battery** (95%), load average 2.43 2.29 2.81, no thermal warnings.
- From 20:25 onward: AC power.

## Step 1 — rescore kernel screen (`rescore-run{1,2,3}.txt`, `step1-conditions.txt`)

- All premises passed in all three runs:
  - `half` conversion == decode for all 65536 codes
  - max |cos − oracle|: mac-main 1.176e-8, mac-pr 1.176e-8, half-neon 1.428e-8, half-seq 7.749e-8
  - top-10 10/10 for every arm
- Runs: 20:28:19–20:28:24, about 2 s each.
- Conditions: load average 2.59 → 2.55; AC power; no thermal warnings before or after any run.

Pooled p50 / p95 in µs; n = 6000 per arm per run.

| arm | run 1 | run 2 | run 3 |
|---|---|---|---|
| mac-main | 43.33 / 47.08 | 39.71 / 50.38 | 40.38 / 46.33 |
| mac-pr | 22.42 / 24.17 | 21.75 / 25.58 | 22.50 / 23.92 |
| half-neon | 21.50 / 22.62 | 19.38 / 23.92 | 21.46 / 22.38 |
| half-seq | 103.46 / 119.00 | 102.67 / 120.00 | 103.42 / 117.12 |
| dot-accelerate | 6.08 / 6.75 | 5.92 / 7.42 | 6.00 / 6.75 |
| dot-neon | 5.29 / 5.88 | 5.21 / 6.75 | 5.25 / 5.92 |
| dot-seq | 42.46 / 48.04 | 42.46 / 52.96 | 42.46 / 47.79 |

Pooled p50, against mac-main:
- mac-pr, half-neon: faster in all three runs
- half-seq: slower in all three runs

Dot-only arms, against dot-accelerate:
- dot-neon: faster in all three runs
- dot-seq: slower in all three runs

## Step 2 — bge-reranker-v2-m3, ORT 1.28.0 only (aborted)

Files:
- `ort128-aborted-swap.txt` (stdout and stderr)
- `out-ort128.jsonl`: 487 samples, written when the run was interrupted
- `step2-conditions-aborted-swap.txt`
- `step2-stack-2219.txt`, `step2-stack-2229.txt` (`sample` output)

No `out-ort128-summary.json`. No ORT 1.31 files.

**Why it was aborted:**
- The script keeps all 10 sessions loaded in one process.
- Process physical footprint: **52.9 GB** (peak 53.0 GB) on a 32 GB machine.
- System swap: 45.5 of 46 GB used, 83,145,784 swapouts.
- The timing loop started at 20:33:16 and was still running at 22:31, about 2 h in.
- Every timing sample includes paging time, so the latency numbers below are **invalid for comparison**. They are reported only as raw evidence.
- Stopped with SIGINT at 22:31. Python flushed its output, then hung during shutdown. Killed with SIGTERM at 22:41.

### Accuracy vs fp32-cpu-t1 (ORT 1.28.0)

Accuracy does not depend on timing, so these numbers are not affected by the paging.

| arm | Kendall τ-b | top-1 | max \|Δlogit\| | load s (in-process) | timed? |
|---|---|---|---|---|---|
| fp32-cpu-t1 | 1.0000 | 1.00 | 0.0000 | 2.0 | yes |
| int8-shipped-t1 | 0.8036 | 1.00 | 0.7620 | 0.5 | DROPPED (τ < 0.85) |
| int8-shipped-t4 | 0.8036 | 1.00 | 0.7620 | 0.3 | DROPPED |
| dyn-qint8-t1 | 0.8571 | 1.00 | 1.1506 | 0.7 | yes |
| dyn-qint8-t4 | 0.8571 | 1.00 | 1.1506 | 0.3 | yes |
| dyn-quint8-t1 | 0.8036 | 1.00 | 0.9668 | 0.2 | DROPPED |
| dyn-quint8-t4 | 0.8036 | 1.00 | 0.9668 | 0.2 | DROPPED |
| fp16-coreml-gpu | 0.9488 | 1.00 | 0.0273 | 4.7 | yes |
| fp32-coreml-gpu | 1.0000 | 1.00 | 0.0001 | 5.5 | yes |
| fp16-coreml-all | 0.9488 | 1.00 | 0.0273 | 7.3 | yes |

- CoreML partitioning:
  - fp32 graph: 51 partitions, 1312 of 1420 nodes
  - fp16 graph: 51 partitions, 1309 of 1417 nodes
- Peak RSS per arm: **N/A**. The `--rss` child step runs after timing, and timing never finished. The run had not reached the summary step either.

### Latency per 8×256 batch, partial samples (SWAP-CONTAMINATED, not comparable)

Values are p50 / p95 in ms; n is the number of samples. There are no rows for the 4 arms dropped by the accuracy check.

| arm | pass 0 | pass 1 | pass 2 (partial) | all |
|---|---|---|---|---|
| fp32-cpu-t1 | 1887.4 / 11518.0 (30) | 1854.2 / 11759.7 (30) | 2045.9 / 13089.4 (20) | 1879.0 / 11838.2 (80) |
| dyn-qint8-t1 | 1770.8 / 4593.8 (30) | 1755.9 / 4956.6 (30) | 1780.3 / 5865.4 (20) | 1763.8 / 4956.6 (80) |
| dyn-qint8-t4 | 1118.5 / 2693.6 (30) | 1088.2 / 2835.5 (30) | 1089.0 / 2976.6 (20) | 1107.0 / 2835.5 (80) |
| fp16-coreml-gpu | 6415.6 / 131309.3 (30) | 6565.2 / 137786.2 (30) | 5883.9 / 152813.4 (20) | 6215.4 / 139856.8 (80) |
| fp32-coreml-gpu | 6349.4 / 117709.9 (30) | 5019.2 / 127141.7 (30) | 5528.8 / 123319.6 (25) | 5694.8 / 121344.6 (85) |
| fp16-coreml-all | 6930.8 / 128783.7 (30) | 5795.9 / 142883.6 (30) | 6874.4 / 136305.3 (22) | 6384.9 / 136305.3 (82) |

Stack samples (`sample`, 3 s each):
- 22:19: the main thread was in `MatMul<float>`, running `ArmKleidiAI::MlasGemmBatch` → `kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa`. So ORT 1.28.0 already dispatches fp32 MatMul to the KleidiAI SME2 kernels on this M4.
- 22:29: the main thread was inside a CoreML partition (`MLNeuralNetworkEngine` → `Espresso::net::__forward`). Its hot frames were `Espresso::cast_kernel_cpu` (fp16→fp32 `vConvert_Planar16FtoPlanarF`), `elementwise_kernel_cpu` and `BNNSEngine::inner_product_kernel`, which are Espresso CPU kernels.

## Deviations and fixes

1. **Machine not fully quiet.**
   - Chrome (13–22% CPU), OrbStack, Adobe Desktop Service, the VS Code C# language server and Claude Code stayed running in the background.
   - Load average was about 2.3–2.6 throughout.
   - Step 0 was recorded on battery; steps 1 and 2 ran on AC.
2. **File creation:** the files were extracted from PROMPT.md with a regex script, not typed by hand.
3. **Toolchains:**
   - Rust: the existing Homebrew rustc/cargo 1.99.0 was used; rustup was not installed.
   - Python: the venvs use Homebrew Python 3.14.8.
   - ORT resolved to 1.28.0 and 1.31.0.
4. **Fix: symlinked model data.**
   - The first step 2 attempt failed in both versions inside `quant_pre_process`, before any timing ran. The error was `onnx ... ValidationError: Data of TensorProto ... model.onnx_data ... is a symbolic link`. The Hugging Face cache stores blobs behind symlinks.
   - Fix: `cp -cRL <snapshot> ~/pamin-m4/model`, an APFS clone with the links resolved, and `MODEL_DIR=~/pamin-m4/model`. The script was not changed.
   - Failed logs: `ort1{28,31}-failed-symlink.txt`, `step2-conditions-failed-symlink.txt`.
5. **Step 2 aborted because the process ran out of memory and swapped** (details above). Not fixed: the smallest fix is no longer small. Loading fewer sessions at a time would change the interleaved rotation design.
6. **Runbook issues observed** (not fixed):
   - Python 3.14 uses a 128 KiB file buffer, so progress is invisible until exit. `-u` and `flush=True` would expose it.
   - The `sme` grep in `meta` also matches `kern.boot*` lines, because the boot UUID contains "BF16".
   - Step 1 took about 2 s per run, not about 5 min. Step 2 timing took more than 2 h, against an estimate of 40–60 min.
