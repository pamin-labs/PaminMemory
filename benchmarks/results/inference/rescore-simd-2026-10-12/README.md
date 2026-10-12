# Host rescore kernels: SIMD dot product and fp16 conversion

**Status: provisional.** All three passes below ran beside unrelated Cargo
builds on the same host. A clean rerun is pending: no competing jobs, p50 and
p95 pooled over every pass, and forced AVX2 and AVX-512 arms rotated against
each other. Until it lands, treat these figures as a screen, not a validated
result.

A component screen of the host-side part of `recall_dense`
(`crates/pamin-index/src/projection.rs`). For each candidate, that code widens
the stored fp16 vector and takes two dot products (its norm, and its dot with
the query). The screen does **not** run `Engine::search_reranked`, query zvec,
iterate documents or sort, so it says nothing about whole-search latency (see
the last section).

## Arms

Each arm is one commit's implementation of the same work: 1 query norm, then
100 candidates × 1024 dimensions.

| Arm | Commit | fp16 → f32 | Dot product |
| --- | --- | --- | --- |
| `main` | `ad33d2d` | scalar `decode`, one code at a time | sequential f32 sum |
| `dot` | `255fbe1` | scalar `decode` | runtime-dispatched kernel (`crates/pamin-index/src/vector_math.rs`); AVX-512F on this host |
| `convert` | `01614fb` | `half` 2.7.1 slice conversion (F16C on this host) | as `dot` |

## Results

The `main` and `dot` columns compare `dot` with its predecessor `main`. The
`dot` and `convert` columns compare `convert` with its predecessor `dot`. The
`main` and `convert` columns compare the stack with `main`.

- Host: Intel Xeon @ 2.10 GHz (AVX-512F/BW/VL/VNNI/BF16/FP16, F16C), 4 vCPUs, shared.
- Build: Rust 1.98.1, `--release`, one thread.
- Load: during all three passes the 1-minute load average was about 5.6, from
  another agent's Cargo builds on this host. That is the main source of the p95
  spread.
- Method: 16 rounds, with the arm order rotated and reversed between rounds,
  100 calls per arm per round. The figures below are the median of each pass's
  p50, over three passes ([raw output](result.txt)).

| Metric | `main` | `dot` | Δ vs `main` | `convert` | Δ vs `dot` | Stack Δ vs `main` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max abs cosine error vs f64 oracle | 7.75e-8 | 1.54e-8 | −6.21e-8 (−80%) | 1.54e-8 | 0 (0%) | −6.21e-8 (−80%) |
| Top-10 overlap with oracle (of 10) | 10 | 10 | 0 | 10 | 0 | 0 |
| p50 µs per 100×1024 rescore | 241.72 | 131.32 | −110.40 (−45.7%) | 36.67 | −94.65 (−72.1%) | −205.05 (−84.8%) |
| p95 µs | N/A | N/A | N/A | N/A | N/A | N/A |
| Memory | N/A | N/A | N/A | N/A | N/A | N/A |
| Disk | N/A | N/A | N/A | N/A | N/A | N/A |

- **Memory:** not measured. Neither change adds a buffer or keeps one alive:
  both arms allocate one 1024-wide `Vec<f32>` per candidate, as `main` does.
- **Disk:** not measured as a size. The release binary gains the `half` and
  `crunchy` crates' code.
- **p95:** not reported. The per-pass p95 ranged from 247 to 631 µs for `main`
  under the competing builds. It waits for the clean rerun, pooled over all
  passes.
- **Exactness:** a unit test checks the fp16 conversion bit-for-bit against
  `decode` for all 65536 codes. A second unit test checks every dot kernel the
  CPU can run against an f64 sum.

## Other platforms

| Platform | Dot product | fp16 → f32 | Status |
| --- | --- | --- | --- |
| x86-64 with AVX-512F (Intel Skylake-SP and later, AMD Zen 4/5) | AVX-512F | F16C | Timed above |
| x86-64 with AVX2+FMA only (most other Intel and AMD) | AVX2 | F16C | Tested for correctness here. Not timed: dispatch picks AVX-512 on this host. A provisional unrotated screen of the same kernel ran 10.99 µs (AVX2) vs 10.45 µs (AVX-512) per 100 dot-only rows |
| Linux aarch64 (Graviton, Ampere, Raspberry Pi 5) | NEON | `fcvtl` when `fp16` is detected | Unit tests pass under `qemu-aarch64-static`. Timing N/A: emulation is not timing |
| macOS on Apple silicon | NEON, replacing Accelerate `cblas_sdot` ([2026-09-30 screen](../vector-rescore-accelerate-2026-09-30.md)) | `fcvtl` | Timed on an M4: see [Apple M4](#apple-m4) |
| macOS on Intel | as x86-64 (AVX-512/AVX2), replacing Accelerate, by maintainer decision so every x86-64 host runs one implementation | F16C | Compile-checked. Timing N/A: no Intel Mac available. The x86 kernels are timed above on Linux; they have not been timed against Accelerate, so no speed change is claimed for Intel Macs |
| Windows x86-64 | as x86-64 | as x86-64 | Compile-checked |

## Apple M4

A separate machine ran the same work (1 query norm, then 100 candidates ×
1024 dimensions) with macOS arms. The PR's first two commits kept
Accelerate's `cblas_sdot` on macOS; this screen is why the last code commit
moved macOS to the NEON kernel Linux aarch64 already used.

| Arm | fp16 → f32 | Dot product | Implements |
| --- | --- | --- | --- |
| `mac-main` | scalar `decode` | Accelerate `cblas_sdot` | `main` on macOS |
| `mac-pr` | `half` slice conversion (`fcvtl`) | Accelerate `cblas_sdot` | this PR before the NEON switch |
| `half-neon` | `half` slice conversion (`fcvtl`) | NEON, four accumulators | this PR |
| `dot-accelerate`, `dot-neon` | (pre-widened) | as named | the dot product alone, 100 rows |

- Host: Apple M4, 32 GB, on AC power, no thermal warnings before or after.
- Build: `--release`, one thread.
- Load: 1-minute load average 2.55–2.59. Chrome (13–22% CPU), OrbStack,
  Adobe Desktop Service and an editor language server were running.
- Method: three runs back to back. Each run rotates the arm order between
  rounds and pools 6000 calls per arm. The figures are the median over the
  three runs of each run's pooled p50 and p95
  ([per-run figures](m4-result.md)).

| Metric | `mac-main` | `mac-pr` | `half-neon` | Δ vs `mac-pr` | Stack Δ vs `mac-main` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Max abs cosine error vs f64 oracle | ≤ 7.749e-8 | ≤ 7.749e-8 | ≤ 7.749e-8 | N/A (only the bound across all arms was reported) | N/A |
| Top-10 overlap with oracle (of 10) | 10 | 10 | 10 | 0 | 0 |
| p50 µs per 100×1024 rescore | 40.38 | 22.42 | 21.46 | −0.96 (−4.3%) | −18.92 (−46.9%) |
| p95 µs | 47.08 | 24.17 | 22.62 | −1.55 (−6.4%) | −24.46 (−52.0%) |
| Memory | N/A | N/A | N/A | N/A | N/A |
| Disk | N/A | N/A | N/A | N/A | N/A |

The dot product alone: `dot-neon` 5.25 µs p50 and 5.92 µs p95, against
`dot-accelerate` 6.00 and 6.75 per 100 rows. That is −0.75 µs (−12.5%) at p50
and −0.83 µs (−12.3%) at p95. NEON was faster at p50 in all three runs. The
screen is still provisional: the three runs were consecutive and ran beside
the background load listed above.

## Libraries considered

- **`simsimd`:** in the same unrotated dot-only screen, the hand kernel ran
  10.45 µs and `simsimd` 6.x ran 13.83 µs per 100 rows. The hand kernel adds no
  C build step.
- **`half`:** used for the fp16 conversion rather than hand-written
  intrinsics. On aarch64 it needs inline assembly, because stable Rust has no
  `f16` vector type.

## Whole-search impact: not claimed

On the 2026-10-12 XQuAD-R run, one query took 1960–2456 ms at the entry point.
Rescoring is about 0.24 ms of that per call at this candidate count, so this
change is worth well under 0.1% of a query. That is below what an entry-point
run can resolve, so no entry-point arm was run, and no search-latency gain is
claimed. The change exists so that no part of the CPU path stays scalar.

## Reproduce

The harness only produces numbers, so it is kept out of this tree.

- Location: `pamin-memory/research/2026-10-12/raw/rescore-simd/` in the
  `internal-docs` submodule. Its README explains how to run it.
- Arms: the rerun forces each x86 kernel, so AVX2 and AVX-512 are timed
  against each other.
- Premise: it aborts before timing if any arm misses the f64 oracle by more
  than 1e-6 in cosine or changes the top 10.
- Raw output: [`result.txt`](result.txt) is this provisional run's output.
