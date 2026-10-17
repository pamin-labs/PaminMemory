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
- Clock: every µs figure in this README, on both hosts, is **elapsed (wall)
  time** for one call. The harness reads `std::time::Instant` (a monotonic
  clock) around one rescore. The work runs on one thread, with no child process
  and no GPU or NPU. Neither harness captured CPU user or system time, so CPU
  time is N/A in every table. On one thread, CPU time could not exceed the wall
  figure, but nobody measured how far below it the CPU time sits.

| Metric | `main` | `dot` | Δ vs `main` | `convert` | Δ vs `dot` | Stack Δ vs `main` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max abs cosine error vs f64 oracle | 7.75e-8 | 1.54e-8 | −6.21e-8 (−80%) | 1.54e-8 | 0 (0%) | −6.21e-8 (−80%) |
| Top-10 overlap with oracle (of 10) | 10 | 10 | 0 | 10 | 0 | 0 |
| p50 µs per 100×1024 rescore, elapsed | 241.72 | 131.32 | −110.40 (−45.7%) | 36.67 | −94.65 (−72.1%) | −205.05 (−84.8%) |
| p95 µs, elapsed | N/A | N/A | N/A | N/A | N/A | N/A |
| CPU user / system time | N/A | N/A | N/A | N/A | N/A | N/A |
| Memory | N/A | N/A | N/A | N/A | N/A | N/A |
| Disk | N/A | N/A | N/A | N/A | N/A | N/A |

- **Memory:** not measured. Neither change adds a buffer or keeps one alive:
  both arms allocate one 1024-wide `Vec<f32>` per candidate, as `main` does.
- **Disk:** not measured as a size. The release binary gains the `half`
  crate's code. `half` lists `crunchy` only for targets this project does not
  build: `cargo tree -e normal -i crunchy` prints nothing for
  `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin` or
  `aarch64-unknown-linux-gnu`.
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
| macOS on Apple silicon | Accelerate `cblas_sdot`, as on `main` ([2026-09-30 screen](../vector-rescore-accelerate-2026-09-30.md)). The NEON kernel compiles and its tests run here, but `dot` does not call it | `fcvtl` | fp16 step timed on an M4: see [Apple M4](#apple-m4). NEON vs Accelerate: N/A, not measured cleanly (the M4 screen is provisional and covers no M1–M3) |
| macOS on Intel | as x86-64 (AVX-512/AVX2), replacing Accelerate, by maintainer decision so every x86-64 host runs one implementation | F16C | Compile-checked. Timing N/A: no Intel Mac available. The x86 kernels are timed above on Linux; they have not been timed against Accelerate, so no speed change is claimed for Intel Macs |
| Windows x86-64 | as x86-64 | as x86-64 | Compile-checked |

## Apple M4

A separate machine ran the same work (1 query norm, then 100 candidates ×
1024 dimensions) with macOS arms. The PR's first two commits kept
Accelerate's `cblas_sdot` on macOS. This screen briefly moved Apple silicon to
the NEON kernel Linux aarch64 uses; that switch is reverted, and Apple silicon
keeps Accelerate. The screen ran under background load on one M4, so it does
not justify replacing a library kernel on every Apple CPU. The `half-neon` and
`dot-neon` columns below are provisional evidence for a later clean
measurement, not the shipped path.

As on x86, the arms are not builds of the named commits. The harness
(`rescore-simd-m4/` at `internal-docs` `12d4918`) holds a copy of each
commit's kernels. Its `decode` is identical to `ad33d2d`'s, and its NEON kernel
is identical to `df94e8a`'s; both were checked with `diff` on the function
bodies. `half` is 2.7.1 from the harness's lockfile.

| Arm | Commit whose macOS path it copies | fp16 → f32 | Dot product |
| --- | --- | --- | --- |
| `mac-main` | `ad33d2d` (`main`) | scalar `decode` | Accelerate `cblas_sdot` |
| `mac-pr` | `01614fb`; also what this PR ships on Apple silicon since `a148a5e` | `half` slice conversion (`fcvtl`) | Accelerate `cblas_sdot` |
| `half-neon` | `df94e8a` (reverted by `a148a5e`; a candidate, not shipped) | `half` slice conversion (`fcvtl`) | NEON, four accumulators |
| `dot-accelerate`, `dot-neon` | as `mac-pr` and `half-neon` | (pre-widened) | the dot product alone, 100 rows |

- Host: MacBook Air Mac16,12, Apple M4 (4 P + 6 E cores), 32 GB, macOS 27.0.1
  (26A434). On AC power, with no thermal warnings before or after.
- Build: Homebrew `rustc`/`cargo` 1.99.0 (rustup was not installed), against
  Rust 1.98.1 on x86-64. `--release`, one thread. The compiler differs between
  the two hosts, so their figures are not comparable with each other.
- Clock: elapsed (wall) time per call; CPU time N/A (see the x86 section).
- Load: 1-minute load average 2.55–2.59. Chrome (13–22% CPU), OrbStack,
  Adobe Desktop Service and an editor language server were running.
- Method: three runs back to back. Each run rotates the arm order between
  rounds and pools 6000 calls per arm. The figures are the median over the
  three runs of each run's pooled p50 and p95
  ([per-run figures](m4-result.md), [raw output](m4-raw/)).

| Metric | `mac-main` | `mac-pr` | Δ `mac-pr` vs `mac-main` | `half-neon` | Δ vs `mac-pr` | Stack Δ vs `mac-main` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Max abs cosine error vs f64 oracle | 1.176e-8 | 1.176e-8 | 0 (0%) | 1.428e-8 | +2.52e-9 (+21.4%) | +2.52e-9 (+21.4%) |
| Top-10 overlap with oracle (of 10) | 10 | 10 | 0 | 10 | 0 | 0 |
| p50 µs per 100×1024 rescore, elapsed | 40.38 | 22.42 | −17.96 (−44.5%) | 21.46 | −0.96 (−4.3%) | −18.92 (−46.9%) |
| p95 µs, elapsed | 47.08 | 24.17 | −22.91 (−48.7%) | 22.62 | −1.55 (−6.4%) | −24.46 (−52.0%) |
| CPU user / system time | N/A | N/A | N/A | N/A | N/A | N/A |
| Memory | N/A | N/A | N/A | N/A | N/A | N/A |
| Disk | N/A | N/A | N/A | N/A | N/A | N/A |

The `mac-main` → `mac-pr` step is the fp16 conversion alone: both arms use
Accelerate's `cblas_sdot`, so the step is `half`'s slice conversion replacing
the scalar `decode`. Its cosine error is unchanged because the conversion is
exact (it equals `decode` for all 65536 codes in every run) and the dot product
is the same. Memory and disk are N/A for every column: the screen measured
neither, and the M4 run retained no binary sizes.

NEON's cosine error is 2.5e-9 larger than Accelerate's. Both errors are about 70× below the 1e-6 premise, and neither changes the top 10.

The dot product alone: `dot-neon` 5.25 µs p50 and 5.92 µs p95, against
`dot-accelerate` 6.00 and 6.75 per 100 rows. That is −0.75 µs (−12.5%) at p50
and −0.83 µs (−12.3%) at p95. NEON was faster at p50 in all three runs. The
screen is still provisional: the three runs were consecutive and ran beside
the background load listed above. So NEON vs Accelerate is N/A as a validated
result, and Apple silicon keeps Accelerate until a clean, repeated, rotated
measurement says otherwise.

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
