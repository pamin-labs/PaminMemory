# Host rescore kernels: SIMD dot product and fp16 conversion

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
| `dot` | `255fbe1` | scalar `decode` | runtime-dispatched kernel ([`vector_math.rs`](vector_math.rs)); AVX-512F on this host |
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
| p95 µs (pass 2, the quietest) | 247.47 | 146.42 | −101.05 (−40.8%) | 42.63 | −103.79 (−70.9%) | −204.84 (−82.8%) |
| Memory | N/A | N/A | N/A | N/A | N/A | N/A |
| Disk | N/A | N/A | N/A | N/A | N/A | N/A |

- **Memory:** not measured. Neither change adds a buffer or keeps one alive:
  both arms allocate one 1024-wide `Vec<f32>` per candidate, as `main` does.
- **Disk:** not measured as a size. The release binary gains the `half` and
  `crunchy` crates' code.
- **Exactness:** a unit test checks the fp16 conversion bit-for-bit against
  `decode` for all 65536 codes. A second unit test checks every dot kernel the
  CPU can run against an f64 sum.

## Other platforms

| Platform | Dot product | fp16 → f32 | Status |
| --- | --- | --- | --- |
| x86-64 with AVX-512F (Intel Skylake-SP and later, AMD Zen 4/5) | AVX-512F | F16C | Timed above |
| x86-64 with AVX2+FMA only (most other Intel and AMD) | AVX2 | F16C | Tested for correctness here. Not timed: dispatch picks AVX-512 on this host. A provisional unrotated screen of the same kernel ran 10.99 µs (AVX2) vs 10.45 µs (AVX-512) per 100 dot-only rows |
| Linux aarch64 (Graviton, Ampere, Raspberry Pi 5) | NEON | `fcvtl` when `fp16` is detected | Unit tests pass under `qemu-aarch64-static`. Timing N/A: emulation is not timing |
| macOS (Apple M-series, and Intel Macs) | Accelerate `cblas_sdot`, unchanged ([2026-09-30 screen](../vector-rescore-accelerate-2026-09-30.md)) | as Linux aarch64 (Apple silicon) or x86-64 (Intel) | Compile-checked for both macOS targets. Timing N/A: no Mac here |
| Windows x86-64 | as x86-64 | as x86-64 | Compile-checked |

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

1. Make an empty Cargo package.
2. Copy [`bench-Cargo.toml`](bench-Cargo.toml) to its `Cargo.toml` and
   [`bench-Cargo.lock`](bench-Cargo.lock) to its `Cargo.lock`.
3. Copy [`main.rs`](main.rs), [`vector_math.rs`](vector_math.rs) and
   [`decode.rs`](decode.rs) into its `src/`.
4. Run `cargo run --release --locked` three times.
