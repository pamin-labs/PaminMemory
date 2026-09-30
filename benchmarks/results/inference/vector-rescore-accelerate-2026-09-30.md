# Host vector rescore kernel screen on Apple M4

This is a release-built, six-round alternating comparison of four host-side
cosine implementations over 100 synthetic 1024-wide FP32 vectors per call.
The short candidate list matches the order of magnitude of zvec's product
rescore; it does **not** run `Engine::search_reranked`, decode vectors from
zvec, or establish a product search gain. Each timed call includes the host
function boundary and result allocation. The shared Apple Mac16,12 had
variable concurrent load. The source executes 400 calls per arm per round.

| Route | Median per 100 rows | Max score error vs FP64 oracle |
| --- | ---: | ---: |
| Existing Rust FP32 reduction | 115.000 µs | 1.274e-6 |
| SimSIMD, three dot products | 114.807 µs | Within 1.19e-7 of existing reduction |
| NumKong angular | 263.641 µs | 2.9e-8 |
| Apple Accelerate `cblas_sdot` | **13.150 µs** | **2.68e-7** |

On a second deliberately clustered set with FP16-rounded candidates, the
existing reduction matched 5/10 FP64-oracle top-ten IDs (max error 2.348e-6);
Accelerate matched 6/10 (max error 2.80e-7). That one constructed near-tie
set is a numerical stress check, not a retrieval-accuracy estimate. The
kernel result supports trying Accelerate on macOS, while the complete product
quality/latency impact of changing only this primitive remains unmeasured.
The product code uses the system Accelerate framework on macOS and preserves
the old Rust reduction elsewhere. No model, index encoding, or public
configuration changes here.

## Why the small rescore stays on optimized CPU

A separate rotated 100×1024 FP16-rounded vector control measured complete
host calls through ONNX Runtime, including tensor setup. On the same Apple M4,
ORT's optimized CPU FP32 p50 was **38.167 µs** and CoreML `ALL` FP32 was
**113.375 µs**; CoreML CPU+GPU was 115.375 µs and CoreML CPU-only 111.125 µs.
CoreML EP owned 11 nodes, CPU EP one, but this does not establish physical
ANE/GPU utilization. This is a small-kernel control, not `search_reranked`.
The separate Accelerate screen above then found a faster CPU SIMD path.
These two timing studies ran under different host load and are not pooled.

[Raw scores/times](vector-rescore-device-2026-09-30/result.json), source,
both ONNX graphs and lockfile are archived in the
[device-control directory](vector-rescore-device-2026-09-30/).
`probe.rs` takes that directory as its argument after building its Cargo
manifest; it validates finite scores and records assigned EP nodes. The
output's CoreML assignment is still not a hardware utilization trace.

[Exact Rust driver](vector-rescore-accelerate-2026-09-30.rs),
[manifest](vector-rescore-accelerate-2026-09-30-Cargo.toml),
[lockfile](vector-rescore-accelerate-2026-09-30-Cargo.lock), and
[all round timings and numeric checks](vector-rescore-accelerate-2026-09-30-result.txt)
retain the screen. To repeat it on macOS, copy those four files into a
disposable Cargo package as `src/main.rs`, `Cargo.toml`, `Cargo.lock`, then run
`cargo run --release --locked`. Other platforms need their own SIMD kernel
comparison before selecting an implementation.
