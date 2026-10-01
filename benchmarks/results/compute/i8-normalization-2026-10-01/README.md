# Exact int8 pooled-vector normalization

This is a normalization kernel diagnostic, not model inference or whole search latency. Six process-local alternating blocks, 100,000 calls per arm/block, a shared Apple Silicon host, Rust 1.98.1, ordinary `-O`. The source is an immutable record of the scratch program, not a test/harness registered in the workspace.

The median block cost is 614.37896 ns before and 98.775 ns after (-83.92%, 6.22x). The ordered floating reduction is replaced by an exact integer sum, allowing LLVM SIMD without relaxing floating arithmetic. Old conversion/division already use SIMD. Output remains bit-identical; both paths allocate one 4,096-byte vector payload. No model/index encoding change and no query accuracy, delay, RSS or disk improvement is inferred from these kernel timings.

The complete squared sum is at most 1024 * 128^2 = 2^24, representable exactly in both i32 and f32. Product regressions include signed extrema, mixed signed values and the zero-vector error.

Reproduce the archived scratch diagnostic:

```sh
cp frozen-program.rs.txt /tmp/normalization.rs
rustc --edition=2024 -O /tmp/normalization.rs -o /tmp/normalization
/tmp/normalization
```

The summary retains every block and its source SHA-256. On this host, generated code uses `addv.4s` in the new reduction and `fdiv.4s` in both output loops.
