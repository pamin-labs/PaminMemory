# Strict cache-hit replay of streamed CoreML source hashing

This rechecks the [three-round historical measurement](coreml-stream-cache-2026-09-29.md)
through the same real `native::prepare` entry point and its frozen before/after
release binaries. Each arm requires **both** `source.onnx` and `model.onnx`
before timing, checks their expected sizes and the prepared graph SHA256, and
rejects changed inodes, birth times or modification times after the call. The
original 1,136,209,678-byte model source and prepared copy have the same
SHA256 at the start and end of each 30-round run. The stricter second run also
checks frozen binary SHA256 and at least 25% system free memory before every
arm. Disk free space must stay above 10 GiB. A failed premise exits without a
reported speed result.

Two separate alternating before/after, after/before runs were made on the
shared Apple M4. Each has 30 paired rounds; they are kept separate because
host load differed.

| Measure | First 30 rounds | Stricter second 30 rounds |
| --- | ---: | ---: |
| Before cache-hit process wall median | 0.570403 s | 0.818536 s |
| After cache-hit process wall median | 0.555832 s | 0.809023 s |
| Paired geometric after/before ratio | 0.98134 | 1.01160 |
| Rounds after was faster | 24/30 | 13/30 |
| Two-sided paired sign-flip p, 100,000 draws | 0.00005 | 0.42124 |
| Before median peak process RSS | 1,155,055,616 B | 1,155,088,384 B |
| After median peak process RSS | 18,874,368 B | 18,874,368 B |

The first run suggests a small time gain, while the stricter replay does not.
Even the arm medians and within-round ratios disagree in the second run. Thus
the old **−7.1% timing claim is rejected**; this shared-host experiment does
not establish a repeatable latency gain or loss. The roughly 1.136 GB lower
process peak is reproduced in both runs and follows the removal of the full
source allocation. It is per-process preparation memory, not steady model
residency or system-wide RSS. Both arms produced the same prepared graph
SHA256, `3ece88f7a06766959c38ad0cca861902ccd4530c15df0971f267abb1092c4109`.
No model inference, search accuracy, whole-search latency or total disk was
measured here.

[First 30 raw paired rows](coreml-stream-cache-recheck-2026-09-30-earlier-rows.json),
[stricter second 30 rows](coreml-stream-cache-recheck-2026-09-30-rows.json),
and the [exact stricter replay and summariser](../../harnesses/stream-cache-recheck-2026-09-30/README.md)
retain both the accepted memory result and the failed timing replication. The
frozen binary, source and model identities remain in the
[original protocol](coreml-stream-cache-2026-09-29.json).
