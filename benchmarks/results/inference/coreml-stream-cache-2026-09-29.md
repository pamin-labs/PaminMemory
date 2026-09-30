# CoreML preparation: stream the source digest on a cache hit

**Timing correction (2026-09-30):** the three-round `−7.1%` figure below is
rejected. [Two strict 30-round cache-hit replays](coreml-stream-cache-recheck-2026-09-30.md)
kept the source and prepared files unchanged, reproduced the process-memory
reduction, and disagreed on the small latency effect. There is no accepted
preparation-speed percentage from this shared host.

This measures the real `native::prepare` function on an existing prepared BGE
export, not inference or complete search. The source is 1,136,209,678 bytes.
Both arms have the same ARM SHA-256 implementation and release dependencies.
The control references the unchanged `native.rs` and `prepared.rs` at `90b0225`;
other modules are identical. Separate processes run before/after, after/before,
before/after. [Evidence and reproduction source](coreml-stream-cache-2026-09-29.json)
include binary hashes, module hashes, raw-output hashes and all six measurements.

| Metric | Before | After | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Prepared graph SHA-256 | `3ece88f7a06766959c38ad0cca861902ccd4530c15df0971f267abb1092c4109` | Identical in all six runs | 0 changed graphs | N/A |
| Historical three-round cache-hit median | 1.165626 s | 1.083086 s | -0.082540 s | Rejected timing estimate; see correction above |
| Maximum process RSS, median | 1,155,006,464 B | 18,825,216 B | -1,136,181,248 B | -98.4% |
| Prepared graph bytes | 924,562 B | 924,562 B | 0 B | 0% |
| Prepared source-copy bytes | 1,136,209,678 B | 1,136,209,678 B | 0 B | 0% |
| Retrieval accuracy / search latency | Not measured | Not measured | N/A | N/A |

| Round | Before seconds | After seconds | Before RSS bytes | After RSS bytes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.947484 | 1.107999 | 1,153,597,440 | 18,825,216 |
| 2 | 1.147208 | 1.078013 | 1,155,006,464 | 18,972,672 |
| 3 | 1.165626 | 1.083086 | 1,155,039,232 | 18,792,448 |

## Mechanism and limits

Previously the full source was retained in a `Vec` before checking the cache.
The new path reuses the existing streaming SHA-256 helper. It still verifies
all input bytes and preserves the cache key and graph. A miss copies the source
into a private snapshot, verifies the snapshot digest, and rewrites those same
bytes. A changed input is rejected; failed preparation removes its partial copy.

The full cached BGE regression verifies first preparation and a subsequent hit
against the graph produced by the unchanged rewrite function: 49 LayerNorms,
24 GELUs and the FP32 head. Small regressions cover a changed source and failure
cleanup without requiring models.

The original timing is not an accepted direction: three rounds on a shared
machine with other builds active did not replicate in the strict replay. The approximately
1.136 GB reduction matches the removed source allocation. These are independent
process peaks, not the model's steady-state residency or system-wide memory.
Initial preparation still reads a complete snapshot; this change does not share
compiled resources across the 64/128/256 bucket sessions. Model/index disk size
and inference arithmetic are unchanged; retrieval quality is not re-evaluated
by this cache-only experiment.
