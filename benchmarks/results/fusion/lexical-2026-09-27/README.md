# Historical lexical sweep output

These are retained complete stdout tables, grid rows and fold choices used by
the 2026-09-27 ADR. Original machine-specific build paths are redacted; both
original and published hashes are in manifest.json. The historical output did
not retain per-query matrices or full model/index provenance, so it is not a
fully replayable paired artifact and cannot retrospectively supply those facts.

New CHANNELS runs can set `CHANNELS_OUT=<result.json>` to persist all raw paired
nDCG vectors and grid configurations through the shared Diagnosis owner. Keep
stdout beside it for the selected folds and printed family tests. Use the
pinned dataset/model/index setup of the relevant corpus; this diagnostic drives
fusion before reranking and does not measure full-product reranker latency.
