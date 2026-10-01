# Last-identical-list memo: native CPU acceptance

A bounded memo of the last complete successful input list avoids retokenizing a
reranker shortlist whose ordered raw pairs and effective batch limits are
unchanged, while every referenced complete batch remains cached. It retains
batch references and result positions, without texts, encodings or duplicate
scores. Changed inputs or evicted references use the existing full planner.
The sole score cache remains bounded to 4,096 logical score slots.

These measurements compare the complete-batch cache with that memo, using
actual `Engine::search_reranked` at return limits **5 and 10**. They are not a
comparison to the earlier context-invalid pair cache. Full-result limit300
calls follow each timed call **outside its clock**, with separate counters.

Run the inert verifier with Python 3:

```sh
python3 verify.py
python3 test_verify.py
```

The verifier reads sanitized archives, checks file hashes, derives every row,
checks source-declared f32 bits and gold ranks, recomputes costs and rejects
invalid evidence. It never starts a build, database or model. Product
reranking/fusion source snapshots and fixture JSON remain public.
The archive's arithmetic verifier is complemented by the separate
[reproduction package](../../../harnesses/reranker-identical-list-2026-09-30/README.md).
It retains the exact original native probe and selected frozen compilation
inputs, plus commands using an explicitly supplied, hash-bound public shared
controller dependency. A bare checkout of this PR does not automatically contain
that dependency. It can generate a new 36-process/428-call experiment;
PR228's separate 80-process/880-call experiment is not substituted for this one.
Templates become measurement controllers only in a fresh scratch export, under
the repository measurement policy. Original private logs and transformation hashes remain
private; no credential record or complete private seed/clone manifest is
included. Every public stdout was sanitized **before** gzip compression.

The directory date is the **September 30 series label**; public export was
**October 1, 2026 UTC**. Retained UTC log events span 2026-09-30T23:35:20.157972Z
to 2026-09-30T23:42:07.377380Z. Per-process first/last recorded events are in
[receipt.json](receipt.json); exact process/capture start and end and export
clock time were not recorded and remain **unknown**. Log events do not establish
those boundaries. Hardware was AMD EPYC 9V74, with a four-core CPU quota and
affinity to CPUs 0–4; observed machine/cgroup state is retained per process.

## What was checked

The frozen baseline was measured at
`2f5bd087ceb4b65a25dc94932e416d8ac5c29740`, binary
`03a7f804bd8c4c0c319083e2e51cef8072fe4634f6b2ebab07632aa69c3e1d20`.
The memo was measured at `d0b6a14a4f317fea3c1e117f627289aad1b1c964`, binary
`8da6eac4e2014b74fc4a05a09938f2bf55f40986f0f9fc4dfa629a2d083abb04`.
Publication restacking and explanatory documentation edits do not alter those
frozen inputs. The archived measured reranking source is the original source.

Historical build records report all four product libraries and the declared
helper freshly compiled from Git-blob-checked workspace inputs with Rust 1.98.1.
The compiled harness SHA remains its exact recorded historical identity. The
original native probe is now publicly inspectable in the reproduction package;
original private execution/build/database controllers remain privately retained.
The new shared sanitized controllers are prospective reproduction tooling, not
byte-exact reconstructions of those historical controllers or proof of the
historical binaries. The public verifier checks retained product source copies,
recorded identities and raw evidence, without certifying a complete public
source-to-binary artifact binding. Original retrieval input is byte-identical at
both frozen commits; its immutable Git blob reference is retained in the receipt.
Shared third-party artifacts were
reused; their historical build provenance is unknown and they are **not**
claimed freshly source-attested. Executed prepared models, tokenizer assets,
external weights, native ORT 1.28/zvec libraries and ELF loader are hash-bound.
Actual node assignments were CPU: 1,023 embedding nodes and 295 reranker nodes.
The original INT8 source export was already absent under CPU prepared-release
policy; this evidence does not treat it as an available executed asset.

The native fixture has 230 documents, 157 golden queries and 11 graph mentions.
The default flat-buffer coverage is 0; no Optimize/reindex or threshold change
was performed. These are four queries (80/81/101/102), not the full 157-query
quality benchmark. The vector channel budget is 50 and actual visible Accurate
selection offers 30 pairs. Available `Why::Reranked` scores are distinguished
from fresh computation by actual counters.

There are 36 serial fresh processes and 428 timed calls: 32 Accurate processes
in two rounds (baseline→memo, then memo→baseline), plus 4 Off controls. Every
Accurate process has fresh A or B, five hot repeats, the other configuration,
five hot repeats and a new query. A uses default Fusion; B disables only the
supported ngram channel per call as a context control. All 36 complete stdout
archives are retained, including regressions. Fresh native processes are evidenced
by historical per-job controller `Popen` records and separate raw records; the
controller source is privately retained. Native process PID/startticks were not retained, so these records do not directly attest
that identity. Retained PID/startticks identify owned PostgreSQL servers.

Fresh, history and hot fused inputs, raw logits, typed Why fields, limited
results, complete result order and gold ranks agree bit-for-bit across source
variants. Warmed final A/B also matches a fresh process with that same final
configuration. Fresh/new Accurate calls score 30 pairs. The q101 changed list
recomputes 20 rows; q80's configuration change preserves its offered list and
recomputes 0. Every hot memo call has encoding and forward operations skipped;
baseline hot calls encode all pairs while performing 0 forwards. Off performs 0
reranker work. This does not imply zero overall CPU work or wall time.

Each initialized disposable cluster was authenticated with `SHOW
data_directory` before native Engine, with owned PID/startticks, executable,
data and port checks and a sibling-listener rejection. Bounded cleanup stopped
every owned server. Each clone started from the same stopped seed; the original
seed's complete bytes and all model/runtime assets stayed unchanged. OOM and
OOM-kill events stayed 0. Minimum free disk remained above 8 GiB. Host available
memory was guarded separately from the qualified finite-cgroup
`max-current+inactive_file` reclaim estimate; reclaim is not guaranteed.

## Descriptive costs and memory

Default A hot wall medians below each use 20 dependent calls across 4 processes
(two fresh-A processes and two B→A histories). Every paired process-block hot
median was lower with the memo. Two rounds and this small workload do not
establish stable population percentiles or strong timing causality.

| Return limit | Query | Baseline hot median | Memo hot median | Absolute difference | Relative difference |
|---|---|---:|---:|---:|---:|
|5|80|3.8845ms|2.8305ms|-1.0540ms|-27.13%|
|5|101|3.6420ms|2.7375ms|-0.9045ms|-24.84%|
|10|80|3.7610ms|2.8970ms|-0.8640ms|-22.97%|
|10|101|3.7050ms|2.5785ms|-1.1265ms|-30.40%|

|Return limit|Query|Within-sample hot wall p95 baseline|Memo|Absolute difference|Relative difference|
|---|---|---:|---:|---:|---:|
|5|80|5.3840ms|4.5435ms|-0.8405ms|-15.61%|
|5|101|4.9524ms|3.4093ms|-1.5431ms|-31.16%|
|10|80|4.6022ms|3.8709ms|-0.7313ms|-15.89%|
|10|101|4.6144ms|3.5699ms|-1.0445ms|-22.64%|

These p95 values interpolate the same 20 dependent hot-call samples; they are
not estimates of stable population tails.

Regressions are retained rather than attributed to the memo:

|Metric/condition|Baseline|Memo|Absolute difference|Relative difference|Observations per variant|
|---|---:|---:|---:|---:|---:|
|Wall median, q101 changed B→A, limit10|598.5880ms|681.7335ms|+83.1455ms|+13.89%|2|
|Wall median, cold A q101, limit5|2298.4750ms|2459.0715ms|+160.5965ms|+6.99%|2|
|Wall median, Off hot, limit5|3.4660ms|4.8380ms|+1.3720ms|+39.58%|1|

Off performed no reranker work. Do not infer cold, new, changed-list or Off speed gains. The JSON includes every group, process block,
raw range and linear-interpolated within-sample percentile.

Wall includes PostgreSQL during search. Per-call process user/system counters
are 100 Hz and exclude PostgreSQL. **Zero ticks are censored**, not measured zero
CPU: each zero field has nominal resolution below 10 ms, or below 20 ms combined.
CPU percent/difference claims are withheld when this resolution dominates.
The JSON preserves observed tick arithmetic only. Process-total `wait4` CPU
was not captured and is N/A; do not reconstruct it from timed calls that omit
startup and diagnostics.

The following RSS/HWM values describe the actual native process, including
model initialization and the outside-clock complete-result diagnostics. They
exclude PostgreSQL memory. They are not the heap cost of the memo.

|Variant|Tier|Observed per-call RSS range (KiB)|Observed per-call HWM range (KiB)|
|---|---|---:|---:|
|baseline|accurate|1,116,940–1,139,868|1,116,940–1,139,868|
|baseline|off|758,696–759,916|758,696–759,916|
|memo|accurate|1,119,056–1,137,740|1,119,056–1,137,740|
|memo|off|756,924–759,536|756,924–759,536|

Paired descriptive memory below uses the median of four process-block medians
for each default A hot group (20 dependent calls). These small signed differences
include diagnostics/allocator state and do not measure memo heap savings.

|Return limit|Query|Metric (KiB)|Baseline|Memo|Absolute difference|Relative difference|
|---|---|---|---:|---:|---:|---:|
|5|80|rss_kib|1,126,254.0|1,126,432.0|+178.0|+0.0158%|
|5|80|hwm_kib|1,126,254.0|1,126,432.0|+178.0|+0.0158%|
|5|101|rss_kib|1,126,072.0|1,125,774.0|-298.0|-0.0265%|
|5|101|hwm_kib|1,126,072.0|1,125,774.0|-298.0|-0.0265%|
|10|80|rss_kib|1,127,558.0|1,127,354.0|-204.0|-0.0181%|
|10|80|hwm_kib|1,127,558.0|1,127,354.0|-204.0|-0.0181%|
|10|101|rss_kib|1,126,118.0|1,126,518.0|+400.0|+0.0355%|
|10|101|hwm_kib|1,126,118.0|1,126,518.0|+400.0|+0.0355%|

Memo metadata is bounded by one plan with at most 4,096 positions and batch
references, and holds no text, encoding or score copies. The 4,096 score-slot
budget is a logical capacity, **not** a 16 KiB total heap bound. Allocator-specific
incremental memo heap was not measured and is N/A.

|Verified quality/work metric|Baseline|Memo|Absolute difference|Relative difference|
|---|---:|---:|---:|---:|
|Output/gold mismatch count across 428 retained timed calls|0|0|0|N/A (zero baseline)|
|Accurate hot calls with observed encoding (160 per variant)|160|0|-160|-100%|
|Accurate hot encode time sum (µs, component counter)|156,902|0|-156,902|-100%|

The encoding counters establish skipped reranker tokenization on these dependent
hot calls; they do not establish a whole-search CPU or population latency gain.
Quality covers the four fixture queries and typed output/rank equivalence only.

## Index disk qualification

|Scope|Index files|Before bytes|After bytes|Absolute difference|Relative difference|Byte identity|
|---|---:|---:|---:|---:|---:|---|
|Original stopped seed|46|22,028,682|22,028,682|0|0%|Unchanged|
|Each of 36 disposable clones|46|22,028,682|22,028,682|0|0%|Four same-size file hashes changed|

Apparent index disk difference is 0 bytes (0%); byte hashes are not invariant.
Memo incremental heap difference and relative change are N/A. Fresh/history/hot
score/order mismatch counts are 0→0, absolute difference 0 and percent change N/A
(the baseline is 0).

The original seed index occupies 22,028,682 bytes. All 36 clones, including Off
and both sources/orders, have that same total before and after. However,
**clone index bytes are not invariant**: each run changes the SHA of
`embedding.index.2.proxima`, `.4.proxima`, `.6.proxima` and `.8.proxima`;
each remains 5,234,688 bytes. Every other index-file digest is unchanged.
The cause was not determined. Disposed post-run files cannot be retroactively
inspected for changed offsets. Before/after aggregates, affected names and
sizes are retained in the receipt, while full private inventories remain
private. This proof establishes matched observed fused results/coverage/gold,
not immutable clone index bytes or a native index-repair mechanism.

There is no default selector, model precision, provider or batch-kernel change.
The diagnostic does not establish broad quality improvements, general
product latency, Apple correctness/performance or a platform-wide default
adoption. Linux unit tests and clippy passed before the documentation-only
restack; actual PR CI and platform gates remain separate.
