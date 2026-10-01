# Read-only Flat-index metadata mutation evidence

Two actual copied-index component runs first changed four Flat-index files at collection drop despite `read_only=true`. Offline complete-file checks found only two source-defined timestamps and their enclosing footer CRC changed. The second run additionally asserted 230 live documents, read one stored FP16 vector and queried 50 hits before drop. Neither run changed the original seed. This is a disk-mutation correctness diagnostic, not an optimization or a product search benchmark.

| Component arm | Run start UTC | Work before collection drop | First changed phase |
| --- | --- | --- | --- |
| Pure open | 2026-10-01T01:02:05.865610+00:00 | SDK initialization, options, collection open | Collection drop |
| Stored-vector query | 2026-10-01T01:36:43.225342+00:00 | Also stats/schema, stored-vector preparation, one SDK vector query (50 hits) | Collection drop |

Exact snapshot observation timestamps are retained in [evidence.json](evidence.json). Preparation includes iterator creation, one `next`, getter, byte copy and document/iterator teardown. The query phase includes construction, output/include-vector settings, execution and result/query teardown. Phase attribution does not identify a runtime call stack. Process exit added no further changes.

Both runs used a synthetic public-input seed of 230 vectors, the same Linux x86_64 shared machine (four-core CPU quota, 16 GiB cgroup limit), zvec-rust/zvec-rust-sys 0.7.2, and the same frozen helper and native library. No Engine, model inference, PostgreSQL, lexical query or build ran in either diagnostic. Stored FP16 bytes were packed two codes per opaque `f32` word, as the product's `half.rs::query` does; no synthetic query vector, decode/reencode or embedding model was used. The helper asserted a 2,048-byte FP16 buffer and native live count 230 in the vector arm. These assertions are a private frozen-helper receipt, not facts reconstructed from the published excerpt alone. The actual query returned 50 hits, also visible in its full stdout.

| Identity | Value |
| --- | --- |
| Historical local source identity (unpublished) | `af917642a1fba637b9a8fc02bffb2c966c50f7ec` |
| Public equivalent production subset | [`47edbae70bbc72b108e4b6b09b938d32edab918c`](https://github.com/pamin-labs/PaminMemory/tree/47edbae70bbc72b108e4b6b09b938d32edab918c), preserved by [tag `evidence/native-readonly-production-source-20261001`](https://github.com/pamin-labs/PaminMemory/tree/evidence%2Fnative-readonly-production-source-20261001) |
| Native SHA256 | `58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a` |
| Frozen helper binary SHA256 | `f7799f4ae77d8b48a3f7e058b63f89ba7a10312475e5150aea28dd10df448228` |
| Vendor source layout | Alibaba zvec `1ab7975dfc2d2160054bafff614831b7099cd930` |
| Options | `read_only=true`, `enable_mmap=true`, `max_buffer_size=67108864` |
| Actual `.proxima` mapping permissions | `rw-s` (writable shared), at open and through query |

Helper/control and runtime asset identities are retained in the JSON. Its vendor source-file hash list covers exactly `index_format.h`, `flat_index_format.h`, `flat_streamer.cc`, `flat_streamer_entity.cc` and `c_api.cc` at their recorded paths; it does not retain an `index_mapping.cc` hash. Fresh compilation of pamin-core, pamin-index and the helper is attested by the private historical build receipt; the complete helper and build state are unavailable publicly.

The [public production-input manifest](public-source-binding.json) binds 35 tracked production/dependency declaration files to exact Git blobs, sizes and SHA256 in the preserved public tree. Those files are byte-identical to the historical local source and its private before/after build-input receipts. This is a limited source-equivalence binding, not a claim that the historical binary was compiled from the public commit or that its full build can be reconstructed. It covers core/index production sources, all workspace Cargo manifests, root Cargo.lock and toolchain declaration; tracked Cargo config paths are absent, and the private config-discovery receipt recorded no present configs. Private helper/controller sources, the full historical tree, actual compiler/environment and cached dependency build state are excluded. Cached third-party artifact provenance was not reconstructed. Offline before/after source, config, asset, binary and original-index equality checks are private receipts. Model caches were unused; their historical whole-cache immutability is unmeasured.

## Exact observed change

`embedding.index.{2,4,6,8}.proxima` each remains 5,234,688 bytes. Each decodes as one header/footer block with no next block. Header and segment-table bytes remain exact. The table's `flat.linear_meta` entry determines the timestamp offset; offsets were not guessed from a diff. Each file has eight differing bytes (32 total across four files), within these three fields:

| Field | Source-defined full extent | Actual differing extent | Seed → pure open → vector query |
| --- | --- | --- | --- |
| Footer CRC32C | [1052544,1052548) | Four bytes | 1103889364 → 883591429 → 3065129832 |
| Footer `update_time` | [1052592,1052600) | [1052592,1052594) | 1790801618 → 1790816526 → 1790818603 |
| StreamerLinearMeta `update_time` | [1056776,1056784) | [1056776,1056778) | Same timestamp values |

The two arms each started from the original seed; the arrows compare their independent outputs, not a sequentially mutated clone. Timestamps are Unix realtime seconds: seed 2026-09-30T20:53:38Z, pure-open drop 2026-10-01T01:02:06Z, vector drop 2026-10-01T01:36:43Z. Footer CRCs independently recompute with raw CRC32C seed zero over 128 bytes after clearing the CRC field. Header/table CRCs also independently recompute. Linear header counts remain 64,64,64,38 (sum 230). They are physical persisted-vector counts, distinct from the live native count assertion.

| Metric | Before | After (each arm) | Absolute difference | Percentage change |
| --- | --- | --- | --- | --- |
| Four changed file sizes, total | 20,938,752 B | 20,938,752 B | 0 B | 0% |
| Whole index disk bytes (private receipt) | 22,028,682 B | 22,028,682 B | 0 B | 0% |
| Added/removed index paths (private receipt) | 0 | 0 | 0 | N/A |
| Product accuracy/quality | N/A | N/A | N/A | N/A |
| Product latency p50/p95, CPU, RSS | N/A | N/A | N/A | N/A |

Retained clone/snapshot overhead is diagnostic disk usage, not a product gain. Native test stdout elapsed seconds are raw harness completion times, not isolated latency measurements, and no speedup is claimed.

## Public verification and omitted evidence

[Pure-open stdout](pure-open.log) and [vector stdout](vector.log) are full native stdout copies inspected for private text. No lines required redaction. They carry no IDs, corpus text or vector bytes. Public JSON labels remove project IDs and paths; it includes only the 64-byte header, 128-byte footer, segment table reconstructed from nonzero runs plus zero padding, and 128 bytes containing the StreamerLinearMeta and LinearIndexHeader. Table segment names and arithmetic offsets are metadata; segment payloads are omitted. No database, complete index, features/vectors, UUID, private path, credentials, full source manifest, controller or execution harness is included.

Run only the offline verifier and invariant tests:

```sh
python3 -B benchmarks/results/index/native-readonly-metadata-2026-10-01/verify.py
(cd benchmarks/results/index/native-readonly-metadata-2026-10-01 && python3 -B -m unittest -v test_verify.py test_source_binding.py)
```

The verifier uses raw Castagnoli CRC32C with seed zero, not IEEE CRC32. Its lookup table is built once. Unchanged large tables reuse a two-entry cache keyed by immutable complete bytes: at most 4 MiB of input payload plus Python object overhead is retained; headers/footers and inputs above 2 MiB are not cached. Byte equality, rather than a digest or CRC alone, identifies cached inputs. Each new content still uses the scalar fallback; no optimized CRC32C dependency is required, and no SIMD or product-search speedup is claimed. Mutation checks continue to validate changed content and semantic invariants. Process RSS was not measured for this cache.

The verifier independently checks included header/table/footer CRCs, bounds including data+padding extents, contiguous append offsets and aggregate content extent, expected revision-0 format/segment names, each file's recorded header magic identity, the 64-byte LinearIndexHeader plus serialized IndexMeta extent, count arithmetic, timestamp agreement, source binding, observed changed ranges and allowed field extents. Negatives include invalid CRC, CRC-refreshed unsupported revision/header magic and wrong linear-header metadata extents, format/bounds/chained format and consistent-before/after padding overflow, segment overlap/gap/reordering, wrong declared storage type/count, an extra metadata change with refreshed valid CRC, expanded mask ranges and fabricated phase attribution. Type comes from the declared helper/field receipt; the omitted IndexMeta payload cannot independently establish FP16 type. Phase facts are checked for consistency with logs and receipts, not independently observed by this verifier. The public source manifest is independently pinned by a literal canonical SHA256 receipt. If Git and its public commit objects are available, the verifier also checks all 35 exact mode/path/blob/size/SHA identities with replacement refs and lazy fetching disabled and inherited Git overrides removed; it never requires the private historical commit object. Missing public Git objects explicitly leave the source-byte check unavailable, while metadata checks remain usable. Public revision and blob aliases, changed paths/SHA and widened scope are rejected. Exact vendor source/runtime identity lists and UTC observation records are pinned private-receipt identities: checking their canonical digest binds the publication, but does not independently inspect runtime assets or remeasure events. File, excerpts and phase-excerpt objects have exact key sets; unsupported nested claims are rejected. Log stages, option getters and hit counts must form one ordered sequence, including rehashed-log negatives for options after open/drop and hits before preparation or after drop. Exactly one completed test summary must report one passed and zero failed/ignored/measured/filtered tests; concatenated failed, duplicate or zero-test summaries are rejected even with a refreshed log hash.

Whole-file before/after SHA256 identities are published for comparison and each of the sixteen receipts is pinned with SHA256 format checks. Header magic is also bound to each file's original recorded value: `625105687`, `3208367440`, `1854750413`, and `2318598652` for files 2, 4, 6, and 8 respectively. The pinned `SetupMetaHeader` source generates this field with `std::random_device()()`; it is not a shared format constant. CRC-refreshed zero and cross-file identity substitutions are rejected independently for each file in both arms. Those checks bind recorded identities; they do not recompute the omitted whole files. **They cannot validate omitted payloads.** Full-file masked equality, unchanged feature segments, complete stage inventories and original-seed equality were verified privately against retained bytes. Those are explicitly private verification receipts; public excerpts cannot reproduce that omitted-byte equality proof. The public verifier does not authorize ignoring timestamps or relaxing any index guard.

## Source interpretation and scope

Every published excerpt byte and receipt is also bound by the verifier's literal canonical JSON SHA256 `61b72c36d017b0a79a70dbc9b8484869c6f7095abe08a988aad6a5f6376b2a22`, independently derived from the original [publication commit `444b8e7771f76351a7ec5b2f9c4a276505d39d22`](https://github.com/pamin-labs/PaminMemory/blob/444b8e7771f76351a7ec5b2f9c4a276505d39d22/benchmarks/results/index/native-readonly-metadata-2026-10-01/evidence.json), Git blob `98786c984cb13a517599c9b575618c05e48f7e4e`. Canonicalization uses Python `json.dumps` with sorted keys, separators `(',', ':')`, default ASCII escaping and `allow_nan=False`, then UTF8 and SHA256. The pin is independent of the mutable evidence and source manifest. Consistent before/after edits to otherwise unchecked header `h[9]`, linear-header `lh[2]`, or streamer padding are rejected even when their enclosing CRCs are refreshed and every whole-file SHA256 receipt stays unchanged. Original evidence and raw stdout remain byte-identical. This binds the retained publication; it does not independently prove that the excerpts came from the omitted whole files or verify omitted payload equality.

Vendor source at the recorded revision provides a close path consistent with the observation: Collection destruction closes segments; vector index cleanup calls FlatStreamer close; `flush_linear_meta` writes its realtime timestamp; dirty mmap storage refresh updates the footer timestamp and CRC. Read-only top-level flush guards do not guard this cleanup path. Storage options default `copy_on_write=false`, consistent with actual writable shared mappings. JSON retains the five source-file identities listed above; other causal references below are revision-pinned upstream links without retained file hashes. See [FlatStreamer close](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/flat/flat_streamer.cc), [metadata flush](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/flat/flat_streamer_entity.cc), [IndexMapping refresh](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/framework/index_mapping.cc), and [format layout](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/include/zvec/core/framework/index_format.h).

Source-layout decoding matches the actual metadata bytes. This is not a reproducible source build of the released native binary, and no runtime call stack was captured. Two runs establish only the behavior observed on this Flat component/native pin; they do not establish HNSW, read-write access, product graph/Engine behavior, lexical safety, ranking/self-match, returned-vector completeness, another format/corpus/pin or a general read-only guarantee. The 36 previously disposed clones retain only SHA-level historical evidence; this cannot retroactively diagnose their fields. Native/product defaults and strict byte guards remain unchanged. No whitelist or guard relaxation is proposed.

## Verifier-component cost observation

[Compact recorded receipt](verifier-cost.json) retains full-precision clock
values, source/input identities and the complete table payload SHA256
`fcfab93bd3a47ad3e3ae6a76ae3e408bfbc88ca7cfb779ead5b36ccc614d82ea`.
These are **one shared-host observation per case**, covering the same 16 metadata
excerpts, with no product search, native/SDK, model or database execution.
Timers cover metadata validation, logs and the canonical evidence digest;
interpreter startup/imports, input loading and Git source-byte checks are excluded.
They do not establish statistical latency or a product speedup.

| Metric | Before | After | Absolute difference | Percentage change |
| --- | ---: | ---: | ---: | ---: |
| Wall seconds (cold) | 2.153045189 | 0.716812514 | -1.436232675 | -66.707038% |
| Process CPU seconds (cold) | 2.152929824 | 0.716466029 | -1.436463795 | -66.721348% |
| Scalar CRC table bytes (cold) | 16839680 | 1052480 | -15787200 | -93.750000% |
| Wall seconds (warm) | 2.153045189 | 0.589296649 | -1.563748540 | -72.629620% |
| Process CPU seconds (warm) | 2.152929824 | 0.589027994 | -1.563901830 | -72.640632% |
| Scalar CRC table bytes (warm) | 16839680 | 0 | -16839680 | -100.000000% |
| Maximum retained CRC cache input payload bytes | 0 | 4194304 | +4194304 | N/A (zero baseline) |
| Process RSS bytes | N/A | N/A | N/A | N/A |

All 16 retained tables have one identical complete byte payload. Cold validation
has 1 scalar table scan and 15 cache hits; warm validation has 0 scalar table
scans and 16 hits. Scalar-byte counts exclude byte-key hashing/equality scans,
reconstruction/copying, header/footer CRCs, metadata comparisons and other hash
work: they are **not total memory transfer**. The cache bound covers retained
input payload only; lookup/cache/Python object overhead and process RSS were not
measured. The full 69-test scan audit in the receipt includes changed tables and
deliberate cache tests; it is separate from these three timed observations.

To produce new observations later, use the read-only
[probe](probe_verifier_cost.py) from this directory. Its exact before-comparator
source is now retained as [before-verify.py.in](before-verify.py.in), checked
against the original SHA256 in the receipt before evaluation. The recorded
`775de8a2d0e81a80239df4a0279600da7df68ec0` revision remains provenance; no Git
object, ref, remote fetch or download is required for this component procedure.
Current prospective source/input pins are listed separately from the unchanged
historical measured-source identities. The missing-tree Git availability fix
is outside the timed metadata path. This procedure produces new timings, not
identical historical values, and does not run the invariant suite or product work:

```sh
python3 -B probe_verifier_cost.py
python3 -B probe_verifier_cost.py --execute
```

The first command only prints the plan. The second opts in to before, cold-after
and warm-after metadata verification in one process. No new observation was run
while adding this receipt/procedure; the numbers above retain the prior
verifier-only observations.

The separate [suite scan producer](probe_suite_scans.py) reproduces all five
full-suite counters using the same scalar-function instrumentation: clear the
cache, run `test_verify` and `test_source_binding`, count calls and byte lengths
for inputs above 256 bytes, and count distinct full-input SHA256 values and
calls matching the retained table SHA256. It verifies current suite/input pins
before execution. The suite uses tiny disposable mock/Git directories; published
files remain unchanged. It does not rerun the three cost observations.

```sh
python3 -B probe_suite_scans.py
python3 -B probe_suite_scans.py --execute
python3 -B -m unittest -v test_cost_procedures.py
```

The first command only prints a plan. The explicit suite command can regenerate
69 tests, 17 large scalar calls, 8,424,450 scalar bytes, 13 distinct contents and
two original-table scans. Strengthened Git availability subcases do not change
the metadata CRC workload. These counts include deliberate cache eviction tests
and changed/rehashed tables, and exclude hashing/equality and other byte traffic.
No historical cost value or original native evidence was replaced by this
procedure update.
