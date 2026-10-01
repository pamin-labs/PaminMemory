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
| Product source | `af917642a1fba637b9a8fc02bffb2c966c50f7ec` |
| Native SHA256 | `58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a` |
| Frozen helper binary SHA256 | `f7799f4ae77d8b48a3f7e058b63f89ba7a10312475e5150aea28dd10df448228` |
| Vendor source layout | Alibaba zvec `1ab7975dfc2d2160054bafff614831b7099cd930` |
| Options | `read_only=true`, `enable_mmap=true`, `max_buffer_size=67108864` |
| Actual `.proxima` mapping permissions | `rw-s` (writable shared), at open and through query |

Helper/control and runtime asset identities and relevant vendor source-file hashes are retained in the JSON. Product/helper source was freshly compiled and attested at build time; cached third-party artifact provenance was not reconstructed. Offline before/after source, config, asset, binary and original-index equality checks are private receipts. Model caches were unused; their historical whole-cache immutability is unmeasured.

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
(cd benchmarks/results/index/native-readonly-metadata-2026-10-01 && python3 -B -m unittest -v test_verify.py)
```

The verifier independently checks included header/table/footer CRCs, bounds including data+padding extents, contiguous append offsets and aggregate content extent, expected format/segment names, count arithmetic, timestamp agreement, source binding, observed changed ranges and allowed field extents. Negatives include invalid CRC, CRC-refreshed wrong format/bounds/chained format and consistent-before/after padding overflow, segment overlap/gap/reordering, wrong declared storage type/count, an extra metadata change with refreshed valid CRC, expanded mask ranges and fabricated phase attribution. Type comes from the declared helper/field receipt; the omitted IndexMeta payload cannot independently establish FP16 type. Phase facts are checked for consistency with logs and receipts, not independently observed by this verifier. Exact source/runtime identity lists and UTC observation records are pinned private-receipt identities: checking their canonical digest binds the publication, but does not independently inspect runtime assets or remeasure events. Log option getters and hit counts are parsed with exact cardinality, including rehashed-log negative cases.

Whole-file before/after SHA256 identities are published for comparison. **They cannot validate omitted payloads.** Full-file masked equality, unchanged feature segments, complete stage inventories and original-seed equality were verified privately against retained bytes. Those are explicitly private verification receipts; public excerpts cannot reproduce that omitted-byte equality proof. The public verifier does not authorize ignoring timestamps or relaxing any index guard.

## Source interpretation and scope

Pinned vendor source provides a close path consistent with the observation: Collection destruction closes segments; vector index cleanup calls FlatStreamer close; `flush_linear_meta` writes its realtime timestamp; dirty mmap storage refresh updates the footer timestamp and CRC. Read-only top-level flush guards do not guard this cleanup path. Storage options default `copy_on_write=false`, consistent with actual writable shared mappings. Relevant source identities are in JSON; see [FlatStreamer close](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/flat/flat_streamer.cc), [metadata flush](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/flat/flat_streamer_entity.cc), [IndexMapping refresh](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/framework/index_mapping.cc), and [format layout](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/include/zvec/core/framework/index_format.h).

Source-layout decoding matches the actual metadata bytes. This is not a reproducible source build of the released native binary, and no runtime call stack was captured. Two runs establish only the behavior observed on this Flat component/native pin; they do not establish HNSW, read-write access, product graph/Engine behavior, lexical safety, ranking/self-match, returned-vector completeness, another format/corpus/pin or a general read-only guarantee. The 36 previously disposed clones retain only SHA-level historical evidence; this cannot retroactively diagnose their fields. Native/product defaults and strict byte guards remain unchanged. No whitelist or guard relaxation is proposed.
