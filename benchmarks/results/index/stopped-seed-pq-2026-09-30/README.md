# Persisted PQ in the current stopped Linux seed

The retained stopped restart-floor DiskANN seed has **512 persisted PQ chunks**
although its requested schema says `pq_chunks: 0`. All 10 embedding index files
agree in `DiskAnnPqMeta.chunk_num` and `PqInt8SerPayload.num_chunk`; their document
counts sum to 18,000. Every `diskann.pq_data` section has exactly
`document_count * 512` bytes. The quantizer has 1,024 input dimensions, two
FP16 dimensions per chunk and 256 centroids per chunk, storing uint8 codes.
The stored-vector metadata has 1,026 dimensions following metric conversion.

This is a prospective, read-only inspection of this current stopped seed from
the original archive series labeled 2026-09-30. The inspection capture timestamp
was not recorded; the public export was prepared 2026-10-01 UTC. The private
summary's 2026-09-30 heading is an archive label, not a retained UTC capture
record. This does not verify the state of every historical trial copy,
the 50k synthetic requested-zero/requested-64 runs, macOS, other seeds or Flat
blocks. It establishes no recall, latency, memory, source-build attestation,
post-write visibility or performance improvement. Those metrics are **N/A**.
No native library, database, model, index API, build or maintenance ran.

## Public evidence and omitted bytes

[excerpts.json](excerpts.json) retains per-file identities and sanitized binary
excerpts: the entire 64-byte container header, 128-byte footer, segment table,
80-byte DiskANN metadata prefix, 160-byte PQ header and 44-byte quantizer
prefix. It omits vectors, graph data, PQ codes, centroid values, database
contents and private absolute paths. No whole index is published.

The [pretrial provenance](../restart-floor-disk-2026-09-30/provenance.json)
contains the requested schema, original seed sizes/hashes and Linux runtime
library SHA256 `58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a`.
The per-file public receipt records that private inspection streamed each
current original file and matched all 10 full-file SHA256s to that pretrial
manifest, and checked the complete DiskANN/PQ metadata CRCs. **The excerpts do
not cryptographically prove those full-file hashes or complete-section CRCs.**
Those bytes remain private. Matching recorded hashes to the manifest checks
receipt consistency, not the omitted file contents. Excerpt hashes likewise
check internal consistency, not independent authenticity.

[verify.py](verify.py) independently recomputes container-header/footer/table
CRCs (those complete bytes are included), section layout including padding and
non-overlap of every occupied section extent, document counts,
PQ header/payload agreement and codebook/code-length arithmetic. It checks
reported native/file identity against the public pretrial manifest and prints
`public_whole_file_sha_or_complete_section_crc_verified: false`. Offsets are
parsed from the segment table; they are not assumed from a sample file.
Unknown, legacy, Flat, chained or inconsistent formats fail closed, including
with Python assertions disabled. Zero requested chunks never substitute for
persisted chunks.

## Pinned format source

The [source audit](../zvec-native-source-audit-2026-09-30.md) identifies wrapper
0.7.2 and vendor commit `1ab7975dfc2d2160054bafff614831b7099cd930`. The parser
uses that immutable source's little-endian layout (immutable URLs and source
SHA256s are retained in [source-manifest.json](source-manifest.json)):

- [index_format.h](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/include/zvec/core/framework/index_format.h): container header/footer and segment table.
- [index_unpacker.h](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/include/zvec/core/framework/index_unpacker.h): footer/table location and section offsets.
- [diskann_entity.h](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/diskann/diskann_entity.h): document count and 160-byte PQ header.
- [diskann_builder_entity.cc](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/core/algorithm/diskann/diskann_builder_entity.cc): writes quantizer metadata and `doc_cnt * chunk_num` code bytes.
- [quantizer.h](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/turbo/quantizer/quantizer.h): serialization header.
- [pq_int8_quantizer.cc](https://github.com/alibaba/zvec/blob/1ab7975dfc2d2160054bafff614831b7099cd930/src/turbo/quantizer/pq_int8_quantizer/pq_int8_quantizer.cc): serialized chunk count, FP16 codebook and uint8 codes.

CRC32C uses the raw Castagnoli update seeded at zero, matching the container
implementation, rather than the complemented convention of many CRC libraries.
The source rule for automatic chunks does not by itself attest an installed
binary; the persisted headers are separate evidence for this seed.

## Verification

From this directory, using Python's standard library only:

```sh
python3 verify.py
python3 -m unittest -v test_verify.py
python3 -O -m unittest test_verify.py
```

The verifier returns 10 files, 18,000 documents, requested zero and effective
512. Eighteen tests (16 metadata guards and two archive guards) cover malformed
header/footer/table CRCs, unknown container
with valid CRC, unknown quantizer, legacy header, mismatched chunk counts,
unknown metric, Flat package, code-length arithmetic, zero-as-no-PQ rejection,
omitted-file identity rejection and a synthetic full
metadata CRC corruption. The archive guards accept the retained archive and
reject mutated provenance after checking the archive identity.
Synthetic omitted content is zero-filled; tests do
not read the private seed. CRC-refreshed section overlap and padding-only overlap
fixtures are rejected by both readers before persisted-PQ interpretation.
Adjacent extents and zero-length unpadded sections remain valid. The shared
range check uses each serialized section padding value; it imposes no new
alignment constant or contiguity rule. Complete PQ-data CRC validation remains
unavailable from excerpts and is not added by this layout check.

[inspect_native.py](inspect_native.py) is the standalone offline parser for
holders of original files. Its full-file SHA256 and complete metadata CRC
checks are stronger than the public excerpt verifier. It reads files only,
never imports zvec, and refuses files that differ from the supplied manifest:

```sh
python3 inspect_native.py --seed /path/to/stopped-seed \
  --provenance ../restart-floor-disk-2026-09-30/provenance.json \
  --output /path/to/private-inspection.json
```

The native parser's output includes input file paths and should remain private
until sanitized. Original private receipts are retained separately; this
public export does not replace them. This inspection closes the effective-PQ
question for this seed only. A future comparison still needs an effective-value
guard for each arm; vector-only visibility after a promoted write, flush,
maintenance and read-only reopen remains unmeasured.
