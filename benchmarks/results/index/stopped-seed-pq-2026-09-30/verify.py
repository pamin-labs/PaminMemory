"""Verify public excerpts; never claim omitted file/section bytes were checked."""
import hashlib
import json
from pathlib import Path
import struct

from inspect_native import crc32c


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_row(row):
    excerpts = row['excerpts']

    def read(name, offset, size):
        item = excerpts[name]
        data = bytes.fromhex(item['hex'])
        require((item['offset'], item['bytes'], len(data)) == (offset, size, size),
                f'{name}: invalid offset/length')
        require(hashlib.sha256(data).hexdigest() == item['sha256'],
                f'{name}: excerpt SHA256 mismatch')
        return data

    size = row['bytes']
    header = read('container_header', 0, 64)
    hc, _, version, revision, _, hs, fs, fo, co, *_ = struct.unpack('<IHHIIHHIQQQQQ', header)
    require((version, revision, hs, fs, co) == (2, 0, 64, 128, 64), 'Unknown container format')
    require(crc32c(b'\0' * 4 + header[4:]) == hc, 'Header CRC mismatch')
    footer_offset = fo if fo < 2**31 else size + fo - 2**32
    require(footer_offset == size - 128, 'Unsupported footer location')
    footer = read('container_footer', footer_offset, 128)
    fc, table_crc, _, count, table_size, _, content_size, padding, *tail = struct.unpack('<IIIIII13Q', footer)
    require(crc32c(b'\0' * 4 + footer[4:]) == fc, 'Footer CRC mismatch')
    require(tail[-2] == 0 and tail[-1] == size, 'Chained package unsupported')
    require(0 < count <= 64 and count * 32 <= table_size <= 65536, 'Invalid table length')
    require(co + content_size + padding + table_size == footer_offset, 'Invalid container extent')
    table = read('segment_table', footer_offset - table_size, table_size)
    require(crc32c(table) == table_crc, 'Table CRC mismatch')
    sections = {}
    for i in range(count):
        no, checksum, index, length, section_padding = struct.unpack_from('<IIQQQ', table, i * 32)
        require(count * 32 <= no < len(table), 'Invalid section name offset')
        end = table.index(0, no)
        name = table[no:end].decode('ascii')
        require(name not in sections, 'Duplicate section')
        require(index + length + section_padding <= content_size, 'Section exceeds container')
        sections[name] = dict(offset=co + index, size=length, padding=section_padding, crc32c=checksum)
    require(all(name in sections for name in ['diskann.meta', 'diskann.pq_meta', 'diskann.pq_data']),
            'Unsupported non-DiskANN package')
    require({name: sections[name] for name in row['sections']} == row['sections'], 'Reported sections differ')
    sm, sp, sd = [sections[name] for name in ['diskann.meta', 'diskann.pq_meta', 'diskann.pq_data']]
    require(sm['size'] == 4096, 'Unknown DiskANN metadata format')
    docs, stored_dims, *_ = struct.unpack('<10Q', read('diskann_meta_prefix', sm['offset'], 80))
    require(docs > 0 and stored_dims > 0, 'Invalid documents/dimensions')
    pq = read('diskann_pq_meta_header', sp['offset'], 160)
    quantizer_bytes, chunks = struct.unpack_from('<QQ', pq)
    require(pq[16:] == b'\0' * 144, 'Legacy/unknown PQ header rejected')
    require(0 < chunks <= stored_dims and quantizer_bytes >= 44, 'Invalid PQ dimensions')
    require(sp['size'] == 160 + quantizer_bytes, 'PQ metadata extent mismatch')
    require(sd['size'] == docs * chunks, 'PQ code byte count mismatch')
    q = read('quantizer_fixed_prefix', sp['offset'] + 160, 44)
    magic, qv, qt, dims, metric, payload_size, dtype, reserved = struct.unpack('<IHHIIIHH', q[:24])
    original_dim, quantizer_chunks, chunk_dim, centroids, zero_mean, input_dtype, rotate, payload_reserved = struct.unpack('<IIIIBBBB', q[24:])
    require(magic == 0x52545A51 and (qv, qt, dtype, reserved, metric) == (1, 5, 1, 0, 0), 'Unknown quantizer format')
    require(dims == original_dim and quantizer_chunks == chunks, 'PQ header/payload disagreement')
    require(dims % chunks == 0 and chunk_dim == dims // chunks, 'Invalid chunk dimensions')
    require((centroids, zero_mean, input_dtype, rotate, payload_reserved) == (256, 0, 2, 0, 0), 'Unsupported quantizer payload')
    require(payload_size == quantizer_bytes - 24 == 20 + dims * centroids * 2, 'Codebook length mismatch')
    actual = dict(documents=docs, stored_dimensions=stored_dims, effective_pq_chunks=chunks,
                  quantizer_dimensions=dims, quantizer_payload_chunks=quantizer_chunks,
                  centroids_per_chunk=centroids, pq_data_bytes=sd['size'],
                  pq_metadata_bytes=sp['size'], quantizer_serialized_bytes=quantizer_bytes)
    require(all(row[key] == value for key, value in actual.items()), 'Reported values differ from excerpts')
    require(row['fixed_pq_header_sha256'] == hashlib.sha256(pq).hexdigest(), 'PQ header hash mismatch')
    require(row['quantizer_fixed_prefix_sha256'] == hashlib.sha256(q).hexdigest(), 'Quantizer prefix hash mismatch')
    require(row['quantizer_input_type'] == 'fp16' and row['quantizer_code_type'] == 'uint8', 'Reported dtype mismatch')
    return docs, chunks


def verify(document, provenance):
    require(document['requested_pq_chunks'] == provenance['conversion_schema_and_logical_digest'][1]['schema']['fields']['embedding']['pq_chunks'] == 0,
            'Requested setting mismatch')
    require(document['native_library_sha256'] == provenance['native_zvec_library']['sha256'], 'Native identity mismatch')
    expected = {item['path'].split('<SCRATCH>/seed-disk/', 1)[1]: item
                for item in provenance['seed_files'] if 'embedding.index.' in item['path']}
    require(len(document['files']) == len(expected) == document['file_count'] == 10, 'File count mismatch')
    seen = set()
    documents = 0
    for row in document['files']:
        name = row['seed_relative_path']
        require(name in expected and name not in seen, 'Unknown/duplicate seed file')
        seen.add(name)
        require((row['bytes'], row['sha256']) == (expected[name]['bytes'], expected[name]['sha256']),
                'Recorded file identity differs from public pretrial provenance')
        docs, chunks = verify_row(row)
        require(chunks == document['effective_pq_chunks'] == 512, 'Effective PQ differs')
        documents += docs
    require(documents == document['documents'] == provenance['seed_documents'] == 18000, 'Document count mismatch')
    return dict(file_count=len(seen), documents=documents, requested_pq_chunks=0,
                effective_pq_chunks=512, public_whole_file_sha_or_complete_section_crc_verified=False)


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    provenance_path = here.parent / 'restart-floor-disk-2026-09-30/provenance.json'
    document = json.loads((here / 'excerpts.json').read_text())
    require(hashlib.sha256(provenance_path.read_bytes()).hexdigest() == document['provenance_sha256'], 'Provenance hash mismatch')
    print(json.dumps(verify(document, json.loads(provenance_path.read_text())), sort_keys=True))
