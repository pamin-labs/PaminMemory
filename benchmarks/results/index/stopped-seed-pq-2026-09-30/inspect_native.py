"""Offline metadata-only guard for this stopped Linux seed; never load zvec."""
import argparse, hashlib, json, struct
from pathlib import Path

POLY = 0x82F63B78
TABLE = []
for value in range(256):
    for _ in range(8):
        value = (value >> 1) ^ (POLY if value & 1 else 0)
    TABLE.append(value)

def crc32c(data, seed=0):
    for value in data:
        seed = TABLE[(seed ^ value) & 255] ^ (seed >> 8)
    return seed

def exact(handle, offset, size, total):
    if not (0 <= offset <= total and 0 <= size <= total - offset):
        raise ValueError('Unsupported or malformed index metadata')
    handle.seek(offset)
    data = handle.read(size)
    if not (len(data) == size):
        raise ValueError('Unsupported or malformed index metadata')
    return data

def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(data)
    return result.hexdigest()

def inspect(path, expected):
    size = path.stat().st_size
    if not (size == expected['bytes']):
        raise ValueError('Unsupported or malformed index metadata')
    actual_sha = file_sha(path)
    if not (actual_sha == expected['sha256']):
        raise ValueError('Stopped file differs from retained seed')
    with path.open('rb') as h:
        header = exact(h, 0, 64, size)
        hc, _, version, revision, _, hs, fs, fo, co, *_ = struct.unpack('<IHHIIHHIQQQQQ', header)
        if not ((version, revision, hs, fs, co) == (2, 0, 64, 128, 64)):
            raise ValueError('Unsupported or malformed index metadata')
        if not (crc32c(b'\x00' * 4 + header[4:]) == hc):
            raise ValueError('Unsupported or malformed index metadata')
        footer_offset = fo if fo < 2**31 else size + fo - 2**32
        if not (footer_offset == size - 128):
            raise ValueError('Not a single immutable footer-at-end package')
        footer = exact(h, footer_offset, 128, size)
        fc, table_crc, _, count, table_size, _, content_size, content_padding, *tail = struct.unpack('<IIIIII13Q', footer)
        if not (crc32c(b'\x00' * 4 + footer[4:]) == fc):
            raise ValueError('Unsupported or malformed index metadata')
        if not (tail[-2] == 0 and tail[-1] == size):
            raise ValueError('Chained package unsupported')
        if not (0 < count <= 64 and count * 32 <= table_size <= 65536):
            raise ValueError('Unsupported or malformed index metadata')
        if not (co + content_size + content_padding + table_size == footer_offset):
            raise ValueError('Unsupported or malformed index metadata')
        table = exact(h, footer_offset - table_size, table_size, size)
        if not (crc32c(table) == table_crc):
            raise ValueError('Unsupported or malformed index metadata')
        sections = {}
        for i in range(count):
            no, checksum, index, length, padding = struct.unpack_from('<IIQQQ', table, i * 32)
            if not (count * 32 <= no < len(table)):
                raise ValueError('Unsupported or malformed index metadata')
            stop = table.index(0, no)
            name = table[no:stop].decode('ascii')
            if not (name not in sections):
                raise ValueError('Unsupported or malformed index metadata')
            if not (index + length + padding <= content_size):
                raise ValueError('Unsupported or malformed index metadata')
            sections[name] = dict(offset=co+index, size=length, padding=padding, crc32c=checksum)
        for name in ['diskann.meta', 'diskann.pq_meta', 'diskann.pq_data']:
            if not (name in sections):
                raise ValueError('Not a supported DiskANN package')
        sm, sp, sd = (sections[x] for x in ['diskann.meta', 'diskann.pq_meta', 'diskann.pq_data'])
        if not (sm['size'] == 4096):
            raise ValueError('Unsupported or malformed index metadata')
        disk_meta = exact(h, sm['offset'], sm['size'], size)
        if not (crc32c(disk_meta) == sm['crc32c']):
            raise ValueError('Unsupported or malformed index metadata')
        docs, stored_dims, *_ = struct.unpack_from('<10Q', disk_meta)
        if not (docs > 0 and stored_dims > 0):
            raise ValueError('Unsupported or malformed index metadata')
        pq_meta = exact(h, sp['offset'], 160, size)
        quantizer_bytes, chunks = struct.unpack_from('<QQ', pq_meta)
        if not (pq_meta[16:] == b'\x00' * 144):
            raise ValueError('Legacy/unknown PQ header layout rejected')
        if not (0 < chunks <= stored_dims and quantizer_bytes >= 44):
            raise ValueError('Unsupported or malformed index metadata')
        if not (sp['size'] == 160 + quantizer_bytes):
            raise ValueError('Unsupported or malformed index metadata')
        if not (sd['size'] == docs * chunks):
            raise ValueError('Unsupported or malformed index metadata')
        quantizer = exact(h, sp['offset']+160, 44, size)
        magic, qv, qt, dims, metric, payload_size, dtype, reserved = struct.unpack('<IHHIIIHH', quantizer[:24])
        original_dim, quantizer_chunks, chunk_dim, centroids, zero_mean, input_dtype, rotate, payload_reserved = struct.unpack('<IIIIBBBB', quantizer[24:])
        if not (magic == 1381259857 and (qv, qt, dtype, reserved, metric) == (1, 5, 1, 0, 0)):
            raise ValueError('Unsupported or malformed index metadata')
        if not (dims == original_dim and quantizer_chunks == chunks):
            raise ValueError('Unsupported or malformed index metadata')
        if not (dims % chunks == 0 and chunk_dim == dims // chunks):
            raise ValueError('Unsupported or malformed index metadata')
        if not ((centroids, zero_mean, input_dtype, rotate, payload_reserved) == (256, 0, 2, 0, 0)):
            raise ValueError('Unsupported or malformed index metadata')
        if not (payload_size == quantizer_bytes - 24):
            raise ValueError('Unsupported or malformed index metadata')
        if not (payload_size == 20 + dims * centroids * 2):
            raise ValueError('Unsupported or malformed index metadata')
        # Check serialized PQ metadata CRC without exposing codebook values.
        pq_all = exact(h, sp['offset'], sp['size'], size)
        if not (crc32c(pq_all) == sp['crc32c']):
            raise ValueError('Unsupported or malformed index metadata')
        metadata_bytes_read = 64 + 128 + table_size + 4096 + 160 + 44 + sp['size']
        return dict(path=str(path), bytes=size, sha256=actual_sha,
            provenance_match=True, header_footer_table_crc_valid=True,
            disk_meta_crc_valid=True, pq_meta_crc_valid=True,
            documents=docs, stored_dimensions=stored_dims,
            effective_pq_chunks=chunks, quantizer_dimensions=dims,
            quantizer_payload_chunks=quantizer_chunks,
            quantizer_input_type='fp16', quantizer_code_type='uint8',
            centroids_per_chunk=centroids, pq_data_bytes=sd['size'],
            pq_metadata_bytes=sp['size'], quantizer_serialized_bytes=quantizer_bytes,
            metadata_bytes_read=metadata_bytes_read,
            fixed_pq_header_sha256=hashlib.sha256(pq_meta).hexdigest(),
            quantizer_fixed_prefix_sha256=hashlib.sha256(quantizer).hexdigest(),
            sections={k:sections[k] for k in ['diskann.meta','diskann.pq_meta','diskann.pq_data']})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True, type=Path)
    ap.add_argument('--provenance', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    source = json.loads(args.provenance.read_text())
    expected = {x['path'].split('<SCRATCH>/seed-disk/',1)[1]:x
                for x in source['seed_files'] if 'embedding.index.' in x['path']}
    paths = sorted((args.seed/'index').rglob('embedding.index.*.proxima'))
    if not (len(paths) == len(expected) == 10):
        raise ValueError('Unsupported or malformed index metadata')
    rows = [inspect(p, expected[p.relative_to(args.seed).as_posix()]) for p in paths]
    if not (sum((x['documents'] for x in rows)) == source['seed_documents'] == 18000):
        raise ValueError('Unsupported or malformed index metadata')
    if not ({x['effective_pq_chunks'] for x in rows} == {512}):
        raise ValueError('Unsupported or malformed index metadata')
    output = dict(scope='Offline stopped seed only; no native library loaded; no vector payload decoded',
        native_library_sha256=source['native_zvec_library']['sha256'],
        provenance_sha256=file_sha(args.provenance),
        requested_pq_chunks=source['conversion_schema_and_logical_digest'][1]['schema']['fields']['embedding']['pq_chunks'],
        file_count=len(rows), documents=sum(x['documents'] for x in rows),
        effective_pq_chunks=512, files=rows)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ['file_count','documents','requested_pq_chunks','effective_pq_chunks']}))
