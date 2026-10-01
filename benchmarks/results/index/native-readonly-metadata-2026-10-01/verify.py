#!/usr/bin/env python3
"""Verify published metadata only; never opens an SDK, database or payload file."""
import datetime
import hashlib
import json
import re
import struct
from pathlib import Path

import source_binding

ROOT = Path(__file__).resolve().parent
HEADER = struct.Struct('<IHHIIHHIQQ3Q')
FOOTER = struct.Struct('<IIIIIIQQQQ7QQQ')
SEGMENT = struct.Struct('<IIQQQ')
LINEAR = struct.Struct('<IIQIIIII28s')
NAMES = {'IndexVersion', 'flat.linear_meta', 'IndexMeta', 'flat.linear_list_head', 'flat.features1'}

# SetupMetaHeader assigns magic using std::random_device(), not a format constant.
# Per-file identities from the original publication, commit
# 444b8e7771f76351a7ec5b2f9c4a276505d39d22, evidence.json blob
# 98786c984cb13a517599c9b575618c05e48f7e4e; identical in both arms/phases.
HEADER_MAGIC = {
    'embedding.index.2.proxima': 625105687,
    'embedding.index.4.proxima': 3208367440,
    'embedding.index.6.proxima': 1854750413,
    'embedding.index.8.proxima': 2318598652,
}


# Identity receipts only: omitted whole-file payloads are never verified here.
WHOLE_FILE_SHA256 = {'pure-open': {'embedding.index.2.proxima': ('781e78bad2994332a4fc3614d4d323c5517acfd8e329e941713fa2a92fb79522', 'd9d326f771e174fa829bfe1fec4deb7dabce204937ca8fb702bcc4f45f2f1a67'), 'embedding.index.4.proxima': ('47db1851ed2c2838236b5a144043fc357b73da2c599e438ba91092886ae863bc', 'be169c527efc44c19438835785dbd7448b50235930e23a96eda7e62dd351cf22'), 'embedding.index.6.proxima': ('30959a1299958d26835d004560f7f4b70eca40b020bb9c17759f4ce7ede3d849', '44c714a9a4315da19cf9289861a638d88a82476faffd12908e673d4436d59d71'), 'embedding.index.8.proxima': ('2bb0ea85ce32144e60e82089e619406d0c6fb4d582b470cbec32d57e434c7c35', 'afa255490e3dade9e0ce2b341e728efe5dfe69f371a55bc1a28c7a738d8fe8be')}, 'vector': {'embedding.index.2.proxima': ('781e78bad2994332a4fc3614d4d323c5517acfd8e329e941713fa2a92fb79522', '4e5a3e98656d2ac4a0ee6847bf224383a1e3f49a03bd80a72a61f664ba352ef1'), 'embedding.index.4.proxima': ('47db1851ed2c2838236b5a144043fc357b73da2c599e438ba91092886ae863bc', 'c6816c8d5febcee0fd51be3878061780834b09f214c9777762660e4bb620e2ec'), 'embedding.index.6.proxima': ('30959a1299958d26835d004560f7f4b70eca40b020bb9c17759f4ce7ede3d849', 'e55c616663934b6f3f31427a67acbe2cc80720d0a42c496e1ec048164e950488'), 'embedding.index.8.proxima': ('2bb0ea85ce32144e60e82089e619406d0c6fb4d582b470cbec32d57e434c7c35', '656c7ee3bc65805e1ec5edd3ae250cacbaaf9808aedd6e53743d0d9fa9c0e60d')}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def crc32c(data):
    # Raw Castagnoli recurrence, seed zero; offline metadata, no timing claim.
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82f63b78 if value & 1 else 0)
        table.append(value)
    crc = 0
    for byte in data:
        crc = table[(crc ^ byte) & 255] ^ (crc >> 8)
    return crc



def decode(excerpt, file_size, expected_magic):
    require(set(excerpt) == {'header_hex', 'footer_hex', 'table_size', 'table_nonzero_runs', 'streamer_and_linear_header_hex'}, 'phase excerpt schema/scope')
    header = bytes.fromhex(excerpt['header_hex'])
    footer = bytes.fromhex(excerpt['footer_hex'])
    linear = bytes.fromhex(excerpt['streamer_and_linear_header_hex'])
    require(len(header) == 64 and len(footer) == 128 and len(linear) == 128, 'metadata extent')
    h, f = HEADER.unpack(header), FOOTER.unpack(footer)
    require(h[2] == 2 and h[3] == 0 and h[5:7] == (64, 128), 'wrong format/revision')
    require(h[4] == expected_magic, 'header magic identity receipt binding')
    require(crc32c(b'\0'*4 + header[4:]) == h[0], 'header CRC')
    require(crc32c(b'\0'*4 + footer[4:]) == f[0], 'footer CRC')
    require(f[3] == 5 and f[4] == excerpt['table_size'] and f[4] <= 2**21, 'table bounds')
    require(h[7] - f[4] == 64 and h[8] == h[7] + 128, 'metadata offsets')
    require(f[-2] == 0 and f[-1] == file_size and h[8] + f[6] + f[7] == file_size, 'file bounds or chained format')
    table = bytearray(f[4])
    previous_end = 0
    for offset, text in excerpt['table_nonzero_runs']:
        data = bytes.fromhex(text)
        require(data and all(data) and offset >= previous_end and offset + len(data) <= len(table), 'sparse table bounds')
        table[offset:offset+len(data)] = data
        previous_end = offset + len(data)
    require(crc32c(table) == f[1], 'segment table CRC')
    segments = {}
    append_end = 0
    for i in range(f[3]):
        m = SEGMENT.unpack_from(table, i*32)
        require(f[3]*32 <= m[0] < len(table), 'name offset')
        end = table.find(b'\0', m[0])
        require(end != -1, 'unterminated segment name')
        name = table[m[0]:end].decode('ascii')
        require(name in NAMES and name not in segments, 'wrong segment type/name')
        start = h[8] + m[2]
        require(m[2] + m[3] + m[4] <= f[6] and start + m[3] + m[4] <= file_size, 'segment data/padding bounds')
        require(m[2] == append_end, 'segment append overlap/gap/order')
        append_end = m[2] + m[3] + m[4]
        segments[name] = (start, m[3])
    require(set(segments) == NAMES and append_end == f[6], 'segment set/aggregate extent')
    start, size = segments['flat.linear_meta']
    streamer = struct.unpack_from('<QQQII32s', linear)
    lh = LINEAR.unpack_from(linear, 64)
    # header_size = sizeof(LinearIndexHeader) + serialized IndexMeta size.
    require(size >= 128 and lh[0] + 64 == size and lh[0] - lh[7] == LINEAR.size, 'linear header bounds')
    require(lh[1] in {38, 64} and streamer[3] == 1, 'linear count/type')
    require(streamer[1] == f[9], 'timestamp disagreement')
    return {'h': h, 'f': f, 'count': lh[1], 'timestamp': f[9], 'regions': [(0, header), (64, bytes(table)), (h[7], footer), (start, linear)], 'allowed': [[h[7], h[7]+4], [h[7]+48, h[7]+56], [start+8, start+16]]}


def verify(evidence, check_logs=True, logs=None):
    source_binding.validate(json.loads((ROOT/'public-source-binding.json').read_text()))
    require(set(evidence) == {'seed_documents', 'runs', 'helper_source_sha256', 'vendor_commit', 'runtime_assets', 'sdk', 'native_sha256', 'controller_sha256', 'source_commit', 'format', 'helper_binary_sha256', 'source_files', 'storage_type'}, 'evidence schema/scope')
    require(evidence['sdk'] == 'zvec-rust/zvec-rust-sys 0.7.2', 'SDK binding')
    # These pins bind published private receipts; they do not remeasure assets.
    for field, expected in {'source_files': 'f94a18c3ea9c242c07f2e3ea4bd7909297443ce7ca83532bf466002efd31f5d7', 'runtime_assets': '9b943819d01aff2b735489cece89f74f1b6d7c455f40310f76e34ea43f93f440'}.items():
        actual = hashlib.sha256(json.dumps(evidence[field], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(actual == expected, 'source/runtime receipt binding')
    require(evidence['format'] == 'flat-readonly-metadata-evidence-v1', 'evidence format')
    require(evidence['storage_type'] == 'VectorFp16' and evidence['seed_documents'] == 230, 'declared storage/count')
    require(evidence['source_commit'] == 'af917642a1fba637b9a8fc02bffb2c966c50f7ec' and evidence['vendor_commit'] == '1ab7975dfc2d2160054bafff614831b7099cd930', 'source binding')
    for key in ['native_sha256', 'helper_binary_sha256', 'helper_source_sha256', 'controller_sha256']:
        require(re.fullmatch('[0-9a-f]{64}', evidence[key]) is not None, 'identity')
    require(evidence['native_sha256'] == '58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a' and evidence['helper_binary_sha256'] == 'f7799f4ae77d8b48a3f7e058b63f89ba7a10312475e5150aea28dd10df448228', 'native/helper binding')
    require(evidence['helper_source_sha256'] == 'a959d3e745eef37951b0a38bd6ddef9ad5e299bfec9950ee0e37936f83178c6b' and evidence['controller_sha256'] == '8479633fd34cd50de45987b3b9b3f6b108cbcb0e935c427276f07bb90590736b', 'helper/control binding')
    require([r['arm'] for r in evidence['runs']] == ['pure-open', 'vector'], 'arm set')
    for run in evidence['runs']:
        expected = ['initialized', 'options_readonly', 'opened']
        if run['arm'] == 'vector':
            expected += ['stats_schema', 'vector_prepared', 'vector_queried']
        expected += ['dropped']
        require(set(run) == {'arm', 'started_utc', 'mapping_permissions', 'stages', 'snapshot_observation_utc', 'hits', 'private_full_byte_verification', 'environment', 'first_changed_stage', 'files', 'live_count_asserted', 'options', 'process_exit_delta_file_count', 'phase_delta_file_counts', 'log_sha256'}, 'run schema/scope')
        require(run['private_full_byte_verification'] == {'all_other_bytes_equal': True, 'original_index_unchanged': True, 'added_paths': 0, 'removed_paths': 0, 'size_delta_bytes': 0, 'scope': 'private complete-file verification receipt; omitted payload equality cannot be reproduced from these public excerpts'}, 'private receipt scope')
        require(run['environment'] == {'platform': 'Linux x86_64', 'cpu_quota_cores': 4, 'cgroup_memory_limit_bytes': 17179869184, 'shared_machine': True, 'inference': 'none', 'postgres': 'none', 'temperature': 'no product cold/warm performance measurement'}, 'component environment/scope')
        start_expected = {'pure-open': '2026-10-01T01:02:05.865610+00:00', 'vector': '2026-10-01T01:36:43.225342+00:00'}
        require(run['started_utc'] == start_expected[run['arm']], 'run start receipt')
        observed_pins = {'pure-open': 'ed5867aaad9677e4b32723c5afee2a99a6ef7c57f9f9571e5fd7f7020d96ed90', 'vector': '5edc8dd2c057a79d45402de4d2b0749b5a10a8a288932c5c837c3dc40aad7f90'}
        require(set(run['snapshot_observation_utc']) == set(expected), 'observation stage set')
        started = datetime.datetime.fromisoformat(run['started_utc'])
        observed = [datetime.datetime.fromisoformat(run['snapshot_observation_utc'][stage]) for stage in expected]
        require(all(value.utcoffset() == datetime.timedelta(0) for value in [started, *observed]), 'UTC timezone')
        require(started <= observed[0] and all(a <= b for a,b in zip(observed,observed[1:])) and (observed[-1]-started).total_seconds() < 120, 'observation chronology/bounds')
        require(hashlib.sha256(json.dumps(run['snapshot_observation_utc'],sort_keys=True,separators=(',',':')).encode()).hexdigest() == observed_pins[run['arm']], 'observation receipt binding')
        require(run['stages'] == expected and run['first_changed_stage'] == 'dropped', 'phase attribution')
        require(run['phase_delta_file_counts'] == {s:4 if s == 'dropped' else 0 for s in expected} and run['process_exit_delta_file_count'] == 0, 'late phase fabrication')
        require(run['mapping_permissions'] == 'rw-s' and run['options'] == {'read_only':True,'enable_mmap':True,'max_buffer_size':67108864}, 'options/mapping')
        require(run['hits'] == (50 if run['arm'] == 'vector' else None), 'hit count')
        require(run['live_count_asserted'] == (230 if run['arm'] == 'vector' else None), 'live count')
        require([f['label'] for f in run['files']] == [f'embedding.index.{i}.proxima' for i in [2,4,6,8]], 'file set')
        counts = []
        for file in run['files']:
            require(set(file) == {'label', 'file_size_before', 'file_size_after', 'whole_file_sha256_before', 'whole_file_sha256_after', 'actual_changed_ranges', 'allowed_full_field_ranges', 'excerpts'}, 'file schema/scope')
            require(set(file['excerpts']) == {'before', 'after'}, 'excerpts schema/scope')
            recorded_hashes = (file['whole_file_sha256_before'], file['whole_file_sha256_after'])
            require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None for value in recorded_hashes), 'whole-file identity format')
            require(recorded_hashes == WHOLE_FILE_SHA256[run['arm']][file['label']], 'whole-file identity receipt binding')
            require(file['file_size_before'] == file['file_size_after'] == 5234688, 'file size')
            old = decode(file['excerpts']['before'], file['file_size_before'], HEADER_MAGIC[file['label']])
            new = decode(file['excerpts']['after'], file['file_size_after'], HEADER_MAGIC[file['label']])
            require(old['allowed'] == new['allowed'] == file['allowed_full_field_ranges'], 'allowed mask range')
            require(old['count'] == new['count'], 'count changed')
            counts.append(old['count'])
            ranges = []
            for (off, a), (new_off, b) in zip(old['regions'], new['regions']):
                require(off == new_off and len(a) == len(b), 'metadata relocation')
                start = None
                for i in range(len(a)+1):
                    differs = i < len(a) and a[i] != b[i]
                    if differs:
                        require(any(lo <= off+i < hi for lo, hi in old['allowed']), 'unallowed metadata byte')
                        if start is None: start = off+i
                    elif start is not None:
                        ranges.append([start, off+i]); start = None
            require(ranges == file['actual_changed_ranges'] == [[1052544,1052548],[1052592,1052594],[1056776,1056778]], 'changed ranges')
            require(old['timestamp'] == 1790801618 and new['timestamp'] == (1790816526 if run['arm'] == 'pure-open' else 1790818603), 'run timestamp')
        require(counts == [64,64,64,38] and sum(counts) == 230, 'total count')
        if check_logs:
            log = (ROOT/(run['arm']+'.log')).read_text() if logs is None else logs[run['arm']]
            require(hashlib.sha256(log.encode()).hexdigest() == run['log_sha256'], 'log hash')
            option_event = 'PROBE_OPTIONS read_only=true enable_mmap=true max_buffer_size=67108864'
            expected_events = ['PROBE_STAGE initialized', option_event, 'PROBE_STAGE options_readonly', 'PROBE_STAGE opened']
            if run['arm'] == 'vector':
                expected_events += ['PROBE_STAGE stats_schema', 'PROBE_STAGE vector_prepared', 'PROBE_HITS 50', 'PROBE_STAGE vector_queried']
            expected_events += ['PROBE_STAGE dropped']
            events = [line for line in log.splitlines() if line.startswith(('PROBE_STAGE ', 'PROBE_OPTIONS ', 'PROBE_HITS '))]
            require(events == expected_events, 'ordered probe events')
            result_lines = [line for line in log.splitlines() if 'test result:' in line]
            require(len(result_lines) == 1 and re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(?:\.[0-9]+)?s', result_lines[0]) is not None, 'single positive test completion')
            option_lines = [line for line in log.splitlines() if line.startswith('PROBE_OPTIONS ')]
            require(option_lines == ['PROBE_OPTIONS read_only=true enable_mmap=true max_buffer_size=67108864'], 'raw option getter')
            hit_lines = [line for line in log.splitlines() if line.startswith('PROBE_HITS ')]
            require(hit_lines == (['PROBE_HITS 50'] if run['arm'] == 'vector' else []), 'raw hit count/cardinality')
    return True


if __name__ == '__main__':
    verify(json.loads((ROOT/'evidence.json').read_text()))
    count = source_binding.check_git(json.loads((ROOT/'public-source-binding.json').read_text()))
    print('Both published metadata excerpts verified; omitted payload equality remains a private receipt.')
    print(f'{count} public production-input blobs verified with replacement refs disabled.' if count is not None else 'Public production-input Git objects unavailable: manifest receipt checked, source bytes not independently checked.')
