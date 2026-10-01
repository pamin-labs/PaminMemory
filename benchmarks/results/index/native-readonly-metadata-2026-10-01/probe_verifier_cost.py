#!/usr/bin/env python3
"""Read-only metadata-component probe; prints new observations, never product work.

Uses the hash-bound before source retained in this directory. No fetch/download,
file edits, SDK, database, model, native helper or Cargo invocation is provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import sys

# Keep the probe read-only even when invoked without Python -B.
sys.dont_write_bytecode = True
import verify

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_before(cost):
    source = (ROOT/'before-verify.py.in').read_bytes()
    require(hashlib.sha256(source).hexdigest() == cost['before_verify_sha256'], 'before source identity differs')
    before = {'__file__': str(ROOT/'verify.py'), '__name__': 'before_metadata_component'}
    exec(compile(source, str(ROOT/'before-verify.py.in'), 'exec'), before)
    return before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='opt in to three metadata-only observations')
    args = parser.parse_args()
    if not args.execute:
        print('Plan: retained before CRC source, cold current verifier, then warm current verifier; no product execution.')
        return
    cost = json.loads((ROOT/'verifier-cost.json').read_text())
    for name, expected in cost['probe_inputs_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'recorded input/source identity differs: ' + name)
    before = load_before(cost)
    evidence = verify.source_binding.load_json(ROOT/'evidence.json')
    tables = set()
    excerpts = 0
    for run in evidence['runs']:
        for file in run['files']:
            for excerpt in file['excerpts'].values():
                data = bytearray(excerpt['table_size'])
                for offset, text in excerpt['table_nonzero_runs']:
                    block = bytes.fromhex(text)
                    data[offset:offset+len(block)] = block
                tables.add(bytes(data)); excerpts += 1
    require(excerpts == 16 and len(tables) == 1, 'retained table inventory differs')
    table = next(iter(tables))
    require(len(table) == cost['table_identity']['bytes_each']
            and hashlib.sha256(table).hexdigest() == cost['table_identity']['sha256'], 'retained table bytes differ')
    del tables, table, data
    observations = {}
    verify._crc32c_cached.cache_clear()
    for label, function in [('before', before['verify']), ('after_cold', verify.verify), ('after_warm', verify.verify)]:
        stats_before = verify._crc32c_cached.cache_info()
        wall = time.perf_counter(); cpu = time.process_time()
        require(function(evidence), 'metadata validation did not complete')
        observed = {'wall_seconds': time.perf_counter()-wall, 'process_cpu_seconds': time.process_time()-cpu}
        stats_after = verify._crc32c_cached.cache_info()
        scans = 16 if label == 'before' else stats_after.misses-stats_before.misses
        observed['scalar_table_bytes'] = scans * cost['table_identity']['bytes_each']
        observed['scalar_table_scans'] = scans
        if label != 'before':
            observed['table_cache_hits'] = stats_after.hits-stats_before.hits
        observations[label] = observed
    print(json.dumps({'new_observations': observations, 'scope': cost['scope'],
                      'scan_scope': cost['scan_scope'], 'memory': cost['memory']}, indent=2))


if __name__ == '__main__':
    main()
