#!/usr/bin/env python3
"""Read-only metadata-component probe; prints new observations, never product work.

Requires the before commit object already available locally. No fetch/download,
file edits, SDK, database, model, native helper or Cargo invocation is provided.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import sys

# Keep the probe read-only even when invoked without Python -B.
sys.dont_write_bytecode = True
import verify

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='opt in to three metadata-only observations')
    parser.add_argument('--repository', type=Path, default=ROOT.parents[3])
    args = parser.parse_args()
    if not args.execute:
        print('Plan: local before CRC source, cold current verifier, then warm current verifier; no product execution.')
        return
    cost = json.loads((ROOT/'verifier-cost.json').read_text())
    for name, expected in cost['after_inputs_sha256'].items():
        require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, 'recorded input/source identity differs: ' + name)
    git = shutil.which('git')
    require(git is not None, 'Git required only to read locally available before source')
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env.update(GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1',
               GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_COUNT='0')
    path = 'benchmarks/results/index/native-readonly-metadata-2026-10-01/verify.py'
    result = subprocess.run([git, '--no-replace-objects', '-c', 'core.fsmonitor=false',
                             '-C', str(args.repository), 'show', cost['before_crc_revision'] + ':' + path],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, 'before source unavailable locally; no fetch attempted')
    require(hashlib.sha256(result.stdout).hexdigest() == cost['before_verify_sha256'], 'before source identity differs')
    before = {'__file__': str(ROOT/'verify.py'), '__name__': 'before_metadata_component'}
    exec(compile(result.stdout, str(ROOT/'verify.py'), 'exec'), before)
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
