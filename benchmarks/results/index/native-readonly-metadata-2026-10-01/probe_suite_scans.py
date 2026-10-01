#!/usr/bin/env python3
"""Regenerate metadata invariant-suite scan counts; never product execution.

The tests use small disposable mock/Git fixture directories. Published files
are read only. Counts exclude byte-key hashing/equality and other memory traffic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import verify

ROOT = Path(__file__).resolve().parent


def collect(suite, table_sha256):
    verify._crc32c_cached.cache_clear()
    calls = []
    scalar = verify._crc32c_scalar
    def counted(data):
        if len(data) > 256:
            calls.append((len(data), hashlib.sha256(data).hexdigest()))
        return scalar(data)
    with patch.object(verify, '_crc32c_scalar', side_effect=counted):
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    if not result.wasSuccessful():
        raise ValueError('invariant/source suite failed; no successful scan audit')
    return {'tests': result.testsRun, 'large_scalar_calls': len(calls),
            'large_scalar_bytes': sum(size for size, _ in calls),
            'distinct_large_scalar_contents': len({sha for _, sha in calls}),
            'original_table_scalar_scans': sum(sha == table_sha256 for _, sha in calls)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('Plan: test_verify and test_source_binding with scalar CRC call/byte/content counters; no cost timing or product execution.')
        return
    cost = json.loads((ROOT/'verifier-cost.json').read_text())
    for name, expected in cost['suite_probe_inputs_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('suite source/input identity differs: ' + name)
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_verify', 'test_source_binding'])
    audit = collect(suite, cost['table_identity']['sha256'])
    print(json.dumps({'regenerated_scan_audit': audit, 'scope': cost['full_suite_scan_audit']['scope'],
                      'scan_scope': cost['scan_scope'], 'cost_measurements_rerun': False}, indent=2))


if __name__ == '__main__':
    main()
