#!/usr/bin/env python3
"""Mutate temporary archive copies, refresh digests, and require semantic rejection."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def refresh(root):
    manifest = root / 'provenance.json'
    p = json.loads(manifest.read_text())
    for name in p['archive_files']:
        p['archive_files'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name, record in p['redactions'].items():
        record['published_sha256'] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest.write_text(json.dumps(p, indent=2) + '\n')


def mutate_json(path, change, jsonl=False):
    if jsonl:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        row = json.loads(path.read_text())
        change(row)
        path.write_text(json.dumps(row, indent=2) + '\n')


def run_case(label, mutate=None, flags=(), refresh_hashes=True):
    with tempfile.TemporaryDirectory(prefix='graph-archive-negative-') as temp:
        root = Path(temp) / 'archive'
        shutil.copytree(ROOT, root)
        if mutate:
            mutate(root)
        if refresh_hashes:
            refresh(root)
        result = subprocess.run([sys.executable, *flags, str(root / 'verify.py')], capture_output=True, text=True)
        if result.returncode == 0 or 'PASS:' in result.stdout:
            raise SystemExit(f'FAIL: {label} was accepted')
        print(f'PASS: rejected {label}')


if __name__ == '__main__':
    success = subprocess.run([sys.executable, str(ROOT / 'verify.py')], capture_output=True, text=True)
    if success.returncode:
        raise SystemExit(success.stderr)
    run_case('Python -O', flags=('-O',))
    run_case('raw provider mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[2]['fields'].update(assigned_nodes={'CUDAExecutionProvider':1023}), True))
    run_case('raw runtime mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[0]['fields'].update(runtime_info='ORT Build Info: changed'), True))
    run_case('raw mapped-path mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[0].update(runtime_maps=['${ORT_LIB}/other.so']), True))
    run_case('provenance provider mutation with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: next(iter(p['arms'][0]['inference']['selected_graphs'].values())).update(assigned_nodes={'CPUExecutionProvider':1022})))
    run_case('comparison score mutation with refreshed hashes', lambda r: mutate_json(r/'comparison.json', lambda c: c['invariants'][0].update(scored=.7)))
    run_case('test-log marker mutation with refreshed hashes', lambda r: (r/'baseline.log').write_text((r/'baseline.log').read_text().replace('fixture ... ok','fixture ... FAILED')))
    run_case('displayed table mutation with refreshed hashes', lambda r: (r/'README.md').write_text((r/'README.md').read_text().replace('+139.999996%', '+141.000000%')))
    run_case('missing redaction record with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['redactions'].pop('baseline.log')))
    run_case('missing asset pin with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0]['asset_hashes_before'].pop('${ZVEC_LIB}/libzvec_c_api.so')))
    run_case('malformed compiled source map with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0].update(source_hashes={'bogus':'not-a-digest'})))
    run_case('changed compiled source digest with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0]['source_hashes'].update({'Cargo.lock':'0'*64})))
    run_case('missing vector rank50 with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row.update(non_graph=[h for h in row['non_graph'] if not any(x['channel']=='vector' and x['rank']==50 for x in h['ranks'])])))
    run_case('duplicate channel rank with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: next(x for h in row['non_graph'] for x in h['ranks'] if x['channel']=='vector' and x['rank']==50).update(rank=49)))
    run_case('changed controlled edge endpoint with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row['known_edges'][0].update({'from':'arbitrarytopic'})))
    run_case('changed controlled edge confidence with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row['known_edges'][0].update(confidence=.9)))
    run_case('extra file with refreshed known hashes', lambda r: (r/'extra.txt').write_text('extra\n'))
    run_case('missing required file', lambda r: (r/'baseline.usage.json').unlink(), refresh_hashes=False)
