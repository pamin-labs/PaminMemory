#!/usr/bin/env python3
"""Exercise evidence guards on temporary copies; never invoke inference."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Negative verification refuses -O/PYTHONOPTIMIZE.')
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('component_verify', root/'verify.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def mutate_json(name, fn):
    def change(target):
        path = target/name
        value = v.load(path)
        fn(value)
        save(path, value)
    return change


def mutate_raw(fn):
    def change(target):
        data = v.rows(target/'fixed-raw.jsonl.gz')
        fn(data)
        (target/'fixed-raw.jsonl.gz').write_bytes(gzip.compress(('\n'.join(json.dumps(row, ensure_ascii=False) for row in data)+'\n').encode(), mtime=0))
        lines = v.payload(target/'fixed-raw.log.gz').decode().splitlines()
        at = 0
        for position, line in enumerate(lines):
            if 'CACHE_ACCEPT_JSON ' in line:
                lines[position] = 'CACHE_ACCEPT_JSON '+json.dumps(data[at], ensure_ascii=False)
                at += 1
        (target/'fixed-raw.log.gz').write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(), mtime=0))
        component = [row for row in data if row['kind'] in ('rank', 'history_comparison', 'complete')]
        (target/'fixed-component.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in component))
    return change


def rank(rows, phase):
    return next(r for r in rows if r['kind'] == 'rank' and r['phase'] == phase)


fixtures = [
    ('missing required manifest member', mutate_json('manifest.json', lambda j: j['files'].pop('original-oracle.json')), 'manifest includes every required'),
    ('f32 logit bit corruption', mutate_raw(lambda rows: rank(rows, 'fresh')['ranked'][0].update(bits=0)), 'f32 score/bit identity'),
    ('hot forward work corruption', mutate_raw(lambda rows: rank(rows, 'hot').update(batches=1)), 'hot zero new forward work'),
    ('history work corruption', mutate_raw(lambda rows: rank(rows, 'baseline->pooled').update(tokens=0)), 'fresh planned work'),
    ('plan physical shape corruption', mutate_raw(lambda rows: next(r for r in rows if r['kind'] == 'plan')['batches'][0].update(physical_shape=[4, 128])), 'actual logical/physical shape'),
    ('launch tuning leak', mutate_json('fixed-launch.json', lambda j: j['environment'].update(PAMIN_RERANK_BATCH='1')), 'launch safe config'),
    ('executed external weight drift', mutate_json('provenance-after.json', lambda j: j['assets'][v.GRAPH].update(sha256='0'*64)), 'all executed assets unchanged'),
    ('build binding corruption', mutate_json('fixed-build-binding.json.gz', lambda j: j['pamin-index'].update(sha256='0'*64)), 'measurement/build binding'),
    ('comparison omission', mutate_json('complete-same-list-comparisons.json', lambda j: j.pop()), 'all240 actual joined'),
    ('summary false gain', mutate_json('summary.json', lambda j: j['cases'][0].update(changed_same_list_score_bits=0)), 'recomputed summary'),
    ('percentage corruption', mutate_json('metrics.json', lambda j: next(iter(j.values())).update(percentage_change=1)), 'before/after/delta/percentage'),
    ('released source guard erased', mutate_json('original/qualified-postvalidation.json.gz', lambda j: j.update(original_whole_asset_guard='PASSED')), 'original asset guard remains qualified'),
]
with tempfile.TemporaryDirectory(prefix='pamin-cache-public-') as temporary:
    target = Path(temporary)/'evidence'
    for label, mutate, expected in fixtures:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(root, target)
        mutate(target)
        try:
            v.verify(target, check_hashes=False)
        except ValueError as error:
            v.require(expected in str(error), label+' failed for unexpected reason '+str(error))
        else:
            raise SystemExit('Corrupted evidence accepted: '+label)
    shutil.rmtree(target)
    shutil.copytree(root, target)
    path = target/'fixed-raw.jsonl.gz'
    path.write_bytes(path.read_bytes()+b'changed')
    try:
        v.verify(target)
    except ValueError as error:
        v.require('archive hash' in str(error), 'outer archive hash guard')
    else:
        raise SystemExit('Archive hash corruption accepted')
for program in ('verify.py', 'negative-fixtures.py'):
    for optimized, env in ((True, dict(os.environ)), (False, dict(os.environ, PYTHONOPTIMIZE='1'))):
        command = [sys.executable]+(['-O'] if optimized else [])+[str(root/program)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        v.require(result.returncode != 0 and 'refuses' in result.stderr, 'optimization refusal '+program)
print('Rejected 12 semantic corruptions, 1 archive corruption and 4 optimized-Python launches; temporary copies only.')
