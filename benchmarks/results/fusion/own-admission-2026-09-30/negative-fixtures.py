#!/usr/bin/env python3
"""Corrupt temporary copies; verify semantic checks and optimization refusal."""
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
spec = importlib.util.spec_from_file_location('own_verify', root/'verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
repo = root.parents[3]
relative = root.relative_to(repo)
harness_relative = Path('benchmarks/harnesses/fusion-own-admission-2026-09-30')


def save(path, value):
    data = (json.dumps(value, ensure_ascii=False)+'\n').encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)


def change_rows(target, mutation):
    """Keep JSONL and raw log consistent so checks beyond hash/log parity run."""
    rows_path = target/'pooled.jsonl.gz'
    rows = [json.loads(line) for line in verifier.read(rows_path).decode().splitlines()]
    mutation(rows)
    rows_path.write_bytes(gzip.compress(('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows)+'\n').encode(), mtime=0))
    log_path = target/'pooled.log.gz'
    lines = verifier.read(log_path).decode().splitlines()
    at = 0
    for position, line in enumerate(lines):
        if 'POOL_ACCEPT_JSON ' in line:
            lines[position] = 'POOL_ACCEPT_JSON '+json.dumps(rows[at], ensure_ascii=False)
            at += 1
    log_path.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(), mtime=0))


def rows_mutation(fn):
    return lambda target: change_rows(target, fn)


def change_json(name, fn):
    def change(target):
        path = target/name
        value = verifier.load(path)
        fn(value)
        save(path, value)
    return change


def provider_change(target):
    path = target/'pooled.log.gz'
    text = verifier.read(path).decode().replace('"CPUExecutionProvider":295', '"CPUExecutionProvider":296')
    path.write_bytes(gzip.compress(text.encode(), mtime=0))


def remove_score(rows):
    hit = next(h for h in rows[0]['complete'] if any(w['kind'] == 'reranked' for w in h['why']))
    hit['why'] = [w for w in hit['why'] if w['kind'] != 'reranked']
    rows[0]['top10'] = rows[0]['complete'][:10]


def reintroduce_controller(target):
    path = target.parents[3]/harness_relative/'historical-run-own.py.in.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'controller')


fixtures = [
    ('historical derived identity drift', change_json('sanitization.json', lambda value: next(iter(value['corrected_derived_records']['original_records'].values())).update(original_sha256='0'*64)), 'historical derived metadata identities'),
    ('default certification drift', change_json('scope.json', lambda value: value.update(shipped_default_certification=True)), 'historical publication/default scope'),
    ('provenance certification drift', change_json('provenance.json', lambda value: value['publication_scope'].update(validated_quality_benchmark=True)), 'provenance publication/default scope'),
    ('provenance description drift', change_json('provenance.json', lambda value: value.update(scope='validated default product benchmark')), 'provenance diagnostic scope'),
    ('historical settings drift', change_json('provenance.json', lambda value: value.update(configuration_scope='complete shipped defaults')), 'provenance historical configuration scope'),
    ('summary certification drift', change_json('summary.json.gz', lambda value: value['publication_scope'].update(public_execution_reproducible=True)), 'summary publication/default scope'),
    ('immutable Git reference drift', change_json('git-inputs.json', lambda value: value.update(revision='0'*40)), 'pinned Git input references'),
    ('public corpus drift', change_json('inputs/corpus/queries.json', lambda value: value[0].update(query='invented')), 'pinned Git fixture bytes'),
    ('family design drift', change_json('family-input.json', lambda value: value.update(design='independent row signs')), 'declared family design'),
    ('family input drift', change_json('family-input.json', lambda value: value['cells'][0]['differences'].__setitem__(0, 1)), 'native 157-ID family input alignment'),
    ('family query order drift', change_json('family-input.json', lambda value: value['query_ids'].__setitem__(1, 0)), 'family query-ID order'),
    ('removed inventory drift', change_json('removed-harness-inventory.json', lambda value: value['files'].pop()), 'removed harness inventory'),
    ('controller reintroduced', reintroduce_controller, 'removed measurement harness reintroduced'),
    ('paired ID duplication', rows_mutation(lambda rows: rows[1].update(query_id=0)), 'paired query IDs'),
    ('native gold drift', rows_mutation(lambda rows: rows[0]['relevant'].append('invented_gold')), 'native fixture pairing'),
    ('offered budget drift', rows_mutation(lambda rows: rows[0].update(offered_candidates=31)), 'native offered30–31 counts'),
    ('product prefix drift', rows_mutation(lambda rows: rows[0]['top10'][0].update(fusion_score=99)), 'actual prefix identity'),
    ('native selection drift', rows_mutation(lambda rows: rows[0]['selected_fused_positions'].__setitem__(-1, 31)), 'native main+graph selection'),
    ('available score missing', rows_mutation(remove_score), 'offered/available Why identities'),
    ('full fused drift', rows_mutation(lambda rows: rows[0]['fused'][0].update(fusion_score=99)), 'all fused rows identical'),
    ('raw provider node drift', provider_change, 'raw provider assignment'),
    ('flat buffer mislabel', change_json('seed-index-identity.json', lambda value: value['before'].update(vector_completeness=1.0)), 'raw index before'),
    ('percentage drift', change_json('metrics.json', lambda value: value['cross_lingual/ndcg10'].update(percentage_change=100)), 'metric delta/percentage'),
    ('statistical p drift', change_json('native-statistics.json', lambda value: value['cells'][0].update(raw_p=0.04)), 'native Monte Carlo raw p'),
    ('statistical mean drift', change_json('native-statistics.json', lambda value: value['cells'][0].update(delta=0.04)), 'native paired delta'),
    ('group count drift', change_json('summary.json.gz', lambda value: value['groups']['cross_lingual'].update(queries=42)), 'group size'),
    ('candidate case drift', change_json('changed-ndcg-cases.json.gz', lambda value: value[0]['relevant_details']['baseline'][0].update(rank=99)), 'changed-case relevant scores'),
]

with tempfile.TemporaryDirectory(prefix='pamin-own-evidence-') as temporary:
    copy_repo = Path(temporary)
    copy_root = copy_repo/relative
    copy_harness = copy_repo/harness_relative
    for label, mutate, expected in fixtures:
        if copy_harness.exists():
            shutil.rmtree(copy_harness)
        if copy_root.exists():
            shutil.rmtree(copy_root)
        shutil.copytree(root, copy_root)
        mutate(copy_root)
        try:
            verifier.verify(copy_root, check_hashes=False)
        except ValueError as error:
            verifier.require(expected in str(error), label+' failed for unexpected reason: '+str(error))
        else:
            raise SystemExit('Negative fixture accepted: '+label)
    shutil.rmtree(copy_root)
    shutil.copytree(root, copy_root)
    path = copy_root/'pooled.log.gz'
    path.write_bytes(path.read_bytes()+b'tampered')
    try:
        verifier.verify(copy_root)
    except ValueError as error:
        verifier.require('archive hash' in str(error), 'outer archive hash detection')
    else:
        raise SystemExit('Archive hash corruption accepted')

for program in ('verify.py', 'negative-fixtures.py'):
    for optimized, environment in [(True, dict(os.environ)), (False, dict(os.environ, PYTHONOPTIMIZE='1'))]:
        command = [sys.executable]+(['-O'] if optimized else [])+[str(root/program)]
        result = subprocess.run(command, capture_output=True, text=True, env=environment)
        verifier.require(result.returncode != 0 and ('refuses' in result.stderr), program+' optimization refusal')
print(f'Rejected{len(fixtures)}semantic corruptions,1archive hash corruption and4optimized-Python launches; temporary copies only.')
