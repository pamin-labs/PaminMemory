#!/usr/bin/env python3
"""Standalone, read-only component evidence verification. No inference/builds."""
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import statistics
import sys

if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Verification refuses -O/PYTHONOPTIMIZE.')

ARMS = ('refactor', 'fixed')
QUERIES = (80, 101)
LISTS = ('baseline', 'pooled')
GRAPH = '${MODEL_CACHE}/prepared/eecdcf109c0c08402aa8f893fc25d2d4/attention.onnx'
GRAPH_SHA = '24cc5ad23811a65c6a4648d8be0b79a7ad67a180e2db234b08c899c882ac5164'
ORT = '${ORT_RUNTIME}/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0'
NATIVE = '${CARGO_HOME}/lib/pamin/libzvec_c_api.so'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def payload(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def load(path):
    return json.loads(payload(path))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(path):
    return [json.loads(line) for line in payload(path).decode().splitlines()]


def marker(log, name):
    return [json.loads(line.split(name+' ', 1)[1]) for line in log.splitlines() if name+' ' in line]


def bits(value):
    require(type(value) in (float, int) and math.isfinite(value), 'finite recorded logit')
    return struct.unpack('<I', struct.pack('<f', value))[0]


def required_files():
    files = {'README.md', 'manifest.json', 'provenance.json', 'public-fixture.json',
             'summary.json', 'complete-same-list-comparisons.json', 'refactor-component.jsonl',
             'fixed-component.jsonl', 'original-oracle.json', 'exact-fixture.json', 'sanitization.json',
             'provenance-before.json.gz', 'provenance-after.json', 'metrics.json', 'verify.py',
             'negative-fixtures.py'}
    for arm in ARMS:
        files.update(arm+'-'+name for name in ('launch.json', 'provider.json', 'raw.log.gz', 'raw.jsonl.gz', 'build-binding.json.gz', 'clean.log.gz'))
        for crate in ('index', 'engine'):
            files.update(f'{arm}-pamin-{crate}-build.{ext}.gz' for ext in ('log', 'jsonl'))
        files.update(f'sources/{arm}-{name}' for name in ('scratch-reranking.rs.in.gz', 'product-reranking.rs.in.gz',
                    'encoder.rs.in.gz', 'tokenizer.rs.in.gz', 'prepared.rs.in.gz', 'Cargo.lock.gz'))
    files.update('sources/'+name+'.in.gz' for name in ('run.py', 'prepare-build.py', 'causal.rs.append'))
    files.update('original/'+name+'.gz' for name in ('raw.jsonl', 'reranker.log', 'provenance-before.json', 'qualified-postvalidation.json', 'asset-guard-failure.json'))
    files.update('original/'+name for name in ('reranker-launch.json', 'reranker-provider.json'))
    return files


def provider(log, expected, assets):
    decoder = json.JSONDecoder()
    records = []
    for line in log.splitlines():
        if 'ONNX graph execution-provider assignment' not in line:
            continue
        graph, _ = decoder.raw_decode(line.split('model_graph=', 1)[1])
        nodes, end = decoder.raw_decode(line.split('assigned_nodes=', 1)[1])
        require(not line.split('assigned_nodes=', 1)[1][end:].strip(), 'provider trailing data')
        require(graph == GRAPH and nodes == {'CPUExecutionProvider': 295}, 'actual CPU provider graph/nodes')
        records.append({'graph': graph, 'sha256': assets[graph]['sha256'], 'nodes': nodes})
    require(len(records) == 1 and records == expected and records[0]['sha256'] == GRAPH_SHA, 'provider identity binding')


def ranked_identity(ranked):
    return [(row['fused_position'], row['bits']) for row in ranked]


def check_rank(record, ids):
    ranked = record['ranked']
    require(record['offered'] == 30 and len(ranked) == 30 and {r['fused_position'] for r in ranked} == set(ids), 'all30 offered output scores')
    require([r['raw_model_rank'] for r in ranked] == list(range(1, 31)), 'complete raw ranks')
    for r in ranked:
        require(bits(r['score']) == r['bits'], 'f32 score/bit identity')
    positions = {position: at for at, position in enumerate(ids)}
    expected = sorted(ranked, key=lambda r: (-r['score'], positions[r['fused_position']]))
    require(ranked == expected, 'raw score sorting and original-position ties')
    require(record['cache_hits']+record['cache_misses'] == record['offered'], 'offered hit/miss accounting')
    require(record['cache_misses'] == record['fresh_scored'], 'successful fresh work accounting')
    for key in ('tokens', 'padded_tokens', 'batches', 'encode_us', 'forward_us', 'wall_us', 'remembered'):
        require(type(record[key]) is int and record[key] >= 0, 'nonnegative work '+key)


def batches(plan):
    return plan['batches']


def totals(groups):
    return {'fresh_scored': sum(len(g['lengths']) for g in groups),
            'tokens': sum(sum(g['lengths']) for g in groups),
            'padded_tokens': sum(g['physical_shape'][0]*g['physical_shape'][1] for g in groups),
            'batches': len(groups)}


def signature(group, case):
    documents = {d['fused_position']: d['shown'] for d in case['documents']}
    return (case['query'], tuple(documents[at] for at in group['fused_positions']),
            len(group['lengths']), max(group['lengths']), tuple(group['physical_shape']))


def missing_pair_groups(plan, first_ids):
    """Predecessor plans just missed pairs; reconstruct from observed lengths."""
    length_map = {at: length for batch in batches(plan) for at, length in zip(batch['fused_positions'], batch['lengths'])}
    offered = [p['fused_position'] for p in plan['pairs']]
    missed = [at for at in offered if at not in first_ids]
    missed.sort(key=lambda at: (length_map[at], offered.index(at)))
    groups = []
    for at in missed:
        length = length_map[at]
        if groups and len(groups[-1]['lengths']) < 4 and (len(groups[-1]['lengths'])+1)*length <= 512:
            groups[-1]['fused_positions'].append(at)
            groups[-1]['lengths'].append(length)
            groups[-1]['physical_shape'] = [len(groups[-1]['lengths']), length]
        else:
            groups.append({'fused_positions': [at], 'lengths': [length], 'physical_shape': [1, length]})
    return groups


def verify(root, check_hashes=True):
    manifest = load(root/'manifest.json')
    require(set(manifest['files']) == required_files()-{'manifest.json'}, 'manifest includes every required member')
    if check_hashes:
        for relative, digest in manifest['files'].items():
            path = root/relative
            require(path.resolve().is_relative_to(root.resolve()), 'manifest traversal')
            require(sha(path.read_bytes()) == digest, 'archive hash '+relative)
        sanitation = load(root/'sanitization.json')
        for relative, record in sanitation['files'].items():
            path = root/relative
            require(sha(path.read_bytes()) == record['archive_sha256'], 'sanitized archive hash')
            data = payload(path)
            require(sha(data) == record['sanitized_payload_sha256'], 'sanitized payload hash')
            original = data.decode()
            for replacement in sanitation['prefixes']:
                original = original.replace(replacement['replacement'], replacement['original'])
            require(sha(original.encode()) == record['original_sha256'], 'path-only original reconstruction')
    before = load(root/'provenance-before.json.gz')
    after = load(root/'provenance-after.json')
    require(before['assets'] == after['assets'] and after['executed_assets_unchanged'], 'all executed assets unchanged')
    require(after['source_export_still_absent'] and not after['pg_opened'] and after['source_binary_unchanged'], 'run scope/unchanged sources')
    assets = before['assets']
    require(assets[GRAPH]['sha256'] == GRAPH_SHA and assets[ORT]['sha256'] == '1461ef7cc3d9e49982591721683cc3e3a55580aeca9a5254e7aac47b75ee4bab', 'prepared graph/runtime identity')
    require(assets[NATIVE]['sha256'] == '58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a', 'native runtime identity')
    require('absent before run' in before['source_export']['status'] and before['source_export']['path'] not in assets, 'released source absent before/after')
    fixture = load(root/'exact-fixture.json')
    require(sha(payload(root/'exact-fixture.json')) == before['fixture_sha256'], 'exact measured fixture identity')
    cleaned = json.loads(json.dumps({'schema': fixture['schema'], 'cases': fixture['cases']}))
    for case in cleaned['cases']:
        for document in case['documents']:
            document.pop('topic_id')
    require(cleaned == load(root/'public-fixture.json'), 'public fixture derives from exact input')
    cases = {case['query_id']: case for case in fixture['cases']}
    require(set(cases) == set(QUERIES), 'two public queries only')
    oracle = load(root/'original-oracle.json')
    require(set(oracle) == {'80', '101'} and sha(payload(root/'original-oracle.json')) == before['oracle_sha256'], 'original strict oracle identity')
    original_raw = rows(root/'original/raw.jsonl.gz')
    original_log = payload(root/'original/reranker.log.gz').decode()
    require(marker(original_log, 'BATCH_COMPOSITION_JSON') == original_raw, 'original raw-log rows')
    original_provenance = load(root/'original/provenance-before.json.gz')
    qualification = load(root/'original/qualified-postvalidation.json.gz')
    require(qualification['original_whole_asset_guard'].startswith('FAILED') and qualification['executed_graph_weights_tokenizers_runtime_unchanged'], 'original asset guard remains qualified')
    original_provider = [{'graph': r['model_graph'], 'nodes': r['assigned_nodes'], 'sha256': r['sha256']}
                         for r in load(root/'original/reranker-provider.json')]
    provider(original_log, original_provider, original_provenance['assets'])
    require(next(r for r in original_raw if r['kind'] == 'input_identity')['fixture'] == fixture,
            'original oracle exact same public fixture')
    for asset in assets:
        if asset.endswith('.source'):
            require(qualification['cleanup_policy']['source_record_sha256'] == assets[asset]['sha256'],
                    'original prepared source identity marker')
        else:
            require(qualification['assets_after'].get(asset) == assets[asset], 'oracle executed assets match new pair')
    sanitation = load(root/'sanitization.json')
    require(sanitation['files']['sources/run.py.in.gz']['original_sha256'] == before['runner_sha256'],
            'prospective runner source identity')
    public_provenance = load(root/'provenance.json')
    public_original = [r for r in original_raw if r['kind'] == 'public_rank']
    require(len(public_original) == 16, '16 original fresh ranks')
    for row in public_original:
        require(row['ranked'] == oracle[str(row['query_id'])][row['arm']], 'original fresh oracle bit/order parity')
        require(row['offered'] == row['fresh_scored_pairs'] == 30, 'original fresh cache reset')
        for score in row['ranked']:
            require(bits(score['logit']) == score['bits'], 'original f32 bits')
    outputs = {}
    raw_variants = {}
    plans_by_arm = {}
    for arm in ARMS:
        binding = load(root/(arm+'-build-binding.json.gz'))
        require(binding == next(b for b in before['variants'] if b['arm'] == arm), 'measurement/build binding')
        require(binding['fresh_path_crates'], 'four path crates fresh')
        require(binding['revision'] == public_provenance[arm]['measured_commit'], 'measured revision summary')
        require(not any(k.startswith('PAMIN_') for k in binding['environment']), 'clean build tuning environment')
        require(binding['commands'][0][2:] == ['clean', '--release', '-p', 'pamin-core', '-p', 'pamin-store', '-p', 'pamin-index', '-p', 'pamin-engine'], 'clean all path crates before build')
        require(all(command[:2] == ['${CARGO_HOME}/bin/cargo', '+1.98.1'] for command in binding['commands']), 'pinned build toolchain')
        source_map = binding['source_files']
        require(sha(json.dumps(source_map, sort_keys=True).encode()) == binding['source_sha256'], 'complete source inventory digest')
        require(sha(payload(root/f'sources/{arm}-scratch-reranking.rs.in.gz')) == binding['scratch_reranking_sha256'] == source_map['crates/pamin-index/src/reranking.rs'], 'measured scratch source hash')
        require(sha(payload(root/f'sources/{arm}-product-reranking.rs.in.gz')) == binding['product_reranking_sha256'], 'product source hash')
        for name in ('encoder.rs', 'tokenizer.rs', 'prepared.rs'):
            require(sha(payload(root/f'sources/{arm}-{name}.in.gz')) == source_map['crates/pamin-index/src/'+name], 'measured model source '+name)
        require(sha(payload(root/f'sources/{arm}-Cargo.lock.gz')) == source_map['Cargo.lock'], 'measured lockfile')
        appendix = payload(root/'sources/causal.rs.append.in.gz').decode()
        appendix = appendix.replace('const EXPECT_BATCH_CACHE: bool = false;', 'const EXPECT_BATCH_CACHE: bool = '+str(arm == 'fixed').lower()+';')
        require(payload(root/f'sources/{arm}-product-reranking.rs.in.gz').decode()+'\n'+appendix == payload(root/f'sources/{arm}-scratch-reranking.rs.in.gz').decode(), 'only measured harness append')
        compiler = []
        for crate in ('index', 'engine'):
            messages = rows(root/f'{arm}-pamin-{crate}-build.jsonl.gz')
            require(any(m['reason'] == 'build-finished' and m['success'] for m in messages), 'successful compiler output')
            record = binding['pamin-'+crate]
            require({'sha256': record['sha256'], 'bytes': record['bytes']} == public_provenance[arm]['frozen_binaries'][crate], 'public frozen binary identity')
            artifact = record['compiler_artifact']
            require(artifact in messages and artifact['fresh'] is False and artifact['profile']['test'], 'actual fresh compiler binary artifact')
            require(artifact['manifest_path'].startswith(binding['source']+'/crates/pamin-'+crate+'/'), 'build source path')
            compiler.extend(m for m in messages if m['reason'] == 'compiler-artifact')
        for crate in ('core', 'store', 'index', 'engine'):
            require(any(m['target']['name'] == 'pamin_'+crate and not m['fresh'] and m['manifest_path'].startswith(binding['source']+'/') for m in compiler), 'fresh path dependency '+crate)
        launch = load(root/(arm+'-launch.json'))
        require(launch['binary_sha256'] == binding['pamin-index']['sha256'] and launch['command'][0] == binding['pamin-index']['binary'], 'actual launched frozen binary')
        require(launch['command'][1:] == ['--ignored', '--exact', 'reranking::scratch_cache_acceptance::causal', '--nocapture', '--test-threads=1'], 'exact component target')
        require(launch['assets_unchanged'] and launch['environment'] == before['environment'], 'launch safe config')
        environment = launch['environment']
        require({k: v for k, v in environment.items() if k.startswith('PAMIN_')} == {'PAMIN_DEVICE': 'cpu'}, 'no inherited PAMIN tuning')
        require(set(environment) <= {'HOME', 'USER', 'PATH', 'LANG', 'LC_ALL', 'TZ', 'TMPDIR', 'RUSTUP_HOME', 'CARGO_HOME', 'ORT_LIB_LOCATION', 'ORT_PREFER_DYNAMIC_LINK', 'CARGO_TARGET_DIR', 'LD_LIBRARY_PATH', 'PAMIN_DEVICE', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'CACHE_ACCEPT_INPUT', 'CACHE_ACCEPT_ORACLE', 'CACHE_ACCEPT_MODEL_CACHE'}, 'safe launch allowlist')
        raw = rows(root/(arm+'-raw.jsonl.gz'))
        log = payload(root/(arm+'-raw.log.gz')).decode()
        require(raw == marker(log, 'CACHE_ACCEPT_JSON') and len(raw) == 43, '43 complete raw records per variant')
        require('test result: ok. 1 passed; 0 failed;' in log and raw[-1]['kind'] == 'complete', 'successful component execution')
        provider(log, load(root/(arm+'-provider.json')), assets)
        loaded = next(row for row in raw if row['kind'] == 'loaded')
        require(loaded['maximum_tokens'] == 256 and any(ORT in line for line in loaded['loaded_libraries']) and any(NATIVE in line for line in loaded['loaded_libraries']), 'actual loaded runtime/libraries')
        require('da9b5e364c' in loaded['ort_build_info'], 'recorded ORT build')
        require(next(r for r in raw if r['kind'] == 'input')['expected_batch_cache'] == (arm == 'fixed'), 'measured cache arm')
        require(next(r for r in raw if r['kind'] == 'input')['fixture_sha256'] == before['fixture_sha256'], 'raw input identity')
        component = [r for r in raw if r['kind'] in ('rank', 'history_comparison', 'complete')]
        require(component == rows(root/(arm+'-component.jsonl')), 'component derives from complete raw records')
        calls = [r for r in raw if r['kind'] == 'rank']
        require(len(calls) == 24 and Counter(r['phase'] for r in calls) == {'fresh': 8, 'hot': 8, 'baseline->pooled': 4, 'pooled->baseline': 4}, 'all24 rank calls per variant')
        plans = [r for r in raw if r['kind'] == 'plan']
        require(len(plans) == 8, 'all fresh plans')
        by_plan = {(p['query_id'], p['arm']): p for p in plans}
        for plan in plans:
            case = cases[plan['query_id']]
            ids = case[plan['arm']]
            require(len(ids) == 30 and len(set(ids)) == 30, 'fixture30 unique offered positions')
            docs = {d['fused_position']: d['shown'] for d in case['documents']}
            expected_pairs = [{'fused_position': at, 'query_sha256': sha(case['query'].encode()), 'document_sha256': sha(docs[at].encode())} for at in ids]
            require(plan['pairs'] == expected_pairs, 'planned exact query/document hashes')
            flattened = [at for group in batches(plan) for at in group['fused_positions']]
            length_map = {at: length for group in batches(plan) for at, length in zip(group['fused_positions'], group['lengths'])}
            require(flattened == sorted(ids, key=lambda at: (length_map[at], ids.index(at))), 'stable sorted complete plan')
            for group in batches(plan):
                require(len(group['lengths']) == len(group['fused_positions']) <= 4 and group['physical_shape'] == [len(group['lengths']), max(group['lengths'])], 'actual logical/physical shape')
                require(group['physical_shape'][0]*group['physical_shape'][1] <= 512 or len(group['lengths']) == 1, 'default padded token budget')
        indexed = {(r['query_id'], r['phase'], r['arm'], r['trial']): r for r in calls}
        require(len(indexed) == 24, 'unique rank call identities')
        for row in calls:
            case = cases[row['query_id']]
            check_rank(row, case[row['arm']])
            if row['phase'] == 'hot':
                fresh = indexed[(row['query_id'], 'fresh', row['arm'], row['trial'])]
                require(row['ranked'] == fresh['ranked'] and row['cache_hits'] == 30 and row['cache_misses'] == 0, 'all16 immediate hot parity')
                require(all(row[k] == 0 for k in ('fresh_scored', 'tokens', 'padded_tokens', 'batches', 'forward_us')), 'hot zero new forward work')
            elif row['phase'] == 'fresh' or row['trial'] == 0:
                require(ranked_identity(row['ranked']) == ranked_identity(oracle[str(row['query_id'])][row['arm']]), 'fresh original/refactor/fixed bit/rank parity')
                require(row['cache_hits'] == 0, 'fresh cache reset')
                for key, value in totals(batches(by_plan[(row['query_id'], row['arm'])])).items():
                    require(row[key] == value, 'fresh planned work '+key)
            else:
                first = row['phase'].split('->')[0]
                final_plan = by_plan[(row['query_id'], row['arm'])]
                first_plan = by_plan[(row['query_id'], first)]
                if arm == 'fixed':
                    known = {signature(group, case) for group in batches(first_plan)}
                    missed = [group for group in batches(final_plan) if signature(group, case) not in known]
                else:
                    missed = missing_pair_groups(final_plan, set(case[first]))
                for key, value in totals(missed).items():
                    require(row[key] == value, 'changed-history planned work '+key)
                require(row['cache_hits'] == 30-row['fresh_scored'], 'changed-history full-batch hit count')
        comparisons = [r for r in raw if r['kind'] == 'history_comparison']
        require(len(comparisons) == 8, 'all history result flags')
        for comparison in comparisons:
            actual = indexed[(comparison['query_id'], comparison['history'], comparison['arm'], comparison['step'])]
            match = ranked_identity(actual['ranked']) == ranked_identity(oracle[str(actual['query_id'])][actual['arm']])
            require(comparison['matches_fresh_same_final_list'] == match, 'history flag derived from full scores')
            if arm == 'fixed' or comparison['step'] == 0:
                require(match, 'fixed same-list history independence')
        require(raw[-1]['public_lifetime_fresh_scored'] == sum(r['fresh_scored'] for r in calls), 'lifetime successful model work')
        raw_variants[arm] = raw
        outputs[arm] = indexed
        plans_by_arm[arm] = plans
    require(plans_by_arm['refactor'] == plans_by_arm['fixed'], 'fresh batch plans preserved')
    complete = []
    expected_cases = []
    for arm in ARMS:
        for query in QUERIES:
            for history in ('baseline->pooled', 'pooled->baseline'):
                final = history.split('->')[1]
                warm = outputs[arm][(query, history, final, 1)]
                fresh = next(r for key, r in outputs[arm].items() if key[:3] == (query, 'fresh', final))
                f = {r['fused_position']: r for r in fresh['ranked']}
                w = {r['fused_position']: r for r in warm['ranked']}
                score_changes = sum(f[at]['bits'] != w[at]['bits'] for at in f)
                rank_changes = sum(f[at]['raw_model_rank'] != w[at]['raw_model_rank'] for at in f)
                expected_cases.append({'variant': arm, 'query_id': query, 'history': history, 'cache_hits': warm['cache_hits'], 'fresh_scored': warm['fresh_scored'], 'changed_same_list_score_bits': score_changes, 'changed_same_list_raw_ranks': rank_changes})
                for at in f:
                    complete.append({'variant': arm, 'query_id': query, 'history': history, 'fused_position': at, 'fresh': f[at], 'warm_history': w[at]})
    require(len(complete) == 240 and complete == load(root/'complete-same-list-comparisons.json'), 'all240 actual joined comparisons')
    summary = load(root/'summary.json')
    require(summary['cases'] == expected_cases and summary['fresh_original_refactor_fixed_bits_and_ranks_identical'] and summary['all_hot_new_forward_work_zero'], 'recomputed summary')
    metrics = load(root/'metrics.json')
    computed = metric_values(expected_cases, raw_variants, before['clock_ticks_per_second'])
    require(metrics == computed, 'before/after/delta/percentage metrics')
    return 'Verified 86 raw records/48 rank calls, 16 hot zero-forward calls, 240 same-list comparisons, original fresh oracle, CPU 295 provider nodes, unchanged executed assets and 4 fresh binary build records. Two-query component scope only.'


def metric_values(cases, raw_variants, ticks_per_second):
    metrics = {}
    for before, after in zip(cases[:4], cases[4:]):
        for field in ('changed_same_list_score_bits', 'changed_same_list_raw_ranks', 'fresh_scored'):
            a, b = before[field], after[field]
            metrics[f"q{before['query_id']}/{before['history']}/{field}"] = {'before': a, 'after': b, 'absolute_difference': b-a, 'percentage_change': 100*(b-a)/a if a else 'N/A'}
    metrics['All16hot/new_forward_pairs'] = {'before': 0, 'after': 0, 'absolute_difference': 0, 'percentage_change': 'N/A'}
    component = {}
    for arm, raw in raw_variants.items():
        calls = [row for row in raw if row['kind'] == 'rank']
        start = raw[0]['process_before_load']
        final = raw[-1]['final_process']
        component[arm] = {
            'Hot component median wall us (8 calls)': statistics.median(r['wall_us'] for r in calls if r['phase'] == 'hot'),
            'Component rank-call wall sum us (24 calls)': sum(r['wall_us'] for r in calls),
            'Child component final HWM KiB': final['hwm_kib'],
            'Child aggregate CPU seconds (unequal fresh work)': (final['utime_ticks']+final['stime_ticks']-start['utime_ticks']-start['stime_ticks'])/ticks_per_second,
        }
    for key in component['refactor']:
        a, b = component['refactor'][key], component['fixed'][key]
        metrics[key] = {'before': a, 'after': b, 'absolute_difference': b-a, 'percentage_change': 100*(b-a)/a}
    for key in ('Engine p50/p95 latency', 'Broad nDCG/recall', 'Whole Engine-service-device memory', 'Model-index-cache-temp disk'):
        metrics[key] = {label: 'N/A' for label in ('before', 'after', 'absolute_difference', 'percentage_change')}
    return metrics


if __name__ == '__main__':
    try:
        print(verify(Path(__file__).resolve().parent))
    except (ValueError, KeyError, TypeError, OSError, IndexError, StopIteration, json.JSONDecodeError) as error:
        raise SystemExit('Verification failed: '+str(error))
