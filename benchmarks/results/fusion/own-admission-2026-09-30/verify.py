#!/usr/bin/env python3
"""Read-only evidence verification; no models, databases, binaries or compilers."""
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Verification refuses -O/PYTHONOPTIMIZE; run ordinary Python.')

GROUPS = {'monolingual': 62, 'cross_lingual': 43, 'lexical': 32, 'relational': 20}
METRICS = ('ndcg10', 'recall10', 'recall50')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, label):
    require(type(a) in (int, float) and math.isfinite(a) and abs(a-b) < 1e-13, label)


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def load(path):
    return json.loads(read(path))


def marker(log, name):
    return [json.loads(line.split(name+' ', 1)[1]) for line in log.splitlines() if name+' ' in line]


def selected(row, pooled):
    traces = row['fused']
    head = min(30, len(traces))
    graph = []
    for at in range(head, len(traces)):
        channels = [w for w in traces[at]['why'] if w['kind'] == 'channel']
        if channels and all(w['channel'] == 'graph' for w in channels):
            score = channels[-1]['score'] or 0.0
            if score >= 0.5:
                graph.append((at, score))
    extras = [at for at, score in sorted(graph, key=lambda pair: (-pair[1], pair[0]))[:30]]
    baseline = list(range(head)) + extras
    if not pooled:
        return sorted(baseline)
    chosen = []
    for channel in ('lexical_segmented', 'lexical_ngram', 'vector'):
        at = next((at for at, h in enumerate(traces) if any(
            w['kind'] == 'channel' and w['channel'] == channel and w['rank'] == 1
            and w['weight'] > 0 and w['contribution'] > 0 for w in h['why'])), None)
        if at is not None and at not in chosen and len(chosen) < head:
            chosen.append(at)
    for at in range(head):
        if len(chosen) == head:
            break
        if at not in chosen:
            chosen.append(at)
    return sorted(chosen + extras)


def values(row):
    rel = set(row['relevant'])
    require(rel and len(rel) == len(row['relevant']), 'nonempty unique gold')
    ranked = row['complete']
    gain = sum(1/math.log2(at+2) for at, h in enumerate(ranked[:10]) if h['topic'] in rel)
    ideal = sum(1/math.log2(at+2) for at in range(min(len(rel), 10)))
    return {'ndcg10': gain/ideal,
            'recall10': sum(h['topic'] in rel for h in ranked[:10])/len(rel),
            'recall50': sum(h['topic'] in rel for h in ranked[:50])/len(rel)}


def providers(log, arm, provenance, expected):
    decoder = json.JSONDecoder()
    records = []
    prefix = '${EXPERIMENT}/acceptance/own-results-v2/'+arm+'-home/models/'
    for line in log.splitlines():
        if 'ONNX graph execution-provider assignment' not in line:
            continue
        graph, _ = decoder.raw_decode(line.split('model_graph=', 1)[1])
        nodes, end = decoder.raw_decode(line.split('assigned_nodes=', 1)[1])
        require(not line.split('assigned_nodes=', 1)[1][end:].strip(), 'provider trailing data')
        require(graph.startswith(prefix), 'provider graph path')
        graph = '${MODEL_CACHE}/'+graph[len(prefix):]
        require(set(nodes) == {'CPUExecutionProvider'} and type(nodes['CPUExecutionProvider']) is int,
                'actual CPU provider required')
        asset = provenance['assets_before'][graph]
        records.append({'model_graph': graph, 'assigned_nodes': nodes, 'sha256': asset['sha256']})
    records.sort(key=lambda record: record['model_graph'])
    require(records == expected, 'raw provider assignment differs from summary/assets')
    require([list(x['assigned_nodes'].values())[0] for x in records] == ([1023] if arm == 'seed' else [1023, 295]),
            'expected actual graph node counts')


def rng(seed):
    while True:
        seed = (seed*6364136223846793005+1442695040888963407) & ((1 << 64)-1)
        yield (seed >> 33) % 2


def paired_p(differences):
    observed = sum(differences)/len(differences)
    if not any(differences):
        return 1.0
    draws = rng(0x5eed600d15c0)
    extreme = 0
    for _ in range(10000):
        flipped = sum(d if next(draws) == 0 else -d for d in differences)/len(differences)
        extreme += abs(flipped) >= abs(observed)
    return (extreme+1)/10001


def family_p(settings):
    sparse = [[(at, d) for at, d in enumerate(row) if abs(d) > 1e-9] for row in settings]
    norms = [math.sqrt(sum(d*d for _, d in row)) for row in sparse]
    observed = [abs(sum(d for _, d in row))/norm if norm else 0 for row, norm in zip(sparse, norms)]
    reached = [0]*len(settings)
    draws = rng(0xfa311e5d7a7e)
    for _ in range(10000):
        signs = [next(draws) for _ in range(157)]
        largest = max(abs(sum(-d if signs[at] else d for at, d in row))/norm if norm else 0
                      for row, norm in zip(sparse, norms))
        for at, value in enumerate(observed):
            reached[at] += largest >= value
    return [1.0 if value == 0 else (count+1)/10001 for value, count in zip(observed, reached)]


def verify(root, check_hashes=True):
    repo = root.parents[3]
    harness = repo/'benchmarks/harnesses/fusion-own-admission-2026-09-30'
    if check_hashes:
        manifest = load(root/'manifest.json')
        for relative, digest in manifest['files'].items():
            path = repo/relative
            require(path.resolve().is_relative_to(repo.resolve()), 'manifest path traversal')
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'archive hash '+relative)
        sanitation = load(root/'sanitization.json')
        for relative, record in sanitation['files'].items():
            path = repo/relative
            require(hashlib.sha256(path.read_bytes()).hexdigest() == record['archive_sha256'], 'sanitized archive hash')
            payload = read(path)
            require(hashlib.sha256(payload).hexdigest() == record['sanitized_uncompressed_sha256'], 'sanitized payload hash')
            reconstructed = payload.decode()
            for replacement in sanitation['path_prefix_replacements']:
                reconstructed = reconstructed.replace(replacement['replacement'], replacement['original'])
            require(hashlib.sha256(reconstructed.encode()).hexdigest() == record['original_sha256'],
                    'reconstructed original path-only payload hash')
            require(len(payload) == record['sanitized_bytes'] and len(reconstructed.encode()) == record['original_bytes'],
                    'original/sanitized byte counts')
    provenance = load(root/'provenance.json')
    require(provenance['source_revision'] == '13ee710c9df865f1dac98dc77a8108e438ddc539', 'source revision')
    require('UNKNOWN' in read(root/'historical-config-limit.md').decode(), 'historical knobs UNKNOWN')
    require(provenance['assets_before'] == provenance['assets_after'], 'assets before/after')
    summary = load(root/'summary.json.gz')
    require(summary['queries'] == 157 and summary['changed_candidate_queries'] == 29, 'summary counts')
    queries = load(harness/'tests/corpus/queries.json')
    memories = load(harness/'tests/corpus/memories.json')
    require(len(queries) == 157 and len(memories) == 230, 'fixture counts')
    for name in ('memories.json', 'queries.json'):
        require(hashlib.sha256((harness/'tests/corpus'/name).read_bytes()).hexdigest() == provenance['dataset_files'][name], 'native fixture hash')
    sources = load(root/'sources.json')
    require(sources['revision'] == provenance['source_revision'], 'frozen source revision')
    for arm, metadata in sources['variants'].items():
        require(metadata['channel_pool'] == (arm == 'pooled'), 'source selection mode')
        for relative, digest in metadata['overlaid_sha256'].items():
            path = harness/arm/'src'/(Path(relative).name+'.in')
            require(hashlib.sha256(read(path)).hexdigest() == digest, 'frozen overlaid source identity')
    binary_records = load(root/'binaries.json')
    require(binary_records == provenance['binaries'], 'retained binary identities')
    for arm in ('baseline', 'pooled'):
        engine = read(harness/arm/'src/engine.rs.in').decode()
        require('const GRAPH_SHOWN_FROM: f32 = 0.5;' in engine and 'const GRAPH_SHOWN_AT_MOST: usize = 30;' in engine, 'native graph policy')
        require('const EXPERIMENTAL_CHANNEL_POOL: bool = '+str(arm == 'pooled').lower()+';' in engine, 'frozen arm toggle')
    a = read(harness/'baseline/src/engine.rs.in').decode()
    b = read(harness/'pooled/src/engine.rs.in').decode()
    require(a.replace('const EXPERIMENTAL_CHANNEL_POOL: bool = false;', 'const EXPERIMENTAL_CHANNEL_POOL: bool = true;') == b, 'only arm source difference')
    harness_sha = hashlib.sha256(read(harness/'tests/scratch_pool_retrieval.rs.in')).hexdigest()
    require(all(record['source_sha256'] == harness_sha for record in binary_records), 'test-source identities')
    arms = {}
    expected_identity = None
    for arm in ('seed', 'baseline', 'pooled'):
        log = read(root/(arm+'.log.gz')).decode()
        require('test result: ok. 1 passed; 0 failed;' in log, 'successful native test '+arm)
        identity = load(root/(arm+'-index-identity.json'))
        require(marker(log, 'POOL_INDEX_IDENTITY') == [identity['before']], 'raw index before')
        require(marker(log, 'POOL_INDEX_AFTER') == [identity['after']], 'raw index after')
        require(marker(log, 'POOL_GRAPH_IDENTITY') == [identity['graph']], 'raw graph identity')
        require(identity['before'] == {'documents': 230, 'mode': 'native-default-flat-buffer',
                                      'project': 'eval-accuracy-8257bd6e24a243c4', 'vector_completeness': 0.0}, 'flat-buffer premise')
        require(identity['after'] == {'documents': 230, 'project': identity['before']['project'], 'vector_completeness': 0.0}, 'unchanged coverage0')
        require(identity['graph'] == [['mentions', 11]], 'native graph count')
        if expected_identity is None:
            expected_identity = identity
        require(identity == expected_identity, 'same seed/arm index identity')
        providers(log, arm, provenance, summary['seed_provider'] if arm == 'seed' else summary['providers'][arm])
        if arm == 'seed':
            require(not marker(log, 'POOL_ACCEPT_JSON'), 'seed was not measured')
            continue
        rows = [json.loads(line) for line in read(root/(arm+'.jsonl.gz')).decode().splitlines()]
        require(rows == marker(log, 'POOL_ACCEPT_JSON'), 'raw log JSONL equality')
        require(len(rows) == 157 and [row['query_id'] for row in rows] == list(range(157)), 'paired query IDs')
        require(Counter(row['group'] for row in rows) == GROUPS, 'native group counts')
        require(Counter(row['offered_candidates'] for row in rows) == {30: 155, 31: 2}, 'native offered30–31 counts')
        for row, query in zip(rows, queries):
            require(row['dataset'] == 'own' and row['excluded'] == [], 'own dataset')
            require(all(row[key] == query[key] for key in ('query', 'group', 'relevant')), 'native fixture pairing')
            for field in ('fused', 'complete', 'top10'):
                hits = row[field]
                require([hit['rank'] for hit in hits] == list(range(1, len(hits)+1)), 'contiguous ranks')
                require(len({hit['topic_id'] for hit in hits}) == len(hits), 'unique live topic IDs')
            require(row['top10'] == row['complete'][:10] and len(row['top10']) == 10, 'actual prefix identity')
            require({h['topic_id'] for h in row['fused']} == {h['topic_id'] for h in row['complete']}, 'complete fused union')
            require(sum(any(w.get('channel') == 'vector' for w in h['why']) for h in row['fused']) == 50, 'native vector recall50')
            positions = row['selected_fused_positions']
            require(positions == selected(row, arm == 'pooled'), 'native main+graph selection')
            offered = {row['fused'][at]['topic_id'] for at in positions}
            scored = {h['topic_id'] for h in row['complete'] if any(w['kind'] == 'reranked' for w in h['why'])}
            require(offered == scored and len(offered) == row['offered_candidates'], 'offered/available Why identities')
            measured = values(row)
            native = row['native_metrics'][row['group']]
            close(native['ndcg'], measured['ndcg10'], 'native nDCG arithmetic')
            close(native['recall'], measured['recall50'], 'native recall arithmetic')
            require(native['queries'] == 1 and native['per_query'] == [native['ndcg']], 'native per-query Scores')
            deep = sum(h['topic'] in row['relevant'] for h in row['complete'][10:50])
            require(native['deep'] == deep and native['with_work'] == int(deep > 0), 'native deep metrics')
        arms[arm] = rows
    require(summary['providers']['baseline'] == summary['providers']['pooled'], 'same providers')
    require(len(summary['rows']) == 157, 'summary paired rows')
    for before, after, paired in zip(arms['baseline'], arms['pooled'], summary['rows']):
        require(before['fused'] == after['fused'], 'all fused rows identical')
        require(before['offered_candidates'] == after['offered_candidates'], 'same per-query budget')
        require(all(paired[key] == before[key] == after[key] for key in ('query_id', 'query', 'group', 'relevant', 'offered_candidates')), 'summary query pair')
        offered_before = {before['fused'][at]['topic'] for at in before['selected_fused_positions']}
        offered_after = {after['fused'][at]['topic'] for at in after['selected_fused_positions']}
        require(sorted(offered_after-offered_before) == paired['added_candidates'] and sorted(offered_before-offered_after) == paired['removed_candidates'], 'candidate churn')
        for arm, row in (('baseline', before), ('pooled', after)):
            require(paired[arm+'_top10'] == row['top10'], 'summary actual top10')
            for metric, value in values(row).items():
                close(paired[arm+'_'+metric], value, 'summary metric '+metric)
    require([row['query_id'] for row in arms['baseline'] if row['offered_candidates'] == 31] == [1, 62], 'two native graph extras')
    changed_cases = [row for row in summary['rows'] if row['baseline_ndcg10'] != row['pooled_ndcg10']]
    require([row['query_id'] for row in changed_cases] == [80, 101], 'changed ranking cases')
    cases = load(root/'changed-ndcg-cases.json.gz')
    require(len(cases) == len(changed_cases), 'changed-case count')
    for case, paired in zip(cases, changed_cases):
        require(all(case[key] == value for key, value in paired.items()), 'retained changed-case details')
        for arm in ('baseline', 'pooled'):
            complete = arms[arm][case['query_id']]['complete']
            details = [{'topic': topic, 'rank': hit['rank'], 'why': hit['why']}
                       for topic in case['relevant'] for hit in complete if hit['topic'] == topic]
            require(case['relevant_details'][arm] == details, 'changed-case relevant scores')
    metrics = load(root/'metrics.json')
    for group, count in GROUPS.items():
        rows = [row for row in summary['rows'] if row['group'] == group]
        retained = summary['groups'][group]
        require(retained['queries'] == count, 'group size')
        for metric in METRICS:
            before = sum(row['baseline_'+metric] for row in rows)/count
            after = sum(row['pooled_'+metric] for row in rows)/count
            close(retained['baseline_'+metric], before, 'group before mean')
            close(retained['pooled_'+metric], after, 'group after mean')
            expected = {'before': before, 'after': after, 'absolute_difference': after-before, 'percentage_change': 100*(after-before)/before}
            for key, value in expected.items():
                close(metrics[group+'/'+metric][key], value, 'metric delta/percentage')
        require(retained['changed_candidate_queries'] == sum(bool(row['added_candidates']) for row in rows), 'group candidate churn')
        require(retained['ndcg_wins'] == sum(row['pooled_ndcg10']-row['baseline_ndcg10'] > 1e-9 for row in rows), 'group wins')
        require(retained['ndcg_losses'] == sum(row['pooled_ndcg10']-row['baseline_ndcg10'] < -1e-9 for row in rows), 'group losses')
    for key, record in metrics.items():
        if '/' not in key or key not in [g+'/'+m for g in GROUPS for m in METRICS]:
            require(set(record.values()) == {'N/A'}, 'unmeasured costs N/A')
    stats = load(root/'native-statistics.json')['cells']
    require(len(stats) == 12 and len({(c['group'], c['metric']) for c in stats}) == 12, '12 statistical cells')
    source = read(harness/'native-statistics.rs.in').decode()
    matched = re.search(r'let settings: Vec<Vec<f64>> = vec!\[(.*)\];\nlet family', source, re.S)
    settings = ast.literal_eval('['+matched.group(1).replace('vec!', '')+']')
    expected_settings = [[row['pooled_'+cell['metric']]-row['baseline_'+cell['metric']] if row['group'] == cell['group'] else 0 for row in summary['rows']] for cell in stats]
    require(settings == expected_settings, 'native 157-ID family input alignment')
    family = family_p(settings)
    for at, cell in enumerate(stats):
        differences = [row['pooled_'+cell['metric']]-row['baseline_'+cell['metric']] for row in summary['rows'] if row['group'] == cell['group']]
        close(cell['delta'], sum(differences)/len(differences), 'native paired delta')
        require(cell['wins'] == sum(d > 1e-9 for d in differences) and cell['losses'] == sum(d < -1e-9 for d in differences), 'native paired W/L')
        require(cell['ties'] == len(differences)-cell['wins']-cell['losses'], 'native paired ties')
        close(cell['raw_p'], paired_p(differences), 'native Monte Carlo raw p')
        close(cell['family_p'], family[at], 'native Monte Carlo family p')
    require(marker(read(root/'native-statistics.log').decode(), 'STAT_JSON') == stats, 'raw native statistics records')
    return 'Verified 314 actual Engine traces, 157 paired IDs, identical fused lists,30–31 selected candidates, coverage0, raw CPU nodes1023/295, independent metrics and12 native Monte Carlo cells. Historical tuning UNKNOWN; no build attestation or adoption.'


if __name__ == '__main__':
    try:
        print(verify(Path(__file__).resolve().parent))
    except (ValueError, KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as error:
        raise SystemExit('Verification failed: '+str(error))
