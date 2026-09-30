#!/usr/bin/env python3
"""Read-only verification of the finished DiskANN archive, including recomputation."""
import gzip,hashlib,importlib.machinery,importlib.util,json,math,statistics,sys
if sys.flags.optimize:
    raise SystemExit("Verification requires Python assertions; run without -O/-OO or PYTHONOPTIMIZE.")
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parent
repo=root.parents[3]
manifest=json.loads((root/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
    assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==digest,relative
def load(name,filename):
    loader=importlib.machinery.SourceFileLoader(name,str(filename))
    spec=importlib.util.spec_from_loader(loader.name,loader)
    module=importlib.util.module_from_spec(spec);loader.exec_module(module);return module
code=repo/'benchmarks/harnesses/restart-floor-2026-09-30'
runner=load('archived_disk_runner',code/'run.py.in')
calculator=load('archived_disk_calculator',code/'analyze.py.in')
raw=[json.loads(x) for x in (root/'raw.jsonl').read_text().splitlines()]
expected_commits={'predecessor':'f57f9c218d03d88666b3cc89fae9ae7e9eed2e50','candidate':'11493c1388f74b087db23136a94e4b8feed1efe3','main':'315c10242ddf7a1cec3bccbf550a942320e09557'}
assert set(r['arm'] for r in raw)==set(expected_commits)
assert len([r for r in raw if r['phase']=='process_total'])==9
assert {(r['arm'],r['repetition']) for r in raw}=={(arm,rep) for arm in expected_commits for rep in range(3)}, 'unexpected raw repetition'
expected_queries=[j*(18000-2)//25+1 for j in range(1,25)]
warm={}
for arm,commit in expected_commits.items():
    for rep in range(3):
        rows=[r for r in raw if r['arm']==arm and r['repetition']==rep]
        assert rows and all(r['commit']==commit for r in rows)
        phases={phase:[r for r in rows if r['phase']==phase] for phase in {r['phase'] for r in rows}}
        for phase in ['open','write_and_memory_drain','durability_flush','maintenance','search_cold','new_write_search','closed_index','process_total']:assert len(phases[phase])==1,(arm,rep,phase)
        assert Counter(r['phase'] for r in rows)==Counter({**{p:1 for p in ['open','write_and_memory_drain','durability_flush','maintenance','search_cold','new_write_search','closed_index','process_total']},'search_warmup':2,'search_warm':24}), (arm,rep,'unexpected phase multiset')
        assert [r['extra']['query_document'] for r in phases['search_warm']]==expected_queries, 'query workload differs from fixed order'
        assert phases['open'][0]['extra']['documents']==18000
        maintenance=phases['maintenance'][0]
        assert maintenance['extra']['documents']==18001
        assert maintenance['extra']['optimize_completed_delta']==(0 if arm=='candidate' else 1)
        complete=maintenance['extra']['completeness']
        if arm=='candidate':assert complete<1 and math.isclose(complete,18000/18001,abs_tol=1e-6)
        else:assert complete==1
        assert phases['durability_flush'][0]['extra']['flushed']==1
        closed=phases['closed_index'][0]
        assert closed['wall_ms'] is None and closed['measurement_kind']=='diagnostic'
        new=phases['new_write_search'][0]
        assert new['extra']['known_new_topic_retrieved'] and 'restart-proof' in new['extra']['topics']
        for row in rows:
            if row['phase']!='process_total':
                assert row['index']=='disk'
                assert row['clock_ticks_per_second']==100
                for kind in ['user','system']:
                    expected=None if row['process_before'] is None or row['process_after'] is None else (row['process_after'][kind+'_ticks']-row['process_before'][kind+'_ticks'])/row['clock_ticks_per_second']
                    assert row['cpu_'+kind+'_seconds']==expected, 'derived CPU field disagrees with retained tick delta'
                assert set(row['actual_providers'])=={'embedding','reranker'}
                for provider in row['actual_providers'].values():
                    assert set(provider['assigned_nodes'])=={'CPUExecutionProvider'} and provider['assigned_nodes']['CPUExecutionProvider']>0
        warm[(arm,rep)]={r['extra']['query_document']:r['extra']['topics'] for r in phases['search_warm']}
        assert len(warm[(arm,rep)])==24 and all(len(topics)==10 for topics in warm[(arm,rep)].values())
        for row in phases['search_warm']:
            rank=next((i+1 for i,topic in enumerate(row['extra']['topics']) if topic==f'incident-{row["extra"]["query_document"]}'),None)
            assert rank==row['extra']['rank']
        log=gzip.decompress((root/'logs'/f'{rep}-{arm}.log.gz').read_bytes()).decode()
        assert 'test result: ok. 1 passed;' in log and 'synchronous pread()' in log
        actual_providers=runner.cpu_provider_assignments(log)
        assert actual_providers==maintenance['actual_providers']
        logged=[json.loads(x.removeprefix('RESTART_JSON ')) for x in log.splitlines() if x.startswith('RESTART_JSON ')]
        # Each native log observation binds exactly one archived product row.
        product_rows=[r for r in rows if r['phase']!='process_total']
        assert len(logged)==len(product_rows), (arm,rep,'raw/log cardinality differs')
        unmatched=list(product_rows)
        for logrow in logged:
            matches=[i for i,r in enumerate(unmatched) if all(key in r and r[key]==value for key,value in logrow.items())]
            assert len(matches)==1, (arm,rep,'native log observation has no unique raw row')
            unmatched.pop(matches[0])
        assert not unmatched, (arm,rep,'raw observations absent from native log')
for reference in ['predecessor','main']:
    assert sum(warm[(reference,rep)]==warm[('candidate',rep)] for rep in range(3))==3
provenance=json.loads((root/'provenance.json').read_text())
assert manifest['files'][str((code/'harness.rs.in').relative_to(repo))]==provenance['harness']['sha256'], 'harness missing from manifest or recorded digest differs'
assert hashlib.sha256((code/'harness.rs.in').read_bytes()).hexdigest()==provenance['harness']['sha256'], 'harness disagrees with recorded compiled source'
assert len((code/'harness.rs.in').read_bytes())==provenance['harness']['bytes']
seed_log=gzip.decompress((root/'logs/seed-disk.log.gz').read_bytes()).decode()
seed_rows=[json.loads(line.removeprefix('RESTART_JSON ')) for line in seed_log.splitlines() if line.startswith('RESTART_JSON ')]
assert Counter(row['phase'] for row in seed_rows)==Counter({'open':1,'seed_complete':1}), 'unexpected seed log phases'
assert next(row for row in seed_rows if row['phase']=='seed_complete')==provenance['seed_complete_diagnostic'], 'seed diagnostic disagrees with native seed log'
assert provenance['seed_complete_diagnostic']['extra']['completeness']==1
assert provenance['seed_complete_diagnostic']['extra']['index_disk'][0]==provenance['all_files_including_floor_marker']
seed_files=provenance['seed_files']
assert len(seed_files)==len({entry['path'] for entry in seed_files}), 'duplicate seed inventory path'
floor_markers=[entry for entry in seed_files if entry['path'].endswith('/.pamin-optimized-files')]
assert len(floor_markers)==1
floor_bytes=b'v1 273\n'
assert floor_markers[0]['bytes']==len(floor_bytes) and floor_markers[0]['sha256']==hashlib.sha256(floor_bytes).hexdigest(), 'floor marker bytes differ'
# Stopped seed inventory follows index close; file count can change after the diagnostic.
assert provenance['index']=='disk'  and provenance['saved_floor']==273 and provenance['all_files_including_floor_marker']==274
expected_profile=['gpahal/bge-m3-onnx-int8','topic','disk','named','reversed-keys']
assert provenance['native_profile']==expected_profile
profile_assets=[entry for entry in provenance['seed_files'] if entry['path'].endswith('/profile')]
assert len(profile_assets)==1
profile_bytes='\n'.join(expected_profile).encode()
assert profile_assets[0]['bytes']==len(profile_bytes) and profile_assets[0]['sha256']==hashlib.sha256(profile_bytes).hexdigest()
conversion=provenance['conversion_schema_and_logical_digest']
conversion_log=gzip.decompress((root/'logs/disk-conversion.log.gz').read_bytes()).decode()
observed=[json.loads(line.split('DISK_SETUP_JSON ',1)[1]) for line in conversion_log.splitlines() if 'DISK_SETUP_JSON ' in line]
assert observed==conversion, 'conversion provenance disagrees with retained setup log'
assert [row['phase'] for row in observed]==['before','after']
assert hashlib.sha256((code/'disk_schema.rs.in').read_bytes()).hexdigest()==provenance['conversion_helper_artifact']['scratch_source_sha256']
assert conversion[0]['logical_digest']==conversion[1]['logical_digest']==[18000,'eb2483ff691e5e245079745e4745934620caf7ae692deea3766c84511c061d6b']
assert conversion[1]['same_all_stored_document_bits'] and not conversion[1]['floor_written']
schema=conversion[1]['schema'];assert schema['segment_documents']==2000
vector=schema['fields']['embedding']
assert {key:vector[key] for key in ['dtype','dimension','index_type','metric','degree','build_list','pq_chunks','quantize','quantizer_rotate']}=={'dtype':22,'dimension':1024,'index_type':5,'metric':3,'degree':64,'build_list':100,'pq_chunks':0,'quantize':0,'quantizer_rotate':False}
libraries=json.loads((root/'post-trial-libraries.json').read_text())
assert libraries['recorded_utc'] and libraries['scope']
assert set(libraries['libraries'])=={'runtime_library','native_zvec_library'}
for key, entry in libraries['libraries'].items():
    assert entry==provenance[key], 'post-trial runtime/native identity mismatch'
post=json.loads((root/'post-trial-assets.json').read_text())
assert post['all_recorded_source_graphs_and_prepared_external_weights_unchanged']
pre_assets=provenance['source_assets']+provenance['prepared_graphs_and_external_weights']+provenance['prepared_source_metadata']
def inventory(entries):
    assert len(entries)==len({entry['path'] for entry in entries}), 'duplicate asset path'
    return {entry['path']:(entry['bytes'],entry['sha256']) for entry in entries}
assert inventory(post['assets'])==inventory(pre_assets), 'post-trial inventory differs from pretrial provenance'
assert all(x['sha256']==x['posttrial_sha256'] and x['unchanged'] for x in post['assets'])
assert any(x['path'].endswith('.onnx.data') for x in post['assets'])
assert calculator.summarize(raw)==json.loads((root/'summary.json').read_text())
for binary in json.loads((root/'binaries.json').read_text()):
    assert binary['sha256']==binary['current_pretrial_hash']['sha256']==binary['posttrial_hash']['sha256']
    assert binary['current_pretrial_hash']['bytes']==binary['posttrial_hash']['bytes']
    assert binary['posttrial_hash']['recorded_utc']
episode=json.loads((root/'episode-elapsed.json').read_text())
for arm in expected_commits:
    values=[x['wall_seconds'] for x in raw if x['phase']=='process_total' and x['arm']==arm]
    assert episode['arms'][arm]=={'process_seconds':values,'median_seconds':statistics.median(values)}
for arm in ['predecessor','main']:
    a=episode['arms'][arm]['median_seconds'];b=episode['arms']['candidate']['median_seconds']
    assert episode['comparisons'][arm]=={'before':a,'after':b,'absolute_difference':b-a,'percentage_change':100*(b-a)/a}
labels=[
('Known-topic recall@10 (synthetic)','known_topic_recall_at_10',1,''),
('Known-topic MRR@10 (synthetic)','known_topic_mrr_at_10',1,''),
('Optimize jobs per first upkeep tick','optimize_jobs',1,''),
('Full-process elapsed median, includes diagnostics','episode',1,' s'),
('Engine open wall median','open_ms',1,' ms'),
('Write + urgent drain wall median','write_ms',1,' ms'),
('Durability flush wall median','durability_flush_ms',1,' ms'),
('Maintenance wall median','maintenance_ms',1,' ms'),
('Maintenance process CPU median, 10ms tick counters','maintenance_cpu_s',1,' s'),
('First-search wall median, model load included','first_search_ms',1,' ms'),
('Warm search p50, median across processes','warm_p50_ms',1,' ms'),
('Warm search p95, median across processes','warm_p95_ms',1,' ms'),
('Peak process RSS','peak_process_rss_bytes',2**20,' MiB'),
('RSS after maintenance','maintenance_rss_bytes',2**20,' MiB'),
('Open index apparent bytes','index_apparent_bytes',2**20,' MiB'),
('Open index allocated bytes','index_allocated_bytes',2**20,' MiB'),
('Closed index apparent bytes','closed_index_apparent_bytes',2**20,' MiB'),
('Closed index allocated bytes','closed_index_allocated_bytes',2**20,' MiB'),
('Vector graph completeness, hybrid visibility checked','vector_completeness',1,'')]
summary=json.loads((root/'summary.json').read_text())
displayed=[line for line in (root/'README.md').read_text().splitlines() if line.startswith('|')]
expected=[]
for reference in ['predecessor','main']:
    expected+=['| Metric | Before | After | Absolute difference | Percentage change |','| --- | ---: | ---: | ---: | ---: |']
    for label,key,divisor,unit in labels:
        if key=='episode':
            value=episode['comparisons'][reference]
            before,after,delta,percent=[value[k] for k in ['before','after','absolute_difference','percentage_change']]
        else:
            value=summary['comparisons'][reference]['metric_differences'][key]
            before,after,delta,percent=[value[k] for k in ['before','after','absolute_delta','percent_delta']]
        if key=='first_search_ms':
            expected.append(f'| {label} | {before/divisor:.6f}{unit} | {after/divisor:.6f}{unit} | Withheld: unstable three-process sample | Withheld: unstable three-process sample |')
        else:
            expected.append(f'| {label} | {before/divisor:.6f}{unit} | {after/divisor:.6f}{unit} | {delta/divisor:+.6f}{unit} | {percent:+.3f}% |')
assert displayed==expected, 'published README tables disagree with recomputed summaries'
load('retrospective_restart_binding',root/'rebuild/verify.py').verify(repo=repo)
print('verified DiskANN:9 processes,72/72 paired ordered top10 per reference,work counts,CPU providers,new-write visibility,native profile/schema/digest,external weight identities and recomputed metrics')
