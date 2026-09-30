#!/usr/bin/env python3
"""Read-only verification of the finished DiskANN archive, including recomputation."""
import gzip,hashlib,importlib.machinery,importlib.util,json,math,statistics,sys
if sys.flags.optimize:
    raise SystemExit("Verification requires Python assertions; run without -O/-OO or PYTHONOPTIMIZE.")
sys.dont_write_bytecode=True
from pathlib import Path
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
warm={}
for arm,commit in expected_commits.items():
    for rep in range(3):
        rows=[r for r in raw if r['arm']==arm and r['repetition']==rep]
        assert rows and all(r['commit']==commit for r in rows)
        phases={phase:[r for r in rows if r['phase']==phase] for phase in {r['phase'] for r in rows}}
        for phase in ['open','write_and_memory_drain','durability_flush','maintenance','search_cold','new_write_search','closed_index','process_total']:assert len(phases[phase])==1,(arm,rep,phase)
        assert len(phases['search_warmup'])==2 and len(phases['search_warm'])==24
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
        # Archived rows add labels/CPU deltas; raw product observations must match logs.
        for logrow in logged:
            match=[r for r in rows if r['phase']==logrow['phase'] and r.get('extra')==logrow.get('extra')]
            assert len(match)==1
            for key,value in logrow.items():assert match[0][key]==value
for reference in ['predecessor','main']:
    assert sum(warm[(reference,rep)]==warm[('candidate',rep)] for rep in range(3))==3
provenance=json.loads((root/'provenance.json').read_text())
assert provenance['index']=='disk' and provenance['saved_floor']==273 and provenance['all_files_including_floor_marker']==274
assert len(provenance['native_profile'])==5 and provenance['native_profile'][2]=='disk'
conversion=provenance['conversion_schema_and_logical_digest']
assert conversion[0]['logical_digest']==conversion[1]['logical_digest']==[18000,'eb2483ff691e5e245079745e4745934620caf7ae692deea3766c84511c061d6b']
assert conversion[1]['same_all_stored_document_bits'] and not conversion[1]['floor_written']
schema=conversion[1]['schema'];assert schema['segment_documents']==2000
vector=schema['fields']['embedding']
assert {key:vector[key] for key in ['dtype','dimension','index_type','metric','degree','build_list','pq_chunks','quantize','quantizer_rotate']}=={'dtype':22,'dimension':1024,'index_type':5,'metric':3,'degree':64,'build_list':100,'pq_chunks':0,'quantize':0,'quantizer_rotate':False}
post=json.loads((root/'post-trial-assets.json').read_text())
assert post['all_recorded_source_graphs_and_prepared_external_weights_unchanged']
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
print('verified DiskANN:9 processes,72/72 paired ordered top10 per reference,work counts,CPU providers,new-write visibility,native profile/schema/digest,external weight identities and recomputed metrics')
