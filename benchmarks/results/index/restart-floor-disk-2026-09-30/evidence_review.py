"""Read-only semantic evidence guards; no historical runner mutation or inference."""
import re
import hashlib,json,math,statistics
from pathlib import PurePosixPath

def measurement_annotations(row):
    """Bind runner-added scopes and native disk observations before arithmetic."""
    assert row.get('conditions')=='new Engine process; warm OS/model file cache; isolated copied stopped DB/index', 'cache/isolation conditions differ from historical runner'
    assert row.get('cpu_scope')=='process threads only; PostgreSQL excluded', 'CPU measurement scope differs from historical runner'
    assert row.get('rss_scope')=='process only', 'RSS measurement scope differs from historical runner'
    if row['phase'] in {'maintenance','closed_index'}:
        usage=row['extra'].get('index_disk')
        assert type(usage) is list and len(usage)==3 and all(type(v) is int and v>=0 for v in usage), 'index_disk must contain three nonnegative integers (booleans refused)'

def memory_status(snapshot):
    """Require the complete /proc status observations before any RSS arithmetic."""
    status=snapshot.get('rss_status')
    assert isinstance(status,list) and len(status)==2, 'RSS/HWM observations must be complete and unique'
    values={}
    for line in status:
        assert isinstance(line,str), 'RSS/HWM observation must be text'
        match=re.fullmatch(r'(VmRSS|VmHWM):\s+([0-9]+) kB',line)
        assert match is not None, 'RSS/HWM observation must be well-formed and nonnegative'
        key,value=match.groups()
        assert key not in values, 'RSS/HWM observations must be complete and unique'
        values[key]=int(value)
    assert set(values)=={'VmRSS','VmHWM'}, 'RSS/HWM observations must be complete and unique'
    assert values['VmHWM']>=values['VmRSS'], 'RSS exceeds process high-water mark'
    return values

def optimize_job_count(value,expected):
    assert type(value) is int and value>=0, 'optimize job count must be a nonnegative integer (booleans refused)'
    assert value==expected, 'arm optimize job count differs'

def real_measurement(value, positive=False):
    assert type(value) in {int,float} and math.isfinite(value) and (value>0 if positive else value>=0), 'invalid real timing observation (booleans refused)'


HWM_ANOMALIES={
    'disk':[{'arm':'candidate','repetition':2,'previous':[9,'search_warm','process_before',1770044], 'current':[9,'search_warm','process_after',1769984]}],
    'memory':[{'arm':'main','repetition':2,'previous':[2,'durability_flush','process_before',1263960], 'current':[2,'durability_flush','process_after',1263580]},
              {'arm':'candidate','repetition':1,'previous':[5,'search_warmup','process_before',1771088], 'current':[5,'search_warmup','process_after',1770628]}],
}
HWM_SCOPE='Exact retained historical anomalies only; no tolerance. HWM-derived peak process RSS is uncertified/N/A. Sampled VmRSS observations remain observations, not lifetime peaks. Original raw/log/calculator numbers are preserved.'
HWM_TABLE_ROW='| Peak process RSS | N/A: historical HWM chronology uncertified | N/A: historical HWM chronology uncertified | N/A | N/A |'


def hwm_allowance(index,arm,repetition):
    return [item for item in HWM_ANOMALIES[index] if item['arm']==arm and item['repetition']==repetition]


def process_observations(rows, historical_anomalies=()):
    """Validate one process before arithmetic; bind declared historical defects."""
    previous=None;previous_ticks=None;anomalies=[]
    for position,row in enumerate(rows):
        if row['phase']=='process_total':
            real_measurement(row['wall_seconds'],positive=True)
            continue
        if row['phase']=='closed_index':assert row['wall_ms'] is None
        else:real_measurement(row['wall_ms'])
        for key in ['cpu_user_seconds','cpu_system_seconds']:
            if row[key] is not None:real_measurement(row[key])
        for field in ['process_before','process_after']:
            snapshot=row[field]
            if snapshot is None:continue
            ticks=tuple(snapshot[key] for key in ['user_ticks','system_ticks'])
            assert all(type(value) is int and value>=0 for value in ticks), 'invalid process CPU tick counters (booleans refused)'
            assert previous_ticks is None or all(current>=old for current,old in zip(ticks,previous_ticks)), 'process cumulative CPU counters decreased within/across phases'
            previous_ticks=ticks
            hwm=memory_status(snapshot)['VmHWM'];observation=[position,row['phase'],field,hwm]
            if previous is not None and hwm<previous[-1]:
                anomalies.append({'arm':row.get('arm'),'repetition':row.get('repetition'),'previous':previous,'current':observation})
            previous=observation
    assert anomalies==list(historical_anomalies), 'process high-water mark decreased within/across phases or declared historical anomaly changed'
    return anomalies


def historical_hwm_review(raw,index,raw_sha256):
    sampled=[];anomalies=[]
    for arm in ['main','predecessor','candidate']:
        for repetition in range(3):
            rows=[r for r in raw if r['arm']==arm and r['repetition']==repetition]
            anomalies+=process_observations(rows,hwm_allowance(index,arm,repetition))
            observations=[memory_status(r[field])['VmRSS'] for r in rows if r['phase']!='process_total'
                          for field in ['process_before','process_after'] if r[field] is not None]
            sampled.append({'arm':arm,'repetition':repetition,'observations':len(observations),
                            'maximum_observed_rss_kib':max(observations,default=None)})
    assert anomalies==HWM_ANOMALIES[index], 'historical high-water anomaly set differs'
    return {'schema':1,'index':index,'raw_sha256':raw_sha256,'scope':HWM_SCOPE,
            'exact_historical_anomalies':anomalies,'hwm_peak_certified':False,'published_hwm_peak':'N/A',
            'sampled_rss_observations':sampled}


def qualified_summary(calculated,hwm_review,calculator_sha256):
    """Supersede derived numeric HWM peaks; never mutate retained raw/calculator."""
    assert hwm_review['hwm_peak_certified'] is False and hwm_review['published_hwm_peak']=='N/A', 'historical peak cannot be certified'
    assert hwm_review['exact_historical_anomalies']==HWM_ANOMALIES[hwm_review['index']], 'historical anomaly binding differs'
    assert all(re.fullmatch(r'[0-9a-f]{64}',value) for value in [hwm_review['raw_sha256'],calculator_sha256]), 'invalid derivation source hash'
    legacy=(json.dumps(calculated,indent=2)+'\n').encode()
    result=json.loads(legacy)
    key='peak_process_rss_bytes'
    for arm in result['arms'].values():
        for process in arm['processes']:process[key]=None
        arm['median_of_process_metrics'][key]=None
    for comparison in result['comparisons'].values():
        comparison['metric_differences'][key]={field:None for field in ['before','after','absolute_delta','percent_delta']}
    result['metric_certification']={key:{'certified':False,'status':'N/A','reason':'historical VmHWM chronology uncertified'}}
    result['summary_derivation']={'schema':1,'method':'archived calculator recomputation followed by explicit HWM qualification; supersedes numeric summary only',
        'raw_sha256':hwm_review['raw_sha256'],'archived_calculator_sha256':calculator_sha256,
        'superseded_numeric_summary_sha256':hashlib.sha256(legacy).hexdigest()}
    return result


def binary_binding(binaries,provenance,expected_commits):
    assert len(binaries)==len(expected_commits), 'incomplete binary arm inventory'
    assert len({b['arm'] for b in binaries})==len(binaries), 'duplicate binary arm'
    frozen=provenance['frozen_arms']
    assert len(frozen)==len(expected_commits) and len({b['arm'] for b in frozen})==len(frozen), 'duplicate/incomplete pretrial inventory'
    actual={b['arm']:b for b in binaries};pre={b['arm']:b for b in frozen}
    assert set(actual)==set(pre)==set(expected_commits), 'binary arm set'
    assert len({b['binary'] for b in binaries})==len(binaries), 'duplicate executable path'
    for arm,commit in expected_commits.items():
        b,p=actual[arm],pre[arm]
        assert {k:b[k] for k in ['arm','commit','binary','sha256','current_pretrial_hash']}==p, 'arm-keyed pretrial executable differs'
        assert b['commit']==commit and b['current_pretrial_hash']['path']==b['binary'], 'arm/commit/path binding'
        assert b['sha256']==b['current_pretrial_hash']['sha256']==b['posttrial_hash']['sha256'], 'executable digest drift'
        assert b['current_pretrial_hash']['bytes']==b['posttrial_hash']['bytes']>0 and b['posttrial_hash']['path']==b['binary'], 'executable size/path drift'
        assert b['posttrial_hash']['recorded_utc']

def runner_binding(path,provenance):
    b=path.read_bytes();record=provenance['runner']
    assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256'], 'historical runner bytes/digest differ'

def provider_binding(provider,role,provenance,bindings):
    record=bindings['roles'][role]
    expected_source={'embedding':('model_quantized.onnx','2b34e84df040034d4b9eabb62383a87c18955822','gpahal--bge-m3-onnx-int8--model_quantized.onnx.source'),'reranker':('model_int8.onnx','6f5ff65298512715a1e669753bc754d2bc8f367b','onnx-community--bge-reranker-v2-m3-ONNX--onnx--model_int8.onnx.source')}
    name,revision,metadata_name=expected_source[role]
    assert PurePosixPath(record['source_graph']).name==name and record['revision']==revision and PurePosixPath(record['source_metadata']['path']).name==metadata_name, 'role/source artifact differs'
    path=provider['model_graph'];prefix='<SCRATCH>/results-disk/'
    assert path.startswith(prefix), 'provider graph outside historical process copy'
    rest=path[len(prefix):].split('/',1)
    assert len(rest)==2 and rest[0] in {f'{rep}-{arm}' for rep in range(3) for arm in ['main','predecessor','candidate']}, 'unknown provider process'
    assert rest[1].startswith('models/prepared/'), 'model symlink path differs'
    normalized=provenance['seed_models_symlink_target']+'/'+rest[1].removeprefix('models/')
    assert normalized==record['prepared_graph'], 'actual provider role/graph differs'
    assert provider['assigned_nodes']=={'CPUExecutionProvider':record['cpu_nodes']}, 'actual provider node count differs'
    inventory={e['path']:e for e in provenance['prepared_graphs_and_external_weights']}
    assert record['prepared_graph'] in inventory and record['external_data'] in inventory, 'prepared graph/external data absent'
    assert PurePosixPath(record['prepared_graph']).parent==PurePosixPath(record['external_data']).parent, 'external data directory differs'
    metadata=record['source_metadata'];b=metadata['text'].encode()
    observed=next(e for e in provenance['prepared_source_metadata'] if e['path']==metadata['path'])
    assert len(b)==observed['bytes'] and hashlib.sha256(b).hexdigest()==observed['sha256'], 'captured metadata not bound to historical inventory'
    lines=metadata['text'].splitlines();assert len(lines)==4 and lines[3].startswith('sha256 ')
    source=next(e for e in provenance['source_assets'] if e['path']==record['source_graph'])
    assert PurePosixPath(source['path']).name==lines[0] and source['bytes']==int(lines[1]) and source['sha256']==lines[3].removeprefix('sha256 '), 'prepared/source identity differs'
    assert int(lines[2])>0 and '/snapshots/'+record['revision']+'/' in source['path'], 'source revision/mtime identity differs'

def timing_review(summary,raw):
    timing_keys=['open_ms','write_ms','durability_flush_ms','maintenance_ms','maintenance_cpu_s','first_search_ms','warm_p50_ms','warm_p95_ms','episode']
    result={}
    for reference in ['predecessor','main']:
        result[reference]={}
        for key in timing_keys:
            if key=='episode':
                values=lambda arm:[r['wall_seconds'] for r in sorted((r for r in raw if r['arm']==arm and r['phase']=='process_total'),key=lambda r:r['repetition'])]
            else:
                values=lambda arm:[p[key] for p in sorted(summary['arms'][arm]['processes'],key=lambda p:p['repetition'])]
            a,b=values(reference),values('candidate')
            spread=lambda xs:(max(xs)-min(xs))/statistics.median(xs) if statistics.median(xs) else (0 if max(xs)==min(xs) else None)
            sa,sb=spread(a),spread(b);d=[y-x for x,y in zip(a,b)]
            reasons=[]
            if sa is None or sa>.10:reasons.append('reference spread >10% of median')
            if sb is None or sb>.10:reasons.append('candidate spread >10% of median')
            if min(d)<0<max(d):reasons.append('paired delta sign reversal')
            result[reference][key]={'before_process_values':a,'after_process_values':b,'reference_spread_fraction':sa,'candidate_spread_fraction':sb,'paired_deltas':d,'withhold_comparison':bool(reasons),'reasons':reasons}
    # Also retain a screen for each actual timed phase/query row, including
    # warmups and newly written-memory search that have no published table row.
    for reference in ['predecessor','main']:
        groups={}
        for r in raw:
            if r['arm'] not in [reference,'candidate'] or r.get('wall_ms') is None:continue
            key=r['phase']+('/query-'+str(r['extra']['query_document']) if 'query_document' in r.get('extra',{}) else '')
            groups.setdefault(key,{}).setdefault(r['arm'],[]).append(r)
        screens={}
        for key,arms in sorted(groups.items()):
            a=sorted(arms[reference],key=lambda r:r['repetition']);b=sorted(arms['candidate'],key=lambda r:r['repetition'])
            assert len(a)==len(b)==3 and [r['repetition'] for r in a]==[0,1,2], 'raw timing replica grouping differs'
            for metric in ['wall_ms','cpu_user_seconds','cpu_system_seconds']:
                av=[r[metric] for r in a];bv=[r[metric] for r in b]
                if any(v is None for v in av+bv):continue
                spread=lambda xs:(max(xs)-min(xs))/statistics.median(xs) if statistics.median(xs) else (0 if max(xs)==min(xs) else None)
                sa,sb=spread(av),spread(bv);d=[y-x for x,y in zip(av,bv)];reasons=[]
                if sa is None or sa>.10:reasons.append('reference spread >10% of median')
                if sb is None or sb>.10:reasons.append('candidate spread >10% of median')
                if min(d)<0<max(d):reasons.append('paired delta sign reversal')
                screens[key+'/'+metric]={'before_process_values':av,'after_process_values':bv,'reference_spread_fraction':sa,'candidate_spread_fraction':sb,'paired_deltas':d,'withhold_comparison':bool(reasons),'reasons':reasons}
        result[reference]['raw_call_timing']=screens
    return result


def historical_rank(value,expected):
    assert value is None or (type(value) is int and value>0), 'rank must be None or an actual positive integer (booleans refused)'
    assert value==expected, 'query rank differs from actual returned position'

def historical_hardware(provenance,readme):
    expected={'kernel':'Linux 6.18.44 x86_64 GNU/Linux','cpu_model':'AMD EPYC 9V74 80-Core Processor','cgroup_cpu_max':'400000 100000','cgroup_memory_max':'17179869184'}
    assert all(provenance.get(key)==value for key,value in expected.items()), 'retained hardware annotation differs'
    lines=[line for line in readme.splitlines() if line.startswith('- Linux ')]
    assert lines==['- Linux 6.18.44, AMD EPYC 9V74, cgroup 4 CPU cores and 16 GiB RAM. One workload process at a time; no builds or other model experiments during timing. The host is shared, so external interference is not controlled.'], 'retained hardware README annotation differs'

def seed_endpoint(provenance, record):
    """Late surviving-file endpoint identity; never historical per-copy proof."""
    entries=provenance['seed_files']
    assert record.get('exclusions')==['all symlinks','server.json','postmaster.pid','.pgpass','pgpass'] and provenance.get('credential_exclusions')==['server.json','pgpass'], 'seed inventory exclusion policy differs'
    seed=PurePosixPath('<SCRATCH>/seed-disk')
    for entry in entries:
        path=PurePosixPath(entry['path'])
        assert str(path)==entry['path'] and '..' not in path.parts and path.is_relative_to(seed) and path!=seed and path.name not in {'server.json','postmaster.pid','.pgpass','pgpass'}, 'seed inventory path outside recorded stopped-seed scope'
    inventory={entry['path']:[entry['bytes'],entry['sha256']] for entry in entries}
    assert len(inventory)==len(entries)
    encoded=json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()
    assert record['canonical_inventory_sha256']==hashlib.sha256(encoded).hexdigest()
    assert record['file_count']==len(entries)
    assert record['total_bytes']==sum(entry['bytes'] for entry in entries)
    assert record['matches_pretrial_inventory'] and record['seed_postgresql_stopped'] and record['metadata_stable_during_capture']
    assert record['seed_models_symlink_target']==provenance['seed_models_symlink_target']=='<MODEL_CACHE>'
    assert record['historical_per_copy_attested'] is False
    assert record['started_utc'] and record['finished_utc'] and record['started_utc']<=record['finished_utc']
    assert 'not historical per-copy attestation' in record['scope']
    assert record['canonical_encoding']=='UTF-8 JSON object path -> [bytes, sha256], sorted keys, separators comma/colon'

PREAD_FALLBACK="DiskAnn: no async I/O backend available: io_uring is unavailable and libaio could not be loaded. Enable io_uring or install libaio (e.g. 'apt-get install libaio1', or 'libaio1t64' on Ubuntu 24.04+) and retry. DiskAnn will use synchronous pread(); performance may be degraded."
def disk_backend(log):
    diagnostics=[line for line in log.splitlines() if re.search(r'io_uring|libaio|pread\(\)|async I/O|I/O backend|diskann_file_reader\.cc',line,re.I)]
    prefix=r"\[ WARN \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ diskann_file_reader\.cc:57\] "
    assert len(diagnostics)==1 and re.fullmatch(prefix+re.escape(PREAD_FALLBACK),diagnostics[0]), 'Disk native backend diagnostic must be exactly one synchronous pread fallback without conflicting backend evidence'

def successful_test_log(log):
    lines=[line for line in log.splitlines() if line.strip()]
    summaries=[line for line in lines if 'test result:' in line]
    pattern=r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in [0-9]+(?:\.[0-9]+)?s'
    assert len(summaries)==1 and re.fullmatch(pattern,summaries[0]) and lines[-1]==summaries[0], 'retained test log must end with exactly one complete successful one-test summary (zero filtered)'

def conversion_schema(before,after):
    # Complete canonical schema emitted by the historical helper. It did not
    # record HNSW degree/ef parameters; this check cannot certify absent values.
    text={'dimension':0,'dtype':2,'index_type':11,'metric':1,'nullable':False,'quantize':0,'quantizer_rotate':False}
    expected={'segment_documents':2000,'fields':{
        'id':{'dimension':0,'dtype':2,'nullable':False},
        'content_ngram':dict(text),'content_segmented':dict(text),
        'embedding':{'dimension':1024,'dtype':22,'index_type':1,'metric':3,'nullable':False,'quantize':0,'quantizer_rotate':False}}}
    assert before==expected, 'complete retained preconversion HNSW schema differs'
    assert set(after)==set(before) and after['segment_documents']==before['segment_documents'] and set(after['fields'])==set(before['fields']), 'conversion schema structure/segment parameters differ'
    for name in ['id','content_ngram','content_segmented']:
        assert after['fields'][name]==before['fields'][name], 'conversion changed nonembedding schema field'
    embedding=dict(before['fields']['embedding'],index_type=5,degree=64,build_list=100,pq_chunks=0)
    assert after['fields']['embedding']==embedding, 'complete retained postconversion DiskANN schema differs'
