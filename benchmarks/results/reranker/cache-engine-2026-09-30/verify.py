#!/usr/bin/env python3
"""Read-only archived Engine evidence verification; no Git, model, Cargo or DB."""
import collections,gzip,hashlib,io,json,os,statistics,sys,tarfile
from pathlib import Path
if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
 raise SystemExit('Refuse optimized Python: run python -B verify.py')
ROOT=Path(__file__).resolve().parent
EXPECTED={'binding/analyze.py.in.gz', 'raw/r0-refactor-q80-accurate-A.jsonl.gz', 'binding/refactor-source-members.json', 'raw/r1-fixed-q101-accurate-B-providers.json', 'raw/r0-fixed-q101-accurate-B.log.gz', 'raw/r1-refactor-q101-accurate-B.jsonl.gz', 'raw/r0-refactor-q80-accurate-A-providers.json', 'binding/run-controller.log.gz', 'raw/provenance-after.json', 'raw/r0-refactor-q101-accurate-B.jsonl.gz', 'raw/r0-fixed-q80-accurate-A-providers.json', 'raw/r0-fixed-q101-accurate-B-launch.json', 'raw/r0-fixed-q80-accurate-A.log.gz', 'raw/r0-fixed-q101-accurate-A-graph.json', 'raw/r1-fixed-q80-accurate-B-providers.json', 'raw/r1-fixed-q80-accurate-A-providers.json', 'raw/r1-refactor-q101-accurate-A.log.gz', 'raw/r0-fixed-q80-accurate-A-launch.json', 'binding/refactor-pamin-engine-build.log.gz', 'raw/r1-refactor-q101-accurate-A-launch.json', 'binding/run.py.in.gz', 'raw/r0-refactor-q80-accurate-A.log.gz', 'raw/r0-refactor-q80-accurate-B-providers.json', 'raw/r1-refactor-q80-accurate-B.log.gz', 'raw/r1-refactor-q80-accurate-A.jsonl.gz', 'raw/r1-fixed-q101-accurate-A.log.gz', 'raw/r0-refactor-q101-accurate-A-launch.json', 'raw/r1-fixed-q80-accurate-B-graph.json', 'raw/r0-fixed-q80-off-A.log.gz', 'raw/r0-fixed-q101-accurate-A.jsonl.gz', 'verify.py', 'binding/refactor-pamin-engine-build.jsonl.gz', 'raw/r0-fixed-q80-accurate-B-providers.json', 'raw/r0-refactor-q101-accurate-B.log.gz', 'raw/r1-fixed-q101-accurate-A.jsonl.gz', 'raw/r0-refactor-q101-accurate-A-graph.json', 'raw/r1-fixed-q101-accurate-B-launch.json', 'raw/r0-fixed-q80-off-A.jsonl.gz', 'raw/r1-fixed-q101-accurate-A-graph.json', 'raw/r0-fixed-q101-accurate-A-providers.json', 'raw/r1-fixed-q101-accurate-B.jsonl.gz', 'binding/fixed-source.tar.gz', 'raw/r1-fixed-q80-accurate-A-graph.json', 'raw/r0-refactor-q80-accurate-A-launch.json', 'raw/r1-fixed-q80-accurate-A.log.gz', 'raw/r0-refactor-q101-accurate-A-providers.json', 'binding/refactor-clean.log.gz', 'binding/build.py.in.gz', 'raw/r1-refactor-q101-accurate-B-graph.json', 'raw/r0-refactor-q80-off-A.log.gz', 'raw/r0-refactor-q101-accurate-B-launch.json', 'raw/r0-fixed-q101-accurate-A-launch.json', 'raw/provenance-before.json', 'binding/fixed-source-members.json', 'resource-comparisons.md', 'raw/r0-fixed-q80-accurate-A.jsonl.gz', 'raw/r0-refactor-q101-accurate-B-graph.json', 'binding/refactor-source.tar.gz', 'raw/cost-observations.json', 'raw/r1-refactor-q80-accurate-A.log.gz', 'raw/r0-refactor-q80-accurate-B.log.gz', 'raw/r1-refactor-q101-accurate-A-providers.json', 'raw/r1-refactor-q101-accurate-A-graph.json', 'raw/r0-refactor-q101-accurate-B-providers.json', 'raw/r0-refactor-q80-off-A-graph.json', 'binding/refactor-frozen.json', 'raw/r1-refactor-q80-accurate-B-providers.json', 'raw/r0-fixed-q101-accurate-A.log.gz', 'sanitization.json', 'raw/r1-refactor-q80-accurate-A-providers.json', 'raw/r1-fixed-q80-accurate-A-launch.json', 'raw/r0-refactor-q80-off-A-launch.json', 'raw/r0-fixed-q80-accurate-B.log.gz', 'raw/r1-refactor-q101-accurate-A.jsonl.gz', 'raw/r0-fixed-q80-accurate-B-launch.json', 'raw/r1-fixed-q80-accurate-B-launch.json', 'raw/r0-fixed-q80-accurate-A-graph.json', 'raw/r1-fixed-q80-accurate-A.jsonl.gz', 'raw/r0-refactor-q101-accurate-A.log.gz', 'README.md', 'binding/build-controller.log.gz', 'raw/r1-refactor-q101-accurate-B-providers.json', 'raw/r0-refactor-q101-accurate-A.jsonl.gz', 'raw/r0-fixed-q80-off-A-launch.json', 'binding/fixed-frozen.json', 'raw/r0-fixed-q101-accurate-B-providers.json', 'raw/r1-fixed-q80-accurate-B.jsonl.gz', 'raw/r0-fixed-q80-off-A-providers.json', 'raw/r0-refactor-q80-accurate-A-graph.json', 'raw/summary.json', 'raw/r1-refactor-q80-accurate-A-launch.json', 'raw/r1-refactor-q80-accurate-B-graph.json', 'raw/r0-fixed-q80-accurate-B.jsonl.gz', 'raw/r1-refactor-q80-accurate-B.jsonl.gz', 'binding/fixed-pamin-engine-build.jsonl.gz', 'raw/r0-fixed-q80-off-A-graph.json', 'raw/r0-fixed-q101-accurate-B-graph.json', 'binding/frozen-variants.json', 'raw/r0-fixed-q80-accurate-B-graph.json', 'raw/r1-fixed-q101-accurate-B.log.gz', 'raw/r0-refactor-q80-accurate-B-graph.json', 'raw/r0-refactor-q80-accurate-B-launch.json', 'raw/r1-fixed-q101-accurate-B-graph.json', 'raw/r0-refactor-q80-accurate-B.jsonl.gz', 'raw/r1-refactor-q101-accurate-B-launch.json', 'raw/r1-refactor-q80-accurate-B-launch.json', 'raw/r1-refactor-q80-accurate-A-graph.json', 'negative_fixtures.py', 'raw/r1-fixed-q101-accurate-A-launch.json', 'raw/r0-refactor-q80-off-A-providers.json', 'raw/r0-fixed-q101-accurate-B.jsonl.gz', 'raw/r1-fixed-q80-accurate-B.log.gz', 'binding/probe.rs.append.in.gz', 'binding/fixed-pamin-engine-build.log.gz', 'raw/r1-refactor-q101-accurate-B.log.gz', 'raw/r0-refactor-q80-off-A.jsonl.gz', 'raw/r1-fixed-q101-accurate-A-providers.json', 'binding/prepare.py.in.gz', 'binding/fixed-clean.log.gz'}
FIELDS=['offered','scored','characters','tokens','padded_tokens','batches','encode_us','forward_us']
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(name):
 b=(ROOT/name).read_bytes();return gzip.decompress(b) if name.endswith('.gz') else b
def obj(name):return json.loads(read(name))
def original(b,m):
 for old,new in m['prefixes'].items():b=b.replace(new.encode(),old.encode())
 return b

def verify(hashes=True):
 manifest=obj('manifest.json');actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
 need(set(manifest['files'])==EXPECTED and actual==EXPECTED|{'manifest.json'},'archive member set mismatch')
 if hashes:
  for name,h in manifest['files'].items():need(sha((ROOT/name).read_bytes())==h,'archive hash '+name)
 sanit=obj('sanitization.json')
 if hashes:
  for n,r in sanit['files'].items():
   b=read(n);need(sha(b)==r['payload_sha256'] and sha(original(b,sanit))==r['original_sha256'],'sanitization binding '+n)
 before=obj('raw/provenance-before.json');after=obj('raw/provenance-after.json');variants=obj('binding/frozen-variants.json')
 need(before['variants']==variants,'launch variant bindings')
 need(before['assets']==after['assets'] and before['seed']==after['seed'] and after['all_owned_pg_stopped'],'post-run asset/seed guard')
 need(sha(original(read('binding/run.py.in.gz'),sanit))==before['runner_sha256'],'runner binding')
 for v in variants:
  arm=v['arm'];need(v==obj('binding/'+arm+'-frozen.json'),'variant report')
  need(sha(json.dumps(v['source_files'],sort_keys=True).encode())==v['source_sha256'],'complete source inventory binding')
  members=obj('binding/'+arm+'-source-members.json')
  with tarfile.open(fileobj=io.BytesIO(read('binding/'+arm+'-source.tar.gz')),mode='r:') as tar:
   need(set(tar.getnames())==set(members),'source archive set')
   for member in tar:
    b=tar.extractfile(member).read();r=members[member.name]
    need(sha(b)==r['payload_sha256'] and sha(original(b,sanit))==r['original_sha256']==v['source_files'][member.name],'source binding '+member.name)
   need(sha(original(tar.extractfile('crates/pamin-index/src/reranking.rs').read(),sanit))==v['product_reranking_sha256'],'production reranking binding')
   queries=json.loads(tar.extractfile('crates/pamin-engine/tests/corpus/queries.json').read())
   corpus=json.loads(tar.extractfile('crates/pamin-engine/tests/corpus/memories.json').read())
   need(len(queries)==157 and len(corpus)==230,'public corpus binding')
   probe=tar.extractfile('crates/pamin-engine/tests/scratch_cache_engine.rs').read()
   base=tar.extractfile('crates/pamin-engine/tests/retrieval.rs').read()
   need(probe==base+b'\n'+read('binding/probe.rs.append.in.gz'),'exact appended harness binding')
  messages=[json.loads(l) for l in read('binding/'+arm+'-pamin-engine-build.jsonl.gz').splitlines() if l.startswith(b'{')]
  artifacts=[r for r in messages if r.get('reason')=='compiler-artifact'];binary=v['pamin-engine']
  need(binary['compiler_artifact'] in artifacts and not binary['compiler_artifact']['fresh'],'fresh measured binary emission')
  for c in ['pamin-core','pamin-store','pamin-index','pamin-engine']:
   need(any(a['target']['name']==c.replace('-','_') and not a['fresh'] and v['source'] in a['manifest_path'] for a in artifacts),'fresh path crate '+c)
 rows_by={};count=hot=0
 for name in sorted(n for n in manifest['files'] if n.startswith('raw/r') and n.endswith('.jsonl.gz')):
  process=Path(name).name.removesuffix('.jsonl.gz');rows=[json.loads(s) for s in read(name).splitlines()]
  log=read('raw/'+process+'.log.gz').decode();native=[json.loads(l.split('CACHE_ENGINE_ROW ',1)[1]) for l in log.splitlines() if 'CACHE_ENGINE_ROW ' in l]
  need('test result: ok. 1 passed;' in log and len(rows)==len(native),'native process success/rows')
  launch=obj('raw/'+process+'-launch.json');v=next(v for v in variants if '-'+v['arm']+'-' in process)
  need(launch['binary_sha256']==v['pamin-engine']['sha256'] and launch['command'][0]==v['pamin-engine']['binary'],'launch binary identity')
  env=launch['environment'];need({k:env[k] for k in env if k.startswith('PAMIN_')}=={'PAMIN_DEVICE':'cpu','PAMIN_PROFILE':'accuracy','PAMIN_EVAL_HOME':'${ENGINE_EXPERIMENT}/results/'+process+'-home'},'explicit PAMIN knobs')
  need(not any(k.startswith('ORT_') and k not in ['ORT_LIB_LOCATION','ORT_PREFER_DYNAMIC_LINK'] or k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','HF_TOKEN'] for k in env),'inherited tuning/credentials')
  sequence=env['CACHE_SEQUENCE'].split(',');need(len(rows)==len(sequence),'sequence length');need(env['CACHE_TIER'] in ['accurate','off'],'tier')
  providers=obj('raw/'+process+'-providers.json');need(len(providers)==(2 if env['CACHE_TIER']=='accurate' else 1),'provider count')
  native_providers=[];decoder=json.JSONDecoder()
  for line in log.splitlines():
   if 'ONNX graph execution-provider assignment' in line:
    graph,_=decoder.raw_decode(line.split('model_graph=',1)[1]);graph='${MODEL_CACHE}/'+graph.split('/models/',1)[1] if '/models/' in graph else graph;nodes,_=decoder.raw_decode(line.split('assigned_nodes=',1)[1]);native_providers.append(dict(graph=graph,nodes=nodes,sha256=before['assets'][graph]['sha256']))
  need(providers==sorted(native_providers,key=lambda p:p['graph']),'raw provider assignments')
  for p in providers:need(p['nodes'] in [{'CPUExecutionProvider':1023},{'CPUExecutionProvider':295}] and before['assets'][p['graph']]['sha256']==p['sha256'],'actual provider asset')
  need(obj('raw/'+process+'-graph.json')==[ ['mentions',11] ],'seed graph topology')
  for i,(r,n) in enumerate(zip(rows,native)):
   d={k:r['after'][k]-r['before'][k] for k in FIELDS};need(r['actual_delta']==d and all(x>=0 for x in d.values()),'actual counters')
   need({k:x for k,x in r.items() if k!='actual_delta'}==n,'raw log/jsonl equality')
   need(r['step']==i and r['arm']==sequence[i] and r['tier']==env['CACHE_TIER'],'actual call order/config')
   q=int(env['CACHE_QUERY_ID']);need(r['query_id']==(q+1 if r['arm']=='N' else q) and r['query']==queries[r['query_id']]['query'],'query identity')
   need(r['documents']==230 and r['coverage']==0 and len(r['fused'])==len(r['complete']),'actual index/complete result')
   offered=[r['fused'][at]['topic_id'] for at in r['selected_fused_positions']];scores={h['topic_id'] for h in r['complete'] if h['reranked_bits']}
   need(set(offered)==scores and len(offered)==d['offered'] and d['scored']<=d['offered'],'actual offered identity')
   need(d['offered']==(30 if r['tier']=='accurate' else 0),'30 actual budget')
   need(sum(any(w.get('kind')=='channel' and w['channel']=='vector' for w in h['why']) for h in r['fused'])==50,'native vector candidates')
   need(any('${ORT_RUNTIME}/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0' in s for s in r['loaded_libraries']),'loaded runtime')
   if i==0 or r['arm']=='N':need(d['scored']==d['offered'],'fresh/new actual work')
   if r['tier']=='off':need(d['scored']==d['batches']==d['forward_us']==0,'off control')
   if i>0 and r['arm']==rows[i-1]['arm']:
    need(r['fused']==rows[i-1]['fused'] and r['complete']==rows[i-1]['complete'],'hot bits/order')
    need(all(d[k]==0 for k in ['scored','tokens','padded_tokens','batches','forward_us']),'hot forward work')
    if r['tier']=='accurate':hot+=1
  rows_by[process]=rows;count+=len(rows)
 need(len(rows_by)==18 and count==214 and hot==160,'18 processes/214 calls/160 hot calls')
 comparisons=[]
 for repeat in [0,1]:
  for variant in ['refactor','fixed']:
   for q in [80,101]:
    for initial,final in [('A','B'),('B','A')]:
     rows=rows_by[f'r{repeat}-{variant}-q{q}-accurate-{initial}'];warm=rows[6];fresh=rows_by[f'r{repeat}-{variant}-q{q}-accurate-{final}'][0]
     need(warm['fused']==fresh['fused'],'same final fused list')
     def scores(r):return {h['topic_id']:h['reranked_bits'] for h in r['complete'] if h['reranked_bits']}
     f,w=scores(fresh),scores(warm);need(set(f)==set(w),'same final offered IDs')
     changed=sum(f[k]!=w[k] for k in f);positions=sum(a['topic_id']!=b['topic_id'] for a,b in zip(fresh['complete'],warm['complete']))
     result_bits=sum(a['score_bits']!=b['score_bits'] for a,b in zip(sorted(fresh['complete'],key=lambda x:x['topic_id']),sorted(warm['complete'],key=lambda x:x['topic_id'])))
     need((changed,positions,result_bits)==((20,25,0) if variant=='refactor' and q==101 else (0,0,0)),'expected actual history result')
     comparisons.append(dict(repeat=repeat,variant=variant,query_id=q,history=initial+'->'+final,changed_logits=changed,changed_product_positions=positions,changed_product_result_score_bits=result_bits,actual_delta=warm['actual_delta'],selected_changed=rows[0]['selected_fused_positions']!=warm['selected_fused_positions'] or [rows[0]['fused'][i]['topic_id'] for i in rows[0]['selected_fused_positions']]!=[warm['fused'][i]['topic_id'] for i in warm['selected_fused_positions']]))
 for repeat in [0,1]:
  for q in [80,101]:
   for arm in ['A','B']:
    a,b=[rows_by[f'r{repeat}-{v}-q{q}-accurate-{arm}'] for v in ['refactor','fixed']]
    for i in [0,12]:need(a[i]['fused']==b[i]['fused'] and a[i]['complete']==b[i]['complete'],'fresh/new cross-variant oracle')
 summary=obj('raw/summary.json');need(comparisons==summary['comparisons'] and all(summary[k] is True for k in ['fresh_product_bits_equal','hot_actual_forward_zero','source_binary_assets_unchanged','pg_all_stopped']),'summary derived from raw')
 # Recompute every resource observation, including cold, changed, off/new and hot.
 groups=collections.defaultdict(list);blocks=[]
 for process,rows in rows_by.items():
  variant='refactor' if '-refactor-' in process else 'fixed'
  for r in rows:
   phase='off' if r['tier']=='off' else 'cold' if r['step']==0 else 'new-query' if r['arm']=='N' else 'changed-call' if r['step']==6 and r['query_id']==101 else 'unchanged-call' if r['step']==6 else 'hot'
   groups[(variant,r['query_id'],r['arm'],phase)].append((process,r))
  for arm in ['A','B']:
   hs=[r for r in rows if r['tier']=='accurate' and r['arm']==arm and r['step'] not in [0,6]]
   if hs:blocks.append(dict(variant=variant,process=process,query_id=hs[0]['query_id'],arm=arm,calls=len(hs),median_wall_us=statistics.median(r['wall_us'] for r in hs),median_encode_us=statistics.median(r['actual_delta']['encode_us'] for r in hs)))
 observations=[]
 for (variant,q,arm,phase),prs in sorted(groups.items()):
  rs=[r for _,r in prs];observations.append(dict(variant=variant,query_id=q,arm=arm,phase=phase,calls=len(rs),processes=len({p for p,_ in prs}),median_wall_us=statistics.median(r['wall_us'] for r in rs),min_wall_us=min(r['wall_us'] for r in rs),max_wall_us=max(r['wall_us'] for r in rs),total_process_user_ticks=sum(r['process_after']['utime_ticks']-r['process_before']['utime_ticks'] for r in rs),total_process_system_ticks=sum(r['process_after']['stime_ticks']-r['process_before']['stime_ticks'] for r in rs),median_encode_us=statistics.median(r['actual_delta']['encode_us'] for r in rs),fresh_scores=sum(r['actual_delta']['scored'] for r in rs),actual_offered=sum(r['actual_delta']['offered'] for r in rs),max_process_hwm_kib=max(r['process_after']['hwm_kib'] for r in rs)))
 cost=obj('raw/cost-observations.json');need(observations==cost['rows'] and blocks==cost['hot_process_blocks'],'all resource observations derived from raw')
 print('PASS: 18 processes, 214 calls, 160 hot calls; 16 history comparisons; fresh/new oracles; complete resource/source/assets/provider/config bindings')
 return count
if __name__=='__main__':verify()
