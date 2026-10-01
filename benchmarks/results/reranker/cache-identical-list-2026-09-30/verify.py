#!/usr/bin/env python3
"""Read-only verifier for sanitized inert evidence. Never runs native/build/PG/models."""
import sys
if sys.flags.optimize:raise SystemExit('Evidence verifier refuses -O/PYTHONOPTIMIZE')
import argparse,copy,gzip,hashlib,json,math,re,statistics,struct
from pathlib import Path
KEYS=['offered','scored','characters','tokens','padded_tokens','batches','encode_us','forward_us']
ZERO=['scored','characters','tokens','padded_tokens','batches','forward_us']
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(x):return hashlib.sha256(x).hexdigest()
def jsonfile(path):return json.loads(path.read_text())
def delta(r,a='before',b='after'):return {k:r[b][k]-r[a][k] for k in KEYS}
def selected(r):return [r['fused'][p]['topic_id'] for p in r['selected_fused_positions']]
def typed_hits(hits):
 data=copy.deepcopy(hits)
 for position,h in enumerate(data,1):
  require(h['rank']==position,'one-based rank positions')
  require(isinstance(h['score_bits'],int) and 0<=h['score_bits']<2**32,'invalid fusion f32 bits')
  bits=[];fused=0.0
  for w in h['why']:
   fields=['score','weight','contribution'] if w['kind']=='channel' else ['score'] if w['kind']=='reranked' else []
   if w['kind']=='channel':
    contribution=struct.unpack('<f',struct.pack('<f',w['contribution']))[0];fused=struct.unpack('<f',struct.pack('<f',fused+contribution))[0]
   for k in fields:
    if w[k] is not None:
     require(isinstance(w[k],(int,float)) and math.isfinite(w[k]),'nonfinite source-declared f32')
     raw=struct.unpack('<I',struct.pack('<f',w[k]))[0]
     if w['kind']=='reranked':bits.append(raw)
     w[k]={'declared_f32_bits':raw}
  require(bits==h['reranked_bits'],'native raw Why::Reranked and bit-array disagree')
  require(struct.unpack('<I',struct.pack('<f',fused))[0]==h['score_bits'],'fusion Why contributions and f32 scorebits disagree')
 return json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False)
def percentile(xs,p):
 xs=sorted(xs);at=(len(xs)-1)*p;lo=math.floor(at);hi=math.ceil(at);return xs[lo]+(xs[hi]-xs[lo])*(at-lo)
def describe(xs):return {'n':len(xs),'median':statistics.median(xs),'p95_linear_sample':percentile(xs,.95),'min':min(xs),'max':max(xs)}
def jobs():
 result=[]
 for rep,order in enumerate([['baseline','memo'],['memo','baseline']]):
  for arm in order:
   for limit in [5,10]:
    for q in [80,101]:
     for initial in ['A','B']:result.append((f'r{rep}-{arm}-l{limit}-q{q}-accurate-{initial}',rep,arm,limit,q,'accurate',initial))
 for arm in ['baseline','memo']:
  for limit in [5,10]:result.append((f'r0-{arm}-l{limit}-q80-off-A',0,arm,limit,80,'off','A'))
 return result

def verify(root):
 root=Path(root);manifest=jsonfile(root/'manifest.json')
 actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
 require(actual==set(manifest),'archive file set changed')
 for name,record in manifest.items():
  data=(root/name).read_bytes();require(len(data)==record['bytes'] and sha(data)==record['sha256'],'archive checksum '+name)
 receipt=jsonfile(root/'receipt.json');cost=jsonfile(root/'cost-and-gold.json');summary=jsonfile(root/'acceptance-summary.json')
 require(receipt['seed']['documents']==230 and receipt['seed']['coverage']==0.0 and receipt['seed']['nativegraph_mentions']==11,'native seed premise')
 require(receipt['seed']['full_inventory_before_digest']==receipt['seed']['full_inventory_after_digest'],'original seed mutated')
 require(receipt['seed']['index_disk_bytes']==22028682,'seed index bytes')
 require(len(receipt['processes'])==36 and cost['processes']==36 and cost['timed_calls']==428,'process/call totals')
 require(summary['36_processes_complete'] and summary['428_timed_calls'] and summary['all_owned_pg_stopped'] and summary['full300_diagnostics_outside_clock'],'acceptance scope')
 require(set(b['variant'] for b in receipt['source_bindings'])=={'baseline','memo'},'two source arms')
 frozen={'baseline':('2f5bd087ceb4b65a25dc94932e416d8ac5c29740','03a7f804bd8c4c0c319083e2e51cef8072fe4634f6b2ebab07632aa69c3e1d20'),'memo':('d0b6a14a4f317fea3c1e117f627289aad1b1c964','8da6eac4e2014b74fc4a05a09938f2bf55f40986f0f9fc4dfa629a2d083abb04')}
 for b in receipt['source_bindings']:
  require((b['measured_commit'],b['binary_sha256'])==frozen[b['variant']],'exact frozen source/binary identity')
  require(b['tracked_plus_harness_inventory_before_digest']==b['tracked_plus_harness_inventory_after_digest'],'compiled inputs changed')
  require(b['all4_and_helper_fresh'] and set(a['target'] for a in b['artifacts'])=={'pamin_core','pamin_store','pamin_index','pamin_engine','scratch_cache_product_limits'},'fresh four productlibs/helper')
  require(all(a['fresh'] is False and a['source_path'].startswith('<SOURCE:'+b['variant']+'>/') and a['manifest_path'].startswith('<SOURCE:'+b['variant']+'>/') for a in b['artifacts']),'fresh source path')
  require(sha((root/'sources'/f'{b["variant"]}-reranking.rs').read_bytes())==b['reranking_source_sha256'],'measured source copy binding')
  require(b['native_harness_sha256']=='1473c07b397282d3a5bc833b2e8d3a94e824703a4219012251d2cd80af027b50','exact historical native harness identity')
 require({p.name for p in (root/'sources').iterdir() if p.is_file()}=={'baseline-reranking.rs','memo-reranking.rs','fusion.rs','memories.json','queries.json'},'public product source/data scope without measurement drivers')
 require(receipt['public_source_scope_qualification']=={'policy': 'Number-producing measurement drivers remain private scratch artifacts; public verifier only recalculates retained evidence.', 'private_preservation': 'Whole pre-cleanup evidence bundle retained byte-for-byte with checked manifest and restricted permissions; no private paths published.', 'historical_native_harness_sha256': '1473c07b397282d3a5bc833b2e8d3a94e824703a4219012251d2cd80af027b50', 'native_harness_binding_scope': 'Historical recorded identity only; compiled measurement harness and execution/build/PG controllers are not publicly inspectable. Existing historical build freshness/input checks are retained records, not a complete public source-to-binary attestation.', 'retained_public_sources': ['baseline-reranking.rs', 'memo-reranking.rs', 'fusion.rs', 'memories.json', 'queries.json'], 'original_retrieval_input': {'repository_path': 'crates/pamin-engine/tests/retrieval.rs', 'commits': ['2f5bd087ceb4b65a25dc94932e416d8ac5c29740', 'd0b6a14a4f317fea3c1e117f627289aad1b1c964'], 'git_blob_sha1': '40a27403e0d9e53ec58dd1770efcc1648e04accb', 'sha256': '1e0dd4e1f6d137ebf3ca80c3eca2ab3a0313beafbc84e43e10b1e63f7ab0b73f', 'scope': 'Byte-identical original input at both frozen commits; modified compiled target is privately retained.'}},'exact public/private source qualification and immutable Git references')
 fusion=(root/'sources/fusion.rs').read_text();require('score: Option<f32>' in fusion and 'weight: f32' in fusion and 'contribution: f32' in fusion and 'score: f32' in fusion,'declared Why field types')
 canonical=lambda z:json.dumps(z,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
 require(sha(canonical(receipt['assets']))==receipt['assets_before_inventory_digest']==receipt['assets_after_inventory_digest'],'recorded endpoint asset fingerprints')
 required_roles={
  'embedding':{'graph':'<MODEL_CACHE>/prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx','external_weights':['<MODEL_CACHE>/prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx.data'],'tokenizer':'<MODEL_CACHE>/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/tokenizer.json'},
  'accurate':{'graph':'<MODEL_CACHE>/prepared/eecdcf109c0c08402aa8f893fc25d2d4/attention.onnx','prepared_base_graph':'<MODEL_CACHE>/prepared/eecdcf109c0c08402aa8f893fc25d2d4/model.onnx','external_weights':['<MODEL_CACHE>/prepared/eecdcf109c0c08402aa8f893fc25d2d4/model.onnx.data'],'tokenizer':'<MODEL_CACHE>/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/tokenizer.json'}}
 require(receipt['asset_roles']==required_roles,'complete executed model asset roles')
 for role in required_roles.values():
  paths=[role['graph'],role['tokenizer'],*role['external_weights']]+([role['prepared_base_graph']] if 'prepared_base_graph' in role else [])
  require(all(p in receipt['assets'] and receipt['assets'][p]['bytes']>0 and re.fullmatch('[0-9a-f]{64}',receipt['assets'][p]['sha256']) for p in paths),'mandatory graph/weights/tokenizer coverage')
 controls=receipt['runtime_controls'];require(controls['device']=='cpu' and controls['profile']=='accuracy' and controls['ORT']=='1.28.0' and controls['model_precision']=='CPU INT8' and controls['pair_budget']==30 and controls['return_limits']==[5,10] and controls['diagnostic_limit']==300,'runtime controls')
 require(receipt['model_source_export']['state'].startswith('absent throughout'),'source-export scope')
 required_libraries={'<WORKSPACE>/.onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0','<WORKSPACE>/.cargo/lib/pamin/libzvec_c_api.so'}
 require(required_libraries<=set(receipt['assets']),'mandatory native libraries unbound')
 require(receipt['loader']['path'] in receipt['assets'] and receipt['assets'][receipt['loader']['path']]['sha256']==receipt['loader']['sha256'],'loader asset binding')
 cfg=receipt['build_config'];require(cfg['RUSTC_WRAPPER']==cfg['RUSTC_WORKSPACE_WRAPPER']==cfg['CARGO_ENCODED_RUSTFLAGS']==cfg['CARGO_ENCODED_RUSTDOCFLAGS']=='','compiler flags/wrappers controlled')
 for name in ['cargo','rustc','rustdoc']:require('1.98.1' in receipt['compiler'][name] and receipt['tools'][name]['sha256'],'pinned compiler')
 gold=jsonfile(root/'sources/queries.json');mem=jsonfile(root/'sources/memories.json');require(len(gold)==157 and len(mem)==230,'native corpus cardinality');topics={m['topic'] for m in mem}
 capture=receipt['capture_provenance'];require(capture['series_label']=='2026-09-30' and capture['public_export_date_utc']=='2026-10-01' and capture['public_export_time_utc']=='unknown','series/export date scope')
 processes={p['name']:p for p in receipt['processes']};require(len(processes)==36 and set(processes)=={j[0] for j in jobs()},'exact registered process matrix')
 rowsby={};groups={};calls=[];quality=[];usage=[]
 for name,rep,arm,limit,q,tier,initial in jobs():
  p=processes[name];require(p['variant']==arm and p['returncode']==0 and p['fullyvalidated_stopped_clone_disposed'],'process success/disposal')
  require(p['actual_loader_mapping_verified'],'actual native loader mapping')
  pg=p['pg'];require(pg['data_directory_verified_before_native_fixture'] and pg['stop'].get('process_pidfile_and_port_gone'),'authenticated before Engine/owned stop proof')
  require(pg['identity']['data_directory'].endswith(name+'-home/postgres/data') and pg['identity']['postgres_executable'].startswith('<TRIALS>/'+name+'-home/postgres/install/'),'owned data/executable')
  require(pg['identity']['pid']==pg['stop']['owned_pid'],'same PID stopped')
  env=p['environment'];require(env['CACHE_RETURN_LIMIT']==str(limit) and env['CACHE_QUERY_ID']==str(q) and env['CACHE_TIER']==tier and env['PAMIN_DEVICE']=='cpu' and env['PAMIN_PROFILE']=='accuracy','effective requested configuration')
  require(not any(k in env for k in ['LD_PRELOAD','LD_AUDIT','PGSERVICE','PGHOST','PGPASSWORD','XDG_CACHE_HOME']),'inherited runtime overrides absent')
  require({k for k in env if k.startswith('PAMIN_')}=={'PAMIN_DEVICE','PAMIN_PROFILE','PAMIN_EVAL_HOME'},'unexpected product knob')
  require(all(env[k]=='<MODEL_CACHE>' for k in ['HF_HOME','HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE']) and env['HF_HUB_OFFLINE']=='1','pinned offline cache')
  require(p['minimum_sample_disk_free']>=8*1024**3,'8GiB live disk reserve')
  for event in ['oom','oom_kill']:
   parse=lambda s:dict(l.split() for l in s.splitlines())
   require(parse(p['hardware_before']['cgroup']['memory.events']).get(event,'0')==parse(p['hardware_after']['cgroup']['memory.events']).get(event,'0')=='0','OOM event')
  require(p['hardware_before']['affinity']==p['hardware_after']['affinity'] and p['hardware_before']['cgroup']['cpu.max']==p['hardware_after']['cgroup']['cpu.max']=='400000 100000\n','CPU quota/affinity')
  index=p['index'];require(index['file_count']==46 and index['before_inventory_digest']==receipt['seed']['index_inventory_digest'],'matched original index inventory')
  require(index['changed_files']==[{'name':f'embedding.index.{n}.proxima','before_bytes':5234688,'after_bytes':5234688} for n in [2,4,6,8]],'four changed file sizes')
  require(index['before_bytes']==index['after_bytes']==receipt['seed']['index_disk_bytes'],'index disk total')
  require(index['before_inventory_digest']!=index['after_inventory_digest'] and index['changed_file_names']==[f'embedding.index.{n}.proxima' for n in [2,4,6,8]],'preserved unexplained four-file mutation')
  text=gzip.decompress((root/'raw'/(name+'.log.gz')).read_bytes()).decode()
  timestamps=re.findall(r'(?m)^(2026-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z)',text)
  require(capture['processes'][name]=={'first_recorded_log_event_utc':min(timestamps) if timestamps else 'unknown','last_recorded_log_event_utc':max(timestamps) if timestamps else 'unknown','process_start_utc':'unknown','process_end_utc':'unknown'},'recorded event timestamp provenance')
  require('/workspace' not in text and '/home/agent' not in text and not re.search(r'(?:password\s*[=:]|postgres(?:ql)?://|authorization\s*:|token=|session[_-]?url)',text,re.I),'public sanitization')
  require('test result: ok. 1 passed;' in text,'native test completion')
  decoder=json.JSONDecoder();providers=[]
  for line in text.splitlines():
   if 'ONNX graph execution-provider assignment' in line:
    graph,_=decoder.raw_decode(line.split('model_graph=',1)[1]);nodes,_=decoder.raw_decode(line.split('assigned_nodes=',1)[1]);require(set(nodes)=={'CPUExecutionProvider'} and nodes['CPUExecutionProvider']>0,'actual node providers');require(p['models_symlink_target']=='<MODEL_CACHE>' and graph.startswith('<TRIALS>/'+name+'-home/models/'),'owned graph path through pinned models symlink');resolved_graph=graph.replace('<TRIALS>/'+name+'-home/models/','<MODEL_CACHE>/',1);require(resolved_graph in receipt['assets'],'selected graph asset binding');expected_nodes={required_roles['embedding']['graph']:1023};expected_nodes.update({required_roles['accurate']['graph']:295} if tier=='accurate' else {});require(resolved_graph in expected_nodes and nodes=={'CPUExecutionProvider':expected_nodes[resolved_graph]},'executed graph role/nodecount binding');providers.append((graph,nodes))
  require(len(providers)==(2 if tier=='accurate' else 1) and len({graph for graph,_ in providers})==len(providers),'actual selected graph cardinality')
  require(sorted(n['CPUExecutionProvider'] for _,n in providers)==([295,1023] if tier=='accurate' else [1023]),'known CPU node counts')
  graph_rows=[json.loads(l.split('CACHE_ENGINE_GRAPH ',1)[1]) for l in text.splitlines() if 'CACHE_ENGINE_GRAPH ' in l];require(len(graph_rows)==1 and sum(n for _,n in graph_rows[0])==11,'actual native graph11')
  rows=[json.loads(l.split('CACHE_ENGINE_ROW ',1)[1]) for l in text.splitlines() if 'CACHE_ENGINE_ROW ' in l];rowsby[name]=rows
  sequence=env['CACHE_SEQUENCE'].split(',');expected_sequence=([initial]*6+(['B'] if initial=='A' else ['A'])*6+['N']) if tier=='accurate' else ['A','A','N'];require(sequence==expected_sequence and len(rows)==len(sequence)==p['rows'],'actual history length')
  for step,(mode,r) in enumerate(zip(sequence,rows)):
   require(r['step']==step and r['arm']==mode and r['query_id']==(q+1 if mode=='N' else q) and r['return_limit']==limit and r['tier']==tier,'exact matrix row')
   require(r['query']==gold[r['query_id']]['query'],'public native query')
   require(type(r['wall_us']) is int and r['wall_us']>0,'positive integer timed wall cost')
   require(all(type(r[s][k]) is int and r[s][k]>=0 for s in ['process_before','process_after'] for k in ['utime_ticks','stime_ticks']) and all(r['process_after'][k]>=r['process_before'][k] for k in ['utime_ticks','stime_ticks']),'unsigned monotonic process CPU ticks')
   require(all(type(r[s][k]) is int and r[s][k]>0 for s in ['process_before','process_after'] for k in ['rss_kib','hwm_kib']) and all(r[s]['hwm_kib']>=r[s]['rss_kib'] for s in ['process_before','process_after']),'positive process RSS/HWM scope')
   require(r['documents']==230 and r['coverage']==0.0,'matched flat corpus premise')
   for field in ['fused','complete','limited']:
    require(all(h['topic'] in topics for h in r[field]),'unexpected fixture topic');typed_hits(r[field])
   require(len(r['limited'])==limit and typed_hits(r['limited'])==typed_hits(r['complete'][:limit]),'timed/full prefix')
   require(len({h['topic_id'] for h in r['complete']})==len(r['complete']) and {h['topic_id'] for h in r['complete']}=={h['topic_id'] for h in r['fused']},'complete result permutation of fused IDs')
   mapping={h['topic_id']:(h['topic'],h['state'],h['seed'],h['score_bits']) for h in r['fused']}
   require(all(mapping[h['topic_id']]==(h['topic'],h['state'],h['seed'],h['score_bits']) for h in r['complete']),'fused result identity/provenance preserved')
   require(all(type(v) is int and v>=0 for key in ['before','after','diagnostic_before','diagnostic_after'] for v in r[key].values()),'unsigned integer lifetime counters')
   require(r['diagnostic_before']==r['after'],'timed/diagnostic lifetime counter continuity')
   if step:require(r['before']==rows[step-1]['diagnostic_after'],'between-call lifetime counter continuity')
   require(len(r['fused'])==len(r['complete']) and len({h['topic_id'] for h in r['fused']})==len(r['fused']),'full fused IDs unique')
   require(sum(any(w.get('kind')=='channel' and w.get('channel')=='vector' for w in h['why']) for h in r['fused'])==50,'vector channel budget')
   pos=r['selected_fused_positions'];require(all(isinstance(p,int) and 0<=p<len(r['fused']) for p in pos) and len(set(pos))==len(pos),'selected positions')
   visible=len(pos)>=2 and pos[0]<limit;require(r['can_be_seen']==visible,'can_be_seen actual selected positions')
   d=delta(r);diag=delta(r,'diagnostic_before','diagnostic_after');require(all(v>=0 for v in [*d.values(),*diag.values()]),'successful cumulative counters')
   if tier=='off':require(all(d[k]==diag[k]==0 for k in KEYS) and not visible,'Off avoids reranker')
   else:
    require(r['after']['maximum_tokens']==controls['maximum_tokens']==256,'loaded truncation configuration')
    require(visible and len(pos)==d['offered']==diag['offered']==30,'actual30offered visible at productlimit')
    scored=[h['topic_id'] for h in r['complete'] if h['reranked_bits']];require(set(scored)==set(selected(r)) and len(scored)==30,'actual Why selected score IDs')
    require(all(diag[k]==0 for k in ZERO),'outside-clock diagnostic no forwards')
    if arm=='memo':require(diag['encode_us']==0,'memo diagnostic skip tokenization')
    if step==0 or mode=='N':require(d['scored']==30,'fresh/new real model pairs')
   require(all(any(path in line for line in r['loaded_libraries']) for path in required_libraries),'actual mapped ORT/zvec')
   phase=('off_cold' if step==0 else 'off_hot' if step==1 else 'off_new') if tier=='off' else 'cold' if step==0 else 'new' if mode=='N' else 'changed' if step==6 else 'hot'
   key=(limit,r['query_id'],tier,mode,phase,arm)
   item={'process':name,'step':step,'wall_us':r['wall_us'],'encode_us':d['encode_us'],'forward_us':d['forward_us'],'scored':d['scored'],'batches':d['batches'],'cpu_ticks':sum(r['process_after'][k]-r['process_before'][k] for k in ['utime_ticks','stime_ticks']),'rss_kib':r['process_after']['rss_kib'],'hwm_kib':r['process_after']['hwm_kib']}
   groups.setdefault(key,[]).append(item);calls.append({'group':list(key),**item})
   rank={h['topic']:h['rank'] for h in r['complete']};relevant={t:rank.get(t) for t in gold[r['query_id']]['relevant']};require(all(v is not None for v in relevant.values()),'gold relevant retained')
   gain=sum(1/math.log2(v+1) for v in relevant.values() if v<=limit);ideal=sum(1/math.log2(v+1) for v in range(1,min(len(relevant),limit)+1));quality.append({'process':name,'step':step,'qid':r['query_id'],'limit':limit,'relevant_ranks':relevant,'ndcg_at_limit':gain/ideal})
  for old,new in zip(rows,rows[1:]):
   if old['arm']==new['arm'] and old['query_id']==new['query_id']:
    for field in ['fused','complete','limited']:require(typed_hits(old[field])==typed_hits(new[field]),'hot exact typed bits/order')
    d=delta(new);require(all(d[k]==0 for k in ZERO),'hot zero fresh model operations')
    if arm=='memo':require(d['encode_us']==0,'memo hot no encoding')
    elif tier=='accurate':require(d['encode_us']>0,'baseline hot encoding observed')
 for name,rep,arm,limit,q,tier,initial in jobs():
  rows=rowsby[name]
  if arm=='baseline':
   peer=rowsby[name.replace('-baseline-','-memo-')]
   for old,new in zip(rows,peer):
    for field in ['fused','complete','limited']:require(typed_hits(old[field])==typed_hits(new[field]),'cross-source exact typed outputs')
    require(all(delta(old)[k]==delta(new)[k] for k in ['offered','scored','tokens','padded_tokens','batches']),'same actual forward work')
  if tier=='accurate':
   warm=rows[6];fresh=rowsby[name[:-1]+warm['arm']][0]
   for field in ['fused','complete','limited']:require(typed_hits(warm[field])==typed_hits(fresh[field]),'warm/fresh same final context')
   require((selected(rows[0])!=selected(warm))==(q==101),'changed shortlist hypothesis')
   require(delta(warm)['scored']==(20 if q==101 else 0),'whole-group history hypothesis')
 keyed=lambda rows:{(r['process'],r['step']):r for r in rows}
 require(len(calls)==428 and keyed(calls)==keyed(cost['calls']) and keyed(quality)==keyed(cost['quality']),'full raw derivation of retained calls/gold')
 # Stored cost arrays use sorted filename order; compare keyed canonical records.
 table={tuple(r['group']):r for r in cost['groups']}
 require(set(table)==set(groups),'all cost groups retained')
 for key,rows in groups.items():
  entry=table[key];require(entry['wall_us']==describe([r['wall_us'] for r in rows]) and entry['encode_us']==describe([r['encode_us'] for r in rows]) and entry['forward_us']==describe([r['forward_us'] for r in rows]) and entry['hwm_kib']==describe([r['hwm_kib'] for r in rows]) and entry['cpu_ticks']==describe([r['cpu_ticks'] for r in rows]),'cost derivation')
  expected={p:statistics.median(r['wall_us'] for r in rows if r['process']==p) for p in {r['process'] for r in rows}};require(entry['process_block_medians_us']==expected,'independent process blocks')
 pairs={tuple(r['group']):r for r in cost['paired_costs']}
 for key,rows in groups.items():
  if key[-1]!='baseline':continue
  new=groups[key[:-1]+('memo',)];p=pairs[key[:-1]];oldmed=statistics.median(r['wall_us'] for r in rows);newmed=statistics.median(r['wall_us'] for r in new)
  require(p['baseline_sample_median_us']==oldmed and p['memo_sample_median_us']==newmed and p['difference_us']==newmed-oldmed and p['percent_change']==100*(newmed-oldmed)/oldmed,'descriptive wall costs including regressions')
 memory=jsonfile(root/'paired-memory.json');require(len(memory)==8,'memory groups')
 for entry in memory:
  blocks={}
  for arm in ['baseline','memo']:
   rows=groups[(entry['return_limit'],entry['query_id'],'accurate','A','hot',arm)];field=entry['metric'];require(field in ['rss_kib','hwm_kib'],'memory metric')
   blocks[arm]={p:statistics.median(r[field] for r in rows if r['process']==p) for p in sorted({r['process'] for r in rows})}
  old=statistics.median(blocks['baseline'].values());new=statistics.median(blocks['memo'].values())
  require(entry['process_blocks']==blocks and entry['baseline']==old and entry['memo']==new and entry['difference']==new-old and entry['percent_change']==100*(new-old)/old,'paired process memory derivation')
 require(cost['gold_exact_equal'] and cost['source_typed_why_bits_equal'],'accepted typed/gold fields')
 return {'processes':36,'timed_calls':428,'limits':[5,10],'full300_diagnostic_outside_clock':True,'exact_fresh_history_hot_bits_and_gold':True,'memo_hot_encoding_and_model_operations_skipped':True,'original_seed_assets_unchanged_with_four_unexplained_clone_proxima_changes':True,'scope':'two rounds; descriptive wall costs, CPU100Hz zero ticks censored, diagnostics included in process memory, no broad quality/platform/default claim'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
if __name__=='__main__':main()
