"""Serial owned native Engine runner. Requires explicit grant and --execute."""
import sys
if sys.flags.optimize:raise SystemExit("Guarded runner refuses -O/PYTHONOPTIMIZE")
import argparse,fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
import guards as g
import build_guard as b
ROOT=Path(__file__).resolve().parent
KEYS=['offered','scored','characters','tokens','padded_tokens','batches','encode_us','forward_us']
def delta(row,a='before',z='after'):return {k:row[z][k]-row[a][k] for k in KEYS}
def save(out,name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def selected(row):return [row['fused'][p]['topic_id'] for p in row['selected_fused_positions']]
def validate(row,arm):
 assert row['documents']==230 and row['coverage']==0.0
 assert row['limited']==row['complete'][:len(row['limited'])]
 assert len(row['limited'])==row['return_limit']
 d=delta(row);diag=delta(row,'diagnostic_before','diagnostic_after');assert min(d.values())>=0 and min(diag.values())>=0
 assert sum(any(w.get('kind')=='channel' and w.get('channel')=='vector' for w in h['why']) for h in row['fused'])==50
 if row['tier']=='off':assert d['offered']==d['scored']==diag['offered']==0
 else:
  assert row['can_be_seen'] and len(selected(row))==d['offered']==diag['offered']==30
  scored=[h['topic_id'] for h in row['complete'] if h['reranked_bits']];assert set(scored)==set(selected(row)) and len(scored)==30
  assert all(len(h['reranked_bits'])<=1 for h in row['complete'])
  assert all(diag[k]==0 for k in ['scored','tokens','padded_tokens','batches','forward_us'])
  if arm=='memo':assert diag['encode_us']==0
  if row['step']==0 or row['arm']=='N':assert d['scored']==30
 for lib in g.LIBRARIES[:2]:assert any(str(lib.resolve()) in line for line in row['loaded_libraries']),'mapped native library absent'
 return d,diag

def launch(home,env,command,log,usage):
 # Authentication and listener/PID authority precede the first native Engine.
 database=None;process=None;record={'hardware_before':g.hardware()}
 try:
  database=g.pg.start(home,env);record['owned_postgres']=database
  g.pg.check(home,database['identity']);g.pg.show_data_directory(g.pg.layout(home),env)
  with log.open('x') as output:
   process=subprocess.Popen(command,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();samples=[];loader_seen=False
   while process.poll() is None:
    g.pg.check(home,database['identity']);r=g.headroom()
    old_events=dict(l.split() for l in record['hardware_before']['cgroup']['memory.events'].splitlines());new_events=dict(l.split() for l in r['cgroup']['memory.events'].splitlines())
    assert all(old_events.get(k,'0')==new_events.get(k,'0') for k in ['oom','oom_kill']),'OOM changed during native execution'
    try:
     maps=Path(f'/proc/{process.pid}/maps').read_text()
     mapped=[Path(l.split()[-1]).resolve() for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/')]
     loader_seen=loader_seen or g.LIBRARIES[2].resolve() in mapped
    except FileNotFoundError:pass
    samples.append({'elapsed':time.monotonic()-start,'disk_free':r['disk_free'],'cgroup':r['cgroup']})
    assert time.monotonic()-start<600,'native process exceeded600s'
    time.sleep(1)
   record['resource_samples']=samples;record['returncode']=process.returncode;record['actual_loader_mapping_verified']=loader_seen
   assert loader_seen,'expected ELFloader not observed in native process mappings'
   assert process.returncode==0,'native Engine failure; preserve owned clone/log'
  g.pg.check(home,database['identity']);g.pg.show_data_directory(g.pg.layout(home),env)
 finally:
  original_error=sys.exc_info()[1];cleanup_error=None
  if original_error is not None:record['original_failure_class']=type(original_error).__name__
  try:
   if process is not None and process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
  except BaseException as error:
   cleanup_error=error;record['native_cleanup_failure_class']=type(error).__name__
  finally:
   try:
    # A partial helper.start failure still stops ONLY verified owned layout.
    record['stop']=g.pg.stop(home,env,database['identity'] if database else None)
   except BaseException as error:
    cleanup_error=error;record['stop_failure_class']=type(error).__name__
   finally:
    try:
     record['hardware_after']=g.hardware()
     old_events=dict(l.split() for l in record['hardware_before']['cgroup']['memory.events'].splitlines());new_events=dict(l.split() for l in record['hardware_after']['cgroup']['memory.events'].splitlines())
     if any(old_events.get(k,'0')!=new_events.get(k,'0') for k in ['oom','oom_kill']):
      error=RuntimeError('OOM changed at process endpoint');cleanup_error=error;record['resource_failure_class']=type(error).__name__
    except BaseException as error:cleanup_error=error;record['hardware_failure_class']=type(error).__name__
    usage.write_text(json.dumps(record,indent=2)+'\n')
  if original_error is None and cleanup_error is not None:raise cleanup_error

 return record

def jobs(arms):
 work=[]
 for rep,order in enumerate([arms,list(reversed(arms))]):
  for a in order:
   for limit in [5,10]:
    for q in [80,101]:
     for seq in ['A,A,A,A,A,A,B,B,B,B,B,B,N','B,B,B,B,B,B,A,A,A,A,A,A,N']:work.append((rep,a,limit,q,'accurate',seq))
 for a in arms:
  for limit in [5,10]:work.append((0,a,limit,80,'off','A,A,N'))
 return work

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--slot-granted',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if not args.execute:print('CODE-ONLY:36processes428timed@5/@10 calls; no native PG/model launched');return
 assert args.slot_granted,'explicit exclusive live slot required';assert not args.output.exists()
 build=json.loads((ROOT/'build-results/provenance.json').read_text());assert len(build['arms'])==2
 assets=g.assets();seed=g.inventory(g.SEED);assert assets==build['assets_after'] and seed==build['seed_after']
 g.headroom(256*1024**2);args.output.mkdir();save(args.output,'before.json',{'assets':assets,'seed':seed,'hardware':g.hardware(),'owned_helper_sha256':g.digest(g._PG_PATH),'loader_realpath':str(g.LIBRARIES[2].resolve()),'loader_sha256':g.digest(g.LIBRARIES[2]),'source_build_binding_sha256':g.digest(ROOT/'build-results/provenance.json'),'controller':{p.name:g.digest(p) for p in [ROOT/'run.py',ROOT/'guards.py',g._PG_PATH]},'scope':'fresh4product/helper only; transitive shareddependency provenance historicalunknown'})
 results={};controller_pins={p:g.digest(p) for p in [ROOT/'run.py',ROOT/'guards.py',ROOT/'build_guard.py',ROOT/'build-results/provenance.json',g._PG_PATH]}
 with Path('<SCRATCH>/.pamin-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for rep,a,limit,q,tier,seq in jobs(build['arms']):
   assert all(g.digest(p)==h for p,h in controller_pins.items()),'controller/helper altered after freeze'
   name=f'r{rep}-{a["arm"]}-l{limit}-q{q}-{tier}-{seq[0]}';home=args.output/(name+'-home');binary=Path(a['binary']);source=Path(a['source'])
   assert g.digest(binary)==a['binary_sha256'];assert b.inputs(source,a['commit'],a['allowed_untracked'])==a['inputs_after'];assert g.assets()==assets and g.inventory(g.SEED)==seed
   g.clone(g.SEED,home);initial_inventory=g.inventory(home);env,scrubbed=g.runtime_environment(home)
   env.update(CACHE_RETURN_LIMIT=str(limit),CACHE_QUERY_ID=str(q),CACHE_TIER=tier,CACHE_SEQUENCE=seq)
   command=[str(binary),'--ignored','--exact','cache_product_limit_probe','--nocapture','--test-threads=1']
   save(args.output,name+'-launch.json',{'command':command,'environment':env,'scrubbed_inherited_names':scrubbed,'clone_inventory_before':initial_inventory,'binary_sha256':g.digest(binary),'loader_sha256':g.digest(g.LIBRARIES[2])})
   print('LIMIT_ENGINE_START',name,flush=True);launch(home,env,command,args.output/(name+'.log'),args.output/(name+'-usage.json'))
   text=(args.output/(name+'.log')).read_text();assert 'test result: ok. 1 passed;' in text
   providers=[];decoder=json.JSONDecoder()
   for line in text.splitlines():
    if 'ONNX graph execution-provider assignment' in line:
     graph,_=decoder.raw_decode(line.split('model_graph=',1)[1]);nodes,_=decoder.raw_decode(line.split('assigned_nodes=',1)[1]);assert set(nodes)=={'CPUExecutionProvider'} and nodes['CPUExecutionProvider']>0
     path=Path(graph).resolve();assert str(path) in assets and g.digest(path)==assets[str(path)]['sha256'];providers.append({'graph':str(path),'nodes':nodes})
   assert len(providers)==(2 if tier=='accurate' else 1)
   rows=[json.loads(l.split('CACHE_ENGINE_ROW ',1)[1]) for l in text.splitlines() if 'CACHE_ENGINE_ROW ' in l];assert len(rows)==len(seq.split(','))
   for step,(expected_arm,row) in enumerate(zip(seq.split(','),rows)):
    assert row['step']==step and row['arm']==expected_arm and row['query_id']==(q+1 if expected_arm=='N' else q) and row['return_limit']==limit and row['tier']==tier,'actual native row differs from requested matrix'
    row['actual_delta'],row['diagnostic_delta']=validate(row,a['arm'])
   for prev,row in zip(rows,rows[1:]):
    if prev['arm']==row['arm'] and prev['query_id']==row['query_id']:
     assert prev['fused']==row['fused'] and prev['complete']==row['complete'],'hot bit/order drift'
     assert all(row['actual_delta'][k]==0 for k in ['scored','tokens','padded_tokens','batches','forward_us'])
     if a['arm']=='memo':assert row['actual_delta']['encode_us']==0
   assert g.assets()==assets and g.inventory(g.SEED)==seed
   assert all(g.digest(p)==h for p,h in controller_pins.items()),'controller/helper changed during job'
   assert g.digest(binary)==a['binary_sha256'] and b.inputs(source,a['commit'],a['allowed_untracked'])==a['inputs_after'],'source/binary changed during job'
   save(args.output,name+'-providers.json',providers);(args.output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));results[name]=rows
   save(args.output,name+'-clone-after.json',g.inventory(home));g.pg.stop(home,env);assert not (home/'postgres/data/postmaster.pid').exists()
   # Only successful stopped fully validated clones are removable; failure exits
   # before this point, leaving the disposable home and original full raw intact.
   shutil.rmtree(home);save(args.output,name+'-cleanup.json',{'validated_owned_clone_removed':str(home),'original_seed_unchanged':g.inventory(g.SEED)==seed});g.headroom()
   print('LIMIT_ENGINE_COMPLETE',name,flush=True)
 comparisons=[]
 for rep in [0,1]:
  for limit in [5,10]:
   for q in [80,101]:
    for initial,final in [('A','B'),('B','A')]:
     src=[]
     for arm in ['baseline','memo']:
      history=results[f'r{rep}-{arm}-l{limit}-q{q}-accurate-{initial}']
      changed=selected(history[0])!=selected(history[6]);actual_work=history[6]['actual_delta']['scored']
      assert changed==(q==101),'changed-shortlist context hypothesis changed; abort rather than force'
      assert actual_work==(20 if q==101 else 0),'whole-group history workload hypothesis changed; abort rather than force'
      fresh=results[f'r{rep}-{arm}-l{limit}-q{q}-accurate-{final}'][0];warm=history[6]
      assert warm['fused']==fresh['fused'] and warm['complete']==fresh['complete'] and warm['limited']==fresh['limited'],'same-final-list repair regressed'
      for at in [0,6,12]:src.append((arm,at,history[at]))
     for at in [0,6,12]:
      old=next(r for arm,i,r in src if arm=='baseline' and i==at);new=next(r for arm,i,r in src if arm=='memo' and i==at)
      assert old['fused']==new['fused'] and old['complete']==new['complete'] and old['limited']==new['limited']
      assert all(old['actual_delta'][k]==new['actual_delta'][k] for k in ['offered','scored','tokens','padded_tokens','batches'])
     comparisons.append({'repeat':rep,'limit':limit,'qid':q,'history':initial+'->'+final,'same_final_bits_order_equal':True,'selected_order_changed':q==101,'changed_fresh_rows':20 if q==101 else 0})
 save(args.output,'summary.json',{'comparisons':comparisons,'36_processes_complete':True,'428_timed_calls':True,'all_owned_pg_stopped':True,'limits':[5,10],'full300_diagnostics_outside_clock':True,'scope':'small CPU controls; no broadquality or stablelatencyclaim'})
 save(args.output,'after.json',{'assets':g.assets(),'seed':g.inventory(g.SEED),'hardware':g.hardware()})
if __name__=='__main__':main()
