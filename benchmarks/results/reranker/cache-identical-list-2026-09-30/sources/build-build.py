"""Authorized serial build-only stage; never executes frozen helpers/PG/models."""
import fcntl,json,os,shutil,signal,subprocess,time
from pathlib import Path
import guards as g
import build_guard as b
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'build-results';REPO=Path('<SCRATCH>/reranker-batch-cache');TARGET=Path('<WORKSPACE>/PaminMemory/target')
ARMS=[('baseline','2f5bd087ceb4b65a25dc94932e416d8ac5c29740'),('memo','d0b6a14a4f317fea3c1e117f627289aad1b1c964')]
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def monitor(command,source,env,out,err,initial):
 samples=[];p=subprocess.Popen(command,cwd=source,env=env,stdout=out,stderr=err,start_new_session=True);start=time.monotonic()
 try:
  while p.poll() is None:
   r=g.headroom();samples.append({'elapsed':time.monotonic()-start,'disk_free':r['disk_free'],'cgroup':r['cgroup']})
   for k in ['oom','oom_kill']:
    def value(s):return dict(l.split() for l in s.splitlines()).get(k,'0')
    assert value(r['cgroup']['memory.events'])==value(initial['cgroup']['memory.events']),'OOM changed'
   assert time.monotonic()-start<900,'build exceeded900s'
   time.sleep(2)
  assert p.returncode==0,'build failed; raw preserved'
 except BaseException:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  raise
 finally:write(source.name+'-resources.json',samples)
def main():
 assert not OUT.exists();initial=g.headroom(1024**3);parent,scrubbed=g.clean_environment();env,tools=b.build_environment(parent)
 env.update(CARGO_INCREMENTAL='0',CARGO_BUILD_JOBS='4',LD_LIBRARY_PATH='<WORKSPACE>/.cargo/lib/pamin:<WORKSPACE>/.onnxruntime/onnxruntime-linux-x64-1.28.0/lib')
 OUT.mkdir();asset_before=g.assets();seed_before=g.inventory(g.SEED);tool_before=b.tool_identities(tools)
 versions={name:subprocess.check_output([str(path),'-Vv' if name!='rustdoc' else '-V'],env=env,text=True) for name,path in tools.items()};assert versions['rustc'].startswith('rustc 1.98.1 ')
 configs={}
 for parent in [Path('<WORKSPACE>'),ROOT,REPO,Path('<WORKSPACE>/.cargo')]:
  for file in ['config','config.toml']:
   path=(parent/file if parent.name=='.cargo' else parent/'.cargo'/file)
   if path.is_file():configs[str(path)]={'sha256':g.digest(path),'text':path.read_text()}
 provenance={'before':initial,'assets_before':asset_before,'seed_before':seed_before,'tools':tool_before,'versions':versions,'environment':env,'scrubbed_names':scrubbed,'cargo_configs':configs,'scope':'all tracked product/source inputs against Git blobs plus fresh4productlibs/helper; shared transitive third-party artifacts historical provenance unknown, not freshly source-attested','guard_source':{p.name:g.digest(p) for p in [ROOT/'build.py',ROOT/'build_guard.py',ROOT/'guards.py',ROOT/'prepare.py',ROOT/'probe.rs.append',ROOT/'PLAN.md']},'arms':[]};write('provenance.json',provenance)
 with Path('<SCRATCH>/.pamin-measurement.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm,commit in ARMS:
   source=ROOT/'sources'/arm;assert not source.exists();source.parent.mkdir(exist_ok=True)
   subprocess.run(['git','-C',str(REPO),'worktree','add','--detach',str(source),commit],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   subprocess.run(['/usr/bin/python3',str(ROOT/'prepare.py'),str(source)],check=True)
   allowed={b.TARGET:g.digest(source/b.TARGET)};before=b.inputs(source,commit,allowed);g.headroom()
   cargo=str(tools['cargo']);clean=[cargo,'clean','--release',*[v for n in sorted(b.PRODUCTS) for v in ['-p',n]]]
   command=[cargo,'test','--release','--offline','--locked','-p','pamin-engine','--test','scratch_cache_product_limits','--no-run','--message-format=json']
   print('MEMO_BUILD_START',arm,flush=True)
   with (OUT/(arm+'-clean.log')).open('x') as log:subprocess.run(clean,cwd=source,env=env,stdout=log,stderr=log,check=True)
   with (OUT/(arm+'.cargo.jsonl')).open('x') as out,(OUT/(arm+'.build.log')).open('x') as err:monitor(command,source,env,out,err,initial)
   after=b.inputs(source,commit,allowed);assert before==after;assert b.tool_identities(tools)==tool_before
   records=[json.loads(l) for l in (OUT/(arm+'.cargo.jsonl')).read_text().splitlines() if l.startswith('{')];artifacts,exe=b.fresh_artifacts(records,source)
   binary=OUT/(arm+'-frozen');shutil.copy2(exe,binary);assert g.digest(binary)==g.digest(exe)
   assert g.assets()==asset_before and g.inventory(g.SEED)==seed_before
   provenance['arms'].append({'arm':arm,'commit':commit,'source':str(source),'inputs_before':before,'inputs_after':after,'allowed_untracked':allowed,'command':command,'clean':clean,'artifacts':artifacts,'binary':str(binary),'binary_sha256':g.digest(binary),'binary_bytes':binary.stat().st_size,'hardware_after':g.hardware()});write('provenance.json',provenance)
   print('MEMO_BUILD_FROZEN',arm,g.digest(binary),flush=True)
 provenance.update(after=g.hardware(),assets_after=g.assets(),seed_after=g.inventory(g.SEED));write('provenance.json',provenance);print('DONE_BUILD_ONLY_NO_NATIVE_PG_MODEL',flush=True)
if __name__=='__main__':main()
