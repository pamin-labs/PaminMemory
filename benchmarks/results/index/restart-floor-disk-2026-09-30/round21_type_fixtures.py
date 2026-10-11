"""Small resealed archive copies for strict evidence-type regressions."""
import gzip,hashlib,json,re,runpy,shutil,subprocess,sys,tempfile
from pathlib import Path
import test_evidence_review as archive
BASE='94a38bace6b34b3c579b23f0c8d1d285596435c9'
def run(kind):
 before='--before' in sys.argv;names=set()
 for rel in [archive.DISK,archive.HNSW]:
  names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
 gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
 cases={'provider':[True,1.0],'flags':['matches_pretrial_inventory','seed_postgresql_stopped','metadata_stable_during_capture'],'seed':[('bytes',-1),('bytes',True),('bytes',1.0),('sha256','not-a-sha256'),('sha256','A'*64),('sha256','a'*63)],'topics':['integer','mapping']}[kind]
 if not before and kind=='provider':cases += [-1,0,'1']
 count=0
 for rel in ([archive.HNSW] if kind=='seed' else [archive.DISK] if kind=='flags' else [archive.DISK,archive.HNSW]):
  for case in cases:
   with tempfile.TemporaryDirectory(prefix='restart-evidence-types-') as directory:
    repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
    for name in names:
     dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
    if before:
     for name in [str(archive.DISK/'verify.py'),str(archive.HNSW/'verify.py'),str(archive.DISK/'evidence_review.py')]:
      (repo/name).write_bytes(subprocess.check_output(['git','-C',str(archive.REPO),'show',BASE+':'+name]))
    raw=repo/rel/'raw.jsonl';rows=[json.loads(s) for s in raw.read_text().splitlines()]
    if kind=='provider':
     archive.update(repo/archive.DISK/'provider-bindings.json',lambda x:[r.update(cpu_nodes=case) for r in x['roles'].values()])
     for row in rows:
      for provider in row.get('actual_providers',{}).values():provider['assigned_nodes']['CPUExecutionProvider']=case
    elif kind=='flags':archive.update(repo/archive.DISK/'post-review-seed.json',lambda x:x.update({case:'false'}))
    elif kind=='seed':
     def change(x):
      item=next(e for e in x['stopped_seed_index_files'] if not e['path'].endswith('/profile') and not e['path'].endswith('/.pamin-optimized-files'));item[case[0]]=case[1]
     archive.update(repo/archive.HNSW/'provenance.json',change)
    else:
     for row in rows:
      if row['phase']=='search_warm':
       topics=row['extra']['topics']
       if case=='mapping':row['extra']['topics']={t:True for t in topics}
       else:
        index=next(i for i in reversed(range(len(topics))) if topics[i]!=f"incident-{row['extra']['query_document']}");topics[index]=123
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if kind in {'provider','topics'}:
     for log in (repo/rel/'logs').glob('*.log*'):
      if not re.fullmatch(r'[012]-(main|predecessor|candidate)\.log(?:\.gz)?',log.name):continue
      compressed=log.suffix=='.gz';text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();rep=int(log.name[0]);arm=log.name.split('-',1)[1].split('.',1)[0];lines=text.splitlines()
      for at,line in enumerate(lines):
       if kind=='provider' and 'ONNX graph execution-provider assignment' in line:lines[at]=re.sub(r'assigned_nodes=\{"CPUExecutionProvider":\d+\}', 'assigned_nodes='+json.dumps({'CPUExecutionProvider':case}),line)
       if kind=='topics' and line.startswith('RESTART_JSON '):
        native=json.loads(line.removeprefix('RESTART_JSON '))
        if native['phase']=='search_warm':native['extra']['topics']=next(r['extra']['topics'] for r in rows if r['arm']==arm and r['repetition']==rep and r['phase']=='search_warm' and r['extra']['query_document']==native['extra']['query_document']);lines[at]='RESTART_JSON '+json.dumps(native)
      content=('\n'.join(lines)+'\n').encode()
      if compressed:log.write_bytes(gzip.compress(content,mtime=0))
      else:log.write_bytes(content)
    review=runpy.run_path(str(repo/archive.DISK/'evidence_review.py'));code=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in';calc=runpy.run_path(str(code));hwm=review['historical_hwm_review'](rows,'disk' if rel==archive.DISK else 'memory',hashlib.sha256(raw.read_bytes()).hexdigest());(repo/rel/'hwm-review.json').write_text(json.dumps(hwm,indent=2)+'\n');summary=review['qualified_summary'](calc['summarize'](rows),hwm,hashlib.sha256(code.read_bytes()).hexdigest());(repo/rel/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive.manifests(repo);result=archive.execute(repo,rel)
    if before:assert result.returncode==0,result.stderr
    else:
     reason={'provider':'provider node count must be an actual positive integer','flags':'seed endpoint flags must be exactly true','seed':'seed file identity','topics':'query topics must be an actual list of strings'}[kind]
     if kind=='provider' and case in [-1,0,'1']:assert result.returncode!=0,result.stderr
     else:assert result.returncode!=0 and reason in result.stderr,result.stderr
    count+=1
 print(('BEFORE accepted ' if before else 'PASS rejected ')+str(count)+' resealed '+kind+' mutations')
