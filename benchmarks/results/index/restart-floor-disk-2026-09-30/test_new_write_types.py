"""Matched/resealed new-write field mutations; immutable original fixtures."""
import gzip,hashlib,json,runpy,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive
before='--before' in sys.argv;BASE='7b6cb52dff947aad5a8f8b805dee24c78bb706aa'
names=set()
for rel in [archive.DISK,archive.HNSW]:
 names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
cases=[('topics','restart-proof'),('known_new_topic_retrieved',1),('topics',{'restart-proof':True}),('topics',['restart-proof',1]),('known_new_topic_retrieved','true')]
if not before:cases += [('topics',None),('topics',[]),('known_new_topic_retrieved',False),('known_new_topic_retrieved',0)]
for rel in [archive.DISK,archive.HNSW]:
 for field,value in cases:
  arm='main';phase='new_write_search'
  with tempfile.TemporaryDirectory(prefix='restart-new-write-') as directory:
   repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
   for name in names:
    dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
   if before:
    name=str(rel/'verify.py');(repo/name).write_bytes(subprocess.check_output(['git','-C',str(archive.REPO),'show',BASE+':'+name]))
   raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
   selected=next(r for r in rows if r['arm']==arm and r['repetition']==0 and r['phase']==phase);selected['extra'][field]=value
   raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
   compressed=rel==archive.DISK;log=repo/rel/'logs'/('0-'+arm+'.log'+('.gz' if compressed else ''))
   text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();lines=text.splitlines()
   for at,line in enumerate(lines):
    if line.startswith('RESTART_JSON '):
     native=json.loads(line.removeprefix('RESTART_JSON '))
     if native['phase']==phase:native['extra']=selected['extra'];lines[at]='RESTART_JSON '+json.dumps(native)
   text='\n'.join(lines)+'\n'
   if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
   else:log.write_text(text)
   # Refresh derived receipts so rejection must come from result semantics.
   review=runpy.run_path(str(repo/archive.DISK/'evidence_review.py'))
   index='disk' if compressed else 'memory';hwm=review['historical_hwm_review'](rows,index,hashlib.sha256(raw.read_bytes()).hexdigest())
   (repo/rel/'hwm-review.json').write_text(json.dumps(hwm,indent=2)+'\n')
   if before:
    code=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in';calc=runpy.run_path(str(code))
    calculated=calc['summarize'](rows);summary=review['qualified_summary'](calculated,hwm,hashlib.sha256(code.read_bytes()).hexdigest())
    (repo/rel/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
   archive.manifests(repo);result=archive.execute(repo,rel)
   if before:assert result.returncode==0,result.stderr
   else:assert result.returncode!=0 and ('new-write topics' if field=='topics' and value!=[] else 'new-write topic visibility' if field=='topics' else 'new-write retrieved flag') in result.stderr,result.stderr
print('BEFORE: both archives accepted10 matched/resealed malformed new-write results' if before else 'PASS: both archives reject18 matched/resealed malformed new-write results')
