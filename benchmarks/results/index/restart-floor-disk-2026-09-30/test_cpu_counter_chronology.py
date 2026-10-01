"""Resealed equal-delta counter drops; bounded archive copies, no product run."""
import gzip,hashlib,importlib.util,json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive
BASE='ec96e57695bee67b04299a3da9ed6611a068e899';before='--before' in sys.argv
names=set()
for rel in [archive.DISK,archive.HNSW]:
 names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
for rel in [archive.DISK,archive.HNSW]:
 for key in ['user_ticks','system_ticks']:
  with tempfile.TemporaryDirectory(prefix='restart-cpu-chronology-') as directory:
   repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
   for name in names:
    dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
   if before:
    source=str(archive.DISK/'evidence_review.py');(repo/source).write_bytes(subprocess.check_output(['git','-C',str(archive.REPO),'show',BASE+':'+source]))
   raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
   process=[r for r in rows if r['arm']=='main' and r['repetition']==0 and r['phase']!='process_total']
   selected=next(r for r in process if r['phase']=='search_cold');prior=process[process.index(selected)-1]['process_after'][key];assert prior>0
   before_ticks=selected['process_before'][key];after_ticks=selected['process_after'][key]
   shift=before_ticks-prior+1;assert shift>0 and after_ticks>=shift
   selected['process_before'][key]-=shift;selected['process_after'][key]-=shift
   assert selected['process_after'][key]-selected['process_before'][key]==after_ticks-before_ticks>=0
   raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
   compressed=rel==archive.DISK;log=repo/rel/'logs'/('0-main.log'+('.gz' if compressed else ''))
   text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();lines=text.splitlines()
   for at,line in enumerate(lines):
    if line.startswith('RESTART_JSON '):
     native=json.loads(line.removeprefix('RESTART_JSON '))
     if native['phase']=='search_cold':
      for field in ['process_before','process_after']:native[field]=selected[field]
      lines[at]='RESTART_JSON '+json.dumps(native)
   text='\n'.join(lines)+'\n'
   if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
   else:log.write_text(text)
   if before:
    spec=importlib.util.spec_from_file_location('review',repo/archive.DISK/'evidence_review.py');review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    hwm=review.historical_hwm_review(rows,'disk' if rel==archive.DISK else 'memory',hashlib.sha256(raw.read_bytes()).hexdigest());(repo/rel/'hwm-review.json').write_text(json.dumps(hwm,indent=2)+'\n')
    summary=json.loads((repo/rel/'summary.json').read_text());summary['summary_derivation']['raw_sha256']=hwm['raw_sha256'];(repo/rel/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
   archive.manifests(repo);result=archive.execute(repo,rel)
   if before:assert result.returncode==0,result.stderr
   else:assert result.returncode!=0 and 'process cumulative CPU counters decreased within/across phases' in result.stderr,result.stderr
print('BEFORE: both archives accepted4 matched/resealed nonnegative-delta cumulative tick drops' if before else 'PASS: both archives reject4 matched/resealed cumulative user/system counter drops despite unchanged nonnegative within-phase deltas')
