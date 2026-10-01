"""Resealed path/scope negatives in small disposable retained archive copies."""
import hashlib,json,runpy,shutil,subprocess,sys,tempfile
from pathlib import Path
import test_evidence_review as archive
BASE='741423999d7acdeeefd48afb54d991b27342cc86'
def run(kind):
 before='--before' in sys.argv;names=set()
 for rel in [archive.DISK,archive.HNSW]:names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
 gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
 cases=['${SCRATCH}/unrelated/foreign.index','${SCRATCH}/seed-memory/index/../outside','${SCRATCH}/seed-memory/index/./data','${SCRATCH}/seed-memory/index//data','${SCRATCH}/seed-memory/index','${SCRATCH}/seed-memory/index-other/data'] if kind=='paths' else ['missing','setup only; searches excluded',True]
 count=0
 for rel in ([archive.HNSW] if kind=='paths' else [archive.DISK,archive.HNSW]):
  for case in cases:
   with tempfile.TemporaryDirectory(prefix='restart-scope-path-') as directory:
    repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
    for name in names:
     dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
    if before:
     for name in [str(archive.DISK/'verify.py'),str(archive.HNSW/'verify.py'),str(archive.DISK/'evidence_review.py')]:
      (repo/name).write_bytes(subprocess.check_output(['git','-C',str(archive.REPO),'show',BASE+':'+name]))
    if kind=='paths':
     def change(x):next(e for e in x['stopped_seed_index_files'] if not e['path'].endswith('/profile') and not e['path'].endswith('/.pamin-optimized-files'))['path']=case
     archive.update(repo/rel/'provenance.json',change)
    else:
     raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()];row=next(r for r in rows if r['phase']=='process_total')
     if case=='missing':row.pop('includes')
     else:row['includes']=case
     raw.write_text(''.join(json.dumps(row)+'\n' for row in rows));review=runpy.run_path(str(repo/archive.DISK/'evidence_review.py'));code=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in';calc=runpy.run_path(str(code));hwm=review['historical_hwm_review'](rows,'disk' if rel==archive.DISK else 'memory',hashlib.sha256(raw.read_bytes()).hexdigest());(repo/rel/'hwm-review.json').write_text(json.dumps(hwm,indent=2)+'\n');summary=review['qualified_summary'](calc['summarize'](rows),hwm,hashlib.sha256(code.read_bytes()).hexdigest());(repo/rel/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    archive.manifests(repo);result=archive.execute(repo,rel)
    if before:assert result.returncode==0,result.stderr
    else:assert result.returncode!=0 and ('seed inventory path outside' if kind=='paths' else 'full-process includes scope differs') in result.stderr,result.stderr
    count+=1
 print(('BEFORE accepted ' if before else 'PASS rejected ')+str(count)+' resealed '+kind+' mutations')
