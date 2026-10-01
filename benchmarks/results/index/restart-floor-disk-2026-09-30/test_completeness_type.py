"""Resealed matching raw/native corruptions reject bool/nonfinite completeness."""
import gzip,json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive
if __name__=='__main__':
 names=set()
 for rel in [archive.DISK,archive.HNSW]:
  names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
 gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
 for rel in [archive.DISK,archive.HNSW]:
  for value in [True,False,'1.0',float('nan'),float('inf'),float('-inf')]:
   with tempfile.TemporaryDirectory(prefix='restart-completeness-negative-') as directory:
    repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
    for name in names:
     dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
    raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
    row=next(row for row in rows if row['phase']=='maintenance' and row['arm']=='main');row['extra']['completeness']=value
    raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    compressed=rel==archive.DISK;log=repo/rel/'logs'/f"{row['repetition']}-{row['arm']}.log{'.gz' if compressed else ''}"
    text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();lines=text.splitlines()
    at=next(i for i,line in enumerate(lines) if line.startswith('RESTART_JSON ') and json.loads(line.removeprefix('RESTART_JSON '))['phase']=='maintenance')
    observed=json.loads(lines[at].removeprefix('RESTART_JSON '));observed['extra']['completeness']=value;lines[at]='RESTART_JSON '+json.dumps(observed);text='\n'.join(lines)+'\n'
    if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
    else:log.write_text(text)
    archive.manifests(repo);result=archive.execute(repo,rel)
    assert result.returncode!=0 and ('completeness must be finite numeric' in result.stderr or 'native log observation' in result.stderr or 'native/raw chronology' in result.stderr),result.stderr
 print('PASS: both archives reject 12 resealed raw/log bool, string and nonfinite completeness corruptions')
