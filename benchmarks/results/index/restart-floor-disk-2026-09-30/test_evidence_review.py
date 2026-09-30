#!/usr/bin/env python3
"""Read-only positive and refreshed-manifest semantic negatives in temporary copies."""
import gzip,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3]
DISK=ROOT.relative_to(REPO);HNSW=DISK.with_name('restart-floor-2026-09-30')

def update(path,change):
    value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value,indent=2)+'\n')
def manifests(repo):
    for rel in [DISK,HNSW]:
        p=repo/rel/'manifest.json';value=json.loads(p.read_text())
        for name in value['files']:value['files'][name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
        p.write_text(json.dumps(value,indent=2)+'\n')
def execute(repo,rel=DISK):
    return subprocess.run([sys.executable,'-B',str(repo/rel/'verify.py')],capture_output=True,text=True)
def missing_binary(repo):update(repo/DISK/'binaries.json',lambda x:x.pop())
def duplicate_arm(repo):update(repo/DISK/'binaries.json',lambda x:x.__setitem__(1,x[0]))
def wrong_commit(repo):update(repo/DISK/'binaries.json',lambda x:x[0].update(commit='0'*40))
def wrong_path(repo):update(repo/DISK/'binaries.json',lambda x:x[0].update(binary='<SCRATCH>/bin/unbound'))
def wrong_bytes(repo):update(repo/DISK/'binaries.json',lambda x:x[0]['current_pretrial_hash'].update(bytes=1))
def runner(repo):
    p=repo/'benchmarks/harnesses/restart-floor-2026-09-30/run.py.in';p.write_text(p.read_text()+'\n# changed historical runner\n')
def failed_seed(repo):
    p=repo/DISK/'logs/seed-disk.log.gz';s=gzip.decompress(p.read_bytes()).decode();s=s.replace('test result: ok. 1 passed; 0 failed;','test result: FAILED. 0 passed; 1 failed;');p.write_bytes(gzip.compress(s.encode(),mtime=0))
def failed_conversion(repo):
    p=repo/DISK/'logs/disk-conversion.log.gz';s=gzip.decompress(p.read_bytes()).decode();s=s.replace('test result: ok. 1 passed; 0 failed;','test result: FAILED. 0 passed; 1 failed;');p.write_bytes(gzip.compress(s.encode(),mtime=0))
def wrong_role_graph(repo):update(repo/DISK/'provider-bindings.json',lambda x:x['roles']['embedding'].update(prepared_graph=x['roles']['reranker']['prepared_graph']))
def wrong_metadata(repo):update(repo/DISK/'provider-bindings.json',lambda x:x['roles']['embedding']['source_metadata'].update(text='changed\n'))
def wrong_source_revision(repo):update(repo/DISK/'provider-bindings.json',lambda x:x['roles']['embedding'].update(revision='0'*40))
def wrong_external_data(repo):update(repo/DISK/'provider-bindings.json',lambda x:x['roles']['embedding'].update(external_data=x['roles']['reranker']['external_data']))
def changed_screen(repo):update(repo/DISK/'timing-review.json',lambda x:x['predecessor']['write_ms'].update(withhold_comparison=False))
def shown_unstable(repo):
    p=repo/DISK/'README.md';s=p.read_text();s=s.replace('| Write + urgent drain wall median | 1421.857852 ms | 1388.009180 ms | Withheld: unstable three-process sample | Withheld: unstable three-process sample |','| Write + urgent drain wall median | 1421.857852 ms | 1388.009180 ms | -33.848672 ms | -2.381% |');p.write_text(s)
def hnsw_screen(repo):update(repo/HNSW/'timing-review.json',lambda x:x['predecessor']['warm_p50_ms'].update(withhold_comparison=False))
checks=[missing_binary,duplicate_arm,wrong_commit,wrong_path,wrong_bytes,runner,failed_seed,failed_conversion,wrong_role_graph,wrong_metadata,wrong_source_revision,wrong_external_data,changed_screen,shown_unstable,hnsw_screen]
if __name__=='__main__':
    for rel in [DISK,HNSW]:
        result=execute(REPO,rel);assert result.returncode==0,result.stderr
    names=set()
    for rel in [DISK,HNSW]:names.update(json.loads((REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
    gitdir=subprocess.check_output(['git','-C',str(REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
    for change in checks:
        with tempfile.TemporaryDirectory(prefix='restart-review-negative-') as td:
            repo=Path(td);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
            for name in names:
                dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dst)
            change(repo);manifests(repo);result=execute(repo,HNSW if change==hnsw_screen else DISK)
            assert result.returncode!=0,change.__name__+' corruption accepted'
            assert 'AssertionError' in result.stderr and 'manifest' not in result.stderr.split('AssertionError')[-1],change.__name__+' did not reach semantic guard: '+result.stderr
    print('PASS: both positive archives and15 refreshed-manifest semantic negatives; no Cargo/models/DB')
