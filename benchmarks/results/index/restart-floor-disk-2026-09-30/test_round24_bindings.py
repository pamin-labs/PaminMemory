"""Resealed negatives for Disk hardware, asset identity and seed cardinality bindings."""
import json,shutil,subprocess,sys,tempfile
sys.dont_write_bytecode=True
from pathlib import Path
import test_evidence_review as archive

def hardware(repo):archive.update(repo/archive.DISK/'provenance.json',lambda x:x['hardware'].update(kernel='CP/M',cpu_model='Z80',cgroup_memory_max='1024'))
def asset(repo):
    for name,key in [('provenance.json','prepared_graphs_and_external_weights'),('post-trial-assets.json','assets')]:
        def change(x,key=key):
            item=next(e for e in x[key] if e['path'].endswith('attention.onnx'));item.update(bytes=-1,sha256='not-a-sha256')
            if 'posttrial_sha256' in item:item['posttrial_sha256']='not-a-sha256'
        archive.update(repo/archive.DISK/name,change)
def seed(repo):
    import hashlib
    def change(x):
        dropped=next(e for e in x['seed_files'] if not e['path'].endswith('server.json'));x['seed_files'].remove(dropped)
    archive.update(repo/archive.DISK/'provenance.json',change)
    entries=json.loads((repo/archive.DISK/'provenance.json').read_text())['seed_files']
    inventory={e['path']:[e['bytes'],e['sha256']] for e in entries}
    digest=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    archive.update(repo/archive.DISK/'post-review-seed.json',lambda x:x.update(file_count=len(entries),total_bytes=sum(e['bytes'] for e in entries),canonical_inventory_sha256=digest))

CASES=[(hardware,archive.DISK,'retained Disk hardware provenance differs'),
       (asset,archive.DISK,'model asset identity must have nonnegative integer bytes and canonical SHA256'),
       (seed,archive.DISK,'retained complete seed inventory size differs')]

if __name__=='__main__':
    names=set()
    for rel in [archive.DISK,archive.HNSW]:names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
    gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
    for change,rel,reason in CASES:
        with tempfile.TemporaryDirectory(prefix='restart-round24-') as td:
            repo=Path(td);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
            for name in names:
                dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
            change(repo);archive.manifests(repo);result=archive.execute(repo,rel)
            assert result.returncode!=0 and reason in result.stderr,(change.__name__,result.stderr[-2000:])
    print(f'PASS: {len(CASES)} resealed Disk hardware, asset identity and complete-seed-size negatives')
