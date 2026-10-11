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

def library_scope(repo):archive.update(repo/archive.DISK/'post-trial-libraries.json',lambda x:x.update(scope='pretrial values copied after run; files were not rehashed'))
def hnsw_seed(repo):
    def change(x):
        item=next(e for e in x['stopped_seed_index_files'] if not e['path'].endswith('/profile') and not e['path'].endswith('/.pamin-optimized-files'))
        item['path']=item['path'].rsplit('/',1)[0]+'/fabricated-segment'
    archive.update(repo/archive.HNSW/'provenance.json',change)
def unbound_cross_archive(repo):
    path=repo/archive.HNSW/'manifest.json';manifest=json.loads(path.read_text())
    manifest['files'].pop(str(archive.DISK/'provider-bindings.json'));path.write_text(json.dumps(manifest,indent=2)+'\n')

CASES=[(hardware,archive.DISK,'retained Disk hardware provenance differs'),
       (asset,archive.DISK,'model asset identity must have nonnegative integer bytes and canonical SHA256'),
       (seed,archive.DISK,'retained complete seed inventory size differs'),
       (library_scope,archive.DISK,'post-trial library rehash scope differs'),
       (hnsw_seed,archive.HNSW,'HNSW stopped seed inventory digest differs'),
       (unbound_cross_archive,archive.HNSW,'HNSW manifest must bind cross-archive input')]

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
    print(f'PASS: {len(CASES)} resealed hardware, asset, seed-inventory, library-scope and manifest-binding negatives')
