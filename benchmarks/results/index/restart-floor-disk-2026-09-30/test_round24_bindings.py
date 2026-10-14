"""Resealed negatives for Disk hardware, asset identity, seed cardinality and post-trial status, plus HNSW provenance and README-claim bindings."""
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
def truthy_post_status(repo):archive.update(repo/archive.DISK/'post-trial-assets.json',lambda x:x.update(all_recorded_source_graphs_and_prepared_external_weights_unchanged=1))
def truthy_asset_status(repo):archive.update(repo/archive.DISK/'post-trial-assets.json',lambda x:x['assets'][0].update(unchanged='yes'))
def hnsw_runtime(repo):archive.update(repo/archive.HNSW/'provenance.json',lambda x:x.update(runtime_commit='0'*40))
def hnsw_toolchain(repo):archive.update(repo/archive.HNSW/'provenance.json',lambda x:x.update(cargo='cargo 0.0.0'))
def hnsw_library(repo):archive.update(repo/archive.HNSW/'provenance.json',lambda x:x['native_zvec_library'].update(sha256='0'*64))
def hnsw_library_path(repo):archive.update(repo/archive.HNSW/'provenance.json',lambda x:x['runtime_library'].update(path='${ORT_DIST}/lib/libonnxruntime.so'))
def hnsw_model_identity(field,value):
    # Change the HNSW and consumed Disk tokenizer entries consistently, so only
    # the identity-value check (not cross-archive equality) can reject them.
    def change(repo):
        for rel,key in [(archive.HNSW,'model_assets'),(archive.DISK,'source_assets')]:
            archive.update(repo/rel/'provenance.json',lambda x,key=key:next(e for e in x[key] if e['path'].endswith('bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/tokenizer.json')).update({field:value}))
    change.__name__=f'hnsw_model_{field}_{value!r}';return change
def hnsw_cpu_sentence(repo):
    p=repo/archive.HNSW/'README.md';p.write_text(p.read_text().replace('0.44 s for main','0.40 s for main'))
def hnsw_trigger_qualifier(repo):
    p=repo/archive.HNSW/'README.md';p.write_text(p.read_text().replace(' not retained per-arm evidence.',' confirmed per arm.'))

CASES=[(hardware,archive.DISK,'retained Disk hardware provenance differs'),
       (asset,archive.DISK,'model asset identity must have nonnegative integer bytes and canonical SHA256'),
       (seed,archive.DISK,'retained complete seed inventory size differs'),
       (library_scope,archive.DISK,'post-trial library rehash scope differs'),
       (hnsw_seed,archive.HNSW,'HNSW stopped seed inventory digest differs'),
       (unbound_cross_archive,archive.HNSW,'HNSW manifest must bind cross-archive input'),
       (truthy_post_status,archive.DISK,'post-trial asset status must be exact true'),
       (truthy_asset_status,archive.DISK,'post-trial asset status must be exact true'),
       (hnsw_runtime,archive.HNSW,'HNSW runtime provenance differs from retained Disk record: runtime_commit'),
       (hnsw_toolchain,archive.HNSW,'HNSW runtime provenance differs from retained Disk record: cargo'),
       (hnsw_library,archive.HNSW,'HNSW runtime provenance differs from retained Disk record: native_zvec_library'),
       (hnsw_library_path,archive.HNSW,'HNSW runtime provenance differs from retained Disk record: runtime_library'),
       *[(hnsw_model_identity(field,value),archive.HNSW,'HNSW model identity must have nonnegative integer bytes and canonical SHA256') for field,value in [('bytes',-1),('bytes',True),('bytes',1.0),('sha256','A'*64),('sha256','a'*63)]],
       (hnsw_cpu_sentence,archive.HNSW,'HNSW published upkeep CPU sentence differs from recomputed summary'),
       (hnsw_trigger_qualifier,archive.HNSW,'HNSW optimize-trigger qualification missing from README')]

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
    print(f'PASS: {len(CASES)} resealed hardware, asset, seed-inventory, library-scope, manifest-binding, post-trial status, HNSW runtime/model-identity and README-claim negatives')
