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
def hnsw_readme_cell(repo,reference,column,replacement):
    p=repo/HNSW/'README.md';lines=p.read_text().splitlines();section='predecessor';changed=0
    for at,line in enumerate(lines):
        if line.startswith('## Combined'):section='main'
        if section==reference and line.startswith('| Warm search p95: median across processes |'):
            cells=line.split('|');cells[column]=' '+replacement+' ';lines[at]='|'.join(cells);changed+=1
    assert changed==1, 'expected stable HNSW row missing'
    p.write_text('\n'.join(lines)+'\n')
def hnsw_predecessor_delta(repo):hnsw_readme_cell(repo,'predecessor',4,'-999.000 ms')
def hnsw_predecessor_percent(repo):hnsw_readme_cell(repo,'predecessor',5,'-99.000%')
def hnsw_main_delta(repo):hnsw_readme_cell(repo,'main',4,'-999.000 ms')
def hnsw_main_percent(repo):hnsw_readme_cell(repo,'main',5,'-99.000%')
def hnsw_stable_withheld(repo):hnsw_readme_cell(repo,'main',4,'Withheld: unstable three-process sample')
def hnsw_missing_row(repo):
    p=repo/HNSW/'README.md';lines=p.read_text().splitlines();index=next(i for i,l in enumerate(lines) if l.startswith('| Maintenance wall median |'));lines.pop(index);p.write_text('\n'.join(lines)+'\n')
def uncensored_maintenance_cpu(repo):
    p=repo/DISK/'README.md';s=p.read_text().replace('<0.020 s at combined counter resolution (0 observed ticks) | Withheld: censored counter observation | Withheld: censored counter observation','0.000000 s | -27.320000 s | -100.000%',1);p.write_text(s)
def wrong_cpu_resolution_bound(repo):
    p=repo/DISK/'README.md';s=p.read_text().replace('<0.020 s at combined counter resolution','<0.001 s at combined counter resolution',1);p.write_text(s)
checks=[uncensored_maintenance_cpu,wrong_cpu_resolution_bound,missing_binary,duplicate_arm,wrong_commit,wrong_path,wrong_bytes,runner,failed_seed,failed_conversion,wrong_role_graph,wrong_metadata,wrong_source_revision,wrong_external_data,changed_screen,shown_unstable,hnsw_screen,hnsw_predecessor_delta,hnsw_predecessor_percent,hnsw_main_delta,hnsw_main_percent,hnsw_stable_withheld,hnsw_missing_row]
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
            change(repo);manifests(repo);result=execute(repo,HNSW if change.__name__.startswith('hnsw_') else DISK)
            assert result.returncode!=0,change.__name__+' corruption accepted'
            assert 'AssertionError' in result.stderr and 'manifest' not in result.stderr.split('AssertionError')[-1],change.__name__+' did not reach semantic guard: '+result.stderr
            if change.__name__.startswith('hnsw_') and change!=hnsw_screen:
                assert 'HNSW complete displayed rows disagree' in result.stderr,change.__name__+' did not reach exact table guard: '+result.stderr
    print(f'PASS: both positive archives and {len(checks)} refreshed-manifest semantic negatives; no Cargo/models/DB')
