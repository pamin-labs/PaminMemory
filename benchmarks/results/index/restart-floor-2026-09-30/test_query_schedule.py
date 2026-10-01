#!/usr/bin/env python3
"""Read-only archive checks; temporary copies reseal matching raw/log corruptions."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[3]
HNSW=ROOT.relative_to(REPO);DISK=HNSW.with_name('restart-floor-disk-2026-09-30')

def copy_archive(repo):
    for rel in [HNSW,DISK,Path('benchmarks/harnesses/restart-floor-2026-09-30')]:
        shutil.copytree(REPO/rel,repo/rel)
    gitdir=subprocess.check_output(['git','-C',str(REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
    (repo/'.git').symlink_to(gitdir,target_is_directory=True)

def reseal(repo):
    for rel in [HNSW,DISK]:
        path=repo/rel/'manifest.json';manifest=json.loads(path.read_text())
        for name in manifest['files']:
            manifest['files'][name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest,indent=2)+'\n')

def corrupt(repo,phase,change):
    path=repo/HNSW/'raw.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    # Mutate all nine processes alike: uniqueness and cross-arm paired IDs still
    # agree. Native RESTART_JSON records agree too, so only the schedule catches it.
    for arm in ['main','predecessor','candidate']:
        for rep in range(3):
            targets=[r for r in rows if r['arm']==arm and r['repetition']==rep and r['phase']==phase]
            change(targets)
            log=repo/HNSW/'logs'/f'{rep}-{arm}.log';lines=log.read_text().splitlines();index=0
            for at,line in enumerate(lines):
                if line.startswith('RESTART_JSON '):
                    row=json.loads(line.removeprefix('RESTART_JSON '))
                    if row['phase']==phase:
                        row['extra']['query_document']=targets[index]['extra']['query_document'];index+=1
                        lines[at]='RESTART_JSON '+json.dumps(row)
            assert index==len(targets)
            log.write_text('\n'.join(lines)+'\n')
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

def wrong_id(rows):rows[0]['extra']['query_document']+=1
def wrong_order(rows):
    first,second=rows[0]['extra']['query_document'],rows[1]['extra']['query_document']
    rows[0]['extra']['query_document']=second;rows[1]['extra']['query_document']=first

def execute(repo):
    return subprocess.run([sys.executable,'-B',str(repo/HNSW/'verify.py')],capture_output=True,text=True)

if __name__=='__main__':
    checks=[('search_cold',wrong_id),('search_warmup',wrong_id),('search_warmup',wrong_order),('search_warm',wrong_id),('search_warm',wrong_order)]
    for check in [None,*checks]:
        with tempfile.TemporaryDirectory(prefix='restart-query-schedule-') as td:
            repo=Path(td);copy_archive(repo)
            if check:corrupt(repo,*check)
            reseal(repo);result=execute(repo)
            if check:
                assert result.returncode!=0, str(check)+' corruption accepted'
                guard='HNSW exact ordered '+('warm' if check[0]=='search_warm' else 'cold/warmup')+' query schedule differs'
                assert guard in result.stderr, str(check)+' did not reach schedule guard: '+result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: positive HNSW archive and 5 matching raw/log, refreshed-manifest schedule negatives across all 9 processes; no Cargo/models/DB')
