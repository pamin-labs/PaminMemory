#!/usr/bin/env python3
"""Adjacent seed and raw/native chronology boundaries; no measurement."""
import gzip,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_query_schedule as t

def change_json(path,change):
    value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value,indent=2)+'\n')
def seed_profile(repo):change_json(repo/t.HNSW/'provenance.json',lambda x:next(e for e in x['stopped_seed_index_files'] if e['path'].endswith('/profile')).update(sha256='0'*64))
def seed_floor(repo):change_json(repo/t.HNSW/'provenance.json',lambda x:next(e for e in x['stopped_seed_index_files'] if e['path'].endswith('/.pamin-optimized-files')).update(bytes=1))
def seed_documents(repo):change_json(repo/t.HNSW/'provenance.json',lambda x:x.update(seed_documents=18001))
def chronology(repo,rel,raw_also):
    log=repo/rel/'logs'/('0-main.log.gz' if rel==t.DISK else '0-main.log')
    lines=(gzip.decompress(log.read_bytes()).decode() if rel==t.DISK else log.read_text()).splitlines()
    positions=[i for i,line in enumerate(lines) if line.startswith('RESTART_JSON ')]
    a,b=positions[-2:];lines[a],lines[b]=lines[b],lines[a];value='\n'.join(lines)+'\n'
    if rel==t.DISK:log.write_bytes(gzip.compress(value.encode(),mtime=0))
    else:log.write_text(value)
    if raw_also:
        path=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        positions=[i for i,row in enumerate(rows) if row['arm']=='main' and row['repetition']==0]
        a,b=positions[-3:-1];rows[a],rows[b]=rows[b],rows[a]
        # Per-phase summary/timing/table cells do not change when rows reorder.
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

if __name__=='__main__':
    checks=[(t.HNSW,seed_profile,'seed profile/floor'),(t.HNSW,seed_floor,'seed profile/floor'),(t.HNSW,seed_documents,'seed corpus')]
    for rel in [t.HNSW,t.DISK]:
        for raw_also in [False,True]:checks.append((rel,lambda repo,rel=rel,raw_also=raw_also:chronology(repo,rel,raw_also),'product phase chronology' if raw_also else 'native/raw chronology'))
    for rel,change,guard in checks:
        with tempfile.TemporaryDirectory(prefix='restart-boundaries-') as td:
            repo=Path(td);t.copy_archive(repo);change(repo);t.reseal(repo)
            result=subprocess.run([sys.executable,'-B',str(repo/rel/'verify.py')],capture_output=True,text=True)
            assert result.returncode!=0 and guard in result.stderr,result.stderr
    print('PASS: 7 resealed seed/profile/floor and matching raw/log chronology negatives')
