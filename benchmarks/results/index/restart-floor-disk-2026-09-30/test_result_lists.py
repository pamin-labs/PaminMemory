#!/usr/bin/env python3
"""Matching raw/log duplicate result lists with refreshed artifact hashes."""
import gzip,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_fixture',ROOT.with_name('restart-floor-2026-09-30')/'test_query_schedule.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
def duplicate_non_target(extra):
    topics=extra['topics'];target=f"incident-{extra['query_document']}";positions=[i for i,x in enumerate(topics) if x!=target]
    topics[positions[-1]]=topics[positions[0]]
def corrupt(repo):
    path=repo/t.DISK/'raw.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    for row in rows:
        if row['phase']=='search_warm':duplicate_non_target(row['extra'])
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    for p in (repo/t.DISK/'logs').glob('[012]-*.log.gz'):
        lines=gzip.decompress(p.read_bytes()).decode().splitlines()
        for at,line in enumerate(lines):
            if line.startswith('RESTART_JSON '):
                row=json.loads(line.removeprefix('RESTART_JSON '))
                if row['phase']=='search_warm':duplicate_non_target(row['extra']);lines[at]='RESTART_JSON '+json.dumps(row)
        p.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
if __name__=='__main__':
    for mutation in [False,True]:
        with tempfile.TemporaryDirectory(prefix='disk-result-list-') as td:
            repo=Path(td);t.copy_archive(repo)
            if mutation:corrupt(repo)
            t.reseal(repo);result=subprocess.run([sys.executable,'-B',str(repo/t.DISK/'verify.py')],capture_output=True,text=True)
            if mutation:assert result.returncode!=0 and 'Disk ordered top10 topics must be unique' in result.stderr,result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: positive Disk archive and matching all-arm raw/log duplicate-topics negative with refreshed hashes; target ranks and paired equality preserved; no Cargo/models/DB')
