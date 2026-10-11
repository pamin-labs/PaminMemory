#!/usr/bin/env python3
"""Matching raw/native-log and resealed Disk initial-query schedule negatives."""
import gzip,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_tests',ROOT.with_name('restart-floor-2026-09-30')/'test_query_schedule.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def corrupt(repo,phase,change):
    path=repo/t.DISK/'raw.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    for arm in ['main','predecessor','candidate']:
        for rep in range(3):
            targets=[r for r in rows if r['arm']==arm and r['repetition']==rep and r['phase']==phase];change(targets)
            log=repo/t.DISK/'logs'/f'{rep}-{arm}.log.gz';lines=gzip.decompress(log.read_bytes()).decode().splitlines();index=0
            for at,line in enumerate(lines):
                if line.startswith('RESTART_JSON '):
                    row=json.loads(line.removeprefix('RESTART_JSON '))
                    if row['phase']==phase:
                        row['extra']['query_document']=targets[index]['extra']['query_document'];index+=1
                        lines[at]='RESTART_JSON '+json.dumps(row)
            assert index==len(targets);log.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

if __name__=='__main__':
    checks=[('search_cold',t.wrong_id),('search_warmup',t.wrong_id),('search_warmup',t.wrong_order)]
    for check in [None,*checks]:
        with tempfile.TemporaryDirectory(prefix='disk-initial-query-') as td:
            repo=Path(td);t.copy_archive(repo)
            if check:corrupt(repo,*check)
            t.reseal(repo);result=subprocess.run([sys.executable,'-B',str(repo/t.DISK/'verify.py')],capture_output=True,text=True)
            if check:assert result.returncode!=0 and 'Disk exact ordered cold/warmup query schedule differs' in result.stderr,result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: Disk positive and 3 matching raw/log, resealed cold/warmup negatives across 9 processes')
