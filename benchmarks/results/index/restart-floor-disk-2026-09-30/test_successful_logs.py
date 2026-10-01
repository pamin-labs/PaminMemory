#!/usr/bin/env python3
"""Resealed success-marker negatives preserve matching raw/native observations."""
import gzip,importlib.util,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_tests',ROOT.with_name('restart-floor-2026-09-30')/'test_query_schedule.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def mutate(log,case):
    summary=next(line for line in log.splitlines() if line.startswith('test result:'))
    if case=='failed_after':return log+'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n'
    if case=='multiple':return log+summary+'\n'
    if case=='zero':return log.replace(summary,summary.replace('1 passed','0 passed'))
    if case=='filtered':return log.replace(summary,summary.replace('0 filtered out','1 filtered out'))
    if case=='truncated':return log.replace(summary,'test result: ok. 1 passed;')
    if case=='not_final':return log+'unexpected trailing diagnostic\n'
    raise AssertionError(case)

if __name__=='__main__':
    count=0
    for rel in [t.HNSW,t.DISK]:
        for case in [None,'failed_after','multiple','zero','filtered','truncated','not_final']:
            with tempfile.TemporaryDirectory(prefix='restart-success-log-') as td:
                repo=Path(td);t.copy_archive(repo)
                if case:
                    suffix='.log.gz' if rel==t.DISK else '.log'
                    for arm in ['main','predecessor','candidate']:
                        for rep in range(3):
                            path=repo/rel/'logs'/f'{rep}-{arm}{suffix}'
                            log=gzip.decompress(path.read_bytes()).decode() if rel==t.DISK else path.read_text()
                            changed=mutate(log,case)
                            assert [l for l in log.splitlines() if l.startswith('RESTART_JSON ')]==[l for l in changed.splitlines() if l.startswith('RESTART_JSON ')]
                            if rel==t.DISK:path.write_bytes(gzip.compress(changed.encode(),mtime=0))
                            else:path.write_text(changed)
                t.reseal(repo);result=subprocess.run([sys.executable,'-B',str(repo/rel/'verify.py')],capture_output=True,text=True)
                if case:
                    assert result.returncode!=0 and 'retained test log must end with exactly one complete successful one-test summary' in result.stderr,result.stderr
                    count+=1
                else:assert result.returncode==0,result.stderr
    print(f'PASS: both archives and {count} resealed final-success negatives, matching raw/native observations across all9 processes')
