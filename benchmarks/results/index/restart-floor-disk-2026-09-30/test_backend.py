#!/usr/bin/env python3
"""Resealed retained-log backend contradictions; no runtime/model/database."""
import gzip,importlib.util,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_fixture',ROOT.with_name('restart-floor-2026-09-30')/'test_query_schedule.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
def corrupt(repo,mutation):
    for p in (repo/t.DISK/'logs').glob('[012]-*.log.gz'):
        lines=gzip.decompress(p.read_bytes()).decode().splitlines();at=next(i for i,x in enumerate(lines) if 'DiskAnn will use synchronous pread();' in x)
        if mutation=='duplicate':lines.insert(at,lines[at])
        elif mutation=='missing':lines.pop(at)
        elif mutation=='contradictory':lines[at]=lines[at].replace('will use synchronous pread();','will not use synchronous pread();')
        else:lines.append('[ INFO diskann_file_reader.cc:99] DiskAnn: selected '+mutation+' async I/O backend; synchronous pread() disabled.')
        p.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
if __name__=='__main__':
    for mutation in [None,'duplicate','missing','contradictory','io_uring','libaio']:
        with tempfile.TemporaryDirectory(prefix='disk-backend-') as td:
            repo=Path(td);t.copy_archive(repo)
            if mutation:corrupt(repo,mutation)
            t.reseal(repo);result=subprocess.run([sys.executable,'-B',str(repo/t.DISK/'verify.py')],capture_output=True,text=True)
            if mutation:assert result.returncode!=0 and 'Disk native backend diagnostic must be exactly one' in result.stderr,result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: positive Disk archive and 5 refreshed-hash backend negatives across all9 processes; no Cargo/models/DB')
