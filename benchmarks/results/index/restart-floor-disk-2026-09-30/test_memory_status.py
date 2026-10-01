#!/usr/bin/env python3
"""Resealed raw/log corruptions must fail before archive RSS recomputation."""
import gzip,json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive

ROOT=archive.ROOT;REPO=archive.REPO

if __name__=='__main__':
    names=set()
    for rel in [archive.DISK,archive.HNSW]:
        names.update(json.loads((REPO/rel/'manifest.json').read_text())['files'])
        names.add(str(rel/'manifest.json'))
    gitdir=subprocess.check_output(['git','-C',str(REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
    corruptions=[['VmRSS: -1 kB','VmHWM: 2 kB'],['VmRSS: 1 kB','VmHWM: -1 kB'],['VmRSS: 1 kB','VmRSS: 2 kB'],['VmHWM: 1 kB','VmHWM: 2 kB'],['VmRSS: 1 kB'],['VmRSS: +1 kB','VmHWM: 2 kB'],['VmRSS: 1 MB','VmHWM: 2 kB'],['VmRSS: 3 kB','VmHWM: 2 kB'],['VmRSS: 1 kB','VmHWM: 2 kB','VmRSS: 1 kB']]
    for rel in [archive.DISK,archive.HNSW]:
        for field in ['process_before','process_after']:
            for status in corruptions:
                with tempfile.TemporaryDirectory(prefix='restart-rss-negative-') as temp:
                    repo=Path(temp);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
                    for name in names:
                        dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dst)
                    raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
                    row=rows[0];row[field]['rss_status']=status
                    raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    compressed=rel==archive.DISK
                    log=repo/rel/'logs'/f"{row['repetition']}-{row['arm']}.log{'.gz' if compressed else ''}"
                    text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text()
                    lines=text.splitlines()
                    index=next(i for i,line in enumerate(lines) if line.startswith('RESTART_JSON '))
                    observation=json.loads(lines[index].removeprefix('RESTART_JSON '));observation[field]['rss_status']=status
                    lines[index]='RESTART_JSON '+json.dumps(observation);text='\n'.join(lines)+'\n'
                    if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
                    else:log.write_text(text)
                    archive.manifests(repo);result=archive.execute(repo,rel)
                    assert result.returncode!=0 and ('RSS/HWM' in result.stderr or 'RSS exceeds' in result.stderr),result.stderr
    print('PASS: both archives reject resealed negative, duplicate, missing, malformed and contradictory RSS/HWM before analysis')
