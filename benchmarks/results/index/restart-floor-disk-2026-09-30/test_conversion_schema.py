#!/usr/bin/env python3
"""Matching conversion-log/provenance and resealed canonical-schema negatives."""
import gzip,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('schedule_tests',ROOT.with_name('restart-floor-2026-09-30')/'test_query_schedule.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def corrupt(repo,case):
    root=repo/t.DISK;path=root/'provenance.json';provenance=json.loads(path.read_text())
    records=provenance['conversion_schema_and_logical_digest'];before,after=[r['schema'] for r in records]
    if case=='already_disk':before['fields']['embedding']['index_type']=5
    elif case=='before_dimension':before['fields']['embedding']['dimension']=512
    elif case=='before_quantize':before['fields']['embedding']['quantize']=1
    elif case=='both_segment':before['segment_documents']=after['segment_documents']=1000
    elif case=='before_nullable':before['fields']['id']['nullable']=True
    elif case=='both_text':before['fields']['content_ngram']['metric']=after['fields']['content_ngram']['metric']=2
    elif case=='after_text':after['fields']['content_segmented']['index_type']=1
    elif case=='after_nullable':after['fields']['embedding']['nullable']=True
    elif case=='after_extra':after['fields']['embedding']['unexpected']=0
    elif case=='before_missing':del before['fields']['embedding']['quantizer_rotate']
    else:raise AssertionError(case)
    path.write_text(json.dumps(provenance,indent=2)+'\n')
    log=root/'logs/disk-conversion.log.gz';lines=gzip.decompress(log.read_bytes()).decode().splitlines();index=0
    for at,line in enumerate(lines):
        if 'DISK_SETUP_JSON ' in line:
            prefix=line.split('DISK_SETUP_JSON ',1)[0]
            lines[at]=prefix+'DISK_SETUP_JSON '+json.dumps(records[index]);index+=1
    assert index==2
    log.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))

if __name__=='__main__':
    checks=['already_disk','before_dimension','before_quantize','both_segment','before_nullable','both_text','after_text','after_nullable','after_extra','before_missing']
    for case in [None,*checks]:
        with tempfile.TemporaryDirectory(prefix='disk-conversion-schema-') as td:
            repo=Path(td);t.copy_archive(repo)
            if case:corrupt(repo,case)
            t.reseal(repo);result=subprocess.run([sys.executable,'-B',str(repo/t.DISK/'verify.py')],capture_output=True,text=True)
            if case:
                guards=['complete retained preconversion HNSW schema differs','conversion changed nonembedding schema field','complete retained postconversion DiskANN schema differs']
                assert result.returncode!=0 and any(g in result.stderr for g in guards),result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: Disk positive and10 matching conversion-log/provenance resealed canonical-schema negatives; no historical schema values invented')
