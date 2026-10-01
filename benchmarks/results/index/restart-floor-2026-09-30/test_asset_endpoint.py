#!/usr/bin/env python3
"""Read-only endpoint negatives reseal manifests without altering historical files."""
import json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_query_schedule as t

def drop_path(x):x['prehashed_source_tokenizer_onnx_assets_unchanged'].pop()
def duplicate_path(x):x['prehashed_source_tokenizer_onnx_assets_unchanged'][-1]=x['prehashed_source_tokenizer_onnx_assets_unchanged'][0]
def release_original(x):x['prehashed_source_tokenizer_onnx_assets_unchanged'].pop(0)
def invented_release(x):x['released_sources']=[x['prehashed_source_tokenizer_onnx_assets_unchanged'].pop(2)]
def drop_external(x):x['prepared_external_weights_post_trial'].pop()
def wrong_bytes(x):x['prepared_external_weights_post_trial'][0]['bytes']+=1
def wrong_sha(x):x['prepared_external_weights_post_trial'][0]['sha256']='0'*64
def wrong_mtime(x):x['prepared_external_weights_post_trial'][0]['mtime_ns']+=1
def certified_scope(x):x['prepared_external_weights_post_trial'][0]['hash_scope']='independently hashed before timing'

if __name__=='__main__':
    checks=[drop_path,duplicate_path,release_original,invented_release,drop_external,wrong_bytes,wrong_sha,wrong_mtime,certified_scope]
    for change in [None,*checks]:
        with tempfile.TemporaryDirectory(prefix='hnsw-asset-endpoint-') as td:
            repo=Path(td);t.copy_archive(repo)
            if change:
                path=repo/t.HNSW/'post-trial-assets.json';value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value,indent=2)+'\n')
            t.reseal(repo);result=t.execute(repo)
            if change:assert result.returncode!=0 and 'HNSW ' in result.stderr and 'endpoint' in result.stderr,result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: HNSW positive and 9 resealed endpoint path/identity/receipt/release/scope negatives')
