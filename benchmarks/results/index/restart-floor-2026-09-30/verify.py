#!/usr/bin/env python3
"""Read-only verifier and independent invocation of the archived calculator."""
import hashlib,importlib.machinery,importlib.util,json
from pathlib import Path
root=Path(__file__).resolve().parent
repo=root.parents[3]
manifest=json.loads((root/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
    assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==digest, relative
raw=[json.loads(line) for line in (root/'raw.jsonl').read_text().splitlines()]
assert len([r for r in raw if r['phase']=='process_total'])==9
for arm in ['main','predecessor','candidate']:
    for rep in range(3):
        rows=[r for r in raw if r['arm']==arm and r['repetition']==rep]
        upkeep=[r for r in rows if r['phase']=='maintenance']
        assert len(upkeep)==1 and upkeep[0]['extra']['optimize_completed_delta']==(arm!='candidate')
        warm=[r for r in rows if r['phase']=='search_warm']
        assert len(warm)==24 and len({r['extra']['query_document'] for r in warm})==24
        assert len([r for r in rows if r['phase']=='search_cold'])==1
        assert len([r for r in rows if r['phase']=='search_warmup'])==2
        new=next(r for r in rows if r['phase']=='new_write_search')
        assert new['extra']['known_new_topic_retrieved'] and 'restart-proof' in new['extra']['topics']
        for r in rows:
            if 'actual_providers' in r:
                assert set(r['actual_providers'])=={'embedding','reranker'}
                for provider in r['actual_providers'].values():
                    assert set(provider['assigned_nodes'])=={'CPUExecutionProvider'}
                    assert provider['assigned_nodes']['CPUExecutionProvider']>0
        for reference in ['main','predecessor']:
            if arm=='candidate':
                before={r['extra']['query_document']:r['extra']['topics'] for r in raw if r['arm']==reference and r['repetition']==rep and r['phase']=='search_warm'}
                assert all(r['extra']['topics']==before[r['extra']['query_document']] for r in warm)
source=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in'
loader=importlib.machinery.SourceFileLoader('archived_restart_analysis',str(source))
spec=importlib.util.spec_from_loader(loader.name,loader)
module=importlib.util.module_from_spec(spec);loader.exec_module(module)
assert module.summarize(raw)==json.loads((root/'summary.json').read_text())
print('verified: 9 rotated processes; 72/72 paired ordered lists per reference; actual providers, maintenance work, new-write visibility and recomputed metrics')
