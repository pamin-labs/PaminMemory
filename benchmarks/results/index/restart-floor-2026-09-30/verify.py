#!/usr/bin/env python3
"""Read-only verifier and independent invocation of the archived calculator."""
import hashlib,importlib.machinery,importlib.util,json,sys
if sys.flags.optimize:
    raise SystemExit("Verification requires Python assertions; remove -O/-OO or PYTHONOPTIMIZE.")
sys.dont_write_bytecode=True
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
summary=module.summarize(raw)
assert summary==json.loads((root/'summary.json').read_text())
review_path=repo/'benchmarks/results/index/restart-floor-disk-2026-09-30/evidence_review.py'
review_loader=importlib.machinery.SourceFileLoader('restart_evidence_review',str(review_path))
review_spec=importlib.util.spec_from_loader(review_loader.name,review_loader)
review=importlib.util.module_from_spec(review_spec);review_loader.exec_module(review)
review.runner_binding(repo/'benchmarks/harnesses/restart-floor-2026-09-30/run.py.in',json.loads((root/'provenance.json').read_text()))
memory_binaries=json.loads((root/'binaries.json').read_text())
disk_binaries=json.loads((root.with_name('restart-floor-disk-2026-09-30')/'binaries.json').read_text())
assert len(memory_binaries)==3 and {b['arm'] for b in memory_binaries}=={'main','predecessor','candidate'}, 'HNSW duplicate/incomplete binary arms'
for b in memory_binaries:
    prior=next(p for p in disk_binaries if p['arm']==b['arm'])
    assert b['commit']==prior['commit'] and b['sha256']==prior['sha256'] and b['binary'].replace('${SCRATCH}','<SCRATCH>')==prior['binary'], 'HNSW arm-keyed executable binding'
timing_review=review.timing_review(summary,raw)
assert timing_review==json.loads((root/'timing-review.json').read_text()), 'HNSW timing screen differs'
labels={'First-search wall median (model load included)':'first_search_ms','Maintenance wall median':'maintenance_ms','Warm search p50: median across processes':'warm_p50_ms','Warm search p95: median across processes':'warm_p95_ms'}
reference='predecessor'
for line in (root/'README.md').read_text().splitlines():
    if line.startswith('## Combined'):reference='main'
    if not line.startswith('|'):continue
    cells=[c.strip() for c in line.split('|')];key=labels.get(cells[1])
    if key:
        metric=summary['comparisons'][reference]['metric_differences'][key]
        assert abs(float(cells[2].split()[0])-metric['before'])<.0006 and abs(float(cells[3].split()[0])-metric['after'])<.0006, 'HNSW displayed medians differ'
        withheld=timing_review[reference][key]['withhold_comparison']
        assert (cells[4].startswith('Withheld') and cells[5].startswith('Withheld'))==withheld, 'HNSW unstable timing comparison shown'
binding_source=repo/'benchmarks/results/index/restart-floor-disk-2026-09-30/rebuild/verify.py'
binding_loader=importlib.machinery.SourceFileLoader('retrospective_restart_binding',str(binding_source))
binding_spec=importlib.util.spec_from_loader(binding_loader.name,binding_loader)
binding=importlib.util.module_from_spec(binding_spec);binding_loader.exec_module(binding)
binding.verify(repo=repo)
print('verified: 9 rotated processes; 72/72 paired ordered lists per reference; actual providers, maintenance work, new-write visibility and recomputed metrics')
