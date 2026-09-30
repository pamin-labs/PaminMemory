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
# Bind every displayed row/cell, including stable difference and percentage
# cells; matching medians or a non-Withheld marker alone is insufficient.
labels=[
('Known-topic recall@10 (synthetic)','known_topic_recall_at_10',1,'',6),
('Known-topic MRR@10 (synthetic)','known_topic_mrr_at_10',1,'',6),
('Optimize jobs per first upkeep tick','optimize_jobs',1,'',3),
('Maintenance wall median','maintenance_ms',1,' ms',3),
('First-search wall median (model load included)','first_search_ms',1,' ms',3),
('Warm search p50: median across processes','warm_p50_ms',1,' ms',3),
('Warm search p95: median across processes','warm_p95_ms',1,' ms',3),
('Peak process RSS','peak_process_rss_bytes',2**20,' MiB',3),
('RSS after maintenance','maintenance_rss_bytes',2**20,' MiB',3),
('Closed index allocated bytes','closed_index_allocated_bytes',2**20,' MiB',3),
('Closed index apparent bytes','closed_index_apparent_bytes',2**20,' MiB',3),
('Vector graph completeness (hybrid visibility checked)','vector_completeness',1,'',6)]
expected=[]
for reference in ['predecessor','main']:
    expected+=['| Metric | Before | After | Absolute difference | Percentage change |','| --- | ---: | ---: | ---: | ---: |']
    for label,key,divisor,unit,precision in labels:
        metric=summary['comparisons'][reference]['metric_differences'][key]
        before,after,delta,percent=[metric[k] for k in ['before','after','absolute_delta','percent_delta']]
        if key in timing_review[reference] and timing_review[reference][key]['withhold_comparison']:
            difference=percentage='Withheld: unstable three-process sample'
        else:
            difference=f'{delta/divisor:+.{precision}f}{unit}'
            percentage=f'{percent:+.3f}%' if percent is not None else 'N/A'
        expected.append(f'| {label} | {before/divisor:.{precision}f}{unit} | {after/divisor:.{precision}f}{unit} | {difference} | {percentage} |')
displayed=[line for line in (root/'README.md').read_text().splitlines() if line.startswith('|')]
assert displayed==expected, 'HNSW complete displayed rows disagree with recomputed summary/timing policy'
binding_source=repo/'benchmarks/results/index/restart-floor-disk-2026-09-30/rebuild/verify.py'
binding_loader=importlib.machinery.SourceFileLoader('retrospective_restart_binding',str(binding_source))
binding_spec=importlib.util.spec_from_loader(binding_loader.name,binding_loader)
binding=importlib.util.module_from_spec(binding_spec);binding_loader.exec_module(binding)
binding.verify(repo=repo)
print('verified: 9 rotated processes; 72/72 paired ordered lists per reference; actual providers, maintenance work, new-write visibility and recomputed metrics')
