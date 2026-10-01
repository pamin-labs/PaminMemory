#!/usr/bin/env python3
"""Read-only verifier and independent invocation of the archived calculator."""
import hashlib,importlib.machinery,importlib.util,json,math,struct,sys
if sys.flags.optimize:
    raise SystemExit("Verification requires Python assertions; remove -O/-OO or PYTHONOPTIMIZE.")
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parent
repo=root.parents[3]
manifest=json.loads((root/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
    assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==digest, relative
runner_path=repo/'benchmarks/harnesses/restart-floor-2026-09-30/run.py.in'
runner_loader=importlib.machinery.SourceFileLoader('archived_hnsw_runner',str(runner_path))
runner_spec=importlib.util.spec_from_loader(runner_loader.name,runner_loader)
runner=importlib.util.module_from_spec(runner_spec);runner_loader.exec_module(runner)
review_path=repo/'benchmarks/results/index/restart-floor-disk-2026-09-30/evidence_review.py'
review_loader=importlib.machinery.SourceFileLoader('restart_evidence_review',str(review_path))
review_spec=importlib.util.spec_from_loader(review_loader.name,review_loader)
review=importlib.util.module_from_spec(review_spec);review_loader.exec_module(review)
raw=[json.loads(line) for line in (root/'raw.jsonl').read_text().splitlines()]
assert len([r for r in raw if r['phase']=='process_total'])==9
assert {(r['arm'],r['repetition']) for r in raw}=={(arm,rep) for arm in ['main','predecessor','candidate'] for rep in range(3)}, 'HNSW unexpected arm/repetition'
for arm in ['main','predecessor','candidate']:
    for rep in range(3):
        rows=[r for r in raw if r['arm']==arm and r['repetition']==rep]
        expected_order=['open','write_and_memory_drain','durability_flush','maintenance','search_cold','search_warmup','search_warmup']+['search_warm']*24+['new_write_search','closed_index','process_total']
        assert [r['phase'] for r in rows]==expected_order, 'HNSW product phase chronology differs'
        expected_phases=Counter({**{p:1 for p in ['open','write_and_memory_drain','durability_flush','maintenance','search_cold','new_write_search','closed_index','process_total']},'search_warmup':2,'search_warm':24})
        assert Counter(r['phase'] for r in rows)==expected_phases, 'HNSW unexpected phase multiset'
        log=(root/'logs'/f'{rep}-{arm}.log').read_text()
        review.successful_test_log(log)
        observed_providers=runner.cpu_provider_assignments(log)
        for row in rows:
            if row['phase']=='process_total':
                assert math.isfinite(row['wall_seconds']) and row['wall_seconds']>0, 'HNSW invalid process elapsed'
                continue
            assert row['wall_ms'] is None if row['phase']=='closed_index' else math.isfinite(row['wall_ms']) and row['wall_ms']>=0, 'HNSW invalid phase clock'
            for snapshot in [row['process_before'],row['process_after']]:
                if snapshot is not None:assert all(isinstance(snapshot[k],int) and snapshot[k]>=0 for k in ['user_ticks','system_ticks']), 'HNSW negative process counters'
            for k in ['cpu_user_seconds','cpu_system_seconds']:
                assert row[k] is None or math.isfinite(row[k]) and row[k]>=0, 'HNSW negative CPU delta'
            assert row['index']=='memory' and row['clock_ticks_per_second']==100, 'HNSW measurement configuration differs'
            assert row['actual_providers']==observed_providers, 'HNSW provider assignment differs from process log'
            for kind in ['user','system']:
                expected=None if row['process_before'] is None or row['process_after'] is None else (row['process_after'][kind+'_ticks']-row['process_before'][kind+'_ticks'])/row['clock_ticks_per_second']
                assert row['cpu_'+kind+'_seconds']==expected, 'HNSW derived CPU differs from retained ticks'
        logged=[json.loads(line.removeprefix('RESTART_JSON ')) for line in log.splitlines() if line.startswith('RESTART_JSON ')]
        unmatched=[r for r in rows if r['phase']!='process_total']
        assert len(logged)==len(unmatched), 'HNSW raw/log cardinality differs'
        assert all(all(key in r and r[key]==value for key,value in observation.items()) for observation,r in zip(logged,unmatched)), 'HNSW native/raw chronology differs'
        for observation in logged:
            matches=[i for i,r in enumerate(unmatched) if all(key in r and r[key]==value for key,value in observation.items())]
            assert len(matches)==1, 'HNSW native log observation has no unique raw row'
            unmatched.pop(matches[0])
        assert not unmatched, 'HNSW raw observations absent from native log'
        upkeep=[r for r in rows if r['phase']=='maintenance']
        assert len(upkeep)==1 and upkeep[0]['extra']['optimize_completed_delta']==(arm!='candidate')
        assert next(r for r in rows if r['phase']=='open')['extra']['documents']==18000, 'HNSW open document count differs'
        assert upkeep[0]['extra']['documents']==18001, 'HNSW post-write document count differs'
        complete=struct.unpack('f',struct.pack('f',18000/18001))[0] if arm=='candidate' else 1.0
        assert upkeep[0]['extra']['completeness']==complete, 'HNSW arm maintenance completeness differs'
        assert next(r for r in rows if r['phase']=='write_and_memory_drain')['extra']=={'applied':1,'completed':2,'pending':1 if arm=='candidate' else 2}, 'HNSW single-write drain premise differs'
        assert next(r for r in rows if r['phase']=='durability_flush')['extra']['flushed']==1, 'HNSW durability flush premise differs'
        closed=next(r for r in rows if r['phase']=='closed_index')
        assert closed['wall_ms'] is None and closed['measurement_kind']=='diagnostic', 'HNSW close diagnostic included in product clock'
        warm=[r for r in rows if r['phase']=='search_warm']
        # Bind the exact ordered synthetic query schedule, not merely uniqueness.
        # The archived harness uses the stopped seed's 18,000 documents.
        n=18000
        initial=[r['extra']['query_document'] for r in rows if r['phase'] in ['search_cold','search_warmup']]
        assert [r['phase'] for r in rows if r['phase'] in ['search_cold','search_warmup']]==['search_cold','search_warmup','search_warmup'] and initial==[0,n//2,n-1], 'HNSW exact ordered cold/warmup query schedule differs'
        assert [r['extra']['query_document'] for r in warm]==[j*(n-2)//25+1 for j in range(1,25)], 'HNSW exact ordered warm query schedule differs'
        assert len([r for r in rows if r['phase']=='search_cold'])==1
        assert len([r for r in rows if r['phase']=='search_warmup'])==2
        for r in rows:
            if 'query_document' in r.get('extra',{}):
                topics=r['extra']['topics'];rank=next((i+1 for i,t in enumerate(topics) if t==f"incident-{r['extra']['query_document']}"),None)
                assert len(topics)==len(set(topics))==10 and r['extra']['rank']==rank, 'HNSW query hit/rank premise differs'
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
review.runner_binding(repo/'benchmarks/harnesses/restart-floor-2026-09-30/run.py.in',json.loads((root/'provenance.json').read_text()))
memory_binaries=json.loads((root/'binaries.json').read_text())
disk_binaries=json.loads((root.with_name('restart-floor-disk-2026-09-30')/'binaries.json').read_text())
assert len(memory_binaries)==3 and {b['arm'] for b in memory_binaries}=={'main','predecessor','candidate'}, 'HNSW duplicate/incomplete binary arms'
for b in memory_binaries:
    prior=next(p for p in disk_binaries if p['arm']==b['arm'])
    assert b['commit']==prior['commit'] and b['sha256']==prior['sha256'] and b['binary'].replace('${SCRATCH}','<SCRATCH>')==prior['binary'], 'HNSW arm-keyed executable binding'
disk_root=root.with_name('restart-floor-disk-2026-09-30')
disk_provenance=json.loads((disk_root/'provenance.json').read_text())
provider_bindings=json.loads((disk_root/'provider-bindings.json').read_text())
memory_provenance=json.loads((root/'provenance.json').read_text())
assert memory_provenance['seed_documents']==18000, 'HNSW seed corpus premise differs'
seed_entries=memory_provenance['stopped_seed_index_files']
assert len(seed_entries)==len({e['path'] for e in seed_entries})==271, 'HNSW stopped seed inventory differs'
for suffix,content in [('/profile',b'gpahal/bge-m3-onnx-int8\ntopic\nmemory\nnamed\nreversed-keys'),('/.pamin-optimized-files',b'v1 271\n')]:
    selected=[e for e in seed_entries if e['path'].endswith(suffix)]
    assert len(selected)==1 and selected[0]['bytes']==len(content) and selected[0]['sha256']==hashlib.sha256(content).hexdigest(), 'HNSW seed profile/floor identity differs'
memory_entries=memory_provenance['model_assets']
assert len(memory_entries)==len({e['path'] for e in memory_entries})==7, 'HNSW duplicate/incomplete prehashed model inventory'
memory_assets={e['path'].replace('${MODEL_CACHE}','<MODEL_CACHE>'):e for e in memory_entries}
disk_assets={e['path']:e for e in disk_provenance['source_assets']+disk_provenance['prepared_graphs_and_external_weights']}
# Check this HNSW endpoint itself: the later Disk inventory is a separate
# observation, not a replacement for missing HNSW endpoint evidence.
post=json.loads((root/'post-trial-assets.json').read_text())
assert set(post)=={'prehashed_source_tokenizer_onnx_assets_unchanged','prepared_external_weights_post_trial'}, 'HNSW unexpected endpoint scope/release transition'
expected_paths=set()
for role,binding in provider_bindings['roles'].items():
    source=Path(binding['source_graph'])
    tokenizer=(source.parent if role=='embedding' else source.parent.parent)/'tokenizer.json'
    expected_paths.update([binding['source_graph'],str(tokenizer),binding['prepared_graph'],str(Path(binding['prepared_graph']).parent/'model.onnx')])
assert set(memory_assets)==expected_paths, 'HNSW prehashed model role/path inventory differs'
unchanged=[p.replace('${MODEL_CACHE}','<MODEL_CACHE>') for p in post['prehashed_source_tokenizer_onnx_assets_unchanged']]
assert len(unchanged)==len(set(unchanged))==7 and set(unchanged)==set(memory_assets), 'HNSW complete unchanged endpoint pathset differs'
# Both original CPU exports are explicitly included in the unchanged receipt.
# No released-source transition was recorded; do not invent one retrospectively.
external=post['prepared_external_weights_post_trial']
assert len(external)==len({e['path'] for e in external})==2, 'HNSW incomplete/duplicate external-weight endpoint'
expected_external={b['external_data'] for b in provider_bindings['roles'].values()}
assert {e['path'].replace('${MODEL_CACHE}','<MODEL_CACHE>') for e in external}==expected_external, 'HNSW external-weight endpoint pathset differs'
receipt_mtimes={'eecdcf109c0c08402aa8f893fc25d2d4':1790793669634082575,'ac146c082d1526dd1cd10597ae4a0ffc':1790793000506673216}
for entry in external:
    normalized=entry['path'].replace('${MODEL_CACHE}','<MODEL_CACHE>')
    assert set(entry)=={'path','bytes','sha256','mtime_ns','hash_scope'}, 'HNSW external-weight endpoint scope differs'
    assert normalized in disk_assets and all(entry[k]==disk_assets[normalized][k] for k in ['bytes','sha256']), 'HNSW external-weight endpoint identity differs'
    assert entry['mtime_ns']==receipt_mtimes[Path(normalized).parent.name] and entry['hash_scope']=='post-trial identity; not independently hashed before timing', 'HNSW external-weight endpoint receipt differs'
for path,entry in memory_assets.items():
    assert path in disk_assets and all(entry[key]==disk_assets[path][key] for key in ['bytes','sha256']), 'HNSW recorded model inventory differs from retained role binding'
for row in raw:
    if row['phase']=='process_total':continue
    for role,provider in row['actual_providers'].items():
        prefix=f"${{EVAL_HOMES}}/{row['repetition']}-{row['arm']}/models/"
        assert provider['model_graph'].startswith(prefix), 'HNSW provider process/model path differs'
        normalized='<MODEL_CACHE>/'+provider['model_graph'].removeprefix(prefix)
        binding=provider_bindings['roles'][role]
        assert normalized==binding['prepared_graph'] and normalized in memory_assets and binding['source_graph'] in memory_assets, 'HNSW provider graph absent from retained model inventory'
        mapped=dict(provider,model_graph=provider['model_graph'].replace('${EVAL_HOMES}/','<SCRATCH>/results-disk/',1))
        review.provider_binding(mapped,role,disk_provenance,provider_bindings)
expected_commits={b['arm']:b['commit'] for b in memory_binaries}
assert all(r['commit']==expected_commits[r['arm']] for r in raw), 'HNSW raw commit differs from retained arm executable'
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
