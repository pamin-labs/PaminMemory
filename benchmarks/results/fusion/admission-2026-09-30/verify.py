#!/usr/bin/env python3
"""Read-only archive/premise verifier; recomputes metrics from full raw traces."""
import gzip,hashlib,json,sys
if sys.flags.optimize:
 raise SystemExit("Verification requires Python assertions; remove -O/PYTHONOPTIMIZE")
from pathlib import Path
def provider_records(log, arm, expected, assets):
 """Tie original assignment records to their resolved graph identity."""
 decoder=json.JSONDecoder();records=[]
 prefix='${EXPERIMENT}/paired/'+arm+'-home/models/'
 graph_hashes={a['path']:a['sha256'] for a in assets}
 for line in log.splitlines():
  if 'ONNX graph execution-provider assignment' not in line:continue
  graph,_=decoder.raw_decode(line.split('model_graph=',1)[1])
  nodes,end=decoder.raw_decode(line.split('assigned_nodes=',1)[1])
  assert not line.split('assigned_nodes=',1)[1][end:].strip(), 'trailing provider record data'
  assert isinstance(graph,str) and graph.startswith(prefix), 'unexpected graph path'
  graph='${MODEL_CACHE}/'+graph[len(prefix):]
  assert graph in graph_hashes, 'graph absent from pretrial asset identities'
  assert isinstance(nodes,dict) and set(nodes)=={'CPUExecutionProvider'}, 'unexpected provider'
  assert type(nodes['CPUExecutionProvider']) is int and nodes['CPUExecutionProvider']>0, 'invalid node count'
  records.append({'model_graph':graph,'assigned_nodes':nodes,'graph_sha256':graph_hashes[graph]})
 assert len(records)==2 and records==expected, 'raw provider records differ from summary/assets'
 return records

root=Path(__file__).resolve().parent
repo=root.parents[3]
manifest=json.loads((root/'manifest.json').read_text())
for relative,digest in manifest['files'].items():
 assert hashlib.sha256((repo/relative).read_bytes()).hexdigest()==digest,relative
summary=json.loads((root/'paired-summary.json').read_text())
assets=json.loads((root/'pretrial-assets.json').read_text())
audit=json.loads((root/'build-audit.json').read_text())
assert audit['binding_status']=='retrospective_inputs_only_not_build_attestation'
assert audit['historical_tuning_status']=='unrecorded_inherited_overrides'
for arm in ['baseline','pooled']:
 retained=audit['arms'][arm]
 build=gzip.decompress((root/retained['build_log']).read_bytes())
 lock=gzip.decompress((root/retained['helper_lockfile']).read_bytes())
 assert hashlib.sha256(build).hexdigest()==retained['sanitized_build_log_sha256']
 assert hashlib.sha256(lock).hexdigest()==retained['helper_lockfile_sha256']
 assert f'variants/{arm}/crates/pamin-engine'.encode() in build
 assert f'helpers/{arm}'.encode() in build and b'Finished `release` profile' in build
 assert retained['helper_source_sha256']==hashlib.sha256((repo/'benchmarks/harnesses/fusion-admission-2026-09-30'/f'{arm}.rs.in').read_bytes()).hexdigest()
 assert retained['helper_manifest_sha256']==hashlib.sha256((repo/'benchmarks/harnesses/fusion-admission-2026-09-30'/f'{arm}-Cargo.toml.in').read_bytes()).hexdigest()
sources=json.loads((root/'sources.json').read_text())
for arm in ['baseline','pooled']:
 for relative,digest in sources['variants'][arm]['overlaid_sha256'].items():
  assert audit['arms'][arm]['variant_files'][relative]==digest
before_files=audit['arms']['baseline']['variant_files'];after_files=audit['arms']['pooled']['variant_files']
assert before_files.keys()==after_files.keys()
assert [p for p in before_files if before_files[p]!=after_files[p]]==['crates/pamin-engine/src/engine.rs']
arms={}
for arm in ['baseline','pooled']:
 text=gzip.decompress((root/f'{arm}.jsonl.gz').read_bytes()).decode()
 rows=[json.loads(line) for line in text.splitlines()]
 assert len(rows)==24 and len({r['query_document'] for r in rows})==24
 log=gzip.decompress((root/f'{arm}.log.gz').read_bytes()).decode()
 log_rows=[json.loads(line[len('IDENTIFIER_JSON '):]) for line in log.splitlines() if line.startswith('IDENTIFIER_JSON ')]
 assert log_rows==rows
 provider_records(log,arm,summary['actual_provider_graphs'][arm],assets['prepared_assets'])
 for r in rows:
  assert r['channel_pool']==(arm=='pooled')
  assert r['expected_documents']==18001 and r['channel_depth']==50 and r['reranker_depth']==30
  inputs=r['reranker_inputs'];assert len(inputs)==30
  assert len({v['topic'] for v in inputs})==30
  assert {v['topic'] for v in inputs}=={v['topic'] for v in r['final_all'] if any(w['kind']=='reranked' for w in v['why'])}
  assert r['product_top10']==r['final_all'][:10]
  assert {v['topic'] for v in r['fused']}=={v['topic'] for v in r['final_all']}
  assert r['target_fused_rank']==next(v['rank'] for v in r['fused'] if v['topic']==r['target'])
  assert r['target_final_rank']==next(v['rank'] for v in r['final_all'] if v['topic']==r['target'])
  assert r['target_entered_reranker']==any(v['topic']==r['target'] for v in inputs)
  for entries in r['channels_from_complete_trace'].values():
   assert [e['rank'] for e in entries]==list(range(1,len(entries)+1))
 arms[arm]=rows
providers=summary['actual_provider_graphs']
assert providers['baseline']==providers['pooled'] and len(providers['baseline'])==2
for p in providers['baseline']:
 assert set(p['assigned_nodes'])=={'CPUExecutionProvider'} and p['assigned_nodes']['CPUExecutionProvider']>0
assert summary['queries']==24 and summary['pairs_per_query_each_arm']==30
changed=0;recovered=[];lost=[]
for before,after,retained in zip(arms['baseline'],arms['pooled'],summary['rows']):
 assert before['query']==after['query']==retained['query'] and before['target']==after['target']==retained['target']
 assert before['fused']==after['fused']
 b={v['topic'] for v in before['reranker_inputs']};a={v['topic'] for v in after['reranker_inputs']}
 assert sorted(a-b)==retained['added_candidates'] and sorted(b-a)==retained['removed_candidates']
 changed+=bool(a-b)
 for arm,row in [('baseline',before),('pooled',after)]:
  assert row['target_final_rank']==retained[f'{arm}_final_rank']
  assert row['target_entered_reranker']==retained[f'{arm}_entered']
  assert [v['topic'] for v in row['product_top10']]==retained[f'{arm}_top10']
  assert next(v['why'] for v in row['final_all'] if v['topic']==row['target'])==retained[f'{arm}_target_why']
 if before['target_final_rank']>10>=after['target_final_rank']:recovered.append(before['target'])
 if before['target_final_rank']<=10<after['target_final_rank']:lost.append(before['target'])
assert changed==summary['changed_candidate_queries']==10
assert recovered==summary['recovered_targets'] and lost==summary['lost_targets']==[]
metrics=json.loads((root/'metrics.json').read_text())
for name,reciprocal in [('Known-target recall@10',False),('Known-target MRR@10',True)]:
 values=[]
 for arm in ['baseline','pooled']:
  ranks=[r['target_final_rank'] for r in arms[arm]]
  n=sum(k<=10 for k in ranks)
  assert n==summary[f'{arm}_targets_top10']
  values.append(sum((1/k if reciprocal else 1) if k<=10 else 0 for k in ranks)/24)
 a,b=values
 assert metrics[name]=={'before':a,'after':b,'absolute_difference':b-a,'percentage_change':100*(b-a)/a}
assert sum(r['target_entered_reranker'] for r in arms['baseline'])==21
assert all(r['target_entered_reranker'] for r in arms['pooled'])
print('Verified 48 actual Engine traces, same fused candidates, 30 offered candidates/search, raw providers and recomputed diagnostic metrics; historical build binding/tuning remain unverified.')
