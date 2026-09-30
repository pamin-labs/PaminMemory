#!/usr/bin/env python3
"""Read-only checks of retained synthetic component evidence; no runtime loading."""
import hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-6)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(hit):return next(w['score'] for w in hit['why'] if w.get('kind')=='channel' and w['channel']=='graph')
def path(hit):return next(w for w in hit['why'] if w.get('kind')=='path')
def load(name):return json.loads((ROOT/name).read_text())

provenance=load('provenance.json');comparison=load('comparison.json')
assert provenance['source_base']=='13ee710c9df865f1dac98dc77a8108e438ddc539'
assert provenance['scope'].startswith('native search_fused component')
for name,expected in provenance['source_files'].items():assert sha(ROOT/'source'/name)==expected
for name,record in provenance['redactions'].items():
 assert sha(ROOT/name)==record['published_sha256'] and re.fullmatch('[0-9a-f]{64}',record['original_sha256'])
arm_records={r['arm']:r for r in provenance['arms']}
for arm,weak_rank in [('baseline',23),('scored',22)]:
 row=load(f'{arm}.jsonl');record=arm_records[arm]
 assert row['record']=='fixture' and row['arm']==arm and row['documents']==241
 assert row['weak_rank']==weak_rank and close(row['weak_relevance'],11/(10+weak_rank))
 weak=next(r for r in row['non_graph'] if r['topic']==row['weak'])
 assert min(r['rank'] for r in weak['ranks'])==weak_rank
 assert all(1<=r['rank']<=50 for h in row['non_graph'] for r in h['ranks'])
 visible={r['topic'] for r in row['non_graph']}
 assert not visible.intersection(row['target_labels']) and row['early_stop']['target'] not in visible
 assert len(row['known_edges'])==9 and all(close(a,b) for a,b in zip(sorted(e['confidence'] for e in row['known_edges']),sorted([1,.8,.1,.01,1,.8,.7,.1,1])))
 assert 'test scratch_scored_graph_finite_fixture ... ok' in (ROOT/f'{arm}.log').read_text()
 hits={h['topic']:h for h in row['targets']};assert len(hits)==3
 values=[score(hits[name]) for name in row['target_labels']]
 expected=[row['weak_relevance'],.1,.05] if arm=='baseline' else [.8,.5,.5]
 assert all(close(a,b) for a,b in zip(values,expected))
 first=path(hits[row['target_labels'][0]])
 assert first['hops']==1 and first['from']==(row['weak'] if arm=='baseline' else row['strong'])
 for i in [1,2]:
  trace=path(hits[row['target_labels'][i]]);assert trace['from']==row['strong']
  assert trace['hops']==(1 if arm=='baseline' and i==1 else 2)
 early=row['early_stop'];assert early['decoys']==60 and close(early['expected_score'],.5)
 assert early['reached']==(arm=='scored')
 if arm=='scored':
  assert close(score({'why':early['why']}),.5)
  trace=path({'why':early['why']});assert trace['from']==row['strong'] and trace['via']==early['via'] and trace['hops']==2
 fresh=record['fresh_compiler_artifacts'];assert all(r['fresh'] is False for r in fresh)
 assert {r['target'] for r in fresh}=={'pamin_core','pamin_store','pamin_index','pamin_engine','scratch_scored_fixture','scratch_scored_multihop'}
 assert record['asset_hashes_before']==record['asset_hashes_after']
 assert record['resources']['oom_before']==record['resources']['oom_after']==0
 assert record['resources']['oom_kill_before']==record['resources']['oom_kill_after']==0
 assert record['process_usage']['exit_status']==0
 assert record['inference']['runtime_info'][0].startswith('ORT Build Info:')
 graphs=record['inference']['selected_graphs'];assert len(graphs)==1
 graph=next(iter(graphs.values()));assert graph['assigned_nodes']=={'CPUExecutionProvider':1023}
 assert graph['sha256']=='51040ce485c0c3a9f9e46fdf1847ee125aa8e2dadbcf97a49a636bb47cd42ea4'
 assert graph['companion_assets_sha256']['${MODEL_CACHE}/prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx.data']=='2600d5896ddedb06f0d1627179ea1c39da1ef075e170598cd467016dc06e8c0b'
 assert record['inference']['mapped_native_libraries_sha256']['${ORT_LIB}/libonnxruntime.so.1.28.0']=='1461ef7cc3d9e49982591721683cc3e3a55580aeca9a5254e7aac47b75ee4bab'
assert arm_records['baseline']['inference']==arm_records['scored']['inference']
assert comparison['cross_arm_non_graph_identical'] is False and comparison['early_stop_recovered'] is True
assert [r['expected'] for r in comparison['invariants']]==[.8,.5,.5]
for p in ROOT.rglob('*'):
 if not p.is_file() or p.name=='verify.py':continue
 data=p.read_text()
 assert not re.search(r'(?:/workspace/(?:scratch|\.pamin|\.cargo|\.onnxruntime|PaminMemory)|/home/|postgres(?:ql)?://|Bearer\s+[A-Za-z0-9]|claude\.ai/|app://)',data),f'private path/credential/session marker: {p.name}'
 assert p.suffix not in {'.onnx','.bin','.data','.so'},'binary/model material must not be published'
print('PASS: four native component cases, exact weak-rank arithmetic, fresh builds, pinned inference/assets, disclosed limits, sanitized text-only evidence')
