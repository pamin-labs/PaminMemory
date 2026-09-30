#!/usr/bin/env python3
"""Read-only checks of retained synthetic component evidence; no runtime loading."""
import hashlib,json,math,re,sys,struct
if sys.flags.optimize:
 raise SystemExit("FAIL: Python optimization disables assertions; run without -O/-OO")
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
EXPECTED_FILES = {'provenance.json', 'source/migrations/V10__settled_jobs_leave.sql', 'source/migrations/V12__job_subject_once.sql', 'source/migrations/V1__initial.sql', 'source/migrations/V8__topics_by_recency.sql', 'retrospective-sql-audit.json', 'source/migrations/V5__cascade_outbox.sql', 'source/graph_trace.rs.in', 'source/migrations/V9__state_content_from_span.sql', 'comparison.json', 'source/migrations/V4__current_state_pointer.sql', 'scored.trace.jsonl', 'scored.jsonl', 'README.md', 'scored.usage.json', 'scored.log', 'source/migrations/V11__retrieval_signals_leave.sql', 'test_verify.py', 'source/migrations/V3__shard_key_and_indexes.sql', 'source/prepare.py', 'source/migrations/V6__topic_name_index.sql', 'baseline.log', 'baseline.trace.jsonl', 'baseline.usage.json', 'source/migrations/V14__source_versions_index_once.sql', 'platform-observation.json', 'source/migrations/V13__edge_endpoints_on_versions.sql', 'source/migrations/V7__one_document_per_topic.sql', 'source/fixture.rs.in', 'baseline.jsonl', 'source/experimental-traversal.patch', 'verify.py', 'source/migrations/V2__relationships.sql'}
assert {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()} == EXPECTED_FILES, 'required file inventory differs'
assert set(provenance['archive_files']) == EXPECTED_FILES - {'provenance.json'}, 'hash inventory differs'
for name, digest in provenance['archive_files'].items():
 assert sha(ROOT/name) == digest, f'archive hash differs: {name}'
assert set(provenance['source_files']) == {'fixture.rs.in','graph_trace.rs.in','experimental-traversal.patch','prepare.py'}
assert set(provenance['redactions']) == {f'{a}.{e}' for a in ['baseline','scored'] for e in ['jsonl','log','trace.jsonl','usage.json']}
ASSET_PINS = {'${MODEL_CACHE}/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/config.json': '68bf436cb7a210655f599b3944065f12a81317291b32e78bd4338b5745cda63a',
 '${MODEL_CACHE}/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/model_quantized.onnx': {'expected_transition': 'present '
                                                                                                                                                    'to '
                                                                                                                                                    'absent '
                                                                                                                                                    'after '
                                                                                                                                                    'successful '
                                                                                                                                                    'native '
                                                                                                                                                    'CPU '
                                                                                                                                                    'preparation; '
                                                                                                                                                    'prepared '
                                                                                                                                                    'assets '
                                                                                                                                                    'pinned',
                                                                                                                             'present': True,
                                                                                                                             'sha256': '16de7ea1146ca427e14938ec3e9abfdcaff0e6ac76434cd693ac35d761250bcb',
                                                                                                                             'source_identity_sha256': '16de7ea1146ca427e14938ec3e9abfdcaff0e6ac76434cd693ac35d761250bcb'},
 '${MODEL_CACHE}/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/special_tokens_map.json': '8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835',
 '${MODEL_CACHE}/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/tokenizer.json': '249df0778f236f6ece390de0de746838ef25b9d6954b68c2ee71249e0a9d8fd4',
 '${MODEL_CACHE}/models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/tokenizer_config.json': 'b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80',
 '${MODEL_CACHE}/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/config.json': '122e922dcfed6503c8721e6fe1daf090340c3d95ca7f3aa3a72730b321a51cfd',
 '${MODEL_CACHE}/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/onnx/model_int8.onnx': {'expected_transition': 'present '
                                                                                                                                                                   'to '
                                                                                                                                                                   'absent '
                                                                                                                                                                   'after '
                                                                                                                                                                   'successful '
                                                                                                                                                                   'native '
                                                                                                                                                                   'CPU '
                                                                                                                                                                   'preparation; '
                                                                                                                                                                   'prepared '
                                                                                                                                                                   'assets '
                                                                                                                                                                   'pinned',
                                                                                                                                            'present': False,
                                                                                                                                            'sha256': None,
                                                                                                                                            'source_identity_sha256': '912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d'},
 '${MODEL_CACHE}/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/special_tokens_map.json': '8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835',
 '${MODEL_CACHE}/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/tokenizer.json': '8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3',
 '${MODEL_CACHE}/models--onnx-community--bge-reranker-v2-m3-ONNX/snapshots/6f5ff65298512715a1e669753bc754d2bc8f367b/tokenizer_config.json': 'b87c8703482b0300d3da30e201519aa641f6a450f5eb5bf1e624afbf70c74d80',
 '${MODEL_CACHE}/prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx': '51040ce485c0c3a9f9e46fdf1847ee125aa8e2dadbcf97a49a636bb47cd42ea4',
 '${MODEL_CACHE}/prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx.data': '2600d5896ddedb06f0d1627179ea1c39da1ef075e170598cd467016dc06e8c0b',
 '${MODEL_CACHE}/prepared/eecdcf109c0c08402aa8f893fc25d2d4/attention.onnx': '24cc5ad23811a65c6a4648d8be0b79a7ad67a180e2db234b08c899c882ac5164',
 '${MODEL_CACHE}/prepared/eecdcf109c0c08402aa8f893fc25d2d4/model.onnx': '7231486a34a1d71b151f9f0f8a224f1d7d454dfaad450117b5cc179a2d350bb8',
 '${MODEL_CACHE}/prepared/eecdcf109c0c08402aa8f893fc25d2d4/model.onnx.data': '06b529ab95974bcf4b21c0ac9e649816534c2e2ebb44990cb15462872754199c',
 '${ORT_LIB}/libonnxruntime.so.1.28.0': '1461ef7cc3d9e49982591721683cc3e3a55580aeca9a5254e7aac47b75ee4bab',
 '${ZVEC_LIB}/libzvec_c_api.so': '58381ac7b12afd5eeae3dc10325914a28fc3157061291bb693a9ed757d815b8a'}
BINARY_PINS = {'baseline': '485cbd023d9e5c8b1b4cdd3f94419199fe634a1a66a9d67d44dd90b7a1facdd1', 'scored': 'ac308a0b9438eb897d1601d185583026d2f07b1a268b47b7b1082e56daf72bd0'}
PAYLOAD_MAP_DIGESTS = {'baseline': 'b2cba4a77820f63e0ea869b5099f631705aae850dee17e6a2a4d287e29c4397b', 'scored': 'b2cba4a77820f63e0ea869b5099f631705aae850dee17e6a2a4d287e29c4397b'}
CURRENT_PLATFORM_PINS = {'cpu_model': 'AMD EPYC 9V74 80-Core Processor', 'kernel': '6.18.44', 'architecture': 'x86_64', 'cpu_quota': '400000 100000', 'cpu_affinity': [0, 1, 2, 3, 4], 'memory_max_bytes': 17179869184}
SQL_MAP_DIGEST = '6264b33b89775579e92b4d903a16b1fed165942498d5f2b3acc024b38e834513'
inventory_scope = provenance['source_inventory_scope']
assert inventory_scope['original_captured_entries_per_arm'] == 52
assert inventory_scope['original_status'] == 'partial: migration SQL omitted'
assert inventory_scope['original_sql_build_attestation'] == 'N/A: not captured'
assert inventory_scope['retrospective_sql_audit'] == 'retrospective-sql-audit.json'
audit = load('retrospective-sql-audit.json')
assert audit['source_base'] == provenance['source_base']
assert audit['original_build_sql_attestation'] == 'N/A: SQL omitted from the original 52-entry pre/post-build inventory'
assert re.fullmatch(r'2026-09-30T[0-9:.]+\+00:00', audit['captured_at_utc'])
sql_sources = audit['migration_sources']
assert len(sql_sources) == 14
assert hashlib.sha256(json.dumps({k:v['sha256'] for k,v in sql_sources.items()},sort_keys=True,separators=(',',':')).encode()).hexdigest() == SQL_MAP_DIGEST
assert {v['published_file'] for v in sql_sources.values()} == {name for name in EXPECTED_FILES if name.startswith('source/migrations/')}
for name, source in sql_sources.items():
 assert name == 'crates/pamin-store/migrations/' + Path(source['published_file']).name
 assert source['matches_recorded_git_base'] is True
 assert sha(ROOT/source['published_file']) == source['sha256']
 assert (ROOT/source['published_file']).stat().st_size == source['bytes']
assert set(audit['arms']) == {'baseline','scored'}
for arm, binary in audit['arms'].items():
 assert binary['binary_sha256'] == BINARY_PINS[arm]
 assert set(binary['migration_payloads']) == set(sql_sources)
 assert hashlib.sha256(json.dumps(binary['migration_payloads'],sort_keys=True,separators=(',',':')).encode()).hexdigest() == PAYLOAD_MAP_DIGESTS[arm], 'retrospective payload capture differs'
 for name, payload in binary['migration_payloads'].items():
  assert payload['sha256'] == sql_sources[name]['sha256'] and payload['bytes'] == sql_sources[name]['bytes']
  assert type(payload['first_binary_offset']) is int and 0 <= payload['first_binary_offset'] <= binary['binary_bytes'] - payload['bytes']
platform_observation = load(provenance['hardware_observation'])
assert re.fullmatch(r'2026-09-30T[0-9:.]+\+00:00', platform_observation['captured_at_utc'])
assert platform_observation['historical'] == {'cpu_model':'N/A: not captured','kernel':'N/A: not captured','cpu_quota':'N/A: not captured','cpu_affinity':'N/A: not captured','memory_max_bytes':{'baseline':17179869184,'scored':17179869184}}
assert platform_observation['current'] == CURRENT_PLATFORM_PINS, 'dated current platform capture differs'
assert platform_observation['current']['cpu_model'] and platform_observation['current']['kernel'] and platform_observation['current']['cpu_affinity']
assert platform_observation['current']['memory_max_bytes'] == 17179869184
assert platform_observation['current']['cpu_quota'] == '400000 100000'

arm_records={r['arm']:r for r in provenance['arms']}
assert len(provenance['arms']) == 2 and set(arm_records) == {'baseline','scored'}
SOURCE_PATHS = {'crates/pamin-engine/tests/graph_trace/mod.rs', 'crates/pamin-cli/Cargo.toml', 'crates/pamin-store/src/error.rs', 'crates/pamin-index/src/half.rs', 'crates/pamin-index/src/tokenizer.rs', 'crates/pamin-index/src/encoder.rs', 'crates/pamin-index/src/hub.rs', 'crates/pamin-index/src/inference.rs', 'crates/pamin-index/src/reshape.rs', 'crates/pamin-index/src/segmentation.rs', 'crates/pamin-core/src/version.rs', 'crates/pamin-core/src/graph.rs', 'benchmarks/results/inference/vector-rescore-device-2026-09-30/Cargo.toml', 'Cargo.lock', 'crates/pamin-store/src/lib.rs', 'crates/pamin-core/src/lib.rs', 'crates/pamin-index/src/onnx.rs', 'crates/pamin-index/src/reranking.rs', 'crates/pamin-core/Cargo.toml', 'crates/pamin-store/src/workspace.rs', 'crates/pamin-engine/Cargo.toml', 'crates/pamin-index/src/error.rs', 'crates/pamin-core/src/env.rs', 'crates/pamin-index/src/descriptors.rs', 'Cargo.toml', 'crates/pamin-core/src/id.rs', 'crates/pamin-store/src/sql.rs', 'crates/pamin-index/src/lib.rs', 'crates/pamin-core/src/channel.rs', 'crates/pamin-index/Cargo.toml', 'crates/pamin-index/src/prepared.rs', 'crates/pamin-store/src/repository.rs', 'crates/pamin-core/src/fusion.rs', 'crates/pamin-index/src/projection.rs', 'crates/pamin-engine/src/reshape.rs', 'crates/pamin-index/src/attention.rs', 'crates/pamin-store/src/database.rs', 'crates/pamin-engine/src/cascade.rs', 'benchmarks/results/inference/vector-rescore-accelerate-2026-09-30-Cargo.toml', 'crates/pamin-core/src/cascade.rs', 'crates/pamin-core/src/ledger.rs', 'crates/pamin-engine/tests/scratch_scored_fixture.rs', 'crates/pamin-store/src/migrate.rs', 'crates/pamin-index/src/native.rs', 'crates/pamin-store/Cargo.toml', 'crates/pamin-store/src/jobs.rs', 'crates/pamin-engine/src/engine.rs', 'crates/pamin-core/src/filter.rs', 'crates/pamin-engine/src/lib.rs', 'crates/pamin-store/src/graph.rs', 'crates/pamin-engine/tests/harness/mod.rs', 'crates/pamin-index/src/embedding.rs'}
SOURCE_MAP_DIGESTS = {'baseline': 'a1559290a3d29640e927e2570191c90e3958caf75c224f7d70e61ac5f1107171', 'scored': '57410269c8ae2da52143afbb1c0b8e7a8644582785f2dc44a0e331bbce3eb0fa'}
for arm, record in arm_records.items():
 assert record['binary_sha256'] == BINARY_PINS[arm], 'recorded binary digest differs'
 sources = record['source_hashes']
 assert set(sources) == SOURCE_PATHS, 'compiled source inventory differs'
 assert all(re.fullmatch('[0-9a-f]{64}',digest) for digest in sources.values()), 'malformed source digest'
 assert hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':')).encode()).hexdigest() == SOURCE_MAP_DIGESTS[arm], 'recorded compiled source map differs'
 assert sources['crates/pamin-engine/tests/scratch_scored_fixture.rs'] == sha(ROOT/'source/fixture.rs.in')
 assert sources['crates/pamin-engine/tests/graph_trace/mod.rs'] == sha(ROOT/'source/graph_trace.rs.in')
assert {name for name in SOURCE_PATHS if arm_records['baseline']['source_hashes'][name] != arm_records['scored']['source_hashes'][name]} == {'crates/pamin-engine/src/engine.rs'}, 'cross-arm source relationship differs'
assert sha(ROOT/'source/fixture.rs.in') == '0daa185fcf9eff2d174ce13569407ff972827cacecffce15996740352cdbaab1', 'exact measured fixture differs'
assert 'let query = \"quartzanchor orbital navigation calibration beacon\";' in (ROOT/'source/fixture.rs.in').read_text()
rows = {}

for arm,weak_rank in [('baseline',23),('scored',22)]:
 row=load(f'{arm}.jsonl');record=arm_records[arm];rows[arm]=row
 launch = record['launch_binding']
 assert re.fullmatch('[0-9a-f]{64}', launch['original_launch_sha256'])
 assert launch['command'] == ['${FROZEN_BINARY}','scratch_scored_graph_finite_fixture','--exact','--ignored','--nocapture','--test-threads=1']
 assert launch['GRAPH_OUT'] == '${RESULTS}/'+arm+'.jsonl' and launch['GRAPH_TRACE'] == '${RESULTS}/'+arm+'.trace.jsonl'
 assert launch['effective_product_settings'] == {'PAMIN_PROFILE':'accuracy','PAMIN_DEVICE':'cpu','GRAPH_ARM':arm,'GRAPH_CLK_TCK':'100'}

 assert row['query'] == 'quartzanchor orbital navigation calibration beacon', 'native fixture query differs'
 assert row['record']=='fixture' and row['arm']==arm and row['documents']==241
 assert row['expected_scores'] == [.8,.5,.5]
 assert row['weak_rank']==weak_rank and close(row['weak_relevance'],11/(10+weak_rank))
 weak=next(r for r in row['non_graph'] if r['topic']==row['weak'])
 assert min(r['rank'] for r in weak['ranks'])==weak_rank
 channels = {'lexical_segmented','lexical_ngram','vector'}
 assert {r['channel'] for h in row['non_graph'] for r in h['ranks']} == channels
 assert len(row['non_graph']) == len({h['topic'] for h in row['non_graph']}) == len({h['id'] for h in row['non_graph']})
 for channel in channels:
  assert sorted(r['rank'] for h in row['non_graph'] for r in h['ranks'] if r['channel']==channel) == list(range(1,51)), 'incomplete/duplicate top50 rank inventory'
 assert all(r['score'] is None or math.isfinite(r['score']) for h in row['non_graph'] for r in h['ranks'])
 visible={r['topic'] for r in row['non_graph']}
 assert not visible.intersection(row['target_labels']) and row['early_stop']['target'] not in visible
 # Fixture chooses the first 67 hidden topics in repository name order.
 hidden = sorted(set(f'islandnode{n:04}' for n in range(240)) - visible)[:67]
 assert len(hidden)==67 and row['strong']=='quartzanchor'
 shared,longer,bridge,samehop,low,high = hidden[:6]
 assert row['target_labels'] == [shared,longer,samehop]
 assert row['early_stop']['via']==bridge and row['early_stop']['target']==hidden[6]
 f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
 topology = [(row['weak'],shared,1),(row['strong'],shared,.8),(row['strong'],longer,.1),(row['strong'],bridge,.01),(bridge,longer,1),(row['strong'],low,.8),(row['strong'],high,.7),(low,samehop,.1),(high,samehop,1)]
 assert sorted((e['from'],e['to'],e['confidence']) for e in row['known_edges']) == sorted((a,b,f32(c)) for a,b,c in topology), 'controlled endpoint/confidence topology differs'

 log = (ROOT/f'{arm}.log').read_text()
 assert re.findall(r'^test scratch_scored_graph_finite_fixture \.\.\. ok$', log, re.M) == ['test scratch_scored_graph_finite_fixture ... ok']
 assert len(re.findall(r'^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in [0-9.]+s$', log, re.M)) == 1
 # Historical fixture writes GRAPH_OUT directly, so no stdout JSON linkage exists.
 assert provenance['fixture_log_linkage'] == 'not captured: native fixture persisted GRAPH_OUT directly; stdout has test success only'
 assert 'std::env::var("GRAPH_OUT").expect("output path")' in (ROOT/'source/fixture.rs.in').read_text()
 events = [json.loads(line) for line in (ROOT/f'{arm}.trace.jsonl').read_text().splitlines()]
 assert len(events) == 3
 runtime, loaded, assigned = events
 assert [e['fields']['message'] for e in events] == ['graph fixture runtime','loaded ONNX graph','ONNX graph execution-provider assignment']
 assert [e['target'] for e in events] == ['scratch_scored_fixture::graph_trace','pamin_index::inference','pamin_index::inference']
 info = record['inference']
 assert info['runtime_info'] == [runtime['fields']['runtime_info']] == ['ORT Build Info: git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, build type=Release']
 assert set(info['selected_graphs']) == {loaded['fields']['model_graph']} == {assigned['fields']['model_graph']}
 model = loaded['fields']['model_graph']
 assert info['selected_graphs'][model]['assigned_nodes'] == assigned['fields']['assigned_nodes'] == {'CPUExecutionProvider':1023}
 assert info['selected_graphs'][model]['sha256'] == ASSET_PINS[model]
 assert info['selected_graphs'][model]['companion_assets_sha256'] == {k:ASSET_PINS[k] for k in [model,model+'.data']}
 mapped = info['mapped_native_libraries_sha256']
 assert all(e['runtime_maps'] == list(mapped) for e in events)
 assert mapped == {k:ASSET_PINS[k] for k in mapped}
 assert set(mapped) == {'${ORT_LIB}/libonnxruntime.so.1.28.0'}
 assert record['asset_hashes_before'] == record['asset_hashes_after'] == ASSET_PINS
 assert record['process_usage'] == load(f'{arm}.usage.json')

 hits={h['topic']:h for h in row['targets']};assert len(hits)==3
 values=[score(hits[name]) for name in row['target_labels']]
 expected=[row['weak_relevance'],.1,.05] if arm=='baseline' else [.8,.5,.5]
 assert all(close(a,b) for a,b in zip(values,expected))
 first=path(hits[row['target_labels'][0]])
 assert first['hops']==1 and first['from']==first['via']==(row['weak'] if arm=='baseline' else row['strong'])
 assert first['asserted_from']==first['via'] and first['asserted_to']==shared
 for i in [1,2]:
  trace=path(hits[row['target_labels'][i]]);assert trace['from']==row['strong']
  assert trace['hops']==(1 if arm=='baseline' and i==1 else 2)
  via = row['strong'] if arm=='baseline' and i==1 else (bridge if i==1 else (low if arm=='baseline' else high))
  assert trace['via']==trace['asserted_from']==via and trace['asserted_to']==row['target_labels'][i]
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
computed = []
for i, case in enumerate(['cross_origin','later_hop','same_hop']):
 values = []
 for arm in ['baseline','scored']:
  row = rows[arm]
  values.append(score(next(h for h in row['targets'] if h['topic'] == row['target_labels'][i])))
 before, after = values
 computed.append({'case':case,'baseline':before,'scored':after,'expected':rows['scored']['expected_scores'][i], 'absolute_delta':after-before,'percent_change':100*(after-before)/before})
early = {'case':'early_stop','baseline':None,'scored':score({'why':rows['scored']['early_stop']['why']}),'expected':rows['scored']['early_stop']['expected_score'],'absolute_delta':None,'percent_change':None}
assert comparison['invariants'] == computed
assert comparison['early_stop'] == early
assert comparison['cross_arm_non_graph_identical'] == (rows['baseline']['non_graph'] == rows['scored']['non_graph']) is False
assert comparison['early_stop_recovered'] == (not rows['baseline']['early_stop']['reached'] and rows['scored']['early_stop']['reached']) is True
labels = ['Two origins reach one target','A later, stronger arrival from one origin','A stronger route at the same hop','Two-hop answer after 60 one-hop decoys']
lines = []
for label, values in zip(labels, computed+[early]):
 before = 'Absent' if values['baseline'] is None else f"{values['baseline']:.8f}"
 delta = 'N/A' if values['absolute_delta'] is None else f"+{values['absolute_delta']:.8f}"
 percent = 'N/A' if values['percent_change'] is None else f"+{values['percent_change']:.6f}%"
 lines.append(f"| {label} | {before} | {values['scored']:.8f} | {values['expected']:g} | {delta} | {percent} |")
readme = (ROOT/'README.md').read_text()
start = readme.index('| Controlled case |')
stop = readme.index('\n\n', start)
assert readme[start:stop].splitlines() == ['| Controlled case | Disabled arm | Scored arm | Explicit oracle | Absolute score change | Score change |','| --- | ---: | ---: | ---: | ---: | ---: |'] + lines, 'displayed score table differs from raw evidence'
for p in ROOT.rglob('*'):
 if not p.is_file() or p.name in {'verify.py','test_verify.py'}:continue
 data=p.read_text()
 assert not re.search(r'(?:/workspace/(?:scratch|\.pamin|\.cargo|\.onnxruntime|PaminMemory)|/home/|postgres(?:ql)?://|Bearer\s+[A-Za-z0-9]|claude\.ai/|app://)',data),f'private path/credential/session marker: {p.name}'
 assert p.suffix not in {'.onnx','.bin','.data','.so'},'binary/model material must not be published'
print('PASS: four native component cases, exact weak-rank arithmetic, fresh builds, exact query/binary pins, retrospective SQL payloads, disclosed historical hardware limits, pinned inference/assets, sanitized text-only evidence')
