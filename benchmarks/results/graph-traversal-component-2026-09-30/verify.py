#!/usr/bin/env python3
"""Read-only checks of retained synthetic component evidence; no runtime loading."""
import hashlib,json,math,re,sys,struct
if sys.flags.optimize:
 raise SystemExit("FAIL: Python optimization disables assertions; run without -O/-OO")
from pathlib import Path
ROOT=Path(__file__).resolve().parent
# Complete approved shell blocks, including final newline; prospective guards.
# Membership checks alone permit a later assignment to override a pinned value.
BUILD_RECIPE_SHA256='6808938a2e3d50deb5f244939e30844f1eb67d951324abbd442b0c22973b6e9f'
RUNTIME_RECIPE_SHA256='8c9404312e9a8f20fb8fade6c2380597aa4155a606c7ba7b94528980156c85d4'

def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-6)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def graph_only_evidence(hit):
 why=hit['why']
 graph=[w for w in why if w.get('kind')=='channel' and w.get('channel')=='graph']
 assert len(graph)==1, 'reached target must have exactly one graph-channel record'
 paths=[w for w in why if w.get('kind')=='path']
 assert len(paths)==1, 'reached target must have exactly one path record'
 assert paths[0].get('edge')=='related_to' and paths[0].get('derivation')=='deterministic', 'controlled path edge/derivation differs'
 assert len(why)==2, 'graph-only target complete Why must contain only graph and path records'
 return graph[0],paths[0]
# Frozen native Why values from the reviewed 2026-09-30 capture, not values
# supplied by the mutable comparison/manifest. UUID-dependent ranks apply
# only to these retained runs, not to independently seeded reproductions.
GRAPH_RECORD_PINS = {
 'baseline': [(4,0.020202022045850754),(6,0.01772727258503437),(7,0.01719697006046772)],
 'scored': [(1,0.025151517242193222),(4,0.02196969836950302),(7,0.02196969836950302),(8,0.01613636501133442)]}
def validate_graph_record(hit,arm,index):
 graph,_=graph_only_evidence(hit)
 assert set(graph)=={'kind','channel','rank','score','weight','contribution'}, 'graph channel fields differ'
 assert type(graph['rank']) is int and graph['rank']==GRAPH_RECORD_PINS[arm][index][0], 'retained graph rank differs'
 for key in ['score','weight','contribution']:
  assert type(graph[key]) in {int,float} and math.isfinite(graph[key]), 'invalid graph numeric field'
 assert graph['weight']==0.30000001192092896, 'retained graph weight differs'
 assert graph['contribution']==GRAPH_RECORD_PINS[arm][index][1], 'retained graph contribution differs'
def score(hit):return graph_only_evidence(hit)[0]['score']
def path(hit):return graph_only_evidence(hit)[1]
def fixture_content(topic):
 # Exact content formula in the hash-pinned measured fixture, indexed by origin.
 if topic=='quartzanchor':return 'orbital navigation calibration beacon'
 assert re.fullmatch(r'islandnode[0-9]{4}',topic) and 0 <= int(topic[10:]) < 240, 'invalid fixture origin topic'
 n=int(topic[10:])
 return f'orbital navigation calibration beacon archival report category {n % 17} revision {n}'
def load(name):return json.loads((ROOT/name).read_text())

provenance=load('provenance.json');comparison=load('comparison.json')
assert provenance['source_base']=='13ee710c9df865f1dac98dc77a8108e438ddc539'
MAIN_REF_NOT_RUN = '315c10242ddf7a1cec3bccbf550a942320e09557'
assert type(provenance.get('compared_main_ref_not_run')) is str and provenance['compared_main_ref_not_run']==MAIN_REF_NOT_RUN, 'prepared unrun main reference differs'
assert type(provenance.get('mapped_library_identity_scope')) is str and provenance['mapped_library_identity_scope']=='mapped paths and pinned file digests; no inode identity captured', 'mapped-library identity limitation differs'
assert provenance['scope']=='native search_fused component reproduction; independently seeded UUID projects; no product quality/speed conclusion', 'component provenance scope differs'
# Independent SHA-256 of Git blob 760cd200eebba61025615ea5a72384110f0ac9b5
# at reviewed commit a7eccf37096bbc1f9c6bdf670ba718459d05ff22.
assert sha(ROOT/'source/experimental-traversal.patch') == '9ab5997e25a5c7e33fc9c582dbeafc142a0792214216a3d318f7ddb380ab8eba', 'recorded traversal patch differs'
# Independent digest of the unchanged preparation script at published 271a875.
assert sha(ROOT/'source/prepare.py') == 'f3812344fb1fa5c81a56f174bf884858968b81a2fea5624e5cf6b450a9621c6b', 'recorded preparation script differs'
for name,expected in provenance['source_files'].items():assert sha(ROOT/'source'/name)==expected
ORIGINAL_REDACTION_PINS = {'baseline.jsonl': '302e70b2463379af148eaf8c3ced4f4d3606c1fc865af1b1bc38d1d69597929e',
 'baseline.log': '6f5b1350ed22fcf72dbc0a38dddd65ea99dace15f875298b0ae15becb077e025',
 'baseline.trace.jsonl': '5936b7d0cea61761299147254a2af812cadfc74d3fa126abf33b95ab3a5d6236',
 'baseline.usage.json': '7f675ce48907b49028a49dc8d3b531ec54a89b2fde695e024700bf70a0b3ac37',
 'scored.jsonl': '149ca7521684db7a8a621c5b8283841456ad9c08b1d9a5283a55d1ee6928a0fa',
 'scored.log': '59b0ae1975940198f03ec6ce7a8cb0ae9efc11c3752c2a097bf1e7046a7385be',
 'scored.trace.jsonl': '6efe6ec43b56afafd8fe5db7cf158eab5c5808c4bae28b6282ac1ee59f16287c',
 'scored.usage.json': 'e273d12d107522410ea3af260c4092e55b8c152a772576c7d0438d6bd5ee3d15'}
assert provenance['toolchain'] == {'rustc': 'rustc 1.98.1 (48a229cea 2026-09-01)\nbinary: rustc\ncommit-hash: 48a229ceaefd4985c50990b14116b6d856af0985\ncommit-date: 2026-09-01\nhost: x86_64-unknown-linux-gnu\nrelease: 1.98.1\nLLVM version: 22.1.8\n', 'cargo': 'cargo 1.98.1 (797e8a9bc 2026-08-05)\nrelease: 1.98.1\ncommit-hash: 797e8a9bca276c1c9f9f738d2a20f484fa4eea9d\ncommit-date: 2026-08-05\nhost: x86_64-unknown-linux-gnu\nlibgit2: 1.9.4 (sys:0.21.0 vendored)\nlibcurl: 8.21.0-DEV (sys:0.4.90+curl-8.21.0 vendored ssl:OpenSSL/3.6.3)\nssl: OpenSSL 3.6.3 9 Jun 2026\nos: Debian 13.0.0 (trixie) [64-bit]\n', 'build_flags': {'profile': 'release', 'jobs': 2, 'offline': True, 'locked': True, 'incremental': False, 'rustflags': None, 'encoded_rustflags': None, 'wrappers': None, 'native_link': 'explicit dynamic ORT1.28.0', 'inherited_product_knobs': 'cleared'}, 'runtime_product_settings': {'profile': 'accuracy', 'device': 'cpu', 'channel_depth': 50, 'graph_depth': 2, 'reranker': 'not called; component scope', 'rayon_omp_environment_threads': 4, 'pamin_inference_threads': None, 'ort_threads': 'product available_parallelism default; not directly recorded'}}, 'recorded toolchain/settings differ'
SHARED_MACHINE = 'exclusive model/build slot; shared file cache and reclaim pressure; non-timing component reproduction'
LAUNCH_SETTINGS_PINS = {'baseline': '56ff2d0db433b43726412700f2b1758f380c6cae31dd8767950edc2aa9405f4f', 'scored': '2cbe0450735170c4d2b0ba49f227b89378f95ad301b4c1ec6474053d2aa3cd37'}
ORIGINAL_LAUNCH_PINS = {'baseline': '72038565da7e2273384ee28a369df2a8eaf84f8eb4402eeb92b8b84f23af1f3d', 'scored': '33243aa7e35538e086352863f693250a535b5f0e6b66a62effe84af33c87ca65'}
for name,record in provenance['redactions'].items():
 assert type(record.get('changed')) is bool and record['changed'] == (record['original_sha256'] != record['published_sha256']), 'redaction changed flag differs'
 assert record['original_sha256'] == ORIGINAL_REDACTION_PINS[name], 'original redaction binding differs'
 assert sha(ROOT/name)==record['published_sha256'] and re.fullmatch('[0-9a-f]{64}',record['original_sha256'])
EXPECTED_FILES = {'source/multihop.rs.in', 'provenance.json', 'source/migrations/V10__settled_jobs_leave.sql', 'source/migrations/V12__job_subject_once.sql', 'source/migrations/V1__initial.sql', 'source/migrations/V8__topics_by_recency.sql', 'retrospective-sql-audit.json', 'source/migrations/V5__cascade_outbox.sql', 'source/graph_trace.rs.in', 'source/migrations/V9__state_content_from_span.sql', 'comparison.json', 'source/migrations/V4__current_state_pointer.sql', 'scored.trace.jsonl', 'scored.jsonl', 'README.md', 'scored.usage.json', 'scored.log', 'source/migrations/V11__retrieval_signals_leave.sql', 'test_verify.py', 'source/migrations/V3__shard_key_and_indexes.sql', 'source/prepare.py', 'source/migrations/V6__topic_name_index.sql', 'baseline.log', 'baseline.trace.jsonl', 'baseline.usage.json', 'source/migrations/V14__source_versions_index_once.sql', 'platform-observation.json', 'source/migrations/V13__edge_endpoints_on_versions.sql', 'source/migrations/V7__one_document_per_topic.sql', 'source/fixture.rs.in', 'baseline.jsonl', 'source/experimental-traversal.patch', 'verify.py', 'source/migrations/V2__relationships.sql'}
assert {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()} == EXPECTED_FILES, 'required file inventory differs'
assert set(provenance['archive_files']) == EXPECTED_FILES - {'provenance.json'}, 'hash inventory differs'
for name, digest in provenance['archive_files'].items():
 assert sha(ROOT/name) == digest, f'archive hash differs: {name}'
assert set(provenance['source_files']) == {'fixture.rs.in','multihop.rs.in','graph_trace.rs.in','experimental-traversal.patch','prepare.py'}
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
BINARY_BYTES_PINS = {'baseline': 15330992, 'scored': 15331248}
PAYLOAD_MAP_DIGESTS = {'baseline': 'b2cba4a77820f63e0ea869b5099f631705aae850dee17e6a2a4d287e29c4397b', 'scored': 'b2cba4a77820f63e0ea869b5099f631705aae850dee17e6a2a4d287e29c4397b'}
CURRENT_PLATFORM_PINS = {'cpu_model': 'AMD EPYC 9V74 80-Core Processor', 'kernel': '6.18.44', 'architecture': 'x86_64', 'cpu_quota': '400000 100000', 'cpu_affinity': [0, 1, 2, 3, 4], 'memory_max_bytes': 17179869184}
SQL_MAP_DIGEST = '6264b33b89775579e92b4d903a16b1fed165942498d5f2b3acc024b38e834513'
inventory_scope = provenance['source_inventory_scope']
assert inventory_scope['original_captured_entries_per_arm'] == 52
assert inventory_scope['original_status'] == 'partial: migration SQL omitted'
assert inventory_scope['original_sql_build_attestation'] == 'N/A: not captured'
assert inventory_scope['retrospective_sql_audit'] == 'retrospective-sql-audit.json'
audit = load('retrospective-sql-audit.json')
assert audit.get('scope') == 'Retrospective read-only inspection of preserved original binaries and frozen SQL sources; no historical rebuild or runtime execution. This is not a build-time source attestation.', 'retrospective SQL audit scope differs'
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
 assert type(binary.get('binary_bytes')) is int and binary['binary_bytes'] == BINARY_BYTES_PINS[arm], 'retained binary byte length differs: '+arm
 assert set(binary['migration_payloads']) == set(sql_sources)
 assert hashlib.sha256(json.dumps(binary['migration_payloads'],sort_keys=True,separators=(',',':')).encode()).hexdigest() == PAYLOAD_MAP_DIGESTS[arm], 'retrospective payload capture differs'
 for name, payload in binary['migration_payloads'].items():
  assert payload['sha256'] == sql_sources[name]['sha256'] and payload['bytes'] == sql_sources[name]['bytes']
  assert type(payload['first_binary_offset']) is int and 0 <= payload['first_binary_offset'] <= binary['binary_bytes'] - payload['bytes']
platform_observation = load(provenance['hardware_observation'])
assert platform_observation['scope']=='Current platform observation after the original runs; never substituted for missing historical fields.', 'current platform scope differs'
assert platform_observation['comparison']=='Original memory.max equals current memory.max for both arms. CPU model/kernel/quota/affinity equality with original execution is unknown.', 'historical hardware limitation differs'
assert re.fullmatch(r'2026-09-30T[0-9:.]+\+00:00', platform_observation['captured_at_utc'])
assert platform_observation['historical'] == {'cpu_model':'N/A: not captured','kernel':'N/A: not captured','cpu_quota':'N/A: not captured','cpu_affinity':'N/A: not captured','memory_max_bytes':{'baseline':17179869184,'scored':17179869184}}
assert platform_observation['current'] == CURRENT_PLATFORM_PINS, 'dated current platform capture differs'
assert platform_observation['current']['cpu_model'] and platform_observation['current']['kernel'] and platform_observation['current']['cpu_affinity']
assert platform_observation['current']['memory_max_bytes'] == 17179869184
assert platform_observation['current']['cpu_quota'] == '400000 100000'

RESOURCE_PINS = {'baseline': {'estimated_headroom_bytes': 8437116928,
              'oom_after': 0,
              'oom_before': 0,
              'oom_kill_after': 0,
              'oom_kill_before': 0,
              'reclaim_pressure': True,
              'scope': 'max-current+inactive_file; reclaim not guaranteed; no '
                       'cache drop'},
 'scored': {'estimated_headroom_bytes': 8410218496,
            'oom_after': 0,
            'oom_before': 0,
            'oom_kill_after': 0,
            'oom_kill_before': 0,
            'reclaim_pressure': True,
            'scope': 'max-current+inactive_file; reclaim not guaranteed; no '
                     'cache drop'}}
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
retrospective = provenance['retrospective_multihop_source']
assert retrospective['scope'] == 'retrospective equality of preserved baseline/scored scratch source copies; original build-time source byte attestation N/A: omitted from the 52-entry source maps'
assert re.fullmatch(r'2026-10-01T[0-9:.]+\+00:00', retrospective['captured_at_utc'])
assert retrospective['source'] == 'crates/pamin-engine/tests/scratch_scored_multihop.rs'
assert retrospective['published_file'] == 'source/multihop.rs.in'
assert retrospective['sha256'] == sha(ROOT/retrospective['published_file']) == '70d5127e265a475edc3ededc871dc91cd79a0bfb56a4e6feca4d1501fd206c00'
assert retrospective['bytes'] == (ROOT/retrospective['published_file']).stat().st_size
assert retrospective['arm_source_sha256'] == {arm:retrospective['sha256'] for arm in ['baseline','scored']}
rows = {}

for arm,weak_rank in [('baseline',23),('scored',22)]:
 row=load(f'{arm}.jsonl');record=arm_records[arm];rows[arm]=row
 resources=record['resources']
 assert resources==RESOURCE_PINS[arm], 'retained resource-pressure record differs'
 assert type(resources['reclaim_pressure']) is bool and all(type(resources[key]) is int for key in resources if key.endswith('_bytes') or key.startswith('oom_')), 'resource-pressure record types differ'
 launch = record['launch_binding']
 assert launch.get('scope') == 'projection of retained original launch; local path prefixes replaced by placeholders', 'launch projection scope differs'
 assert launch['original_launch_sha256'] == ORIGINAL_LAUNCH_PINS[arm]
 assert launch['command'] == ['${FROZEN_BINARY}','scratch_scored_graph_finite_fixture','--exact','--ignored','--nocapture','--test-threads=1']
 assert launch['GRAPH_OUT'] == '${RESULTS}/'+arm+'.jsonl' and launch['GRAPH_TRACE'] == '${RESULTS}/'+arm+'.trace.jsonl'
 settings=launch['effective_product_settings']
 assert settings['PAMIN_PROFILE']=='accuracy' and settings['PAMIN_DEVICE']=='cpu' and settings['GRAPH_ARM']==arm and settings['GRAPH_CLK_TCK']=='100'
 assert settings['GRAPH_SHARED_MACHINE']==SHARED_MACHINE
 assert settings['PAMIN_EVAL_HOME']=='${WORKSPACES}/'+arm
 assert settings['GRAPH_OUT']==launch['GRAPH_OUT'] and settings['GRAPH_TRACE']==launch['GRAPH_TRACE']
 assert hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()==LAUNCH_SETTINGS_PINS[arm], 'complete normalized launch projection differs'

 assert len(row['graph_process_cpu_user_system_seconds'])==2
 for value in [row['elapsed_ms'],row['early_stop']['elapsed_ms'],row['process_lifetime_high_water_kib'],*row['graph_process_cpu_user_system_seconds']]:
  assert type(value) in {int,float} and math.isfinite(value) and value>=0, 'invalid raw elapsed/CPU/RSS'
 assert row['shared_machine'] == SHARED_MACHINE, 'retained interference declaration differs'
 assert row['query'] == 'quartzanchor orbital navigation calibration beacon', 'native fixture query differs'
 assert row['record']=='fixture' and row['arm']==arm and row['documents']==241
 assert row.get('setup') == 'native write/drain, explicit OptimizeIndex queue/drain; runtime defaults preserved', 'recorded native fixture setup differs'
 assert row['expected_scores'] == [.8,.5,.5]
 assert row['weak_rank']==weak_rank and close(row['weak_relevance'],11/(10+weak_rank))
 channels = {'lexical_segmented','lexical_ngram','vector'}
 fixture_topics = {'quartzanchor'} | {f'islandnode{n:04d}' for n in range(240)}
 for hit in row['non_graph']:
  assert hit['topic'] in fixture_topics, 'non-graph topic outside fixture inventory'
  assert hit['ranks'] and all(rank.get('channel') in channels for rank in hit['ranks']), 'non-graph row needs allowed channel evidence'
  for rank in hit['ranks']:
   assert type(rank.get('rank')) is int and rank['rank']>0, 'non-graph rank must be a positive integer'
   assert rank.get('score') is None or (type(rank['score']) in {int,float} and math.isfinite(rank['score'])), 'non-graph score must be real non-boolean or null'
 assert row['strong']=='quartzanchor' and sum(h['topic']=='quartzanchor' for h in row['non_graph'])==1, 'strong origin must occur exactly once in retained inventory'
 seed_window=row['non_graph'][:63]
 assert sum(h['topic']=='quartzanchor' for h in seed_window)==1, 'strong origin must occur exactly once in fixture seed window'
 assert any(hit['topic']==row['weak'] for hit in seed_window), 'weak origin must occur in first 63 retained fused results'
 chosen=next(hit for hit in seed_window if hit['topic']!=row['strong'] and min(rank['rank'] for rank in hit['ranks'])>=22)
 assert chosen['topic']==row['weak'], 'weak origin must be first eligible topic in recorded fixture seed window'
 weak=next(r for r in seed_window if r['topic']==row['weak'])
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
 test_lines=[line for line in log.splitlines() if line.startswith('test ') and not line.startswith('test result:')]
 assert test_lines == ['test scratch_scored_graph_finite_fixture ... ok'], 'complete fixture test markers differ'
 summaries=[line for line in log.splitlines() if line.startswith('test result:')]
 assert len(summaries)==1 and re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in [0-9]+(?:\.[0-9]+)?s',summaries[0]), 'complete fixture result summary differs'
 assert log.rstrip().splitlines()[-1]==summaries[0], 'fixture result must be the final log line'
 assert not re.search(r'(?m)^failures:|\bFAILED\b|thread .+ panicked at',log), 'fixture log contains a failure marker'
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
 usage=record['process_usage']
 assert set(usage)=={'wall_seconds','user_seconds','system_seconds','maximum_process_rss_kib','exit_status','scope'}
 for metric in ['wall_seconds','user_seconds','system_seconds','maximum_process_rss_kib']:
  value=usage[metric]
  assert type(value) in {int,float} and math.isfinite(value) and value>=0, 'invalid duration/RSS: '+metric
 assert usage['maximum_process_rss_kib']==row['process_lifetime_high_water_kib'], 'raw and process usage RSS differ'
 assert usage['scope']=='Linux wait4 of native test process; setup included, independent services excluded'

 assert len(row['targets'])==3 and len({h['topic'] for h in row['targets']})==3, 'exactly three unique target records required'
 hits={h['topic']:h for h in row['targets']};assert set(hits)==set(row['target_labels'])
 for i,name in enumerate(row['target_labels']):
  origin=row['weak'] if arm=='baseline' and i==0 else row['strong']
  validate_graph_record(hits[name],arm,i)
  assert hits[name].get('seed') == fixture_content(origin), 'target seed content differs from expected origin: '+arm+' '+name
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
  validate_graph_record({'why':early['why']},arm,3)
  assert close(score({'why':early['why']}),.5)
  trace=path({'why':early['why']});assert trace['from']==row['strong'] and trace['via']==early['via'] and trace['hops']==2
  assert trace['asserted_from']==early['via'] and trace['asserted_to']==early['target']
 else:assert early['why'] is None, 'absent early-stop target must have no evidence'
 fresh=record['fresh_compiler_artifacts'];assert all(r['fresh'] is False for r in fresh)
 assert len(fresh)==6 and len({r['target'] for r in fresh})==6, 'exactly six fresh target records required'
 assert all(r['source']==('crates/pamin-engine/tests/'+r['target']+'.rs' if r['target'].startswith('scratch_') else 'crates/'+r['target'].replace('_','-')+'/src/lib.rs') for r in fresh)
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
assert comparison['scope'] == 'native Engine component fixture; fresh UUID projects may change tie orders; no timing claim', 'component comparison scope differs'
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
build_commands=[block for block in re.findall(r'```sh\n(.*?)```',readme,re.S) if 'cargo test -p pamin-engine' in block]
assert len(build_commands)==1, 'exact native-library build reproduction command required'
assert hashlib.sha256(build_commands[0].encode()).hexdigest()==BUILD_RECIPE_SHA256, 'pinned native-library build environment missing or changed: complete approved recipe differs'
runtime_commands=[block for block in re.findall(r'```sh\n(.*?)```',readme,re.S) if '"$BINARY" scratch_scored_graph_finite_fixture' in block]
assert len(runtime_commands)==1 and hashlib.sha256(runtime_commands[0].encode()).hexdigest()==RUNTIME_RECIPE_SHA256, 'pinned native-library runtime loader environment missing or changed: complete approved recipe differs'
assert re.findall(r'The recorded local main ref was `([^`]+)`;',readme)==[MAIN_REF_NOT_RUN], 'README prepared main reference differs'
assert readme.count(f'The recorded local main ref was `{MAIN_REF_NOT_RUN}`;\nthat separate main arm was prepared but was not executed here.')==1, 'README prepared main execution limitation differs'
assert 'PAMIN_EVAL_HOME="$WORKSPACE" PAMIN_PROFILE=accuracy PAMIN_DEVICE=cpu' in readme, 'concrete reproduction workspace binding missing'
assert '--test scratch_scored_fixture --test scratch_scored_multihop' in readme, 'reproduction must compile both retained targets'
start = readme.index('| Controlled case |')
stop = readme.index('\n\n', start)
assert readme[start:stop].splitlines() == ['| Controlled case | Disabled arm | Scored arm | Explicit oracle | Absolute score change | Score change |','| --- | ---: | ---: | ---: | ---: | ---: |'] + lines, 'displayed score table differs from raw evidence'
for p in ROOT.rglob('*'):
 if not p.is_file() or p.name in {'verify.py','test_verify.py'}:continue
 data=p.read_text()
 assert not re.search(r'(?:/workspace/(?:scratch|\.pamin|\.cargo|\.onnxruntime|PaminMemory)|/home/|postgres(?:ql)?://|Bearer\s+[A-Za-z0-9]|claude\.ai/|app://)',data),f'private path/credential/session marker: {p.name}'
 assert p.suffix not in {'.onnx','.bin','.data','.so'},'binary/model material must not be published'
print('PASS: four native component cases, exact weak-rank arithmetic, fresh builds, exact query/binary pins, retrospective SQL payloads, disclosed historical hardware limits, pinned inference/assets, sanitized text-only evidence')
