#!/usr/bin/env python3
"""Read-only retrospective binding; requires local Git objects for the three revisions."""
import ast,gzip,hashlib,json,subprocess,sys
from pathlib import Path
if sys.flags.optimize:raise SystemExit('Rebuild verification requires assertions; remove -O/PYTHONOPTIMIZE')
sys.dont_write_bytecode=True
TARGET='crates/pamin-engine/tests/scratch_restart_floor.rs'
PRODUCTS={'pamin-core','pamin-store','pamin-index','pamin-engine'}
REBUILD_SCOPE='Named-package clean in a shared Cargo target freshly compiled only the four product libraries and selected integration-test helper. All 538 third-party compiler artifacts per arm were cached. Their source-to-artifact provenance is unknown; byte equality does not establish full source-to-artifact attestation.'
def historical_paths(repo):
    # Recover the documented original locations from the immutable historical
    # builder, without executing it or embedding raw checkout paths here.
    tree=ast.parse((repo/'benchmarks/harnesses/restart-floor-2026-09-30/build.py.in').read_text())
    root=next(Path(ast.literal_eval(node.value.args[0])) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='root' for t in node.targets))
    env=next({key.arg:ast.literal_eval(key.value) for key in node.value.keywords} for node in tree.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='update')
    return {'${REBUILD_AUDIT}':str(root.parent/'restart-floor-rebuild-audit-new'),
            '${SCRATCH}':str(root),'${TARGET_DIR}':env['CARGO_TARGET_DIR'],
            '${REPO}':str(Path(env['CARGO_TARGET_DIR']).parent),
            '${ORT_ROOT}':str(Path(env['ORT_LIB_LOCATION']).parents[1]),
            '${CARGO_HOME}':env['CARGO_HOME'],'${RUSTUP_HOME}':env['RUSTUP_HOME']}
def sha(data):return hashlib.sha256(data).hexdigest()
def text(root,name):return gzip.decompress((root/name).read_bytes()).decode()
def original_bytes(value,paths):
    for public,private in paths.items():value=value.replace(public,private)
    return value.encode()
def compiler(value,checkout):
    rows=[json.loads(line) for line in value.splitlines() if line.startswith('{')]
    selected=[r for r in rows if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name','').replace('_','-') in PRODUCTS|{'scratch-restart-floor'}]
    assert len(selected)==5 and {r['target']['name'].replace('_','-') for r in selected}==PRODUCTS|{'scratch-restart-floor'}
    for r in selected:
        name=r['target']['name'].replace('_','-')
        expected=checkout+'/crates/'+('pamin-engine' if name=='scratch-restart-floor' else name)+'/Cargo.toml'
        assert r['manifest_path']==expected, 'compiler used another source tree'
    assert any(r.get('reason')=='build-finished' and r['success'] is True for r in rows)
    return selected
def source_inputs(repo,commit):
    tree=subprocess.check_output(['git','ls-tree','-r','-z',commit],cwd=repo)
    entries=[entry.split(b'\t',1) for entry in tree.split(b'\0') if entry]
    header=[a.split() for a,b in entries]
    blobs=subprocess.check_output(['git','cat-file','--batch'],cwd=repo,input=b'\n'.join(h[2] for h in header if h[0]!=b'160000')+b'\n')
    offset=0;result={}
    for (meta,name),h in zip(entries,header):
        mode,kind,oid=h;name=name.decode()
        if mode==b'160000':result[name]={'kind':'gitlink','commit':oid.decode()};continue
        end=blobs.index(b'\n',offset);blob_header=blobs[offset:end].split();assert blob_header[:2]==[oid,b'blob']
        count=int(blob_header[2]);value=blobs[end+1:end+count+1];offset=end+count+2
        if mode==b'120000':result[name]={'kind':'symlink','target':value.decode(),'sha256':sha(value)}
        else:result[name]={'kind':'file','bytes':count,'sha256':sha(value)}
    return result
def verify(root=None,repo=None,audit=None,scope_review=None):
    root=Path(root or Path(__file__).resolve().parent)
    disk=root.parent
    repo=Path(repo or disk.parents[3])
    audit=audit or json.loads(text(root,'audit.json.gz'))
    scope_review=scope_review or json.loads((root/'scope-review.json').read_text())
    assert scope_review['recorded_utc']
    assert scope_review['scope']==REBUILD_SCOPE, 'rebuild qualification wording differs'
    assert scope_review['historical_audit_sha256']==sha((root/'audit.json.gz').read_bytes())
    assert scope_review['historical_procedure_sha256']==sha((root/'audit.py.in').read_bytes())
    assert scope_review['target_scope']=='shared Cargo target; cargo clean --release -p only for the four product packages'
    assert scope_review['fresh_product_and_helper_artifacts_per_arm']==5
    assert scope_review['third_party_source_to_artifact_provenance']=='unknown' and scope_review['full_source_to_artifact_attestation'] is False, 'rebuild qualification falsely certifies cached third-party provenance'
    assert set(scope_review['cached_third_party_artifacts_per_arm'])=={'main','predecessor','candidate'}
    paths=historical_paths(repo)
    historical=json.loads((disk/'binaries.json').read_text())
    memory=json.loads((disk.with_name('restart-floor-2026-09-30')/'binaries.json').read_text())
    provenance=json.loads((disk/'provenance.json').read_text())
    memory_provenance=json.loads((disk.with_name('restart-floor-2026-09-30')/'provenance.json').read_text())
    for value,record_path in [(provenance,'rebuild/scope-review.json'),(memory_provenance,'../restart-floor-disk-2026-09-30/rebuild/scope-review.json')]:
        assert value['retrospective_rebuild_scope_review']=={'record':record_path,'qualification':REBUILD_SCOPE}, 'rebuild provenance qualification differs'
    assert audit['all_three_byte_for_byte_identical'] is True and audit['started_utc'] and audit['finished_utc']
    assert set(audit['arms'])=={r['arm'] for r in historical}=={'main','predecessor','candidate'}
    assert audit['toolchain']==audit['toolchain_after'] and audit['toolchain'].startswith('rustc 1.98.1 ')
    assert audit['cargo']==audit['cargo_after'] and audit['cargo'].startswith('cargo 1.98.1 ')
    before,after=audit['before'],audit['after']
    script=original_bytes((root/'audit.py.in').read_text(),paths)
    assert sha(script)==before['script']['sha256']==after['script']['sha256']
    assert len(script)==before['script']['bytes']==after['script']['bytes']
    assert before['arms']==after['arms'] and before['libraries']==after['libraries']
    assert before['build_environment']==after['build_environment']
    assert before['current_inherited_pamin_environment']==after['current_inherited_pamin_environment']=={}
    env=before['build_environment']
    assert env['CARGO_TARGET_DIR']=='${TARGET_DIR}' and env['CARGO_BUILD_JOBS']=='4'
    assert env['ORT_PREFER_DYNAMIC_LINK']=='1' and env['ORT_LIB_LOCATION']=='${ORT_ROOT}/onnxruntime-linux-x64-1.28.0/lib'
    for name in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER']:assert env[name] is None
    assert set(before['libraries'])=={'runtime_library','native_zvec_library'}
    for key,value in before['libraries'].items():
        assert (value['sha256'],value['bytes'])==(provenance[key]['sha256'],provenance[key]['bytes'])
    harness=(repo/'benchmarks/harnesses/restart-floor-2026-09-30/harness.rs.in').read_bytes()
    assert before['original_harness']['sha256']==after['original_harness']['sha256']==sha(harness)
    for old in historical:
        arm=old['arm'];commit=old['commit'];record=audit['arms'][arm];prior=before['arms'][arm]
        assert prior['commit']==commit
        memory_old=next(r for r in memory if r['arm']==arm)
        assert memory_old['commit']==commit and memory_old['sha256']==old['sha256']
        assert record['byte_for_byte_identical'] is True
        for binary in [prior['historical_binary'],record['historical_binary_after'],record['rebuilt_binary']]:
            assert binary['sha256']==old['sha256'] and binary['bytes']==old['current_pretrial_hash']['bytes']==old['posttrial_hash']['bytes']
        assert record['inputs_before']==record['inputs_after']==prior['snapshot']
        snapshot=prior['snapshot'];assert snapshot['tracked_status']=='clean' and snapshot['tracked_regular_contents_equal_git_blobs'] is True
        expected=source_inputs(repo,commit)
        expected[TARGET]={'kind':'selected_ignored_test','bytes':len(harness),'sha256':sha(harness)}
        assert len(expected)==len(snapshot['inputs'])==384 and snapshot['inputs']==expected, 'source inputs disagree with pristine revision/selected harness'
        unselected=set(snapshot['unselected_ignored_tests'])
        assert unselected==({'crates/pamin-index/tests/scratch_disk_schema.rs','crates/pamin-index/tests/scratch_legacy_schema.rs'} if arm=='candidate' else set())
        checkout='${SCRATCH}/checkouts/'+arm
        original_json=text(root,f'original-{arm}-build.jsonl.gz')
        assert sha(original_bytes(original_json,paths))==prior['original_compiler_json']['sha256']
        assert len(original_bytes(original_json,paths))==prior['original_compiler_json']['bytes']
        assert compiler(original_json,checkout)==prior['original_selected_compiler_artifacts']
        original_log=original_bytes(text(root,f'original-{arm}-build.log.gz'),paths)
        assert sha(original_log)==prior['original_build_log']['sha256'] and len(original_log)==prior['original_build_log']['bytes']
        fresh_json=text(root,f'{arm}-build.jsonl.gz');selected=compiler(fresh_json,checkout)
        assert selected==record['compiler_artifacts'] and all(r['fresh'] is False for r in selected)
        all_rows=[json.loads(line) for line in fresh_json.splitlines() if line.startswith('{')]
        all_artifacts=[r for r in all_rows if r.get('reason')=='compiler-artifact']
        third_party=[r for r in all_artifacts if r not in selected]
        assert len(third_party)==scope_review['cached_third_party_artifacts_per_arm'][arm]==538 and all(r['fresh'] is True for r in third_party), 'rebuild cached third-party scope differs from compiler records'
        for name,identity in record['logs'].items():
            raw=original_bytes(text(root,name+'.gz'),paths)
            assert sha(raw)==identity['sha256'] and len(raw)==identity['bytes'], 'retained build log identity differs'
        assert record['clean_command']==['cargo','+1.98.1','clean','--release',*[x for p in sorted(PRODUCTS) for x in ['-p',p]]]
        assert record['build_command']==['cargo','+1.98.1','test','-p','pamin-engine','--test','scratch_restart_floor','--no-run','--release','--offline','--locked','--message-format=json']
    table=[line for line in (root/'README.md').read_text().splitlines() if line.startswith('|')]
    expected_table=['| Arm | Revision | Executable bytes | Rebuilt SHA256 equals original |','| --- | --- | ---: | --- |']
    for arm in ['main','predecessor','candidate']:
        old=next(r for r in historical if r['arm']==arm)
        expected_table.append(f"| {arm} | {old['commit']} | {old['current_pretrial_hash']['bytes']} | {old['sha256']} |")
    assert table==expected_table, 'rebuild identity table differs from verified archive'
    return audit
if __name__=='__main__':
    verify()
    print('Verified 3 byte-identical retrospective rebuilds: HNSW/Disk frozen identities, pristine revision + 384 inputs/arm, original/fresh raw compiler logs, all4+helper fresh:false, toolchain/flags/runtime unchanged; 538 cached third-party artifacts/arm have unknown source-to-artifact provenance; no full source-to-artifact attestation or historical build-time/tuning certification.')
