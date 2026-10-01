#!/usr/bin/env python3
"""Mutate temporary archive copies, refresh digests, and require semantic rejection."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
# The archive has an exact file inventory: imports must never write caches.
sys.dont_write_bytecode = True
import tempfile

ROOT = Path(__file__).resolve().parent


def refresh(root):
    manifest = root / 'provenance.json'
    p = json.loads(manifest.read_text())
    for name in p['archive_files']:
        p['archive_files'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in p['source_files']:
        p['source_files'][name] = hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest()
    for name, record in p['redactions'].items():
        record['published_sha256'] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    manifest.write_text(json.dumps(p, indent=2) + '\n')


def mutate_json(path, change, jsonl=False):
    if jsonl:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    else:
        row = json.loads(path.read_text())
        change(row)
        path.write_text(json.dumps(row, indent=2) + '\n')


def run_case(label, mutate=None, flags=(), refresh_hashes=True, expected_error=None):
    with tempfile.TemporaryDirectory(prefix='graph-archive-negative-') as temp:
        root = Path(temp) / 'archive'
        shutil.copytree(ROOT, root)
        if mutate:
            mutate(root)
        if refresh_hashes:
            refresh(root)
        result = subprocess.run([sys.executable, *flags, str(root / 'verify.py')], capture_output=True, text=True)
        if result.returncode == 0 or 'PASS:' in result.stdout:
            raise SystemExit(f'FAIL: {label} was accepted')
        if expected_error is not None and expected_error not in result.stderr:
            raise SystemExit(f'FAIL: {label} rejected for wrong reason: {result.stderr}')
        print(f'PASS: rejected {label}')


def reject_optimized_prepare(label, flags=(), optimize_env=None):
    with tempfile.TemporaryDirectory(prefix='graph-prepare-negative-') as temp:
        output = Path(temp)/'never-created'
        env = os.environ.copy()
        env.pop('PYTHONOPTIMIZE', None)
        if optimize_env is not None:
            env['PYTHONOPTIMIZE'] = optimize_env
        result = subprocess.run([sys.executable, *flags, str(ROOT/'source/prepare.py'), '--source', str(Path(temp)/'absent-input'), '--out', str(output)], env=env, capture_output=True, text=True)
        if result.returncode == 0 or output.exists() or 'Python optimization disables assertions' not in result.stderr:
            raise SystemExit(f'FAIL: prepare {label} did not reject before input/output access')
        print(f'PASS: rejected prepare {label} before writes')


def round12_interval_channel_cases():
    for arm in ['baseline','scored']:
        for metric in ['elapsed_ms','early_elapsed_ms','user','system']:
            def excessive(root,arm=arm,metric=metric):
                def change(row):
                    if metric=='elapsed_ms':row['elapsed_ms']=1_000_000_000
                    elif metric=='early_elapsed_ms':row['early_stop']['elapsed_ms']=1_000_000_000
                    else:row['graph_process_cpu_user_system_seconds'][0 if metric=='user' else 1]=1000
                mutate_json(root/(arm+'.jsonl'),change)
            run_case('excessive interval '+arm+'/'+metric,excessive,expected_error='graph interval '+('CPU' if metric in ['user','system'] else 'wall')+' time exceeds whole-process total')
        def duplicate_channel(root,arm=arm):
            def change(row):
                donor=next(h for h in row['non_graph'] if len(h['ranks'])>1 and any(rank['channel']=='lexical_ngram' for rank in h['ranks']))
                receiver=next(h for h in row['non_graph'] if h is not donor and any(rank['channel']=='lexical_ngram' for rank in h['ranks']))
                rank=next(rank for rank in donor['ranks'] if rank['channel']=='lexical_ngram')
                donor['ranks'].remove(rank);receiver['ranks'].append(rank)
            mutate_json(root/(arm+'.jsonl'),change)
        run_case('coherently moved duplicate channel '+arm,duplicate_channel,expected_error='non-graph hit must have one rank per channel')


def round10_native_type_cases():
    for arm in ['baseline','scored']:
        for value in [True,False,'1',None,float('inf')]:
            run_case('edge confidence type '+arm+'/'+repr(value),lambda root,arm=arm,value=value:mutate_json(root/(arm+'.jsonl'),lambda row:next(e for e in row['known_edges'] if e['confidence']==1).update(confidence=value)),expected_error='controlled edge confidence must be finite non-boolean')
        for value in [False,0.0,'0',None]:
            def change_status(root,arm=arm,value=value):
                mutate_json(root/(arm+'.usage.json'),lambda u:u.update(exit_status=value))
                mutate_json(root/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['process_usage'].update(exit_status=value))
            run_case('exit status type '+arm+'/'+repr(value),change_status,expected_error='process exit status must be integer zero')
        for value in ['not-a-uuid',None,True,123,'01a0f44292f37151b129111520058494']:
            run_case('native topic UUID '+arm+'/'+repr(value),lambda root,arm=arm,value=value:mutate_json(root/(arm+'.jsonl'),lambda row:row['non_graph'][0].update(id=value)),expected_error='non-graph identifier must be a canonical UUID')


def complete_non_graph_cases():
    for arm in ['baseline','scored']:
        for field,value,error in [('rank',True,'non-graph rank must be a positive integer'),('score',True,'non-graph score must be real non-boolean or null')]:
            run_case('boolean non-graph '+arm+'/'+field,lambda root,arm=arm,field=field,value=value:mutate_json(root/(arm+'.jsonl'),lambda row:row['non_graph'][0]['ranks'][0].update({field:value})),expected_error=error)
        def missing(root,arm=arm):
            def change(row):
                visible={h['topic'] for h in row['non_graph']}
                unused=next(f'islandnode{n:04}' for n in reversed(range(240)) if f'islandnode{n:04}' not in visible)
                next(h for h in row['non_graph'] if h['topic']=='quartzanchor')['topic']=unused
            mutate_json(root/(arm+'.jsonl'),change)
        run_case('missing strong seed '+arm,missing,expected_error='strong origin must occur exactly once in retained inventory')
        def duplicate(root,arm=arm):
            mutate_json(root/(arm+'.jsonl'),lambda row:row['non_graph'].append(copy.deepcopy(row['non_graph'][0])))
        run_case('duplicate strong seed '+arm,duplicate,expected_error='strong origin must occur exactly once in retained inventory')
        def outside(root,arm=arm):
            def change(row):
                strong=next(h for h in row['non_graph'] if h['topic']=='quartzanchor')
                row['non_graph'].remove(strong);row['non_graph'].append(strong)
            mutate_json(root/(arm+'.jsonl'),change)
        run_case('strong outside seed window '+arm,outside,expected_error='strong origin must occur exactly once in fixture seed window')


def complete_graph_record_cases():
    for arm in ['baseline','scored']:
        for index in range(4 if arm=='scored' else 3):
            for field,value,error in [('rank',999,'retained graph rank differs'),('rank',True,'retained graph rank differs'),('weight',-10,'retained graph weight differs'),('contribution',999,'retained graph contribution differs'),('score',True,'invalid graph numeric field')]:
                def mutate(root,arm=arm,index=index,field=field,value=value):
                    def change(row):
                        why=row['targets'][index]['why'] if index<3 else row['early_stop']['why']
                        next(w for w in why if w.get('channel')=='graph')[field]=value
                    mutate_json(root/(arm+'.jsonl'),change)
                run_case('complete graph '+arm+'/'+str(index)+'/'+field+'/'+str(value),mutate,expected_error=error)


def round7_graph_only_cases():
    for arm in ['baseline', 'scored']:
        for index in range(4 if arm == 'scored' else 3):
            for channel in ['vector', 'lexical_segmented', 'lexical_ngram']:
                def extra_channel(root, arm=arm, index=index, channel=channel):
                    def change(row):
                        why = row['targets'][index]['why'] if index < 3 else row['early_stop']['why']
                        why.append({'kind': 'channel', 'channel': channel, 'rank': 1, 'score': 1.0})
                    mutate_json(root / (arm + '.jsonl'), change)
                run_case('extra non-graph Why ' + arm + '/' + str(index) + '/' + channel,
                         extra_channel, expected_error='complete Why must contain only graph and path records')
            def extra_kind(root, arm=arm, index=index):
                def change(row):
                    why = row['targets'][index]['why'] if index < 3 else row['early_stop']['why']
                    why.append({'kind': 'unrelated', 'value': 1})
                mutate_json(root / (arm + '.jsonl'), change)
            run_case('extra other Why kind ' + arm + '/' + str(index), extra_kind,
                     expected_error='complete Why must contain only graph and path records')


def round7_weak_window_cases():
    def reseed(root, arm, outside):
        def change(row):
            window = row['non_graph'][63:] if outside else row['non_graph'][:63]
            replacement = next(hit for hit in window if hit['topic'] not in {row['weak'], row['strong']}
                               and min(rank['rank'] for rank in hit['ranks']) == row['weak_rank'])
            old, new = row['weak'], replacement['topic']
            def content(topic):
                number = int(topic[10:])
                return f'orbital navigation calibration beacon archival report category {number % 17} revision {number}'
            row['weak'] = new
            for edge in row['known_edges']:
                for key in ['from', 'to']:
                    if edge[key] == old:
                        edge[key] = new
            for hit in row['targets']:
                if hit['seed'] == content(old):
                    hit['seed'] = content(new)
                for why in hit['why']:
                    if why.get('kind') == 'path':
                        for key in ['from', 'via', 'asserted_from', 'asserted_to']:
                            if why.get(key) == old:
                                why[key] = new
        mutate_json(root / (arm + '.jsonl'), change)
    for arm in ['baseline', 'scored']:
        run_case('coherent out-of-window weak reseed ' + arm,
                 lambda root, arm=arm: reseed(root, arm, True),
                 expected_error='weak origin must occur in first 63 retained fused results')
        run_case('coherent later eligible weak reseed ' + arm,
                 lambda root, arm=arm: reseed(root, arm, False),
                 expected_error='weak origin must be first eligible topic in recorded fixture seed window')


def round7_native_environment_cases():
    build_settings = ['env -u ORT_LIB_PATH', 'ORT_LIB_LOCATION="$ORT_LIB"', 'ORT_PREFER_DYNAMIC_LINK=1',
                      'ZVEC_LIB_DIR="$ZVEC_LIB"', 'ZVEC_AUTO_BUILD=0',
                      'LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB"']
    for setting in build_settings:
        for replacement in ['', setting.replace('=', '=_wrong_', 1) if '=' in setting else 'env -u WRONG_PATH']:
            def change_build(root, setting=setting, replacement=replacement):
                path = root / 'README.md'
                text = path.read_text()
                start = text.index('\nenv -u ORT_LIB_PATH')
                end = text.index('```', start)
                path.write_text(text[:start] + text[start:end].replace(setting, replacement, 1) + text[end:])
            run_case('native build setting ' + setting + ' -> ' + repr(replacement), change_build,
                     expected_error='pinned native-library build environment missing or changed')
    for replacement in ['', 'LD_LIBRARY_PATH="$ORT_LIB"']:
        def change_runtime(root, replacement=replacement):
            path = root / 'README.md'
            text = path.read_text()
            prefix, command = text.split('LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB"', 1)
            path.write_text(prefix + 'LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB"' +
                            command.replace('LD_LIBRARY_PATH="$ORT_LIB:$ZVEC_LIB"', replacement, 1))
        run_case('native runtime loader setting -> ' + repr(replacement), change_runtime,
                 expected_error='pinned native-library runtime loader environment missing or changed')
    # Earlier approved tokens must not hide later effective overrides. Every
    # mutation refreshes the README's archive digest before semantic checks.
    for override in ['ORT_LIB_LOCATION="$WRONG_ORT"','ORT_PREFER_DYNAMIC_LINK=0',
                     'ZVEC_LIB_DIR="$WRONG_ZVEC"','ZVEC_AUTO_BUILD=1',
                     'LD_LIBRARY_PATH="$WRONG_LIB"']:
        def later_build(root, override=override):
            path=root/'README.md';text=path.read_text()
            old='CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 cargo +"$TOOLCHAIN" test -p pamin-engine'
            assert text.count(old)==1
            path.write_text(text.replace(old,override+' \\\n'+old,1))
        run_case('later effective native build override '+override,later_build,
                 expected_error='pinned native-library build environment missing or changed')
    def later_runtime(root):
        path=root/'README.md';text=path.read_text()
        old='PAMIN_EVAL_HOME="$WORKSPACE" PAMIN_PROFILE=accuracy PAMIN_DEVICE=cpu'
        assert text.count(old)==1
        path.write_text(text.replace(old,'LD_LIBRARY_PATH="$WRONG_LIB" \\\n'+old,1))
    run_case('later effective native runtime loader override',later_runtime,
             expected_error='pinned native-library runtime loader environment missing or changed')


def round8_cases():
    run_case('replacement preparation script', lambda r: (r/'source/prepare.py').write_text('not Python\n'), expected_error='recorded preparation script differs')
    for arm in ['baseline', 'scored']:
        for index in range(4 if arm == 'scored' else 3):
            for key, value in [('edge', 'contradicts'), ('derivation', 'explicit')]:
                def change_path(root, arm=arm, index=index, key=key, value=value):
                    def change(row):
                        why = row['targets'][index]['why'] if index < 3 else row['early_stop']['why']
                        next(record for record in why if record['kind']=='path')[key] = value
                    mutate_json(root/(arm+'.jsonl'), change)
                run_case('controlled path '+arm+'/'+str(index)+'/'+key, change_path, expected_error='controlled path edge/derivation differs')
        for topic, ranks, error in [
            ('fabricatedtopic', [], 'non-graph topic outside fixture inventory'),
            ('islandnode0240', [], 'non-graph topic outside fixture inventory'),
            ('islandnode0239', [], 'non-graph row needs allowed channel evidence'),
            ('islandnode0239', [{'channel':'unsupported','rank':1,'score':1}], 'non-graph row needs allowed channel evidence')]:
            def unsupported(root, arm=arm, topic=topic, ranks=ranks):
                def change(row):
                    existing = next((hit for hit in row['non_graph'] if hit['topic']==topic), None)
                    if existing is not None:
                        existing['ranks'] = ranks
                    else:
                        row['non_graph'].append({'topic':topic,'id':'00000000-0000-0000-0000-000000000000','ranks':ranks})
                mutate_json(root/(arm+'.jsonl'), change)
            run_case('unsupported non-graph '+arm+'/'+topic+'/'+str(ranks), unsupported, expected_error=error)
        run_case('overstated launch scope '+arm, lambda r,arm=arm: mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['launch_binding'].update(scope='exact unredacted historical launch')),expected_error='launch projection scope differs')
    retained=json.loads((ROOT/'provenance.json').read_text())
    for name, record in retained['redactions'].items():
        run_case('incorrect redaction flag '+name, lambda r,name=name,record=record:mutate_json(r/'provenance.json',lambda p:p['redactions'][name].update(changed=not record['changed'])),expected_error='redaction changed flag differs')
    # Exercise only harmless stand-ins, never Cargo or the native fixture.
    import re, shlex
    readme=(ROOT/'README.md').read_text()
    blocks=re.findall(r'```sh\n(.*?)```',readme,re.S)
    build=next(block for block in blocks if 'CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0' in block)
    for original,replacement in [('unset RUSTUP_TOOLCHAIN','unset'),('TOOLCHAIN=1.98.1-x86_64-unknown-linux-gnu','TOOLCHAIN=stable'),('cd "$ARMS/$ARM"','cd "."'),('ARMS=$(cd "$ARMS" && pwd -P)','ARMS="$ARMS"')]:
        def changed_selector(root,original=original,replacement=replacement):
            path=root/'README.md';text=path.read_text();assert original in text;path.write_text(text.replace(original,replacement,1))
        run_case('toolchain recipe selector '+original,changed_selector,expected_error='pinned native-library build environment missing or changed')
    rust_keys=['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER','RUSTUP_TOOLCHAIN','RUSTC','RUSTDOC']
    with tempfile.TemporaryDirectory(prefix='graph-recipe-inert-') as temp:
        stub=Path(temp)/'cargo'
        toolchain=json.loads((ROOT/'provenance.json').read_text())['toolchain']
        for name in ['cargo','rustc']:
            stub=Path(temp)/name
            code='import json,os,sys\n'
            code+='assert sys.argv[1]=="+1.98.1-x86_64-unknown-linux-gnu"\n'
            code+='if sys.argv[2:]==["-Vv"]: print(os.environ.get("STUB_VERSION_"+'+repr(name)+','+repr(toolchain[name])+'),end="")\n'
            code+='else:\n result=dict(os.environ);result["__argv__"]=sys.argv[1:];result["__cwd__"]=os.getcwd();print(json.dumps(result))\n'
            stub.write_text('#!'+sys.executable+'\n'+code);stub.chmod(0o700)
        env=os.environ.copy();env.update({key:'must-be-cleared' for key in rust_keys})
        env.update(PATH=temp+os.pathsep+env['PATH'],ORT_LIB='/retained-ort',ZVEC_LIB='/retained-zvec',ARMS=str(Path(temp)/'prepared arms'))
        for arm in ['baseline','scored']:
            manifest=Path(env['ARMS'])/arm/'Cargo.toml'
            manifest.parent.mkdir(parents=True);manifest.write_text('# synthetic '+arm+' manifest\n')
            env['ARM']=arm
            # Run from the evidence directory, where real Cargo would otherwise
            # discover the enclosing repository. Only the Python stand-in runs.
            result=subprocess.run(['sh','-c',build],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            captured=json.loads(result.stdout)
            assert captured['__cwd__']==str(manifest.parent), 'build working directory must be generated arm'
            assert not set(rust_keys).intersection(captured), 'build recipe inherited flags/wrappers'
            args=captured['__argv__'];position=args.index('--manifest-path')
            assert args[position+1]==str(manifest) and manifest.is_file(), 'build recipe must select generated arm manifest'
            assert [args[i+1] for i,value in enumerate(args[:-1]) if value=='--test']==['scratch_scored_fixture','scratch_scored_multihop']
            print('PASS: inert build selects pinned toolchain, '+arm+' generated manifest/CWD and clears seven compiler selectors/flags/wrappers')
        absolute_arms=Path(env['ARMS'])
        env['ARMS']=os.path.relpath(absolute_arms,ROOT)
        for arm in ['baseline','scored']:
            env['ARM']=arm
            result=subprocess.run(['sh','-c',build],cwd=ROOT,env=env,capture_output=True,text=True)
            assert result.returncode==0,result.stderr
            captured=json.loads(result.stdout);manifest=absolute_arms/arm/'Cargo.toml'
            args=captured['__argv__'];position=args.index('--manifest-path')
            assert args[position+1]==str(manifest) and captured['__cwd__']==str(manifest.parent), 'relative ARMS must canonicalize before child cd'
            print('PASS: inert build canonicalizes relative ARMS for '+arm+' manifest/CWD')
        import re
        other_os=re.sub(r'^os: .*$', 'os: synthetic alternate distribution [64-bit]',toolchain['cargo'],flags=re.M)
        alternate=dict(env,STUB_VERSION_cargo=other_os)
        result=subprocess.run(['sh','-c',build],cwd=ROOT,env=alternate,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['__cwd__']==str(absolute_arms/env['ARM']), 'alternate distro must retain generated working directory'
        print('PASS: Cargo stable identity accepts alternate distribution verbose metadata')
        for field in ['release','commit-hash','host']:
            changed=re.sub('^'+field+r': .*$',field+': wrong stable identity',toolchain['cargo'],flags=re.M)
            wrong=dict(env,STUB_VERSION_cargo=changed)
            result=subprocess.run(['sh','-c',build],cwd=ROOT,env=wrong,capture_output=True,text=True)
            assert result.returncode!=0 and not result.stdout, 'wrong Cargo '+field+' reached mocked build'
            print('PASS: incorrect Cargo '+field+' rejects before mocked build')
        for name in ['rustc','cargo']:
            wrong=dict(env,**{'STUB_VERSION_'+name:'wrong recorded version'})
            result=subprocess.run(['sh','-c',build],cwd=ROOT,env=wrong,capture_output=True,text=True)
            assert result.returncode!=0 and not result.stdout, 'wrong '+name+' version reached build'
            print('PASS: incorrect '+name+' verbose version rejects before mocked build')
    launch=next(block for block in blocks if '"$BINARY" scratch_scored_graph_finite_fixture' in block)
    code=shlex.split(launch[launch.index('python3 -c '):].replace('\\\n',' '))[2]
    env=os.environ.copy();env.update(PAMIN_EVAL_HOME='/retained-workspace',PAMIN_PROFILE='accuracy',PAMIN_DEVICE='cpu',PAMIN_PREPARED='off',PAMIN_FUSED_ATTENTION='off',PAMIN_INFERENCE_THREADS='999',PAMIN_FUTURE_KNOB='unexpected')
    for threads in [{},{'RAYON_NUM_THREADS':'99','OMP_NUM_THREADS':'97'},{'RAYON_NUM_THREADS':'','OMP_NUM_THREADS':''}]:
        environment={key:value for key,value in env.items() if key not in {'RAYON_NUM_THREADS','OMP_NUM_THREADS'}};environment.update(threads)
        result=subprocess.run([sys.executable,'-c',code,sys.executable,'-c','import json,os;print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("PAMIN_") or k in {"RAYON_NUM_THREADS","OMP_NUM_THREADS"}}))'],env=environment,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)=={'PAMIN_EVAL_HOME':'/retained-workspace','PAMIN_PROFILE':'accuracy','PAMIN_DEVICE':'cpu','RAYON_NUM_THREADS':'4','OMP_NUM_THREADS':'4'}, 'launch inherited product knobs or conflicting thread counts'
        print('PASS: inert launch shim clears product knobs and assigns both recorded4 thread counts '+repr(threads))
    for key in ['RAYON_NUM_THREADS','OMP_NUM_THREADS']:
        run_case('runtime thread recipe '+key,lambda root,key=key:(root/'README.md').write_text((root/'README.md').read_text().replace(key+'="4"',key+'="9"')),expected_error='pinned native-library runtime loader environment missing or changed')



if __name__ == '__main__':
    from test_future_accounting import resource_gate
    print('RESOURCE_ADMISSION '+json.dumps(resource_gate(),sort_keys=True))
    success = subprocess.run([sys.executable, str(ROOT / 'verify.py')], capture_output=True, text=True)
    if success.returncode:
        raise SystemExit(success.stderr)
    if '--round8-only' in sys.argv:
        round8_cases()
        raise SystemExit(0)
    round8_cases()
    if '--round7-native-env-only' in sys.argv:
        round7_native_environment_cases()
        raise SystemExit(0)
    if '--round7-weak-only' in sys.argv:
        round7_weak_window_cases()
        raise SystemExit(0)
    round7_graph_only_cases()
    if '--round7-graph-only' in sys.argv:
        raise SystemExit(0)
    round7_weak_window_cases()
    round7_native_environment_cases()
    run_case('non-patch traversal replacement with refreshed hashes',lambda r:(r/'source/experimental-traversal.patch').write_text('not a patch\n'),expected_error='recorded traversal patch differs')
    run_case('changed traversal patch with refreshed hashes',lambda r:(r/'source/experimental-traversal.patch').write_text((r/'source/experimental-traversal.patch').read_text()+'\n# changed\n'),expected_error='recorded traversal patch differs')
    for arm in ['baseline','scored']:
        wall=json.loads((ROOT/(arm+'.usage.json')).read_text())['wall_seconds']
        for duration in ['9999','9'*400,format(wall+.01,'.8f')]:
            def changed_duration(root,arm=arm,duration=duration):
                import re
                log=root/(arm+'.log');text,count=re.subn(r'finished in [0-9]+(?:\.[0-9]+)?s','finished in '+duration+'s',log.read_text())
                assert count==1;log.write_text(text)
            run_case('log duration beyond process wall '+arm+'/'+duration[:20],changed_duration,expected_error='fixture test duration exceeds whole-process wall time')
    for arm in ['baseline','scored']:
        for label,extra,error in [
            ('extra failed test','test additional_case ... FAILED\n','complete fixture test markers differ'),
            ('duplicate success test','test scratch_scored_graph_finite_fixture ... ok\n','complete fixture test markers differ'),
            ('extra failed summary','test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.00s\n','complete fixture result summary differs'),
            ('extra success summary','test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 1.00s\n','complete fixture result summary differs'),
            ('failure section','failures:\n    extra_case\n','fixture result must be the final log line'),
            ('panic marker','thread fixture panicked at error\n','fixture result must be the final log line')]:
            run_case(label+' '+arm,lambda r,arm=arm,extra=extra:(r/(arm+'.log')).write_text((r/(arm+'.log')).read_text()+extra),expected_error=error)
        def failed_before_summary(r,arm=arm):
            p=r/(arm+'.log');s=p.read_text();p.write_text(s.replace('test result:', 'failures:\n    bad_case\n\ntest result:'))
        run_case('failure section before final success '+arm,failed_before_summary,expected_error='fixture log contains a failure marker')
        run_case('wrong filtered count '+arm,lambda r,arm=arm:(r/(arm+'.log')).write_text((r/(arm+'.log')).read_text().replace('1 filtered out','0 filtered out')),expected_error='complete fixture result summary differs')
    for arm in ['baseline','scored']:
        for setup in ['existing project reused; no writes or optimization','native write/drain; no OptimizeIndex','',None,True,{},['native write/drain, explicit OptimizeIndex queue/drain; runtime defaults preserved']]:
            run_case('contradictory setup '+arm+' '+repr(setup),lambda r,arm=arm,setup=setup:mutate_json(r/(arm+'.jsonl'),lambda row:row.update(setup=setup)),expected_error='recorded native fixture setup differs')
        run_case('missing setup '+arm,lambda r,arm=arm:mutate_json(r/(arm+'.jsonl'),lambda row:row.pop('setup')),expected_error='recorded native fixture setup differs')
    retained=json.loads((ROOT/'provenance.json').read_text())
    main_ref='315c10242ddf7a1cec3bccbf550a942320e09557'
    for ref in ['0'*40, retained['source_base'], 'nonexistent', '', None, True, [main_ref]]:
        run_case('prepared unrun main reference '+repr(ref),lambda r,ref=ref:mutate_json(r/'provenance.json',lambda p:p.update(compared_main_ref_not_run=ref)),expected_error='prepared unrun main reference differs')
    run_case('missing prepared unrun main reference',lambda r:mutate_json(r/'provenance.json',lambda p:p.pop('compared_main_ref_not_run')),expected_error='prepared unrun main reference differs')
    def matched_unrun_reference(r):
        mutate_json(r/'provenance.json',lambda p:p.update(compared_main_ref_not_run='0'*40))
        p=r/'README.md';p.write_text(p.read_text().replace(main_ref,'0'*40))
    run_case('matched fabricated unrun main reference and README',matched_unrun_reference,expected_error='prepared unrun main reference differs')
    for label,change,error in [
        ('different SHA',lambda text:text.replace(main_ref,'0'*40),'README prepared main reference differs'),
        ('missing reference',lambda text:text.replace('The recorded local main ref was `'+main_ref+'`;','No main reference retained.'),'README prepared main reference differs'),
        ('contradictory second reference',lambda text:text+'\nThe recorded local main ref was `'+('0'*40)+'`;\n','README prepared main reference differs'),
        ('executed arm claim',lambda text:text.replace('that separate main arm was prepared but was not executed here.','that separate main arm was prepared and executed here.'),'README prepared main execution limitation differs'),
        ('missing non-execution limitation',lambda text:text.replace('that separate main arm was prepared but was not executed here.','that separate main arm was prepared.'),'README prepared main execution limitation differs')]:
        def mutate_readme(r,change=change):
            p=r/'README.md';p.write_text(change(p.read_text()))
        run_case('README prepared main '+label,mutate_readme,expected_error=error)
    for scope in ['mapped inode identity captured and verified', 'mapped paths and pinned file digests', '', None, True, 0, ['mapped paths and pinned file digests; no inode identity captured']]:
        run_case('mapped-library identity scope '+repr(scope),lambda r,scope=scope:mutate_json(r/'provenance.json',lambda p:p.update(mapped_library_identity_scope=scope)),expected_error='mapped-library identity limitation differs')
    run_case('missing mapped-library identity scope',lambda r:mutate_json(r/'provenance.json',lambda p:p.pop('mapped_library_identity_scope')),expected_error='mapped-library identity limitation differs')
    for arm in ['baseline','scored']:
        row=json.loads((ROOT/(arm+'.jsonl')).read_text())
        weak_n=int(row['weak'][10:])
        weak_content=f'orbital navigation calibration beacon archival report category {weak_n % 17} revision {weak_n}'
        strong_content='orbital navigation calibration beacon'
        for index in range(3):
            opposite=strong_content if arm=='baseline' and index==0 else weak_content
            for value in ['unrelated origin content',opposite,'',None,123,{'content':row['targets'][index]['seed']}]:
                run_case('target seed '+arm+' '+str(index)+' '+repr(value),lambda r,arm=arm,index=index,value=value:mutate_json(r/(arm+'.jsonl'),lambda row:row['targets'][index].update(seed=value)),expected_error='target seed content differs from expected origin: '+arm)
            run_case('missing target seed '+arm+' '+str(index),lambda r,arm=arm,index=index:mutate_json(r/(arm+'.jsonl'),lambda row:row['targets'][index].pop('seed')),expected_error='target seed content differs from expected origin: '+arm)
    for scope in ['Contemporaneous build-time SQL source attestation.', 'Historical rebuild verified the SQL sources.', '', None]:
        run_case('retrospective SQL scope '+str(scope),lambda r,scope=scope:mutate_json(r/'retrospective-sql-audit.json',lambda a:a.update(scope=scope)),expected_error='retrospective SQL audit scope differs')
    run_case('missing retrospective SQL scope',lambda r:mutate_json(r/'retrospective-sql-audit.json',lambda a:a.pop('scope')),expected_error='retrospective SQL audit scope differs')
    audit=json.loads((ROOT/'retrospective-sql-audit.json').read_text())
    for arm in ['baseline','scored']:
        length=audit['arms'][arm]['binary_bytes']
        for value in [length+1, length+1024*1024, length-1, 0, -1, float(length), True, str(length), None]:
            run_case('binary length '+arm+' '+repr(value),lambda r,arm=arm,value=value:mutate_json(r/'retrospective-sql-audit.json',lambda a:a['arms'][arm].update(binary_bytes=value)),expected_error='retained binary byte length differs: '+arm)
        run_case('missing binary length '+arm,lambda r,arm=arm:mutate_json(r/'retrospective-sql-audit.json',lambda a:a['arms'][arm].pop('binary_bytes')),expected_error='retained binary byte length differs: '+arm)
    run_case('product overclaim in provenance scope',lambda r:mutate_json(r/'provenance.json',lambda p:p.update(scope='native search_fused component; validated product accuracy and speed improvement')),expected_error='component provenance scope differs')
    for key,value in [('scope','Historical platform capture from the measured runs.'),('comparison','Current CPU/kernel/quota/affinity exactly match the original runs.')]:
        run_case('historical hardware overclaim '+key,lambda r,key=key,value=value:mutate_json(r/'platform-observation.json',lambda p:p.update({key:value})),expected_error='current platform scope differs' if key=='scope' else 'historical hardware limitation differs')
    for arm in ['baseline','scored']:
        recorded=next(a for a in retained['arms'] if a['arm']==arm)['resources']
        for key in recorded:
            value=False if key=='reclaim_pressure' else 'guaranteed reclaim; exclusive machine' if key=='scope' else -1 if key=='estimated_headroom_bytes' else 1
            run_case('resource-pressure '+arm+' '+key,lambda r,arm=arm,key=key,value=value:mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['resources'].update({key:value})),expected_error='retained resource-pressure record differs')
        for key in ['reclaim_pressure','estimated_headroom_bytes','oom_before']:
            value=1 if key=='reclaim_pressure' else False if key=='oom_before' else float(recorded[key])
            run_case('resource-pressure strict type '+arm+' '+key,lambda r,arm=arm,key=key,value=value:mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['resources'].update({key:value})),expected_error='resource-pressure record types differ')
        for index in range(3+(arm=='scored')):
            for evidence_kind in ['graph','path']:
                for mutation in ['duplicate','missing']:
                    def contradictory_why(r,arm=arm,index=index,evidence_kind=evidence_kind,mutation=mutation):
                        def change(row):
                            why=row['targets'][index]['why'] if index<3 else row['early_stop']['why']
                            matches=[w for w in why if w.get('kind')=='path'] if evidence_kind=='path' else [w for w in why if w.get('kind')=='channel' and w.get('channel')=='graph']
                            assert len(matches)==1
                            if mutation=='missing':why.remove(matches[0])
                            else:
                                duplicate=copy.deepcopy(matches[0])
                                if evidence_kind=='graph':duplicate['score']=999
                                else:duplicate['asserted_to']='contradictory-target'
                                why.append(duplicate)
                        mutate_json(r/(arm+'.jsonl'),change)
                    run_case(mutation+' '+evidence_kind+' Why '+arm+' target '+str(index),contradictory_why,expected_error='exactly one '+('graph-channel' if evidence_kind=='graph' else 'path')+' record')
    for value in ['2099-01-01',True,None,20260930]:
        run_case('experiment date '+repr(value),lambda root,value=value:mutate_json(root/'provenance.json',lambda p:p.update(date=value)),expected_error='recorded experiment date differs')
    run_case('missing experiment date',lambda root:mutate_json(root/'provenance.json',lambda p:p.pop('date')),expected_error='recorded experiment date differs')
    for arm in ['baseline','scored']:
        for value in ['removed FAILED test and rewrote summary','',None,True]:
            run_case('original log scope '+arm+'/'+repr(value),lambda root,arm=arm,value=value:mutate_json(root/'provenance.json',lambda p:p['redactions'][arm+'.log'].update(scope=value)),expected_error='original log formatting-only redaction scope differs')
        run_case('missing original log scope '+arm,lambda root,arm=arm:mutate_json(root/'provenance.json',lambda p:p['redactions'][arm+'.log'].pop('scope')),expected_error='original log formatting-only redaction scope differs')
    for value in ['future builds need not capture SQL; original build fully attested','',None,True]:
        run_case('future compiler capture '+repr(value),lambda root,value=value:mutate_json(root/'provenance.json',lambda p:p['source_inventory_scope'].update(future_requirement=value)),expected_error='future source-capture requirement differs')
    run_case('missing future compiler capture',lambda root:mutate_json(root/'provenance.json',lambda p:p['source_inventory_scope'].pop('future_requirement')),expected_error='future source-capture requirement differs')
    for metric in ['General retrieval accuracy','Product search latency p50/p95','Graph-attributed memory','Graph-attributed disk']:
        def overclaim(root,metric=metric):
            path=root/'README.md'
            path.write_text(path.read_text().replace('| '+metric+' | N/A | N/A | N/A | N/A |','| '+metric+' | 100 ms | 50 ms | -50 ms | -50% |'))
        run_case('unmeasured table '+metric,overclaim,expected_error='unmeasured resource table must remain N/A')
    for script in ['verify.py','test_verify.py']:
        # Split detection strings keep this published test source free of the
        # private markers it deliberately injects into temporary archives.
        for marker in ['/ho'+'me/alice/private-project','postgres'+'ql://user:password@host/db','Bearer '+'credential123']:
            def leaked_script(root,script=script,marker=marker):
                path=root/script;path.write_text(path.read_text()+'\n# '+marker+'\n')
            run_case('published script private marker '+script+'/'+marker.split(':')[0],leaked_script,expected_error='private path/credential/session marker: '+script)
    run_case('modified privacy pattern',lambda root:(root/'verify.py').write_text((root/'verify.py').read_text().replace('Bearer\\s+','Bearer\\s*')),expected_error='approved private-marker pattern differs')
    for source,label in [('retrospective-sql-audit.json','SQL'),('platform-observation.json','platform'),('provenance.json','multihop')]:
        for value in ['2026-09-30T:+00:00','2026-09-30T99:99:99+00:00','2026-09-30T23:02:04.562272+00:00',None,True]:
            def timestamp(root,source=source,value=value):
                def change(item):
                    target=item['retrospective_multihop_source'] if source=='provenance.json' else item
                    target['captured_at_utc']=value
                mutate_json(root/source,change)
            run_case('capture timestamp '+source+'/'+repr(value),timestamp,expected_error='retained '+label+' capture timestamp differs')
    run_case('missing generated arm manifest flag',lambda root:(root/'README.md').write_text((root/'README.md').read_text().replace('  --manifest-path "$ARMS/$ARM/Cargo.toml" \\\n','')),expected_error='complete approved recipe differs')
    for replacement in ['was compiled and passed.','was compiled and run successfully.']:
        run_case('invented frontier stress execution '+replacement,lambda root,replacement=replacement:(root/'README.md').write_text((root/'README.md').read_text().replace('was compiled but not run.',replacement)),expected_error='README compiled-but-unrun frontier-stress limitation differs')
    run_case('invented README historical loaded zvec identity',lambda root:(root/'README.md').write_text((root/'README.md').read_text().replace('mapping was not captured: historical loaded-Zvec identity is **N/A**.','mapping was captured and verified: historical loaded-Zvec identity is **verified**.')),expected_error='README historical loaded zvec N/A limitation differs')
    run_case('invented historical loaded zvec identity',lambda root:mutate_json(root/'provenance.json',lambda p:p.update(historical_loaded_zvec_identity='loaded zvec verified')),expected_error='historical loaded zvec limitation differs')
    run_case('historical runner source overclaim',lambda root:mutate_json(root/'provenance.json',lambda p:p['future_reproduction'].update(scope='original historical runner bytes')),expected_error='prospective runner scope differs')
    run_case('future accounting source mutation',lambda root:(root/'source/future-accounting.py.in').write_text((root/'source/future-accounting.py.in').read_text()+'\n# changed\n'),expected_error='future accounting source bytes differ')
    for arm in ['baseline','scored']:
        run_case('finite raw score replacement '+arm,lambda root,arm=arm:mutate_json(root/(arm+'.jsonl'),lambda row:row['non_graph'][0]['ranks'][0].update(score=999.0)),expected_error='retained non-graph raw inventory differs')
        for index in range(4 if arm=='scored' else 3):
            for value in [True,1.0,'1']:
                def hop_type(root,arm=arm,index=index,value=value):
                    def change(row):
                        why=row['targets'][index]['why'] if index<3 else row['early_stop']['why']
                        next(w for w in why if w['kind']=='path')['hops']=value
                    mutate_json(root/(arm+'.jsonl'),change)
                run_case('non-native hop type '+arm+'/'+str(index)+'/'+repr(value),hop_type,expected_error='path hop count must be a positive integer')
        for suffix in ['jsonl','trace.jsonl','usage.json']:
            for value in ['changed scores/timings/providers','',True,None]:
                run_case('JSON scope '+arm+'.'+suffix+'/'+repr(value),lambda root,arm=arm,suffix=suffix,value=value:mutate_json(root/'provenance.json',lambda p:p['redactions'][arm+'.'+suffix].update(scope=value)),expected_error='JSON redaction scope differs')
            run_case('missing JSON scope '+arm+'.'+suffix,lambda root,arm=arm,suffix=suffix:mutate_json(root/'provenance.json',lambda p:p['redactions'][arm+'.'+suffix].pop('scope')),expected_error='JSON redaction scope differs')
    round12_interval_channel_cases()
    round10_native_type_cases()
    complete_non_graph_cases()
    complete_graph_record_cases()
    for name in retained['redactions']:
        run_case('original redaction digest '+name, lambda r,name=name: mutate_json(r/'provenance.json',lambda p:p['redactions'][name].update(original_sha256='0'*64)))
    for arm in ['baseline','scored']:
        run_case('raw interference declaration '+arm,lambda r,arm=arm:mutate_json(r/(arm+'.jsonl'),lambda row:row.update(shared_machine='exclusive machine; no interference')))
        for key in next(a for a in retained['arms'] if a['arm']==arm)['launch_binding']['effective_product_settings']:
            run_case('launch projection '+arm+' '+key,lambda r,arm=arm,key=key:mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['launch_binding']['effective_product_settings'].update({key:'changed'})))
        run_case('original launch digest '+arm,lambda r,arm=arm:mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['launch_binding'].update(original_launch_sha256='0'*64)))
    def leaves(value,prefix=()):
        for key,item in value.items():
            if isinstance(item,dict):yield from leaves(item,prefix+(key,))
            else:yield prefix+(key,)
    for keys in leaves(retained['toolchain']):
        def toolchain_mutation(r,keys=keys):
            def change(p):
                item=p['toolchain']
                for key in keys[:-1]:item=item[key]
                item[keys[-1]]='bogus'
            mutate_json(r/'provenance.json',change)
        run_case('toolchain '+'/'.join(keys),toolchain_mutation)
    for field in ['product_latency_p50_ms','corpus_accuracy','fabricated']:
        run_case('extra comparison field '+field,lambda root,field=field:mutate_json(root/'comparison.json',lambda value:value.update({field:{'baseline':100,'scored':50}})),expected_error='comparison top-level schema differs')
    for key,value in [('early_stop_recovered',1),('cross_arm_non_graph_identical',0)]:
        run_case('nonboolean comparison status '+key,lambda root,key=key,value=value:mutate_json(root/'comparison.json',lambda row:row.update({key:value})),expected_error='comparison status fields must be actual booleans')
    run_case('comparison scope',lambda r:mutate_json(r/'comparison.json',lambda c:c.update(scope='validated product accuracy and speed improvement')))
    for arm in ['baseline','scored']:
        for reached in [0,1,0.0,1.0,None,'false','true',[],{}]:
            run_case('nonboolean early-stop '+arm+'/'+repr(reached),lambda r,arm=arm,reached=reached:mutate_json(r/(arm+'.jsonl'),lambda row:row['early_stop'].update(reached=reached)),expected_error='early-stop reached must be an actual boolean')
    for arm,count in [('baseline',3),('scored',4)]:
        for index in range(count):
            for field in ['confidence','rank','fabricated']:
                def extra_path(root,arm=arm,index=index,field=field):
                    def changed(row):
                        why=row['targets'][index]['why'] if index<3 else row['early_stop']['why']
                        next(record for record in why if record['kind']=='path')[field]=999
                    mutate_json(root/(arm+'.jsonl'),changed)
                run_case('extra Why::Path field '+arm+'/'+str(index)+'/'+field,extra_path,expected_error='retained Why::Path field set differs')
    for endpoint in ['asserted_from','asserted_to']:
        run_case('early-stop '+endpoint,lambda r,endpoint=endpoint:mutate_json(r/'scored.jsonl',lambda row:next(w for w in row['early_stop']['why'] if w['kind']=='path').update({endpoint:'unrelatedtopic'})))
    def absent_evidence(r):
        scored=json.loads((r/'scored.jsonl').read_text())
        mutate_json(r/'baseline.jsonl',lambda row:row['early_stop'].update(why=scored['early_stop']['why']))
    run_case('evidence on absent early-stop result',absent_evidence)
    for arm in ['baseline','scored']:
        def duplicate_target(r,arm=arm):
            def change(row):
                duplicate=copy.deepcopy(row['targets'][0]);next(w for w in duplicate['why'] if w['kind']=='path')['from']='fabricatedseed'
                row['targets'].insert(0,duplicate)
            mutate_json(r/(arm+'.jsonl'),change)
        run_case('duplicate target '+arm,duplicate_target)
        run_case('missing target '+arm,lambda r,arm=arm:mutate_json(r/(arm+'.jsonl'),lambda row:row['targets'].pop()))
        for metric in ['wall_seconds','user_seconds','system_seconds','maximum_process_rss_kib']:
            for invalid in [-1,float('nan'),float('inf'),True]:
                def invalid_usage(r,arm=arm,metric=metric,invalid=invalid):
                    mutate_json(r/(arm+'.usage.json'),lambda u:u.update({metric:invalid}))
                    mutate_json(r/'provenance.json',lambda p:next(a for a in p['arms'] if a['arm']==arm)['process_usage'].update({metric:invalid}))
                run_case('invalid process usage '+arm+' '+metric+' '+str(invalid),invalid_usage)
    for arm in ['baseline','scored']:
        for metric in ['elapsed_ms','early_elapsed_ms','process_lifetime_high_water_kib','graph_cpu_user','graph_cpu_system']:
            for invalid in [-1,float('nan'),float('inf'),True]:
                def invalid_raw_usage(r,arm=arm,metric=metric,invalid=invalid):
                    def change(row):
                        if metric=='early_elapsed_ms':row['early_stop']['elapsed_ms']=invalid
                        elif metric.startswith('graph_cpu_'):row['graph_process_cpu_user_system_seconds'][0 if metric=='graph_cpu_user' else 1]=invalid
                        else:row[metric]=invalid
                    mutate_json(r/(arm+'.jsonl'),change)
                run_case('invalid raw usage '+arm+' '+metric+' '+str(invalid),invalid_raw_usage)
    run_case('missing sixth compiler artifact',lambda r:mutate_json(r/'provenance.json',lambda p:p['arms'][0]['fresh_compiler_artifacts'].pop()))
    run_case('duplicate fresh compiler artifact',lambda r:mutate_json(r/'provenance.json',lambda p:p['arms'][0]['fresh_compiler_artifacts'].append(p['arms'][0]['fresh_compiler_artifacts'][-1])))
    run_case('wrong multihop artifact source',lambda r:mutate_json(r/'provenance.json',lambda p:p['arms'][0]['fresh_compiler_artifacts'][-1].update(source='wrong.rs')))
    run_case('multihop source bytes',lambda r:(r/'source/multihop.rs.in').write_text((r/'source/multihop.rs.in').read_text()+'\n// changed\n'))
    for key,value in [('scope','historically attested build input'),('sha256','0'*64),('bytes',0),('source','wrong.rs'),('published_file','source/fixture.rs.in')]:
        run_case('retrospective multihop binding '+key,lambda r,key=key,value=value:mutate_json(r/'provenance.json',lambda p:p['retrospective_multihop_source'].update({key:value})))
    run_case('multihop arm source digest',lambda r:mutate_json(r/'provenance.json',lambda p:p['retrospective_multihop_source']['arm_source_sha256'].update(scored='0'*64)))
    run_case('reproduction workspace omitted',lambda r:(r/'README.md').write_text((r/'README.md').read_text().replace('PAMIN_EVAL_HOME="$WORKSPACE" ','')))
    run_case('reproduction multihop compile target omitted',lambda r:(r/'README.md').write_text((r/'README.md').read_text().replace(' --test scratch_scored_multihop','')))
    # A tiny source-only preparation proves both exact helpers survive scratch
    # cleanup and are hashed. It never invokes Cargo or an executable helper.
    with tempfile.TemporaryDirectory(prefix='graph-prepare-positive-') as temp:
        source=Path(temp)/'input';out=Path(temp)/'prepared'
        engine=source/'crates/pamin-engine/src/engine.rs';engine.parent.mkdir(parents=True)
        engine.write_text('const USE_SCORED_GRAPH_RECALL: bool = false;\nseed_relevance(&named, lists, options.k)\nk: fusion.k()\n')
        (source/'crates/pamin-engine/Cargo.toml').write_text('[dev-dependencies]\n')
        (source/'Cargo.lock').write_text('[[package]]\nname = "pamin-engine"\ndependencies = [\n "pamin-core",\n]\n[[package]]\nname = "pamin-core"\n')
        tests=source/'crates/pamin-engine/tests';tests.mkdir();(tests/'scratch_discard.rs').write_text('discard')
        result=subprocess.run([sys.executable,str(ROOT/'source/prepare.py'),'--source',str(source),'--out',str(out)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        prepared=json.loads((out/'prepared-source-hashes.json').read_text())
        for arm in ['baseline','scored']:
            assert not (out/arm/'crates/pamin-engine/tests/scratch_discard.rs').exists()
            for target,artifact in [('scratch_scored_fixture','fixture.rs.in'),('scratch_scored_multihop','multihop.rs.in')]:
                relative=arm+'/crates/pamin-engine/tests/'+target+'.rs'
                expected=hashlib.sha256((ROOT/'source'/artifact).read_bytes()).hexdigest()
                assert prepared[relative]==hashlib.sha256((out/relative).read_bytes()).hexdigest()==expected
        print('PASS: preparation retains and hashes both exact scratch target sources')
    with tempfile.TemporaryDirectory(prefix='graph-prepare-source-negative-') as temp:
        root=Path(temp)/'archive';shutil.copytree(ROOT,root)
        (root/'source/multihop.rs.in').write_text('changed source')
        out=Path(temp)/'never-created'
        result=subprocess.run([sys.executable,str(root/'source/prepare.py'),'--source',str(Path(temp)/'absent-input'),'--out',str(out)],capture_output=True,text=True)
        assert result.returncode!=0 and not out.exists() and 'preserved multihop source differs' in result.stderr
        print('PASS: preparation rejects changed multihop source before copying')
    run_case('Python -O', flags=('-O',))
    run_case('Python -OO', flags=('-OO',))
    reject_optimized_prepare('-O', flags=('-O',))
    reject_optimized_prepare('-OO', flags=('-OO',))
    reject_optimized_prepare('PYTHONOPTIMIZE=1', optimize_env='1')
    reject_optimized_prepare('PYTHONOPTIMIZE=2', optimize_env='2')
    run_case('raw query mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row.update(query='different query')))
    run_case('binary SHA mutation with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0].update(binary_sha256='0'*64)))
    run_case('retrospective binary SHA mutation with refreshed hashes', lambda r: mutate_json(r/'retrospective-sql-audit.json', lambda a: a['arms']['baseline'].update(binary_sha256='0'*64)))
    run_case('retrospective SQL digest mutation with refreshed hashes', lambda r: mutate_json(r/'retrospective-sql-audit.json', lambda a: next(iter(a['migration_sources'].values())).update(sha256='0'*64)))
    run_case('retrospective SQL payload offset mutation with refreshed hashes', lambda r: mutate_json(r/'retrospective-sql-audit.json', lambda a: next(iter(a['arms']['baseline']['migration_payloads'].values())).update(first_binary_offset=-1)))
    run_case('current CPU model mutation with refreshed hashes', lambda r: mutate_json(r/'platform-observation.json', lambda a: a['current'].update(cpu_model='different CPU')))
    run_case('historical SQL inventory overclaim with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['source_inventory_scope'].update(original_status='complete')))
    run_case('current hardware substituted as historical with refreshed hashes', lambda r: mutate_json(r/'platform-observation.json', lambda a: a['historical'].update(cpu_quota=a['current']['cpu_quota'])))
    run_case('published migration bytes mutation with refreshed hashes', lambda r: (r/'source/migrations/V1__initial.sql').write_text('SELECT 1;\n'))
    run_case('measured fixture query mutation with refreshed hashes', lambda r: (r/'source/fixture.rs.in').write_text((r/'source/fixture.rs.in').read_text().replace('let query = \"quartzanchor orbital navigation calibration beacon\";', 'let query = \"different query\";')))
    run_case('raw provider mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[2]['fields'].update(assigned_nodes={'CUDAExecutionProvider':1023}), True))
    run_case('raw runtime mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[0]['fields'].update(runtime_info='ORT Build Info: changed'), True))
    run_case('raw mapped-path mutation with refreshed hashes', lambda r: mutate_json(r/'baseline.trace.jsonl', lambda rows: rows[0].update(runtime_maps=['${ORT_LIB}/other.so']), True))
    run_case('provenance provider mutation with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: next(iter(p['arms'][0]['inference']['selected_graphs'].values())).update(assigned_nodes={'CPUExecutionProvider':1022})))
    run_case('comparison score mutation with refreshed hashes', lambda r: mutate_json(r/'comparison.json', lambda c: c['invariants'][0].update(scored=.7)))
    run_case('test-log marker mutation with refreshed hashes', lambda r: (r/'baseline.log').write_text((r/'baseline.log').read_text().replace('fixture ... ok','fixture ... FAILED')))
    run_case('displayed table mutation with refreshed hashes', lambda r: (r/'README.md').write_text((r/'README.md').read_text().replace('+139.999996%', '+141.000000%')))
    run_case('missing redaction record with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['redactions'].pop('baseline.log')))
    run_case('missing asset pin with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0]['asset_hashes_before'].pop('${ZVEC_LIB}/libzvec_c_api.so')))
    run_case('malformed compiled source map with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0].update(source_hashes={'bogus':'not-a-digest'})))
    run_case('changed compiled source digest with refreshed hashes', lambda r: mutate_json(r/'provenance.json', lambda p: p['arms'][0]['source_hashes'].update({'Cargo.lock':'0'*64})))
    run_case('missing vector rank50 with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row.update(non_graph=[h for h in row['non_graph'] if not any(x['channel']=='vector' and x['rank']==50 for x in h['ranks'])])))
    run_case('duplicate channel rank with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: next(x for h in row['non_graph'] for x in h['ranks'] if x['channel']=='vector' and x['rank']==50).update(rank=49)))
    run_case('changed controlled edge endpoint with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row['known_edges'][0].update({'from':'arbitrarytopic'})))
    run_case('changed controlled edge confidence with refreshed hashes', lambda r: mutate_json(r/'baseline.jsonl', lambda row: row['known_edges'][0].update(confidence=.9)))
    run_case('extra file with refreshed known hashes', lambda r: (r/'extra.txt').write_text('extra\n'))
    run_case('missing required file', lambda r: (r/'baseline.usage.json').unlink(), refresh_hashes=False)
