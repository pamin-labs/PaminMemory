#!/usr/bin/env python3
"""Prospective builds only; preserve archived build.py.in and historical outputs."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
if sys.flags.optimize:raise SystemExit('Build guards require assertions; remove -O/PYTHONOPTIMIZE')

TARGET='crates/pamin-engine/tests/scratch_cache_product_limits.rs'
PRODUCTS={'pamin-core','pamin-store','pamin-index','pamin-engine'}
TOOLCHAIN=Path('<WORKSPACE>/.rustup/toolchains/1.98.1-x86_64-unknown-linux-gnu/bin')
CONTROLLED_CARGO={'CARGO_HOME','CARGO_TARGET_DIR','CARGO_BUILD_JOBS'}
def reject_overrides(parent):
    unexpected=sorted(key for key in parent if
        (key.startswith('CARGO_') and key not in CONTROLLED_CARGO) or
        key.startswith(('RUSTC','RUSTDOC')) or key in {'RUSTFLAGS','RUSTUP_TOOLCHAIN'})
    assert not unexpected, 'refuse inherited compiler/Cargo overrides: '+', '.join(unexpected)
def build_environment(parent):
    reject_overrides(parent)
    tools={name:TOOLCHAIN.joinpath(name).resolve(strict=True) for name in ['cargo','rustc','rustdoc']}
    env=dict(parent,RUSTUP_HOME='<WORKSPACE>/.rustup',CARGO_HOME='<WORKSPACE>/.cargo',
             CARGO_TARGET_DIR='<WORKSPACE>/PaminMemory/target',CARGO_BUILD_JOBS='4',
             ORT_LIB_LOCATION='<WORKSPACE>/.onnxruntime/onnxruntime-linux-x64-1.28.0/lib',ORT_PREFER_DYNAMIC_LINK='1',
             RUSTC=str(tools['rustc']),RUSTDOC=str(tools['rustdoc']),
             RUSTC_WRAPPER='',RUSTC_WORKSPACE_WRAPPER='',
             CARGO_ENCODED_RUSTFLAGS='',CARGO_ENCODED_RUSTDOCFLAGS='')
    env['PATH']=str(TOOLCHAIN)+os.pathsep+env.get('PATH','')
    return env,tools
def tool_identities(tools):
    return {name:{'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)} for name,path in tools.items()}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def git(checkout,*args):
    return subprocess.check_output(['git',*args],cwd=checkout)
def inputs(checkout,commit,allowed):
    assert git(checkout,'rev-parse','HEAD').decode().strip()==commit
    assert not git(checkout,'status','--porcelain','--untracked-files=no'), 'tracked changes refused'
    assert not git(checkout,'diff','--cached','--name-only',commit), 'index differs from expected commit'
    tree=[entry.split('\t',1) for entry in git(checkout,'ls-tree','-r','-z',commit).decode().split('\0')[:-1]]
    tracked={name:[header.split()[0],header.split()[2]] for header,name in tree}
    # Read immutable HEAD blobs in one batch. Status can hide worktree changes
    # under assume-unchanged/skip-worktree; index contents are not the authority.
    blobs=subprocess.check_output(['git','cat-file','--batch'],cwd=checkout,
        input=('\n'.join(entry[1] for entry in tracked.values() if entry[0]!='160000')+'\n').encode())
    offset=0;expected={}
    for name,(mode,oid) in tracked.items():
        if mode=='160000':continue
        end=blobs.index(b'\n',offset);header=blobs[offset:end].split()
        assert header[:2]==[oid.encode(),b'blob']
        size=int(header[2]);content=blobs[end+1:end+1+size];offset=end+size+2
        expected[name]={'bytes':size,'sha256':hashlib.sha256(content).hexdigest()}
    assert TARGET not in tracked, 'scratch target must remain untracked'
    extra=set()
    for options in [[],['--ignored']]:
        extra.update(git(checkout,'ls-files','--others',*options,'--exclude-standard','-z').decode().split('\0')[:-1])
    assert extra<=set(allowed), 'unexpected untracked/ignored source or dependency'
    for name in extra:
        assert (checkout/name).is_file() and not (checkout/name).is_symlink(), 'scratch input must be a regular file'
        assert digest(checkout/name)==allowed[name], 'scratch content differs from explicit manifest'
    result={}
    for name in sorted(set(tracked)|extra):
        if name in tracked and tracked[name][0]=='160000':
            result[name]={'kind':'gitlink','commit':tracked[name][1]};continue
        path=checkout/name
        if name in tracked and tracked[name][0]=='120000':
            assert path.is_symlink(), 'tracked symlink replaced'
            target=os.fsencode(os.readlink(path))
            assert {'bytes':len(target),'sha256':hashlib.sha256(target).hexdigest()}==expected[name], 'tracked link target differs from HEAD git blob'
            result[name]={'kind':'symlink','bytes':len(target),'sha256':hashlib.sha256(target).hexdigest(),'target':os.fsdecode(target)}
            continue
        assert path.is_file() and not path.is_symlink(), 'non-file/symlink source requires separate explicit audit'
        result[name]={'bytes':path.stat().st_size,'sha256':digest(path)}
        if name in tracked:assert result[name]==expected[name], 'tracked regular file differs from HEAD git blob'
    assert 'Cargo.lock' in result and 'Cargo.toml' in result
    return result
def fresh_artifacts(records,checkout):
    selected=[x for x in records if x.get('reason')=='compiler-artifact' and x.get('target',{}).get('name','').replace('_','-') in PRODUCTS|{'scratch-cache-product-limits'}]
    assert {x['target']['name'].replace('_','-') for x in selected}==PRODUCTS|{'scratch-cache-product-limits'}
    for x in selected:
        assert x['fresh'] is False, 'product/helper reused'
        assert Path(x['manifest_path']).resolve().is_relative_to(checkout.resolve()), 'product compiled from another checkout'
    tests=[x for x in selected if x['target']['name']=='scratch_cache_product_limits' and x.get('executable')]
    assert len(tests)==1 and any(x.get('reason')=='build-finished' and x['success'] is True for x in records)
    return selected,Path(tests[0]['executable'])
