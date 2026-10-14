#!/usr/bin/env python3
"""Prepare prospective scratch arms with explicit disabled-outcome assertions.
This is not the historical preparation script or captured fixture source.
Input is a scratch checkout of the recorded base with experimental-traversal.patch
applied. No tracked input file is changed, and nothing is built or run here.
"""
import sys
if sys.flags.optimize:
 raise SystemExit("FAIL: Python optimization disables assertions; run without -O/-OO/PYTHONOPTIMIZE")
import argparse, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
source=a.source.resolve();out=a.out.resolve();assert not out.exists() and source not in out.parents
artifact=Path(__file__).parent
assert hashlib.sha256((artifact/'fixture-future.rs.in').read_bytes()).hexdigest() == 'ecf3d00607854c0f8773941ad9b813b35ec73a46add2cc37f64b9bbbe1a03373', 'prospective fixture source differs'
assert hashlib.sha256((artifact/'multihop.rs.in').read_bytes()).hexdigest() == '70d5127e265a475edc3ededc871dc91cd79a0bfb56a4e6feca4d1501fd206c00', 'preserved multihop source differs'
# Bind the input to the recorded base plus patch: every source file the
# preparation does not rewrite must equal the retained baseline arm's bytes.
provenance=json.loads((artifact.parent/'provenance.json').read_text())
baseline=next(arm['source_hashes'] for arm in provenance['arms'] if arm['arm']=='baseline')
rewritten={'Cargo.lock','crates/pamin-engine/Cargo.toml','crates/pamin-engine/tests/scratch_scored_fixture.rs','crates/pamin-engine/tests/graph_trace/mod.rs'}
for name,expected in baseline.items():
 if name in rewritten:continue
 path=source/name;assert path.is_file() and not path.is_symlink(),'recorded source input missing: '+name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'source differs from recorded base '+provenance['source_base']+' plus patch: '+name
# Cargo also compiles files the map does not list (a new module, a build
# script); every Rust source or manifest under the four compiled crates must be one
# the map records or one this preparation writes.
compiled={str(f.relative_to(source)) for crate in [source/'crates'/name for name in ['pamin-core','pamin-store','pamin-index','pamin-engine']] for f in crate.rglob('*') if f.is_file() and (f.suffix=='.rs' or f.name=='Cargo.toml') and 'tests' not in f.relative_to(crate).parts}
unrecorded=sorted(compiled-set(baseline))
assert not unrecorded,'unrecorded build input in source: '+', '.join(unrecorded)
# migrate.rs embeds every migration with include_str!, so the SQL files are
# build inputs too: require exactly the archived migration set, byte for byte.
audit=json.loads((artifact.parent/'retrospective-sql-audit.json').read_text())['migration_sources']
embedded={str(f.relative_to(source)) for crate in [source/'crates'/name for name in ['pamin-core','pamin-store','pamin-index','pamin-engine']] for f in crate.rglob('*.sql') if 'tests' not in f.relative_to(crate).parts}
assert embedded==set(audit),'embedded migration SQL set differs from archived copies: '+', '.join(sorted(embedded^set(audit)))
for name,entry in audit.items():
 archived=(artifact.parent/entry['published_file']).read_bytes();assert hashlib.sha256(archived).hexdigest()==entry['sha256'],'archived migration copy differs: '+name
 path=source/name;assert path.is_file() and not path.is_symlink() and path.read_bytes()==archived,'embedded migration SQL differs from archived copy: '+name
switch='const USE_SCORED_GRAPH_RECALL: bool = false;'
engine=(source/'crates/pamin-engine/src/engine.rs').read_text();assert engine.count(switch)==1
assert 'seed_relevance(&named, lists, options.k)' in engine and 'k: fusion.k()' in engine
for arm in ['baseline','scored']:
 target=out/arm
 shutil.copytree(source,target,ignore=shutil.ignore_patterns('.git','target','internal-docs','.agents','.claude','.codex','.aws','scratch_*.rs','__pycache__'))
 if arm=='scored':(target/'crates/pamin-engine/src/engine.rs').write_text(engine.replace(switch,switch.replace('false','true')))
 manifest=target/'crates/pamin-engine/Cargo.toml';manifest.write_text(manifest.read_text()+'\n# Scratch-only runtime evidence binding.\nort = { workspace = true }\n')
 lock=target/'Cargo.lock';text=lock.read_text();start=text.index('name = "pamin-engine"');end=text.index('\n[[package]]',start);block=text[start:end]
 if '\n "ort",' not in block:lock.write_text(text[:start]+block.replace(' "pamin-core",',' "ort",\n "pamin-core",',1)+text[end:])
 tests=target/'crates/pamin-engine/tests';shutil.copyfile(artifact/'fixture-future.rs.in',tests/'scratch_scored_fixture.rs');shutil.copyfile(artifact/'multihop.rs.in',tests/'scratch_scored_multihop.rs');(tests/'graph_trace').mkdir();shutil.copyfile(artifact/'graph_trace.rs.in',tests/'graph_trace/mod.rs')
 files={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'prepared-source-hashes.json').write_text(json.dumps(files,indent=2)+'\n')
