#!/usr/bin/env python3
"""Prepare scratch-only disabled/enabled source copies for the native fixture.
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
 tests=target/'crates/pamin-engine/tests';shutil.copyfile(artifact/'fixture.rs.in',tests/'scratch_scored_fixture.rs');(tests/'graph_trace').mkdir();shutil.copyfile(artifact/'graph_trace.rs.in',tests/'graph_trace/mod.rs')
 files={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'prepared-source-hashes.json').write_text(json.dumps(files,indent=2)+'\n')
