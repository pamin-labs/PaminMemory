#!/usr/bin/env python3
"""Adversarial temporary copies; never modifies retained evidence or runs models."""
import contextlib,gzip,importlib.util,io,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evidence',BASE/'verify.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def put(p,o):
 b=(json.dumps(o,ensure_ascii=False)+'\n').encode();p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b)
def rows_change(root,process,change):
 p=root/'raw'/f'{process}.jsonl.gz';rows=[json.loads(l) for l in gzip.decompress(p.read_bytes()).splitlines()];change(rows)
 p.write_bytes(gzip.compress((''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)).encode(),mtime=0))
 log=root/'raw'/f'{process}.log.gz';lines=gzip.decompress(log.read_bytes()).decode().splitlines();i=0
 for at,l in enumerate(lines):
  if 'CACHE_ENGINE_ROW ' in l:
   n={k:x for k,x in rows[i].items() if k!='actual_delta'};lines[at]=l.split('CACHE_ENGINE_ROW ',1)[0]+'CACHE_ENGINE_ROW '+json.dumps(n,ensure_ascii=False);i+=1
 log.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
def wrong_counter(root):
 def change(rs):rs[1]['after']['scored']+=1;rs[1]['actual_delta']['scored']+=1
 rows_change(root,'r0-fixed-q101-accurate-A',change)
def wrong_order(root):
 def change(rs):rs[6]['complete'][0],rs[6]['complete'][1]=rs[6]['complete'][1],rs[6]['complete'][0]
 rows_change(root,'r0-fixed-q101-accurate-A',change)
def coverage(root):rows_change(root,'r0-fixed-q80-accurate-A',lambda rs:rs[0].update(coverage=1))
def provider(root):
 p=root/'raw/r0-fixed-q80-accurate-A-providers.json';o=json.loads(p.read_text());o[0]['nodes']={'CUDAExecutionProvider':1023};put(p,o)
def tuning(root):
 p=root/'raw/r0-fixed-q80-accurate-A-launch.json';o=json.loads(p.read_text());o['environment']['ORT_NUM_THREADS']='1';put(p,o)
def assets(root):
 p=root/'raw/provenance-after.json';o=json.loads(p.read_text());o['seed'][next(iter(o['seed']))]='0'*64;put(p,o)
def summary(root):
 p=root/'raw/summary.json';o=json.loads(p.read_text());o['comparisons'][0]['changed_logits']=1;put(p,o)
def costs(root):
 p=root/'raw/cost-observations.json';o=json.loads(p.read_text());o['rows'][0]['median_wall_us']+=1;put(p,o)
def binary(root):
 p=root/'raw/r0-fixed-q80-accurate-A-launch.json';o=json.loads(p.read_text());o['binary_sha256']='0'*64;put(p,o)
def missing(root):
 p=root/'raw/r0-fixed-q80-accurate-A-log-not-real';p.write_text('unexpected')
with contextlib.redirect_stdout(io.StringIO()):v.verify()
checks=[wrong_counter,wrong_order,coverage,provider,tuning,assets,summary,costs,binary,missing]
for mutate in checks:
 with tempfile.TemporaryDirectory(prefix='engine-evidence-negative-') as td:
  root=Path(td)/'evidence';shutil.copytree(BASE,root);mutate(root);v.ROOT=root
  try:
   with contextlib.redirect_stdout(io.StringIO()):v.verify(hashes=False)
  except (ValueError,KeyError):pass
  else:raise SystemExit('negative accepted: '+mutate.__name__)
# Integrity guard independently rejects a changed declared file.
with tempfile.TemporaryDirectory(prefix='engine-evidence-hash-') as td:
 root=Path(td)/'evidence';shutil.copytree(BASE,root);(root/'README.md').write_text('changed');v.ROOT=root
 try:v.verify()
 except ValueError:pass
 else:raise SystemExit('hash corruption accepted')
for args,env in [(['-O'],{}),(['-OO'],{}),([],{ 'PYTHONOPTIMIZE':'1'}),([],{ 'PYTHONOPTIMIZE':'0'})]:
 result=subprocess.run([sys.executable,*args,'-B',str(BASE/'verify.py')],env=os.environ|env,capture_output=True,text=True)
 if result.returncode==0 or 'Refuse optimized Python' not in result.stderr:raise SystemExit('optimized Python accepted')
print('PASS:10 semantic/membership negatives,1 hash negative,4 optimized-Python refusals')
