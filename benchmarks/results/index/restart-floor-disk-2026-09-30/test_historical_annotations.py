"""Bounded resealed historical-condition/rank/hardware/endpoint negatives."""
import gzip,hashlib,json,runpy,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive
BEFORE='--before' in sys.argv;BASE='a25771cd699b173b7164ee1033b1506d16298b63'
names=set()
for rel in [archive.DISK,archive.HNSW]:
 names.update(json.loads((archive.REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
gitdir=subprocess.check_output(['git','-C',str(archive.REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
cases=[(rel,case) for rel in [archive.DISK,archive.HNSW] for case in ['conditions_changed','conditions_missing','rank_bool','rank_float']]
cases += [(archive.HNSW,'hardware_'+field) for field in ['kernel','cpu_model','cgroup_cpu_max','cgroup_memory_max']]
cases += [(archive.HNSW,'hardware_readme')]
cases += [(archive.DISK,case) for case in ['seed_outside','seed_traversal','seed_exclusion_removed','seed_exclusion_added','seed_credential_exclusion']]
for rel,case in cases:
 with tempfile.TemporaryDirectory(prefix='restart-historical-annotations-') as directory:
  repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
  for name in names:
   dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive.REPO/name,dst)
  if BEFORE:
   for name in [str(archive.DISK/'evidence_review.py'),str(rel/'verify.py')]:
    (repo/name).write_bytes(subprocess.check_output(['git','-C',str(archive.REPO),'show',BASE+':'+name]))
  raw=repo/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
  if case.startswith(('conditions','rank')):
   selected=rows[0] if case.startswith('conditions') else next(r for r in rows if r['phase']=='search_warm' and type(r['extra']['rank']) is int and r['extra']['rank']==1)
   if case=='conditions_changed':selected['conditions']='cold caches; shared live database'
   elif case=='conditions_missing':selected.pop('conditions')
   else:
    selected['extra']['rank']=True if case=='rank_bool' else 1.0
    compressed=rel==archive.DISK;log=repo/rel/'logs'/f"{selected['repetition']}-{selected['arm']}.log{'.gz' if compressed else ''}"
    lines=(gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text()).splitlines();matched=0
    for at,line in enumerate(lines):
     if line.startswith('RESTART_JSON '):
      native=json.loads(line.removeprefix('RESTART_JSON '))
      if native['phase']==selected['phase'] and native['extra']['query_document']==selected['extra']['query_document']:
       native['extra']['rank']=selected['extra']['rank'];lines[at]='RESTART_JSON '+json.dumps(native);matched+=1
    assert matched==1
    text='\n'.join(lines)+'\n'
    if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
    else:log.write_text(text)
   raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
   review=runpy.run_path(str(repo/archive.DISK/'evidence_review.py'));index='disk' if rel==archive.DISK else 'memory'
   hwm=review['historical_hwm_review'](rows,index,hashlib.sha256(raw.read_bytes()).hexdigest());(repo/rel/'hwm-review.json').write_text(json.dumps(hwm,indent=2)+'\n')
   code=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in';calc=runpy.run_path(str(code));summary=review['qualified_summary'](calc['summarize'](rows),hwm,hashlib.sha256(code.read_bytes()).hexdigest())
   (repo/rel/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(repo/rel/'timing-review.json').write_text(json.dumps(review['timing_review'](summary,rows),indent=2)+'\n')
  elif case=='hardware_readme':
   path=repo/rel/'README.md';path.write_text(path.read_text().replace('AMD EPYC 9V74','Z80'))
  elif case.startswith('hardware'):
   path=repo/rel/'provenance.json';value=json.loads(path.read_text());value[case.removeprefix('hardware_')]='contradictory retained hardware';path.write_text(json.dumps(value,indent=2)+'\n')
  else:
   path=repo/rel/'provenance.json';provenance=json.loads(path.read_text());endpoint=repo/rel/'post-review-seed.json';receipt=json.loads(endpoint.read_text())
   if case.startswith(('seed_outside','seed_traversal')):
    entry=next(e for e in provenance['seed_files'] if '/postgres/data/' in e['path']);entry['path']='<SCRATCH>/unrelated/data' if case=='seed_outside' else '<SCRATCH>/seed-disk/../unrelated/data'
    inventory={e['path']:[e['bytes'],e['sha256']] for e in provenance['seed_files']};receipt['canonical_inventory_sha256']=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   elif case=='seed_exclusion_removed':receipt['exclusions'].remove('all symlinks')
   elif case=='seed_exclusion_added':receipt['exclusions'].append('index')
   else:provenance['credential_exclusions']=['everything']
   path.write_text(json.dumps(provenance,indent=2)+'\n');endpoint.write_text(json.dumps(receipt,indent=2)+'\n')
  archive.manifests(repo);result=archive.execute(repo,rel)
  if BEFORE:assert result.returncode==0,case+': '+result.stderr
  else:
   message=('cache/isolation conditions' if case.startswith('conditions') else 'rank must be None or an actual positive integer' if case.startswith('rank') else 'retained hardware' if case.startswith('hardware') else 'seed inventory path' if case.startswith(('seed_outside','seed_traversal')) else 'seed inventory exclusion policy')
   assert result.returncode!=0 and message in result.stderr,case+': '+result.stderr
print(('BEFORE: accepted ' if BEFORE else 'PASS: rejected ')+str(len(cases))+' matched/recomputed/resealed condition, rank, HNSW hardware and stopped-seed path/exclusion mutations')

if not BEFORE:
 review=runpy.run_path(str(archive.REPO/archive.DISK/'evidence_review.py'))
 for value in [None,1,10]:review['historical_rank'](value,value)
 for value in [True,False,0,-1,1.0,'1']:
  try:review['historical_rank'](value,1)
  except AssertionError as error:assert 'rank must be None or an actual positive integer' in str(error)
  else:raise AssertionError('invalid synthetic rank accepted')
 print('PASS: rank null/positive integer positives and6 invalid type/range cases before equality')
