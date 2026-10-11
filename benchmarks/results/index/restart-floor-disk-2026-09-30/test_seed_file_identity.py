"""Recomputed stopped-seed endpoint negatives; immutable retained archive."""
import copy,hashlib,json,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent;repo=root.parents[3];source=root/'evidence_review.py'
before='--before' in sys.argv;text=source.read_text()
if before:text=subprocess.check_output(['git','show','3a84478496fcfb0e225c9fe65c54c41b8ada1679:'+str(source.relative_to(repo))],cwd=repo,text=True)
g={};exec(compile(text,str(source),'exec'),g)
p=json.loads((root/'provenance.json').read_text());r=json.loads((root/'post-review-seed.json').read_text());g['seed_endpoint'](p,r)
cases=[('bytes',-1),('bytes',True),('bytes',1.0),('sha256','not-a-hash'),('sha256','A'*64),('sha256','a'*63)]
for field,value in cases:
 provenance=copy.deepcopy(p);record=copy.deepcopy(r);entry=next(x for x in provenance['seed_files'] if not x['path'].endswith('/profile') and not x['path'].endswith('/.pamin-optimized-files'));entry[field]=value
 inventory={x['path']:[x['bytes'],x['sha256']] for x in provenance['seed_files']}
 record['canonical_inventory_sha256']=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest();record['total_bytes']=sum(x['bytes'] for x in provenance['seed_files'])
 if before:g['seed_endpoint'](provenance,record)
 else:
  try:g['seed_endpoint'](provenance,record)
  except AssertionError as error:assert 'seed file identity' in str(error)
  else:raise AssertionError('malformed seed file identity accepted')
print('BEFORE: six rederived impossible seed identities accepted' if before else 'PASS: complete seed endpoint positive and six rederived impossible identities rejected')
