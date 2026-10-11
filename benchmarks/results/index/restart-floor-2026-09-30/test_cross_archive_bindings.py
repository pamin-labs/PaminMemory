"""Standalone HNSW must authenticate sibling model files without Disk verifier."""
import hashlib,json,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_query_schedule as archive
before='--before' in sys.argv
inputs=[str(archive.DISK/name) for name in ['provenance.json','provider-bindings.json']]
cases=['consistent_pair'] if before else ['provider_only','provenance_only','consistent_pair']
for case in cases:
 with tempfile.TemporaryDirectory(prefix='cross-archive-model-binding-') as directory:
  repo=Path(directory);archive.copy_archive(repo)
  manifest=repo/archive.HNSW/'manifest.json';m=json.loads(manifest.read_text())
  if before:
   for name in inputs:m['files'].pop(name,None)
   manifest.write_text(json.dumps(m,indent=2)+'\n')
  else:assert all(name in m['files'] for name in inputs), 'standalone cross-archive model inputs not authenticated'
  binding=repo/archive.DISK/'provider-bindings.json';provenance=repo/archive.DISK/'provenance.json'
  b=json.loads(binding.read_text());p=json.loads(provenance.read_text());metadata=b['roles']['embedding']['source_metadata']
  lines=metadata['text'].splitlines();lines[2]=str(int(lines[2])+1);metadata['text']='\n'.join(lines)+'\n';blob=metadata['text'].encode()
  observed=next(row for row in p['prepared_source_metadata'] if row['path']==metadata['path']);observed.update(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
  if case!='provenance_only':binding.write_text(json.dumps(b,indent=2)+'\n')
  if case!='provider_only':provenance.write_text(json.dumps(p,indent=2)+'\n')
  # Keep the HNSW manifest fixed. Even consistent edits must not pass.
  result=archive.execute(repo)
  if before:assert result.returncode==0,result.stderr
  else:
   expected=inputs[1] if case=='provider_only' else inputs[0]
   assert result.returncode!=0 and expected in result.stderr,result.stderr
print('BEFORE: unauthenticated consistent sibling metadata/inventory edits passed standalone HNSW' if before else 'PASS: standalone HNSW rejects3 individual/consistent sibling provenance/model-binding edits against fixed manifest hashes')
