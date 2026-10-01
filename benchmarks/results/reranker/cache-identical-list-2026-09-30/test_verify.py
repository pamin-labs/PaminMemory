#!/usr/bin/env python3
"""Synthetic mutations of sanitized evidence; no native/build/PG/model calls."""
import sys
if sys.flags.optimize:raise SystemExit('Negative verifier tests refuse -O/PYTHONOPTIMIZE')
import copy,gzip,hashlib,json,shutil,struct,subprocess,tempfile
from pathlib import Path
import verify
ROOT=Path(__file__).resolve().parent

def seal(root):
 manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def edit_json(root,name,action):
 p=root/name;data=json.loads(p.read_text());action(data);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def edit_rows(root,name,action):
 p=root/'raw'/(name+'.log.gz');lines=gzip.decompress(p.read_bytes()).decode().splitlines();rows=[json.loads(l.split('CACHE_ENGINE_ROW ',1)[1]) for l in lines if 'CACHE_ENGINE_ROW ' in l];action(rows);i=0
 for at,line in enumerate(lines):
  if 'CACHE_ENGINE_ROW ' in line:lines[at]=line.split('CACHE_ENGINE_ROW ',1)[0]+'CACHE_ENGINE_ROW '+json.dumps(rows[i],ensure_ascii=False);i+=1
 p.write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))
def rewrite_log(root,name,action):
 p=root/'raw'/(name+'.log.gz');p.write_bytes(gzip.compress(action(gzip.decompress(p.read_bytes()).decode()).encode(),mtime=0))
def mutate_logit(root):
 def change(rows):
  r=rows[0];target=next(h for h in r['complete'] if h['reranked_bits']);identifier=target['topic_id'];bits=target['reranked_bits'][0]^1;score=struct.unpack('<f',struct.pack('<I',bits))[0]
  for field in ['complete','limited']:
   for h in r[field]:
    if h['topic_id']==identifier:
     h['reranked_bits']=[bits]
     for w in h['why']:
      if w['kind']=='reranked':w['score']=score
 edit_rows(root,'r0-baseline-l5-q80-accurate-A',change)

def delete_asset(root,suffix):
 def change(p):
  p['assets'].pop(next(k for k in p['assets'] if k.endswith(suffix)))
  fingerprint=hashlib.sha256(json.dumps(p['assets'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  p['assets_before_inventory_digest']=p['assets_after_inventory_digest']=fingerprint
 edit_json(root,'receipt.json',change)
def shift_ranks(root):
 for p in sorted((root/'raw').glob('*.log.gz')):
  def shift(rows):
   for r in rows:
    for field in ['fused','complete','limited']:
     for h in r[field]:h['rank']+=1
  edit_rows(root,p.name.removesuffix('.log.gz'),shift)
 def change(cost):
  for row in cost['quality']:
   row['relevant_ranks']={k:v+1 for k,v in row['relevant_ranks'].items()};limit=row['limit'];ranks=row['relevant_ranks'];gain=sum(1/verify.math.log2(v+1) for v in ranks.values() if v<=limit);ideal=sum(1/verify.math.log2(v+1) for v in range(1,min(len(ranks),limit)+1));row['ndcg_at_limit']=gain/ideal
 edit_json(root,'cost-and-gold.json',change)
def fusion_bit_drift(root):
 def change(rows):
  identifier=rows[0]['fused'][0]['topic_id']
  for field in ['fused','complete','limited']:
   for h in rows[0][field]:
    if h['topic_id']==identifier:h['score_bits']^=1
 edit_rows(root,'r0-baseline-l5-q80-accurate-A',change)
def continuity_drift(root):
 def change(rows):
  for key in ['before','after','diagnostic_before','diagnostic_after']:rows[1][key]['characters']+=1
 edit_rows(root,'r0-baseline-l5-q80-accurate-A',change)
def off_character_work(root):
 def change(rows):
  for step,r in enumerate(rows):
   for key in ['before','after','diagnostic_before','diagnostic_after']:
    if step!=0 or key!='before':r[key]['characters']+=1
 edit_rows(root,'r0-baseline-l5-q80-off-A',change)
def negative_counters(root):
 def change(rows):
  for row in rows:
   for snapshot in ['before','after','diagnostic_before','diagnostic_after']:
    row[snapshot]['characters']-=1000000
 edit_rows(root,'r0-baseline-l5-q80-accurate-A',change)
def wrong_graph_role(root):
 rewrite_log(root,'r0-baseline-l5-q80-accurate-A',lambda text:text.replace('prepared/ac146c082d1526dd1cd10597ae4a0ffc/model.onnx','models--gpahal--bge-m3-onnx-int8/snapshots/2b34e84df040034d4b9eabb62383a87c18955822/tokenizer.json'))
def main():
 print('positive',verify.verify(ROOT))
 mutations=[
 ('source copy binding',lambda r:(r/'sources/memo-reranking.rs').write_bytes((r/'sources/memo-reranking.rs').read_bytes()+b'\n// synthetic altered source\n')),
 ('cold raw logit drift',mutate_logit),
 ('memo hot encoding',lambda r:edit_rows(r,'r0-memo-l5-q80-accurate-A',lambda rows:rows[1]['after'].__setitem__('encode_us',rows[1]['before']['encode_us']+1))),
 ('wrong query/limit',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0].__setitem__('return_limit',10))),
 ('duplicate selected position',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0]['selected_fused_positions'].__setitem__(-1,rows[0]['selected_fused_positions'][0]))),
 ('can_be_seen skip',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0].__setitem__('can_be_seen',False))),
 ('executed graph role substitution',wrong_graph_role),
 ('negative lifetime counter shift',negative_counters),
 ('provider mismatch',lambda r:rewrite_log(r,'r0-baseline-l5-q80-accurate-A',lambda text:text.replace('CPUExecutionProvider','CUDAExecutionProvider'))),
 ('inherited cache override',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['environment'].__setitem__('HF_HOME','/foreign'))),
 ('unauthenticated PG',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['pg'].__setitem__('data_directory_verified_before_native_fixture',False))),
 ('foreign stop PID',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['pg']['stop'].__setitem__('owned_pid',999999))),
 ('hidden index mutation',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['index'].__setitem__('changed_file_names',[]))),
 ('OOM endpoint',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['hardware_after']['cgroup'].__setitem__('memory.events','oom 1\noom_kill 0\n'))),
 ('fabricated wall cost',lambda r:edit_json(r,'cost-and-gold.json',lambda p:p['groups'][0]['wall_us'].__setitem__('median',1))),
 ('fabricated gold rank',lambda r:edit_json(r,'cost-and-gold.json',lambda p:p['quality'][0]['relevant_ranks'].__setitem__(next(iter(p['quality'][0]['relevant_ranks'])),999))),
 ('fresh product lib reused',lambda r:edit_json(r,'receipt.json',lambda p:p['source_bindings'][0]['artifacts'][0].__setitem__('fresh',True))),
 ('runtime library unbound',lambda r:delete_asset(r,'libonnxruntime.so.1.28.0')),
 ('external weights unbound',lambda r:delete_asset(r,'ac146c082d1526dd1cd10597ae4a0ffc/model.onnx.data')),
 ('tokenizer unbound',lambda r:delete_asset(r,'6f5ff65298512715a1e669753bc754d2bc8f367b/tokenizer.json')),
 ('model asset roles incomplete',lambda r:edit_json(r,'receipt.json',lambda p:p['asset_roles']['accurate'].__setitem__('external_weights',[]))),
 ('frozen source identity false',lambda r:edit_json(r,'receipt.json',lambda p:p['source_bindings'][0].__setitem__('measured_commit','0'*40))),
 ('frozen binary identity false',lambda r:edit_json(r,'receipt.json',lambda p:p['source_bindings'][0].__setitem__('binary_sha256','0'*64))),
 ('changed proxima size false',lambda r:edit_json(r,'receipt.json',lambda p:p['processes'][0]['index']['changed_files'][0].__setitem__('after_bytes',1))),
 ('global rank quality shift',shift_ranks),
 ('fusion bits inconsistent',fusion_bit_drift),
 ('lifetime counter discontinuity',continuity_drift),
 ('Off character work',off_character_work),
 ('negative raw wall cost',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0].__setitem__('wall_us',-1))),
 ('negative raw process memory',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0]['process_after'].__setitem__('rss_kib',-1))),
 ('reversed process CPU ticks',lambda r:edit_rows(r,'r0-baseline-l5-q80-accurate-A',lambda rows:rows[0]['process_after'].__setitem__('utime_ticks',rows[0]['process_before']['utime_ticks']-1))),
 ('memory cost fabricated',lambda r:edit_json(r,'paired-memory.json',lambda p:p[0].__setitem__('difference',100))),
 ]
 for label,change in mutations:
  with tempfile.TemporaryDirectory(prefix='cache-memo-evidence-negative-') as tmp:
   root=Path(tmp)/'proof';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'));change(root);seal(root)
   try:verify.verify(root)
   except ValueError as error:
    if str(error).startswith('archive '):raise SystemExit('FAIL semantic mutation rejected only by manifest: '+label)
    print('PASS rejected',label,'—',str(error),flush=True)
   else:raise SystemExit('FAIL accepted '+label)
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)/'proof';shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__'));p=root/'README.md';p.write_bytes(p.read_bytes()+b'corrupt')
  try:verify.verify(root)
  except ValueError:print('PASS checksum corruption')
  else:raise SystemExit('FAIL checksum corruption')
 for flag in ['-O','-OO']:
  result=subprocess.run([sys.executable,flag,str(ROOT/'verify.py')],capture_output=True,text=True)
  if result.returncode==0 or 'refuses' not in result.stderr:raise SystemExit('FAIL optimized Python accepted')
  print('PASS rejected',flag)
 print('PASS positive +32semantic/1hash/2optimized negatives; no native/model/PG/build executed')
if __name__=='__main__':main()
