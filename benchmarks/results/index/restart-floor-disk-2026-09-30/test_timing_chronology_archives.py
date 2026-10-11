"""Bounded resealed raw/log negatives, and optional original before-fix attacks."""
import gzip,importlib.util,json,shutil,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_evidence_review as archive
REPO=archive.REPO;before='--before' in sys.argv;BASE='5abd91af44d5b8e2de5590aedb6326bf504f28ef'
names=set()
for rel in [archive.DISK,archive.HNSW]:
 names.update(json.loads((REPO/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
gitdir=subprocess.check_output(['git','-C',str(REPO),'rev-parse','--absolute-git-dir'],text=True).strip()
cases=['wall_bool','hwm_within'] if before else ['wall_bool','process_wall_bool','cpu_bool','hwm_within','hwm_between','historical_anomaly_changed','historical_anomaly_removed']
for rel in [archive.DISK,archive.HNSW]:
 for case in cases:
  with tempfile.TemporaryDirectory(prefix='restart-timing-chronology-') as directory:
   repo=Path(directory);(repo/'.git').symlink_to(gitdir,target_is_directory=True)
   for name in names:
    dst=repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dst)
   if before:
    for name in [str(archive.DISK/'evidence_review.py'),str(rel/'verify.py'),str(rel/'README.md')]:
     (repo/name).write_bytes(subprocess.check_output(['git','-C',str(REPO),'show',BASE+':'+name]))
   raw=repo/rel/'raw.jsonl';rows=[json.loads(l) for l in raw.read_text().splitlines()]
   selected=next(r for r in rows if r['phase']=='new_write_search');field=None
   if case=='wall_bool':selected['wall_ms']=True;field='wall_ms'
   elif case=='process_wall_bool':selected=next(r for r in rows if r['phase']=='process_total');selected['wall_seconds']=True
   elif case=='cpu_bool':selected['cpu_user_seconds']=False;field='cpu_user_seconds'
   elif case in ['hwm_within','hwm_between']:
    if case=='hwm_between':selected=next(r for r in rows if r['phase']=='write_and_memory_drain')
    field='process_before' if case=='hwm_within' else 'process_after'
    selected[field]['rss_status']=[l if l.startswith('VmRSS:') else 'VmHWM: 2000000 kB' for l in selected[field]['rss_status']]
   else:
    index='disk' if rel==archive.DISK else 'memory'
    spec=importlib.util.spec_from_file_location('review',REPO/archive.DISK/'evidence_review.py');review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    anomaly=review.HWM_ANOMALIES[index][0];process=[r for r in rows if r['arm']==anomaly['arm'] and r['repetition']==anomaly['repetition']]
    selected=process[anomaly['previous'][0]];field=anomaly['previous'][2]
    value=anomaly['previous'][-1]+1
    selected[field]['rss_status']=[l if l.startswith('VmRSS:') else f'VmHWM: {value} kB' for l in selected[field]['rss_status']]
    if case=='historical_anomaly_removed':
     maximum=max(int(l.split()[1]) for row in process if row['phase']!='process_total' for f in ['process_before','process_after'] if row[f] is not None for l in row[f]['rss_status'] if l.startswith('VmHWM:'))
     for row in process:
      if row['phase']=='process_total':continue
      for f in ['process_before','process_after']:
       if row[f] is not None:row[f]['rss_status']=[l if l.startswith('VmRSS:') else f'VmHWM: {maximum} kB' for l in row[f]['rss_status']]
   raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
   if field is not None:
    compressed=rel==archive.DISK;log=repo/rel/'logs'/f"{selected['repetition']}-{selected['arm']}.log{'.gz' if compressed else ''}"
    text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();lines=text.splitlines()
    product=[r for r in rows if r['arm']==selected['arm'] and r['repetition']==selected['repetition'] and r['phase']!='process_total']
    native_lines=[i for i,l in enumerate(lines) if l.startswith('RESTART_JSON ')];assert len(product)==len(native_lines)
    for at,row in zip(native_lines,product):
     observation=json.loads(lines[at].removeprefix('RESTART_JSON '))
     fields=['process_before','process_after'] if case=='historical_anomaly_removed' else [field]
     for key in fields:
      if key in observation:observation[key]=row[key]
     lines[at]='RESTART_JSON '+json.dumps(observation)
    text='\n'.join(lines)+'\n'
    if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
    else:log.write_text(text)
   if before:
    spec=importlib.util.spec_from_file_location('review',repo/archive.DISK/'evidence_review.py');review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
    summary=json.loads((repo/rel/'summary.json').read_text());(repo/rel/'timing-review.json').write_text(json.dumps(review.timing_review(summary,rows),indent=2)+'\n')
   archive.manifests(repo);result=archive.execute(repo,rel)
   if before:assert result.returncode==0,result.stderr
   else:
    assert result.returncode!=0,result.stdout
    reason='real timing' if 'bool' in case else 'high-water mark'
    assert reason in result.stderr,result.stderr
print(('BEFORE: both archives accepted resealed bool wall and impossible new HWM decrease' if before else 'PASS: both archives reject14 resealed bool timings and added/changed/removed HWM anomalies before arithmetic'))
