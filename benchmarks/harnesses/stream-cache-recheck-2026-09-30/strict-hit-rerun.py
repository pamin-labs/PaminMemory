import hashlib,json,os,pathlib,subprocess,time,shutil
root=pathlib.Path('/private/tmp/pamin-review-prs/stream-cache-recheck');root.mkdir(exist_ok=True)
evidence=json.load(open('benchmarks/results/inference/coreml-stream-cache-2026-09-29.json'))['protocol']
source=pathlib.Path(evidence['source']);cache=pathlib.Path(evidence['cache']);graph=pathlib.Path(evidence['prepared']);copied=graph.with_name('source.onnx');expected='3ece88f7a06766959c38ad0cca861902ccd4530c15df0971f267abb1092c4109'
cmd=['--exact','scratch_cached_native_prepare_probe','--ignored','--nocapture']
def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def premise(full=False):
 assert source.is_file() and copied.is_file() and graph.is_file(), 'not a cache hit: missing source/model copy'
 assert source.stat().st_size==copied.stat().st_size==1136209678
 assert digest(graph)==expected
 if full:assert digest(source)==digest(copied)==evidence['source_sha256']
 return {'graph_inode':graph.stat().st_ino,'source_inode':copied.stat().st_ino,'graph_birth':graph.stat().st_birthtime,'source_birth':copied.stat().st_birthtime,'graph_mtime_ns':graph.stat().st_mtime_ns,'source_mtime_ns':copied.stat().st_mtime_ns}
initial=premise(full=True)
for item in evidence['binaries'].values():assert digest(pathlib.Path(item['path']))==item['sha256']
rows=[]
for round in range(30):
 for arm in (['before','after'] if round%2==0 else ['after','before']):
  assert shutil.disk_usage(root).free>10*2**30
  assert int(subprocess.check_output(['memory_pressure'],text=True).split('System-wide memory free percentage: ')[1].split('%')[0])>=25
  assert digest(pathlib.Path(evidence['binaries'][arm]['path']))==evidence['binaries'][arm]['sha256']
  previous=premise();env=os.environ.copy();env.update(PAMIN_NATIVE_MODEL_SOURCE=str(source),PAMIN_NATIVE_TEST_CACHE=str(cache),PAMIN_EXPECT_PREPARED_PATH=str(graph))
  start=time.monotonic();out=root/f'{round+1}-{arm}.log'
  with out.open('w') as f:proc=subprocess.Popen([evidence['binaries'][arm]['path'],*cmd],env=env,stdout=f,stderr=subprocess.STDOUT);done,stat,usage=os.wait4(proc.pid,0)
  exit=os.waitstatus_to_exitcode(stat);proc.returncode=exit
  assert exit==0,(arm,round,'see log')
  following=premise();assert previous==following,'cache files replaced during a nominal hit'
  row={'round':round+1,'arm':arm,'wall_seconds':time.monotonic()-start,'peak_process_rss_bytes':usage.ru_maxrss,'cpu_user_seconds':usage.ru_utime,'cpu_system_seconds':usage.ru_stime,'cache_hit_premise':'both files/content/inodes/birth times unchanged'}
  rows.append(row);(root/'rows.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
assert premise(full=True)==initial
print('60 strict-premise rotated stream-cache arms completed')
