"""Fail-closed preparation/measurement guards. Import never launches native/PG/models."""
import sys
if sys.flags.optimize:raise SystemExit("Guarded runner refuses -O/PYTHONOPTIMIZE")
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
GRAPH=Path('<SCRATCH>/graph-scored-measure')
sys.path.insert(0,str(GRAPH))
import importlib.util
_PG_PATH=Path(__file__).resolve().parent/"owned_postgres.frozen.py"
_spec=importlib.util.spec_from_file_location("memo_owned_postgres",_PG_PATH)
pg=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(pg)
ROOT=Path(__file__).resolve().parent
SEED=Path('<SCRATCH>/fusion-pool-prototype-tools/acceptance/own-results-v2/seed-home')
CACHE=Path('<MODEL_CACHE>')
RESERVE=8*1024**3
LIBRARIES=[Path('<WORKSPACE>/.onnxruntime/onnxruntime-linux-x64-1.28.0/lib/libonnxruntime.so.1.28.0'),Path('<WORKSPACE>/.cargo/lib/pamin/libzvec_c_api.so'),Path('/lib64/ld-linux-x86-64.so.2')]
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def inventory(root):
 return {str(p.relative_to(root)):({'symlink':os.readlink(p)} if p.is_symlink() else {'bytes':p.stat().st_size,'sha256':digest(p)}) for p in sorted(Path(root).rglob('*')) if p.is_symlink() or p.is_file()}
def assets():
 roots=[CACHE/'prepared',CACHE/'models--gpahal--bge-m3-onnx-int8',CACHE/'models--onnx-community--bge-reranker-v2-m3-ONNX']
 files=[p for r in roots for p in r.rglob('*') if p.is_file() and p.name!='used']+LIBRARIES
 return {str(p.resolve()):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(set(files))}
def hardware():
 cg=Path('/sys/fs/cgroup')
 return {'utc_ns':time.time_ns(),'disk_free':shutil.disk_usage(ROOT).free,'affinity':sorted(os.sched_getaffinity(0)),'cpuinfo':Path('/proc/cpuinfo').read_text(),'meminfo':Path('/proc/meminfo').read_text(),'load':list(os.getloadavg()),'cgroup':{n:(cg/n).read_text() for n in ['cpu.max','cpu.stat','cpuset.cpus.effective','memory.current','memory.max','memory.stat','memory.events'] if (cg/n).exists()}}
def headroom(extra=0):
 x=hardware();assert x['disk_free']>=RESERVE+extra,'8GiB reserve plus projected growth absent'
 available=int(next(l.split()[1] for l in x['meminfo'].splitlines() if l.startswith('MemAvailable:')))*1024
 assert available>=2*1024**3,'host memory available below2GiB'
 cg=x['cgroup']
 if cg['memory.max'].strip()!='max':
  stat=dict(l.split() for l in cg['memory.stat'].splitlines());uncommitted=int(cg['memory.max'])-int(cg['memory.current']);inactive=int(stat.get('inactive_file',0));estimate=uncommitted+inactive
  assert estimate>=2*1024**3,'estimated reclaimable cgroup headroom below2GiB'
  x['cgroup_headroom']={'uncommitted':uncommitted,'inactive_file':inactive,'estimate':estimate,'scope':'max-current+inactive_file estimate; reclaim not guaranteed, no forced cache drop'}
 return x
def clean_environment():
 # Capture inherited *names* only; do not print tokens/credentials. Reject
 # product/compiler/timing knobs, scrub all remaining cache/loader overrides.
 overrides=sorted(k for k in os.environ if k.startswith(('PAMIN_','ORT_','RUSTC','RUSTDOC')) or k in {'RUSTFLAGS','RUSTUP_TOOLCHAIN','OMP_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'} or k.startswith('CARGO_') and k not in {'CARGO_HOME','CARGO_TARGET_DIR','CARGO_BUILD_JOBS'})
 assert not overrides,'unexpected inherited tuning/compiler overrides: '+','.join(overrides)
 env={k:os.environ[k] for k in ['HOME','USER','LANG','LC_ALL','TZ','TMPDIR'] if k in os.environ}
 env['PATH']='/usr/local/bin:/usr/bin:/bin'
 return env,sorted(k for k in os.environ if k.startswith(('HF_','HUGGINGFACE_','PG','LD_','XDG_','ORT_','PAMIN_','CARGO_','RUST')))
def runtime_environment(home):
 env,scrubbed=clean_environment()
 env.update(PAMIN_DEVICE='cpu',PAMIN_PROFILE='accuracy',PAMIN_EVAL_HOME=str(home),HF_HOME=str(CACHE),HF_HUB_CACHE=str(CACHE),HUGGINGFACE_HUB_CACHE=str(CACHE),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',LD_LIBRARY_PATH='<WORKSPACE>/.cargo/lib/pamin:<WORKSPACE>/.onnxruntime/onnxruntime-linux-x64-1.28.0/lib')
 assert 'LD_PRELOAD' not in env and 'LD_AUDIT' not in env
 return env,scrubbed
def clone(seed,home):
 assert not home.exists() and not (seed/'postgres/data/postmaster.pid').exists()
 before=inventory(seed);size=sum(v.get('bytes',0) for v in before.values());headroom(size+32*1024**2)
 shutil.copytree(seed,home,symlinks=True)
 assert inventory(home)==before,'clone differs before setup metadata'
 (home/'.graph-disposable-clone').write_text('owned disposable memo acceptance clone\n')
 # Native attaches via captured localhost/server record only AFTER our owned
 # helper starts+authenticates exactly this cluster. Rewrite only clone setup.
 path=home/'server.json';record=json.loads(path.read_text());install=list((home/'postgres/install').glob('*/bin/pg_ctl'));assert len(install)==1
 record['installation_dir']=str(install[0].parent.parent.resolve());path.write_text(json.dumps(record)+'\n')
 pg.preflight(home);assert inventory(seed)==before
 return before
