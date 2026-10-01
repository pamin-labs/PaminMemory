#!/usr/bin/env python3
"""Pure scope/disk negatives; optional resealed archive checks need offline slot."""
import gzip,importlib.machinery,importlib.util,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
if sys.flags.optimize:raise SystemExit('Negative guard checks require assertions')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
loader=importlib.machinery.SourceFileLoader('measurement_review',str(ROOT/'evidence_review.py'))
spec=importlib.util.spec_from_loader(loader.name,loader);review=importlib.util.module_from_spec(spec);loader.exec_module(review)

def row(phase='maintenance'):
    return {'phase':phase,'cpu_scope':'process threads only; PostgreSQL excluded','rss_scope':'process only','extra':{'index_disk':[1,100,512]}}

class PureGuards(unittest.TestCase):
    def test_disk_before_arithmetic(self):
        invalid=[None,[],[1,2],[1,2,3,4],[1,-100,-100],[True,1,1],[1,False,1],[1,1,True],[1,1.0,2],[1,'2',3]]
        for phase in ['maintenance','closed_index']:
            review.measurement_annotations(row(phase))
            for usage in invalid:
                changed=row(phase);changed['extra']['index_disk']=usage
                with self.assertRaisesRegex(AssertionError,'index_disk'):
                    review.measurement_annotations(changed)
                    self.fail('would reach disk summary arithmetic')
        zero=row();zero['extra']['index_disk']=[0,0,0];review.measurement_annotations(zero)
    def test_scopes_on_all_product_rows(self):
        for phase in ['open','maintenance','closed_index','search_warm']:
            for field in ['cpu_scope','rss_scope']:
                for value in [None,False,'whole service including PostgreSQL and devices']:
                    changed=row(phase);changed[field]=value
                    with self.assertRaisesRegex(AssertionError,'measurement scope'):review.measurement_annotations(changed)
                changed=row(phase);changed.pop(field)
                with self.assertRaisesRegex(AssertionError,'measurement scope'):review.measurement_annotations(changed)

def resealed_archives():
    import test_evidence_review as archive
    repo=archive.REPO;names=set()
    for rel in [archive.DISK,archive.HNSW]:
        names.update(json.loads((repo/rel/'manifest.json').read_text())['files']);names.add(str(rel/'manifest.json'))
    gitdir=subprocess.check_output(['git','-C',str(repo),'rev-parse','--absolute-git-dir'],text=True).strip()
    code=repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in'
    calc_loader=importlib.machinery.SourceFileLoader('retained_calculator',str(code))
    calc_spec=importlib.util.spec_from_loader(calc_loader.name,calc_loader);calc=importlib.util.module_from_spec(calc_spec);calc_loader.exec_module(calc)
    changes=[(phase,'index_disk',usage) for phase in ['maintenance','closed_index'] for usage in [[1,-100,-100],[True,1,1]]]
    changes += [('open','cpu_scope','process plus PostgreSQL and devices'),('open','rss_scope','whole service')]
    for rel in [archive.DISK,archive.HNSW]:
        for phase,field,value in changes:
            with tempfile.TemporaryDirectory(prefix='restart-annotations-negative-') as temp:
                target=Path(temp);(target/'.git').symlink_to(gitdir,target_is_directory=True)
                for name in names:
                    dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,dst)
                raw=target/rel/'raw.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
                selected=next(r for r in rows if r['phase']==phase)
                (selected['extra'] if field=='index_disk' else selected)[field]=value
                raw.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if field=='index_disk':
                    compressed=rel==archive.DISK;log=target/rel/'logs'/f"{selected['repetition']}-{selected['arm']}.log{'.gz' if compressed else ''}"
                    text=gzip.decompress(log.read_bytes()).decode() if compressed else log.read_text();lines=text.splitlines()
                    at=next(i for i,line in enumerate(lines) if line.startswith('RESTART_JSON ') and json.loads(line.removeprefix('RESTART_JSON '))['phase']==phase)
                    observation=json.loads(lines[at].removeprefix('RESTART_JSON '));observation['extra']['index_disk']=value
                    lines[at]='RESTART_JSON '+json.dumps(observation);text='\n'.join(lines)+'\n'
                    if compressed:log.write_bytes(gzip.compress(text.encode(),mtime=0))
                    else:log.write_text(text)
                    # Regenerate the derived summary too: guard must fail on
                    # raw measurements before stale tables/calculator checks.
                    (target/rel/'summary.json').write_text(json.dumps(calc.summarize(rows),indent=2)+'\n')
                archive.manifests(target);result=archive.execute(target,rel)
                expected='index_disk' if field=='index_disk' else 'measurement scope'
                assert result.returncode!=0 and expected in result.stderr,result.stderr
    print('PASS: both archives reject 12 resealed raw/log/derived-summary or runner-scope mutations')

if __name__=='__main__':
    run_archives='--archives' in sys.argv
    if not unittest.main(argv=[sys.argv[0]],exit=False).result.wasSuccessful():raise SystemExit(1)
    if run_archives:resealed_archives()
