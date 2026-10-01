#!/usr/bin/env python3
"""Matching raw/log mutations recompute summaries and displayed completeness cells."""
import importlib.machinery,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
sys.dont_write_bytecode=True
import test_query_schedule as t

def load(name,path):
    loader=importlib.machinery.SourceFileLoader(name,str(path));spec=importlib.util.spec_from_loader(name,loader);module=importlib.util.module_from_spec(spec);loader.exec_module(module);return module

def mutate(repo,phase,change,arms=None):
    path=repo/t.HNSW/'raw.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    for arm in arms or ['main','predecessor','candidate']:
        for rep in range(3):
            target=next(r for r in rows if r['arm']==arm and r['repetition']==rep and r['phase']==phase);change(target)
            log=repo/t.HNSW/'logs'/f'{rep}-{arm}.log';lines=log.read_text().splitlines()
            for at,line in enumerate(lines):
                if line.startswith('RESTART_JSON '):
                    observation=json.loads(line.removeprefix('RESTART_JSON '))
                    if observation['phase']==phase:
                        lines[at]='RESTART_JSON '+json.dumps({key:target[key] for key in observation});break
            log.write_text('\n'.join(lines)+'\n')
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    calculator=load('calculator',repo/'benchmarks/harnesses/restart-floor-2026-09-30/analyze.py.in');summary=calculator.summarize(rows)
    (repo/t.HNSW/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    review=load('review',repo/t.DISK/'evidence_review.py')
    (repo/t.HNSW/'timing-review.json').write_text(json.dumps(review.timing_review(summary,rows),indent=2)+'\n')
    path=repo/t.HNSW/'README.md';lines=path.read_text().splitlines();reference='predecessor'
    for at,line in enumerate(lines):
        if line.startswith('## Combined'):reference='main'
        if line.startswith('| Vector graph completeness (hybrid visibility checked) |'):
            metric=summary['comparisons'][reference]['metric_differences']['vector_completeness']
            lines[at]=f"| Vector graph completeness (hybrid visibility checked) | {metric['before']:.6f} | {metric['after']:.6f} | {metric['absolute_delta']:+.6f} | {metric['percent_delta']:+.3f}% |"
    path.write_text('\n'.join(lines)+'\n')

def extra(key,value):return lambda row:row['extra'].__setitem__(key,value)
def negative_cpu(row):
    row['process_before']['user_ticks']=row['process_after']['user_ticks']+1;row['cpu_user_seconds']=-0.01

if __name__=='__main__':
    checks=[('open',extra('documents',18001),None,'HNSW open document count'),('maintenance',extra('documents',18000),None,'HNSW post-write document count'),('maintenance',extra('completeness',1.0),['candidate'],'HNSW arm maintenance completeness'),('maintenance',extra('completeness',0.5),['main','predecessor'],'HNSW arm maintenance completeness'),('write_and_memory_drain',extra('applied',2),None,'HNSW single-write drain premise'),('durability_flush',extra('flushed',0),None,'HNSW durability flush premise'),('search_cold',negative_cpu,None,'invalid real timing observation'),('search_warm',extra('rank',10),None,'HNSW query hit/rank premise')]
    for check in [None,*checks]:
        with tempfile.TemporaryDirectory(prefix='hnsw-product-premise-') as td:
            repo=Path(td);t.copy_archive(repo)
            if check:mutate(repo,*check[:3])
            t.reseal(repo);result=t.execute(repo)
            if check:assert result.returncode!=0 and check[3] in result.stderr,result.stderr
            else:assert result.returncode==0,result.stderr
    print('PASS: HNSW positive and 8 matching raw/log, regenerated summary/table, resealed premise negatives')
