"""Only tiny fake files and mocked orchestration; no native/model/PG execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from contextlib import nullcontext
import test_templates as fixtures
from test_templates import load

ROOT = Path(__file__).resolve().parent
BASE = 'a9fcb08401d8abe3c1d5671d71ee74c7831427ba'
BEFORE = '--before' in sys.argv
if BEFORE:
    sys.argv.remove('--before')


class ResumeAndOff(unittest.TestCase):
    setUp = fixtures.Templates.setUp
    tearDown = fixtures.Templates.tearDown

    def pg_identity(self):
        return {'files':{'bin/postgres':{'bytes':1,'sha256':'a'*64,'kind':'regular_file'}},'pg_config':{'version':'PostgreSQL 17.6'},'installation_sha256':'b'*64,'scope':'prospective owned PostgreSQL installation and mapped runtime; historical build identity N/A','server_executable':{'bytes':1,'sha256':'a'*64,'installation_file':'bin/postgres'},'mapped_libraries':[{'bytes':1,'sha256':'c'*64,'location':'external/libfake.so'}]}

    def baseline(self, module, filename):
        if BEFORE:
            path = self.work / filename
            relative = str((ROOT / (filename + '.in')).relative_to(ROOT.parents[2]))
            path.write_bytes(subprocess.check_output(['git', 'show', BASE + ':' + relative], cwd=ROOT))
            return load(module, path)
        return self.run if filename == 'run.py' else self.rows

    def test_interrupted_matrix_resumes_only_incomplete_and_checks_plan(self):
        run = self.baseline('reproduction_run', 'run.py')
        seed = self.work / 'seed'; (seed / 'index').mkdir(parents=True)
        (seed / 'index/data').write_bytes(b'fake index')
        (seed / '.graph-disposable-clone').touch()
        (seed / 'server.json').write_text('{}')
        (self.work / 'source-receipts.json').write_text('{}')
        binary = self.work / 'fake-binary'; binary.write_bytes(b'never execute')
        identity = self.pg_identity()
        pins = {arm: {'path':str(binary),'sha256':self.common.digest(binary),'bytes':binary.stat().st_size,'revision':self.common.MANIFEST['revisions'][arm]} for arm in ['main','stack']}
        pins['seed']=dict(pins['main'])
        (self.work / 'build/binaries.json').write_text(json.dumps(pins))
        jobs = self.common.schedule(); called = []; interrupt = [True]
        def native(home, env, executable, prefix):
            name = prefix.name.split('-attempt')[0]; called.append(name)
            Path(str(prefix)+'.log').write_text(name)
            if interrupt[0] and len(called)==2:
                raise RuntimeError('mock interrupted attempt')
            if getattr(self,'mutate_final',False) and len(called)==81:
                (seed/'index/data').write_bytes(b'changed seed during last process')
            return {'samples':[], 'native_wall_seconds':1, 'postgres_build_identity':identity}
        def parsed(text, job, bound, *args):
            self.assertEqual(text, job['name'])
            return {'documents':230}, [{'step':step,'job':job['name']} for step in range(len(job['sequence'].split(',')))]
        with patch.object(self.common,'exclusive',side_effect=nullcontext), \
             patch.object(self.common,'source_inputs'),patch.object(self.common,'assets',return_value={}), \
             patch.object(self.common,'guard',return_value={'mock':True}), \
             patch.object(self.common,'environment',return_value={}), \
             patch.object(self.common,'disk_sizes',return_value={'files':1,'logical_bytes':1,'allocated_bytes':4096}), \
             patch.object(self.pg,'preflight'),patch.object(self.pg,'layout',return_value={},create=True),\
             patch.object(self.pg,'installation_identity',return_value={k:identity[k] for k in ['files','pg_config','installation_sha256']},create=True),patch.object(run,'native',side_effect=native), \
             patch.object(run,'parse',side_effect=parsed), \
             patch.object(self.rows,'validate',return_value={'mock_validated':True}), \
             patch.object(self.rows,'hot_guard'),patch.object(self.rows,'oracle_checks',return_value=[]), \
             patch.object(self.rows,'shown_source_identity'), \
             patch.object(sys,'argv',['run','--execute','--exclusive']):
            with self.assertRaisesRegex(RuntimeError,'mock interrupted attempt'):
                run.main()
            self.assertTrue((self.work/'results'/(jobs[0]['name']+'.json')).is_file())
            interrupt[0]=False
            if BEFORE:
                with self.assertRaises(FileExistsError):run.main()
                self.assertEqual(len(called),2)
                return
            if getattr(self,'mutate_final',False):
                with self.assertRaisesRegex(ValueError,'run plan changed before completion'):run.main()
                self.assertFalse((self.work/'results/complete.json').exists())
                self.assertEqual(len(called),81)
                return
            def retained_active(home):
                if str(home).endswith(jobs[1]['name']+'-attempt0-home'):
                    raise RuntimeError('mock retained PostgreSQL active')
            self.pg.preflight.side_effect=retained_active
            with self.assertRaisesRegex(RuntimeError,'mock retained PostgreSQL active'):run.main()
            self.assertEqual(len(called),2)
            self.pg.preflight.side_effect=None
            run.main()
            self.assertEqual(len(called),81)
            self.assertEqual(called.count(jobs[0]['name']),1)
            self.assertEqual(called.count(jobs[1]['name']),2)
            self.assertEqual(len(list((self.work/'results').glob('r*.json'))),81) # run identity +80
            self.assertTrue(json.loads((self.work/'results/complete.json').read_text())['complete'])
            # Retain interrupted evidence, use a fresh attempt without signaling/deleting it.
            self.assertTrue((self.work/'results'/(jobs[1]['name']+'-attempt0-home')).is_dir())
            run.main();self.assertEqual(len(called),81)
            (self.work/'source-receipts.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'resume plan identity differs'):run.main()
            self.assertEqual(len(called),81)

    def test_seed_mutation_during_run_prevents_complete(self):
        if BEFORE:self.skipTest('resume API absent on base')
        self.mutate_final=True
        self.test_interrupted_matrix_resumes_only_incomplete_and_checks_plan()

    def test_checkpoint_publication_and_corruption(self):
        if BEFORE:self.skipTest('new atomic/checkpoint API is absent on base')
        out=self.work/'results';out.mkdir();target=out/'job.json'
        duplicate=out/'duplicate.json';duplicate.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,'duplicate checkpoint'):self.run.read_json(duplicate)
        with patch.object(self.run.os,'replace',side_effect=RuntimeError('mock power loss')):
            with self.assertRaisesRegex(RuntimeError,'mock power loss'):self.run.atomic_save(target,{'value':1})
        self.assertFalse(target.exists());self.assertTrue(list(out.glob('*.pending')))
        self.run.atomic_save(target,{'value':1});self.assertEqual(json.loads(target.read_text()),{'value':1})
        with self.assertRaisesRegex(ValueError,'preserve published'):self.run.atomic_save(target,{})
        job=self.common.schedule()[0];actual=[{'step':step} for step in range(13)]
        prefix=out/(job['name']+'-attempt0');log=Path(str(prefix)+'.log');log.write_text('retained log')
        identity=self.pg_identity();packet={'job':job,'rows':actual,'opened':{},'validated':[{}]*13,'usage':{'postgres_build_identity':identity},'postgres_build_identity':identity}
        packet['checkpoint']={'attempt':0,'identity_sha256':'identity','packet_sha256':hashlib.sha256(self.run.canonical(packet).encode()).hexdigest(),'log_sha256':self.common.digest(log)}
        with patch.object(self.run,'parse',return_value=({},actual)),patch.object(self.rows,'validate',return_value={}),patch.object(self.rows,'hot_guard'):
            self.assertEqual(self.run.checkpoint(packet,job,'identity',{},[],out,identity),packet)
            wrong=copy.deepcopy(identity);wrong['installation_sha256']='d'*64
            with self.assertRaisesRegex(ValueError,'PostgreSQL installation differs'):self.run.bind_postgres(wrong,identity)
            for field,value,message in [('attempt',True,'invalid checkpoint attempt'),('identity_sha256','changed','identity differs'),('packet_sha256','changed','packet hash differs'),('log_sha256','changed','log differs')]:
                bad=copy.deepcopy(packet);bad['checkpoint'][field]=value
                with self.assertRaisesRegex(ValueError,message):self.run.checkpoint(bad,job,'identity',{},[],out,identity)
            bad=copy.deepcopy(packet);bad['validated'][0]={'changed':True}
            bad['checkpoint']['packet_sha256']=hashlib.sha256(self.run.canonical({k:v for k,v in bad.items() if k!='checkpoint'}).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,'validation differs'):self.run.checkpoint(bad,job,'identity',{},[],out,identity)

    def test_off_cross_source_cold_and_new_all_hit_views(self):
        rows=self.baseline('rows','rows.py');results={}
        hit={'rank':1,'topic_id':'id','topic':'topic','state':{'content':'text'},'seed':None,'why':[{'kind':'channel','score':1.0,'weight':1.0,'contribution':1.0}],'score_bits':1065353216,'reranked_bits':[]}
        for round_ in range(4):
            for limit in [5,10]:
                for source in ['main','stack']:
                    job={'round':round_,'source':source,'limit':limit,'query_id':80,'tier':'off','sequence':'A,A,N'}
                    actual=[{kind:[copy.deepcopy(hit)] for kind in ['limited','complete','fused']} for step in range(3)]
                    results[(round_,source,limit,80,'off','A')]={'job':job,'rows':actual}
        self.assertEqual(rows.oracle_checks(results),[])
        for step in [0,2]:
            for kind in ['limited','complete','fused']:
                bad=copy.deepcopy(results);bad[(0,'stack',5,80,'off','A')]['rows'][step][kind][0]['state']['content']='changed retrieval'
                if BEFORE:self.assertEqual(rows.oracle_checks(bad),[])
                else:
                    with self.assertRaisesRegex(AssertionError,'Off cold/new cross-source'):rows.oracle_checks(bad)


if __name__=='__main__':unittest.main()
