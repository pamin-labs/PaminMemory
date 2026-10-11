"""Synthetic retained-packet integrity tests; no native/model/PG execution."""
import ast
import copy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch
import test_templates as fixtures

ROOT=Path(__file__).resolve().parent
BASE='c85510c629de1adf395fd32c32dcd7c38663c22c'
BEFORE='--before' in sys.argv
if BEFORE:sys.argv.remove('--before')


class AnalysisCheckpoints(unittest.TestCase):
    setUp=fixtures.Templates.setUp
    tearDown=fixtures.Templates.tearDown

    def retained(self):
        if BEFORE:
            name=str((ROOT/'analyze.py.in').relative_to(ROOT.parents[2]))
            (self.work/'analyze.py').write_bytes(subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT))
            self.analyzer=fixtures.load('reproduction_analyze',self.work/'analyze.py')
        out=self.work/'results';out.mkdir()
        (self.work/'source-receipts.json').write_text('{}')
        inputs=['config.json','sources.json','source-receipts.json','seed-receipt.json',
                'build/binaries.json','main/crates/pamin-engine/tests/corpus/queries.json',
                'run.py','rows.py','common.py','owned_postgres.py']
        if not BEFORE:inputs += ['analyze.py','metrics.py']
        jobs=self.common.schedule()
        installation={'files':{'bin/postgres':{'bytes':1,'sha256':'a'*64,'kind':'regular_file'}},'pg_config':{'version':'PostgreSQL 17.6'},'installation_sha256':'b'*64}
        build=dict(installation,scope='prospective owned PostgreSQL installation and mapped runtime; historical build identity N/A',server_executable={'bytes':1,'sha256':'a'*64,'installation_file':'bin/postgres'},mapped_libraries=[{'bytes':1,'sha256':'c'*64,'location':'external/libfake.so'}])
        bound={str((Path(self.common.CONFIG[pin['root']])/pin['relative']).resolve()):pin for pin in self.common.MANIFEST['assets']}
        plan={'schema':1,'jobs':jobs,'binaries':json.loads((self.work/'build/binaries.json').read_text()),'assets':bound,
              'inputs':{name:self.common.digest(self.work/name) for name in inputs},'seed_inventory':{},'postgres_installation':installation,
              'scope':'new synthetic method checkpoints only; no historical numeric identity; interrupted attempts retained'}
        identity=hashlib.sha256(self.run.canonical(plan).encode()).hexdigest()
        self.common.save(out/'run-identity.json',plan)
        for job in jobs:
            actual=[{'step':step,'job':job['name']} for step in range(len(job['sequence'].split(',')))]
            packet={'job':job,'rows':actual,'validated':[{} for row in actual],'opened':{},'usage':{'native_wall_seconds':1,'samples':[],'postgres_build_identity':build},'postgres_build_identity':build,
                    'host_conditions':{'before':{'affinity':[0,1]},'after':{'affinity':[0,1]}},
                    'postgres_disk':{'before':{'logical_bytes':1,'allocated_bytes':4096,'files':1},
                                     'after':{'logical_bytes':1,'allocated_bytes':4096,'files':1}}}
            log=out/(job['name']+'-attempt0.log');log.write_text(job['name'])
            packet['checkpoint']={'attempt':0,'identity_sha256':identity,'packet_sha256':hashlib.sha256(self.run.canonical(packet).encode()).hexdigest(),'log_sha256':self.common.digest(log)}
            self.common.save(out/(job['name']+'.json'),packet)
        complete={'complete':True,'processes':80,'calls':880,'classifications':[],'regenerated_fixture':True,'historical_numeric_identity_claimed':False,'run_identity_sha256':identity}
        if not BEFORE:
            complete.update(packet_file_sha256=self.run.packet_file_digests(out,jobs),
                            packet_binding_scope=self.run.COMPLETE_PACKET_SCOPE)
        self.common.save(out/'complete.json',complete)
        return out,jobs

    def analyze(self):
        def parsed(text,job,bound,*args):
            self.assertEqual(text,job['name'])
            return {},[{'step':step,'job':job['name']} for step in range(len(job['sequence'].split(',')))]
        with ExitStack() as stack:
            # The actual loader/checkpoint/read_json functions run. Product row
            # semantics and parsed native log payloads are deliberately mocked.
            stack.enter_context(patch.object(runpy,'run_path',return_value=vars(self.run)))
            source=stack.enter_context(patch.object(self.common,'source_inputs'))
            stack.enter_context(patch.object(self.run,'parse',side_effect=parsed))
            stack.enter_context(patch.object(self.rows,'validate',return_value={}))
            stack.enter_context(patch.object(self.rows,'hot_guard'))
            stack.enter_context(patch.object(self.rows,'oracle_checks',return_value=[]))
            report=stack.enter_context(patch.object(self.analyzer,'report',return_value={'calls':880}))
            stack.enter_context(patch.object(self.analyzer,'write_tables',side_effect=lambda path,result:path.write_text('mock report\n')))
            forbidden=stack.enter_context(patch.object(self.pg,'installation_identity',side_effect=AssertionError('pg_config forbidden')))
            self.analyzer.main()
            forbidden.assert_not_called()
            self.assertEqual(len(report.call_args.args[0]),80)
            if not BEFORE:
                source.assert_called_once()
                self.assertEqual(report.call_args.kwargs['provider_binding'],{'device':'cpu','execution_provider':'CPUExecutionProvider','scope':'retained actual per-process log assignment/graph/runtime bindings revalidated by run.checkpoint'})

    def test_complete_checkpoint_positive_without_pg_config(self):
        self.retained();self.analyze()
        self.assertTrue((self.work/'metrics.json').is_file())

    def test_packet_or_native_log_mutation(self):
        out,jobs=self.retained();path=out/(jobs[0]['name']+'.json');original=path.read_text()
        for case in ['packet','log']:
            path.write_text(original)
            log=out/(jobs[0]['name']+'-attempt0.log');log.write_text(jobs[0]['name'])
            if case=='packet':
                value=json.loads(path.read_text());value['usage']['native_wall_seconds']=9999;path.write_text(json.dumps(value))
            else:log.write_text('changed retained native log')
            with self.assertRaisesRegex(ValueError,'complete packet file digest differs' if case=='packet' else 'checkpoint log differs'):self.analyze()
            self.assertFalse((self.work/'metrics.json').exists())

    def test_controller_fields_with_refreshed_selfhash_are_bound_by_completion(self):
        out,jobs=self.retained();path=out/(jobs[0]['name']+'.json');original=path.read_text()
        changes=[('native-wall',lambda packet:packet['usage'].update(native_wall_seconds=9999)),
                 ('samples',lambda packet:packet['usage'].update(samples=[{'native':{'VmRSS':999,'VmHWM':999},'pg_main':{'VmRSS':999,'VmHWM':999}}])),
                 ('host',lambda packet:packet.update(host_conditions={'before':{'affinity':[99]},'after':{'affinity':[99]}})),
                 ('postgres-disk',lambda packet:packet.update(postgres_disk={'before':{'logical_bytes':999},'after':{'logical_bytes':999}}))]
        for name,change in changes:
            with self.subTest(name=name):
                packet=json.loads(original);change(packet)
                packet['checkpoint']['packet_sha256']=hashlib.sha256(self.run.canonical({k:v for k,v in packet.items() if k!='checkpoint'}).encode()).hexdigest()
                path.write_text(json.dumps(packet))
                with self.assertRaisesRegex(ValueError,'complete packet file digest differs'):self.analyze()
                self.assertFalse((self.work/'metrics.json').exists())
        path.write_text(original)

    def test_legacy_missing_incomplete_or_invalid_completion_bindings_fail_closed(self):
        out,jobs=self.retained();path=out/'complete.json';original=path.read_text()
        for mode in ['missing','incomplete','invalid-type','missing-scope']:
            with self.subTest(mode=mode):
                complete=json.loads(original)
                if mode=='missing':complete.pop('packet_file_sha256')
                elif mode=='incomplete':complete['packet_file_sha256'].pop(jobs[0]['name'])
                elif mode=='invalid-type':complete['packet_file_sha256'][jobs[0]['name']]=True
                else:complete.pop('packet_binding_scope')
                path.write_text(json.dumps(complete))
                with self.assertRaises(ValueError):self.analyze()
                self.assertFalse((self.work/'metrics.json').exists())
        path.write_text(original)

    def test_publication_digests_require_packet_to_match_validated_memory(self):
        out,jobs=self.retained();packets={self.run.key(job):self.run.read_json(out/(job['name']+'.json')) for job in jobs}
        self.assertEqual(self.run.packet_file_digests(out,jobs,packets),self.run.read_json(out/'complete.json')['packet_file_sha256'])
        packets[self.run.key(jobs[0])]['usage']['native_wall_seconds']=9999
        with self.assertRaisesRegex(ValueError,'validated in-memory checkpoint'):
            self.run.packet_file_digests(out,jobs,packets)

    def test_report_implementation_mutation(self):
        self.retained()
        for name in ['metrics.py','analyze.py']:
            with self.subTest(name=name):
                source=self.work/name;old=source.read_bytes()
                source.write_bytes(old+b'\n# altered report behavior\ndef report(*args, **kwargs): return {"changed_report": True}\n' if name=='analyze.py' else old+b'\ndef metric_tables(*args, **kwargs): return {"changed_report": True}\n')
                if BEFORE:
                    self.analyze();(self.work/'metrics.json').unlink()
                else:
                    with patch.object(runpy,'run_path',side_effect=AssertionError('unvalidated controller loaded')):
                        with self.assertRaisesRegex(ValueError,'method/configuration input changed: '+name):self.analyzer.main()
                    self.assertFalse((self.work/'metrics.json').exists())
                source.write_bytes(old)

    def test_runner_bootstrap_inventory_exactly_agrees(self):
        # Read literal inventories without executing run_identity/pg_config.
        def inventory(name, variable):
            tree=ast.parse((ROOT/name).read_text())
            values=[ast.literal_eval(node.value) for node in ast.walk(tree)
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in node.targets)]
            self.assertEqual(len(values),1)
            return values[0]
        runner=inventory('run.py.in','files');bootstrap=inventory('analyze.py.in','expected_inputs')
        self.assertEqual(len(runner),12);self.assertEqual(len(set(runner)),12)
        self.assertEqual(set(runner),bootstrap)
        self.assertTrue({'analyze.py','metrics.py'} <= bootstrap)

    def test_completion_plan_current_inputs_and_pg_bindings(self):
        out,jobs=self.retained();complete=out/'complete.json';original=complete.read_text()
        for field,value,reason in [('run_identity_sha256','wrong','plan/cardinality'),('processes',True,'plan/cardinality'),('classifications',[{'changed':True}],'oracle classifications')]:
            complete.write_text(original);changed=json.loads(original);changed[field]=value;complete.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,reason):self.analyze()
        complete.write_text(original)
        for name in ['rows.py','run.py']:
            source=self.work/name;old=source.read_bytes();source.write_bytes(old+b'\nraise AssertionError("modified source must not load")\n')
            with patch.object(runpy,'run_path',side_effect=AssertionError('unvalidated controller loaded')):
                with self.assertRaisesRegex(ValueError,'method/configuration input changed'):self.analyzer.main()
            source.write_bytes(old)
        path=out/(jobs[0]['name']+'.json');packet=json.loads(path.read_text())
        packet['usage']['postgres_build_identity']=copy.deepcopy(packet['postgres_build_identity']);packet['usage']['postgres_build_identity']['installation_sha256']='d'*64
        packet['checkpoint']['packet_sha256']=hashlib.sha256(self.run.canonical({k:v for k,v in packet.items() if k!='checkpoint'}).encode()).hexdigest();path.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError,'complete packet file digest differs'):self.analyze()
        self.assertFalse((self.work/'metrics.json').exists())


if __name__=='__main__':unittest.main()
