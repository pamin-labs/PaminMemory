"""Future accounting API tests; all process operations mocked, tiny files only."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import sys
sys.dont_write_bytecode = True
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parent
loader=importlib.machinery.SourceFileLoader('future_accounting',str(ROOT/'source/future-accounting.py.in'))
spec=importlib.util.spec_from_loader(loader.name,loader)
runner=importlib.util.module_from_spec(spec);loader.exec_module(runner)


def resource_gate():
    # One sequential archive copy plus conservative file/temporary overhead.
    # These tests mock processes; native/shared-host admission is external.
    archive_bytes=sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())
    required=2*archive_bytes+1024**2
    free=shutil.disk_usage(tempfile.gettempdir()).free
    if free<required:raise RuntimeError('insufficient temporary disk for sequential archive-copy fixtures')
    return {'temporary_disk_free_bytes':free,'archive_bytes':archive_bytes,'required_disk_bytes':required,'scope':'conservative fixture copy budget, not a measured peak or native admission'}


class FutureRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):print('RESOURCE_ADMISSION '+json.dumps(resource_gate(),sort_keys=True))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='graph-accounting-mock-')
        self.root=Path(self.temp.name);self.paths={}
        for role,name in [('ort','libonnxruntime.so.1.28.0'),('zvec','libzvec_c_api.so')]:
            path=self.root/name;path.write_text('synthetic '+role);self.paths[role]=path
        self.pins={role:hashlib.sha256(path.read_bytes()).hexdigest() for role,path in self.paths.items()}
        self.environment=patch.dict(runner.os.environ,{'GRAPH_OUT':str(self.root/'result.jsonl')})
        self.environment.start()

    def tearDown(self):self.environment.stop();self.temp.cleanup()

    def test_small_fixture_disk_budget(self):
        required=resource_gate()['required_disk_bytes']
        self.assertLess(required,8*1024**3)
        with patch.object(shutil,'disk_usage',return_value=Mock(free=required)):
            self.assertEqual(resource_gate()['temporary_disk_free_bytes'],required)
        with patch.object(shutil,'disk_usage',return_value=Mock(free=required-1)):
            with self.assertRaisesRegex(RuntimeError,'temporary disk'):resource_gate()

    def test_collector_requires_both_actual_paths_and_hashes(self):
        lines='\n'.join('1-2 r-xp 0 00:00 1 '+str(path) for path in self.paths.values())
        with patch.object(runner,'PINS',self.pins),patch.object(Path,'read_text',return_value=lines):
            self.assertEqual(set(runner.maps(42,self.paths)),{'ort','zvec'})
        wrong=lines.replace(str(self.paths['zvec']),'/other/libzvec_c_api.so')
        with patch.object(runner,'PINS',self.pins),patch.object(Path,'read_text',return_value=wrong):
            with self.assertRaisesRegex(RuntimeError,'unexpected actual mapped zvec path'):runner.maps(42,self.paths)

    def test_capture_retains_wait4_log_and_mapped_receipts(self):
        process=Mock(pid=42)
        accounting=Mock(ru_utime=2.5,ru_stime=.5,ru_maxrss=4096)
        mapped={role:{'path':str(path),'sha256':self.pins[role]} for role,path in self.paths.items()}
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen',return_value=process) as launch,patch.object(runner,'maps',return_value=mapped),patch.object(runner.os,'waitid',return_value=Mock()),patch.object(runner.os,'killpg') as kill,patch.object(runner.os,'wait4',return_value=(42,0,accounting)) as wait:
            report=runner.capture(['mock native'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
        self.assertEqual(report['user_seconds'],2.5);self.assertEqual(report['system_seconds'],.5)
        self.assertEqual(report['maximum_process_rss_kib'],4096);self.assertEqual(report['exit_status'],0)
        self.assertEqual(launch.call_args.kwargs['env']['GRAPH_OUT'],str(self.root/'result.jsonl'))
        self.assertTrue(launch.call_args.kwargs['start_new_session']);wait.assert_called_once_with(42,0);kill.assert_called_once()
        self.assertEqual(json.loads((self.root/'mapped').read_text())['libraries'],mapped)
        self.assertTrue((self.root/'log').exists())

    def test_missing_zvec_refuses_success_but_retains_accounting(self):
        process=Mock(pid=42);accounting=Mock(ru_utime=0,ru_stime=0,ru_maxrss=1)
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen',return_value=process),patch.object(runner,'maps',return_value={}),patch.object(runner.os,'waitid',return_value=Mock()),patch.object(runner.os,'killpg'),patch.object(runner.os,'wait4',return_value=(42,0,accounting)):
            with self.assertRaisesRegex(RuntimeError,'both actual runtime mappings'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
        self.assertTrue((self.root/'usage').is_file());self.assertTrue((self.root/'mapped').is_file())

    def test_early_mapping_conflict_retains_failed_accounting(self):
        process=Mock(pid=42);accounting=Mock(ru_utime=.2,ru_stime=.1,ru_maxrss=2)
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen',return_value=process),patch.object(runner,'maps',side_effect=RuntimeError('unexpected actual mapped zvec path')),patch.object(runner.os,'waitid',return_value=Mock()),patch.object(runner.os,'killpg'),patch.object(runner.os,'wait4',return_value=(42,9,accounting)):
            with self.assertRaisesRegex(RuntimeError,'unexpected actual mapped zvec'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
        usage=json.loads((self.root/'usage').read_text());mapped=json.loads((self.root/'mapped').read_text())
        self.assertFalse(usage['capture_accepted']);self.assertFalse(mapped['accepted'])
        self.assertEqual(usage['user_seconds'],.2);self.assertIn('zvec',mapped['capture_error'])

    def test_timeout_retains_failed_wait4_accounting(self):
        process=Mock(pid=42);accounting=Mock(ru_utime=0,ru_stime=0,ru_maxrss=1)
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen',return_value=process),patch.object(runner,'maps',return_value={}),patch.object(runner.os,'waitid',side_effect=[None,Mock(),Mock()]),patch.object(runner.os,'killpg'),patch.object(runner.os,'wait4',return_value=(42,9,accounting)),patch.object(runner.time,'monotonic',side_effect=[0,1000,1001,1002]):
            with self.assertRaisesRegex(RuntimeError,'native fixture timeout'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
        self.assertFalse(json.loads((self.root/'usage').read_text())['capture_accepted'])

    def test_stuck_killed_child_has_explicit_missing_accounting(self):
        process=Mock(pid=42)
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen',return_value=process),patch.object(runner,'maps',return_value={}),patch.object(runner.os,'waitid',return_value=None),patch.object(runner.os,'killpg'),patch.object(runner.os,'wait4') as wait,patch.object(runner.time,'monotonic',side_effect=[0,1000,1001,1020]):
            with self.assertRaisesRegex(RuntimeError,'accounting unavailable'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
            wait.assert_not_called()
        self.assertTrue((self.root/'log').exists());self.assertFalse((self.root/'usage').exists())
        self.assertIn(process,runner.RETAINED_PROCESSES)

    def test_existing_graph_result_and_dangling_link_are_preserved(self):
        result=self.root/'result.jsonl'
        for kind in ['file','directory','dangling_link']:
            if kind=='file':result.write_text('existing native evidence')
            elif kind=='directory':result.mkdir()
            else:result.symlink_to(self.root/'absent target')
            with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'outputs must be fresh'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
                launch.assert_not_called()
            self.assertFalse((self.root/'log').exists());self.assertFalse((self.root/'usage').exists());self.assertFalse((self.root/'mapped').exists())
            if kind=='file':self.assertEqual(result.read_text(),'existing native evidence');result.unlink()
            elif kind=='directory':result.rmdir()
            else:self.assertTrue(result.is_symlink());self.assertFalse(result.exists());result.unlink()

    def test_graph_result_alias_and_missing_contract_are_rejected(self):
        for target in ['log','usage','mapped']:
            with patch.dict(runner.os.environ,{'GRAPH_OUT':str(self.root/target)}),patch.object(runner.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'output paths must be distinct'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
                launch.assert_not_called()
        with patch.dict(runner.os.environ,{},clear=True),patch.object(runner.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'GRAPH_OUT result path required'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
            launch.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        (self.root/'log').write_text('existing user data')
        with patch.object(runner,'PINS',self.pins),patch.object(runner.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'outputs must be fresh'):runner.capture(['mock'],self.paths,self.root/'log',self.root/'usage',self.root/'mapped')
            launch.assert_not_called()
        self.assertEqual((self.root/'log').read_text(),'existing user data')


if __name__=='__main__':unittest.main()
