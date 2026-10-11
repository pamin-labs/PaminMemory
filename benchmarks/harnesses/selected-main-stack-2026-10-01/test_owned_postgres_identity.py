"""Prospective PostgreSQL identity tests: tiny files and mocked processes only."""
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parent
loader=importlib.machinery.SourceFileLoader('pg_identity',str(ROOT/'owned_postgres.py.in'))
spec=importlib.util.spec_from_loader(loader.name,loader)
pg=importlib.util.module_from_spec(spec);loader.exec_module(pg)


def fixture_headroom():
    free=shutil.disk_usage(ROOT).free
    if free<8*1024**3:raise RuntimeError('fixture requires 8 GiB disk reserve')
    available=int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024
    maximum=Path('/sys/fs/cgroup/memory.max');current=Path('/sys/fs/cgroup/memory.current')
    raw_headroom=None
    if maximum.is_file() and maximum.read_text().strip()!='max':
        raw_headroom=int(maximum.read_text())-int(current.read_text())
        available=min(available,raw_headroom)
    if available<128*1024**2:raise RuntimeError('fixture requires qualified 128 MiB memory headroom')
    return {'disk_free_bytes':free,'raw_cgroup_headroom_bytes':raw_headroom,'qualified_available_memory_bytes':available,'scope':'host MemAvailable capped by current cgroup max-current; cache reclaim not guaranteed'}


class IdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.headroom=fixture_headroom()
        print('RESOURCE_ADMISSION '+json.dumps(cls.headroom,sort_keys=True))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='pg-identity-mock-')
        self.home=Path(self.temp.name);self.install=self.home/'install'
        for name in ['postgres','pg_ctl','psql','pg_config']:
            path=self.install/'bin'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('mock '+name)
        (self.install/'lib').mkdir();(self.install/'lib/libpq.so.5').write_text('mock library')
        (self.install/'share').mkdir();(self.install/'share/build.conf').write_text('mock build configuration')
        self.item={'home':self.home,'install':self.install,'data':self.home/'data','record':{'port':1234}}
        self.expected={'pid':42,'port':1234}

    def tearDown(self):self.temp.cleanup()

    def config(self,command,**kwargs):
        self.assertEqual(kwargs['timeout'],5)
        return Mock(stdout='PostgreSQL 17.6' if command[-1]=='--version' else 'mock '+command[-1])

    def inventory(self):
        with patch.object(pg.subprocess,'run',side_effect=self.config):return pg.installation_identity(self.item,{})

    def test_inventory_binds_executables_libraries_config_and_flags(self):
        before=self.inventory()
        for file in ['bin/postgres','bin/pg_ctl','bin/psql','bin/pg_config','lib/libpq.so.5','share/build.conf']:
            self.assertEqual(before['files'][file]['sha256'],hashlib.sha256((self.install/file).read_bytes()).hexdigest())
        self.assertEqual(set(before['pg_config']),{'version','configure','cc','cflags','ldflags','libs'})
        (self.install/'share/build.conf').write_text('changed')
        self.assertNotEqual(before['installation_sha256'],self.inventory()['installation_sha256'])

    def test_configuration_and_library_change_changes_identity(self):
        before=self.inventory();(self.install/'lib/libpq.so.5').write_text('other library')
        self.assertNotEqual(before,self.inventory())
        with patch.object(pg.subprocess,'run',return_value=Mock(stdout='PostgreSQL 17.5')):
            with self.assertRaisesRegex(RuntimeError,'configuration version differs'):pg.installation_identity(self.item,{})

    def test_build_flags_are_part_of_installation_digest(self):
        before=self.inventory()
        original=self.config
        def changed(command,**kwargs):
            return Mock(stdout='different optimization') if command[-1]=='--cflags' else original(command,**kwargs)
        with patch.object(pg.subprocess,'run',side_effect=changed):
            self.assertNotEqual(before['installation_sha256'],pg.installation_identity(self.item,{})['installation_sha256'])

    def test_escape_symlink_refuses_inventory(self):
        target=self.home/'outside';target.write_text('unowned')
        (self.install/'lib/escape.so').symlink_to(target)
        with self.assertRaisesRegex(RuntimeError,'escapes'):
            self.inventory()

    def test_actual_running_executable_and_escaped_mapped_library_are_bound(self):
        library=self.install/'lib/library with space.so.5'
        library.write_text('mock whitespace library')
        before=self.inventory();binary={key:before['files']['bin/postgres'][key] for key in ['bytes','sha256']}
        encoded=str(library).replace(' ',r'\040')
        maps='100-200 r-xp 0 00:00 1 '+encoded+'\n'
        original=pg.file_digest
        def digest(path):return dict(binary) if str(path)=='/proc/42/exe' else original(path)
        with patch.object(pg,'identity',return_value=self.expected),patch.object(pg,'installation_identity',return_value=before),patch.object(pg,'file_digest',side_effect=digest),patch.object(Path,'read_text',return_value=maps):
            receipt=pg.build_identity(self.item,{},self.expected,before)
        self.assertEqual(receipt['server_executable']['sha256'],binary['sha256'])
        self.assertEqual(receipt['mapped_libraries'][0]['location'],'installation/lib/library with space.so.5')
        self.assertEqual(receipt['mapped_libraries'][0]['sha256'],hashlib.sha256(library.read_bytes()).hexdigest())
        self.assertIn('historical build identity N/A',receipt['scope'])
        self.assertIn('inode identity not attested',receipt['mapped_libraries_scope'])

    def test_changed_installation_and_replaced_running_binary_fail(self):
        before=self.inventory()
        for changed in ['installation','running']:
            after=dict(before,installation_sha256='changed') if changed=='installation' else before
            with self.subTest(changed=changed),patch.object(pg,'identity',return_value=self.expected),patch.object(pg,'installation_identity',return_value=after),patch.object(pg,'file_digest',return_value={'bytes':1,'sha256':'wrong'}):
                with self.assertRaisesRegex(RuntimeError,'changed across launch' if changed=='installation' else 'running PostgreSQL executable differs'):
                    pg.build_identity(self.item,{},self.expected,before)

    def test_start_retains_build_receipt_and_failure_stops_owned_server(self):
        before=self.inventory()
        def execute(command,**kwargs):
            if '--version' in command:return Mock(stdout=Path(command[0]).name+' (PostgreSQL) 17.6')
            value=command[-1].removeprefix('SHOW ')
            return Mock(stdout='170006' if value=='server_version_num' else '' if value=='unix_socket_directories' else pg.NATIVE_SETTINGS.get(value,''))
        for failure in [False,True]:
            with self.subTest(failure=failure),patch.object(pg,'preflight',return_value=self.item),patch.object(pg,'installation_identity',return_value=before),patch.object(pg.subprocess,'run',side_effect=execute),patch.object(pg,'identity',return_value=self.expected),patch.object(pg,'listening',return_value=True),patch.object(pg,'show_data_directory'),patch.object(pg,'client_environment',return_value={}),patch.object(pg,'build_identity',side_effect=RuntimeError('identity mismatch') if failure else None,return_value={'prospective':'mock build'}),patch.object(pg,'stop_item') as stop:
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'identity mismatch'):pg.start(self.home,{})
                    stop.assert_called_once_with(self.item,{},self.expected)
                else:
                    self.assertEqual(pg.start(self.home,{})['postgres_build_identity'],{'prospective':'mock build'})
                    stop.assert_not_called()


if __name__=='__main__':unittest.main()
