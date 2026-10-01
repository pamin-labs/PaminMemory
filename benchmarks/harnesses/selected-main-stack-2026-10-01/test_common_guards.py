"""Model-free mount mapping and restricted-proc cleanup regressions."""
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parent


class CommonGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        (self.work / 'common.py').write_bytes((ROOT / 'common.py.in').read_bytes())
        (self.work / 'sources.json').write_text('{}')
        (self.work / 'config.json').write_text('{}')
        loader = importlib.machinery.SourceFileLoader('guard_fixture', str(self.work / 'common.py'))
        spec = importlib.util.spec_from_loader('guard_fixture', loader)
        self.common = importlib.util.module_from_spec(spec)
        loader.exec_module(self.common)

    def tearDown(self):
        self.temp.cleanup()

    def mount(self, subtree, member, escaped=False):
        mounted = self.work / ('mounted cgroup' if escaped else 'mounted')
        mounted.mkdir(exist_ok=True)
        (mounted / 'memory.max').write_text(str(8 * 1024**3))
        (mounted / 'memory.current').write_text(str(1024**3))
        (mounted / 'cpu.max').write_text('200000 100000')
        membership = self.work / 'membership'
        membership.write_text('0::' + member + '\n')
        mountinfo = self.work / 'mountinfo'
        path = str(mounted).replace(' ', r'\040')
        mountinfo.write_text(f'27 1 0:26 {subtree} {path} rw,nosuid - cgroup2 cgroup rw\n')
        return mounted, membership, mountinfo

    def test_subtree_mount_does_not_repeat_membership_prefix(self):
        mounted, membership, mountinfo = self.mount('/slice/pod', '/slice/pod/worker')
        leaf = mounted / 'worker'
        leaf.mkdir()
        (leaf / 'memory.max').write_text('max')
        (leaf / 'memory.current').write_text('1024')
        (leaf / 'cpu.max').write_text('500000 100000')
        conditions = self.common.cgroup_conditions(mounted, membership, mountinfo)
        self.assertEqual(conditions['cgroup_path'], str(leaf))
        self.assertEqual(conditions['effective_memory_headroom_bytes'], 7 * 1024**3)
        self.assertEqual(conditions['effective_cpu_quota_cores'], 2)
        self.assertEqual(conditions['mounted_filesystem_subtree'], '/slice/pod')
        self.assertEqual(len(conditions['cgroup_ancestors']), 2)

    def test_bind_root_escaped_mountpoint_and_visible_ancestor_boundary(self):
        mounted, membership, mountinfo = self.mount('/slice/pod', '/slice/pod', escaped=True)
        conditions = self.common.cgroup_conditions(None, membership, mountinfo)
        self.assertEqual(conditions['cgroup_path'], str(mounted))
        self.assertEqual(len(conditions['cgroup_ancestors']), 1)
        self.assertIn('invisible ancestor limits unknown', conditions['scope'])
        mountinfo.write_text(f'27 1 0:26 / {str(mounted).replace(" ", r"\040")} rw - cgroup2 cgroup rw\n')
        membership.write_text('0::/\n')
        self.assertEqual(self.common.cgroup_conditions(mounted, membership, mountinfo)['cgroup_path'], str(mounted))

    def test_missing_mapping_traversal_or_leaf_fails_closed(self):
        mounted, membership, mountinfo = self.mount('/slice/pod', '/slice/other')
        with self.assertRaisesRegex(ValueError, 'does not map'):
            self.common.cgroup_conditions(mounted, membership, mountinfo)
        for member in ['/slice/pod/../escape', '/slice/pod/.']:
            membership.write_text('0::' + member + '\n')
            with self.assertRaisesRegex(ValueError, 'path escape'):
                self.common.cgroup_conditions(mounted, membership, mountinfo)
        membership.write_text('0::/slice/pod/missing\n')
        with self.assertRaisesRegex(ValueError, 'effective cgroup missing'):
            self.common.cgroup_conditions(mounted, membership, mountinfo)

    def proc_fixture(self):
        proc = self.work / 'proc'
        for pid, state, group in [(1234, 'Z', 1234), (1235, 'S', 1234), (9876, 'S', 9876)]:
            entry = proc / str(pid)
            entry.mkdir(parents=True)
            (entry / 'stat').write_text(f'{pid} (fixture name) {state} 1 {group}\n')
        return proc

    def test_hidepid_unrelated_uid_is_skipped_owned_live_member_is_verified(self):
        proc = self.proc_fixture()
        original_read = Path.read_text
        original_stat = Path.stat
        def read(path, *args, **kwargs):
            if path == proc / '9876/stat':
                raise PermissionError('restricted unrelated PID')
            return original_read(path, *args, **kwargs)
        def metadata(path, *args, **kwargs):
            if path == proc / '9876':
                return Mock(st_uid=os.geteuid() + 1)
            return original_stat(path, *args, **kwargs)
        with patch.object(self.common, 'Path', side_effect=lambda value: proc if value == '/proc' else Path(value)), \
             patch.object(Path, 'read_text', read), patch.object(Path, 'stat', metadata):
            self.assertEqual(self.common.live_group_members(1234), [1235])
            (proc / '1235/stat').write_text('1235 (fixture) Z 1 1234\n')
            process = Mock(pid=1234)
            with patch.object(self.common, 'exit_status', return_value=0), patch.object(self.common.os, 'killpg') as terminate:
                self.common.stop_group(process)
            terminate.assert_called_once_with(1234, self.common.signal.SIGKILL)
            process.wait.assert_called_once_with(timeout=10)

    def test_unreadable_owned_or_unknown_uid_is_not_reported_drained(self):
        proc = self.proc_fixture()
        for pid in [1234, 1235]:
            with self.subTest(pid=pid):
                original = Path.read_text
                def read(path, *args, **kwargs):
                    if path == proc / str(pid) / 'stat':
                        raise PermissionError('owned process cannot be inspected')
                    return original(path, *args, **kwargs)
                with patch.object(self.common, 'Path', side_effect=lambda value: proc if value == '/proc' else Path(value)), \
                     patch.object(Path, 'read_text', read):
                    with self.assertRaisesRegex(ValueError, 'same-UID process stat inaccessible'):
                        self.common.live_group_members(1234)

    def test_unknown_uid_inspection_failure_is_not_silently_skipped(self):
        proc = self.proc_fixture()
        original_read = Path.read_text
        original_stat = Path.stat
        def read(path, *args, **kwargs):
            if path == proc / '9876/stat':
                raise PermissionError('restricted PID')
            return original_read(path, *args, **kwargs)
        def metadata(path, *args, **kwargs):
            if path == proc / '9876':
                raise PermissionError('UID cannot be classified')
            return original_stat(path, *args, **kwargs)
        with patch.object(self.common, 'Path', side_effect=lambda value: proc if value == '/proc' else Path(value)), \
             patch.object(Path, 'read_text', read), patch.object(Path, 'stat', metadata):
            with self.assertRaises(PermissionError):
                self.common.live_group_members(1234)
