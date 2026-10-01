"""Source/mock checks only: never launch Cargo, native helpers, PG or models."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('materialize216', ROOT / 'materialize.py')
materialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materialize)


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(name, injected=None):
    namespace = {'__name__': 'mock_' + name, '__file__': str(ROOT / (name + '.py.in'))}
    with patch.dict(sys.modules, injected or {}):
        exec(compile((ROOT / (name + '.py.in')).read_text(), namespace['__file__'], 'exec'), namespace)
    return namespace


class Templates(unittest.TestCase):
    def common(self):
        shared = types.ModuleType('shared_common')
        shared.require = require
        return load('common', {'shared_common': shared})

    def test_exact_36_process_428_call_matrix(self):
        jobs = self.common()['schedule']()
        self.assertEqual(len(jobs), 36)
        self.assertEqual(sum(len(j['sequence'].split(',')) for j in jobs), 428)
        self.assertEqual(len({j['name'] for j in jobs}), 36)
        self.assertEqual(sum(j['tier'] == 'off' for j in jobs), 4)
        self.assertEqual([j['source'] for j in jobs[:16]], ['main'] * 8 + ['stack'] * 8)
        self.assertEqual([j['source'] for j in jobs[16:32]], ['stack'] * 8 + ['main'] * 8)

    def test_original_probe_identity(self):
        receipt = json.loads((ROOT / 'source-reconstruction.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'probe.rs.in').read_bytes()).hexdigest(),
                         receipt['native_probe_sha256'])
        self.assertEqual(receipt['native_probe_sha256'],
                         '7a16f16acd939b68a65d8ae4c9a8fcff53e44d76508d4812ef0b8438b5299d2f')

    def test_all_python_sources_parse(self):
        for path in list(ROOT.glob('*.py')) + list(ROOT.glob('*.py.in')):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(), filename=str(path))

    def test_baseline_subset_original_blobs_and_membership(self):
        receipt = json.loads((ROOT / 'baseline-source-manifest.json').read_text())
        self.assertEqual(len(receipt['files']), 124)
        identity = {'archive': 'baseline-source.tar.gz', 'manifest': 'baseline-source-manifest.json',
                    'archive_sha256': materialize.BASELINE_ARCHIVE_SHA256,
                    'manifest_sha256': materialize.BASELINE_MANIFEST_SHA256}
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / 'baseline'
            materialize.export_subset(ROOT, identity, destination)
            self.assertEqual(set(materialize.inventory(destination)), set(receipt['files']))
            self.assertFalse((destination / 'internal-docs').exists())
            self.assertFalse(list(destination.rglob('scratch_*.rs')))

    def test_rehashed_archive_identity_is_not_accepted(self):
        identity = {'archive': 'baseline-source.tar.gz', 'manifest': 'baseline-source-manifest.json',
                    'archive_sha256': '0' * 64, 'manifest_sha256': materialize.BASELINE_MANIFEST_SHA256}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'archive binding'):
                materialize.export_subset(ROOT, identity, Path(temporary) / 'source')

    def test_missing_offline_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                materialize.dependency(Path(temporary))

    def test_changed_offline_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = next(iter(json.loads((ROOT / 'shared-dependency.json').read_text())['files']))
            (Path(temporary) / first).write_bytes(b'changed controller')
            with self.assertRaisesRegex(ValueError, 'dependency differs'):
                materialize.dependency(Path(temporary))

    def test_run_default_never_claims_80_880_or_launches(self):
        common = types.ModuleType('common')
        common.schedule = self.common()['schedule']
        common.require = require
        common.exclusive = lambda: self.fail('default plan must not acquire a live slot')
        shared_run = types.ModuleType('shared_run')
        shared_run.native = lambda *args: self.fail('native launch refused in source/mock test')
        shared_run.parse = lambda *args: self.fail('native parse not needed in default plan')
        injected = {'common': common, 'owned_postgres': types.ModuleType('owned_postgres'),
                    'rows': types.ModuleType('rows'), 'shared_run': shared_run}
        script = load('run', injected)
        with patch.object(sys, 'argv', ['run.py']), patch('builtins.print') as output:
            script['main']()
        plan = json.loads(output.call_args.args[0])
        self.assertEqual((plan['processes'], plan['timed_calls']), (36, 428))

    def test_live_run_requires_explicit_exclusive(self):
        common = types.ModuleType('common')
        common.schedule = self.common()['schedule']
        common.require = require
        shared_run = types.ModuleType('shared_run')
        shared_run.native = shared_run.parse = lambda *args: self.fail('must refuse before native work')
        script = load('run', {'common': common, 'owned_postgres': types.ModuleType('owned_postgres'),
                              'rows': types.ModuleType('rows'), 'shared_run': shared_run})
        with patch.object(sys, 'argv', ['run.py', '--execute']):
            with self.assertRaisesRegex(ValueError, 'exclusive slot'):
                script['main']()

    def test_analysis_rejects_incomplete_matrix(self):
        common = types.ModuleType('common')
        common.schedule = self.common()['schedule']
        common.require = require
        script = load('analyze', {'common': common, 'rows': types.ModuleType('rows')})
        with self.assertRaisesRegex(ValueError, 'process matrix'):
            script['report']([], [])

    def test_descriptive_percentile_matches_published_definition(self):
        common = types.ModuleType('common')
        script = load('analyze', {'common': common, 'rows': types.ModuleType('rows')})
        self.assertAlmostEqual(script['percentile']([1, 2, 3, 4], .95), 3.85)

    def test_predecessor_encoding_allowed_but_memo_encoding_refused(self):
        script = load('rows')
        before = {key: 0 for key in script['STATS']}
        after = dict(before, encode_us=10)
        previous = {'diagnostic_after': before, 'arm': 'A', 'query_id': 80,
                    'fused': [], 'complete': [], 'limited': []}
        row = dict(previous, before=before, after=after)
        script['hot_guard'](previous, row, 'main')
        with self.assertRaisesRegex(AssertionError, 'memo identical hot encoded'):
            script['hot_guard'](previous, row, 'stack')


if __name__ == '__main__':
    unittest.main(verbosity=2)
