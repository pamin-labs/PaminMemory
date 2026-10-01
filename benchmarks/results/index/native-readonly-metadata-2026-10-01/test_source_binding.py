"""Source identity/scope negatives; JSON and read-only Git only."""
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import source_binding


class PublicSourceBinding(unittest.TestCase):
    def setUp(self):
        self.binding = json.loads((source_binding.ROOT/'public-source-binding.json').read_text())

    def test_valid_pinned_manifest(self):
        self.assertTrue(source_binding.validate(self.binding))

    def test_positive_public_git_blobs(self):
        result = source_binding.check_git(self.binding)
        if result is None:
            self.skipTest('public Git objects unavailable; source-byte check unmeasured')
        self.assertEqual(result, 35)

    def test_rehashed_scope_and_identity_negatives(self):
        mutations = {
            'scope':lambda x:x.update(scope='full historical helper and build reproduced'),
            'historical_public':lambda x:x.update(historical_source_public=True),
            'helper_reconstruction':lambda x:x.update(reconstructs_historical_helper_binary=True),
            'commit_alias':lambda x:x.update(public_equivalent_commit='refs/tags/'+source_binding.PUBLIC_TAG),
            'commit':lambda x:x.update(public_equivalent_commit='0'*40),
            'blob_alias':lambda x:x['files'][0].update(git_blob='HEAD:Cargo.lock'),
            'blob':lambda x:x['files'][0].update(git_blob='0'*40),
            'sha':lambda x:x['files'][0].update(sha256='0'*64),
            'path':lambda x:x['files'][0].update(path='crates/pamin-index/src/unmeasured.rs'),
            'private_path':lambda x:x['files'][0].update(path='crates/pamin-index/tests/scratch_native_readonly_probe.rs'),
            'extra_claim':lambda x:x['files'][0].update(full_build_verified=True),
            'bytes':lambda x:x['files'][0].update(bytes=x['files'][0]['bytes']+1),
            'tag':lambda x:x.update(public_tag='main'),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(self.binding); mutate(changed)
                serialized = json.dumps(changed, sort_keys=True, separators=(',', ':')).encode()
                self.assertNotEqual(hashlib.sha256(serialized).hexdigest(), source_binding.MANIFEST_SHA256)
                with self.assertRaises(ValueError):
                    source_binding.validate(json.loads(serialized))

    def test_git_reads_disable_replacement_and_scrub_overrides(self):
        actual_run = subprocess.run; calls = []
        def guarded(command, **kwargs):
            calls.append(command)
            self.assertIn('--no-replace-objects', command)
            self.assertIn('--literal-pathspecs', command)
            self.assertEqual(kwargs['env']['GIT_NO_REPLACE_OBJECTS'], '1')
            self.assertEqual(kwargs['env']['GIT_NO_LAZY_FETCH'], '1')
            self.assertEqual(kwargs['env']['GIT_CONFIG_COUNT'], '0')
            self.assertNotIn('GIT_REPLACE_REF_BASE', kwargs['env'])
            self.assertNotIn('GIT_DIR', kwargs['env'])
            self.assertNotIn(source_binding.HISTORICAL_LOCAL_COMMIT, command)
            return actual_run(command, **kwargs)
        with patch.dict('os.environ', {'GIT_REPLACE_REF_BASE':'refs/evil', 'GIT_CONFIG_COUNT':'1', 'GIT_CONFIG_KEY_0':'core.fsmonitor', 'GIT_CONFIG_VALUE_0':'unsafe-hook', 'GIT_DIR':'/unrelated', 'GIT_NO_LAZY_FETCH':'0'}):
            with patch.object(source_binding.subprocess, 'run', side_effect=guarded):
                result = source_binding.check_git(self.binding)
        self.assertTrue(calls)
        if result is None:self.skipTest('public Git objects unavailable')
        self.assertEqual(result, 35)

    def test_unavailable_objects_reported(self):
        with patch.object(source_binding.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout=b'', stderr=b'unavailable')):
            self.assertIsNone(source_binding.check_git(self.binding))

    def test_missing_promisor_objects_never_enable_lazy_fetch(self):
        for missing in ['commit', 'blob']:
            with self.subTest(missing=missing):
                calls = []
                responses = iter([SimpleNamespace(returncode=128, stdout=b'', stderr=b'missing promisor object')]
                                 if missing == 'commit' else
                                 [SimpleNamespace(returncode=0, stdout=b'commit\n'),
                                  SimpleNamespace(returncode=0, stdout=self.first_tree_entry()),
                                  SimpleNamespace(returncode=128, stdout=b'', stderr=b'missing promisor object')])
                def promisor_read(command, **kwargs):
                    calls.append(command)
                    self.assertEqual(kwargs['env'].get('GIT_NO_LAZY_FETCH'), '1',
                                     'missing promisor object would trigger implicit fetching')
                    return next(responses)
                with patch.dict('os.environ', {'GIT_NO_LAZY_FETCH': '0'}):
                    with patch.object(source_binding.shutil, 'which', return_value='git'):
                        with patch.object(source_binding.subprocess, 'run', side_effect=promisor_read):
                            if missing == 'commit':
                                self.assertIsNone(source_binding.check_git(self.binding))
                            else:
                                with self.assertRaisesRegex(ValueError, '^public Git source bytes/SHA mismatch$'):
                                    source_binding.check_git(self.binding)
                self.assertEqual(len(calls), 1 if missing == 'commit' else 3)

    def test_empty_git_repository_reports_unavailable(self):
        git = source_binding.shutil.which('git')
        if git is None:
            self.skipTest('Git unavailable')
        env = {key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
        with tempfile.TemporaryDirectory() as repository:
            subprocess.run([git, 'init', '--bare', repository], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            self.assertIsNone(source_binding.check_git(self.binding, repository))

    def first_tree_entry(self):
        row = self.binding['files'][0]
        return f"{row['mode']} blob {row['git_blob']}\t{row['path']}\0".encode()

    def check_mock_git(self, responses):
        with patch.object(source_binding.shutil, 'which', return_value='git'):
            with patch.object(source_binding.subprocess, 'run', side_effect=responses):
                source_binding.check_git(self.binding)

    def test_missing_tree_after_available_commit_refused(self):
        responses = [SimpleNamespace(returncode=0, stdout=b'commit\n'),
                     SimpleNamespace(returncode=1, stdout=b'')]
        with self.assertRaisesRegex(ValueError, 'source tree read'):
            self.check_mock_git(responses)

    def test_missing_blob_after_available_commit_refused(self):
        responses = [SimpleNamespace(returncode=0, stdout=b'commit\n'),
                     SimpleNamespace(returncode=0, stdout=self.first_tree_entry()),
                     SimpleNamespace(returncode=1, stdout=b'')]
        with self.assertRaisesRegex(ValueError, 'bytes/SHA mismatch'):
            self.check_mock_git(responses)

    def test_public_object_type_refused(self):
        with patch.object(source_binding.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=b'tag\n', stderr=b'')):
            with self.assertRaisesRegex(ValueError, 'not a commit'):
                source_binding.check_git(self.binding)

    def test_git_byte_corruption_refused(self):
        row = self.binding['files'][0]
        responses = [SimpleNamespace(returncode=0, stdout=b'commit\n'),
                     SimpleNamespace(returncode=0, stdout=self.first_tree_entry()),
                     SimpleNamespace(returncode=0, stdout=b'x'*row['bytes'])]
        with self.assertRaisesRegex(ValueError, 'bytes/SHA mismatch'):
            self.check_mock_git(responses)

    def test_git_tree_alias_refused(self):
        entry = self.first_tree_entry().replace(self.binding['files'][0]['git_blob'].encode(), b'refs/heads/main')
        responses = [SimpleNamespace(returncode=0, stdout=b'commit\n'),
                     SimpleNamespace(returncode=0, stdout=entry)]
        with self.assertRaisesRegex(ValueError, 'mode/path/blob mismatch'):
            self.check_mock_git(responses)


if __name__ == '__main__':
    unittest.main()
