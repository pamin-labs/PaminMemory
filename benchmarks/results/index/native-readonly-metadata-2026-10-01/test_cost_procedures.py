"""Tiny source/procedure checks: no cost timing or product work."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import probe_verifier_cost as probe
import probe_suite_scans as scans


class CostProcedures(unittest.TestCase):
    def test_retained_before_source_loads_without_git(self):
        cost = json.loads((probe.ROOT/'verifier-cost.json').read_text())
        with patch('subprocess.run', side_effect=AssertionError('Git subprocess forbidden')):
            before = probe.load_before(cost)
        self.assertTrue(callable(before['verify']))
        self.assertEqual(before['crc32c'](b'123456789'), 0x58e3fa20)

    def test_changed_comparator_source_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'before-verify.py.in').write_bytes(b'conflicting source')
            with patch.object(probe, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, '^before source identity differs$'):
                    probe.load_before({'before_verify_sha256': '0'*64})

    def test_default_component_and_suite_plans_do_not_execute(self):
        with patch.object(sys, 'argv', ['probe']), patch('builtins.print'):
            with patch.object(probe, 'load_before', side_effect=AssertionError('comparator execution forbidden')):
                probe.main()
            with patch.object(scans, 'collect', side_effect=AssertionError('suite execution forbidden')):
                scans.main()

    def test_suite_scalar_counts_have_reproducible_producer(self):
        data = b'x'*512
        class SameContent(unittest.TestCase):
            def runTest(self):
                scans.verify.crc32c(data)
                scans.verify.crc32c(bytearray(data))
        audit = scans.collect(unittest.TestSuite([SameContent()]), hashlib.sha256(data).hexdigest())
        self.assertEqual(audit, {'tests': 1, 'large_scalar_calls': 1, 'large_scalar_bytes': 512,
                                'distinct_large_scalar_contents': 1, 'original_table_scalar_scans': 1})
