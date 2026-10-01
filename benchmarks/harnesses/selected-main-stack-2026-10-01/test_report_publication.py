"""Pure report failure/resume fixtures; no models, native processes or PG."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_templates as fixtures


class ReportPublication(unittest.TestCase):
    setUp = fixtures.Templates.setUp
    tearDown = fixtures.Templates.tearDown

    def publish(self, result=None):
        result = {'fixture': True} if result is None else result
        with patch.object(self.analyzer, 'write_tables', side_effect=lambda path, value:path.write_text('complete fixture table\n')):
            self.analyzer.publish_reports(result)

    def test_render_failure_exposes_no_final_file_and_retry_succeeds(self):
        def failed(path, result):
            path.write_text('partial staged table')
            raise RuntimeError('mock interrupted rendering')
        with patch.object(self.analyzer, 'write_tables', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'interrupted rendering'):
                self.analyzer.publish_reports({'fixture': True})
        self.assertFalse((self.work / 'metrics.json').exists())
        self.assertFalse((self.work / 'tables.md').exists())
        self.publish()
        self.assertEqual(json.loads((self.work / 'metrics.json').read_text()), {'fixture': True})

    def test_interruption_after_first_publication_resumes_matching_partial_report(self):
        original = self.analyzer.os.link
        def publish(source, target):
            if target.name == 'tables.md':
                raise RuntimeError('mock publication interrupted')
            return original(source, target)
        with patch.object(self.analyzer.os, 'link', side_effect=publish):
            with self.assertRaisesRegex(RuntimeError, 'publication interrupted'):
                self.publish()
        retained = (self.work / 'metrics.json').read_bytes()
        self.assertFalse((self.work / 'tables.md').exists())
        self.publish()
        self.assertEqual((self.work / 'metrics.json').read_bytes(), retained)
        self.assertEqual((self.work / 'tables.md').read_text(), 'complete fixture table\n')
        self.publish() # Exact existing pair is idempotent.

    def test_conflicting_or_symlink_output_is_preserved_before_any_publication(self):
        for name in ['metrics.json', 'tables.md']:
            with self.subTest(name=name):
                target = self.work / name
                target.write_text('retained conflicting bytes')
                with self.assertRaisesRegex(ValueError, 'conflicting existing report'):
                    self.publish()
                self.assertEqual(target.read_text(), 'retained conflicting bytes')
                other = self.work / ('tables.md' if name == 'metrics.json' else 'metrics.json')
                self.assertFalse(other.exists())
                target.unlink()
        target = self.work / 'metrics.json'
        target.symlink_to(self.work / 'missing target')
        with self.assertRaisesRegex(ValueError, 'conflicting existing report'):
            self.publish()
        self.assertTrue(target.is_symlink())
        self.assertFalse((self.work / 'tables.md').exists())
