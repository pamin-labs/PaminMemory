"""Raw retained log bytes, not newline-normalized text, must match the receipt.

Kept outside test_verify.py so the retained 69-test suite-scan audit is unchanged.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import verify


class RawLogBytes(unittest.TestCase):
    def test_retained_logs_accepted(self):
        self.assertTrue(verify.verify(json.loads((verify.ROOT/'evidence.json').read_text())))

    def test_newline_rewritten_log_rejected(self):
        evidence = json.loads((verify.ROOT/'evidence.json').read_text())
        for arm in ['pure-open', 'vector']:
            for newline in [b'\r\n', b'\r']:
                with self.subTest(arm=arm, newline=newline), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    for name in ['pure-open.log', 'vector.log', 'public-source-binding.json']:
                        shutil.copyfile(verify.ROOT/name, root/name)
                    path = root/(arm+'.log')
                    # Text-mode reads normalize these back to the retained text.
                    path.write_bytes(path.read_bytes().replace(b'\n', newline))
                    self.assertEqual(path.read_text(), (verify.ROOT/(arm+'.log')).read_text())
                    with patch.object(verify, 'ROOT', root):
                        with self.assertRaisesRegex(ValueError, '^log hash$'):
                            verify.verify(evidence)


if __name__ == '__main__':
    unittest.main()
