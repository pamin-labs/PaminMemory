"""One reversible fixture for the offline CoreML cache retirement boundary."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("retire_coreml_v1.py")


class Retirement(unittest.TestCase):
    def test_only_v1_is_removed_after_explicit_offline_assertion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            keyed = models / "prepared" / ("a" * 32)
            v1 = keyed / "coreml-all-v1"
            v2 = keyed / "coreml-all-v2"
            v1.mkdir(parents=True)
            v2.mkdir()
            (v1 / "compiled").write_bytes(b"old")
            (v2 / "compiled").write_bytes(b"live")
            outside = root / "outside"
            outside.mkdir()
            (outside / "keep").write_bytes(b"untouched")
            (models / "prepared" / ("b" * 32)).symlink_to(outside)

            def run(*args):
                return subprocess.run(
                    [sys.executable, str(SCRIPT), str(models), *args],
                    capture_output=True,
                    text=True,
                )

            dry = run()
            self.assertEqual(dry.returncode, 0)
            self.assertIn("3 logical bytes", dry.stdout)
            self.assertTrue(v1.exists())
            self.assertNotEqual(run("--apply").returncode, 0)
            self.assertTrue(v1.exists())
            self.assertEqual(run("--apply", "--all-processes-stopped").returncode, 0)
            self.assertFalse(v1.exists())
            self.assertEqual((v2 / "compiled").read_bytes(), b"live")
            self.assertEqual((outside / "keep").read_bytes(), b"untouched")


if __name__ == "__main__":
    unittest.main()
