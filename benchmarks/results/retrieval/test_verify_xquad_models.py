"""The run's actual prepared directory wins over an unused historical copy."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_xquad_models import verify


class LoadedGraphTest(unittest.TestCase):
    def test_current_runtime_path_is_required_even_with_an_unused_valid_copy(self):
        with tempfile.TemporaryDirectory() as home:
            models = Path(home) / "models with spaces"
            hub = models / "models--someone--model"
            (hub / "refs").mkdir(parents=True)
            (hub / "refs/main").write_text("revision")
            snapshot = hub / "snapshots/revision"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_bytes(b"config")
            prepared = models / "prepared"
            prepared.mkdir()
            (prepared / "someone--model--model.onnx.source").write_text("source-hash\n1\n2\n")
            files = {"model.onnx": b"graph", "model.onnx.data": b"weights", "attention.unfused": b"decision"}
            current = prepared / "current-runtime-key"
            unused = prepared / "unused-old-key"
            for directory in [current, unused]:
                directory.mkdir()
                for name, data in files.items():
                    (directory / name).write_bytes(data)
            sha = lambda data: hashlib.sha256(data).hexdigest()
            manifest = [{"role": "reranker", "repository": "someone/model", "revision": "revision",
                         "source_onnx": "model.onnx", "source_onnx_sha256": "source-hash",
                         "files_sha256": {"config.json": sha(b"config")},
                         "prepared_key_on_measured_host": "missing-historical-key",
                         "selected_graph": "model.onnx", "selected_graph_sha256": sha(b"graph"),
                         "prepared_graph_sha256": sha(b"graph"), "prepared_data_sha256": sha(b"weights"),
                         "unfused_decision_sha256": sha(b"decision")}]
            log = Path(home) / "run.log"
            log.write_text("INFO loaded ONNX graph model_graph=" + json.dumps(str(current / "model.onnx")) + "\n")
            verify(models, log, manifest)
            (current / "model.onnx").write_bytes(b"changed graph")
            with self.assertRaisesRegex(AssertionError, "loaded graph must match"):
                verify(models, log, manifest)


if __name__ == "__main__":
    unittest.main()
