"""Check the cached models against the archived XQuAD-R run."""

import hashlib
import json
import sys
from pathlib import Path

workspace_models = Path(sys.argv[1])
hub_cache = workspace_models
models = json.loads((Path(__file__).parent / "xquad-r-model-artifacts.json").read_text())["models"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for model in models:
    directory = hub_cache / ("models--" + model["repository"].replace("/", "--"))
    assert (directory / "refs/main").read_text().strip() == model["revision"]
    snapshot = directory / "snapshots" / model["revision"]
    for name, expected in model["files_sha256"].items():
        assert digest(snapshot / name) == expected, f"{model['role']}: {name} changed"

    label = model["repository"].replace("/", "--") + "--" + model["source_onnx"].replace("/", "--")
    source = workspace_models / "prepared" / (label + ".source")
    assert source.read_text().splitlines()[0] == model["source_onnx_sha256"]
    preferred = workspace_models / "prepared" / model["prepared_key_on_measured_host"]
    candidates = [preferred] + list((workspace_models / "prepared").glob("*/model.onnx"))
    assert any(
        digest(path / "model.onnx") == model["prepared_graph_sha256"]
        and digest(path / "model.onnx.data") == model["prepared_data_sha256"]
        for path in (candidate if candidate.is_dir() else candidate.parent for candidate in candidates)
        if (path / "model.onnx").is_file() and (path / "model.onnx.data").is_file()
    ), f"{model['role']}: prepared ONNX differs"
    print(f"{model['role']}: pinned revision and model files match")
