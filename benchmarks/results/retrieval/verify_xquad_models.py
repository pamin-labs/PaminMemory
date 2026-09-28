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
    assert digest(preferred / "model.onnx") == model["prepared_graph_sha256"]
    assert digest(preferred / "model.onnx.data") == model["prepared_data_sha256"]
    # This is prepared::settled's default selection, not an arbitrary valid
    # copy elsewhere in the cache. A fused graph appearing later must fail
    # verification rather than silently change the executed artifact.
    selected = preferred / ("attention.onnx" if (preferred / "attention.onnx").exists() else "model.onnx")
    assert selected.name == model["selected_graph"], f"{model['role']}: graph selection changed"
    assert digest(selected) == model["selected_graph_sha256"]
    if selected.name == "model.onnx":
        assert digest(preferred / "attention.unfused") == model["unfused_decision_sha256"]
    print(f"{model['role']}: pinned revision and model files match")
