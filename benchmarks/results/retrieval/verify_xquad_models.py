"""Check pinned source files and the graphs actually loaded by an XQuAD run.

Usage: verify_xquad_models.py EVAL_HOME/models xquad-run.log
Prepared hashes describe the archived host. Other runtimes/CPUs may require
fresh evidence; matching an unused old prepared directory is never sufficient.
"""

import hashlib
import json
import sys
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def loaded_graphs(run_log, workspace_models):
    graphs = set()
    prepared = (workspace_models / "prepared").resolve()
    decoder = json.JSONDecoder()
    for line in run_log.read_text().splitlines():
        if "loaded ONNX graph" not in line or "model_graph=" not in line:
            continue
        name, _ = decoder.raw_decode(line.split("model_graph=", 1)[1])
        assert isinstance(name, str)
        path = Path(name).resolve()
        relative = path.relative_to(prepared)
        assert len(relative.parts) == 2, "loaded graph is not a prepared model"
        graphs.add(path)
    assert graphs, "run log does not record any successfully loaded model graph"
    return graphs


def source_record(prepared, model):
    """The lines of the record of the copy made from this model's pinned revision.

    A record names its revision on a `revision <repository>@<commit>` line, under a file name that carries a hash of
    it (`<label>.<tag>.source`); an older build wrote one `<label>.source` with no revision line.
    """
    revision = "revision " + model["repository"] + "@" + model["revision"]
    records = [path.read_text().splitlines() for path in sorted(prepared.glob("*.source"))]
    matching = [lines for lines in records if revision in lines]
    if not matching:
        label = model["repository"].replace("/", "--") + "--" + model["source_onnx"].replace("/", "--")
        legacy = prepared / (label + ".source")
        assert legacy.exists(), f"{model['role']}: no prepared source record names {revision}"
        return legacy.read_text().splitlines()
    assert len(matching) == 1, f"{model['role']}: several prepared source records name {revision}"
    return matching[0]


def default_graph(copy, selected):
    """The graph the product loads from `copy`: the fused one if it exists, else ONNX Runtime's; its last-layer
    trim when the passes are one unpadded row (the run's selection says whether they were); and the word-embedding
    split of whichever that is, when it exists."""
    base = "attention" if (copy / "attention.onnx").exists() else "model"
    if base == "attention" and selected.name.startswith("attention.cls1.") and (copy / "attention.cls1.onnx").exists():
        base = "attention.cls1"
    split = copy / (base + ".split1.onnx")
    return split if split.exists() else copy / (base + ".onnx")


def verify(workspace_models, run_log, models):
    observed = loaded_graphs(run_log, workspace_models)
    observed_hashes = {path: digest(path) for path in observed}
    matched = set()
    for model in models:
        directory = workspace_models / ("models--" + model["repository"].replace("/", "--"))
        assert (directory / "refs/main").read_text().strip() == model["revision"]
        snapshot = directory / "snapshots" / model["revision"]
        for name, expected in model["files_sha256"].items():
            assert digest(snapshot / name) == expected, f"{model['role']}: {name} changed"

        record = source_record(workspace_models / "prepared", model)
        assert len(record) >= 4 and record[0] and record[1].isdigit() and record[2].isdigit(), (
            "prepared source record lacks a digest; rerun prepare_pinned_xquad_models")
        assert record[3] == "sha256 " + model["source_onnx_sha256"], f"{model['role']}: source digest changed"
        candidates = [path for path, actual in observed_hashes.items()
                      if actual == model["selected_graph_sha256"]]
        assert len(candidates) == 1, f"{model['role']}: exactly one loaded graph must match the archived artifact"
        selected = candidates[0]
        preferred = selected.parent
        assert selected.name == model["selected_graph"], f"{model['role']}: graph selection changed"
        assert selected == default_graph(preferred, selected), (
            f"{model['role']}: run did not use the default graph selection")
        assert digest(preferred / "model.onnx") == model["prepared_graph_sha256"]
        assert digest(preferred / "model.onnx.data") == model["prepared_data_sha256"]
        if selected.name in ("model.onnx", "model.split1.onnx"):
            assert digest(preferred / "attention.unfused") == model["unfused_decision_sha256"]
        matched.add(selected)
        print(f"{model['role']}: pinned revision and loaded graph {selected} match")
    assert matched == observed, "run loaded additional unverified model graphs"


if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: verify_xquad_models.py EVAL_HOME/models xquad-run.log"
    models = json.loads((Path(__file__).parent / "xquad-r-model-artifacts.json").read_text())["models"]
    verify(Path(sys.argv[1]), Path(sys.argv[2]), models)
