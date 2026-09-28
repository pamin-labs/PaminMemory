"""Copy product-path XQuAD-R scores from the committed harness output."""

import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
log = root / "xquad-r-current-run.log"
text = log.read_text()
assert "test search_reaches_across_languages ... ok" in text
assert re.findall(r"^\s*the shipped search path, (\w+)\s*$", text, re.M) == ["accuracy"]
commits = re.findall(r"^\s*benchmark code commit: ([0-9a-f]{40})\s*$", text, re.M)
assert len(commits) == 1
assert re.search(r"^\s*index passage: named\s*$", text, re.M)
assert "reranker device: cpu; export: onnx/model_int8.onnx" in text
assert re.findall(r"^\s*indexed documents: (\d+)\s*$", text, re.M) == ["13014"]
rows = re.findall(r"^\s*(cross_lingual|same_language)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", text, re.M)
assert len(rows) == 2 and {row[0] for row in rows} == {"cross_lingual", "same_language"}
assert all(int(row[1]) == 1190 for row in rows)

path = root / "summary-current-xquad.json"
summary = json.loads(path.read_text())
manifest = root / summary["corpus"]["input_sha256_manifest"]
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == summary["corpus"]["input_manifest_sha256"]
profile = root / "xquad-r-index-profile.txt"
assert profile.read_text().splitlines() == [
    "gpahal/bge-m3-onnx-int8", "topic", "memory", "named", "reversed-keys"
]
models = root / "xquad-r-model-artifacts.json"
models_hash = hashlib.sha256(models.read_bytes()).hexdigest()
assert models_hash == "d8616e71f9a49eb88b19066448d109b72acd4beb01c0b67d43a03700bb26c71a"
summary["status"] = "measured product-path run; latency shared a busy machine and is not reported"
summary["path"]["reranker_device"] = "cpu"
summary["path"]["reranker_export"] = "onnx/model_int8.onnx"
summary["source_run"] = {
    "log": log.name,
    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "code_commit": commits[0],
    "index_profile": profile.name,
    "index_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "index_project": "xquad-accuracy-24ad7f1862182925",
    "index_profile_original_mtime": "2026-09-26 16:53:10 +1000",
    "model_artifacts": models.name,
    "model_artifacts_sha256": models_hash,
}
summary["metrics"] = {
    ("cross_language" if group == "cross_lingual" else "same_language"): {
        "ndcg_at_10": float(ndcg),
        "recall_at_50": float(recall),
    }
    for group, _, ndcg, recall in rows
}
path.write_text(json.dumps(summary, indent=2) + "\n")
