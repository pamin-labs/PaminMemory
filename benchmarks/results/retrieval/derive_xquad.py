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
reported_machine = summary["machine"].get("reported_separately", summary["machine"])
summary["machine"] = {
    "verified_from_source_log": False,
    "provenance": "manual host observation; the archived harness did not emit machine metadata",
    "reported_separately": reported_machine,
}
manifest = root / summary["corpus"]["input_sha256_manifest"]
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == summary["corpus"]["input_manifest_sha256"]
profile = root / "xquad-r-index-profile.txt"
assert profile.read_text().splitlines() == [
    "gpahal/bge-m3-onnx-int8", "topic", "memory", "named", "reversed-keys"
]
models = root / "xquad-r-model-artifacts.json"
models_hash = hashlib.sha256(models.read_bytes()).hexdigest()
assert models_hash == "75ff42b82260ba95ea8e26bdbcd7f4f52da05d0819625e2cbab3602643c2d4ee"
embedding_ids = re.findall(r"^\s*index embedding identity: ([0-9a-f]{64})\s*$", text, re.M)
projects = re.findall(r"^\s*index project: (\S+)\s*$", text, re.M)
completeness = re.findall(r"^\s*vector index completeness: ([\d.]+)\s*$", text, re.M)
assert len(embedding_ids) <= 1 and len(projects) <= 1 and len(completeness) <= 1
if embedding_ids:
    model = next(m for m in json.loads(models.read_text())["models"] if m["role"] == "embedder")
    identity = {key: model[key] for key in (
        "repository", "revision", "source_onnx_sha256", "files_sha256"
    )}
    expected_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert embedding_ids == [expected_id]
    assert projects == [f"xquad-accuracy-{summary['corpus']['passage_fingerprint']}-{expected_id[:16]}"]
    assert completeness == ["1.0000"]
if completeness:
    assert float(completeness[0]) == 1.0
summary["status"] = "measured product-path run; latency shared a busy machine and is not reported"
summary["path"]["reranker_device"] = "cpu"
summary["path"]["reranker_export"] = "onnx/model_int8.onnx"
summary["source_run"] = {
    "log": log.name,
    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "code_commit": commits[0],
    "index_profile": profile.name,
    "index_profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    "index_project": projects[0] if projects else "xquad-accuracy-24ad7f1862182925",
    "index_embedding_revision_verified": bool(embedding_ids),
    "index_completeness_asserted_in_run": bool(completeness),
    "index_provenance": (
        "revision-bound project; pinned embedding artifacts and complete vector graph checked before scoring"
        if embedding_ids else
        "the retained index records the repository but not the embedding revision; a fresh revision-bound rerun is pending"
    ),
    "model_artifacts": models.name,
    "model_artifacts_sha256": models_hash,
}
if not embedding_ids:
    summary["source_run"]["index_profile_original_mtime"] = "2026-09-26 16:53:10 +1000"
summary["metrics"] = {
    ("cross_language" if group == "cross_lingual" else "same_language"): {
        "ndcg_at_10": float(ndcg),
        "recall_at_50": float(recall),
    }
    for group, _, ndcg, recall in rows
}
path.write_text(json.dumps(summary, indent=2) + "\n")
