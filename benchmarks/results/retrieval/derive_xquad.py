"""Copy product-path XQuAD-R scores from the committed harness output."""

import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
assert len(sys.argv) == 2 and re.fullmatch(r"[0-9a-f]{40}", sys.argv[1])
log = root / "xquad-r-current-run.log"
text = log.read_text()
assert "test search_reaches_across_languages ... ok" in text
rows = re.findall(r"^\s*(cross_lingual|same_language)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", text, re.M)
assert len(rows) == 2 and {row[0] for row in rows} == {"cross_lingual", "same_language"}
assert all(int(row[1]) == 1190 for row in rows)

path = root / "summary-current-xquad.json"
summary = json.loads(path.read_text())
manifest = root / summary["corpus"]["input_sha256_manifest"]
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == summary["corpus"]["input_manifest_sha256"]
summary["status"] = "measured product-path run; latency shared a busy machine and is not reported"
summary["source_run"] = {
    "log": log.name,
    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    "code_commit": sys.argv[1],
}
summary["metrics"] = {
    ("cross_language" if group == "cross_lingual" else "same_language"): {
        "ndcg_at_10": float(ndcg),
        "recall_at_50": float(recall),
    }
    for group, _, ndcg, recall in rows
}
path.write_text(json.dumps(summary, indent=2) + "\n")
