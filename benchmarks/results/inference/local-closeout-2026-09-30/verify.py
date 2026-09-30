import hashlib, json, subprocess, sys
from pathlib import Path
from summarize import compute
root = Path(__file__).parent
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], name
recomputed = compute(root)
retained = json.loads((root / "summary.json").read_text())
for key in ("arms", "comparisons"):
    assert recomputed[key] == retained[key], key
assert subprocess.check_output([sys.executable, str(root / "tables.py")], text=True).rstrip() + "\n" == (root / "README.md").read_text()
print("Verified 432 raw rows, resource aggregates, paired statistics, and published tables without writing evidence.")
