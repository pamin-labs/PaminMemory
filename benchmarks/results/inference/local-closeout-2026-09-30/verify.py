import hashlib, json, subprocess, sys
from pathlib import Path
from summarize import compute
from fp64 import compute_fp64
from profiling import compute_profiling
import math
root = Path(__file__).parent
manifest = json.loads((root / "manifest.json").read_text())
for name, expected in manifest["files"].items():
    data = (root / name).read_bytes()
    assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], name
recomputed = compute(root)
retained = json.loads((root / "summary.json").read_text())
for key in ("arms", "comparisons"):
    assert recomputed[key] == retained[key], key
def equal(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        for k in left: equal(left[k], right[k])
    elif isinstance(left, float):
        assert math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
    else: assert left == right
equal(compute_fp64(root), retained["fp64"])
equal(compute_profiling(root), retained["profiling"])
assert subprocess.check_output([sys.executable, str(root / "tables.py")], text=True).rstrip() + "\n" == (root / "README.md").read_text()
print("Verified 432 raw rows, resource aggregates, paired statistics, all 5916 FP64 rows against judgements/oracles, and published tables without writing evidence.")

if "--miracl-dir" in sys.argv:
    dataset_dir = Path(sys.argv[sys.argv.index("--miracl-dir") + 1])
    for name, expected in manifest["fp64"]["miracl"]["dataset"]["files"].items():
        path = dataset_dir / name
        assert path.stat().st_size == expected["bytes"], name
        with path.open("rb") as stream: assert hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"], name
    print("Verified exact cached MIRACL inputs for rerun.")
