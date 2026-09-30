"""Check that the archived CHANNELS grids cover every paired query."""

import json
import statistics
from pathlib import Path


root = Path(__file__).resolve().parent
for name, groups, count in [
    ("xquad", ("cross_lingual", "same_language"), 1190),
    ("musique", ("2hop",), 1000),
]:
    evidence = json.loads((root / f"{name}-paired.json").read_text())
    assert len(evidence["variants"]) == 37
    assert len({row["label"] for row in evidence["variants"]}) == 37
    assert all(evidence["whole"][group]["queries"] == count for group in groups)
    assert all(len(evidence["whole"][group]["per_query"]) == count for group in groups)
    for variant in evidence["variants"]:
        assert all(len(variant["scores"][group]["per_query"]) == count for group in groups)
    macro = statistics.mean(
        evidence["whole"][group]["ndcg"] / count for group in groups
    )
    assert f"{macro:.4f}" in (root / f"{name}-stdout.log").read_text()
    print(f"{name}: {count} paired queries, 37 settings, shipped macro nDCG@10 {macro:.4f}")

assert json.loads((root / "xquad-paired.json").read_text())["complete"]
