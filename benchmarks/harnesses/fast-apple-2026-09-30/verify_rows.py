"""Verify that the archived CoreML suffix completes the rejected prefix."""

import json
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[2] / "results/inference"


def rows(suffix):
    return json.loads(
        (RESULTS / f"fast-apple-backend-2026-09-30-{suffix}.json").read_text()
    )


cpu = rows("cpu-rows")
prefix = rows("coreml-rejected-prefix")
resume = rows("coreml-resume-rows")
merged = rows("coreml-rows")
assert cpu["complete"] and merged["complete"]
assert len(cpu["rows"]) == len(merged["rows"]) == 1190
assert not prefix["complete"] and len(prefix["rows"]) == 770
assert resume["complete"] and len(resume["rows"]) == 430
for index in range(10):
    old, new = prefix["rows"][760 + index], resume["rows"][index]
    assert all(old[field] == new[field] for field in
               ("query_id", "ranking", "cross_ndcg", "same_ndcg"))
for index, (observed, expected) in enumerate(zip(
    merged["rows"], prefix["rows"][:760] + resume["rows"]
)):
    assert observed == {**expected, "index": index}
    assert observed["query_id"] == cpu["rows"][index]["query_id"]
print("1190 paired queries and 10 identical overlap rankings verified")
