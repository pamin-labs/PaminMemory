"""Assert retained prototype precision/cost summaries agree with raw rows.

This does not run a benchmark or generate optimization numbers.
"""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).parent / "results/compute/prototype-closeout-2026-10-01"


def rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def verify():
    precision = json.loads((ROOT / "precision-summary.json").read_text())
    query_ids = None
    for file, side in [("xquad-bge.jsonl", "before"), ("xquad-dual.jsonl", "after")]:
        raw = rows(file)
        queries = [r for r in raw if r["kind"] == "query"]
        assert len(queries) == 1190
        ids = [(r["id"], r["language"]) for r in queries]
        assert len(set(ids)) == len(ids)
        if query_ids is not None:
            assert ids == query_ids
        query_ids = ids
        for group, summary in precision["groups"].items():
            scores, recalls = [], []
            for query in queries:
                gold = query["relevance"][group]
                relevant = set(gold["relevant"])
                ranking = [x for x in query["ranked"] if x not in set(gold["exclude"])]
                dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranking[:10]) if x in relevant)
                ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
                scores.append(dcg / ideal)
                recalls.append(len(set(ranking[:50]) & relevant) / len(relevant))
            close(statistics.mean(scores), summary[side])
            close(statistics.mean(recalls), summary["recall50_" + side])
    costs = json.loads((ROOT / "cost-summary.json").read_text())
    for name, arm in costs.items():
        assert len(arm["blocks"]) == 3
        for block in arm["blocks"]:
            raw = rows(f"{name}-{block['block']}.jsonl")
            queries = [r for r in raw if r["kind"] == "query"]
            assert len(queries) == 66 and len({r["id"] for r in queries}) == 66
            assert raw[0]["documents"] == 13014 and all(r["ranked"] for r in queries)
            work = [r for r in raw if r["kind"] == "work"]
            assert len(work) == 1 and work[0]["new_scores"] > 0 and work[0]["offered"] > 0
            values = sorted(r["seconds"] * 1000 for r in queries)
            close(statistics.median(values), block["p50_ms"])
            close(values[math.ceil(0.95 * len(values)) - 1], block["p95_ms"])
            rss = max([raw[0]["warmed_rss_bytes"]] + [r["rss_bytes"] for r in queries])
            assert rss == block["max_sampled_warm_rss_bytes"]
        for cell, median in [("p50_ms", "median_p50_ms"), ("p95_ms", "median_p95_ms"), ("max_sampled_warm_rss_bytes", "median_max_sampled_warm_rss_bytes")]:
            close(statistics.median(b[cell] for b in arm["blocks"]), arm[median])
    print("Verified paired precision and 21 independent cost blocks against retained raw rows.")


if __name__ == "__main__":
    verify()
