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


def verify_cluster_inference(queries, scored, precision):
    import collections
    import numpy as np

    expected = json.loads((ROOT / "cluster-inference.json").read_text())
    assert precision["inference_scope"] == expected, "published inference scope diverged"
    assert expected["queries"] == len(queries) == 1190, "published query count differs from raw rows"
    clusters = collections.defaultdict(list)
    for index, query in enumerate(queries):
        # Sentence keys are language:paragraph:sentence; languages have
        # different sentence segmentation. English identifies the shared paragraph.
        keys = {":".join(x.split(":")[:2]) for group in query["relevance"].values()
                for x in group["relevant"] if x.startswith("en:")}
        assert len(keys) == 1
        clusters[next(iter(keys))].append(index)
    assert len(clusters) == expected["clusters"] == 240
    groups = ["cross_lingual", "same_language"]
    deltas = np.array([[b - a for a, b in zip(scored[(g, "before")], scored[(g, "after")])]
                       for g in groups])
    sums = np.array([[deltas[j, indices].sum() for indices in clusters.values()] for j in range(2)])
    counts = np.array([len(indices) for indices in clusters.values()])
    rng = np.random.Generator(np.random.PCG64(expected["seed"]))
    hits = np.zeros(2, dtype=np.int64)
    observed = np.abs(sums.sum(axis=1))
    bootstraps = []
    assert expected["draws"] == 100000
    for _ in range(100):
        signs = rng.integers(0, 2, size=(1000, len(clusters)), dtype=np.int8) * 2 - 1
        hits += (np.abs(signs @ sums.T) >= observed - 1e-14).sum(axis=0)
        sample = rng.integers(0, len(clusters), size=(1000, len(clusters)))
        bootstraps.append(sums[:, sample].sum(axis=2) / counts[sample].sum(axis=1))
    p = (hits + 1) / 100001
    adjusted = np.empty(2)
    previous = 0
    for rank, index in enumerate(np.argsort(p)):
        previous = max(previous, min(1, (2 - rank) * p[index]))
        adjusted[index] = previous
    bootstrap = np.concatenate(bootstraps, axis=1)
    for index, group in enumerate(groups):
        reference = expected["groups"][group]
        for key, value in reference.items():
            assert precision["groups"][group][key] == value, "published inference field diverged"
        close(float(p[index]), reference["paragraph_cluster_sign_flip_p"])
        close(float(adjusted[index]), reference["holm_p"])
        for actual, target in zip(np.quantile(bootstrap[index], [.025, .975]), reference["cluster_bootstrap_delta_95_ci"]):
            close(float(actual), target)


def verify():
    precision = json.loads((ROOT / "precision-summary.json").read_text())
    query_ids = None
    paired_gold = None
    scored_groups = {}
    baseline_queries = None
    for file, side in [("xquad-bge.jsonl", "before"), ("xquad-dual.jsonl", "after")]:
        raw = rows(file)
        queries = [r for r in raw if r["kind"] == "query"]
        assert len(queries) == 1190
        ids = [(r["id"], r["language"]) for r in queries]
        assert len(set(ids)) == len(ids)
        if query_ids is not None:
            assert ids == query_ids
        query_ids = ids
        gold = [{g: {k: sorted(set(values)) for k, values in labels.items()}
                 for g, labels in q["relevance"].items()} for q in queries]
        if paired_gold is not None:
            assert gold == paired_gold, "paired gold labels changed"
        paired_gold = gold
        if side == "before":
            baseline_queries = queries
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
            scored_groups[(group, side)] = scores
            close(statistics.mean(scores), summary[side])
            close(statistics.mean(recalls), summary["recall50_" + side])
    verify_cluster_inference(baseline_queries, scored_groups, precision)
    costs = json.loads((ROOT / "cost-summary.json").read_text())
    expected = {"main-cpu", "new-cpu", "dual-cpu", "main-auto", "new-auto", "main-auto-repeat", "new-auto-persist-hit"}
    assert set(costs) == expected, "retained timing arm set is incomplete"
    assert sum(len(a["blocks"]) for a in costs.values()) == 21
    for name, arm in costs.items():
        assert len(arm["blocks"]) == 3
        assert {b["block"] for b in arm["blocks"]} == {0, 1, 2}
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
