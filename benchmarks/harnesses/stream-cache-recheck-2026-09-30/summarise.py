"""Summarise the archived strict cache-hit rounds without model dependencies."""

import json
import math
import random
import statistics
from pathlib import Path


results = Path(__file__).resolve().parents[2] / "results/inference"


def summarise(filename):
    rows = json.loads((results / filename).read_text())
    rounds = {}
    for row in rows:
        assert row["cache_hit_premise"] == "both files/content/inodes/birth times unchanged"
        rounds.setdefault(row["round"], {})[row["arm"]] = row
    assert len(rows) == 60 and sorted(rounds) == list(range(1, 31))
    assert all(set(pair) == {"before", "after"} for pair in rounds.values())

    before = [rounds[i]["before"]["wall_seconds"] for i in sorted(rounds)]
    after = [rounds[i]["after"]["wall_seconds"] for i in sorted(rounds)]
    ratios = [a / b for b, a in zip(before, after)]
    differences = [a - b for b, a in zip(before, after)]
    rng = random.Random(20260930)
    observed = abs(statistics.mean(differences))
    draws = 100_000
    extreme = sum(
        abs(statistics.mean(
            difference * (1 if rng.getrandbits(1) else -1)
            for difference in differences
        )) >= observed
        for _ in range(draws)
    )
    return {
        "rounds": len(rounds),
        "before_median_seconds": statistics.median(before),
        "after_median_seconds": statistics.median(after),
        "paired_geometric_mean_ratio": math.exp(statistics.mean(map(math.log, ratios))),
        "after_faster_rounds": sum(a < b for b, a in zip(before, after)),
        "two_sided_sign_flip_p": (extreme + 1) / (draws + 1),
        "before_median_process_rss_bytes": statistics.median(
            rounds[i]["before"]["peak_process_rss_bytes"] for i in rounds
        ),
        "after_median_process_rss_bytes": statistics.median(
            rounds[i]["after"]["peak_process_rss_bytes"] for i in rounds
        ),
    }


print(json.dumps({
    "first_30_rounds": summarise("coreml-stream-cache-recheck-2026-09-30-earlier-rows.json"),
    "binary_checked_30_rounds": summarise("coreml-stream-cache-recheck-2026-09-30-rows.json"),
}, indent=2))
