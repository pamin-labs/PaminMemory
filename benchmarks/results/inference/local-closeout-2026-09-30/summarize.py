import gzip, json, math, random, re, statistics
from pathlib import Path

def compute(root):
    arms = {}; medians = {}
    for tier in ("accurate", "fast"):
        for device in ("all", "gpu", "cpu"):
            runs = []; resources = []
            for i in range(3):
                name = f"{tier}-{device}-{i}"
                run = json.loads(gzip.decompress((root / "rows" / (name + ".json.gz")).read_bytes()))
                assert [r["query"] for r in run] == list(range(0, 1190, 50))
                assert all(r["scored"] == r["offered"] > 0 and r["seconds"] > 0 for r in run)
                runs.append(run)
                text = (root / "rows" / (name + ".resource")).read_text()
                m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
                assert m
                resources.append((float(m[1]), float(m[2]) + float(m[3]), int(re.search(r"(\d+)\s+maximum resident set size", text)[1])))
                attestation = (root / "rows" / (name + ".attestation.txt")).read_text()
                expected = "cpu" if device == "cpu" else "coreml"
                assert f'device="{expected}"' in attestation and "24/24" in attestation
            median = [statistics.median(run[j]["seconds"] for run in runs) for j in range(24)]
            medians[f"{tier}-{device}"] = median
            ordered = sorted(median)
            arms[f"{tier}-{device}"] = {
                "p50_seconds": ordered[11], "p95_seconds": ordered[22],
                "median_24_search_seconds": statistics.median(sum(r["seconds"] for r in run) for run in runs),
                "median_process_cpu_seconds": statistics.median(r[1] for r in resources),
                "median_peak_process_rss_bytes": statistics.median(r[2] for r in resources),
                "median_process_wall_seconds": statistics.median(r[0] for r in resources),
                "ndcg": {g: statistics.mean(r["scores"][g]["ndcg"] for run in runs for r in run) for g in ("same_language", "cross_lingual")},
            }
    comparisons = {}
    for tier, reference in [("accurate", "all"), ("fast", "cpu")]:
        for candidate in ("all", "gpu", "cpu"):
            if candidate == reference: continue
            differences = [math.log(b / a) for a, b in zip(medians[f"{tier}-{reference}"], medians[f"{tier}-{candidate}"])]
            observed = abs(statistics.mean(differences)); rng = random.Random(0)
            count = sum(abs(sum(v * (1 if rng.getrandbits(1) else -1) for v in differences) / 24) >= observed for _ in range(19999))
            p = (count + 1) / 20000
            comparisons[f"{tier}-{candidate}-vs-{reference}"] = {"geometric_mean_time_ratio": math.exp(statistics.mean(differences)), "two_sided_signflip_p": p, "family_bonferroni_p": min(1, p * 4), "query_count": 24}
    return {"arms": arms, "comparisons": comparisons}

if __name__ == "__main__":
    print(json.dumps(compute(Path(__file__).parent), indent=2))
