import gzip, json, math, statistics

def read(root, name):
    return json.loads(gzip.decompress((root / "fp64" / name).read_bytes()))

def ndcg(ranked, relevant):
    discount = lambda rank: 1 / math.log2(rank + 2)
    ideal = sum(discount(i) for i in range(min(10, len(relevant))))
    return sum(discount(i) for i, topic in enumerate(ranked[:10]) if topic in relevant) / ideal if ideal else 0.0

def compute_fp64(root):
    gold = read(root, "judgements.json.gz")
    result = {}
    rows = read(root, "50k-results.json.gz")
    assert len(rows) == 900
    baseline = {r["query"]: r for r in rows if r["mode"] == 0 and r["round"] == 0}
    assert set(baseline) == set(range(100))
    result["50k"] = {}
    for mode in range(3):
        selected = [r for r in rows if r["mode"] == mode]
        assert len(selected) == 300
        assert {(r["query"], r["round"]) for r in selected} == {(q, i) for q in range(100) for i in range(3)}
        original = []; stored = []
        for r in selected:
            b = baseline[r["query"]]
            assert r["ids"] == b["ids"] and r["candidates"] == b["candidates"]
            assert r["gold_original"] == b["gold_original"] and r["gold_stored"] == b["gold_stored"]
            a = sum(i in r["gold_original"] for i in r["ids"]) / 10
            s = sum(i in r["gold_stored"] for i in r["ids"]) / 10
            assert a == r["recall_original"] and s == r["recall_stored"]
            original.append(a); stored.append(s)
        times = sorted(statistics.median(r["seconds"] for r in selected if r["query"] == q) for q in range(100))
        result["50k"][str(mode)] = {"recall_original": statistics.mean(original), "recall_stored": statistics.mean(stored), "p50_seconds": times[49], "p95_seconds": times[94]}
    for corpus, count in [("xquad", 1190), ("miracl", 482)]:
        rows = read(root, corpus + "-results.json.gz")
        assert len(rows) == count * 3
        base = {r["query"]: r for r in rows if r["mode"] == 0}
        assert len(base) == count
        result[corpus] = {"queries": count, "changed_returned_lists": {}, "ndcg": {}}
        for mode in range(3):
            selected = [r for r in rows if r["mode"] == mode]
            assert len(selected) == count and {r["query"] for r in selected} == set(base)
            if mode:
                result[corpus]["changed_returned_lists"][str(mode)] = sum(r["ranked"] != base[r["query"]]["ranked"] for r in selected)
            values = {"same_language": [], "cross_lingual": []} if corpus == "xquad" else []
            for r in selected:
                if corpus == "xquad":
                    key = gold[corpus][str(r["query"])]; own = key["answers"][key["language"]]
                    translations = set(key["answers"].values()) - {own}
                    for group, relevant, excluded in [("same_language", {own}, translations), ("cross_lingual", translations, {own})]:
                        value = ndcg([t for t in r["ranked"] if t not in excluded], relevant)
                        assert math.isclose(value, r["scores"][group]["ndcg"], abs_tol=1e-12)
                        values[group].append(value)
                else:
                    assert r["profile"] == "accuracy"
                    value = ndcg(r["ranked"], set(gold[corpus][r["query"]]))
                    assert math.isclose(value, r["scores"]["ndcg"], abs_tol=1e-12)
                    values.append(value)
            result[corpus]["ndcg"][str(mode)] = {g: sum(v) / count for g, v in values.items()} if corpus == "xquad" else sum(values) / count
    return result
