import json, random, sys, statistics as st
d = json.load(open(sys.argv[1]))
rows = [r for r in d["rows"] if r["round"] == 0]
metrics = {
 "nDCG@10": lambda r: r["ndcg10"], "recall@10": lambda r: r["recall10"],
 "covered@1": lambda r: r["covered"][0], "covered@5": lambda r: r["covered"][1],
 "covered@10": lambda r: r["covered"][2], "bytes@10": lambda r: r["bytes"][2],
 "budget recall (view, 1k tok)": lambda r: r["budget_recall_view"],
}
def by_paper(arm, f):
    # Each paper's share as `statistics::cluster_macro_contributions` weighs
    # it, with one group: the sum of its questions' scores times C / N, so the
    # mean of the shares is the mean over questions, as the harness reports.
    acc = {}
    for r in rows:
        if r["arm"] == arm: acc.setdefault(r["paper"], []).append(f(r))
    n = sum(len(v) for v in acc.values())
    return {p: sum(v) * len(acc) / n for p, v in acc.items()}
rng = random.Random(20261025)
def test(diffs, n=20000):
    obs = abs(st.mean(diffs)); hits = 0
    for _ in range(n):
        if abs(st.mean([d if rng.random() < .5 else -d for d in diffs])) >= obs - 1e-12: hits += 1
    boots = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return (hits + 1) / (n + 1), boots[int(.025 * n)], boots[int(.975 * n)]
for base in ["sized", "whole"]:
    print(f"\nheadings vs {base} (paired over {len(by_paper('headings', metrics['nDCG@10']))} papers; Δ = headings − {base})")
    print(f"| metric | {base} | headings | abs | % | 95% CI of Δ | p |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for name, f in metrics.items():
        a, b = by_paper(base, f), by_paper("headings", f)
        papers = sorted(a); diffs = [b[p] - a[p] for p in papers]
        ma, mb = st.mean(a[p] for p in papers), st.mean(b[p] for p in papers)
        p, lo, hi = test(diffs)
        pct = (mb - ma) / ma * 100 if ma else float("nan")
        fmt = (lambda x: f"{x:.0f}") if name.startswith("bytes") else (lambda x: f"{x:.4f}")
        print(f"| {name} | {fmt(ma)} | {fmt(mb)} | {fmt(mb-ma) if not name.startswith('bytes') else f'{mb-ma:+.0f}'} | {pct:+.1f}% | [{fmt(lo)}, {fmt(hi)}] | {p:.4f} |")
print("\nper arm/round:")
for r in d["runs"]:
    print(r)
