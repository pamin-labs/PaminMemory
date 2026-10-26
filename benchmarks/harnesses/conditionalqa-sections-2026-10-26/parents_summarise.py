import json, random, statistics as st, sys
rows = json.load(open(sys.argv[1]))
pairs = [tuple(p.split(':')) for p in sys.argv[2:]] or [("whole", "sections"), ("sections", "outline")]
M = {
 "all evidence @5": lambda r: r["complete"][1], "all evidence @10": lambda r: r["complete"][2],
 "covered@1": lambda r: r["covered"][0], "covered@5": lambda r: r["covered"][1], "covered@10": lambda r: r["covered"][2],
 "fixed 1k read": lambda r: r["fixed_read"], "fixed 1k read, all": lambda r: r["fixed_read_complete"],
 "bytes@10": lambda r: r["bytes"][2], "parents shown in 10": lambda r: r["parents_shown10"],
}
rng = random.Random(20261025)
def shares(arm, f, spread):
    acc = {}
    for r in rows:
        if r["arm"] == arm and (spread is None or r["spread"] == spread):
            acc.setdefault(r["file"], []).append(f(r))
    n = sum(len(v) for v in acc.values())
    return {k: sum(v) * len(acc) / n for k, v in acc.items()}, n
def test(d, n=10000):
    obs = abs(st.mean(d)); hits = sum(abs(st.mean([x if rng.random() < .5 else -x for x in d])) >= obs - 1e-12 for _ in range(n))
    b = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(n))
    return (hits + 1) / (n + 1), b[int(.025 * n)], b[int(.975 * n)]
arms = {r["arm"] for r in rows}
for base, new in pairs:
    if base not in arms or new not in arms: continue
    for spread in [None, "one section", "with an ancestor", "siblings", "unrelated"]:
        print(f"\n### {new} vs {base} — {spread or 'all questions'}")
        print(f"| metric | {base} | {new} | Δ | % | 95% CI | p |\n|---|---:|---:|---:|---:|---|---:|")
        for name, f in M.items():
            a, n = shares(base, f, spread); b, _ = shares(new, f, spread)
            keys = sorted(a); d = [b[k] - a[k] for k in keys]
            ma, mb = st.mean(a.values()), st.mean(b.values())
            if all(x == 0 for x in d): p, lo, hi = 1.0, 0.0, 0.0
            else: p, lo, hi = test(d)
            pct = f"{(mb-ma)/ma*100:+.1f}%" if ma else "n/a"
            fmt = "{:.0f}" if name.startswith("bytes") else "{:.3f}"
            print(f"| {name} (n={n}) | {fmt.format(ma)} | {fmt.format(mb)} | {mb-ma:+.3f} | {pct} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} |")
