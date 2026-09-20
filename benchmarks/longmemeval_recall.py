"""LongMemEval session recall, at the metric the field publishes and at a strict one.

Run: `python3 benchmarks/longmemeval_recall.py`. No model is called anywhere,
which is what makes this the one retrieval figure here that can sit beside a
competitor's published number.

The point of the baseline column: MemPalace's headline is "96.6% R@5 raw on
LongMemEval", and its own benchmark
doc defines R@5 as whether the labelled session is inside the top five. That is
`recall_any@5`, a looser question than the `recall_all@5` this project has been
reporting -- 35 of these 59 questions carry more than one gold session.

Before quoting our figure at that definition, BM25 has to be measured at it
too. `recall_any@5` over a haystack of about fifty sessions may simply be an
easy question, in which case the number says something about the benchmark and
nothing about the system.

The corpus is rebuilt from the LongMemEval file rather than the per-question
ndjson the original run wrote, because those are gone. The check that this
rebuild matches is `recall_all@5`, which the earlier run put at 0.7966.
"""
import json, math, re, collections

DATA = "/tmp/bench/lme/data/longmemeval_s_cleaned.json"


def tokens(text):
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokens(d) for d in docs]
        self.len = [len(d) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        self.df = collections.Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.n = len(self.docs)
        self.tf = [collections.Counter(d) for d in self.docs]

    def scores(self, query):
        out = [0.0] * self.n
        for term in set(tokens(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def hit(gold, ranked, whole):
    """Is the gold session in this cut -- all of them, or any of them."""
    return 1.0 if (gold <= set(ranked) if whole else gold & set(ranked)) else 0.0


def rebuild(rows, data_path=DATA):
    """Both metrics for both sides, over the questions `rows` covers.

    Returned rather than printed so that benchmarks/summarise.py can write the
    same figures into the committed summary. Two copies of a retriever drift,
    and then a printed table and a committed file disagree and neither is
    wrong.
    """
    by_id = {x["question_id"]: x for x in json.load(open(data_path))}
    bm = {"any@5": 0.0, "any@10": 0.0, "all@5": 0.0}
    pm = {"any@5": 0.0, "any@10": 0.0, "all@5": 0.0}
    by_type = collections.defaultdict(lambda: [0.0, 0.0, 0])
    multi_gold = 0

    for r in rows:
        entry = by_id[r["question_id"]]
        gold = set(entry["answer_session_ids"])
        multi_gold += 1 if len(gold) > 1 else 0

        topics, texts = [], []
        for sid, session in zip(entry["haystack_session_ids"],
                                entry["haystack_sessions"]):
            for turn in session:
                content = (turn.get("content") or "").strip()
                if content:
                    topics.append(sid)
                    texts.append(content)

        order = sorted(zip(topics, BM25(texts).scores(entry["question"])),
                       key=lambda p: -p[1])
        sessions, seen = [], set()
        for sid, _ in order:
            if sid not in seen:
                seen.add(sid)
                sessions.append(sid)

        # Every cell measured, including the ones that come to 1.0. This
        # printed `any@10` for pamin as the literal 1.0, which is what it
        # comes to and is still not a measurement of anything -- and a
        # committed summary built on a literal is worse than no summary.
        ranked = r["ranked_top10"]
        for metric, cut, whole in (("any@5", 5, False), ("any@10", 10, False),
                                   ("all@5", 5, True)):
            bm[metric] += hit(gold, sessions[:cut], whole)
            pm[metric] += hit(gold, ranked[:cut], whole)

        t = by_type[r["question_type"]]
        t[0] += hit(gold, sessions[:5], False)
        t[1] += hit(gold, ranked[:5], False)
        t[2] += 1

    n = len(rows)
    return {
        "questions": n,
        "multi_gold": multi_gold,
        "bm25": {k: round(v / n, 4) for k, v in bm.items()},
        "pamin": {k: round(v / n, 4) for k, v in pm.items()},
        "by_type": {t: {"questions": c, "bm25": round(b / c, 4),
                        "pamin": round(p / c, 4)}
                    for t, (b, p, c) in sorted(by_type.items())},
    }


def main():
    rows = [json.loads(l) for l in open("/tmp/bench/lme_pamin.jsonl") if l.strip()]
    print(f"{len(rows)} questions\n", flush=True)
    out = rebuild(rows)
    bm, pm = out["bm25"], out["pamin"]

    print(f"{'metric':<28}{'BM25':>9}{'pamin':>9}")
    print(f"{'recall_any@5  (their R@5)':<28}{bm['any@5']:>9.4f}{pm['any@5']:>9.4f}")
    print(f"{'recall_any@10':<28}{bm['any@10']:>9.4f}{pm['any@10']:>9.4f}")
    print(f"{'recall_all@5':<28}{bm['all@5']:>9.4f}{pm['all@5']:>9.4f}")
    print(f"\n  (the earlier run put BM25 recall_all@5 at 0.7966 -- this rebuild "
          f"gives {bm['all@5']:.4f})")
    print("\nrecall_any@5 by question type:")
    for t, split in out["by_type"].items():
        print(f"  {t:<28} n={split['questions']:>2}   "
              f"BM25 {split['bm25']:.3f}   pamin {split['pamin']:.3f}")


if __name__ == "__main__":
    main()
