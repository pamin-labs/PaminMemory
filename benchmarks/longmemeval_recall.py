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


rows = [json.loads(l) for l in open("/tmp/bench/lme_pamin.jsonl") if l.strip()]
by_id = {x["question_id"]: x for x in json.load(open(DATA))}
print(f"{len(rows)} questions\n", flush=True)

bm_any5 = bm_all5 = bm_any10 = 0.0
pm_any5 = pm_all5 = 0.0
by_type = collections.defaultdict(lambda: [0.0, 0.0, 0])

for r in rows:
    entry = by_id[r["question_id"]]
    gold = set(entry["answer_session_ids"])

    topics, texts = [], []
    for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        for i, turn in enumerate(session):
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

    bm_any5 += 1.0 if gold & set(sessions[:5]) else 0.0
    bm_any10 += 1.0 if gold & set(sessions[:10]) else 0.0
    bm_all5 += 1.0 if gold <= set(sessions[:5]) else 0.0

    pa = 1.0 if gold & set(r["ranked_top10"][:5]) else 0.0
    pm_any5 += pa
    pm_all5 += 1.0 if gold <= set(r["ranked_top10"][:5]) else 0.0
    t = by_type[r["question_type"]]
    t[0] += 1.0 if gold & set(sessions[:5]) else 0.0
    t[1] += pa
    t[2] += 1

n = len(rows)
print(f"{'metric':<28}{'BM25':>9}{'pamin':>9}")
print(f"{'recall_any@5  (their R@5)':<28}{bm_any5/n:>9.4f}{pm_any5/n:>9.4f}")
print(f"{'recall_any@10':<28}{bm_any10/n:>9.4f}{1.0:>9.4f}")
print(f"{'recall_all@5':<28}{bm_all5/n:>9.4f}{pm_all5/n:>9.4f}")
print(f"\n  (the earlier run put BM25 recall_all@5 at 0.7966 -- this rebuild "
      f"gives {bm_all5/n:.4f})")
print("\nrecall_any@5 by question type:")
for t in sorted(by_type):
    b, p, c = by_type[t]
    print(f"  {t:<28} n={c:>2}   BM25 {b/c:.3f}   pamin {p/c:.3f}")
