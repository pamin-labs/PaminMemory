"""MIRACL: 131,924 real passages, 482 human-judged queries, one language.

The one dataset in this category whose relevance judgements come from people
rather than from a single pooled run of five systems' top ten -- 5,092
judgements over 482 queries, 1.89 relevant passages each, over the same
`miracl-v1.0-sw` dev split `crates/pamin-engine/tests/monolingual.rs` already
scores Påmin Memory alone on. This is the first time mem0 and MemPalace are
run against it directly, rather than quoted from their own papers or a
leaderboard.

There is no reader and no judge: a passage either carries a docid a human
marked relevant to the query or it does not, so this is scored the same way
LongMemEval's retrieval stage is, and for the same reason -- nothing here
depends on which model is doing the judging.

mem0 cannot be scored on this metric for the same structural reason it cannot
be scored on LongMemEval's: it extracts facts on write, so a returned memory
belongs to no passage and there is no bijection to invent. `run.py` refuses
rather than guessing one. Its write-path cost (calls, seconds, dollars) is
still measurable -- run it with `--questions 0` to ingest without asking it
for a retrieval it cannot give.

Every arm's identity for a passage is put through the same ASCII-safe
transform before comparison, because Påmin Memory's own CLI collapses
anything outside `[A-Za-z0-9_]` in a topic name and MIRACL's docids
(`article#paragraph`) are not in that alphabet. Starting every id inside that
alphabet makes Påmin Memory's collapse a no-op instead of a second,
undocumented identity for the same passage, and keeps every arm's retrieved
id comparable to the same gold string -- including the one arm (`bm25`) that
never touches it.
"""
import collections
import gzip
import json
import math
import os
import re

NAME = "miracl"
RETRIEVAL_ONLY = True
DEFAULT_PATH = "/tmp/bench/miracl"

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def _safe_id(docid):
    return "miraclsw_" + _UNSAFE.sub("_", docid)


def read(path=DEFAULT_PATH, limit=None):
    """One entry: the whole sampled corpus, ingested exactly once.

    `MIRACL_MAX_DOCS` caps the corpus for checking the harness runs; it is
    not the benchmark and the result is not comparable with anyone's -- the
    same rule `crosslingual.rs` and `monolingual.rs` already apply to this
    same corpus.
    """
    topics = {}
    with open(os.path.join(path, "topics.miracl-v1.0-sw-dev.tsv"),
              encoding="utf-8") as f:
        for line in f:
            qid, text = line.rstrip("\n").split("\t", 1)
            topics[qid] = text

    qrels = collections.defaultdict(set)
    with open(os.path.join(path, "qrels.miracl-v1.0-sw-dev.tsv"),
              encoding="utf-8") as f:
        for line in f:
            qid, _, docid, rel = line.split()
            if rel == "1":
                qrels[qid].add(_safe_id(docid))

    max_docs = int(os.environ.get("MIRACL_MAX_DOCS", "0")) or None
    passages = []
    with gzip.open(os.path.join(path, "docs-0.jsonl.gz"), "rt",
                    encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            row = json.loads(line)
            text = f"{row.get('title', '')} {row.get('text', '')}".strip()
            passages.append((row["docid"], text))

    queries = [(qid, text, qrels[qid]) for qid, text in topics.items()
               if qrels[qid]]
    entry = {"passages": passages, "queries": queries}
    return [entry][:limit] if limit else [entry]


def unit_id(entry):
    return "miracl-v1.0-sw-dev"


def turns_of(entry):
    """Every passage, as its own one-turn session.

    There is no conversation here, so each passage stands alone the way
    `longmemeval.py` already lets a turn stand alone inside a session -- the
    shape this dataset needs already existed, it had just never been asked
    for at passage scale.
    """
    out = []
    for docid, text in entry["passages"]:
        safe = _safe_id(docid)
        out.append({"id": safe, "session": safe, "when": "",
                    "speaker": "", "text": text, "record": text})
    return out


def questions(entry, per_unit=None, seed=None):
    """Every judged query, or the first `per_unit` of them.

    `per_unit=0` is deliberate and different from `None`: it ingests the
    corpus and asks for nothing back, which is how an arm that cannot be
    scored on this metric (mem0) still gets its write-path cost measured.
    """
    queries = entry["queries"]
    if per_unit is not None:
        queries = queries[:per_unit]
    return [(qid, {"question": text, "relevant": gold}, None)
            for qid, text, gold in queries]


def sessions_of(ids):
    """Each passage is already its own session; nothing to roll up."""
    seen, ranked = set(), []
    for pid in ids:
        if pid not in seen:
            seen.add(pid)
            ranked.append(pid)
    return ranked


def ndcg(ranked, gold, k):
    """Binary nDCG@k, trec_eval's `ndcg_cut`: MIRACL's qrels hold {0, 1}."""
    dcg = sum(1 / math.log2(i + 2) for i, pid in enumerate(ranked[:k]) if pid in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def recall(ranked, gold, k):
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def score(qa, ranked):
    gold = qa["relevant"]
    return {"recall@10": recall(ranked, gold, 10), "ndcg@10": ndcg(ranked, gold, 10)}


def label(qa):
    # One language throughout, by MIRACL's own design -- inventing a split
    # here would report a number nobody else reports.
    return "swahili"
