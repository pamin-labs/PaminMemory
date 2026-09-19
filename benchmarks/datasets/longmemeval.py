"""LongMemEval: one haystack per question, scored on retrieval alone.

The other half of this benchmark needs a reader and a judge; this half does
not, and that is why it is here. Recall@k and NDCG@k against the gold sessions
are the only figures in this category that can be set beside another
project's without a model in between, and they cost nothing to score.

Each question carries its own haystack of about fifty sessions, so a unit here
is a question rather than a conversation -- fifty-nine ingests an arm instead
of ten, which is most of why the LOCOMO comparison ran first.

Indexed at turn granularity and rolled up to sessions. Concatenating a session
would hand the embedder a ten-thousand-character passage to truncate, and
measure the truncation.
"""
import collections
import json
import math
import random

NAME = "longmemeval"
DEFAULT_PATH = "/tmp/bench/lme/data/longmemeval_s_cleaned.json"


def read(path=DEFAULT_PATH, limit=None):
    """The questions, with the abstention set dropped.

    Their correct answer is a refusal, which a retrieval metric cannot score,
    and the benchmark's own scorer drops them too.
    """
    data = json.load(open(path))
    pool = [x for x in data if "_abs" not in x["question_id"]]
    return pool[:limit] if limit else pool


def unit_id(entry):
    return entry["question_id"]


def sample(entries, n, seed=20260917):
    """Stratified by question type, deterministic, same set for every arm."""
    by_type = collections.defaultdict(list)
    for entry in entries:
        by_type[entry["question_type"]].append(entry)
    rng = random.Random(seed)
    chosen = []
    for kind in sorted(by_type):
        bucket = by_type[kind][:]
        rng.shuffle(bucket)
        chosen.extend(bucket[: max(1, round(n * len(bucket) / len(entries)))])
    return chosen[:n]


def turns_of(entry):
    """Every turn of every session in this question's haystack.

    The role is kept in the text rather than dropped into a field nothing
    indexes: several question types turn on who said a thing.
    """
    out = []
    for sid, session in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        for i, turn in enumerate(session):
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            out.append({
                "id": f"{sid}__t{i}",
                "session": sid,
                "when": "",
                "speaker": turn.get("role", "user"),
                "text": text,
                "record": f"{turn.get('role', 'user')}: {text}",
            })
    return out


def sessions_of(passages):
    """Session-level ranking: a session ranks where its best turn ranked."""
    ranked, seen = [], set()
    for topic in passages:
        sid = topic.rsplit("__t", 1)[0]
        if sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


def ndcg_any(ranked, gold, k):
    """Any gold session counts as relevant, as the benchmark scores it."""
    dcg = sum(1 / math.log2(i + 2) for i, s in enumerate(ranked[:k]) if s in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def recall_all(ranked, gold, k):
    """One when every gold session is inside the top k, as the benchmark does.

    `recall@50` is not worth reporting on LongMemEval-S: its haystack holds
    about fifty sessions, so the figure approaches one by construction.
    """
    return 1.0 if gold and gold.issubset(set(ranked[:k])) else 0.0


def score(entry, ranked_sessions):
    gold = set(entry["answer_session_ids"])
    return {
        "recall_all@5": recall_all(ranked_sessions, gold, 5),
        "ndcg_any@5": ndcg_any(ranked_sessions, gold, 5),
        "recall_all@10": recall_all(ranked_sessions, gold, 10),
        "ndcg_any@10": ndcg_any(ranked_sessions, gold, 10),
    }


def label(entry):
    return entry["question_type"]
