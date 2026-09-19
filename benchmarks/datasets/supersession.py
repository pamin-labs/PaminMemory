"""Supersession: does an arm answer with the fact that still holds?

LongMemEval's `knowledge-update` questions each carry two gold sessions. The
earlier one states a fact, the later one replaces it, and the answer key gives
the later value. Every other metric in this directory would score an arm that
returns the earlier value exactly the way it scores an arm that returns
nothing, and those are not the same failure: one is a gap, the other is a
memory confidently reciting something it was told had stopped being true.

So the verdict here is three-way -- current, stale, or neither -- and the
figure worth reporting is the stale rate, not the accuracy. The construction
follows Supersede (arXiv:2606.27472), which reached the same place from the
other direction: it built a synthetic supersession corpus first, found it
saturated, and fell back to these questions scored on answers.

Two things the data makes easy to get wrong, both checked rather than assumed:

  The `_1` and `_2` suffixes on the gold session ids are not the timeline.
  One of the 78 has `_1` dated after `_2`, so which fact is stale is decided
  by `haystack_dates` and never by the name.

  Eight of the 78 have no answer-bearing turn in the earlier session at all:
  the two sessions are a plan and its confirmation rather than a value and its
  replacement, and there is no stale value to be wrong about. Scoring them
  three ways would invent a category the data does not have, so they are
  dropped by the premise check below and 70 remain.

Dates are carried in `when` and not folded into `record`, which is the
opposite of what the LOCOMO adapter does, because here the date is the
variable under test: an arm that gets it for free in the passage text cannot
tell us whether reading it off the ledger was worth anything.
"""
import json
import re

NAME = "supersession"
DEFAULT_PATH = "/tmp/bench/lme/data/longmemeval_s_cleaned.json"

DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2}) \(\w{3}\) (\d{2}):(\d{2})")


READER = """You are answering a question from a record of past conversations.

Some records contradict others: the same fact was stated once and then
restated later with a different value. Answer with the value that holds now,
not the one it replaced. Where a record carries a date, the later date wins.

Retrieved records:
{context}

Question: {question}

Answer in as few words as possible -- a name, a number, a short phrase. If the
records do not contain the answer, reply exactly: NO ANSWER IN RECORDS."""

JUDGE = """A fact changed. Decide which version of it a predicted answer gives.

Question: {question}
The value that holds now: {current}
What was said earlier, before it changed: {stale}
Predicted answer: {predicted}

Wording will differ, and that does not matter: a number written differently is
the same number, a name with or without a surname is the same person, extra
detail that contradicts neither is still the same answer. What matters is
which of the two values the prediction commits to.

Reply with exactly one word:
  CURRENT -- it gives the value that holds now
  STALE   -- it gives the earlier value, the one that was replaced
  NEITHER -- it gives some third value, refuses, or says it does not know"""


def rfc3339(stamp):
    """`2023/05/23 (Tue) 13:01` as a timestamp the CLI accepts."""
    m = DATE.match((stamp or "").strip())
    if not m:
        return None
    year, month, day, hour, minute = m.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:00Z"


def _pair(entry):
    """The two gold sessions, earlier first, with their answer-bearing turns.

    Ordered by date rather than by the `_1`/`_2` suffix, because for one of
    the 78 those disagree and the suffix is the one that is wrong.
    """
    where = {s: i for i, s in enumerate(entry["haystack_session_ids"])}
    ordered = sorted(entry["answer_session_ids"],
                     key=lambda s: entry["haystack_dates"][where[s]])
    out = []
    for sid in ordered:
        session = entry["haystack_sessions"][where[sid]]
        out.append({
            "id": sid,
            "date": entry["haystack_dates"][where[sid]],
            "stated": [t["content"].strip() for t in session if t.get("has_answer")],
        })
    return out


def read(path=DEFAULT_PATH, limit=None):
    """The knowledge-update questions that actually have a stale value.

    The premise check is the point of this function: an entry whose earlier
    gold session says nothing about the question cannot produce a STALE
    answer, so keeping it would put a floor under every arm's stale rate that
    has nothing to do with the arm.
    """
    pool = [x for x in json.load(open(path))
            if x["question_type"] == "knowledge-update"]
    kept = []
    for entry in pool:
        earlier, later = _pair(entry)
        if earlier["date"] >= later["date"]:
            continue
        if not earlier["stated"] or not later["stated"]:
            continue
        kept.append(entry)
    if not kept:
        raise RuntimeError(f"no knowledge-update questions survived the premise "
                           f"check in {path}; the file is not LongMemEval-S")
    return kept[:limit] if limit else kept


def unit_id(entry):
    return entry["question_id"]


def turns_of(entry):
    """Every turn of the haystack, with its session's date beside it.

    Beside it, not inside `record`: which arms may read the date is the
    experiment. An arm that wants it in the passage text puts it there itself
    and says so in its name.
    """
    out = []
    for sid, date, session in zip(entry["haystack_session_ids"],
                                  entry["haystack_dates"],
                                  entry["haystack_sessions"]):
        for i, turn in enumerate(session):
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            out.append({
                "id": f"{sid}__t{i}",
                "session": sid,
                "when": rfc3339(date),
                "speaker": turn.get("role", "user"),
                "text": text,
                "record": f"{turn.get('role', 'user')}: {text}",
            })
    return out


def questions(entry, per_unit=1, seed=None):
    """One question per haystack; `per_unit` and `seed` are the shared shape."""
    earlier, _later = _pair(entry)
    reference = {"current": entry["answer"], "stale": "\n".join(earlier["stated"])}
    return [(entry["question_id"], {"question": entry["question"]}, reference)]


def judged(chat, question, reference, predicted):
    """Three columns rather than one, because two of them are failures.

    Returned as a mapping so the row keeps the verdict alongside `correct`:
    an arm's accuracy and its stale rate do not move together, and a page that
    reports only the first cannot say which kind of memory it has.
    """
    verdict = chat(JUDGE.format(question=question, current=reference["current"],
                                stale=reference["stale"],
                                predicted=predicted)).upper()
    for name in ("CURRENT", "STALE", "NEITHER"):
        if name in verdict:
            return {"correct": 1.0 if name == "CURRENT" else 0.0,
                    "verdict": name.lower()}
    return {"correct": 0.0, "verdict": "unparsed"}


def label(qa):
    return "knowledge-update"
