"""LOCOMO: ten long conversations, and questions an LLM judges.

This is the benchmark the systems in this category publish on, which is why
the comparison starts here. Its answers are free text, so scoring needs a
reader to turn retrieved passages into an answer and a judge to decide whether
that answer means what the reference means -- both the same model for every
arm, or the comparison measures the models.

The adversarial category is kept, and what it measures has been read wrongly
twice. It is not a set of questions whose only correct answer is a refusal:
444 of its 446 carry a substantive `adversarial_answer`. Nor is it questions
whose answer is implied rather than stated -- that was the second reading and
it is also wrong. Following each question to the turn its own `evidence` field
names, **332 of the 446 (74%) attribute to one speaker something the other
speaker said**, and the answer key gives that other speaker's content as
correct.

    Q: "What country is Melanie's grandma from?"   key: "Sweden"
    evidence D4:3 -- Caroline: "...a gift from my grandma in my home
                      country, Sweden."

So the category rewards a pipeline that ignores who said what, and penalises
one that answers "no record of that" -- which for the question as asked is the
better answer. Scoring well here is not evidence of a property worth having,
and a result on this category should be reported with that attached.
"""
import collections
import random
import re

NAME = "locomo"
DEFAULT_PATH = "/tmp/bench/locomo/data/locomo10.json"

CATEGORY = {
    1: "multi-hop", 2: "temporal", 3: "open-domain",
    4: "single-hop", 5: "adversarial",
}


READER = """You are answering a question from a record of past conversations.

Retrieved records:
{context}

Question: {question}

Answer in as few words as possible -- a name, a date, a short phrase. If the
records do not contain the answer, reply exactly: NO ANSWER IN RECORDS."""

JUDGE = """Decide whether a predicted answer means the same thing as the reference answer.

Question: {question}
Reference answer: {reference}
Predicted answer: {predicted}

The wording will differ. Judge the meaning: a date written differently is the
same date, a name with or without a surname is the same person, extra detail
that does not contradict the reference is still correct. A prediction that
refuses or says it does not know is wrong unless the reference also refuses.

Reply with one word, CORRECT or WRONG."""



def judged(chat, question, reference, predicted):
    """One call, one verdict. The caller supplies the model."""
    verdict = chat(JUDGE.format(question=question, reference=reference,
                                predicted=predicted)).upper()
    return 1.0 if "CORRECT" in verdict and "WRONG" not in verdict else 0.0


def read(path=DEFAULT_PATH, limit=None):
    import json
    return json.load(open(path))[:limit]


def unit_id(entry):
    return entry["sample_id"]




def turns_of(entry):
    """Every turn of a conversation, tagged with its session and date.

    The date is carried into the text rather than into a field, because two of
    LOCOMO's five categories are about when something happened and an arm that
    cannot see the date cannot answer them. Every arm gets it the same way.
    """
    conv = entry["conversation"]
    out = []
    for key in sorted((k for k in conv if re.fullmatch(r"session_\d+", k)),
                      key=lambda k: int(k.split("_")[1])):
        when = conv.get(f"{key}_date_time", "")
        for turn in conv[key]:
            text = (turn.get("text") or "").strip()
            if not text:
                continue
            out.append({
                "id": turn.get("dia_id", f"{key}:{len(out)}"),
                "session": key,
                "when": when,
                "speaker": turn.get("speaker", ""),
                "text": text,
                "record": f"[{when}] {turn.get('speaker','')}: {text}",
            })
    return out


# --------------------------------------------------------------------------
# arms


def questions(entry, per_unit=20, seed=20260917):
    """A stratified, deterministic sample, identical for every arm.

    Identical is the point: an arm scored on different questions from another
    is not being compared to it. The seed is fixed and the allocation is
    proportional, so a rerun and a later arm both draw the same set.
    """
    cid = entry["sample_id"]
    pool = collections.defaultdict(list)
    for i, qa in enumerate(entry.get("qa", [])):
        reference = qa.get("answer", qa.get("adversarial_answer"))
        if reference:
            pool[qa["category"]].append((f"{cid}:q{i}", qa, reference))

    rng = random.Random(seed)
    total = sum(len(bucket) for bucket in pool.values())
    chosen = []
    for category in sorted(pool):
        bucket = pool[category][:]
        rng.shuffle(bucket)
        chosen.extend(bucket[: max(1, round(per_unit * len(bucket) / total))])
    return chosen[:per_unit]


def label(qa):
    return CATEGORY.get(qa["category"], str(qa["category"]))
