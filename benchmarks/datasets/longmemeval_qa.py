"""LongMemEval: the other half -- answer accuracy, reader and judge included.

`longmemeval.py` scores the retrieval stage alone, which mem0 cannot be
scored on at all (it rewrites turns into facts with no session identity to
rank). This module exists so mem0 has a LongMemEval column anyway: an LLM
reader answers from whatever an arm hands back, an LLM judge grades it, same
shape as `locomo.py`, same reason -- free text needs a reader and a judge,
and comparing their outputs needs the same model behind every arm's reader.

It reuses `longmemeval.py`'s haystack parsing (`turns_of`, `unit_id`,
`sample`) rather than copying it: the haystack is the haystack regardless of
which half of the benchmark is being scored, and two copies of that parser
drift.

Every question participates here, not just the 470 `longmemeval.py` keeps.
The 30 abstention questions it drops are unscoreable on a retrieval metric --
there is no session to rank toward -- but they are answerable text, same as
LOCOMO's adversarial category, and the judge can tell a correct refusal from
a wrong one.
"""
import json

from . import longmemeval as lme

NAME = "longmemeval_qa"
RETRIEVAL_ONLY = False
DEFAULT_PATH = lme.DEFAULT_PATH

unit_id = lme.unit_id
turns_of = lme.turns_of
sample = lme.sample


READER = """You are answering a question from a record of past conversation sessions.

Retrieved records:
{context}

Question: {question}

Answer in as few words as possible -- a name, a date, a short phrase. If the
records do not contain the answer, say so plainly rather than guessing."""

JUDGE = """Decide whether a predicted answer means the same thing as the reference answer.

Question: {question}
Reference answer: {reference}
Predicted answer: {predicted}

The wording will differ. Judge the meaning: a date written differently is the
same date, a name with or without a surname is the same person, extra detail
that does not contradict the reference is still correct. Some reference
answers are themselves a correction -- "you did not mention X, you mentioned
Y" -- and a prediction that says the information was not given, or names the
same thing the reference names, is correct for those; a prediction that
invents the missing detail is wrong.

Reply with one word, CORRECT or WRONG."""


def judged(chat, question, reference, predicted):
    verdict = chat(JUDGE.format(question=question, reference=reference,
                                predicted=predicted)).upper()
    return 1.0 if "CORRECT" in verdict and "WRONG" not in verdict else 0.0


def read(path=DEFAULT_PATH, limit=None):
    """Every question, abstention included -- see the module docstring."""
    return json.load(open(path))[:limit] if limit else json.load(open(path))


def questions(entry, per_unit=None, seed=None):
    """One question per haystack, same shape as `longmemeval.py`'s."""
    return [(entry["question_id"], {
        "question": entry["question"],
        "question_type": entry["question_type"],
    }, entry["answer"])]


def label(qa):
    return qa["question_type"]
