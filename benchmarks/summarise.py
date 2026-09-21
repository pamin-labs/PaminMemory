#!/usr/bin/env python3
"""Aggregate a run's raw rows into the summaries this repository commits.

`results/` holds two kinds of thing and only one of them belongs in git. The
raw rows are the evidence: one per question per arm, written continuously
while a run is in flight, far too many to review in a diff. What is committed
is this script's output -- one `summary-*.json` per table in
docs/benchmarks.md, so that a reader of that page has an artifact to check it
against instead of a promise.

    python3 benchmarks/summarise.py /path/to/raw

Two rules hold the output honest, and both exist because the alternative was
tried somewhere else in this repository and produced a number nobody could
trace:

  Nothing is copied from the page. Every figure here is recomputed from the
  rows, and where the recomputation disagrees with the prose the recomputation
  is what gets written. A summary edited to match its own documentation is not
  evidence of anything.

  A figure the rows cannot support is `null` with a note saying why, never a
  plausible-looking value. That applies to the machine most of all: the first
  runs predate the `machine` object, and a summary that invented one would
  make an unverifiable row look like a verified one.
"""
import argparse
import collections
import json
import math
import os
import statistics
from datetime import datetime, timezone

import longmemeval_recall

# The five LOCOMO files, in the order they were run. Deliberately a list of
# names rather than a glob: the same directory holds smoke tests, a
# pre-instrumentation run and a repository smoke run, and a glob would quietly
# fold those into the totals.
LOCOMO_FILES = ["compare.jsonl", "compare2.jsonl", "compare3.jsonl",
                "compare4.jsonl", "compare5.jsonl"]

# The arms docs/benchmarks.md reports, in the order its accuracy table lists
# them, with the shortlist each one asked for.
LOCOMO_ARMS = [("bm25", 10), ("pamin", 10), ("pamin-ledger", 10),
               ("mem0-k10", 10), ("mempalace", 10), ("mem0-k30", 30),
               ("mempalace-wide", 30), ("pamin-wide", 30)]

# Readings taken and then withdrawn. Kept in the summary rather than dropped,
# because "these arms were measured twice and the first reading was wrong" is
# itself a result -- it is where the noise floor on this benchmark comes from.
LOCOMO_WITHDRAWN = {
    "mem0": (10, "mem0's search keyword is `top_k`, not `limit`; the unknown "
                 "keyword went into **kwargs and the arm ran at mem0's own "
                 "default of 20 passages. `mem0ai[nlp]` was also missing, so "
                 "its BM25 channel lemmatised nothing. Superseded by "
                 "mem0-k10."),
    "mem0-wide": (30, "the same two faults, so this arm also ran at 20 "
                      "passages rather than the 30 it asked for. Superseded "
                      "by mem0-k30."),
}

# Every pair docs/benchmarks.md tests, including the two that compare an arm
# against itself at a different shortlist and the one that compares the two
# withdrawn mem0 runs. That last pair is the noise floor and is the reason the
# others are read as cautiously as they are.
LOCOMO_PAIRS = [("pamin-wide", "bm25"), ("mempalace-wide", "bm25"),
                ("mem0-k30", "bm25"), ("pamin-wide", "pamin"),
                ("mempalace-wide", "pamin"), ("pamin", "bm25"),
                ("pamin-wide", "mem0-k10"), ("pamin-wide", "mem0-k30"),
                ("pamin-wide", "mempalace-wide"), ("mempalace-wide", "mem0-k30"),
                ("pamin", "mem0-k10"), ("pamin", "mempalace"),
                ("pamin-ledger", "pamin"), ("mem0-k30", "mem0-k10"),
                ("mem0", "mem0-wide")]

SUPERSESSION_FILES = ["sup/sup.jsonl", "sup/sup2.jsonl", "sup/sup3.jsonl"]
SUPERSESSION_ARMS = ["bm25", "pamin", "pamin-dated", "pamin-valid",
                     "pamin-valid-read"]
SUPERSESSION_PAIRS = [("pamin", "bm25"), ("pamin-dated", "pamin"),
                      ("pamin-valid", "pamin"), ("pamin-valid-read", "pamin"),
                      ("pamin-valid-read", "pamin-valid"),
                      ("pamin-valid-read", "pamin-dated")]

VERDICTS = ["current", "stale", "neither", "unparsed"]

# The re-run that reads each question five times. Its own file, not folded in
# with the three above: it answers the same questions with a different
# instrument and on a different binary, so averaging the two would destroy the
# one thing worth knowing, which is whether they agree.
POWER_FILE = "power/power.jsonl"
POWER_ARMS = ["pamin-dated", "pamin-valid", "pamin-valid-read"]

# The latency harness's rows. It printed and saved nothing until it was given
# `--out`, which is why the table it produced sat on the page for a day with
# no artifact behind it.
LATENCY_FILE = "lat/latency.jsonl"
POWER_PAIRS = [("pamin-valid-read", "pamin-dated"),
               ("pamin-valid-read", "pamin-valid"),
               ("pamin-valid", "pamin-dated")]


def read_rows(raw, name):
    path = os.path.join(raw, name)
    if not os.path.exists(path):
        raise SystemExit(f"{name} is not in {raw}; this script summarises a "
                         f"run's raw rows and cannot invent the ones missing")
    return [json.loads(line) for line in open(path) if line.strip()]


def read_json(raw, name):
    path = os.path.join(raw, name)
    if not os.path.exists(path):
        raise SystemExit(f"{name} is not in {raw}")
    return json.load(open(path))


def source(raw, name, count):
    """Where a figure came from, and when, by filename alone.

    The raw tree lives outside the repository, so its absolute path describes
    one machine's layout and nothing a reader can use. The name travels; the
    path does not.

    The date is the file's mtime. The rows carry no timestamp of their own, so
    this is when the file was last appended to rather than when the run began,
    and the key says `mtime` so that nobody reads it as the second thing.
    """
    when = datetime.fromtimestamp(os.path.getmtime(os.path.join(raw, name)),
                                  timezone.utc)
    return {"file": name, "rows": count,
            "mtime_utc": when.strftime("%Y-%m-%dT%H:%M:%SZ")}


def machine_of(rows):
    """The machine these rows were taken on, or None when they predate it.

    Half of what a run reports -- latency, resident memory, wall clock -- is a
    property of the box and not of the software, so a file of rows without
    this cannot be merged with another honestly. The runs that produced
    `compare*.jsonl` were written before `run.py` stamped it, and the only
    truthful thing a summary can say about those is that it does not know.
    """
    seen = {json.dumps(r["machine"], sort_keys=True)
            for r in rows if r.get("machine")}
    if not seen:
        return None
    if len(seen) > 1:
        raise SystemExit("these rows come from more than one machine, so no "
                         "single figure over them means anything")
    return json.loads(seen.pop())


def stamped(rows, when_missing):
    """The machine these rows carry, or null and the reason there is none.

    The note is emitted only when there is nothing to stamp, so that a rerun
    over rows that do carry a machine does not leave an explanation of an
    absence standing next to a present one. That is the shape most of this
    repository's stale prose has taken.
    """
    where = machine_of(rows)
    return {"machine": where} if where else {"machine": None,
                                             "machine_note": when_missing}


def mcnemar(a_only, b_only):
    """The exact binomial test on the discordant pairs.

    Every arm answers the same questions, so the only information is in the
    questions where two arms disagree; the totals repeat whatever both got
    right. Exact rather than chi-square because these counts run down to eight.
    """
    n = a_only + b_only
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
        / 2 ** n)
    return {"discordant": n, "a_only": a_only, "b_only": b_only,
            "p": round(p, 4)}


def paired(by_arm, a, b, won):
    """Discordant counts over the questions both arms answered."""
    shared = set(by_arm[a]) & set(by_arm[b])
    return mcnemar(
        sum(1 for q in shared if won(by_arm[a][q]) and not won(by_arm[b][q])),
        sum(1 for q in shared if not won(by_arm[a][q]) and won(by_arm[b][q])))


def correct(row):
    return row["correct"] > 0


def last_per_unit(rows, key):
    """The last row written for each unit, for the columns stamped per unit.

    Ingest cost is measured once per unit and copied onto every row of it, so
    a sum needs one row per unit. Which one is not always a free choice: a
    resumed run re-ingests the unit it was interrupted in, and the rows written
    afterwards carry the second ingest. Seven of MemPalace's ten conversations
    carry two, and the pick moves its ingest total between 500 s and 545 s.

    Last-written is taken because it describes the store the run finished with.
    It is a choice, not a measurement, which is why the summary reports how
    many units were ingested more than once beside the totals.
    """
    out = {}
    for row in rows:
        out[row[key]] = row
    return out


def total(units, field):
    """Sum a per-unit field over the units, counting a missing one as zero."""
    return sum(unit.get(field, 0) for unit in units.values())


def billed(units):
    """What the ingest cost, or None when these rows never recorded it.

    `ingest_cost_usd` arrived late; the first four LOCOMO files have no such
    field, so summing it there gives a confident $0 for an arm that spent
    real money. An arm that made no model calls did pay nothing and can say
    so -- no calls, no bill -- and everything else says it does not know.
    """
    if any("ingest_cost_usd" in unit for unit in units.values()):
        return round(total(units, "ingest_cost_usd"), 4)
    return 0.0 if total(units, "ingest_llm_calls") == 0 else None


def cost_run_usd(cost_run, arm):
    """What the separate cost-mode run billed this arm, or None if it skipped it."""
    billed_usd = cost_run.get(arm, {}).get("write", {}).get("cost_usd")
    return None if billed_usd is None else round(billed_usd, 4)


def locomo_corpus(by_file):
    """How big the LOCOMO corpus each arm read actually was.

    Counted rather than written down. The turn count lives only in raw.jsonl:
    the five compare files record the conversation but not its length, and a
    corpus size copied from the page into a summary of the page is not a check
    on anything.
    """
    turns = {r["unit"]: r["turns"] for r in by_file["raw.jsonl"]}
    questions = {r["question_id"] for r in by_file["compare.jsonl"]}
    return {"name": "locomo", "conversations": len(turns),
            "questions": len(questions), "turns": sum(turns.values())}


def locomo_accuracy(raw, by_file):
    rows = [r for name in LOCOMO_FILES for r in by_file[name]]
    by_arm = collections.defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["question_id"]] = row

    def arm_entry(name):
        group = list(by_arm[name].values())
        return {"rows": len(group),
                "shortlist": int(statistics.median(r["retrieved"] for r in group)),
                "accuracy": round(statistics.mean(r["correct"] for r in group), 4)}

    types = collections.defaultdict(dict)
    for name, _ in LOCOMO_ARMS:
        for row in by_arm[name].values():
            types[row["category_name"]].setdefault(name, []).append(row["correct"])

    return {
        "table": "LOCOMO question answering, judged accuracy",
        "published_in": "docs/benchmarks.md, 'What running it actually showed'",
        "dataset": dict(locomo_corpus(by_file),
                        sampling="stratified over the ten conversations; every "
                                 "arm answers the same questions"),
        "scoring": "an LLM reader answers from the arm's passages and an LLM "
                   "judge grades the answer; both are the same model for every "
                   "arm",
        **stamped(rows, "these rows predate the `machine` object run.py now "
                        "stamps on every one. The run log records the "
                        "container benchmarks/README.md describes, but nothing "
                        "in the rows themselves proves it, so no machine is "
                        "claimed here."),
        "sources": [source(raw, name, len(by_file[name]))
                    for name in LOCOMO_FILES],
        "arms": {name: dict(arm_entry(name), asked_for=asked)
                 for name, asked in LOCOMO_ARMS},
        "withdrawn": {name: dict(arm_entry(name), asked_for=asked,
                                 superseded_because=why)
                      for name, (asked, why) in LOCOMO_WITHDRAWN.items()},
        "by_question_type": {
            label: dict({"questions": len(next(iter(scores.values())))},
                        **{name: round(statistics.mean(values), 4)
                           for name, values in scores.items()})
            for label, scores in sorted(types.items())},
        "mcnemar": [dict({"a": a, "b": b}, **paired(by_arm, a, b, correct))
                    for a, b in LOCOMO_PAIRS],
    }


def locomo_cost(raw, by_file, cost_run, memcost, memcost_fresh):
    rows = [r for name in LOCOMO_FILES for r in by_file[name]]
    by_arm = collections.defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    write = {}
    for name, _ in LOCOMO_ARMS:
        units = last_per_unit(by_arm[name], "conversation")
        reingested = sum(
            1 for unit in units
            if len({(r["ingest_seconds"], r["ingest_llm_seconds"], r["store_mb"])
                    for r in by_arm[name] if r["conversation"] == unit}) > 1)
        write[name] = {
            "conversations": len(units),
            "ingest_seconds": round(total(units, "ingest_seconds"), 1),
            "llm_calls": total(units, "ingest_llm_calls"),
            "llm_seconds": round(total(units, "ingest_llm_seconds"), 1),
            "embedded_texts": total(units, "ingest_embed_texts"),
            "cost_usd": billed(units),
            "cost_usd_cost_run": cost_run_usd(cost_run, name),
            # The cost run counted its own calls, and they do not always agree
            # with the accuracy run's rows: MemPalace is 21 here and 20 there
            # for what is the same ingest. The page cites the pair, so both
            # sides of it belong in the artifact.
            "llm_calls_cost_run": (cost_run.get(name, {})
                                   .get("write", {}).get("llm_calls")),
            "store_mb": total(units, "store_mb"),
            "units_ingested_more_than_once": reingested,
        }

    query = {}
    for name, _ in LOCOMO_ARMS:
        group = by_arm[name]
        measured = cost_run.get(name, {}).get("query", {})
        query[name] = {
            "questions": len(group),
            "passages": int(statistics.median(r["retrieved"] for r in group)),
            "context_bytes_mean": round(
                statistics.mean(r["context_chars"] for r in group), 1),
            "prompt_tokens_median": measured.get("median_prompt_tokens"),
        }

    fresh, existing = memcost_fresh, {e["arm"]: e for e in memcost}
    resident = {
        "bm25": dict(resident_pss_mb=existing["bm25"]["resting_pss_mb"],
                     store_mb=existing["bm25"]["store_mb_absolute"],
                     embedder_resident_mb=existing["bm25"]["embedder_resident_mb"],
                     source="memcost.json"),
        "pamin": dict(resident_pss_mb=fresh["resting_pss_mb"],
                      store_mb=fresh["store_mb_added"],
                      embedder_resident_mb=fresh["embedder_resident_mb"],
                      source="memcost-pamin-fresh.json",
                      note="a workspace created for this measurement. "
                           "memcost.json holds a second reading of `pamin` "
                           "over a workspace that already held other corpora: "
                           f"{existing['pamin']['resting_pss_mb']:,} MB "
                           f"resting, {existing['pamin']['store_mb_added']} "
                           "MB added."),
        "mem0": dict(resident_pss_mb=existing["mem0"]["resting_pss_mb"],
                     store_mb=existing["mem0"]["store_mb_absolute"],
                     embedder_resident_mb=existing["mem0"]["embedder_resident_mb"],
                     source="memcost.json"),
        # No PSS reading of MemPalace was taken. The accuracy run sampled its
        # holder process as RSS, which is a different quantity over a
        # different tree, so it is not carried across into this column.
        "mempalace": dict(resident_pss_mb=None, store_mb=None,
                          embedder_resident_mb=None, source=None,
                          note="memcost.py was never run against MemPalace, "
                               "so there is no PSS reading of it here."),
    }

    return {
        "table": "what each LOCOMO arm spends, on the write side and per question",
        "published_in": "docs/benchmarks.md, 'What each arm spends' and "
                        "'Memory and disk'",
        "dataset": locomo_corpus(by_file),
        **stamped(rows, "these rows predate the `machine` object. Everything "
                        "in this file except the call counts, the token counts "
                        "and the bytes is a property of the box it ran on, so "
                        "without a machine those columns cannot be compared "
                        "with a reading from anywhere else."),
        "sources": ([source(raw, name, len(by_file[name]))
                     for name in LOCOMO_FILES]
                    + [source(raw, "tokencost.json", len(cost_run)),
                       source(raw, "memcost.json", len(memcost)),
                       source(raw, "memcost-pamin-fresh.json", 1)]),
        "write": write,
        "write_notes": [
            "`pamin-wide` reads the store `pamin` built and `mem0-k30` shares "
            "`mem0-k10`'s ingest, so their rows are a re-open rather than a "
            "second write. `mempalace-wide` did re-ingest.",
            "Two cost columns because there are two runs. `cost_usd` is the "
            "accuracy run's own counter and is null for the arms whose rows "
            "predate it. `cost_usd_cost_run` is the separate cost-mode run in "
            "tokencost.json over the same ten conversations, and it is where "
            "the MemPalace figure on the page comes from. The two MemPalace "
            "arms made the same twenty-odd calls and that run billed them "
            f"${cost_run_usd(cost_run, 'mempalace')} and "
            f"${cost_run_usd(cost_run, 'mempalace-wide')}, so it is not stable "
            "to within a factor of three and should not be quoted to the cent.",
            "`store_mb` is the sum over all ten conversations of what each "
            "ingest added. The one-conversation figure is a different "
            "measurement and is under `resident_and_disk`.",
            "Ingest seconds and model seconds were taken while a second run "
            "shared the machine and the endpoint, so they describe the pair. "
            "The call counts, the embedded-text counts and the bytes are not "
            "affected.",
        ],
        "query": query,
        "query_notes": [
            "Context bytes are the mean of the passage contents each arm "
            "returned, measured on every question.",
            "Prompt tokens are `cl100k_base` over the reader prompt, the same "
            "template for every arm, and come from the separate cost-mode run "
            "in tokencost.json. They are null for the three arms that run "
            "never covered: `pamin-ledger`, and mem0 at either shortlist. "
            "tokencost.json does hold a `mem0` entry, but at 20 passages -- "
            "it is the withdrawn arm, so it is not reused for either of the "
            "corrected ones.",
        ],
        "resident_and_disk": {
            "unit": "conv-26, 419 turns, the second smallest of the ten",
            "metric": "PSS over the arm's own process tree, because "
                      "PostgreSQL's backends share one buffer pool and RSS "
                      "charges it to each of them",
            "arms": resident,
        },
    }


def locomo_mempalace_no_llm(raw, by_file, cost_run):
    rows = (by_file["raw.jsonl"] + by_file["compare4.jsonl"]
            + by_file["compare.jsonl"] + by_file["compare2.jsonl"])
    by_arm = collections.defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["question_id"]] = row

    # MemPalace's re-ingest of the largest conversation stopped responding on
    # the fill-in pass, so the wide raw arm is one question short. Every figure
    # in this section is over the questions all six arms answered, because a
    # mode compared over 199 against a mode compared over 198 is not a paired
    # comparison.
    compared = ["mempalace", "mempalace-wide", "mempalace-raw",
                "mempalace-raw-wide", "pamin", "pamin-wide"]
    shared = set.intersection(*(set(by_arm[name]) for name in compared))

    accuracy = {name: round(statistics.mean(by_arm[name][q]["correct"]
                                            for q in shared), 4)
                for name in compared}
    ingest = {}
    for name in ["mempalace", "mempalace-wide", "mempalace-raw",
                 "mempalace-raw-wide"]:
        units = last_per_unit(list(by_arm[name].values()), "unit"
                              if name.startswith("mempalace-raw")
                              else "conversation")
        ingest[name] = {
            "conversations": len(units),
            "ingest_seconds": round(total(units, "ingest_seconds"), 1),
            "llm_calls": total(units, "ingest_llm_calls"),
            "cost_usd": billed(units),
            "cost_usd_cost_run": cost_run_usd(cost_run, name),
            "embedded_texts": (total(units, "ingest_embedded")
                               + total(units, "ingest_embed_texts")),
        }

    return {
        "table": "MemPalace with its LLM off, which is the mode it publishes",
        "published_in": "docs/benchmarks.md, 'MemPalace with its LLM off'",
        "dataset": dict(locomo_corpus(by_file), questions=len(shared),
                        questions_note="one short of the set the other LOCOMO "
                                       "tables use: MemPalace's re-ingest of "
                                       "the largest conversation stopped "
                                       "responding on the fill-in pass, so "
                                       "`mempalace-raw-wide` never answered "
                                       "it, and every arm here is scored over "
                                       "the questions all six answered."),
        "machine": machine_of(rows),
        "machine_note": "from raw.jsonl, which carries it. The four arms drawn "
                        "from compare.jsonl, compare2.jsonl and compare4.jsonl "
                        "carry none; they ran on the same container, but the "
                        "rows do not say so.",
        "sources": [source(raw, name, len(by_file[name]))
                    for name in ["raw.jsonl", "compare4.jsonl", "compare.jsonl",
                                 "compare2.jsonl"]],
        "arms": {
            "mempalace": {"mode": "default, LLM entity refinement on",
                          "shortlist": 10, "accuracy": accuracy["mempalace"]},
            "mempalace-wide": {"mode": "default, LLM entity refinement on",
                               "shortlist": 30,
                               "accuracy": accuracy["mempalace-wide"]},
            "mempalace-raw": {"mode": "init --no-llm, as published",
                              "shortlist": 10,
                              "accuracy": accuracy["mempalace-raw"]},
            "mempalace-raw-wide": {"mode": "init --no-llm, as published",
                                   "shortlist": 30,
                                   "accuracy": accuracy["mempalace-raw-wide"]},
            "pamin": {"mode": "no model on the write path in any mode",
                      "shortlist": 10, "accuracy": accuracy["pamin"]},
            "pamin-wide": {"mode": "no model on the write path in any mode",
                           "shortlist": 30, "accuracy": accuracy["pamin-wide"]},
        },
        "ingest": ingest,
        "ingest_note": "the default mode's rows predate the cost counter, so "
                       "`cost_usd` is null for it and `cost_usd_cost_run` "
                       "carries the separate cost-mode run's figure instead. "
                       "The raw mode needs neither: it made no model calls, so "
                       "there is no bill to record.",
        "mcnemar": [dict({"a": a, "b": b}, **paired(by_arm, a, b, correct))
                    for a, b in [("mempalace-raw", "mempalace"),
                                 ("mempalace-raw-wide", "mempalace-wide"),
                                 ("pamin", "mempalace-raw"),
                                 ("pamin-wide", "mempalace-raw-wide")]],
    }


def supersession(raw, by_file):
    rows = [r for name in SUPERSESSION_FILES for r in by_file[name]]
    by_arm = collections.defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["question_id"]] = row

    arms = {}
    for name in SUPERSESSION_ARMS:
        group = list(by_arm[name].values())
        counted = collections.Counter(r["verdict"] for r in group)
        units = last_per_unit(group, "unit")
        arms[name] = {
            "rows": len(group),
            **{kind: round(counted[kind] / len(group), 4) for kind in VERDICTS},
            # Not rounded to an integer: 70 questions make every median the
            # midpoint of two, and truncating it loses the token the page
            # rounds the other way.
            "prompt_tokens_median": statistics.median(
                r["prompt_tokens"] for r in group),
            "store_mb_median": statistics.median(
                u["store_mb"] for u in units.values()),
        }

    return {
        "table": "supersession: is the answer the value that still holds",
        "published_in": "docs/benchmarks.md, 'What it said'",
        "dataset": {"name": "longmemeval-knowledge-update",
                    "questions": len(by_arm["bm25"]),
                    "turns": sum(r["turns"] for r in
                                 last_per_unit(list(by_arm["bm25"].values()),
                                               "unit").values()),
                    "sampling": "LongMemEval-S's knowledge-update category; "
                                "each question carries its own haystack, and "
                                "each arm answers all of them"},
        "scoring": "an LLM judge reads the answer and returns one of current, "
                   "stale or neither -- the value that holds, the value it "
                   "replaced, or something else. `current` is what `correct` "
                   "counts.",
        **stamped(rows, "these rows carry no machine."),
        "sources": [source(raw, name, len(by_file[name]))
                    for name in SUPERSESSION_FILES],
        "arms": arms,
        "arm_notes": [
            "The arms differ in one thing: where the date lives. `bm25` and "
            "`pamin` have none, `pamin-dated` puts it in the passage text, "
            "`pamin-valid` puts it in the ledger without showing the reader, "
            "`pamin-valid-read` shows it.",
            "Two arms ran beside each other to halve the wall clock, sharing "
            "the machine and the endpoint, so the timings and the LLM counters "
            "describe the pair. The verdicts, the prompt sizes and the bytes "
            "are unaffected, which is why only those are reported here.",
            "`store_mb_median` is the disk one question's haystack adds. "
            "`pamin-valid-read` reads the store `pamin-valid` built, so its "
            "figure is a re-open and not a second write.",
        ],
        "mcnemar": [dict({"a": a, "b": b},
                         **paired(by_arm, a, b,
                                  lambda r: r["verdict"] == "current"))
                    for a, b in SUPERSESSION_PAIRS],
        "mcnemar_note": "on `current`, not on the stale rate, because a "
                        "question that moved from stale to neither is not the "
                        "gain this is testing for.",
    }


def supersession_power(raw, by_file, first_run):
    """The same 70 questions, read five times each, majority verdict.

    The first run left `pamin-valid-read` 0.900 against `pamin-dated` 0.814 at
    p = 0.0703 on eight discordant pairs. This re-run exists to find out
    whether that was the reader being noisy, so the two things it adds are the
    two the first run could not report: how often five reads of one question
    agree, and how much of each arm's retrieved set the other arm also
    retrieved.

    Both answers are recorded whichever way they came out, and they came out
    against the hypothesis this run was built on.
    """
    rows = by_file[POWER_FILE]
    by_arm = collections.defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["unit"]] = row

    arms = {}
    for name in POWER_ARMS:
        group = list(by_arm[name].values())
        counted = collections.Counter(r["verdict"] for r in group)
        spread = collections.Counter(r["agreement"] for r in group)
        arms[name] = {
            "rows": len(group),
            **{kind: round(counted[kind] / len(group), 4) for kind in VERDICTS},
            "reads_agreed_mean": round(
                statistics.mean(r["agreement"] for r in group), 4),
            "unanimous": spread[1.0],
            "agreement_spread": {str(k): v for k, v in sorted(spread.items(),
                                                              reverse=True)},
        }

    a, b = "pamin-valid-read", "pamin-dated"
    shared = sorted(set(by_arm[a]) & set(by_arm[b]))
    overlap = [len(set(by_arm[a][u]["topics"]) & set(by_arm[b][u]["topics"]))
               / max(1, len(by_arm[a][u]["topics"])) for u in shared]

    # Where the two arms disagree, with how much of the retrieval they shared.
    # A flip at overlap 1.00 is presentation and nothing else; a flip below it
    # could be either, and the page cannot tell which.
    discordant = []
    for u in shared:
        first, second = by_arm[a][u], by_arm[b][u]
        if correct(first) != correct(second):
            discordant.append({
                "unit": u, "won": a if correct(first) else b,
                f"{a}_verdict": first["verdict"],
                f"{a}_reads_agreed": first["agreement"],
                f"{b}_verdict": second["verdict"],
                f"{b}_reads_agreed": second["agreement"],
                "retrieved_overlap": round(
                    len(set(first["topics"]) & set(second["topics"]))
                    / max(1, len(first["topics"])), 2)})

    agreed = {}
    for name in POWER_ARMS:
        before = {r["unit"]: r["verdict"] for r in first_run
                  if r["arm"] == name}
        both = sorted(set(before) & set(by_arm[name]))
        same = sum(1 for u in both if before[u] == by_arm[name][u]["verdict"])
        agreed[name] = {"identical_verdicts": same, "of": len(both),
                        "rate": round(same / len(both), 4) if both else None}

    return {
        "table": "supersession, re-measured with five reads and a majority "
                 "verdict",
        "published_in": "docs/benchmarks.md, 'What it said'",
        "question": "is `pamin-valid-read` better than `pamin-dated`, or was "
                    "the first run's p = 0.0703 a reader that could not answer "
                    "the same question twice",
        "dataset": {"name": "longmemeval-knowledge-update",
                    "questions": len(by_arm[a]),
                    "reads_per_question": 5,
                    "sampling": "the same 70 questions the first run used"},
        "scoring": "five reads of one captured retrieval, each judged current, "
                   "stale or neither; the arm's verdict is the majority, and a "
                   "tie resolves to the least favourable of the tied verdicts.",
        **stamped(rows, "these rows carry no machine."),
        "sources": [source(raw, POWER_FILE, len(rows))],
        "arms": arms,
        "mcnemar": [dict({"a": x, "b": y},
                         **paired(by_arm, x, y,
                                  lambda r: r["verdict"] == "current"))
                    for x, y in POWER_PAIRS],
        "retrieved_overlap": {
            "of": f"{a} against {b}",
            "median": round(statistics.median(overlap), 4),
            "mean": round(statistics.mean(overlap), 4),
            "min": round(min(overlap), 4),
            "identical_sets": sum(1 for x in overlap if x == 1.0),
            "questions": len(overlap)},
        "discordant_pairs": discordant,
        "against_the_first_run": agreed,
        "notes": [
            "The three arm totals and the decisive p value come out exactly as "
            "the first run reported them -- 0.814, 0.700, 0.900 and 7 to 1 at "
            "p = 0.0703. The per-question verdicts do not, so this is two runs "
            "that happen to sum to the same numbers rather than one run "
            "reproduced.",
            "Reader noise was not the limit. 55 to 57 of the 70 questions are "
            "unanimous across five reads, and the discordant count did not "
            "move, so the eight pairs are the arms disagreeing rather than the "
            "reader being unable to repeat itself. What limits this comparison "
            "is 70 questions.",
            "One more discordant pair in the same direction would settle it: "
            "8 to 1 on nine pairs is p = 0.0391. LongMemEval-S has 78 "
            "knowledge-update questions and 70 of them qualify, so there is no "
            "ninth pair to be had from this corpus.",
            "The retrieval confound is real and partial. `pamin-dated` writes "
            "the date into the indexed content, and only 16 of 70 questions "
            "return the same topic set to both arms, so part of the gap is "
            "retrieval rather than presentation. Three of the seven pairs that "
            "go to `pamin-valid-read` are at overlap 1.00, where nothing but "
            "what the reader was shown can have moved.",
        ],
    }


def latency(raw, rows):
    """Query latency per arm per shortlist, and the reader curve beside it.

    Retrieval and the reader go in one summary because the point of the pair
    is the comparison between them: a sixty-millisecond difference in
    retrieval is not a difference anyone experiences if the reader takes five
    seconds either way, and reporting the first without the second is how this
    page used to overstate what retrieval latency buys.

    Every cell is the median over rounds rather than one run's number. The
    previous table was a single sitting, and that sitting was taken beside a
    shell of this harness's own spin-looping on one of four cores -- which
    cost the arms that embed over HTTP roughly double and cost this project's
    own arm nothing, so the error was invisible in the row most likely to be
    checked. Rounds are what makes that visible: the spread is published.
    """
    retrieval, reader, skipped = {}, {}, []
    for row in rows:
        if row["arm"].startswith("  "):   # a drift warning, not an arm
            continue
        if row["arm"].startswith("reader, "):
            reader.setdefault(row["prompt_tokens"], []).append(row)
            continue
        if row["n"] != 199:
            skipped.append(f"{row['arm']} at {row['limit']}, n={row['n']}")
            continue
        retrieval.setdefault(row["arm"], {}).setdefault(row["limit"], []).append(row)

    def over(got, key):
        return round(statistics.median(g[key] for g in got), 1)

    summarised = {}
    for arm, limits in retrieval.items():
        for limit, got in sorted(limits.items()):
            p50 = [g["p50_ms"] for g in got]
            median = statistics.median(p50)
            summarised.setdefault(arm, {})[f"at_{limit}"] = {
                "n": 199, "rounds": len(got),
                "p50_ms": round(median, 1),
                "p95_ms": over(got, "p95_ms"),
                "mean_ms": over(got, "mean_ms"),
                "p50_per_round": p50,
                "spread": f"{(max(p50) - min(p50)) / median:.0%}",
            }

    return {
        "table": "latency, timed at the same layer on every arm",
        "published_in": "docs/benchmarks.md, 'Latency, timed at the same layer'",
        "corpus": "LOCOMO, the same ten conversations and 199 questions the "
                  "accuracy run uses, rebuilt for this run because the "
                  "workspace that held them was lost with its PostgreSQL data "
                  "directory",
        "method": "every arm at the boundary an application calls, every unit "
                  "warmed three times before anything is timed, each arm run a "
                  "second time last -- two passes that disagree by more than "
                  "half the median print a warning, and none did -- and every "
                  "published cell the median of at least three such rounds, "
                  "with the per-round figures and the spread kept here.",
        **stamped(rows, "these rows carry no machine."),
        "sources": [source(raw, LATENCY_FILE, len(rows))],
        "rows_skipped": skipped,
        "retrieval": summarised,
        "reader": [{"prompt_tokens": tokens, "rounds": len(got),
                    "calls": got[0]["n"],
                    "median_seconds": round(statistics.median(
                        g["p50_ms"] for g in got) / 1000, 2),
                    "min_seconds": round(min(g["min_ms"] for g in got) / 1000, 2),
                    "max_seconds": round(max(g["max_ms"] for g in got) / 1000, 2)}
                   for tokens, got in sorted(reader.items())],
        "notes": [
            "The single largest correction on this page, and it was ours. The "
            "previous figures -- MemPalace 75.4/79.7 ms, mem0 103.4/105.6, the "
            "embedding call 25.8 -- were measured beside a shell of this "
            "session's left spin-looping on one of the box's four cores. On a "
            "quiet box the same arms read 36.3/54.7, 62.9/64.6 and 14.6, so "
            "this project's published lead over MemPalace was roughly double "
            "what it is. The 12:39 rows are kept out of this summary and the "
            "ratio they produced is withdrawn.",
            "What made it hard to see: losing one of four cores cost the two "
            "arms that embed their query over HTTP roughly double, and cost "
            "this project's own socket arm nothing measurable -- 25.7, 26.2, "
            "25.7 across quiet and noisy rounds alike. The arm a reader would "
            "check first was the one arm the noise did not touch. Why the "
            "in-process path is insensitive where the HTTP path is not has not "
            "been established here and is not claimed.",
            "Every arm is 199 questions. mem0's row used to be 20: not because "
            "mem0 clears its store per conversation, which is what the page "
            "said, but because this harness emptied the qdrant directory "
            "before each ingest so that `store_mb` would be one "
            "conversation's bytes. `BENCH_MEM0_KEEP=1` keeps all ten, and the "
            "20-question rows that remain in the raw file are listed under "
            "`rows_skipped` rather than averaged in.",
            "`of which, one embedding HTTP call` is not a memory system. It is "
            "one call to the shared endpoint, which mem0 and MemPalace pay "
            "inside every search and this project does not, because it holds "
            "the model in the process that holds the index. At 14.6 ms it is "
            "40% of MemPalace's ten-passage figure and 23% of mem0's, and it "
            "is the floor under both.",
            "It is served from `embedder.py`, a process that holds one ONNX "
            "session and spawns nothing, rather than from the shim that also "
            "forks `claude -p` for chat. That process was built to test "
            "whether contention with those forks explained the embedding "
            "call's instability. It did not: the two endpoints measure the "
            "same, 27.7 ms against 29.4 through urllib and 31.5 against 31.9 "
            "through the OpenAI client. The instability was the spin loop.",
            "Widening the shortlist from ten passages to thirty costs this "
            "project nothing measurable (25.7 to 25.7) and mem0 nothing (62.9 "
            "to 64.6), and costs MemPalace about half again -- +26%, +46%, "
            "+64%, +52%, +52% over five rounds. An earlier version of this "
            "page withdrew that claim on the strength of three noisy rounds; "
            "it is reinstated on five quiet ones, as a median rather than as "
            "a single figure.",
            "`--rerank fast` on the socket arm, which is the shipped default. "
            "The server was warm: a first pass on a freshly started server "
            "measured 123.4 ms against a repeat of 25.8, the harness's own "
            "drift guard caught it, and the run was taken again rather than "
            "the faster half of it being kept.",
            "The reader curve is 12 calls a size to a hosted model, and the "
            "median is what the page quotes because the tail belongs to the "
            "provider's queue rather than to the context length -- one call "
            "in these rounds took 167 s. The minimum and maximum are recorded "
            "here so that tail is visible without being published as a "
            "property of context length.",
        ],
    }


def longmemeval_retrieval(raw, pamin_rows, recorded):
    """Session retrieval, at the loose metric the field publishes and a strict one.

    The BM25 column cannot be recomputed from the rows: `lme_bm25.json` stored
    three metrics per question and `recall_any@k` was not among them, so the
    baseline has to be re-ranked from the LongMemEval corpus. That is what
    `longmemeval_recall.py` does and it is imported rather than reimplemented
    here -- two copies of a retriever drift, and then two tables disagree and
    neither is wrong.
    """
    rebuilt = longmemeval_recall.rebuild(pamin_rows)

    # The rebuild has to be checked against something, and the one metric both
    # the original BM25 run and the rebuild computed is `recall_all@5`. If
    # these two ever part, the rebuild is ranking a different corpus.
    n = len(pamin_rows)
    as_recorded = {
        side: round(sum(recorded[r["question_id"]][side]["recall_all@5"]
                        for r in pamin_rows) / n, 4)
        for side in ("bm25", "pamin")}

    return {
        "table": "LongMemEval-S session retrieval, no model anywhere",
        "published_in": "docs/benchmarks.md, 'LongMemEval: the published "
                        "metric is saturated'",
        "dataset": {"name": "longmemeval-s", "questions": n,
                    "questions_with_more_than_one_gold_session":
                        rebuilt["multi_gold"],
                    "sampling": "stratified from LongMemEval-S's 500, "
                                "proportional over the same six types"},
        "scoring": "is the labelled session in the top k. No reader and no "
                   "judge, which is what makes this the one figure here that "
                   "can sit beside a published one.",
        "machine": None,
        "machine_note": "lme_pamin.jsonl predates the `machine` object. The "
                        "BM25 column is not a recorded row at all "
                        "-- it is re-ranked by longmemeval_recall.py, so it is "
                        "a property of whichever machine regenerated this "
                        "file, which for a metric with no timing in it changes "
                        "nothing.",
        "sources": [source(raw, "lme_pamin.jsonl", n),
                    source(raw, "lme_bm25.json", len(recorded))],
        "metrics": {
            "recall_any@5": {"note": "the metric the field publishes",
                             "bm25": rebuilt["bm25"]["any@5"],
                             "pamin": rebuilt["pamin"]["any@5"]},
            "recall_any@10": {"bm25": rebuilt["bm25"]["any@10"],
                              "pamin": rebuilt["pamin"]["any@10"]},
            "recall_all@5": {"note": "every gold session in the top five",
                             "bm25": rebuilt["bm25"]["all@5"],
                             "pamin": rebuilt["pamin"]["all@5"]},
            # Stored per question by the original run rather than rebuilt, so
            # unlike the rows above it this one is a recorded measurement and
            # not a re-ranking. The README publishes it and nothing here
            # carried it, which left one published figure with no artifact.
            "ndcg_any@10": {"note": "recorded by the original run, not rebuilt",
                            **{side: round(
                                sum(recorded[r["question_id"]][side]
                                    ["ndcg_any@10"] for r in pamin_rows) / n, 4)
                               for side in ("bm25", "pamin")}},
        },
        "recall_any@5_by_type": rebuilt["by_type"],
        "rebuild_check": {
            "metric": "recall_all@5",
            "as_recorded_in_lme_bm25_json": as_recorded,
            "as_rebuilt": {"bm25": rebuilt["bm25"]["all@5"],
                           "pamin": rebuilt["pamin"]["all@5"]},
            "note": "the per-question ndjson the original BM25 run wrote is "
                    "gone, so the baseline is re-ranked from the LongMemEval "
                    "file. This is the metric both computed; if the two rows "
                    "ever part, the rebuild is ranking a different corpus.",
        },
    }


def write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as sink:
        json.dump(payload, sink, indent=2, ensure_ascii=False)
        sink.write("\n")
    print(f"  wrote {os.path.relpath(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", help="the directory the run's raw rows are in")
    args = parser.parse_args()

    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    by_file = {name: read_rows(args.raw, name)
               for name in LOCOMO_FILES + SUPERSESSION_FILES
               + [POWER_FILE, "raw.jsonl"]}

    cost_run = read_json(args.raw, "tokencost.json")

    write(os.path.join(results, "locomo", "summary-accuracy.json"),
          locomo_accuracy(args.raw, by_file))
    write(os.path.join(results, "locomo", "summary-cost.json"),
          locomo_cost(args.raw, by_file, cost_run,
                      read_json(args.raw, "memcost.json"),
                      read_json(args.raw, "memcost-pamin-fresh.json")))
    write(os.path.join(results, "locomo", "summary-mempalace-no-llm.json"),
          locomo_mempalace_no_llm(args.raw, by_file, cost_run))
    write(os.path.join(results, "longmemeval", "summary-supersession.json"),
          supersession(args.raw, by_file))
    write(os.path.join(results, "longmemeval",
                       "summary-supersession-power.json"),
          supersession_power(
              args.raw, by_file,
              [r for name in SUPERSESSION_FILES for r in by_file[name]]))
    write(os.path.join(results, "locomo", "summary-latency.json"),
          latency(args.raw, read_rows(args.raw, LATENCY_FILE)))
    write(os.path.join(results, "longmemeval", "summary-session-retrieval.json"),
          longmemeval_retrieval(
              args.raw, read_rows(args.raw, "lme_pamin.jsonl"),
              {e["question_id"]: e for e in read_json(args.raw, "lme_bm25.json")}))


if __name__ == "__main__":
    main()
