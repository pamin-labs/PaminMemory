#!/usr/bin/env python3
"""Does surfacing valid time beat writing the date into the passage?

The supersession run left this one open. `pamin-valid-read` answered 0.900
against `pamin-dated`'s 0.814, but only eight of seventy questions were
discordant -- seven one way -- and McNemar put that at p = 0.0703. The lean is
the right size to be real and the sample is too small to say so. At the same
7:1 lean, twice the discordant pairs would be p = 0.0042, so the question is
answerable; it needs a sharper instrument rather than a different corpus, and
LongMemEval has no more knowledge-update questions to give.

Two things this does that the first run did not:

**It reads each question five times and takes the majority.** One reader
sample per arm makes each question a single Bernoulli draw, and a question the
two arms genuinely disagree about can land concordant by luck -- which loses a
discordant pair and with it the only information McNemar uses. Five draws and a
majority make each cell a property of the arm rather than of one sample. The
agreement rate among the five is reported, because it is the measurement noise
this whole comparison sits on and it has never been quantified here.

**It records which topics came back, not just how many.** `pamin-dated` puts
the date into the *indexed* content, so it may be retrieving different passages
rather than merely showing the reader more -- and the first run stored only a
count, so that could not be checked afterwards. If the two arms retrieve
different sets, the difference between them is not the one the page claims it
is. The overlap is measured and reported whichever way it comes out.

Retrieval is captured once per arm and replayed to the reader, so the five
reads see identical context and differ only in the model's own sampling.

    python3 benchmarks/power.py --units 70 --reads 5 --out /tmp/power.jsonl
"""
import argparse
import collections
import concurrent.futures as futures
import json
import math
import os
import statistics
import time

import arms as arms_module
import run as run_module
from datasets import supersession

ARMS = ("pamin-dated", "pamin-valid", "pamin-valid-read")
DECISIVE = ("pamin-valid-read", "pamin-dated")


def majority(verdicts):
    """The verdict the reads agree on, and how strongly.

    Ties resolve to the least favourable outcome rather than the first seen: a
    question the reader cannot answer the same way twice should not be scored
    as though it could.
    """
    counted = collections.Counter(verdicts)
    best = max(counted.values())
    tied = sorted(v for v, n in counted.items() if n == best)
    order = {"stale": 0, "neither": 1, "current": 2}
    return min(tied, key=lambda v: order.get(v, 0)), best / len(verdicts)


def mcnemar(pairs):
    ab = sum(1 for a, b in pairs if a and not b)
    ba = sum(1 for a, b in pairs if b and not a)
    n = ab + ba
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(math.comb(n, k) for k in range(min(ab, ba) + 1)) / 2 ** n)
    return ab, ba, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=70)
    parser.add_argument("--reads", type=int, default=5)
    parser.add_argument("--out", default="/tmp/bench/power/power.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, done = run_module.load(args.out)
    if done:
        print(f"resuming: {len(rows)} rows already in {args.out}", flush=True)
    sink = open(args.out, "a")
    where = run_module.machine()

    for entry in supersession.read(limit=args.units):
        unit = supersession.unit_id(entry)
        if all((name, unit, unit) in done for name in ARMS):
            continue
        turns = supersession.turns_of(entry)
        question = entry["question"]
        _qid, qa, reference = supersession.questions(entry)[0]

        for name in ARMS:
            if (name, unit, unit) in done:
                continue
            arm = arms_module.ARMS[name]()
            started = time.time()
            arm.ingest(unit, turns)
            ingested = time.time() - started

            passages = arm.recall(question)
            # Which topics, not how many. `pamin-dated` indexes the date, so it
            # can retrieve a different set, and a count cannot show that.
            topics = getattr(arm, "_last", None)
            if topics is None:
                out = arm._run(arm.project, ["search", question, "--limit",
                                             str(arms_module.TOP_K), "--json"])
                topics = [h["topic"] for h in json.loads(out)["hits"]]
            assert passages, f"{name} retrieved nothing for {unit}"

            prompt = supersession.READER.format(
                context="\n".join(f"- {p}" for p in passages), question=question)
            # The five reads are independent by construction -- that is the
            # whole point of taking five -- so they run at once. Serially this
            # run is sixteen hours of mostly waiting on an endpoint that
            # answers concurrently; the shim's only lock guards its counters.
            # Nothing about the sample changes: each chain still reads and then
            # judges its own answer, and the results are put back in order so a
            # rerun with the same seeds prints the same row.
            def one_read(_):
                predicted = run_module.chat(prompt)
                return supersession.judged(
                    run_module.chat, question, reference, predicted)["verdict"], predicted

            with futures.ThreadPoolExecutor(max_workers=args.reads) as pool:
                pairs = list(pool.map(one_read, range(args.reads)))
            verdicts = [v for v, _ in pairs]
            answers = [a for _, a in pairs]
            verdict, agreement = majority(verdicts)

            row = {"arm": name, "unit": unit, "question_id": unit,
                   "machine": where, "mode": "quality",
                   "label": supersession.label(qa),
                   "verdict": verdict, "correct": 1.0 if verdict == "current" else 0.0,
                   "agreement": round(agreement, 3), "verdicts": verdicts,
                   "answers": answers, "topics": topics,
                   "ingest_seconds": round(ingested, 1),
                   "cost_measurable": False}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            rows.append(row)
            done.add((name, unit, unit))
            print(f"  {unit} {name:<17} {verdict:<8} agree {agreement:.2f}  "
                  f"ingest {ingested:.0f}s", flush=True)
    sink.close()
    summarise(rows)


def summarise(rows):
    by = collections.defaultdict(dict)
    for row in rows:
        by[row["arm"]][row["unit"]] = row
    present = [a for a in ARMS if a in by]
    if not present:
        return print("no rows")

    print(f"\n{'arm':<20}{'n':>4}{'current':>9}{'stale':>8}{'neither':>9}"
          f"{'reads agreed':>14}")
    for name in present:
        group = list(by[name].values())
        kinds = collections.Counter(r["verdict"] for r in group)
        n = len(group)
        print(f"{name:<20}{n:>4}{kinds['current']/n:>9.3f}{kinds['stale']/n:>8.3f}"
              f"{kinds['neither']/n:>9.3f}"
              f"{statistics.mean(r['agreement'] for r in group):>14.3f}")

    a, b = DECISIVE
    if a in by and b in by:
        shared = sorted(set(by[a]) & set(by[b]))
        ab, ba, p = mcnemar([(by[a][u]["correct"] == 1, by[b][u]["correct"] == 1)
                             for u in shared])
        print(f"\n{a} vs {b}, {len(shared)} paired questions:")
        print(f"  {a} only {ab}, {b} only {ba}, p = {p:.4f}")

        overlap = [len(set(by[a][u]["topics"]) & set(by[b][u]["topics"])) /
                   max(1, len(by[a][u]["topics"])) for u in shared]
        print(f"\nretrieved-set overlap between them: median "
              f"{statistics.median(overlap):.2f}, min {min(overlap):.2f}")
        print("  1.00 means they read the same passages and differ only in what "
              "the reader was shown;")
        print("  below that, part of any difference is retrieval and not "
              "presentation.")


if __name__ == "__main__":
    main()
