#!/usr/bin/env python3
"""Does trimming the JSON an agent reads cost it any accuracy?

Removing a field saves tokens whether or not it was useful, so the saving is
never the question. The question is what the reader does without it, and that
splits three ways: does it answer as well, does it answer as fast, and does it
spend more of its own output getting there.

So retrieval is held completely fixed. Each question's `pamin search --json` is
captured once and replayed to the reader in two shapes:

  full   what ships today -- every field, indented
  trim   the proposal: no `topic_state`, no `source_span`, no `weight` or
         `contribution` inside `why` (cli.md documents both as derivable), open
         intervals absent rather than null, and no indentation

Nothing else differs. Same hits, same order, same contents, same reader, same
judge -- which the premise check below asserts rather than assumes, because two
shapes that quietly disagreed about what was retrieved would produce a
difference in accuracy that had nothing to do with the fields.

    python3 benchmarks/payload.py --units 70 --out /tmp/payload.jsonl

What it said, over all 70:

    shape      n  current   stale  neither   prompt  answer  seconds
    full      70    0.843   0.143    0.014     5148       2     6.53
    trim      70    0.871   0.129    0.000     3322       2     6.17

    McNemar over 70 paired questions: full only 5, trim only 7, p = 0.7744

The trimmed shape costs 35% fewer prompt tokens and is indistinguishable on
every other axis. It is nominally ahead on all of them, and nobody should read
anything into that: twelve discordant questions out of seventy is a coin the
wrong way up twice. What the run can support is the negative -- there is no
sign of a cost -- and seventy questions would not detect a two-point one.

Two things this does not say, and the second is a warning about the harness
rather than a result:

  The seconds are not a clean measurement. The endpoint is shared and the
  machine is not idle, which is why the shapes are compared on tokens and
  verdicts and the timing is printed rather than concluded from.

  **Both arms are contaminated by ingest order, equally.** The payload carries
  `recorded_at`, and this harness imports a haystack in file order. The
  haystacks are not in date order -- none of the 70 are -- but the two gold
  sessions are, in 69 of 70, so the recording time happens to sort the pair
  that matters. That is why both arms here score far above the 0.700 the same
  retrieval scores when the reader is handed bare contents, and it is an
  accident of how LongMemEval lays its answer sessions out. It cancels in the
  comparison, which is what this file is for, and it would not cancel in a
  claim about how well `pamin` answers -- so no such claim is made from it.
"""
import argparse
import collections
import json
import os
import statistics
import time

import arms as arms_module
import run as run_module
from datasets import supersession

DERIVABLE = ("weight", "contribution")
DROPPED = ("topic_state", "source_span")

READER = supersession.READER


def trim(payload):
    """The proposed shape. Only what a caller cannot recover for itself."""
    hits = []
    for hit in payload["hits"]:
        kept = {}
        for key, value in hit.items():
            if key in DROPPED or value is None:
                continue
            if key == "why":
                value = [{k: v for k, v in entry.items() if k not in DERIVABLE}
                         for entry in value]
            kept[key] = value
        hits.append(kept)
    return {"query": payload["query"], "hits": hits}


def shapes(payload):
    """Both renderings, and the assertion that they retrieved the same thing."""
    small = trim(payload)
    assert [h["content"] for h in payload["hits"]] == \
           [h["content"] for h in small["hits"]], \
        "the two shapes disagree about what came back, so any difference in " \
        "accuracy is not about the fields"
    return {"full": json.dumps(payload, indent=2),
            "trim": json.dumps(small, separators=(",", ":"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=int, default=70)
    parser.add_argument("--out", default="/tmp/bench/payload/payload.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, done = run_module.load(args.out)
    if done:
        print(f"resuming: {len(rows)} rows already in {args.out}", flush=True)
    sink = open(args.out, "a")
    where = run_module.machine()

    for entry in supersession.read(limit=args.units):
        unit = supersession.unit_id(entry)
        if all((shape, unit, unit) in done for shape in ("full", "trim")):
            continue

        arm = arms_module.Pamin()
        started = time.time()
        arm.ingest(unit, supersession.turns_of(entry))
        print(f"  {unit}: ingested in {time.time() - started:.0f}s", flush=True)

        # Once. Both shapes are renderings of this one retrieval.
        question = entry["question"]
        raw = json.loads(arm._run(arm.project, [
            "search", question, "--limit", str(arms_module.TOP_K), "--json"]))
        rendered = shapes(raw)

        _qid, qa, reference = supersession.questions(entry)[0]
        for shape, text in rendered.items():
            if (shape, unit, unit) in done:
                continue
            prompt = READER.format(context=text, question=question)
            began = time.time()
            predicted = run_module.chat(prompt)
            answered = time.time() - began
            verdict = supersession.judged(run_module.chat, question, reference,
                                          predicted)
            row = {"arm": shape, "unit": unit, "question_id": unit,
                   "machine": where, "mode": "quality",
                   "label": supersession.label(qa),
                   "prompt_tokens": run_module.tokens_in(prompt),
                   "answer_tokens": run_module.tokens_in(predicted),
                   "answer_seconds": round(answered, 2),
                   "predicted": predicted, **verdict}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            rows.append(row)
            done.add((shape, unit, unit))
            print(f"    {shape:<5} {row['prompt_tokens']:>6} prompt  "
                  f"{row['answer_tokens']:>4} answer  {answered:>5.1f}s  "
                  f"{row['verdict']}", flush=True)
    sink.close()
    summarise(rows)


def summarise(rows):
    by = collections.defaultdict(list)
    for row in rows:
        by[row["arm"]].append(row)
    if not by:
        return print("no rows")
    print(f"\n{'shape':<8}{'n':>4}{'current':>9}{'stale':>8}{'neither':>9}"
          f"{'prompt':>9}{'answer':>8}{'seconds':>9}")
    for shape in sorted(by):
        group = by[shape]
        kinds = collections.Counter(r["verdict"] for r in group)
        n = len(group)
        print(f"{shape:<8}{n:>4}{kinds['current'] / n:>9.3f}{kinds['stale'] / n:>8.3f}"
              f"{kinds['neither'] / n:>9.3f}"
              f"{statistics.median(r['prompt_tokens'] for r in group):>9.0f}"
              f"{statistics.median(r['answer_tokens'] for r in group):>8.0f}"
              f"{statistics.median(r['answer_seconds'] for r in group):>9.2f}")

    paired = sorted(set(r["unit"] for r in by["full"]) &
                    set(r["unit"] for r in by["trim"]))
    if not paired:
        return
    full = {r["unit"]: r for r in by["full"]}
    trim_ = {r["unit"]: r for r in by["trim"]}
    # `correct` is a float, so these have to be counted as predicates rather
    # than summed as they come: `1.0 and True` is `True` and `0.0 and True` is
    # `0.0`, and a total that mixes them is a float that `math.comb` refuses.
    only_full = sum(1 for u in paired
                    if full[u]["correct"] == 1 and trim_[u]["correct"] == 0)
    only_trim = sum(1 for u in paired
                    if trim_[u]["correct"] == 1 and full[u]["correct"] == 0)
    import math
    n = only_full + only_trim
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k)
                                            for k in range(min(only_full, only_trim) + 1)) / 2 ** n)
    print(f"\nMcNemar over {len(paired)} paired questions: "
          f"full only {only_full}, trim only {only_trim}, p = {p:.4f}")


if __name__ == "__main__":
    main()
