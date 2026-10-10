#!/usr/bin/env python3
"""Summarise the shim-run LOCOMO comparison into a committed artifact.

This is a second, narrower summariser rather than an extension of
`summarise.py`. That script is wired to one specific historical run --
hardcoded filenames (`compare.jsonl` .. `compare5.jsonl`), hardcoded arm names
(`mem0-k10`, `mem0-k30`, `pamin-ledger`, `mempalace-wide`, ...) and auxiliary
files (`tokencost.json`, `memcost.json`) that run alone produced. This run has
a different shape -- one file (`results/locomo/quality.jsonl`), five arms
(`bm25`, `pamin`, `pamin-wide`, `mem0`, `mempalace`), each at the single
shortlist it asked for -- and forcing it through the other script's file list
would either crash or silently summarise the wrong rows.

    python3 benchmarks/summarise_shim_locomo.py

Reads `results/locomo/quality.jsonl` in place and writes
`results/locomo/summary-accuracy-shim-2026-10.json` beside the existing
summaries from the earlier run. Nothing here is copied from docs/benchmarks.md;
every figure is recomputed from the rows, the same rule `summarise.py`
documents.
"""
import collections
import json
import math
import os
import statistics

ARMS = ["bm25", "pamin", "pamin-wide", "mem0", "mempalace"]

# A standalone probe, not derived from quality.jsonl -- the raw rows never
# recorded one HTTP embedding call's own latency, only the text counts
# (`ingest_embedded`) and the LLM-only call seconds. Taken against the live
# shim.py right after this run finished, same box, eight calls to
# /v1/embeddings, one input each. The first call is cold (the ONNX session
# loads lazily on first use); the other seven are what mem0 and MemPalace pay
# per query, since both point their embedder at this same endpoint over HTTP
# while `pamin` embeds inside its own process and never makes this call.
EMBEDDING_HTTP_PROBE = {
    "endpoint": "/v1/embeddings on benchmarks/shim.py",
    "when": "immediately after this run finished, same box",
    "calls_ms": [45.6, 22.9, 22.0, 21.1, 20.6, 20.5, 21.3, 21.3],
    "median_ms_excluding_cold_first": 21.3,
    "note": "one box, one sitting, taken after the fact rather than during a "
            "controlled multi-round run -- not the rigor of the dedicated "
            "latency re-run's own 14.6 ms figure, which this is in the same "
            "order of magnitude as. It exists to answer one narrow question: "
            "how much of mem0's and MemPalace's recall_seconds in this run "
            "is a shared endpoint `pamin` never calls, not to replace the "
            "dedicated latency measurement.",
}
PAIRS = [("pamin", "bm25"), ("pamin-wide", "pamin"), ("pamin-wide", "bm25"),
         ("mem0", "bm25"), ("mempalace", "bm25"), ("pamin-wide", "mem0"),
         ("pamin-wide", "mempalace"), ("mem0", "mempalace"),
         ("pamin", "mem0"), ("pamin", "mempalace")]


def mcnemar(a_only, b_only):
    n = a_only + b_only
    p = 1.0 if n == 0 else min(
        1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
        / 2 ** n)
    return {"discordant": n, "a_only": a_only, "b_only": b_only, "p": round(p, 4)}


def paired(by_arm, a, b, won):
    shared = set(by_arm[a]) & set(by_arm[b])
    return mcnemar(
        sum(1 for q in shared if won(by_arm[a][q]) and not won(by_arm[b][q])),
        sum(1 for q in shared if not won(by_arm[a][q]) and won(by_arm[b][q])))


def correct(row):
    return row["correct"] > 0


def last_per_unit(rows, key):
    out = {}
    for row in rows:
        out[row[key]] = row
    return out


def machine_of(rows):
    """The machine(s) these rows were taken on.

    A single container identity across a run this long is not realistic here:
    benchmarks/README.md documents repeated, unpredictable reclaims, and this
    run was resumed from checkpoint after several of them. What is stable
    across every reclaim is the shape (4 cores, 16 GB, same `vm` host); only
    the kernel build string moves. So this reports `None` plus every distinct
    machine seen, rather than either picking one or crashing -- the timing
    and resource columns here are not a single-sitting measurement and should
    not be read as one, which is why accuracy is reported separately from
    latency/cost in the table this feeds.
    """
    seen = {json.dumps(r["machine"], sort_keys=True)
            for r in rows if r.get("machine")}
    if not seen:
        return {"machine": None,
                "machine_note": "no row carries a machine object"}
    distinct = [json.loads(s) for s in seen]
    if len(distinct) == 1:
        return {"machine": distinct[0]}
    return {"machine": None,
            "machine_note": "this run spanned multiple container reclaims; "
                            "every reclaim kept the same shape (4 cores, "
                            "16 GB, host \"vm\") but a different kernel "
                            "build, so no single machine object covers the "
                            "whole run. Distinct machines seen are listed "
                            "here instead, which is why latency and ingest "
                            "timing in this summary are not read as a single "
                            "clean sitting -- see run_conditions.",
            "distinct_machines": distinct}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "results", "locomo", "quality.jsonl")
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows if r["arm"] in ARMS]

    by_arm = collections.defaultdict(dict)
    by_arm_list = collections.defaultdict(list)
    for row in rows:
        by_arm[row["arm"]][row["question_id"]] = row
        by_arm_list[row["arm"]].append(row)

    # Anthropic's published Sonnet list price: $3 per million input tokens.
    # Applied only to the reader's prompt tokens, which is the one input-side
    # quantity every row actually measured (`tokens_in(prompt)`, cl100k_base).
    # It is not a complete query-side bill: the judge call's own prompt was
    # never counted, and the shim never reports real completion-token counts
    # for either call (`run.py`'s `chat()` always gets `completion_tokens: 0`
    # back, by shim.py's own design -- see its "Real counts are not available
    # through the CLI" note), so no output-token cost can be added for either
    # call. This is a lower bound on the per-question bill, not the bill.
    SONNET_INPUT_RATE_USD_PER_TOKEN = 3.00 / 1_000_000

    def ingest_totals(arm):
        units = last_per_unit(by_arm_list[arm], "unit")
        return {
            "ingest_seconds_total": round(
                sum(u["ingest_seconds"] for u in units.values()), 1),
            "ingest_llm_calls_total": sum(
                u["ingest_llm_calls"] for u in units.values()),
            "ingest_cost_usd_total": round(
                sum(u.get("ingest_cost_usd", 0) for u in units.values()), 4),
        }

    def reader_input_cost(arm):
        tokens = sum(r.get("prompt_tokens", 0) for r in by_arm_list[arm])
        return {"reader_prompt_tokens_total": tokens,
                "reader_input_cost_usd": round(
                    tokens * SONNET_INPUT_RATE_USD_PER_TOKEN, 4)}

    def with_partial_total(entry):
        return dict(entry, partial_total_cost_usd=round(
            entry["ingest_cost_usd_total"] + entry["reader_input_cost_usd"], 4))

    accuracy = {
        arm: with_partial_total(dict({
            "rows": len(group),
            "shortlist": int(statistics.median(r["retrieved"] for r in group.values())),
            "accuracy": round(statistics.mean(r["correct"] for r in group.values()), 4),
        }, **ingest_totals(arm), **reader_input_cost(arm)))
        for arm, group in by_arm.items()
    }

    types = collections.defaultdict(dict)
    for arm in ARMS:
        for row in by_arm[arm].values():
            types[row["label"]].setdefault(arm, []).append(row["correct"])
    by_label = {
        label: dict({"questions": len(next(iter(scores.values())))},
                    **{arm: round(statistics.mean(values), 4)
                       for arm, values in scores.items()})
        for label, scores in sorted(types.items())
    }

    ingest = {}
    for arm in ARMS:
        units = last_per_unit(by_arm_list[arm], "unit")
        ingest[arm] = {
            "conversations": len(units),
            "ingest_seconds_median": statistics.median(
                u["ingest_seconds"] for u in units.values()),
            "llm_calls_median": statistics.median(
                u["ingest_llm_calls"] for u in units.values()),
            "llm_seconds_median": statistics.median(
                u["ingest_llm_seconds"] for u in units.values()),
            "embedded_median": statistics.median(
                u["ingest_embedded"] for u in units.values()),
            "store_mb_median": statistics.median(
                u["store_mb"] for u in units.values()),
            "cost_usd_total": round(
                sum(u.get("ingest_cost_usd", 0) for u in units.values()), 4),
        }

    query = {}
    for arm in ARMS:
        group = by_arm_list[arm]
        latencies = sorted(r["recall_seconds"] for r in group)
        query[arm] = {
            "questions": len(group),
            "retrieved_median": statistics.median(r["retrieved"] for r in group),
            "recall_p50_s": round(statistics.median(latencies), 3),
            "recall_p95_s": round(
                latencies[max(0, int(len(latencies) * 0.95) - 1)], 3),
            "prompt_tokens_median": statistics.median(
                r.get("prompt_tokens", 0) for r in group),
        }
    query_note = (
        "recall_seconds (and recall_p50_s/recall_p95_s here) wraps "
        "arm.recall() end to end, so for mem0 and MemPalace -- both point "
        "their embedder at the same shared endpoint this harness runs -- "
        "the query's own embedding HTTP round trip is already inside the "
        "figure, not a cost sitting outside it. Same rule on the write side: "
        "ingest.*_seconds_total/median below is wall clock around "
        "arm.ingest(), not isolated LLM-call time, because that embedding "
        "round trip happens there too, once per stored memory."
    )

    summary = {
        "table": "LOCOMO question answering, judged accuracy -- shim re-run",
        "published_in": "docs/benchmarks.md, 'Re-run through the local shim'",
        "run_conditions": {
            "harness": "benchmarks/run.py --dataset locomo --mode quality "
                       "--arms bm25,pamin,pamin-wide,mem0,mempalace "
                       "--units 10 --questions 20",
            "endpoint": "benchmarks/shim.py, an OpenAI-shaped HTTP endpoint "
                        "over `claude -p` subprocesses -- no Anthropic API "
                        "key is configured in this environment, so every "
                        "model call (reader, judge, mem0's extraction) runs "
                        "through this session's own Claude Code subscription "
                        "rather than a billed API call",
            "model": "sonnet for every arm's reader, judge, and (for mem0) "
                     "write-side extraction; one shared BGE-M3 ONNX embedder "
                     "for every arm that embeds",
            "pamin_rerank": "`accurate` -- `arms.Pamin.recall` and "
                            "`PaminWide.recall` call `pamin search` without "
                            "`--rerank`, so both took whatever the CLI's own "
                            "default is (`pamin search --help`: `[default: "
                            "accurate]`). That is a different setting from "
                            "the other two pamin measurements already on "
                            "this page: the original accuracy run predates "
                            "the full-head `accurate` rerank entirely (this "
                            "page's own opening note), and the dedicated "
                            "latency re-run explicitly set `--rerank fast` "
                            "as the shipped default at the time it was "
                            "written. `accurate` scores the whole fused "
                            "head with a cross-encoder forward pass per "
                            "candidate rather than reordering only the "
                            "non-lexical hits `fast` does, so it should cost "
                            "more per query and may read differently on "
                            "accuracy too -- this run does not isolate which.",
            "cost_usd_caveat": "ingest_cost_usd and its total here are "
                               "whatever `shim.py` assigns per simulated "
                               "call; they are notional accounting, not a "
                               "real bill, since no API key is set. Treat "
                               "them as a relative ingest-cost signal across "
                               "arms, not as dollars actually spent.",
            "container": "reclaimed repeatedly and unpredictably during this "
                         "run (see benchmarks/README.md); the run resumed "
                         "from its own checkpoint file each time, so no row "
                         "here reflects a clean single sitting",
        },
        "dataset": {"name": "locomo", "conversations": 10, "questions": 199},
        "scoring": "an LLM reader answers from the arm's passages and an LLM "
                   "judge grades the answer; both are the same model for "
                   "every arm",
        **machine_of(rows),
        "sources": [{"file": "quality.jsonl", "rows": len(rows)}],
        "arms": accuracy,
        "arms_note": "ingest_seconds_total, ingest_llm_calls_total and "
                     "ingest_cost_usd_total are summed over all 10 "
                     "conversations for that arm -- the one-time write-side "
                     "bill, not a per-question cost. bm25/pamin/pamin-wide "
                     "make no model calls on the write path, so their "
                     "totals are genuinely 0, not missing. mem0's and "
                     "mempalace's dollar totals are notional (see "
                     "run_conditions.cost_usd_caveat) but the call counts "
                     "and seconds are real measurements of this run. "
                     "reader_input_cost_usd is $3/million tokens (Anthropic's "
                     "published Sonnet list price) against the summed reader "
                     "prompt tokens over all 199 questions -- the one "
                     "per-question cost every arm actually measured. "
                     "partial_total_cost_usd is ingest_cost_usd_total plus "
                     "that, and the name says what it is not: it excludes "
                     "the judge call's prompt entirely and every "
                     "completion/output token for both calls, because the "
                     "shim never reports real completion-token counts "
                     "(run.py's chat() always gets completion_tokens: 0 "
                     "back). Treat it as a lower bound on the full "
                     "per-arm bill, not the bill.",
        "by_label": by_label,
        "mcnemar": [dict({"a": a, "b": b}, **paired(by_arm, a, b, correct))
                    for a, b in PAIRS],
        "ingest": ingest,
        "query": query,
        "query_note": query_note,
        "embedding_http_probe": EMBEDDING_HTTP_PROBE,
    }

    out = os.path.join(here, "results", "locomo",
                       "summary-accuracy-shim-2026-10.json")
    with open(out, "w") as sink:
        json.dump(summary, sink, indent=2, ensure_ascii=False)
        sink.write("\n")
    print(f"wrote {os.path.relpath(out, here)}")


if __name__ == "__main__":
    main()
