#!/usr/bin/env python3
"""Run one benchmark across a set of arms.

Two modes, because two different questions are being asked and they cost
wildly different amounts:

  quality -- what each arm gets right. Needs a reader and a judge on LOCOMO;
             needs neither on LongMemEval, where retrieval is scored directly.
  cost    -- what each arm spends. No reader, no judge: the write side is
             measured through the shared endpoint, and the query side is the
             prompt this harness builds, which can simply be counted.

Rows are appended as they finish and every unit is checkpointed, because the
machine this was written on is reclaimed without warning and a run that cannot
resume is a run nobody repeats.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import importlib
import json
import os
import platform
import socket
import statistics
import threading
import time
import urllib.request

import arms as arms_module
import resources

# mem0's own ingest empties its shared qdrant directory on every unit unless
# BENCH_MEM0_KEEP=1 (see arms.py) -- two units ingesting at once would delete
# each other's in-flight store. Forced to one worker regardless of --workers,
# rather than trusted to the caller, because a run that corrupts silently is
# worse than one that stays slow.
SERIAL_ARMS = {"mem0", "mem0-wide"}

SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:8088/v1")


def shard_of(unit, count):
    """Which shard a unit belongs to, stable across processes and runs.

    A property of the unit id alone, not of iteration order or checkpoint
    state, so two processes started at slightly different moments -- with
    slightly different `done` sets already loaded -- still partition the
    same units the same way instead of racing to ingest the same one twice.
    """
    return int(hashlib.md5(unit.encode()).hexdigest(), 16) % count

# Set when more than one run shares this machine and this endpoint.
#
# The write-side cost columns are differences of a counter the endpoint keeps
# for everybody, and the timings are of a machine somebody else is also using.
# Under a second run they measure the pair, not the arm: an ingest that called
# no model at all came back reporting fourteen calls, which is the one number
# on this page that has to be zero.
#
# Stamped on every row rather than remembered in a comment, because the file
# outlives the shell that produced it.
SHARED = os.environ.get("BENCH_SHARED_ENDPOINT") == "1"

# What a shared endpoint and a shared machine make unreadable. Everything
# else -- the verdict, the prompt size, the bytes on disk -- is the arm's own.
CONTENDED = ("ingest_seconds", "ingest_llm_calls", "ingest_llm_seconds",
             "ingest_embedded", "ingest_cost_usd", "peak_pss_mb",
             "recall_seconds")


def machine():
    """Stamped on every result, because half of them do not travel.

    A latency taken on four shared cores is not the same measurement as one
    taken on a laptop, and a file of rows without this cannot be merged with
    another honestly.
    """
    cores = os.cpu_count()
    total = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                total = round(int(line.split()[1]) / 1024 / 1024)
                break
    except OSError:
        pass
    return {"host": socket.gethostname(), "platform": platform.platform(),
            "cores": cores, "memory_gb": total}


def chat(prompt, retries=3):
    body = json.dumps({"model": "sonnet",
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                f"{SHIM}/chat/completions", data=body,
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)["choices"][0]["message"]["content"].strip()
        except Exception as error:
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"the shared endpoint failed {retries} times: {last}")


def tokens_in(text):
    """A named tokenizer, used the same way for every arm.

    Not the model's own: this environment's CLI reports a prompt of any size
    as a constant plus `cache_creation_input_tokens`, and its delta for a
    3,623-character string was 1.9x what any tokenizer gives. What the cost
    table needs is a ratio between arms, and one tokenizer applied to both
    sides gives that even when the absolute figure is approximate.
    """
    import tiktoken
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def load(rows_path):
    """Rows already written, and the questions they cover.

    Keyed per question, not per unit. Keying per unit is what this did first,
    and it loses work silently: a conversation interrupted after its tenth
    answer had written rows, so it counted as done, and the other ten were
    never retried. The run then reported success with a hole in it -- one arm
    came back with 182 of 199 answers and nothing said so.

    The resume key has to be as fine as the thing being written.
    """
    if not os.path.exists(rows_path):
        return [], set()
    rows = [json.loads(line) for line in open(rows_path) if line.strip()]
    return rows, {(r["arm"], r["unit"], r.get("question_id")) for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="locomo",
                        choices=["locomo", "longmemeval", "longmemeval_qa",
                                 "supersession", "miracl"])
    parser.add_argument("--arms", default="bm25,pamin")
    parser.add_argument("--mode", default="quality", choices=["quality", "cost"])
    parser.add_argument("--units", type=int, default=10,
                        help="conversations for LOCOMO, questions for LongMemEval")
    parser.add_argument("--questions", type=int, default=20,
                        help="questions per unit; LongMemEval has one")
    parser.add_argument("--path", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="units ingested at once per arm; 1 keeps the "
                             "original sequential behaviour. mem0's own "
                             "shared-directory ingest forces it to 1 "
                             "regardless of this flag -- see SERIAL_ARMS.")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="run only this process's slice of the units, "
                             "so several independent processes -- pointed "
                             "at their own stores via BENCH_PAMIN_HOME -- "
                             "can split the work without touching the "
                             "same one. 1 keeps every unit in this process.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="which slice this process takes, 0-based; "
                             "ignored when --shard-count is 1.")
    args = parser.parse_args()

    dataset = importlib.import_module(f"datasets.{args.dataset}")
    entries = dataset.read(args.path or dataset.DEFAULT_PATH, args.units)
    out = args.out or os.path.join(
        os.path.dirname(__file__), "results", args.dataset,
        f"{args.mode}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows, done = load(out)
    if done:
        print(f"resuming: {len(rows)} rows already in {out}", flush=True)
    sink = open(out, "a")
    where = machine()

    sink_lock = threading.Lock()

    for name in args.arms.split(","):
        todo = []
        for entry in entries:
            unit = dataset.unit_id(entry)
            if args.shard_count > 1 and shard_of(unit, args.shard_count) != args.shard_index:
                continue
            wanted = {qid for qid, _qa, _ref
                      in dataset.questions(entry, args.questions)} or {None}
            if not all((name, unit, qid) in done for qid in wanted):
                todo.append(entry)

        workers = 1 if name in SERIAL_ARMS else args.workers

        # Five in a row is not bad luck, it is a broken setup, and the run
        # that taught this lesson is the one where a concurrent `cargo build`
        # replaced the binary under an arm: every ingest failed, each was
        # caught and skipped, and the run reported success with no rows in it.
        # A harness that cannot fail cannot be trusted when it passes.
        # Under concurrency "in a row" is approximate rather than strict
        # ordering, but the intent -- stop on a broken setup rather than
        # grinding through every unit the same way -- still holds.
        consecutive = 0
        consecutive_lock = threading.Lock()

        def record(entry, unit_rows, error):
            nonlocal consecutive
            unit = dataset.unit_id(entry)
            if error is not None:
                print(f"  {name} {unit}: INGEST FAILED: {error}", flush=True)
                with consecutive_lock:
                    consecutive += 1
                    hit = consecutive >= 5
                if hit:
                    raise SystemExit(
                        f"{name} failed to ingest {consecutive} units in a "
                        f"row; the last said: {error}")
                return
            with consecutive_lock:
                consecutive = 0
            with sink_lock:
                for row in unit_rows:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    rows.append(row)
                    done.add((name, unit, row.get("question_id")))

        if workers <= 1:
            for entry in todo:
                unit_rows, error = process_unit(dataset, args, name, entry,
                                                where, done)
                record(entry, unit_rows, error)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(process_unit, dataset, args, name, entry,
                               where, done): entry
                    for entry in todo
                }
                for future in concurrent.futures.as_completed(futures):
                    entry = futures[future]
                    unit_rows, error = future.result()
                    record(entry, unit_rows, error)
    sink.close()
    summarise(rows, args.mode)


def process_unit(dataset, args, name, entry, where, done):
    """Ingest and measure one unit. Returns (rows, error); writes nothing.

    Safe to call concurrently for different units of the same arm: each unit
    gets its own arm instance and (per every arm but mem0, which is kept off
    this path -- see SERIAL_ARMS) its own project/collection/palace, so no
    two calls touch the same store. `done` is only read here, never written;
    the caller is the only writer and it only ever writes this unit's own
    keys, so a concurrent reader never observes another unit's result.
    """
    unit = dataset.unit_id(entry)
    arm = arms_module.ARMS[name]()
    turns = dataset.turns_of(entry)

    before = arms_module.shim_stats()
    disk_before = resources.dir_mb(getattr(arm, "store_path", None))
    sampler = resources.Sampler(lambda: resources.tree_for(arm, arms_module.PAMIN_HOME))
    sampler.start()
    try:
        seconds, count = arm.ingest(unit, turns)
    except Exception as error:
        sampler.stop.set()
        return [], error
    sampler.stop.set()
    sampler.join(timeout=5)
    after = arms_module.shim_stats()

    cost = {
        "ingest_seconds": round(seconds, 1),
        "ingest_llm_calls": after.get("calls", 0) - before.get("calls", 0),
        "ingest_llm_seconds": round(
            after.get("seconds", 0.0) - before.get("seconds", 0.0), 1),
        "ingest_embedded": after.get("embed_texts", 0) - before.get("embed_texts", 0),
        "ingest_cost_usd": round(
            after.get("cost_usd", 0.0) - before.get("cost_usd", 0.0), 4),
        "store_mb": resources.dir_mb(getattr(arm, "store_path", None)) - disk_before,
        "peak_pss_mb": max(sampler.samples) if sampler.samples else None,
        "turns": count,
        "cost_measurable": not SHARED,
    }
    print(f"  {name} {unit}: {count} turns in {seconds:.0f}s | "
          f"{cost['ingest_llm_calls']} LLM calls, "
          f"{cost['ingest_embedded']} embedded", flush=True)

    return list(measure(dataset, arm, entry, args, cost, where, name,
                        unit, done)), None


def measure(dataset, arm, entry, args, cost, where, name, unit, done):
    """One row per question, whatever the dataset and the mode.

    Skips the questions already in `done`, so a unit interrupted partway
    through is finished rather than abandoned.
    """
    retrieval_only = getattr(dataset, "RETRIEVAL_ONLY", False)
    if retrieval_only and not hasattr(arm, "recall_ids"):
        # Retrieval needs the identity of what came back, not its text, and
        # only an arm that stores the original turns can give that. An arm
        # that stores rewritten facts -- mem0 extracts them, so its memories
        # belong to no turn -- cannot be scored this way at all, and says so
        # rather than being scored against a mapping invented here.
        raise SystemExit(
            f"{name} stores rewritten memories rather than the turns it was "
            f"given, so it cannot be scored on {dataset.NAME}'s retrieval "
            f"metric; run it on a dataset whose metric reads answers instead")

    for qid, qa, reference in dataset.questions(entry, args.questions):
        if (name, unit, qid) in done:
            continue
        started = time.time()
        if retrieval_only:
            ids = arm.recall_ids(qa["question"])
            elapsed = time.time() - started
            ranked = dataset.sessions_of(ids)
            yield {"arm": name, "unit": unit, "question_id": qid, "machine": where,
                   "mode": args.mode, "label": dataset.label(qa),
                   "recall_seconds": round(elapsed, 3), "retrieved": len(ids),
                   "metrics": dataset.score(qa, ranked), **cost}
            continue
        passages = arm.recall(qa["question"])
        elapsed = time.time() - started
        prompt = dataset.READER.format(
            context="\n".join(f"- {p}" for p in passages), question=qa["question"])
        row = {"arm": name, "unit": unit, "question_id": qid, "machine": where,
               "mode": args.mode, "label": dataset.label(qa),
               "retrieved": len(passages), "recall_seconds": round(elapsed, 3),
               "prompt_tokens": tokens_in(prompt), **cost}
        if args.mode == "quality":
            predicted = chat(prompt) if passages else ""
            row["predicted"] = predicted
            verdict = (dataset.judged(chat, qa["question"], reference, predicted)
                       if passages else 0.0)
            # A judge may return more than a score. Supersession needs three
            # outcomes rather than two, because answering with the fact that
            # was replaced and answering with nothing are different failures
            # and a single `correct` column cannot tell them apart.
            row.update(verdict if isinstance(verdict, dict) else {"correct": verdict})
        yield row


def summarise(rows, mode):
    if not rows:
        return print("no rows")
    by_arm = collections.defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    def per_unit(arm, field):
        seen = {}
        for row in by_arm[arm]:
            seen[row["unit"]] = row.get(field, 0)
        return list(seen.values())

    if mode == "quality" and "correct" in rows[0]:
        print(f"\n=== accuracy, {len(rows)} answers ===")
        for arm in sorted(by_arm):
            group = by_arm[arm]
            print(f"  {arm:<16}{len(group):>5}"
                  f"{statistics.mean(r['correct'] for r in group):>10.3f}")

        if any("verdict" in r for r in rows):
            print("\n=== which answer came back ===")
            kinds = ["current", "stale", "neither", "unparsed"]
            print(f"  {'arm':<20}" + "".join(f"{k:>10}" for k in kinds))
            for arm in sorted(by_arm):
                group = [r for r in by_arm[arm] if "verdict" in r]
                if not group:
                    continue
                line = f"  {arm:<20}"
                for kind in kinds:
                    share = sum(r["verdict"] == kind for r in group) / len(group)
                    line += f"{share:>10.3f}"
                print(line + f"   n={len(group)}")

        print("\n=== by label ===")
        arms = sorted(by_arm)
        print(f"  {'label':<26}" + "".join(f"{a:>16}" for a in arms))
        for label in sorted({r["label"] for r in rows}):
            line = f"  {label:<26}"
            for arm in arms:
                group = [r for r in by_arm[arm] if r["label"] == label]
                line += (f"{statistics.mean(r['correct'] for r in group):>16.3f}"
                         if group else f"{'-':>16}")
            print(line)

    if any(r.get("cost_measurable") is False for r in rows):
        print("\n=== what one unit costs ===")
        print("  Not from this run. It shared the machine and the model "
              "endpoint with another,")
        print(f"  so {', '.join(CONTENDED)} measure the pair rather "
              f"than the arm.")
        print(f"  {'arm':<20}{'store MB':>10}{'prompt tokens':>15}{'retrieved':>11}")
        for arm in sorted(by_arm):
            group = by_arm[arm]
            print(f"  {arm:<20}"
                  f"{statistics.median(per_unit(arm, 'store_mb')):>10.0f}"
                  f"{statistics.median(r.get('prompt_tokens', 0) for r in group):>15.0f}"
                  f"{statistics.median(r.get('retrieved', 0) for r in group):>11.0f}")
        return

    print("\n=== what one unit costs to ingest ===")
    print(f"  {'arm':<16}{'seconds':>9}{'LLM':>6}{'LLM s':>8}{'embedded':>10}"
          f"{'store MB':>10}{'peak PSS':>10}")
    for arm in sorted(by_arm):
        peaks = [v for v in per_unit(arm, "peak_pss_mb") if v]
        print(f"  {arm:<16}"
              f"{statistics.median(per_unit(arm, 'ingest_seconds')):>9.0f}"
              f"{statistics.median(per_unit(arm, 'ingest_llm_calls')):>6.0f}"
              f"{statistics.median(per_unit(arm, 'ingest_llm_seconds')):>8.0f}"
              f"{statistics.median(per_unit(arm, 'ingest_embedded')):>10.0f}"
              f"{statistics.median(per_unit(arm, 'store_mb')):>10.0f}"
              f"{statistics.median(peaks) if peaks else 0:>10.0f}")

    print("\n=== what one question costs to answer ===")
    print(f"  {'arm':<16}{'recall p50':>12}{'recall p95':>12}{'prompt tokens':>15}"
          f"{'retrieved':>11}")
    for arm in sorted(by_arm):
        group = by_arm[arm]
        latencies = sorted(r["recall_seconds"] for r in group)
        print(f"  {arm:<16}{statistics.median(latencies):>12.2f}"
              f"{latencies[max(0, int(len(latencies) * 0.95) - 1)]:>12.2f}"
              f"{statistics.median(r.get('prompt_tokens', 0) for r in group):>15.0f}"
              f"{statistics.median(r.get('retrieved', 0) for r in group):>11.0f}")


if __name__ == "__main__":
    main()
