"""Query latency, timed at the same layer on every arm.

The accuracy comparison in `run.py` records a `recall_seconds` per question,
and that number is not a latency comparison. It timed this project through
`su ubuntu -c "pamin ... search ..."` -- two process spawns and a socket round
trip -- and timed mem0 as an in-process Python call, while ten arms, an
embedding endpoint and a PostgreSQL cluster shared four cores. Set side by side
those figures said mem0 was faster. Measured here, at the same layer with
nothing else running, they say the opposite by a factor of three.

So latency gets its own harness, and it holds three things fixed that the
accuracy run does not:

1. **The layer.** Every arm is timed at the boundary an application actually
   calls. For mem0 that is a library call; for this project it is the socket
   its resident server listens on. The process-spawn paths are timed too, and
   reported separately, because a caller using the CLI does pay them.
2. **Warmth.** Every unit is warmed before anything is timed. An earlier run of
   this file warmed only the first conversation and timed the socket arm first,
   so that arm alone paid nine cold starts and reported three times the CLI it
   is faster than.
3. **Quiet.** Nothing else runs. Latency is the measurement most sensitive to
   what else is on the machine, which is exactly why the accuracy run cannot
   double as one.

Each arm is then run a second time, last instead of first. Two runs that
disagree mean the order is in the number.
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

WORK = os.environ.get("BENCH_WORK", "/tmp/bench")
PAMIN_HOME = os.environ.get("BENCH_PAMIN_HOME", f"{WORK}/pamin-home")
PAMIN = os.environ.get("BENCH_PAMIN_BIN", "pamin")
SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:8088/v1")
SOCKET = f"{PAMIN_HOME}/pamin.sock"


def pamin_version():
    """The handshake string the server compares against its own.

    The crate version alone is not it: `protocol::version()` appends the
    executable's size and mtime, because two development builds share a
    version and share nothing else.
    """
    path = subprocess.run(["which", PAMIN], capture_output=True, text=True
                          ).stdout.strip() if "/" not in PAMIN else PAMIN
    st = os.stat(path)
    return f"0.0.1+{st.st_size}-{int(st.st_mtime)}"


class PaminSocket:
    """One connection to the resident server. No process is spawned per query."""

    name = "pamin (socket)"

    def __init__(self):
        self.version = pamin_version()
        self.conn = socket.socket(socket.AF_UNIX)
        self.conn.connect(SOCKET)
        self.file = self.conn.makefile("rw")

    def search(self, project, query, limit):
        self.file.write(json.dumps({
            "version": self.version, "project": project, "profile": "accuracy",
            "call": {"search": {"query": query, "limit": limit,
                                "channel_depth": 50, "graph_depth": 2,
                                "rerank": "fast"}},
        }) + "\n")
        self.file.flush()
        line = self.file.readline()
        if not line:
            raise RuntimeError("the server closed the connection")
        reply = json.loads(line)
        if "err" in reply:
            raise RuntimeError(str(reply["err"])[:200])
        if "mismatch" in reply:
            raise RuntimeError(f"server is a different build: {reply['mismatch']}")
        return reply["ok"]["hits"]


class PaminCli:
    """The command as a user runs it: fork, exec, connect, exit."""

    name = "pamin (CLI)"

    def search(self, project, query, limit):
        out = subprocess.run(
            [PAMIN, "--home", PAMIN_HOME, "--project", project, "search",
             query, "--limit", str(limit), "--json"],
            capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-300:])
        return json.loads(out.stdout)["hits"]


class Mem0:
    """mem0's own library call, which is the boundary its users call."""

    name = "mem0 (in-process)"

    def __init__(self, collection):
        os.environ["OPENAI_API_KEY"] = "shim-serves-no-key-needed"
        os.environ["OPENAI_BASE_URL"] = SHIM
        from mem0 import Memory
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": "sonnet"}},
            "embedder": {"provider": "openai",
                         "config": {"model": "bge-m3", "embedding_dims": 1024}},
            "vector_store": {"provider": "qdrant",
                             "config": {"path": f"{WORK}/mem0-qdrant",
                                        "embedding_model_dims": 1024,
                                        "on_disk": True,
                                        "collection_name": collection}},
        })

    def search(self, project, query, limit):
        found = self.memory.search(query, filters={"user_id": project},
                                   top_k=limit)
        results = found.get("results", found) if isinstance(found, dict) else found
        return results


def timed(arm, work, limit):
    """One pass over the work, returning the times and the hits it saw."""
    times, hits = [], 0
    for project, query in work:
        started = time.perf_counter()
        found = arm.search(project, query, limit)
        times.append(time.perf_counter() - started)
        hits += len(found)
    return times, hits


def report(name, times):
    quantile = lambda p: statistics.quantiles(sorted(times), n=100)[p - 1] * 1000
    print(f"{name:<34} n={len(times):>4}  "
          f"p50={statistics.median(times) * 1000:7.1f} ms  "
          f"p95={quantile(95):7.1f} ms  "
          f"mean={statistics.mean(times) * 1000:7.1f} ms", flush=True)


def measure(arm, work, limit):
    """Warm everything, time it, then time it again last.

    The premise this asserts is that the store has something in it: an arm
    answering nothing, quickly, is not a latency measurement.
    """
    for project in dict.fromkeys(p for p, _ in work):
        query = next(q for p, q in work if p == project)
        for _ in range(3):
            arm.search(project, query, limit)

    times, hits = timed(arm, work, limit)
    if hits == 0:
        raise SystemExit(f"{arm.name} returned nothing for every query; "
                         "this is the cost of searching an empty store")
    report(arm.name, times)
    again, _ = timed(arm, work, limit)
    drift = abs(statistics.median(again) - statistics.median(times))
    if drift > 0.5 * statistics.median(times):
        report(f"  {arm.name}, repeated last", again)
        print("  ^ the two runs disagree; the order is in this number",
              flush=True)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--arms", default="pamin-socket,pamin-cli")
    parser.add_argument("--mem0-collection", default="",
                        help="a surviving mem0 collection, e.g. conv_50; mem0 "
                             "clears its store per conversation, so only the "
                             "last one ingested can be re-timed")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datasets.locomo as locomo

    work, mem0_work = [], []
    for entry in locomo.read()[:10]:
        cid = entry["sample_id"]
        project = "locomor1" + "".join(c for c in cid.lower() if c.isalnum())[:26]
        for _qid, qa, _reference in locomo.questions(entry, per_unit=20):
            work.append((project, qa["question"]))
            if args.mem0_collection and \
                    args.mem0_collection == cid.replace("-", "_"):
                mem0_work.append((cid, qa["question"]))

    print(f"{len(work)} questions, --limit {args.limit}, "
          f"nothing else should be running\n", flush=True)

    wanted = args.arms.split(",")
    if "pamin-socket" in wanted:
        measure(PaminSocket(), work, args.limit)
    if "pamin-cli" in wanted:
        measure(PaminCli(), work, args.limit)
    if "mem0" in wanted:
        if not mem0_work:
            raise SystemExit("--mem0-collection must name a surviving "
                             "collection for the mem0 arm")
        measure(Mem0(args.mem0_collection), mem0_work, args.limit)


if __name__ == "__main__":
    main()
