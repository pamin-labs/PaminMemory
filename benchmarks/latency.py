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
import shlex
import socket
import statistics
import subprocess
import sys
import time

WORK = os.environ.get("BENCH_WORK", "/tmp/bench")
PAMIN_HOME = os.environ.get("BENCH_PAMIN_HOME", f"{WORK}/pamin-home")
PAMIN = os.environ.get("BENCH_PAMIN_BIN", "pamin")
# The workspace keeps PostgreSQL at mode 700, so the `su` arm has to
# become the user that owns it. Same default and same variable as
# `arms.py`, because the two have to agree about who runs the command.
RUNAS = os.environ.get("BENCH_USER", "ubuntu")
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


class PaminSu:
    """The CLI behind `su`, which is the layer the accuracy run timed.

    Not a layer anyone would choose: the harness runs as root and the
    workspace's PostgreSQL data directory is mode 700, so every `pamin` call in
    the accuracy run went through a `su`. That run reported 170 ms and the
    obvious reading of it was that this project was slower than mem0. This arm
    exists to price the difference rather than assert it, and it is the row
    that explains the other one.
    """

    name = "pamin (CLI behind su)"

    def search(self, project, query, limit):
        out = subprocess.run(
            ["su", RUNAS, "-c",
             f"PAMIN_HOME={shlex.quote(PAMIN_HOME)} {shlex.quote(PAMIN)} "
             f"--project {shlex.quote(project)} search {shlex.quote(query)} "
             f"--limit {limit} --json"],
            capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-300:])
        return json.loads(out.stdout)["hits"]


class Embedding:
    """One embedding call to the shared endpoint, which every arm but this
    project's pays inside its own search.

    mem0 and MemPalace embed the query over HTTP; this project embeds it in the
    process that holds the index. So this is not a memory system's latency at
    all -- it is the floor under the two that call out, reported separately so
    the difference between them and this project can be read as the part that
    is architecture rather than the part that is code.
    """

    name = "of which, one embedding HTTP call"

    def search(self, project, query, limit):
        import urllib.request
        body = json.dumps({"model": "bge-m3", "input": query}).encode()
        request = urllib.request.Request(
            SHIM.rstrip("/") + "/embeddings", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer shim-serves-no-key-needed"})
        with urllib.request.urlopen(request, timeout=600) as r:
            got = json.load(r)
        # The premise, per call rather than once: an endpoint that answered
        # with an error body would be timed as a very fast embedder.
        vector = got["data"][0]["embedding"]
        if len(vector) != 1024:
            raise RuntimeError(f"the endpoint returned {len(vector)} "
                               "dimensions, so this is not the shared embedder")
        return vector


class Mem0:
    """mem0's own library call, which is the boundary its users call.

    One client, switching collection per query, because a local qdrant allows
    exactly one client per storage folder -- ten clients over one directory is
    a `RuntimeError`, not ten clients. `collection_name` is a plain attribute
    its search reads, so this is the same call mem0 would make itself.

    Until `BENCH_MEM0_KEEP` existed the accuracy harness emptied that folder
    before each ingest, so only the last conversation survived a run and this
    arm could be timed over one conversation's twenty questions where every
    other arm had 199.
    """

    name = "mem0 (in-process)"

    def __init__(self, collections):
        os.environ["OPENAI_API_KEY"] = "shim-serves-no-key-needed"
        os.environ["OPENAI_BASE_URL"] = SHIM
        from mem0 import Memory
        self.collections = sorted(collections)
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": "sonnet"}},
            "embedder": {"provider": "openai",
                         "config": {"model": "bge-m3", "embedding_dims": 1024}},
            "vector_store": {"provider": "qdrant",
                             "config": {"path": f"{WORK}/mem0-qdrant",
                                        "embedding_model_dims": 1024,
                                        "on_disk": True,
                                        "collection_name": self.collections[0]}},
        })
        present = {c.name for c in
                   self.memory.vector_store.client.get_collections().collections}
        missing = [c for c in self.collections if c not in present]
        if missing:
            raise SystemExit(
                f"these mem0 collections are not in the store: {missing}. "
                "Ingest with BENCH_MEM0_KEEP=1 to keep all of them.")

    def search(self, project, query, limit):
        self.memory.vector_store.collection_name = project.replace("-", "_")
        found = self.memory.search(query, filters={"user_id": project},
                                   top_k=limit)
        results = found.get("results", found) if isinstance(found, dict) else found
        # The premise, per query: a collection that was never ingested answers
        # with an empty list very quickly, which is the cheapest possible
        # latency and the least honest one.
        if not results:
            raise RuntimeError(
                f"mem0 returned nothing for {project}; that collection is "
                "empty, so this would be timing the cost of searching nothing")
        return results


class MemPalace:
    """MemPalace's own search call, in its own virtualenv.

    Run this arm with that virtualenv's interpreter -- `mp-venv/bin/python
    benchmarks/latency.py --arms mempalace` -- because the reason MemPalace
    has one is that installing it beside mem0 moves `protobuf` past the
    ceiling mem0 declares.

    The accuracy run reaches MemPalace by spawning that interpreter per query,
    which is a Python start and a package import on top of the search. Timed
    here at the call itself, the way an application embedding it would.
    """

    name = "mempalace (in-process)"

    def __init__(self):
        os.environ.setdefault("MEMPALACE_EMBEDDING_MODEL", "openai-compat")
        os.environ.setdefault("MEMPALACE_EMBEDDING_API_URL",
                              SHIM.rsplit("/v1", 1)[0])
        os.environ.setdefault("MEMPALACE_EMBEDDING_API_MODEL", "bge-m3")
        from mempalace import searcher
        self.searcher = searcher
        self.before = shim_embed_count()

    def search(self, project, query, limit):
        # `searcher.search` is the renderer the CLI calls: it prints and
        # returns nothing. `search_memories` is the retrieval, and it is the
        # layer the other arms are timed at.
        found = self.searcher.search_memories(
            query, f"{WORK}/mp-palace/{project}", n_results=limit)
        return found.get("results", [])

    def check_premise(self):
        """The same premise its ingest arm asserts: one shared embedder.

        The provider switch fails silently, and an arm that quietly used its
        own 384-dimension model would be timing a different system.
        """
        if shim_embed_count() - self.before <= 0:
            raise SystemExit(
                "mempalace embedded nothing through the shared endpoint, so it "
                "used its own model; this would be timing a different embedder")


def shim_embed_count():
    import urllib.request
    try:
        with urllib.request.urlopen(SHIM.rsplit("/v1", 1)[0] + "/stats",
                                    timeout=5) as r:
            return json.load(r).get("embed_texts", 0)
    except Exception:
        return 0


def reader_curve(repeats=12):
    """What the reader costs, against how much context it is handed.

    Retrieval latency is one term of what a caller waits for, and on its own
    it answers the wrong question. The other term is the model call that reads
    the passages, and the arms differ most in how many tokens they hand it --
    557 for this project at ten passages, ~1,017 for mem0 at thirty, 1,511 for
    this project at thirty, 5,133 for MemPalace at thirty. If that term
    dominates, a sixty-millisecond difference in retrieval is not a difference
    anyone experiences.

    The filler is real retrieved text rather than a repeated token, so the
    prefill is not unrepresentatively compressible.
    """
    import tiktoken
    import urllib.request

    enc = tiktoken.get_encoding("cl100k_base")
    rows = [json.loads(line) for line in open(f"{WORK}/compare5.jsonl")]
    corpus = " ".join(r["question"] + " " + r["reference"] for r in rows)
    tokens = enc.encode(corpus)

    sizes = {"557  (pamin @10)": 557, "1017 (mem0 @30)": 1017,
             "1511 (pamin @30)": 1511, "5133 (MemPalace @30)": 5133}
    print(f"reader calls, {repeats} at each size\n", flush=True)
    for label, size in sizes.items():
        prompt = (f"Retrieved records:\n{enc.decode(tokens[:max(1, size - 40)])}"
                  "\n\nQuestion: what is the first word of the records?\n"
                  "Answer in as few words as possible.")
        times = []
        for _ in range(repeats):
            body = json.dumps({"model": "sonnet",
                               "messages": [{"role": "user",
                                             "content": prompt}]}).encode()
            request = urllib.request.Request(
                f"{SHIM}/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            started = time.perf_counter()
            urllib.request.urlopen(request, timeout=600).read()
            times.append(time.perf_counter() - started)
        actual = len(enc.encode(prompt))
        # Seconds rather than milliseconds, and no p95: twelve calls to a
        # hosted model give a median worth reading and a tail that is the
        # provider's queue rather than anything about context length.
        ROWS.append({"arm": f"reader, {label.split()[0]} tokens of context",
                     "limit": None, "n": len(times),
                     "prompt_tokens": actual,
                     "p50_ms": round(statistics.median(times) * 1000, 1),
                     "min_ms": round(min(times) * 1000, 1),
                     "max_ms": round(max(times) * 1000, 1)})
        print(f"{label:<24} actual={actual:>5} tok  "
              f"median={statistics.median(times):6.2f} s  "
              f"min={min(times):6.2f} s  max={max(times):6.2f} s", flush=True)


def timed(arm, work, limit):
    """One pass over the work, returning the times and the hits it saw."""
    times, hits = [], 0
    for project, query in work:
        started = time.perf_counter()
        found = arm.search(project, query, limit)
        times.append(time.perf_counter() - started)
        hits += len(found)
    return times, hits


# Every row this run produces, kept so `--out` can write them.
#
# It printed and nothing else for the whole run that produced the published
# latency table, so that table has no artifact behind it: the numbers went from
# a terminal into a document and cannot be rechecked. Every other harness here
# writes rows. This one now does too.
ROWS = []


def report(name, times, limit=None):
    quantile = lambda p: statistics.quantiles(sorted(times), n=100)[p - 1] * 1000
    ROWS.append({"arm": name, "limit": limit, "n": len(times),
                 "p50_ms": round(statistics.median(times) * 1000, 1),
                 "p95_ms": round(quantile(95), 1),
                 "mean_ms": round(statistics.mean(times) * 1000, 1)})
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

    if hasattr(arm, "check_premise"):
        arm.check_premise()

    times, hits = timed(arm, work, limit)
    if hits == 0:
        raise SystemExit(f"{arm.name} returned nothing for every query; "
                         "this is the cost of searching an empty store")
    report(arm.name, times, limit)
    again, _ = timed(arm, work, limit)
    drift = abs(statistics.median(again) - statistics.median(times))
    if drift > 0.5 * statistics.median(times):
        report(f"  {arm.name}, repeated last", again, limit)
        print("  ^ the two runs disagree; the order is in this number",
              flush=True)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default=None,
                        help="append one JSONL row per arm; without it the "
                             "run leaves no evidence behind")
    parser.add_argument("--arms", default="pamin-socket,pamin-cli")
    parser.add_argument("--mem0-collection", default="",
                        help="time mem0 over this one collection, e.g. "
                             "conv_50. Without it the arm takes every "
                             "conversation, which needs a store ingested "
                             "under BENCH_MEM0_KEEP=1")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import datasets.locomo as locomo

    work, mem0_work = [], []
    for entry in locomo.read()[:10]:
        cid = entry["sample_id"]
        project = "locomor1" + "".join(c for c in cid.lower() if c.isalnum())[:26]
        for _qid, qa, _reference in locomo.questions(entry, per_unit=20):
            work.append((project, qa["question"]))
            if not args.mem0_collection or \
                    args.mem0_collection == cid.replace("-", "_"):
                mem0_work.append((cid, qa["question"]))

    print(f"{len(work)} questions, --limit {args.limit}, "
          f"nothing else should be running\n", flush=True)

    wanted = args.arms.split(",")
    if "pamin-socket" in wanted:
        measure(PaminSocket(), work, args.limit)
    if "pamin-cli" in wanted:
        measure(PaminCli(), work, args.limit)
    if "pamin-su" in wanted:
        measure(PaminSu(), work, args.limit)
    if "embedding" in wanted:
        measure(Embedding(), work, args.limit)
    if "reader" in wanted:
        reader_curve()
    if "mempalace" in wanted:
        measure(MemPalace(), [(p.replace("locomor1conv", "conv-"), q)
                              for p, q in work], args.limit)
    if "mem0" in wanted:
        if not mem0_work:
            raise SystemExit("--mem0-collection names no conversation in this "
                             "dataset, so the mem0 arm has nothing to time")
        measure(Mem0({c.replace("-", "_") for c, _ in mem0_work}),
                mem0_work, args.limit)

    if args.out:
        import run as run_module
        where = run_module.machine()
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as sink:
            for row in ROWS:
                sink.write(json.dumps({**row, "machine": where,
                                       "mode": "latency"}) + "\n")
        print(f"\n{len(ROWS)} rows appended to {args.out}", flush=True)


if __name__ == "__main__":
    main()
