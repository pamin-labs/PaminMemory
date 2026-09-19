"""Every memory system this compares, one class each.

An arm is three methods and two declarations: `ingest` puts a conversation in
and returns what it cost, `recall` answers a query with passages, and the
class says where its store lives and which process holds it so the resource
accounting can find them.

Arms never build their own retrieval. Each one drives the interface its
project actually ships -- `pamin search`, `Memory.search`, `mempalace search`
-- because a rewritten pipeline measures the rewrite.
"""
import collections
import json
import math
import os
import re
import shlex
import subprocess
import time

SHIM = os.environ.get("SHIM_URL", "http://127.0.0.1:8088/v1")
PAMIN = os.environ.get("PAMIN_BIN", "target/release/pamin")
PAMIN_HOME = os.environ.get("BENCH_PAMIN_HOME", "/tmp/bench/pamin-home")
RUNAS = os.environ.get("BENCH_USER", "ubuntu")
TOP_K = int(os.environ.get("TOP_K", "10"))
WORK = os.environ.get("BENCH_WORK", "/tmp/bench")


def shim_stats():
    """What the shared endpoint has been asked for so far.

    Differenced across an arm's ingest, this is that arm's inference bill --
    the number that separates a system calling a model on its write path from
    one that does not. It is a result, not an implementation detail.
    """
    import urllib.request
    try:
        base = SHIM.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(f"{base}/stats", timeout=30) as r:
            return json.load(r)
    except Exception:
        return {}


class Bm25:
    """No memory system: Okapi BM25 over the same turns."""

    name = "bm25"
    store_path = None      # in memory only
    holder = None          # held by this process
    WORD = re.compile(r"[a-z0-9]+")

    def ingest(self, conversation_id, turns):
        started = time.time()
        self.turns = turns
        self.docs = [self.WORD.findall(t["record"].lower()) for t in turns]
        self.freqs = [collections.Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.docs)
        df = collections.Counter()
        for f in self.freqs:
            df.update(f.keys())
        n = len(self.docs)
        self.idf = {t: max(0.0, math.log((n - c + 0.5) / (c + 0.5) + 1.0))
                    for t, c in df.items()}
        return time.time() - started, len(turns)

    def recall(self, question):
        scores = [0.0] * len(self.docs)
        for term in self.WORD.findall(question.lower()):
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, f in enumerate(self.freqs):
                tf = f.get(term)
                if not tf:
                    continue
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * self.lengths[i] / self.avg)
                scores[i] += idf * tf * 2.5 / denom
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:TOP_K]
        self._last = [self.turns[i]["id"] for i in ranked]
        return [self.turns[i]["record"] for i in ranked]

    def recall_ids(self, question):
        """What came back, by turn id, for a metric that scores identity."""
        self.recall(question)
        return self._last


class Pamin:
    """The shipped CLI: `pamin import`, then `pamin search`."""

    name = "pamin"
    store_path = PAMIN_HOME
    holder = "pamin"       # the resident server holds the index and the model

    def _run(self, project, args, timeout=3600):
        quoted = " ".join(shlex.quote(a) for a in args)
        out = subprocess.run(
            ["su", RUNAS, "-c",
             f"PAMIN_HOME={PAMIN_HOME} {PAMIN} --project {project} {quoted}"],
            capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            raise RuntimeError(f"pamin {quoted[:80]}: {out.stderr[-400:]}")
        return out.stdout

    def ingest(self, conversation_id, turns):
        # A run tag in the name, because `import` deduplicates: a project
        # left over from an earlier run returns in a second and reports
        # that second as the ingest cost.
        tag = os.environ.get("RUN_TAG", "r1")
        self.project = ("locomo" + tag
                        + re.sub(r"[^a-z0-9]", "", conversation_id.lower())[:26])
        path = f"{WORK}/ndjson/{self.project}.ndjson"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for t in turns:
                f.write(json.dumps({
                    "topic": re.sub(r"[^A-Za-z0-9_]", "_", t["id"]),
                    "content": t["record"],
                }, ensure_ascii=False) + "\n")
        os.chmod(path, 0o644)

        started = time.time()
        self._run(self.project, ["import", "--from", path])
        self._run(self.project, ["cascade", "drain"])
        return time.time() - started, len(turns)

    def recall(self, question):
        out = self._run(self.project,
                        ["search", question, "--limit", str(TOP_K), "--json"])
        hits = json.loads(out)["hits"]
        self._last = [h["topic"] for h in hits]
        return [h["content"] for h in hits]

    def recall_ids(self, question):
        self.recall(question)
        return self._last


class PaminWide:
    """The flat arm with three times the context, to price context by itself.

    If a ledger arm reads better, the first question is whether it read better
    or merely read more. This arm changes nothing but the cut, over the same
    projects the flat arm built, so the difference between it and `pamin` is
    what a bigger shortlist is worth on its own.
    """

    name = "pamin-wide"
    store_path = PAMIN_HOME
    holder = "pamin"
    LIMIT = 30

    def _run(self, project, args, timeout=3600):
        return Pamin._run(self, project, args, timeout)

    def ingest(self, conversation_id, turns):
        # The same projects the flat arm built. `import` deduplicates, so this
        # is idempotent -- and its timing is therefore not an ingest
        # measurement and is not reported as one.
        self.project = ("locomo" + os.environ.get("FLAT_TAG", "v2")
                        + re.sub(r"[^a-z0-9]", "", conversation_id.lower())[:26])
        started = time.time()
        out = self._run(self.project, ["search", "anything", "--limit", "1", "--json"])
        if not json.loads(out)["hits"]:
            raise RuntimeError(
                f"{self.project} is empty; this arm reuses the flat arm's projects "
                f"and cannot build them")
        return time.time() - started, len(turns)

    def recall(self, question):
        out = self._run(self.project,
                        ["search", question, "--limit", str(self.LIMIT), "--json"])
        return [h["content"] for h in json.loads(out)["hits"]]


class PaminLedger:
    """pamin with the parts of it this comparison has never switched on.

    The flat arm writes every turn as an independent memory and puts the date
    in the text. This one uses the ledger: each session is imported under its
    own `--valid-from`, so when a claim started holding is a field rather than
    a prefix, and consecutive turns are linked, so the graph channel can reach
    the rest of an exchange from whichever turn matched.

    Still no model on the write path -- that is the whole point of running it.
    Every command here is one a user can type.
    """

    name = "pamin-ledger"
    store_path = PAMIN_HOME
    holder = "pamin"
    MONTHS = {m: i for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"], start=1)}

    def _run(self, project, args, timeout=3600):
        return Pamin._run(self, project, args, timeout)

    @classmethod
    def rfc3339(cls, when):
        """LOCOMO's `1:56 pm on 8 May, 2023` as a timestamp the CLI accepts."""
        m = re.match(r"(\d+):(\d+)\s*([ap]m)\s+on\s+(\d+)\s+(\w+),\s*(\d{4})",
                     (when or "").strip(), re.I)
        if not m:
            return None
        hour, minute, half, day, month, year = m.groups()
        hour = int(hour) % 12 + (12 if half.lower() == "pm" else 0)
        month = cls.MONTHS.get(month.lower())
        if not month:
            return None
        return f"{int(year):04d}-{month:02d}-{int(day):02d}T{hour:02d}:{int(minute):02d}:00Z"

    def ingest(self, conversation_id, turns):
        tag = os.environ.get("RUN_TAG", "led")
        self.project = ("locomo" + tag
                        + re.sub(r"[^a-z0-9]", "", conversation_id.lower())[:26])
        by_session = collections.defaultdict(list)
        for t in turns:
            by_session[t["session"]].append(t)

        started = time.time()
        dated = 0
        for session in sorted(by_session, key=lambda s: int(s.split("_")[1])):
            group = by_session[session]
            path = f"{WORK}/ndjson/{self.project}-{session}.ndjson"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                for t in group:
                    f.write(json.dumps({
                        "topic": re.sub(r"[^A-Za-z0-9_]", "_", t["id"]),
                        "content": t["record"],
                    }, ensure_ascii=False) + "\n")
            os.chmod(path, 0o644)

            # One import per session, because `--valid-from` is a property of
            # the call rather than of a line: a single import for the whole
            # conversation could only carry one date, which is the flat arm.
            args = ["import", "--from", path]
            stamp = self.rfc3339(group[0]["when"])
            if stamp:
                args += ["--valid-from", stamp]
                dated += 1
            self._run(self.project, args)

        # Consecutive turns linked, so a match anywhere in an exchange can
        # reach the rest of it through the graph channel `search` already has.
        for group in by_session.values():
            for first, second in zip(group, group[1:]):
                self._run(self.project, [
                    "link",
                    re.sub(r"[^A-Za-z0-9_]", "_", first["id"]),
                    re.sub(r"[^A-Za-z0-9_]", "_", second["id"]),
                    "--kind", "related_to",
                ])

        self._run(self.project, ["cascade", "drain"])
        if dated != len(by_session):
            raise RuntimeError(
                f"{dated} of {len(by_session)} sessions carried a parsed date; "
                f"this arm is not testing the ledger if the dates did not land")
        return time.time() - started, len(turns)

    def recall(self, question):
        out = self._run(self.project,
                        ["search", question, "--limit", str(TOP_K), "--json"])
        return [h["content"] for h in json.loads(out)["hits"]]


class Mem0:
    """mem0's own pipeline, with its LLM and embedder pointed at the shim."""

    name = "mem0"
    store_path = f"{WORK}/mem0-qdrant"
    holder = None          # qdrant runs inside this process

    def __init__(self):
        from mem0 import Memory
        # mem0 reaches OpenAI through the SDK's own environment, and the shim
        # is OpenAI-shaped, so pointing the environment at it is enough and
        # needs no per-provider knowledge of where each config key lives.
        os.environ["OPENAI_API_KEY"] = "shim-serves-claude-no-key-needed"
        os.environ["OPENAI_BASE_URL"] = SHIM
        self.Memory = Memory
        self.config = {
            "llm": {"provider": "openai", "config": {"model": "sonnet"}},
            "embedder": {"provider": "openai",
                         "config": {"model": "bge-m3", "embedding_dims": 1024}},
            "vector_store": {"provider": "qdrant",
                             "config": {"path": f"{WORK}/mem0-qdrant",
                                        "embedding_model_dims": 1024,
                                        "on_disk": True}},
        }

    def ingest(self, conversation_id, turns):
        import shutil
        shutil.rmtree(f"{WORK}/mem0-qdrant", ignore_errors=True)
        config = json.loads(json.dumps(self.config))
        config["vector_store"]["config"]["collection_name"] = re.sub(
            r"[^a-z0-9_]", "_", conversation_id.lower())
        self.memory = self.Memory.from_config(config)
        self.user = conversation_id

        # A session at a time, not a turn at a time. mem0 calls the LLM once
        # per `add`, so per-turn ingestion would be four hundred calls a
        # conversation; per-session is nineteen, and it is also the unit mem0's
        # own examples use.
        started = time.time()
        by_session = collections.defaultdict(list)
        for t in turns:
            by_session[t["session"]].append(t)
        for session in sorted(by_session, key=lambda s: int(s.split("_")[1])):
            group = by_session[session]
            messages = [{"role": "user",
                         "content": f"[{t['when']}] {t['speaker']}: {t['text']}"}
                        for t in group]
            self.memory.add(messages, user_id=self.user)
        return time.time() - started, len(turns)

    def recall(self, question):
        found = self.memory.search(
            question, filters={"user_id": self.user}, limit=TOP_K)
        results = found.get("results", found) if isinstance(found, dict) else found
        return [r.get("memory", "") for r in results]


class MemPalace:
    """MemPalace, mining the same conversations as files.

    Its shape is different from the others and the arm reflects that rather
    than working around it: it mines a directory, so each session becomes a
    file and the conversation becomes a wing. That is how a user points it at
    a transcript, and `mine --mode convos` exists for exactly this input.

    It runs in its own virtualenv. Installing it into the shared one moved
    protobuf past the ceiling mem0 declares, while a mem0 measurement was
    running -- the environment is part of the measurement, and a package
    manager is part of the environment.
    """

    name = "mempalace"
    store_path = f"{WORK}/mp-palace"
    holder = None
    LIMIT = None          # falls back to TOP_K
    PY = os.environ.get("MEMPALACE_PYTHON", f"{WORK}/mp-venv/bin/python")

    # Three variables, not two. `..._API_URL` and `..._API_MODEL` only say
    # where the endpoint is; `..._EMBEDDING_MODEL=openai-compat` is what
    # actually switches the provider. Setting the first two alone leaves it on
    # its own 384-dimension embeddinggemma and says nothing about it -- which
    # would have made this a comparison of embedders wearing the costume of a
    # comparison of memory systems.
    ENV = {
        "MEMPALACE_EMBEDDING_MODEL": "openai-compat",
        "MEMPALACE_EMBEDDING_API_URL": SHIM.rsplit("/v1", 1)[0],
        "MEMPALACE_EMBEDDING_API_MODEL": "bge-m3",
    }

    def _run(self, args, timeout=3600):
        env = dict(os.environ, **self.ENV)
        out = subprocess.run([self.PY, "-m", "mempalace", "--palace", self.palace] + args,
                             capture_output=True, text=True, timeout=timeout, env=env)
        if out.returncode != 0:
            raise RuntimeError(f"mempalace {args[0]}: {out.stderr[-400:]}")
        return out.stdout

    def ingest(self, conversation_id, turns):
        import shutil
        self.palace = f"{WORK}/mp-palace/{conversation_id}"
        source = f"{WORK}/mp-source/{conversation_id}"
        shutil.rmtree(self.palace, ignore_errors=True)
        shutil.rmtree(source, ignore_errors=True)
        os.makedirs(source, exist_ok=True)

        by_session = collections.defaultdict(list)
        for t in turns:
            by_session[t["session"]].append(t)
        for session, group in by_session.items():
            with open(os.path.join(source, f"{session}.md"), "w") as f:
                f.write(f"# {session} — {group[0]['when']}\n\n")
                for t in group:
                    f.write(f"**{t['speaker']}**: {t['text']}\n\n")

        before = shim_stats().get("embed_texts", 0)
        started = time.time()
        self._run(["init", source, "--yes",
                   "--llm-provider", "openai-compat",
                   "--llm-endpoint", SHIM, "--llm-model", "sonnet",
                   "--llm-api-key", "shim", "--accept-external-llm"])
        self._run(["mine", source])
        elapsed = time.time() - started

        # The premise: this arm exists to compare memory systems on one
        # embedder, and the provider switch fails silently when it fails.
        embedded = shim_stats().get("embed_texts", 0) - before
        if embedded <= 0:
            raise RuntimeError(
                "mempalace embedded nothing through the shared endpoint, so it "
                "used its own model; this arm would be comparing embedders")
        return elapsed, len(turns)

    def recall(self, question):
        out = self._run(["search", question, "--results", str(self.LIMIT or TOP_K)])
        blocks, current = [], None
        for line in out.splitlines():
            if re.match(r"\s*\[\d+\]\s", line):
                if current:
                    blocks.append("\n".join(current).strip())
                current = []
                continue
            if current is None:
                continue
            if re.match(r"\s*(Source|Match):", line):
                continue
            current.append(line.strip())
        if current:
            blocks.append("\n".join(current).strip())
        return [b for b in blocks if b]


class MemPalaceWide(MemPalace):
    """MemPalace at the wider shortlist, so no arm is alone in having one."""

    name = "mempalace-wide"
    LIMIT = 30


class Mem0Wide(Mem0):
    """mem0 at the same shortlist the wide pamin arm uses.

    Without this the comparison says only that more context helps, and gives
    the extra context to one side. Whatever `--limit 30` is worth, both
    systems have to be allowed it before either is called better.
    """

    name = "mem0-wide"
    LIMIT = 30

    def recall(self, question):
        found = self.memory.search(
            question, filters={"user_id": self.user}, limit=self.LIMIT)
        results = found.get("results", found) if isinstance(found, dict) else found
        return [r.get("memory", "") for r in results]


ARMS = {
    "bm25": Bm25,
    "pamin": Pamin,
    "pamin-wide": PaminWide,
    "pamin-ledger": PaminLedger,
    "mem0": Mem0,
    "mem0-wide": Mem0Wide,
    "mempalace": MemPalace,
    "mempalace-wide": MemPalaceWide,
}
