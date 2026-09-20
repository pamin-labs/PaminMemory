#!/usr/bin/env python3
"""What one answer costs a caller, envelope and all.

The query-side cost table in docs/benchmarks.md counts the passage *contents*
each system hands back, wrapped in this harness's reader template -- the same
template for every arm, which makes that column a ratio between the systems and
not a bill anybody pays. A caller never receives contents. It receives whatever
its memory system's API returns and it puts that in a context window: ids,
scores, timestamps, hashes, provenance, the lot.

This measures that. For one conversation and a handful of its own questions,
each arm is opened one level below `recall()` -- `recall()` exists to hand a
reader passages, and it throws the envelope away -- at the call the arm itself
makes, and the object that call returns is serialised the way a caller would
put it in a prompt and counted with `cl100k_base` through `run.tokens_in`, the
tokenizer the rest of the suite uses.

  pamin        the bytes `pamin search --limit 10 --json` writes to stdout,
               exactly as written, because that is what a caller pipes.
  mem0         the dict `Memory.search` returns, as `json.dumps` with Python's
               default separators.
  mempalace    the dict `searcher.search_memories` returns, the same way. The
               CLI that `arms.MemPalace` shells out to prints a rendering of
               this same dict and drops its fields; an application embedding
               MemPalace gets the dict, and it is the layer
               benchmarks/latency.py already times.

Serialisation is a choice and it is worth naming, because it is the one place
this could flatter somebody. JSON, not compacted: `json.dumps` defaults, which
is the form a caller gets by serialising the object it was handed. `pamin` is
not re-serialised at all -- its stdout is counted as the CLI emitted it, whether
that is generous or tight. So every arm also carries a `compact` column: the
same fields reserialised with every removable space gone. It is the floor a
caller reaches by re-encoding, it is measured the same way for all three, and
it keeps a difference in formatting from reading as a difference in what is
sent.

**What this does not measure.** Not accuracy -- no reader, no judge, nothing is
answered. Not latency. Not whether a caller *should* pass the envelope on: a
caller that wants only the text can extract it, and the point of the ratio is
precisely that doing so is worth something. Not the cost of a wider shortlist;
every arm here is at TOP_K, and a shortlist is priced in the table this one
sits beside. And no arm is re-ingested, so nothing here says anything about
what any of them cost to write.

**The premises, each asserted rather than assumed.**

  Same shortlist. Every arm answers every query with the same number of
  results, or this compares shortlist sizes wearing the costume of a
  comparison of envelopes. Checked across arms, per query, and the run
  refuses to print a table when it fails.

  Something came back. An empty result set has a very cheap envelope, and
  would make whichever arm's store had gone missing look frugal. Each arm's
  own store is one an earlier run built, so this is the failure to expect.

  The contents are real. Every passage is non-empty, so what is being
  measured is a wrapper around something rather than a wrapper around nothing.

  The envelope contains the contents. Its token count is at least that of the
  bare passages it carries; an envelope smaller than its payload means the
  wrong object was serialised.

  One embedder. MemPalace's provider switch fails silently -- three
  environment variables, and setting two leaves it on its own 384-dimension
  model -- so the arm asserts its queries moved the shared endpoint's embed
  counter, the same premise `arms.MemPalace` and `latency.py` assert. An arm
  retrieving with a different embedder returns different passages, and its
  envelope would then be an envelope around something else.

Conditions of the run this file's table was taken under:

    conv-50 of LOCOMO, 568 turns, 8 of its own questions, TOP_K=10,
    cl100k_base, four shared cores with a `cargo clippy` running beside the
    run -- which is why every row carries `shared_endpoint`, though what a
    shared machine makes unreadable is seconds and endpoint counters and
    there are none of either here.

    Every arm answers out of the store an earlier run left behind, and
    nothing is re-ingested: mem0 from the `conv_50` qdrant collection its
    ingest arm last wrote, MemPalace from `mp-palace/conv-50`, `pamin` from a
    workspace this measurement imported the same conversation into through
    `arms.Pamin`, at the shipped `accuracy` profile. Two passes over all
    three arms gave identical counts, which is what to expect of a
    deterministic retrieval and worth one run to know rather than assume.

    python3 benchmarks/envelope.py --unit conv-50 --questions 8

    arm             n  results  envelope   compact  contents   ratio
    pamin           8       10      1316      1316       532    2.43
    mem0            8       10      1675      1519       320    5.24
    mempalace       8       10      4560      4192      2051    2.22

`pamin` prints compact JSON, so its envelope and its compact floor are the
same number; the other two are dicts, and a caller that serialises them
tightly saves 8 to 9 per cent. Every number here is bare passage text against
the whole response. The reader-wrapped prompt the cost table publishes is a
little more -- 610, 394 and 2,128 on these eight questions -- and lines up
with that table's 557, ~384 and 1,756, which were medians over ten
conversations rather than this one.

The ratio is the finding: between 55 and 81 per cent of what a caller pays to
read one of these answers is not the memory. It is also the column that
travels least well, because it divides by the contents -- mem0 rewrites what
it stores and hands back the least text of the three, so it has neither the
largest envelope nor the smallest and the largest ratio by more than double.
Read the envelope column for what a context window pays and the ratio for how
much of that is packaging.

The first version of this file got `pamin` wrong by 2.4x, and the way it went
wrong is the reason for the assertion in that arm. `PAMIN_BIN` pointed at a
prebuilt binary nine hours older than `ad6c7db`, which took `topic_state`,
`source_span`, `weight` and `contribution` out of the envelope; the arm drove
the right command, on the right corpus, through the shipped entry point, and
measured 3,062 tokens for a response the product prints at 1,316. Nothing
about the run looked wrong. The arm now fails when a retired field comes
back, because the shape of the envelope is the measurement and a binary's age
is not visible in any other way.

Two honest gaps. Token counts are approximate in absolute terms, for the
reason `run.tokens_in` gives: what is comparable is the ratio between arms,
counted the same way on both sides. And the three envelopes are not three
renderings of one convention -- each system chose what to return, which is
what is being measured, but it does mean a field-by-field diff of them is a
comparison of API designs and not of retrieval.
"""
import argparse
import collections
import json
import os
import re
import statistics

import arms as arms_module
import run as run_module
from datasets import locomo

TOP_K = arms_module.TOP_K
SHARED = os.environ.get("BENCH_SHARED_ENDPOINT") == "1"


def render(obj):
    """A library's return value, the way a caller would put it in a prompt.

    JSON, because both library arms return JSON-shaped dicts and JSON is what
    such an object travels as. Not compacted: compacting is a choice a caller
    makes, and this is the cost of the answer as it comes. Anything that is
    not JSON-shaped is rendered with `repr` and says so on the row -- a quiet
    `default=str` would hide a field the caller could not have serialised
    either.
    """
    try:
        return json.dumps(obj, ensure_ascii=False), "json"
    except TypeError:
        return repr(obj), "repr"


def compacted(envelope):
    """The same fields with every removable space gone.

    A floor rather than a second measurement: the library arms are serialised
    with `json.dumps` defaults and a CLI prints whatever it prints, so part of
    an envelope can be whitespace a caller could re-encode away. Counted
    identically for all three, it says how much of the gap between arms is
    formatting rather than fields.
    Returns None for an envelope that is not JSON, which is not a failure --
    it is the reason the column can be absent.
    """
    try:
        return run_module.tokens_in(
            json.dumps(json.loads(envelope), ensure_ascii=False,
                       separators=(",", ":")))
    except ValueError:
        return None


class Pamin:
    """`pamin search --json`, which is the command a caller runs.

    The envelope is the CLI's own stdout, counted unmodified. `arms.Pamin`
    parses `hits[].content` out of exactly this string and throws the rest
    away, so the contents here are the same passages the accuracy run read.
    """

    name = "pamin"
    rendering = "cli stdout"

    # Fields `ad6c7db` took out of the envelope: two hit fields nothing can be
    # done with, and the two halves of a `why` entry that `docs/cli.md` gives
    # the formula for. A binary older than that commit is the trap this arm
    # walked into on its first run -- it printed the fat, indented shape and
    # measured 3,062 tokens against the 1,316 the product now prints, a number
    # 2.4x too high and attached to the right command. `PAMIN_BIN` points at a
    # prebuilt binary, nothing checks its age, and the fields are the only
    # thing that tells the difference, so they are the assertion.
    RETIRED = {"topic_state", "source_span"}
    RETIRED_WHY = {"weight", "contribution"}

    def attach(self, unit):
        self.arm = arms_module.Pamin()
        self.arm.project = arms_module.project_for("locomo", unit)
        self.store = f"{arms_module.PAMIN_HOME}:{self.arm.project}"

    def answer(self, question):
        out = self.arm._run(self.arm.project, [
            "search", question, "--limit", str(TOP_K), "--json"])
        hits = json.loads(out)["hits"]
        stale = set()
        for hit in hits:
            stale |= self.RETIRED & set(hit)
            for entry in hit.get("why", []):
                stale |= self.RETIRED_WHY & set(entry)
        if stale:
            raise SystemExit(
                f"{arms_module.PAMIN} still sends {sorted(stale)}, which "
                f"`pamin search --json` stopped sending in ad6c7db; this "
                f"binary predates the shape the product prints and its "
                f"envelope is not the one anybody pays")
        return out, [h["content"] for h in hits]


class Mem0:
    """`Memory.search`, the call `arms.Mem0.recall` makes.

    Attaches to the collection mem0's ingest arm last left behind rather than
    ingesting again: mem0 calls a model once per session and one conversation
    is roughly four hundred seconds, and nothing here needs a fresh store --
    it needs the store the published numbers came from.
    """

    name = "mem0"

    def attach(self, unit):
        arm = arms_module.Mem0()
        arm._assert_lemmatiser()
        config = json.loads(json.dumps(arm.config))
        config["vector_store"]["config"]["collection_name"] = re.sub(
            r"[^a-z0-9_]", "_", unit.lower())
        arm.memory = arm.Memory.from_config(config)
        arm.user = unit
        self.arm = arm
        self.store = config["vector_store"]["config"]["collection_name"]

    def answer(self, question):
        found = self.arm.memory.search(
            question, filters={"user_id": self.arm.user}, top_k=TOP_K)
        results = found.get("results", found) if isinstance(found, dict) else found
        envelope, self.rendering = render(found)
        return envelope, [r.get("memory", "") for r in results]


class MemPalace:
    """`searcher.search_memories`, the retrieval under MemPalace's CLI.

    `arms.MemPalace` spawns that CLI and scrapes its printed blocks, which is
    how a user searches a palace from a shell; an application gets this dict.
    Both read the same `results[].text`, so the contents column here is the
    one the accuracy run used and the envelope is what the CLI's renderer
    dropped.
    """

    name = "mempalace"

    def attach(self, unit):
        os.environ.update(arms_module.MemPalace.ENV)
        from mempalace import searcher
        self.searcher = searcher
        self.palace = f"{arms_module.MemPalace.store_path}/{unit}"
        self.store = self.palace
        self.embedded = arms_module.shim_stats().get("embed_texts", 0)

    def answer(self, question):
        found = self.searcher.search_memories(
            question, self.palace, n_results=TOP_K)
        envelope, self.rendering = render(found)
        return envelope, [r.get("text", "") for r in found.get("results", [])]

    def check_premise(self):
        """One embedder, asserted from the endpoint's counter and not the call.

        `MEMPALACE_EMBEDDING_MODEL=openai-compat` is the variable that switches
        the provider; the two URL variables alone leave it on its own model and
        say nothing. An arm that embedded its queries elsewhere retrieved
        different passages, and its envelope is around something else.
        """
        moved = arms_module.shim_stats().get("embed_texts", 0) - self.embedded
        if moved <= 0:
            raise SystemExit(
                "mempalace embedded nothing through the shared endpoint, so it "
                "used its own model; this would be an envelope around a "
                "different retrieval")


ARMS = {arm.name: arm for arm in (Pamin, Mem0, MemPalace)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="pamin,mem0,mempalace")
    parser.add_argument("--unit", default="conv-50",
                        help="the conversation every arm answers over; it has "
                             "to be one all of them already hold")
    parser.add_argument("--questions", type=int, default=8)
    parser.add_argument("--out", default="/tmp/bench/envelope/envelope.jsonl")
    args = parser.parse_args()

    entry = next((e for e in locomo.read()
                  if locomo.unit_id(e) == args.unit), None)
    if entry is None:
        raise SystemExit(f"{args.unit} is not in {locomo.DEFAULT_PATH}")
    questions = locomo.questions(entry, args.questions)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows, done = run_module.load(args.out)
    if done:
        print(f"resuming: {len(rows)} rows already in {args.out}", flush=True)
    sink = open(args.out, "a")
    where = run_module.machine()

    for name in args.arms.split(","):
        arm = ARMS[name]()
        arm.attach(args.unit)
        print(f"{name}: {arm.store}", flush=True)
        asked = 0
        for qid, qa, _reference in questions:
            if (name, args.unit, qid) in done:
                continue
            envelope, contents = arm.answer(qa["question"])
            asked += 1
            row = measure(name, arm, args.unit, qid, qa["question"],
                          envelope, contents, where)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
            rows.append(row)
            done.add((name, args.unit, qid))
            print(f"  {qid:<14} {row['results']:>3} results  "
                  f"{row['envelope_tokens']:>6} envelope  "
                  f"{row['contents_tokens']:>6} contents  "
                  f"{row['ratio']:>5.2f}x", flush=True)
        # Only when this pass actually asked something: a premise read off a
        # counter cannot be checked by a run that resumed every one of its
        # rows, and a harness that fails on a resume is one nobody re-runs.
        if asked and hasattr(arm, "check_premise"):
            arm.check_premise()
    sink.close()
    summarise(rows)


def measure(name, arm, unit, qid, question, envelope, contents, where):
    """One query's row, and the three premises one arm can check alone."""
    if not contents:
        raise SystemExit(
            f"{name} returned nothing for {qid}; an empty result set has a "
            f"very cheap envelope, and this arm's store ({arm.store}) is one "
            f"an earlier run built and may no longer hold this conversation")
    if len(contents) > TOP_K:
        raise SystemExit(
            f"{name} returned {len(contents)} results for a shortlist of "
            f"{TOP_K}; this arm is not at the shortlist it reports")
    blank = [i for i, text in enumerate(contents) if not (text or "").strip()]
    if blank:
        raise SystemExit(
            f"{name} returned {len(blank)} empty passages for {qid} (at "
            f"{blank[:3]}); this would be a wrapper measured around nothing")

    # The same prompt `run.py` builds, so this column joins onto the published
    # one rather than being a second convention for the same quantity.
    prompt = locomo.READER.format(
        context="\n".join(f"- {p}" for p in contents), question=question)
    bare = run_module.tokens_in("\n".join(contents))
    envelope_tokens = run_module.tokens_in(envelope)
    if envelope_tokens < bare:
        raise SystemExit(
            f"{name}'s envelope for {qid} is {envelope_tokens} tokens around "
            f"{bare} tokens of passages; an envelope cannot be smaller than "
            f"what it carries, so the wrong object was serialised")

    return {"arm": name, "unit": unit, "question_id": qid, "machine": where,
            "mode": "envelope", "store": arm.store,
            "rendering": getattr(arm, "rendering", "json"),
            "results": len(contents),
            "envelope_tokens": envelope_tokens,
            "compact_tokens": compacted(envelope),
            "envelope_bytes": len(envelope.encode()),
            "prompt_tokens": run_module.tokens_in(prompt),
            "contents_tokens": bare,
            "ratio": round(envelope_tokens / bare, 3),
            # Token counts are the arm's own under contention -- what a shared
            # endpoint makes unreadable is seconds and counters, and there are
            # none here. Stamped anyway, because the file outlives the shell.
            "shared_endpoint": SHARED}


def summarise(rows):
    """The table, after the one premise a single arm cannot check by itself."""
    if not rows:
        return print("no rows")
    by_arm = collections.defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    counts = collections.defaultdict(dict)
    for row in rows:
        counts[row["question_id"]][row["arm"]] = row["results"]
    disagree = {qid: seen for qid, seen in counts.items()
                if len(set(seen.values())) > 1}
    if disagree:
        for qid, seen in sorted(disagree.items()):
            print(f"  {qid}: " + ", ".join(f"{a}={n}" for a, n in sorted(seen.items())))
        raise SystemExit(
            f"{len(disagree)} of {len(counts)} queries came back with different "
            f"numbers of results across arms; an envelope over five hits and an "
            f"envelope over ten are not comparable, so no table is printed")

    print(f"\n{'arm':<14}{'n':>4}{'results':>9}{'envelope':>10}{'compact':>10}"
          f"{'contents':>10}{'ratio':>8}")
    for name in sorted(by_arm):
        group = by_arm[name]
        median = lambda field: statistics.median(r[field] for r in group)
        compact = [r["compact_tokens"] for r in group if r.get("compact_tokens")]
        print(f"{name:<14}{len(group):>4}{median('results'):>9.0f}"
              f"{median('envelope_tokens'):>10.0f}"
              f"{statistics.median(compact) if compact else 0:>10.0f}"
              f"{median('contents_tokens'):>10.0f}"
              f"{median('ratio'):>8.2f}")
    print("\ncontents is the bare passage text; the reader-wrapped prompt the "
          "cost table\npublishes is a few tokens more: "
          + ", ".join(f"{name} "
                      f"{statistics.median(r['prompt_tokens'] for r in by_arm[name]):.0f}"
                      for name in sorted(by_arm)))
    if any(r.get("shared_endpoint") for r in rows):
        print("\nsome rows were taken while another run shared the endpoint. "
              "Token counts are\nthe arm's own -- no seconds and no endpoint "
              "counters are reported here -- but the\nrows say so rather than "
              "leaving it to be remembered.")


if __name__ == "__main__":
    main()
