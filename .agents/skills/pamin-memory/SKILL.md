---
name: pamin-memory
description: "Use the `pamin` CLI as durable memory — record what you learn, retrieve it later, and follow the relationships between facts. Use this skill whenever `pamin` is available and you need to remember something across turns or sessions, recall what was decided or observed earlier, look up a fact you were told before, trace how a fact changed, or find the exact text behind a claim. Reach for it whenever the user says remember this, what did we decide, what do we know about X, look it up, or refers to earlier work you cannot see in the current context — even when they never name pamin. Also use it before answering from assumption on a project that has a pamin workspace: check memory first."
---

# Using Påmin Memory

`pamin` is a local memory store. It keeps what you write as evidence, promotes
what is durable into versioned memories, indexes those four ways, and can
explain why any result came back.

You do not need a server, a key, or a config file. The first command provisions
everything and leaves it running.

## The one thing to understand first

Three layers, and confusing them causes almost every wrong answer:

```
evidence    everything ever written, verbatim, never altered  →  pamin grep
memories    the durable claims promoted out of evidence       →  pamin read
index       those memories, made findable four ways           →  pamin search
```

A write always reaches evidence. It reaches the other two **only if the filter
promotes it**. So "I wrote it and `search` cannot find it" is usually not a bug:

```console
$ pamin write --topic oncall "ok"
Held in evidence only: content was too short to carry a durable claim
```

That content is still recoverable with `grep`. It is not a memory.

Two consequences worth holding onto:

**Before writing to a name you invented, ask what is already here.** `pamin
topics <what it is about>` answers by name and by content and says which found
what. `deployment_pipeline` and `deploy_pipeline` are two memories that never
meet again, and nothing will ever tell you it happened. With no query it lists
the most recent, with the total, which is the fastest way to see what a
workspace is about.

- **A held write creates no topic.** Write to a new name, have it held, and the
  name stays unknown to `read`, `neighbors` and the graph. Check `promoted` in
  the JSON rather than assuming the topic now exists.
- **Evidence is never translated or rewritten.** Whatever a memory lost when it
  was condensed is still in the source, and `grep` reaches it.

## Choosing a command

The mistake to avoid is reaching for `search` every time. It ranks, and ranking
is a guess. Three of the four read paths do not guess at all, and when the
ranking is what you doubt, those are the ones that answer.

| You want | Use | Why this one |
| --- | --- | --- |
| Relevant context, you do not know where it lives | `search` | The only ranked path; fuses four channels |
| Everything connected to a topic | `neighbors` | Structure, no ranking anywhere in the path |
| What a specific topic says, or said | `read` | Exact, by version |
| An exact string — an error code, an ID, a path | `grep` | No tokenizer, no model; reaches held evidence too |

`search` when you want relevance, `grep` when you want certainty.

A concrete case: you are looking for error `E5521`. `search` may or may not
rank it first, and if someone wrote only `E5521` the filter held it and `search`
cannot see it at all. `pamin grep E5521` finds both the promoted memory and the
held fragment, and tells you which is which.

## Writing

```bash
pamin write --topic deployment_pipeline "deploys through the ci pipeline"
git log -1 --format=%B | pamin write --topic release_notes   # content from stdin
```

Topics are names you choose. Writing to an existing name appends a version; the
old one stays readable.

**Write claims, not fragments.** The filter holds content too short or too
empty to carry a durable claim, which means `"ok"`, `"yes"`, `"done"` and bare
identifiers never become memories. Write the sentence a future reader would need:
`"the deploy pipeline moved to argo cd on 2026-03-01"` rather than `"argo cd"`.

**Check `promoted` when it matters.** In `--json`, `version: null` and
`promoted: false` mean the filter held it. `source_version` is always set.

**Bulk loading: use `import`, not a loop.** One JSON object per line:

```bash
pamin import --from memories.ndjson
```

Getting a memory into the ledger costs about **3.2 ms this way against 30.1 ms
through repeated `pamin write`** — one process, one open engine, and the index
catching up in rounds instead of after each memory. The file is parsed in full
before anything is recorded, so a malformed last line refuses the import rather
than leaving it half done. Re-importing the same file is safe: unchanged
memories are held rather than duplicated.

If you must loop, use `--defer` and drain once at the end:

```bash
for f in *.md; do pamin write --defer --topic "$(basename "$f" .md)" "$(cat "$f")"; done
pamin cascade drain
```

`--defer` returns without embedding anything. The memory is committed — `read`
and `grep` see it immediately — and only `search` waits for the queue, which
the server works through on its own within about ten seconds; `cascade drain`
makes it catch up at once.

## Searching

```bash
pamin search "how does deployment work" --json
```

Results are **topics at what each says now**. A topic rewritten fourteen times
is one result, not fourteen, and the `version` reported is its current one.
Earlier versions are read, not ranked — `read --version-offset` reaches them.

### Reading the `why` trace

Every hit explains itself. This is the part worth actually reading, because it
tells you how much to trust the hit:

```json
"why": [
  { "kind": "channel", "channel": "lexical_ngram", "rank": 1 },
  { "kind": "channel", "channel": "vector", "rank": 1 },
  { "kind": "channel", "channel": "graph", "rank": 2 },
  { "kind": "path", "from": "oncall_rota", "via": "oncall_rota", "hops": 1, "edge": "depends_on", "derivation": "explicit" }
]
```

- **Several channels agreeing** is a strong hit. The lexical channels matched
  the words, the vector channel matched the meaning, the graph found it from a
  topic you named.
- **`vector` alone** means nothing matched literally. Often right, sometimes a
  near-miss in meaning — worth a second look before you rely on it.
- **`graph` alone** means nothing matched at all and it came back because it is
  connected to something that did. Read the `path` entry to see the route, then
  judge whether that route is relevant to what you asked.

Ranks travel between channels; scores do not, which is why fusion combines
ranks. A `path` entry is verifiable in a way a rank is not — anyone can check
that two topics really are related the way it claims.

### Flags to leave alone

Every search reranks with one cross-encoder, and there is no flag for it:
`--rerank` and `PAMIN_RERANK` were removed, and setting `PAMIN_RERANK` is an
error, as is every other retired settings variable (`docs/cli.md` lists them
and the flags that replaced them). A search against a warm server on four cores costs more than a second,
most of it the reranker (`docs/cli.md`).

**Leave `--channel-depth` and `--graph-depth` alone.** They exist for the
evaluation harness. Raising them costs latency for recall you have no way to
measure from outside. If you want more control over retrieval, use `grep`,
`read` and `neighbors` — paths whose behaviour you can actually verify.

### When to stop searching

Every `search` costs a reranking pass and a page of results in your context.
On a four-core CPU that is more than a second: the 1522 ms `docs/cli.md`
records was measured when the reranker saw twenty candidates, it now sees a
head of thirty plus up to thirty more that only the graph found, and
`docs/measured.md` records slower medians on busier machines. On Apple
silicon, where the reranker runs on CoreML, `docs/measured.md` records a
whole-search p50 of 0.48 s and p95 of 0.81 s. A second search pays off only when
it can find something the first could not. The rules below are a heuristic for
spending that cost; no agent-task measurement yet shows how often a repeat
search finds something new.

**What counts as backed.** A fact is backed when you have content that states
it, from a version that applies to the time the question asks about. Which
version applies depends on the timeline the question is on (see *The two
timelines*):

- *What holds, or held, at a time* (now, unless the question names another):
  the version must satisfy `valid_from <= time < valid_to`, the same half-open
  interval the store uses, and an absent bound is open. `search` returns a
  topic's current state without checking that interval, so a hit can state a
  fact that has lapsed or not yet begun.
- *What we knew or decided at a time*: the version that was current then.
  Step back with `read --version-offset` until `recorded_at` is at or before
  that time. Open validity bounds on the current version do not answer this.

Run every command in these rules with `--json`. The plain output of `read`
leaves out `valid_from` and `valid_to`, and that of `grep` leaves out
`filter_reason`, and the rules need all three. A walk past the oldest surviving
version returns that version again, with an `actual_version_offset` smaller than
the offset asked for. Stop there: if the oldest version does not apply, none
does, so say so rather than reading further.

A search hit carries its topic's full current content, so for a question about
now there is nothing more to `read` when its validity covers now. When it does
not, an earlier version may: step back with `read --version-offset`. `topics`
and `neighbors` return names and edges, not content: `read` each one that looks
relevant before you use it or rule it out.

- **Before searching again, name what is missing.** List the facts the
  question needs and check them against what you already have. A fact missing
  from the results is missing from one page of five, not from memory, so take
  it to the rules below rather than concluding it is not there.
- **Rephrase at most once, and never to the same place.** If the second
  phrasing brings back mostly the same topics (say, four of the five),
  the ranking is probably not the problem and a third phrasing is likely to
  land there too. Switch to a path that does not rank:
  - `grep` for an exact string you expect the answer to contain. It searches
    every version ever written, superseded ones included, so check each hit
    against its topic's version for the time asked about. For a CLI write,
    `source` is `manual:<topic>`; pass `read` the part after the colon. A hit
    whose `filter_decision` is `filtered` depends on its `filter_reason`. A
    restatement repeated the topic's state at the time, so the topic is
    checked like any other hit. For any other reason, no version took the
    content in, and if it was the topic's first write there is no topic to
    `read`. The hit is then the only record of what was written, so use it,
    saying it was recorded but never promoted to a memory. `grep` does not
    report when evidence was recorded, so for a question about what was known
    by a past time, say such a hit is undated rather than counting it.
  - `neighbors --depth 1` from the best topic you found, with `--kind` to
    narrow it. The default depth is two and nothing caps how many it returns,
    which on a well-connected topic can be most of the project. Without
    `--at` it follows every edge not yet retracted, whatever its validity: the
    graph as believed today. For a question about what holds or held at a
    time, now included, pass `--at <rfc3339>` for that time. `--at` filters by
    when an edge was true, not when it was recorded, so no flag rebuilds the
    graph as it was believed at a past time; for a question about what was
    known then, run it without `--at` and say in your report that it followed
    today's edges.

  `topics` is not one of them: it merges name matches with the same ranked
  search, unreranked. But it returns twenty by default against `search`'s
  five, so a topic it shows that you have not seen is worth reading, as is any
  name match. It also shows what the workspace calls things.
- **Stop and say what you tried once every probe has come back empty.** A
  probe that returned as many results as its limit (twenty by default for
  `grep` and `topics`) has not come back empty, whatever it held: narrow it or
  raise `--limit` first.
  1. two searches;
  2. a `grep` with no hit that backs the fact and no held hit you can use as
     above. Unless the casing is part of an identifier, a miss counts only
     after a retry with `-i`;
  3. `topics` for the same subject, reading any result you have not seen;
  4. a one-hop `neighbors` from the best topic, reading the neighbours that
     look relevant. Skip it only when no hit is related to the question.

  Report that those probes found nothing, not that the memory does not exist:
  `grep` matches one exact literal and `search` returns five results by
  default, so differently worded or lower-ranked memories can still be there.
  An answer built from the closest near-miss is worse than an honest "I
  searched for X and Y and found nothing".

## Relationships

Writing a memory that names another topic derives an edge automatically — no
model, no command, nothing to configure. Matching compares segmented tokens, so
it works in any language and will not find `db` inside `debt`.

For a relationship the text does not state, assert it:

```bash
pamin link oncall_rota deployment_pipeline --kind depends_on
```

Kinds: `mentions`, `supports`, `contradicts`, `supersedes`, `related_to`,
`part_of`, `derived_from`, `same_as`, `depends_on`. **Both topics must already
exist** — linking an unknown name is a typo far more often than intent.
Asserting twice is idempotent.

```bash
pamin neighbors rollback_plan --depth 1 --json
```

This is the way to see a derived edge that never ranked high enough to surface
in a search. `--kind` restricts it, repeatably, and `--at <rfc3339>` follows
only edges asserted to hold at that instant.

`pamin unlink` retracts a claim without erasing that it was made. Read
`closed: false` in the output carefully — it means **nothing was open to
retract**, which is not the same as having retracted something.

## The two timelines

Getting these backwards is the usual way a versioned store starts lying.

- **When it is true** — `--valid-from` / `--valid-to`, RFC 3339, on `write` and
  `link`. This is about the world. Both bounds are open by default, which is how
  most claims are actually stated.
- **When we believed it** — recorded for you, and changed by writing a new
  version or by `unlink`. This is about us.

The case that trips people: *it has stopped being true and nobody knows when*.
Do **not** express that as an open `--valid-to`, which asserts the opposite.
Use `pamin unlink`, whose default reason means the relationship ended rather
than that the claim was wrong.

## Operational notes

- **Every command takes `--json`**, and stdout carries only that JSON. Logs go
  to stderr, failures exit non-zero with the reason on stderr. Parse stdout.
- **The first command is slow and the rest are not.** It starts a server holding
  the database, the index and the model. You do not need to start or stop it;
  `pamin stop` exists but running it between commands throws away the thing that
  makes the rest fast. A few later commands are slow too: the first write, which
  downloads BGE-M3 and writes its mapped copy; the first search, which downloads
  and prepares the reranker; the first search after the server has given its
  models back (30 idle minutes, `--model-idle`); and the first search after an
  upgrade, which measures the reranker's execution plan again.
- **Server settings are fixed when the server starts.** Flags such as
  `--device` or `--model-idle`, and `$PAMIN_HOME/config.toml`, decide them. A
  command asking for settings the running server does not have fails and says
  so; run `pamin stop` only if you mean to restart it with them.
- **Projects are namespaces, not tags.** `--project` (or `PAMIN_PROJECT`, one
  of the few variables `pamin` still reads) picks one; nothing crosses between
  them.
- **Results are stable for stable inputs**, ties broken on identifier, so an
  assembled context can be cached rather than rebuilt.
- **`cascade` in a write's JSON** says whether the index caught up before the
  command returned. `applied` means yes; `queued` means the index still owes
  work, which the server runs within seconds and `pamin cascade drain` runs at
  once. The memory is recorded either way.
- **There is one embedding model.** `--profile` and `PAMIN_PROFILE` were
  removed, and setting `PAMIN_PROFILE` is an error. A project indexed under a
  retired profile refuses to open, naming it, until `pamin reindex` rebuilds
  it.

## Why this rather than a notes file

The obvious alternative to `pamin write` is appending to `CLAUDE.md`, a
`NOTES.md`, or whatever the project already has, and for one fact it is
genuinely simpler. It stops working for the same reason a pile of snippets
stops working:

- **Nothing reads it back.** A fact in `CLAUDE.md` is invisible to `search`,
  `grep` and `neighbors`. Split your memory across both and neither place is
  complete, and you will not remember which is which.
- **There is no query.** A notes file is loaded whole or not at all. That is
  fine at ten facts and useless at a thousand, because the answer to "what did
  we decide about connection pools" is a search, not a file.
- **It grows without bound, and you pay for all of it every time.** An
  auto-loaded file is a context tax on every session: recording the hundredth
  fact means paying for the ninety-nine before it, in every turn, forever.
  Retrieval exists precisely so recall costs the size of the answer rather than
  the size of the history.
- **Rewriting loses what was there before.** Editing a line in a notes file
  destroys the previous claim, when it was decided, and why. `pamin write`
  appends a version; `read --version-offset` still reaches the old one and
  `grep` still reaches the evidence behind it.

The place a notes file still wins is instructions you want loaded
unconditionally — conventions, build commands, what not to touch. Those are not
memories, they are configuration, and they belong in `CLAUDE.md`. The
distinction that holds: **if you would ever need to look it up, write it here;
if it must be in front of you before you do anything, write it there.**

## Working memory across a session

A pattern that works well: search before you assume, write once you have
concluded something durable.

```bash
pamin search "postgres connection pool sizing" --json    # before deciding
# ... do the work ...
pamin write --topic pg_pool_sizing \
  "the pool is sized at 2x cores because the cascade worker holds one connection per job"
```

What makes this worth doing rather than keeping notes in context: the write
records *why*, the next session can retrieve it without you, and `neighbors`
will surface it from any topic that mentions it later — including memories
written before the topic existed, since creating a topic links it to earlier
memories that already named it.

Write the reasoning, not just the conclusion. A memory that says
`"pool = 2x cores"` answers one question; the sentence above answers the next
person who wonders whether the number can change.

The full reference — every command, every flag, every JSON shape — is in
[docs/cli.md](../../../docs/cli.md).
