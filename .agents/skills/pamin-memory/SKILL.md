---
name: pamin-memory
description: Use the `pamin` CLI as durable memory — record what you learn, retrieve it later, and follow the relationships between facts. Use this skill whenever `pamin` is available and you need to remember something across turns or sessions, recall what was decided or observed earlier, look up a fact you were told before, trace how a fact changed, or find the exact text behind a claim. Reach for it whenever the user says remember this, what did we decide, what do we know about X, look it up, or refers to earlier work you cannot see in the current context — even when they never name pamin. Also use it before answering from assumption on a project that has a pamin workspace: check memory first.
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
and `grep` see it immediately — and only `search` waits for the queue.

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
  { "kind": "channel", "channel": "lexical_ngram", "rank": 1, "weight": 0.25, "contribution": 0.0227 },
  { "kind": "channel", "channel": "vector", "rank": 1, "weight": 1.0, "contribution": 0.0909 },
  { "kind": "channel", "channel": "graph", "rank": 2, "weight": 1.0, "contribution": 0.0833 },
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

### Flags worth using, and one to leave alone

`--rerank` decides how much to spend reordering:

| | cost | when |
| --- | --- | --- |
| `off` | 39 ms | Every memory is in one language — the pass is nearly free of benefit there |
| `fast` (default) | 204 ms | Mixed languages, ordinary latency budget |
| `accurate` | 508 ms | Mixed languages, and half a second a search is affordable |

`accurate` scores better than `fast` on every corpus measured; `fast` is the
default purely on latency. Only candidates that no lexical channel found are
reordered, which is why a single-language workspace gains almost nothing.

**Leave `--channel-depth` and `--graph-depth` alone.** They exist for the
evaluation harness. Raising them costs latency for recall you have no way to
measure from outside. If you want more control over retrieval, use `grep`,
`read` and `neighbors` — paths whose behaviour you can actually verify.

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
  makes the rest fast.
- **Projects are namespaces, not tags.** `--project` (or `PAMIN_PROJECT`) picks
  one; nothing crosses between them.
- **Results are stable for stable inputs**, ties broken on identifier, so an
  assembled context can be cached rather than rebuilt.
- **`cascade` in a write's JSON** says whether the index caught up before the
  command returned. `applied` means yes; `queued` means `pamin cascade drain`
  still owes work. The memory is recorded either way.
- **Do not change `--profile` casually.** It changes the vector space, and the
  index refuses to open under a different one until `pamin reindex` rebuilds
  that project.

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
