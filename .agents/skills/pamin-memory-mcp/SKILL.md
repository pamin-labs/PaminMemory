---
name: pamin-memory-mcp
description: "Use the `memory_*` MCP tools as durable memory: record what you learn, retrieve it later, and follow the relationships between facts. Use this skill whenever tools named memory_search, memory_read, memory_grep, memory_neighbors, memory_topics, memory_write, memory_link or memory_unlink are available and you need to remember something across turns or sessions, recall what was decided or observed earlier, look up a fact you were told before, trace how a fact changed, or find the exact text behind a claim. Reach for it whenever the user says remember this, what did we decide, what do we know about X, look it up, or refers to earlier work you cannot see in the current context, even when they never name memory. Also use it before answering from assumption: check memory first."
---

# Using Påmin Memory through MCP

The `memory_*` tools are a memory store. It keeps what you write as evidence,
promotes what is durable into versioned memories, indexes them, connects them
through a graph, and can explain why any result came back. Every tool works on
one project, chosen by whoever started the server; no tool takes a project.

The same operations exist as the `pamin` command line, with the same arguments
and results; the `pamin-memory` skill describes that surface.

## The one thing to understand first

Three layers, and confusing them causes almost every wrong answer:

```
evidence    everything ever written, verbatim, never altered  →  memory_grep
memories    the durable claims promoted out of evidence       →  memory_read
index       those memories, findable by ranking and a graph   →  memory_search
```

A write always reaches evidence. It reaches the other two **only if the filter
promotes it**: `memory_write` with content `"ok"` returns `promoted: false`
and `version: null`. That content is still found by `memory_grep`; it is not a
memory, and a held write to a new name creates no topic.

**Before writing to a name you invented, call `memory_topics`** with what it is
about. `deployment_pipeline` and `deploy_pipeline` are two memories that never
meet again. With no query it lists the most recent topics and the total.

## Choosing a tool

`memory_search` ranks, and ranking is a guess. The other read tools do not
guess, and when the ranking is what you doubt, they are the ones that answer.

| You want | Tool | Why this one |
| --- | --- | --- |
| Relevant context, you do not know where it lives | `memory_search` | The only ranked path |
| Everything connected to a topic | `memory_neighbors` | Structure, no ranking |
| What a topic says, or said | `memory_read` | Exact, by version |
| An exact string: an error code, an ID, a path | `memory_grep` | No model; reaches held evidence too |
| Whether a name already exists | `memory_topics` | By name and by content |

## Writing

`memory_write` takes `topic` and `content`. Writing to an existing topic adds a
version; the old one stays readable.

- **Write claims, not fragments.** Content too short or empty to carry a
  durable claim is held. Write the sentence a future reader needs: "the deploy
  pipeline moved to argo cd on 2026-03-01", not "argo cd".
- **Write the reasoning, not just the conclusion.** "The pool is 2x cores
  because the cascade worker holds one connection per job" answers the next
  question too.
- **Check `promoted`** in the result when it matters.
- `cascade` in the result says whether the index caught up before the call
  returned (`applied`) or still owes work (`queued`, done within seconds). The
  memory is recorded either way; set `defer: true` only when writing many in a
  row.

## Searching

`memory_search` returns **topics at what each says now**, five by default
(`limit`). Earlier versions are read, not ranked: `memory_read` with
`version_offset` reaches them.

A hit on a long memory returns the matching span and its neighbours, with a
`span` field. If the answer is not in that window, call `memory_read` with the
hit's `topic`, `span` set to `span.ordinal` and `around` set to how many spans
either side you want, rather than searching again. A hit with no `span` is the
whole memory.

### Reading the `why` trace

Every hit explains how it was found:

- **Several `channel` entries agreeing** is a strong hit.
- **`vector` alone** means nothing matched literally; often right, sometimes a
  near miss in meaning.
- **`graph` alone** means it came back only because it is connected to
  something that matched. Read its `path` entry and judge whether that route
  is relevant. `asserted_from` and `asserted_to` say which way the edge runs.
- **`reranked`** means the cross-encoder read the query and this memory
  together.

### When to stop searching

An uncached search costs about half a second on Apple silicon and two to three
seconds on four CPU cores, plus a page of results in your context. Search again
only when it can find something the first could not.

- **Before searching again, name what is missing.** A fact absent from five
  results is absent from one page, not from memory.
- **Rephrase at most once.** If the second phrasing returns mostly the same
  topics, switch to a tool that does not rank: `memory_grep` for an exact string
  the answer should contain (retry with `ignore_case: true` before counting a
  miss), `memory_topics` for the subject, or `memory_neighbors` with `depth: 1`
  from the best topic you found.
- **Stop and say what you tried** once those come back empty. Report that the
  probes found nothing, not that the memory does not exist. A probe that
  returned as many results as its `limit` has not come back empty.

### Which version backs a fact

- *What holds, or held, at a time*: the version must satisfy
  `valid_from <= time < valid_to`, with an absent bound open. A search hit is
  the current state whatever its validity, so check it.
- *What we knew or decided at a time*: step back with `memory_read` and
  `version_offset` until `recorded_at` is at or before that time. When
  `actual_version_offset` is smaller than the offset asked for, you have
  reached the oldest version; stop there.

`memory_grep` searches every version ever written, so check each hit against
the topic's version for the time asked about. A hit whose `filter_decision` is
`filtered` was never promoted: it is the only record of what was written, so
say so if you use it.

## Relationships

A memory that names another topic gets an edge automatically. For a
relationship the text does not state, call `memory_link` with `from`, `to` and
`kind`: one of `mentions`, `supports`, `contradicts`, `supersedes`,
`related_to` (the default), `part_of`, `derived_from`, `same_as`,
`depends_on`. **Both topics must already exist.** Linking twice is harmless.

`memory_neighbors` lists what is connected to a topic, two hops by default;
use `depth: 1` and `kinds` to keep it small on a well-connected topic, and
`at` (RFC 3339) to follow only edges asserted to hold at that instant.

`memory_unlink` retracts a relationship without erasing that it was claimed.
`reason: "closed"` (the default) means it ended; `"deleted"` means it was never
true. `closed: false` in the result means **nothing was open to retract**.

## The two timelines

- **When it is true**: `validity.valid_from` / `validity.valid_to`, RFC 3339, on
  `memory_write` and `memory_link`. Both are open by default.
- **When we believed it**: recorded for you, changed by writing a new version
  or by `memory_unlink`.

If something stopped being true and nobody knows when, do not set an open
`valid_to`, which asserts the opposite: use `memory_unlink` with its default
reason.

## Operational notes

- **A failed call is a tool result with an error**, and the message says why:
  an unknown topic, a malformed time, a server that could not start.
- **The first call is slow and the rest are not.** It starts a local server
  holding the database, the index and the models, which then stays up. The
  first search also prepares the reranker.
- **Results are stable for stable inputs** on one machine.
