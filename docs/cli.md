# CLI reference

Every command takes `--json`. The usual caller is an agent parsing output rather
than a person reading it, so the text form is a convenience and the JSON form is
the contract — and what that contract leaves out is deliberate. Nothing carries
an identifier no command accepts, and nothing restates a number the reader can
compute from what is already there.

The examples below are real output from a workspace built by the writes in
[Getting started](#getting-started), captured rather than composed.

## Global options

| Option | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--home <path>` | `PAMIN_HOME` | `~/.pamin` | Where the database, index, and downloaded models live |
| `--project <name>` | `PAMIN_PROJECT` | `default` | The memory namespace to operate on |
| `--profile <name>` | `PAMIN_PROFILE` | `accuracy` | Embedding profile: `speed`, `balanced`, or `accuracy` |
| `--json` | | off | Emit JSON instead of text, on one line |
| `--pretty` | | off | Indent that JSON. Requires `--json` |
| | `PAMIN_POSTGRES_DIR` | unset | Use a PostgreSQL already on this machine instead of installing one |
| | `PAMIN_JIT` | `off` | Let PostgreSQL compile query expressions with LLVM |
| | `PAMIN_MODEL_IDLE` | `1800` | Seconds a resident server holds a model nothing is asking for |
| | `PAMIN_INFERENCE_THREADS` | one per core | Threads one forward pass may use |
| | `PAMIN_DEVICE` | a GPU if there is one | `cpu` keeps the reranker off the GPU |
| | `PAMIN_PREPARED` | on | `off` loads a model from its download rather than from a mapped copy |

The JSON is compact because the usual caller pays for every token of it, and
indenting a ten-hit search costs about a thousand of them. `--pretty` is for
the person who has piped it to a terminal.

`PAMIN_POSTGRES_DIR` points at an installation prefix holding `bin/initdb` --
`/usr/lib/postgresql/17` on Debian and Ubuntu, `$(brew --prefix
postgresql@17)` on macOS. Unset, a workspace installs its own copy, which is
the default because it is what makes `pamin` work with nothing else installed;
set, that copy is not downloaded and not stored, which is several hundred
megabytes a workspace does not spend. Two things to know before setting it.
The version requirement is not checked -- you are vouching for the server, and
the migrations expect PostgreSQL 17. And a figure measured against a server
built by somebody else is a figure for that server: fine for checking
behaviour, not interchangeable with the numbers in
[measured.md](measured.md).

`PAMIN_JIT=on` turns on PostgreSQL's LLVM compilation of query expressions.
It is off by default because it is measurably unreachable at every size
measured here, not because compiling is disliked: PostgreSQL only reaches for
it above a plan cost of 100000, and on a 13,014-topic project the read path's
hydration of fifty candidates plans at 84, the widest query the schema can
state plans at 1313, and `pamin grep` plans at 56 because its `ORDER BY`
matches an index and the scan stops early. The first of those does not grow
with the project at all -- it is bounded by `--channel-depth`.

One shape does grow. A `grep` for something the project barely contains has to
walk its whole recency index, and that cost is linear: 995 at 15,224 stored
versions, so roughly 65 for every thousand, reaching 100000 somewhere around a
million and a half. A workspace an agent has been writing to for a year is
exactly the one that gets there, which is why this is a switch and not a
constant. Two notes: it applies at `pamin stop` and the next start, like every
cluster-level setting here; and with it off, a workspace it installed itself
does not keep the 25 MB of LLVM bitcode that only inlining reads.

`PAMIN_MODEL_IDLE` is how long `pamin serve` keeps a model and the indexes
pinning it after nothing has asked for them. It is a memory setting and the
trade is measured on both sides: on a 13,014-document project a server holding
the embedder and the `fast` reranker is 2,263 MB resident and 88-101 MB once it
has given them back, so half an hour of quiet returns about 2.2 GB -- and the
first search afterwards takes 4,528 ms instead of 116, which is 88% of what a
server starting from nothing costs. Thirty minutes is where that stops being a
close call: an agent working in bursts does not wait half an hour between
searches, and a server left running overnight pays it once. Lower it on a
machine where memory is scarcer than four seconds; raise it if a search every
few minutes is worth 2.2 GB to you.

`PAMIN_INFERENCE_THREADS` is how many threads one forward pass may use.
Unset, the inference library uses one per core, which is right for a single
query and wrong for a server: measured on four cores with nothing shared,
throughput *falls* as callers are added, 182 embeddings a second at one worker
to 54 at eight, because the second caller finds the first caller's threads
rather than an idle core. Splitting the cores between callers instead of
between the layers of one pass is the other way to divide them, and which wins
is a property of the machine rather than of this program -- so it is a setting
whose default is what the library already did.

On the CPU, the first load of a model writes a second copy of it into
`models/prepared/`, and every load after that reads the copy. The copy is the
runtime's own optimized form of the graph with its weights in a separate data
file, which the runtime maps from disk instead of copying onto the heap: the
`accurate` reranker holds 271 MB of live memory rather than 822, the `accuracy`
embedder 272 rather than 824, with scores and vectors bit-identical (see
[measured.md](measured.md)). Most of what is left is the tokenizer's
vocabulary, which the embedder and every reranker share: with both loaded, the
two hold 330 MB rather than 582. What it costs is disk. Each copy is larger than the
model it came from, because the weights are also stored in the layout the CPU's
kernels use -- a data file of 874 MB for the 570 MB `accurate` reranker, about
as much for the embedder, 140 MB for the 119 MB `fast` reranker -- and writing
it makes that first load slower, 5.1 s for `accurate`. A copy belongs to the
runtime version and the CPU that wrote it, so an upgrade, or a model directory
moved to a different CPU, writes a new one and leaves the old one in place; it
is safe to delete `models/prepared/` at any time. `PAMIN_PREPARED=off` loads
from the download, for a disk that cannot spare the second copy. When a copy
cannot be written -- a full or read-only disk -- the model loads from the
download anyway and the log says why. Where the runtime left a model's
attention as separate operators -- the `accurate` reranker's int8 export --
the copy also gets a second graph, `attention.onnx`, with each layer's
attention as one fused operator; it is kept only if it scores a probe
bit-for-bit as the first does, and otherwise `attention.unfused` says why.
`PAMIN_FUSED_ATTENTION=off` loads the unfused graph, for measuring one
against the other.

The reranker runs on a GPU when the machine has one, with no flag and no
separate build. Each platform's inference runtime carries the accelerator that
platform has -- CUDA on x86-64 Linux, Core ML on Apple silicon, DirectML on
Windows -- and loading a reranker tries it first and falls back to the CPU when
it will not start. On a GPU it runs the model's half-precision export rather
than the CPU's int8 one, so the scores are close but not identical; which one
ran is logged when the model loads. `PAMIN_DEVICE=cpu` keeps it on the CPU, for
a comparison that has to be like for like or a GPU that belongs to something
else. Embedding stays on the CPU either way: the index was built with the CPU's
vectors and a query has to be embedded the same way to be compared with them.

On Linux the CUDA path has two requirements the program cannot meet for you.
The machine needs the NVIDIA driver, CUDA 13 and cuDNN 9. And the runtime's
provider libraries -- `libonnxruntime_providers_shared.so` and
`libonnxruntime_providers_cuda.so`, 79 MB, built into `target/release` beside
the binary -- have to sit in the same directory as `pamin`, which `cargo
install` does not arrange: copy them next to the installed binary. Missing
either, the reranker runs on the CPU exactly as before.

On Windows the same holds for one file, `DirectML.dll` (18.5 MB), built beside
`pamin.exe`. Without it the program still starts -- Windows 10 and later carry
their own copy in System32 -- but that copy can be older than the runtime
needs, in which case the reranker quietly stays on the CPU. On Apple silicon
nothing needs copying: Core ML is linked into the binary from the system.

What a GPU is worth has not been measured here, because nothing this project
is measured on has one. The ordering is checked instead: `every_device_orders_like_the_cpu`
in `crates/pamin-index/tests/reranking.rs` loads the reranker wherever it lands
and again forced onto the CPU, and asserts the two order clearly separated
candidates the same way.

A handful of other `PAMIN_*` variables exist and are deliberately not listed
here: they shorten a window or a budget so a test can reach a case, and a
caller has no way to evaluate them. They are named where they are read, and a
unit test checks that everything *not* on that list appears in the table
above, so a setting cannot be added without being documented or deliberately
excluded.

`PAMIN_LOG` sets the log filter (`PAMIN_LOG=debug`). Logs go to stderr, so they
never contaminate the JSON on stdout.

Changing `--profile` changes the vector space. The index records the profile it
was built with and refuses to open under a different one, naming `reindex` in
the error rather than silently mixing two spaces.

| Profile | Model | Width | Resident | Per query |
| --- | --- | --- | --- | --- |
| `speed` | multilingual-e5-small | 384 | 465 MB | 13 ms |
| `balanced` | multilingual-e5-base | 768 | 1.1 GB | 26 ms |
| `accuracy` (default) | BGE-M3, int8 weights | 1024 | 560 MB | 35 ms |

The default is the largest model because quantized weights make it the smallest
download and because the gap it closes is the one Påmin Memory is about: on the
evaluation corpus it roughly doubles cross-lingual retrieval against
`balanced`, matches it on same-language queries, and costs nine milliseconds.
`balanced` is kept for those nine milliseconds and for projects already indexed
under it; there is no other reason left to choose it.

Projects are namespaces, not tags. Each has its own index directory, so nothing
crosses between them and a rebuild of one leaves the others alone. That also
means each project carries its own embedding profile: changing `--profile` for
one is a rebuild of that one.

Every command exits non-zero on failure with the reason on stderr:

```console
$ pamin link nope deployment_pipeline --kind depends_on
Error: no topic named nope
```

## Getting started

```console
$ pamin init
Initialized project default in /home/you/.pamin
```

`init` provisions a local PostgreSQL and applies migrations. No Docker, no
configuration. The server is left running between commands so an agent invoking
the CLI repeatedly does not pay startup each time; `pamin stop` shuts it down.

**It will not run as root.** That is PostgreSQL's rule, not Påmin Memory's:
`initdb` refuses, so the bundled cluster cannot be created or started by a root
user. It matters because containers run as root by default, which makes this
the first thing many people hit. Add an unprivileged user and run as that one.
A root process can still be a *client* of a server an unprivileged user
started, which is what makes `docker exec` into a running workspace work.

```console
$ pamin write --topic deployment_pipeline "the deployment pipeline runs on continuous integration and publishes artifacts"
Wrote deployment_pipeline v1
$ pamin write --topic rollback_plan "a rollback reverts the deployment pipeline to the previous tag"
Wrote rollback_plan v1
$ pamin write --topic oncall_rota "the oncall rota rotates every monday morning"
Wrote oncall_rota v1
$ pamin write --topic deployment_pipeline "the deployment pipeline now runs on argo cd"
Wrote deployment_pipeline v2
```

## `pamin write`

Records a memory. Content comes from the argument, or from standard input when
it is omitted:

```console
$ git log -1 --format=%B | pamin write --topic release_notes
```

Evidence is stored before anything judges it, so a write that is not promoted to
a topic state is still recorded and still recoverable:

```console
$ pamin write --topic oncall_rota "ok"
Held in evidence only: content was too short to carry a durable claim
Stored as oncall_rota source version 2
```

```console
$ pamin write --topic oncall_rota "ok" --json
{
  "topic": "oncall_rota",
  "version": null,
  "promoted": false,
  "reason": "content was too short to carry a durable claim",
  "source_version": 3,
  "cascade": "applied",
  "cascade_lagging": false,
  "valid_from": null,
  "valid_to": null
}
```

A `null` version means the filter held it. `source_version` is always set: the
filter decides what reaches the retrieval surface, never what is kept.

A held write does not create the topic either. Writing to a name that does not
exist yet, and having the content held, leaves the evidence and no topic: the
name stays unknown to `read`, `neighbors`, and the graph channel until something
is promoted under it.

`--valid-from` and `--valid-to` state when the claim is asserted to hold, as
RFC 3339. Both are open by default:

```console
$ pamin write --topic winter_timetable "adds two evening departures" \
    --valid-from 2025-11-01T00:00:00Z --valid-to 2026-03-01T00:00:00Z
```

This is separate from when the memory was written. See
[Two kinds of time](#two-kinds-of-time).

Writing also derives relationships. See [Relationships](#relationships).

`cascade` says whether the projection caught up before the command returned.
The write itself commits the evidence, the span, the state and a record of what
the projection is owed, all in one transaction that touches only PostgreSQL;
the index is brought up to date afterwards. `applied` means that happened here.
`queued` means some of it is still owed and `pamin cascade` will run it — the
memory is recorded either way. See [`pamin cascade`](#pamin-cascade).

`--defer` records the memory and leaves the index to catch up later, so the
command returns without embedding anything:

```console
$ pamin write --defer --topic release_notes "cut 1.4.0 from main"
Wrote release_notes v1
```

The memory is committed exactly as it would be otherwise — `pamin read` and
`pamin grep` see it immediately — and only `search` waits for the queue. Use it
when importing in bulk and run `pamin cascade drain` once at the end: one
rebuild of the vector graph instead of one after every write.

`cascade_lagging` is set once the queue passes ten thousand owed jobs, and it
reports what the queue owed when the write looked at it rather than what is left
afterwards. Without it a cascade keeping up and one falling behind look
identical from outside, apart from searches missing the newest memories.

Ten thousand is also where `--defer` stops deferring: a write past it drains
before returning, so an import that ignores the signal still cannot run the
queue away. That costs the importer the work it created rather than pausing it,
which is the only form of backpressure that means anything here — ordinarily
nothing else is draining, so a writer that waited would slow the import and
leave the backlog exactly where it was. A new memory queues three jobs, so an
import pays for a batch about every three thousand of them and never carries
more than ten thousand.

## `pamin import`

Records many memories in one call, from a file of one JSON object per line.

```console
$ pamin import --from memories.ndjson
Imported 2400 memories: 2400 written, 0 held in evidence only
```

```json
{"topic": "deployment_pipeline", "content": "the pipeline runs on argo cd"}
{"topic": "oncall_rota", "content": "the rota rotates every monday morning"}
```

Each memory goes through the same filter, the same language detection and the
same transaction as `pamin write`; what changes is everything around them. One
invocation instead of one per memory, one open engine instead of one per
memory, and the projection catching up in rounds of sixty-four rather than
after each one. Measured on 2,400 memories, getting one into the ledger costs
**3.2 ms here against 30.1 ms** through `pamin write`, and a whole import of
2,400 previously-unseen memories takes 55 s against 119 s — the rest of which
is embedding them, which costs the same either way.

The file is read by whichever process holds the workspace, which is the server
when one is running. Both are on this machine and run as you.

It is parsed in full before the first memory is recorded, so a malformed last
line is a refusal rather than half an import. Importing the same file twice is
not an error: the second time every memory is unchanged, so the filter holds it
in the evidence layer and nothing reaches the index — which is what the `held`
count is reporting.

The importer watches the queue as it goes and pays it down if it passes the
depth that reports the projection behind, so an import cannot leave the index
arbitrarily far behind however large the file is. `cascade_lagging` in `--json`
says whether that happened.

`--valid-from` and `--valid-to` apply to every memory in the file.

## `pamin read`

Reads a topic at a version. `--version-offset` counts back from the current one.

```console
$ pamin read deployment_pipeline
deployment_pipeline v2 (current, 0 of 2 versions, recorded 2026-03-04T09:12:44.325845Z)

the deployment pipeline now runs on argo cd
```

```console
$ pamin read deployment_pipeline --version-offset 1 --json
{
  "topic": "deployment_pipeline",
  "version": 1,
  "content": "the deployment pipeline runs on continuous integration and publishes artifacts",
  "is_current": false,
  "actual_version_offset": 1,
  "oldest_version": 1,
  "latest_version": 2,
  "available_versions": 2,
  "recorded_at": "2026-03-04T09:12:44.098121Z",
  "observed_at": "2026-03-04T09:12:44.092905Z",
  "source_span": "4c570553-5eeb-4b4f-804c-339738c1a632"
}
```

`valid_from` and `valid_to` are absent here rather than null: a memory written
without an interval carries none, and the fields appear only when it has one.
`source_span` is the evidence this version was promoted from, and `pamin grep`
is what reads it.

An offset past the oldest surviving version clamps rather than failing, and
`actual_version_offset` reports how far the read actually reached. A caller
walking backwards can therefore stop when the two stop agreeing instead of
guessing at the depth first.

`recorded_at` and `observed_at` are the two timelines of
[Two kinds of time](#two-kinds-of-time), and comparing `recorded_at` across
versions is how "when did this change" is answered — the question a version
list on its own raises and cannot settle. `valid_from` and `valid_to` are the
asserted truth interval, open unless a writer bounded them. `pamin search`
carries `recorded_at` on each hit for the same reason: ranking says how well a
memory matches, not how current it is.

## `pamin search`

Retrieves across every recall channel and explains the result.

Results are topics, at what each says now. A topic rewritten fourteen times is
one result and not fourteen, and the `version` a hit reports is its current
one. Earlier versions are read rather than ranked: `pamin read
--version-offset` reaches them, and `pamin grep` reaches the evidence behind
them, including what the filter never promoted.

`--channel-depth` sets how many candidates each channel contributes before
fusion (default 50) and `--graph-depth` how many edges the graph walks out
(default 2). Both take `PAMIN_CHANNEL_DEPTH` and `PAMIN_GRAPH_DEPTH`.

These exist for the evaluation harness, which is what settles them. They are
not a tuning surface for ordinary use: raising the depth costs latency for
recall you cannot measure from outside, and an agent that wants control over
retrieval should reach for `grep`, `read`, and `neighbors` rather than adjust
ranking internals it has no way to evaluate.

`--rerank` chooses how much to spend reordering the results, and takes
`PAMIN_RERANK`:

| | what it loads | a search costs | cross-lingual nDCG@10 | same-language |
|---|---|---|---|---|
| `off` | nothing | 99 ms | 0.6114 | 0.7829 |
| `fast` | 119 MB | 359 ms | **+0.0397** | **−0.0060** |
| `accurate` | 571 MB | 1522 ms | **+0.0482** | +0.0006 |

All three rows are one run over the same 1,190 queries, taken when a tier
reranked twenty candidates; it now reranks thirty, which the `accurate` tier
turns into +0.0063 more cross-lingual (`p = 0.0001`) for half again as many
model pairs, and whose wall time has not been re-taken on a quiet machine. The
rows can be read against each other; none of them can be read against a figure published before
this table, and the `off` and `fast` rows moved when the fusion layer changed
underneath them. Paired bootstrap against `off`, 10,000 resamples: cross-lingual
`p = 0.0001` for both tiers; same-language `p = 0.0008` for `fast` and not
significant for `accurate`. `recall@50` is 0.8962 and 0.9571 in **every** arm,
to four decimals — a reranker reorders a shortlist and never changes what is in
it.

Measured on XQuAD-R's 13,014 sentences in eleven languages, through
`Engine::search_reranked` — the call this command makes, one layer below the
process it runs in. [measured.md](measured.md) reports the same corpus and
tiers as whole CLI invocations, 77/251/1241 ms, and the difference between the
two sets of figures is the invocation; neither is wrong and they are not
interchangeable.

What the `off` row is is worth knowing before reading the other two as
overhead. Most of it is not retrieval either: the four channels, fusion and
reading the states back are about 16 ms of it, and the rest is the forward
pass that turns your query into a vector — 68 ms on this profile's model, on
four cores, for a query the server has not been asked before. A resident
server remembers a query's vector, so asking the same thing twice costs the
16 ms alone. [ADR 0001](adr/0001-tech-selection.md) divides all four stages.

`accurate` is the default, on accuracy: it is the best tier on every corpus
measured, and query by query against `fast` it is ahead by 0.0086 cross-lingual
(`p = 0.0015`) and 0.0066 same-language (`p = 0.0001`) on XQuAD-R, and by
0.0411 on MIRACL Swahili (83 queries better, 12 worse, `p = 0.0001`). What that
costs is the latency column: 1522 ms against `fast`'s 359, about a quarter of
the throughput, and 571 MB loaded against 119.

`fast` was the default until it was measured against that order, and it is
**the only tier that measurably damages same-language ranking** — −0.0060 at
`p = 0.0008`, nineteen queries worse against three better, and on MIRACL at the
`speed` profile −0.0154 against no reranking at all (35 better, 58 worse,
`p = 0.014`). Ask for it when a search has to stay under half a second and the
workspace is mostly cross-lingual, which is where it still earns its place.

**That is not a reason to set `off` on a single-language workspace, and this
page used to say it was.** The same-language column above comes from parallel
text, where it is *the same 1,190 queries* as the cross-lingual column scored
against a different answer key — so every query in it still has correct answers
in ten other languages sitting in the index, which a real single-language
workspace does not. On the one genuinely single-language corpus measured, at
this profile, the pass **gains** 0.0201 at `fast` and 0.0496 at `accurate`. Set
`off` to buy back the time if you want the latency; do not set it expecting
better ranking.

The same run measured two more tiers, `balanced` and `noncommercial`, and both
were removed: `fast` beat each of them cross-lingual at under half the latency.
[ADR 0001](adr/0001-tech-selection.md) keeps their rows. Every tier left is
permissively licensed; [NOTICE](../NOTICE) lists what each one downloads and the
chain behind it.

The model is fetched the first time a search asks for one, into the same cache
as the embedding model.

It costs memory while it is loaded, and more than its download suggests: on a
13,014-document project a server serving `off` is 1,625 MB resident, and one
`fast` search takes it to 2,007 or 2,271 MB -- so between 380 MB and 645 MB for
a 130 MB model, the difference being the inference runtime's arenas rather than
the weights. `pamin serve` gives it back after five minutes with nothing asking
for that tier, which returns 368 to 380 MB of it to the operating system; the
arena growth above that stays. A workspace that sets `off` never pays it at
all. Those figures are for `fast`; `accurate`'s model is 571 MB against 119,
and its resident cost has not been taken on its own.

A reranker reads the query and a memory together, which is what lets it correct
an order the channels got wrong, and what makes it cost a forward pass for
every candidate it looks at. Only the candidates no lexical channel found are
reordered, and only into the positions they already hold — so a memory that
shares words with your query comes back where it was, whatever the reranker
thought of it. That is why the same-language column moves by thousandths rather
than by the hundredths the cross-lingual column moves. It does not hold the
column still: a same-language answer the lexical channels happened to miss is
an unlexical candidate like any other, and reordering can carry it down.

On a workspace in one language there are fewer such candidates, so there is
less for the pass to do -- but less is not nothing. On MIRACL, one language
throughout, `accurate` is still worth +0.0257 over `off` (67 queries better, 19
worse, `p = 0.0001`, `speed` profile); it is `fast` that is worth less than
nothing there.

A score depends on the query as well as the memory, so a resident server
remembers the ones it has computed and a repeated search pays nothing for them:
measured at 69.6 ms the first time and 0.0 ms the second, for the same ordering.
Four thousand scores are kept, about a quarter of a megabyte. Without
`pamin serve` there is no process to keep them in, so every command starts
from nothing.

The latencies are from four cores. Published figures for a reranker of this
size are a few milliseconds per candidate rather than the ten measured here,
and the difference is the core count; on an ordinary server `fast` is tens of
milliseconds.

`--graph-depth` accepts 0 to 4 and refuses anything larger. A topic's
neighbourhood grows multiplicatively with each hop and hub topics reach five
figures of degree, so a fifth hop is not a slower query but a differently sized
one. The walk also starts from at most 64 seeds, keeping the topics the query
named by name ahead of the ones the lexical and vector channels supplied.

```console
$ pamin search "how do we deploy" --limit 3
0.1970  deployment_pipeline v2  the deployment pipeline now runs on argo cd
        lexical_ngram#1 vector#1 graph#2 oncall_rota --depends_on-> deployment_pipeline (1hop)
0.1871  oncall_rota v1  the oncall rota rotates every monday morning
        lexical_ngram#3 vector#3 graph#1 oncall_rota --depends_on-> deployment_pipeline (1hop)
0.1811  rollback_plan v1  a rollback reverts the deployment pipeline to the previous tag
        lexical_ngram#2 vector#2 graph#3 rollback_plan --mentions-> deployment_pipeline (1hop)
```

The JSON carries the same trace, shown here with `--pretty` because it is being
read by a person. Without it the same result is one line and 151 tokens against
249, which is what indentation costs on a hit this size -- roughly a third off,
and more on a fuller result because the saving is per line:

```console
$ pamin search "how do we deploy" --limit 1 --json --pretty
{
  "query": "how do we deploy",
  "hits": [
    {
      "topic": "deployment_pipeline",
      "version": 2,
      "content": "the deployment pipeline now runs on argo cd",
      "score": 0.1969697,
      "why": [
        {
          "kind": "channel",
          "channel": "lexical_ngram",
          "rank": 1
        },
        {
          "kind": "channel",
          "channel": "vector",
          "rank": 1
        },
        {
          "kind": "channel",
          "channel": "graph",
          "rank": 2
        },
        {
          "kind": "path",
          "from": "oncall_rota",
          "via": "oncall_rota",
          "hops": 1,
          "asserted_from": "oncall_rota",
          "asserted_to": "deployment_pipeline",
          "edge": "depends_on",
          "derivation": "explicit"
        }
      ],
      "recorded_at": "2026-03-04T09:12:44.325845Z"
    }
  ]
}
```

### Reading the `why` trace

Three kinds of entry, and they answer different questions.

**`channel`** — this result appeared in that channel at that rank, and
contributed a share of its channel's weight to the score. Neither the weight
nor the contribution is sent: the weight is the constant in the table below,
and ten hits of both cost about seven hundred tokens to restate what the reader
already has. Nor is the score the channel gave it, for a different reason —
that is the channel's own quantity in the channel's own units, so a reader
comparing a BM25 score against a cosine similarity would be comparing nothing.
Fusion reads it to decide where inside its channel's share a candidate falls,
and the result of that reaches you as the rank in the fused list. There are
four channels:

| Channel | What it matches | Weight |
| --- | --- | --- |
| `lexical_segmented` | Words, after segmentation. Works in languages written without spaces | 0.125 |
| `lexical_ngram` | Substrings: file paths, error codes, function names, configuration keys | 0.125 |
| `vector` | Meaning, across languages | 1.0 |
| `graph` | Topics connected to what the other channels found | 0.30 |

The graph channel's weight was 1.0 until it was measured, which needed a corpus
with edges in it — every evaluation corpus here derived none, so the channel
returned nothing and its weight could not matter. Given eleven edges to walk it
turns out to cost 0.2794 nDCG@10 on a cross-lingual group at 1.0, against the
0.1673 it earns on queries whose answers are only reachable across an edge.
1.0 and 0.5 are significantly worse on the cross-lingual group once the whole
sweep is priced as one family, and 0.15 and 0.30 cannot be told apart; 0.30 is
kept because nothing supports moving it. `pamin_core::fusion` carries the sweep.

Its scores are also the only ones fusion does *not* rescale, and for the reason
this table's own note gives about comparability. A path strength is
`confidence × decay^(hops − 1)` over a `(0, 1]` confidence, so 0.5 means "one
derived mention" on every query in every project — a quantity that means the
same thing twice, which a BM25 score and a cosine similarity are not. Rescaling
it inside one query would map whatever the best path happened to be onto the
top of the band, so a single weak guess would vote as loudly as an explicit
assertion.

**Each channel's scores decide the order within its share, and the weights
decide the shares.** A BM25 score and a cosine distance are not comparable, so
a channel's scores are only ever compared against that channel's own — mapped
onto the same narrow band that reciprocal rank fusion would have spanned over
the same candidates, `[(k + 1) / (k + n), 1]`, where `n` is how many candidates
the channel returned.

The band is the part that matters, and it is derived rather than chosen. Over
fifty candidates at `k = 10` it is a factor of 5.45, which is narrow enough
that a channel's *weight* decides against another channel's position: a lexical
channel's top hit at an eighth weight lands below the vector channel's
fiftieth, so a strong channel's marginal candidate still makes the list and a
weak channel's confident one does not displace it. Normalising onto `[0, 1]`
instead spans a factor of infinity inside one channel, which inverts that — and
measurably so. A plain weighted sum of standardised scores scores higher on
nDCG@10 in three of four groups and takes XQuAD-R's cross-lingual `recall@50`
from 0.8960 to 0.7765. Nothing recovers a memory that was never returned, so
that trade is declined; `[ADR 0001](adr/0001-tech-selection.md)` has both
tables.

The two lexical channels carry an eighth of a weight each because at full
weight the pair outvotes the other two on exactly the queries where the wording
matches and the meaning does not. They were also once described here as nearly
the same channel, and they are not: Kendall tau-b between their rankings is
0.2816, 0.3188 and 0.2973 on the three corpora this project measures, so they
agree about a third of the time. They share a field, not a ranking. The eighth
each is one number doing the work of two — no sweep has ever moved them
independently, and the n-gram channel is the weaker of the two wherever either
is measured alone. An eighth rather than the quarter that shipped
before because on MIRACL Swahili — 482 questions people asked, judged by
people — the quarter ranked *worse* than the vector channel by itself, and
because the quarter had never been compared against anything smaller than
itself. Three corpora and the sweep behind that are in
[ADR 0001](adr/0001-tech-selection.md).

One weight serves every workspace, and the evidence says that is the wrong
shape rather than the wrong value. What the lexical pair is worth depends on
whether a query and its answer share a language at all: nothing across a
boundary, and a great deal within one. Measured, fusing all four channels ranks
*below* the vector channel alone on cross-lingual queries — 0.6114 against
0.6335 on XQuAD-R, 481 wins to 35, p = 0.0001 — while on the same corpus's
same-language queries the lexical channels are worth +0.1042 and segmented BM25
alone beats the vector channel 0.7299 to 0.6787. A constant cannot be right
about both, and the two costs are what decide which way it should be wrong.

Reading the scores inside the band is what closes part of that gap without
picking a side: over the same 1,190 questions it is worth +0.0037 cross-lingual
and +0.0273 same-language, both significant, with recall unmoved. **It is the
first change here that improves the same-language group rather than charging
it** — every weight this project ever moved took something from that group to
pay for the other one.

A per-channel confidence was built for the same gap and is **off, because it
was measured and refuted**: it makes the largest cross-lingual group
significantly worse, which is the group it existed to help. The `10` is likewise measured here rather than taken from the rank
fusion literature, which uses 60 for lists thousands of results deep; each
channel proposes fifty, and 60 flattens fifty candidates to the point where
being first says almost nothing.

Fusion happens here rather than inside the retrieval engine. The engine offers to
fuse its own channels and that offer is declined: the graph lives in PostgreSQL
where the engine cannot see it, so an engine-fused list would have to be fused
again and its members counted twice, and the per-channel ranks would already be
gone.

**`path`** — accompanies a `graph` entry and says how the graph reached this
result: the topic the walk started `from`, the topic on the other end of the
final edge (`via`), how many edges were crossed, which relationship, and whether
it was asserted by a caller (`explicit`) or derived by the engine
(`deterministic`). At one hop `from` and `via` are the same topic; past that
they are not, and both are needed to follow the route. Nobody can verify a
reciprocal rank; anyone can verify that two topics are related the way the path
claims.

`asserted_from` and `asserted_to` say which way the edge itself runs. The walk
ignores direction, because both ends of a `depends_on` are relevant to recall
— but that means `via` describes the route taken and not the claim, and the
same edge would otherwise read in opposite directions depending on which end
the walk started from. For `depends_on`, `supersedes`, `contradicts`,
`derived_from` and `part_of` the direction *is* the claim, so it is stated
rather than left to be inferred.

**`reranked`** — the cross-encoder decided this result's position, and fusion
did not. The example above carries no such entry, and correctly: a lexical
channel found that result, so the pass left it where fusion put it. It carries nothing else, and the omission is the design rather than a
shortcut: a cross-encoder's score is calibrated against nothing, so it
separates the candidates of one shortlist and means nothing between two
queries, and a number on the wire invites exactly the comparison it cannot
support.

What it does tell you is the part nothing exposed before. A result **with** this
entry was reordered by the model. A result **without** it holds the place
fusion gave it — either a lexical channel found it, so the pass deliberately
left it alone, or it sat below the tier's depth and the model never saw it. So a
line reading `vector#12 reranked` says the fused list had this twelfth and the
model moved it, and a line reading `lexical_segmented#3 vector#7` says the two
channels agreed and no model was consulted. Auditing a ranking needs that
distinction, and before this it was not derivable from anything the command
returned. `--rerank off` produces no entries of this kind at all.

There is no fourth kind. There used to be a `modifier`, a post-fusion
adjustment that lifted a result by its recorded `importance` and by the balance
of outcomes it took part in. Both were read from columns nothing ever wrote, so
each one multiplied every result by exactly 1.0 on every search anyone ran, and
the adjustment was removed rather than left to look like a ranking signal.

## Relationships

The graph connects topics, and each endpoint resolves to whichever version is
current when a query runs. Edges arrive two ways.

### Derived automatically

Writing a memory that names another topic derives an edge to it. No command is
involved:

```console
$ pamin neighbors rollback_plan --depth 1
deployment_pipeline  1 hop  rollback_plan --mentions--> deployment_pipeline (deterministic, 0.50)
```

`rollback_plan` says "reverts the deployment pipeline", which names
`deployment_pipeline`, so the edge exists. Matching compares segmented tokens
rather than substrings, so it works in any language and does not find `db`
inside `debt`. Creating a topic also links it to memories written earlier that
already named it.

Derived edges carry lower confidence than asserted ones, which orders neighbours
at equal distance.

### Asserted explicitly

Derivation only finds relationships the text states. For anything else — a
dependency, a contradiction, a supersession that nobody wrote down — assert it:

```console
$ pamin link oncall_rota deployment_pipeline --kind depends_on
oncall_rota --depends_on--> deployment_pipeline (v1)
$ pamin link oncall_rota deployment_pipeline --kind depends_on
Already linked: oncall_rota --depends_on--> deployment_pipeline (v1)
```

Asserting is idempotent, so re-running it changes nothing.

Kinds: `mentions`, `supports`, `contradicts`, `supersedes`, `related_to`,
`part_of`, `derived_from`, `same_as`, `depends_on`. Both topics must already
exist; linking a name that does not is a typo far more often than it is intent.

`--valid-from` and `--valid-to` bound when the relationship is asserted to hold,
as RFC 3339. Both are open by default, which is how most relationships are
stated. This is separate from when we recorded the claim.

### `pamin unlink`

Retracts a claim. Every row stays, and `--reason` says what the retraction means:

| `--reason` | Meaning | What history keeps |
| --- | --- | --- |
| `closed` (default) | The relationship ended | Queries about earlier instants still find it |
| `deleted` | The claim was wrong | No instant finds it; it never held |

```console
$ pamin unlink oncall_rota deployment_pipeline --kind depends_on
Retracted oncall_rota --depends_on--> deployment_pipeline (closed)
$ pamin unlink oncall_rota deployment_pipeline --kind depends_on --json
{
  "from": "oncall_rota",
  "to": "deployment_pipeline",
  "kind": "depends_on",
  "reason": "closed",
  "closed": false
}
```

`closed: false` in the output means nothing was open to retract, which is
different from having retracted something.

Retracting is how you say "this stopped being true and I do not know when". An
open `--valid-to` means the claim still holds, so using it to mean "it ended at
some unknown point" asserts the opposite. See
[Two kinds of time](#two-kinds-of-time).

### `pamin neighbors`

Walks the graph with no ranking anywhere in the path.

```console
$ pamin neighbors rollback_plan --depth 1 --json
{
  "topic": "rollback_plan",
  "depth": 1,
  "neighbors": [
    {
      "topic": "deployment_pipeline",
      "hops": 1,
      "via": "rollback_plan",
      "edge": "mentions",
      "asserted_from": "rollback_plan",
      "asserted_to": "deployment_pipeline",
      "derivation": "deterministic",
      "confidence": 0.5
    }
  ]
}
```

Search returns what it judges relevant; this returns what is connected. It is
the question to ask when the ranking itself is what you doubt, and the only way
to see a derived edge that never placed high enough to surface in a search.

Traversal ignores edge direction, since both ends of a `depends_on` are relevant
to recall — so the arrow printed is the one the edge was asserted with, and
`asserted_from` / `asserted_to` carry it in the JSON. Without that the same edge
reads one way walked from one end and the opposite way walked from the other,
which for a question like "what depends on this" is the whole answer. `--kind` restricts it, repeatably. `--at <rfc3339>` follows only edges
asserted to hold at that instant, which is how a question about the past avoids
relationships that were only claimed later. `--depth` accepts 0 to 4, for the
reason given under [`pamin search`](#pamin-search).

## `pamin topics`

Every other read command needs a topic name to start from. This is how you get
one.

```console
$ pamin topics --limit 4
recent    secret_rotation
recent    build_cache
recent    access_review
recent    alert_thresholds

Showing 4 of 12 topics
```

The total is there because the page without it means nothing: four of twelve is
most of the story, four of nine thousand is a sample and you should be asking a
narrower question.

With a query it answers twice over and says which route found what, because the
two fail differently:

```console
$ pamin topics "deployment pipeline" --limit 4
both      deployment_pipeline
content   oncall_rota
content   secret_rotation
content   rollback_plan
```

**`name`** — the topic is called that. Exact on the segmenter's whole tokens, so
`deployment pipeline` reaches `deployment_pipeline` and `deploy pipeline` does
not. This is the route that finds a topic nobody has written much about yet.

**`content`** — a memory under that topic matches. Forgiving, and the route that
catches a half-remembered name: `deploy pipeline` finds `deployment_pipeline`
here even though the name index will not.

**`both`** — each found it, which is the strongest signal that this is the topic
you meant.

Reach for this before writing to a name you invented. `deployment_pipeline` and
`deploy_pipeline` are two memories that never meet again, and nothing will ever
tell you that happened.

## `pamin grep`

Finds an exact string in the evidence. No pattern matching, no tokenizer, no
ranking model anywhere in the path.

```console
$ pamin write --topic incident_log "the checkout service returned E5521 during the tuesday outage"
Wrote incident_log v1
$ pamin write --topic incident_log "E5521"
Held in evidence only: content was too short to carry a durable claim
Stored as incident_log source version 2
```

```console
$ pamin grep E5521
manual:incident_log v2 (filtered)
        E5521
manual:incident_log v1 (promoted)
        the checkout service returned E5521 during the tuesday outage
```

The second write never became a memory, so `pamin search` cannot see it — which
is the filter working correctly. It is still evidence, and this is the route to
it:

```console
$ pamin grep E5521 --json
{
  "literal": "E5521",
  "matches": [
    {
      "source": "manual:incident_log",
      "source_version": "54cf4923-c45b-4cfe-a788-6588fb7eae6d",
      "version": 2,
      "filter_decision": "filtered",
      "filter_reason": "content was too short to carry a durable claim",
      "excerpt": "E5521"
    },
    {
      "source": "manual:incident_log",
      "source_version": "7f4b4500-9472-436e-b537-2d1cab87f268",
      "version": 1,
      "filter_decision": "promoted",
      "filter_reason": "promoted to the retrieval surface",
      "excerpt": "the checkout service returned E5521 during the tuesday outage"
    }
  ]
}
```

Every match reports whether it reached the retrieval surface and why. A
mandatory filter is only safe if its mistakes can be found, and this is how they
are found.

`-i` folds case; matching is case sensitive otherwise. `--limit` bounds the
result count.

It reaches superseded versions too, so it answers "what did that memory say
before it was rewritten" without walking the version list. Nothing tokenizes, so
it works on any language, on partial identifiers, and on strings a segmenter
would split.

Use `search` when you want relevance, and `grep` when you want certainty.

## Two kinds of time

Every claim carries two independent timelines, and confusing them is the usual
way a versioned store starts lying.

**When it is true** — `--valid-from` and `--valid-to` on `pamin write` and
`pamin link`. This is about the world. Both bounds are open by default, which is
how most claims are actually stated: a fact is asserted to hold, not asserted to
hold between two dates.

**When we believed it** — recorded automatically, and changed by `pamin unlink`
or by writing a new version. This is about us.

`neighbors --at <rfc3339>` asks the first question; `neighbors` with no `--at`
asks the second.

`search --json` carries both on every hit, as `valid_from`/`valid_to` and
`recorded_at`, so an agent deciding which of two contradicting memories to
believe does not pay a `read` per result. Reach for the first pair. Recording
time is almost never the answer: everything written in one `pamin import`
shares it to within milliseconds, so ordering by it recovers the order the file
was fed in rather than the order the facts became true. The bounds are `null`
when the writer stated none, which is the ordinary case and is not the same as
a claim that stopped holding.

Two cases look like they need a third kind of end date, and do not:

- *It still holds, and nobody knows when it will stop.* That is what an open
  `--valid-to` already means, and it describes almost every claim ever made. A
  separate way to say it would give the same answer at every instant.
- *It has stopped, and nobody knows when.* Recording that as an open
  `--valid-to` asserts the opposite. It is a statement about what we still stand
  behind, so it belongs on the other timeline: `pamin unlink`, which by default
  means the relationship ended rather than that the claim was wrong.

## `pamin reindex`

Discards the projection index and rebuilds it from PostgreSQL.

```console
$ pamin reindex
Rebuilt the index from postgres: 4 states
```

The index holds nothing PostgreSQL cannot reproduce, which is what makes the
retrieval engine replaceable and makes a breaking engine upgrade a rebuild
rather than a migration. Relationships are unaffected: they live in the
authority store, not the index.

Run it after changing `--profile`, or after deleting the index directory. It
rebuilds one project — the one named by `--project` — and leaves the rest alone.

It is also how a grown project resizes its vector segments when no server is
running. The index sizes them from the number of memories it holds when it is
created, which for a project starting from nothing is the smallest size; a
project that has since grown by orders of magnitude keeps that size until the
index is recreated. Rebuilding recomputes it from what the project holds now,
so a project that has outgrown its layout searches faster afterwards. Where the
old index was built the way a new one is, a memory whose text has not changed
keeps the vector it already has rather than being embedded again.

A running server does this on its own. Once a project's index is spread over
more than twice the segments it should be, the server copies it into the right
shape in the background — reading it a batch at a time while it goes on
answering searches and writes, and building the copy's vector graph with the
index free — then swaps the copy in. Writes made during the copy are carried
over before the swap. Nothing is embedded, and for the length of the copy the
disk holds the index twice. `pamin cascade drain` reports a badly shaped index
either way.

A workspace created before projects had separate indexes holds a single shared
one. Opening it would search another project's memories, and ignoring it would
search nothing, so commands report it and `pamin reindex` migrates it.

## `pamin cascade`

Runs the work a write left for the projection.

A write records the memory and, in the same transaction, a record of what the
index still owes it: the embedding, the vector and lexical entries, and the
relationships the content implies. Nothing derived happens inside that
transaction, so a memory is never recorded without its follow-up work also
being recorded — and a process that dies between the two leaves the work owed
rather than lost.

`pamin write` runs the queue before it returns, so ordinarily there is nothing
here to do. These commands are for when there is: a queue left behind by a
process that was killed, writes made with [`--defer`](#pamin-write), work
deferred because something it needed was unavailable, and jobs that failed
often enough to be set aside.

```console
$ pamin cascade drain
Ran 3 jobs, 0 failed, 0 still owed
```

`drain` runs everything that is due and stops. `run` keeps going, waiting for
new work until it is interrupted; it holds the index open for writing the whole
time, so no other command that writes can run alongside it.

Jobs name a subject rather than an event — "bring this topic up to date", not
"this topic changed" — so running one twice leaves the same result as running
it once, and fourteen edits to one topic leave one job rather than fourteen.

A job that fails is tried again an hour later, up to eight times. After that it
is set aside with the error that stopped it, rather than retried forever:

```console
$ pamin cascade failed
Nothing has failed
```

```console
$ pamin cascade failed --json
{
  "failed": []
}
```

`pamin cascade replay` makes those jobs due again, for when whatever broke them
is fixed. `pamin cascade discard` abandons them. Both report how many they
moved:

```console
$ pamin cascade replay
Queued 0 failed jobs to run again
```

Nothing here can lose a memory. The queue drives the index, and the index holds
nothing PostgreSQL cannot reproduce — `pamin reindex` rebuilds it outright.

## `pamin serve`

Holds the database, the index and the model, and answers commands over a socket
at `$PAMIN_HOME/pamin.sock`.

You do not normally run it. Any command that needs a server starts one and
connects, the same way `pamin init` leaves PostgreSQL running so the next
command does not pay for it:

```console
$ pamin read deployment_pipeline    # first call, starts a server
deployment_pipeline v2 (current, 0 of 2 versions)

the deployment pipeline now runs on argo cd
```

What that buys is everything a short-lived process used to rebuild. Before, each
command connected to PostgreSQL, checked migrations, opened the index, and
loaded an embedding model before it did any work of its own.

`pamin serve` runs it in the foreground instead, which is useful when you want
to watch it. A server started in the background writes to
`$PAMIN_HOME/serve.log`; `PAMIN_LOG` sets its level, as everywhere else.

Between requests it looks after the indexes it holds open: it makes applied
writes durable, compacts an index spread over too many files, and reshapes one
spread over too many segments, as `pamin reindex` describes. A reshape logs
`reshaping the index` when it starts and `reshaped the index` with the segment
counts and its duration when it finishes, at the `info` level that
`PAMIN_LOG=info` shows; a reshape that fails logs a warning, which shows by
default, and leaves the index it was copying in service.

One server serves many projects, and it keeps the sixteen most recently used
indexes open; the seventeenth closes the one nobody has touched for longest.
Sixteen is a count of file descriptors, which is what the bound was built for,
and not a budget in bytes, which is what actually runs out: an open index costs
about 100 MB on `speed` and about 205 MB on the default `accuracy`, so sixteen
of them is 1.6 GB or 3.3 GB depending on a flag. On a machine where that is too
much, `PAMIN_OPEN_INDEXES` sets a smaller number. Lowering it costs nothing but
a reopen when a query lands on a project that has fallen out.

`PAMIN_NO_SERVER=1` runs everything in the calling process, as it did before.
The results are identical — it is the same code either way — so this is for
debugging the server itself, and for a caller that would rather have one process
to reason about than a fast one.

It does not combine with a server that is already up. A running server holds the
index open for writing, and the index takes an exclusive lock on its directory,
so a second process opening the same project fails rather than waiting. That is
the lock doing its job: two processes writing one index is what it exists to
prevent. Run `pamin stop` first if you want the in-process path against a
workspace a server is holding.

Every command goes through the server except two. `serve` is the server, and
`stop` is what shuts it down.

The socket is a file, so it inherits the workspace's permissions and cannot be
reached from another machine. There is no authentication, for the same reason:
anyone who can open the socket can already read the workspace.

## `pamin stop`

```console
$ pamin stop
Stopped the local database server
```

Stops the local PostgreSQL, and the resident server if one is up. It is not run
automatically, because the common case is an agent issuing many commands in a
row and paying startup once.

## Notes for agents

- Every command accepts `--json`, and stdout carries only that JSON. Logging
  goes to stderr.
- Failures exit non-zero with the reason on stderr.
- The first command against a workspace is slow and the rest are not: it starts
  a server that holds the database, the index and the model. Nothing needs to
  start or stop it.
- `search` gives ranked context; `neighbors` gives structure; `read` gives a
  specific version; `grep` gives the verbatim evidence including what the filter
  held. Reach for the last three when the ranking is what you doubt.
- Results are stable for stable inputs. Ties break on identifier, so an
  assembled context can be reused rather than rebuilt.
- Evidence is never translated and never rewritten. Anything a memory lost in
  summarizing is still in the source it came from, and `pamin grep` reaches it.
