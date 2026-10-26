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
| `--vector-index <name>` | `PAMIN_VECTOR_INDEX` | `memory` | Vector index a project is built with: `disk` or `memory` |
| `--json` | | off | Emit JSON instead of text, on one line |
| `--pretty` | | off | Indent that JSON. Requires `--json` |

The JSON is compact because the usual caller pays for every token of it, and
indenting a ten-hit search costs about a thousand of them. `--pretty` is for
the person who has piped it to a terminal.

`PAMIN_LOG` sets the log filter (`PAMIN_LOG=debug`). Logs go to stderr, so they
never contaminate the JSON on stdout.

### Server settings

These govern the server process -- which device, how many threads, how long a
model stays loaded -- so they are fixed when the server starts. Every command
takes them, because any command may be the one that starts the server, and
each can also be set in the `[serve]` table of `$PAMIN_HOME/config.toml`, by
the key in the second column. A flag wins over the file, and the file over the
default.

| Flag | `config.toml` key | Default | Meaning |
| --- | --- | --- | --- |
| `--device <auto\|cpu>` | `device` | `auto` | `auto` measures every viable accelerator and CPU plan and picks the fastest; `cpu` forces optimized CPU. Also read from `PAMIN_DEVICE` |
| `--inference-threads <n>` | `inference_threads` | one per core | Threads one forward pass may use, or on the CPU, for the batches of a rerank and stored texts of the embedding model (BGE-M3), how many single-threaded passes run at once |
| `--ep-library <path>` | `ep_libraries` | none | A path to an installed, ABI-compatible NPU plugin EP library; repeat the flag, or list them in the file. Only discovered NPU devices participate in automatic plugin routing. Unavailable/incompatible libraries retain ordinary GPU/CPU fallback |
| `--prepared <true\|false>` | `prepared` | `true` | `false` loads a model from its download rather than from a mapped copy, fetching the download again if it was removed |
| `--model-idle <seconds>` | `model_idle` | `1800` | Seconds a resident server holds a model nothing is asking for |
| `--open-indexes <n>` | `open_indexes` | `16` | How many project indexes a server keeps open at once |
| `--jit <true\|false>` | `jit` | `false` | Let PostgreSQL compile query expressions with LLVM |
| `--postgres-dir <path>` | `postgres_dir` | unset | Use a PostgreSQL already on this machine instead of installing one |

```toml
# $PAMIN_HOME/config.toml
[serve]
device = "cpu"
model_idle = 600
open_indexes = 4
ep_libraries = ["/opt/vendor/lib/libnpu_ep.so"]
```

A relative path is made absolute when settings are resolved: in a flag,
against the directory the command ran in; in the file, against the workspace
(`$PAMIN_HOME`).

A key the file does not know, or a value a flag would refuse, stops every
command with the file's path and the offending line, rather than leaving the
setting at its default. `pamin config` prints what a command would run with and
where each value came from -- `flag`, `env` (`PAMIN_DEVICE`), `config` or
`default` -- without starting anything; `--json` gives the same as an object.

**Settings are fixed when the server starts.** A command sends the settings it
resolved -- its flags over the file over the defaults -- to the running server,
and a server running with different ones refuses the command, naming each
setting that differs and what it is running with, rather than answering with
settings nobody asked for or replacing a server another agent may be using. To
change them, stop the server and run the command again:

```console
$ pamin stop
$ pamin search "deployment" --device cpu    # starts a server with --device cpu
```

A command forwards the flags it was given to the server it starts. A setting a
command leaves to the default does not conflict with one a server was started
with as a flag, so `pamin serve --model-idle 60` answers commands that name no
idle window; one it took from `config.toml` does, so edit the file and stop
the server together.

`--postgres-dir` points at an installation prefix holding `bin/initdb` and
`bin/pg_ctl` -- `/usr/lib/postgresql/17` on Debian and Ubuntu, `$(brew --prefix
postgresql@17)` on macOS. A prefix without them is refused, naming the flag or
the file and what is missing, rather than replaced by a downloaded copy.
Unset, a workspace installs its own copy, which is
the default because it is what makes `pamin` work with nothing else installed;
set, that copy is not downloaded and not stored, which is several hundred
megabytes a workspace does not spend. Two things to know before setting it.
The version requirement is not checked -- you are vouching for the server, and
the migrations expect PostgreSQL 17. And a figure measured against a server
built by somebody else is a figure for that server: fine for checking
behaviour, not interchangeable with the numbers in
[measured.md](measured.md).

`--jit true` turns on PostgreSQL's LLVM compilation of query expressions.
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

`--model-idle` is how long `pamin serve` keeps a model and the indexes
pinning it after nothing has asked for them. It is a memory setting and the
trade is measured on both sides: on a 13,014-document project a server holding
the embedder and the since-removed `fast` reranker was 2,263 MB resident and 88-101 MB once it
has given them back, so half an hour of quiet returns about 2.2 GB -- and the
first search afterwards takes 4,528 ms instead of 116, which is 88% of what a
server starting from nothing costs. Thirty minutes is where that stops being a
close call: an agent working in bursts does not wait half an hour between
searches, and a server left running overnight pays it once. Lower it on a
machine where memory is scarcer than four seconds; raise it if a search every
few minutes is worth 2.2 GB to you.

`--inference-threads` is how many threads one forward pass may use.
Unset, the inference library uses one per core, which is right for a single
query and wrong for a server: measured on four cores with nothing shared,
throughput *falls* as callers are added, 182 embeddings a second at one worker
to 54 at eight, because the second caller finds the first caller's threads
rather than an idle core. Splitting the cores between callers instead of
between the layers of one pass is the other way to divide them, and which wins
is a property of the machine rather than of this program -- so it is a setting
whose default is what the library already did. Stored texts of BGE-M3 divide
them the other way on the CPU: each pass has one thread, and this many passes
run at once, as do the passes of one rerank, one pair each when this is more
than one (at one, a pass takes up to four pairs). A BGE-M3 query, and a rerank
of a single pass, still get one pass of this many threads. Those passes share
one single-threaded session over the model's mapped copy (below), loaded
with the reranker, and by the embedder the first time more than one text is
stored, and kept from then on; at one thread, or when the model loads from its download
rather than a copy, every pass runs one after another on the model's own
session instead.

On the CPU, the first load of a model writes a copy of it into
`models/prepared/`, and every load after that reads the copy. The copy is the
runtime's own optimized form of the graph with its weights in a separate data
file, which the runtime maps from disk instead of copying onto the heap: the
`accurate` reranker holds 271 MB of live memory rather than 822, the `accuracy`
embedder 272 rather than 824, with scores and vectors bit-identical (see
[measured.md](measured.md)). Those are a model as loaded: the passes it runs
side by side (above), and the session they share, hold memory of their own on
top, and for the `accurate`
reranker that is the larger part. Measured through `pamin search` on four
cores over 3,230 memories, when each pass still had a session of its own
rather than one shared session, running a rerank's batches side by side took the
search server from 2,117 MB peak resident to 4,677, and from 855 MB of
anonymous memory at the end of the run to 1,876 (proportional set size
1,726 MB to 2,808). The resident figure counted the shared data file once for
each session; the anonymous figure was what the sessions held of their own.
That has not been re-measured with one shared session.
Most of what was left was the tokenizer's
vocabulary, which the embedder and every reranker share. It is now held in a
compact layout: with both loaded, the two hold 47 MiB of anonymous memory after
use, where the previous layout held 314 MiB in the same measurement (and
582 MiB when each model built its own copy). The per-model figures above were taken
with the previous layout, and with each model's one session: a model also
loads one shared single-threaded session for the passes it runs side by side
-- the reranker when it loads, the embedder the first time it stores more than
one text -- which those figures leave out. [measured.md](measured.md) has what a
whole server holds with it. What it costs is disk. Each copy is larger than the
model it came from, because the weights are also stored in the layout the CPU's
kernels use -- a data file of 874 MB for the 570 MB reranker, and about as
much for the embedder -- and writing
it makes that first load slower, 5.1 s for `accurate`.

Once a copy has loaded, the download it was written from is removed, since
nothing reads it again: each model is on disk once, as its copy. At the
defaults that took the model directory from 2,923 MB to 1,783 MB (see
[measured.md](measured.md)). What that gives up is the source for the *next*
copy. A copy belongs to the runtime version and the CPU that wrote it, so an
upgrade, or a model directory moved to a different CPU, needs a new one, and
that load fetches the model again -- 570 MB for `accurate` or the `accuracy`
embedder -- writes the copy, and removes the download again. **That load needs
the network.** Offline, it fails with an error that names the model and says
why, instead of searching; the fix is to be online for that one load, or to
copy the model directory from a machine that has the file.

Copies are removed by one rule, applied on every load. The copy this version
loads for a model it has loaded before is never removed, however long the model
goes unused: its download has usually been removed, so removing it would make
the next load of that model need the network. Every other copy -- the one an
upgrade or a move to another CPU leaves behind, or one of a model file the hub
has since replaced -- is removed once no running process has it loaded and none
has loaded it for two weeks. So a directory shared by two versions keeps both
copies while both are in use, and a version that goes two weeks without running
while the other does writes its copies again when it next runs -- fetching the
model first, if its download was removed. A model nothing loads any more --
one of the retired profiles or the `fast` reranker -- keeps its copy until it is
deleted; see [Upgrading from profiles and tiers](#upgrading-from-profiles-and-tiers). Delete
`models/prepared/` only after every process using that model directory has
stopped; the next load downloads the model
again. The download is kept when `HF_HOME` is set -- that cache is shared with
other tools -- or when the model directory, or one model's directory inside it,
is a link to somewhere else. A kept download is hashed once, when its record is
first written, and not read again by later loads. `--prepared false` loads from the download,
fetching it again if it was removed, for a measurement that needs the unmapped
load or a disk that cannot spare the copy's extra size. When a copy cannot be
written -- a full or read-only disk -- the model loads from the download and
keeps it, and the log says why.

The tokenizer vocabulary's lookup trie, about 4 MB, is kept in
`models/tokenizers/` the first time it is built, under a name of its
vocabulary's digest, and read from there by later loads instead of being built
again; deleting the directory only makes the next load build it again.

Where the runtime left a model's attention as separate operators -- the
`accurate` reranker's int8 export -- the copy also gets a second graph,
`attention.onnx`, with each layer's attention as one fused operator; it is
kept only if it scores a probe bit-for-bit as the first does, and otherwise
`attention.unfused` says why. `--fused-attention false`, a hidden flag with no
`config.toml` key, loads the unfused graph, for measuring one against the
other.

Where a graph gathers its word embeddings from one quantized table -- both
int8 exports the CPU loads -- the copy also gets `model.split1.onnx` (or
`attention.split1.onnx`), which takes those rows as an input instead, and
`model.split1.table`, which says where the table is in the data file: Påmin
reads the rows a batch needs from the file, so the runtime never maps the
whole table. It is kept only if it scores the same probe bit-for-bit, and
otherwise `model.split1.unsplit` says why. `--split-embeddings false`, also
hidden and with no `config.toml` key, loads the graph it was split from.

CoreML packages written by older versions under
`models/typed-reranker-v3/<SHA-256>/coreml-all-v1`
are not reused by the current `coreml-all-v2` policy. Old processes can still
use a compiled package after its build lock is released, so the application
does not delete v1 automatically. During an **offline** maintenance window,
stop every Påmin server and CLI process using this model directory, then list
the exact v1 packages and their logical bytes:

```sh
python3 maintenance/retire_coreml_v1.py ~/.pamin/models
```

After verifying the list and that those processes remain stopped, remove only
that obsolete namespace:

```sh
python3 maintenance/retire_coreml_v1.py ~/.pamin/models --apply --all-processes-stopped
```

Use your configured `PAMIN_HOME/models` in place of `~/.pamin/models` when it
differs. The script refuses an arbitrary parent path, skips linked prepared
entries and preserves v2 packages and mapped CPU models. Logical bytes are
not a measure of unique APFS allocation; the actual free-space change may
differ.

The `accurate` reranker calibrates viable shared accelerator plans against optimized CPU:
Eligible registered NPU devices are considered first, followed by CUDA on x86-64 Linux, Core ML `ALL` on Apple silicon, or DirectML on Windows, with optimized CPU/SIMD subgraph fallback. The qualified fastest complete-call plan wins; provider order alone does not determine it.
Provider selection is logged when a
model loads; Core ML may itself use CPU, GPU or ANE, and its internal placement
is not established by the provider label. The different weight exports can
produce different scores. `--device cpu` forces CPU for comparable tests
or when another job needs the accelerator. Embedding also tries shared
accelerator plans using the same export, tokenizer and indexed model
identity. Compatibility checks compare vectors with the CPU export; a rejected
plan falls back. The finite startup fixtures are a smoke check, not full-corpus
retrieval certification. With automatic dispatch, a bounded complete model-call
query fixture compares viable plans against interleaved optimized-CPU controls.
Maximum batch/length conformance is checked separately from singleton query
timing, so bulk ingest throughput does not decide the search plan. Its
validated winner is reused for idle reloads. Identified CUDA and Apple CoreML choices can also persist across fresh CLI processes for up to one day; numerical rejects are retried after five minutes. Keys include model snapshot, actual application source/compiler configuration and mapped running-image identity, device inventory and runtime settings. Opaque DirectML/NPU choices remain process-local until stable hardware/driver identity is available. Cache-hit sessions are output-checked, not retimed; starting a new process alone does not establish a cold calibration arm. This estimates the fastest
plan for that fixture; it is not universal per-query autotuning.
Broader retrieval and resource validation remain under review. Changing a model/export or encoding contract needs paired quality
and reindex validation.

On Linux the CUDA path has two requirements the program cannot meet for you.
The machine needs the NVIDIA driver, CUDA 13 and cuDNN 9. And the runtime's
provider libraries -- `libonnxruntime_providers_shared.so` and
`libonnxruntime_providers_cuda.so`, 79 MB, built into `target/release` beside
the binary -- have to sit in the same directory as `pamin-server`, which
`cargo install` does not arrange: copy them next to the installed binary. Missing
either, the reranker runs on the CPU exactly as before. Where there is no sign
of an NVIDIA driver or GPU -- no `/dev/nvidiactl`, no `/dev/dxg` (WSL 2), no
`/proc/driver/nvidia`, and no NVIDIA PCI device -- CUDA is not tried at all, so a
CPU-only machine does not pay a failed provider registration on each server
start.

On Windows the same holds for one file, `DirectML.dll` (18.5 MB), built beside
`pamin-server.exe`. Without it the program still starts -- Windows 10 and later carry
their own copy in System32 -- but that copy can be older than the runtime
needs, in which case the reranker quietly stays on the CPU. On Apple silicon
nothing needs copying: Core ML is linked into the binary from the system.

Apple CoreML has been measured on the full XQuAD-R product path: it wins for
the reranker with static buckets on this machine. (It lost to optimized CPU for
the since-removed `fast` reranker: [conditions and rows](../benchmarks/results/inference/fast-apple-backend-2026-09-30.md).)
CUDA and DirectML still require their own device-specific timing and precision
checks. The `every_device_orders_like_the_cpu` test in
`crates/pamin-index/tests/reranking.rs` checks clearly separated pairs; it is
not a replacement for a corpus-level accuracy result.

A few measured values have no flag and no `config.toml` key: the reranker's
batch and the longest candidate it reads, and how many documents may sit
outside the vector graph before it is built. They are fixed where they were
measured; the evaluation harnesses set them as `PAMIN_EVAL_*` variables, which
`pamin` itself never reads. `pamin search` has two hidden flags of the same
kind; see [`pamin search`](#pamin-search). The only variables `pamin` reads are
`PAMIN_HOME`, `PAMIN_PROJECT`, `PAMIN_VECTOR_INDEX`, `PAMIN_DEVICE` and
`PAMIN_LOG`; the ones that used to configure anything else are refused, as
[Upgrading from settings variables](#upgrading-from-settings-variables)
lists, and a unit test checks this page against what the binary reads.

Every project embeds with one model, and every search reranks with one
reranker. There is nothing to choose between:

| Role | Model | Width | Resident | Per query |
| --- | --- | --- | --- | --- |
| Embedding | BGE-M3, int8 weights (`gpahal/bge-m3-onnx-int8`) | 1024 | 272 MiB live | 37.7 ms idle, 46–47 ms under load |
| Reranking | bge-reranker-v2-m3, int8 weights (`pamin-labs/bge-reranker-v2-m3-sq-int8-onnx`) | | 271 MiB live | see [`pamin search`](#pamin-search) |

Resident is live anonymous memory with the model loaded from its mapped copy,
taken before the vocabulary was shared and compacted and without the session
side-by-side passes add; each download is about 570 MB. The query figures are
a median on four cores, through `Embedder` with nothing else running and in a
server at a load average of 5.6 to 7.2. [measured.md](measured.md) and
[ADR 0001](adr/0001-tech-selection.md) have the runs.

BGE-M3 was the `accuracy` profile while there were others. It was chosen
because quantized weights make it a small download and because the gap it
closes is the one Påmin Memory is about: on the evaluation corpus it roughly
doubled cross-lingual retrieval against multilingual-e5-base (`balanced`),
matched it on same-language queries, and cost nine milliseconds more. The index
records the model it was built with and refuses to open under another, naming
`reindex` in the error rather than silently mixing two vector spaces.

### Upgrading from settings variables

Settings other than the five above were environment variables. Each is now a
flag (and, for a server setting, a `config.toml` key), or gone, and setting one
-- to any value -- fails with a message naming what replaced it, rather than
being ignored by a server that would then run with settings its operator did
not intend. Unset it, and use the replacement:

| Variable | Now |
| --- | --- |
| `PAMIN_INFERENCE_THREADS` | `--inference-threads`, `inference_threads` |
| `PAMIN_EP_LIBRARIES` | `--ep-library`, once per library; `ep_libraries` |
| `PAMIN_PREPARED` | `--prepared <true\|false>`, `prepared` |
| `PAMIN_FUSED_ATTENTION` | `--fused-attention <true\|false>` (hidden) |
| `PAMIN_MODEL_IDLE` | `--model-idle`, `model_idle` |
| `PAMIN_OPEN_INDEXES` | `--open-indexes`, `open_indexes` |
| `PAMIN_JIT` | `--jit <true\|false>`, `jit` |
| `PAMIN_POSTGRES_DIR` | `--postgres-dir`, `postgres_dir` |
| `PAMIN_CHANNEL_DEPTH`, `PAMIN_GRAPH_DEPTH` | `pamin search --channel-depth`, `--graph-depth` |
| `PAMIN_RERANK_DEPTH`, `PAMIN_SEARCH_EFFORT` | `pamin search --rerank-depth`, `--search-effort` (hidden) |
| `PAMIN_RERANK_BATCH`, `PAMIN_RERANK_BATCH_TOKENS`, `PAMIN_RERANK_MAX_TOKENS`, `PAMIN_UNINDEXED_BUDGET`, `PAMIN_CATCH_UP_BATCH` | removed: fixed at their measured values |

`PAMIN_DEVICE` is still read, as the fallback for `--device`.

### Upgrading from profiles and tiers

Earlier versions had embedding profiles (`--profile`, `PAMIN_PROFILE`:
`speed`, `balanced`, `accuracy`, `dual_accuracy`) and reranking tiers
(`--rerank`, `PAMIN_RERANK`: `off`, `fast`, `accurate`). What shipped as
`accuracy` and `accurate` is what every project now uses, so a workspace that
kept the defaults needs nothing: its indexes open as they are, and the
embedder's measured plans are reused. The reranker's are measured again once,
because a CPU pass now takes one pair and its plan key changed with that.

- `--profile` and `--rerank` are unknown arguments. Setting `PAMIN_PROFILE` or
  `PAMIN_RERANK`, to any value, fails with a message naming the setting, as
  `PAMIN_NO_SERVER` does; unset it.
- A project indexed under `speed`, `balanced` or `dual_accuracy` no longer
  opens. The error names the profile it was built with and `pamin reindex`,
  which rebuilds it with BGE-M3 from PostgreSQL. Until then the server does not
  catch up that project's owed work, and says so in its log.
- The retired models' downloads are not deleted, because the model directory
  may be shared. Once no older version of `pamin` uses this model directory,
  you can remove them by hand from `models/` (or `HF_HOME`, if set), as
  whichever of these exist: `models--intfloat--multilingual-e5-small` and
  `models--intfloat--multilingual-e5-base` (`speed`, `balanced`),
  `models--perplexity-ai--pplx-embed-v1-0.6b` and `pplx-int8-v1`
  (`dual_accuracy`), and `models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1`
  (the `fast` reranker). Their prepared copies are not named by model; with
  every process using the directory stopped, deleting `models/prepared/` as a
  whole removes them, and the next load of the two current models fetches and
  prepares them again.
- The reranker now loads from `pamin-labs/bge-reranker-v2-m3-sq-int8-onnx`,
  fetched on the first search after upgrading, and its plans are measured
  again once. On the CPU that is the int8 file, about 570 MB; a host that
  reranks on an accelerator also fetches the fp16 file, about 1.1 GB, which
  is upstream's unchanged but now comes from the new repository. The previous download,
  `models--onnx-community--bge-reranker-v2-m3-ONNX`, is not deleted, for the
  same reason; remove it by hand once no older `pamin` uses the directory. Its
  prepared copy is not removed either, because its record still names it as
  that model's current copy; deleting `models/prepared/` as a whole, as above,
  removes it.

`--vector-index` chooses how a project's vectors are indexed. Both choices
store every vector in half precision and search the same way -- the index
proposes twice the candidates it is asked for, and those are ranked again by an
exact cosine in full precision -- so both return the same neighbours as the
full-precision graph did, within 0.002 of its recall. They differ in where the
index lives and what that costs, measured over MIRACL's 131,924 passages:

| | resident | disk | a vector query | a whole search | a full build | peak while building |
| --- | --- | --- | --- | --- | --- | --- |
| `disk` | 34 MB | 618 MB | 69-79 ms | 2,339 ms | 23 min | +1,353 MB |
| `memory` (default) | 320 MB | 328 MB | 6-10 ms | 2,236 ms | 1.4 min | +721 MB |

A whole search is the median `accurate` search over XQuAD-R's 13,014
documents rather than MIRACL, on a shared four-core machine; the two ranked
every question identically.

`disk` keeps its graph and vectors on disk and reads them per query, which is
why it holds almost nothing resident; it is for a project whose memory is
scarce, and it pays for that with minutes on each `optimize` upkeep runs after
writes (about a second under `memory`). `memory`, the default, holds them
resident, and is the faster and smaller choice everywhere else. A search spends most of a second or more in
the reranker, so the query column is a small share of a `pamin search`; the
build column is not small, and `disk` also pays minutes for an `optimize` after
a few writes where `memory` pays about a second (see
[measured.md](measured.md)). Choose `memory` for a project that is written to
all the time, and wherever resident memory is not what is scarce.

The index records which one it was built with, and changing the setting
changes nothing about an index that exists: opening it as the other one is
refused, naming `reindex`. `pamin reindex` rebuilds it
the new way and reuses every vector it holds. An index built before this
choice existed stored full-precision vectors under an in-memory graph; it is
refused the same way and moved the same way.

Projects are namespaces, not tags. Each has its own index directory, so nothing
crosses between them and a rebuild of one leaves the others alone.

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
`queued` means some of it is still owed, and the server runs it once it has no
request to answer — the memory is recorded either way. See
[`pamin cascade`](#pamin-cascade).

`--defer` records the memory and leaves the index to catch up later, so the
command returns without embedding anything:

```console
$ pamin write --defer --topic release_notes "cut 1.4.0 from main"
Wrote release_notes v1
```

The memory is committed exactly as it would be otherwise — `pamin read` and
`pamin grep` see it immediately — and only `search` waits for the queue. The
server works through the queue on its own, between requests, so the memory
becomes searchable with nothing else run once the server's five-second upkeep
has come round once or twice: 5.8 to 11.6 seconds in five measured runs. Run
`pamin cascade drain` to make the index catch up at once instead, for instance
at the end of a loop of deferred writes; to record a file of memories, use
[`pamin import`](#pamin-import).

`cascade_lagging` is set once the queue passes ten thousand owed jobs, and it
reports what the queue owed when the write looked at it rather than what is left
afterwards. Without it a cascade keeping up and one falling behind look
identical from outside, apart from searches missing the newest memories.

Ten thousand is also where `--defer` stops deferring: a write past it drains
before returning, so an import that ignores the signal still cannot run the
queue away. That costs the importer the work it created rather than pausing it.
A writer that waited would hand the same work to the server, which runs it only
while no request is being answered and in smaller rounds, so waiting would slow
the import by at least what paying costs and bound nothing. A new memory
queues three jobs, so an import pays for a batch about every three thousand of
them and never carries more than ten thousand.

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

## `pamin ingest`

Records Markdown and text files as memories, one per section. Here, in a git
repository `handbook` whose `docs/ops.md` holds a `# Deploy` section with a
`## Rollback` under it:

```console
$ pamin ingest docs
Ingested 1 files as 2 sections: 2 written, 0 unchanged, 0 held in evidence only
```

A path is a file or a directory. A directory is read recursively for `.md`,
`.markdown` and `.txt` files, skipping hidden entries — `.git` among them —
symbolic links, and, inside a git repository, whatever git ignores: its
`.gitignore` files, including any above the directory named,
`.git/info/exclude` and the global excludes of the user the server runs as. A file named on the command line is read whatever it is called
and wherever a link it is leads, and is cut as Markdown whatever it holds, so
`# ` at the start of a line of code is a heading to it. The paths are made
absolute by the client before the request leaves, because the server reading
them may have been started from another directory.

**A file is cut where its headings are.** A heading outside a fenced block —
`#` to `######`, or a paragraph underlined with `===` or `---` — opens a
section, and the section's topic is the file's path and the headings above it,
each cut to 80 characters, joined by ` › `:

```console
$ pamin search "how do we roll back a release" --limit 2    # second hit trimmed
0.1136  handbook/docs/ops.md › Deploy › Rollback v1  ## Rollback

A rollback reverts the deployment pipeline to the previous release tag. The
oncall engineer starts it from the release dashboard, and the previous build is
serving again within ten minutes because its image is still cached on every
node in the cluster.
        lexical_segmented#1 lexical_ngram#1 vector#1 reranked
0.1042  handbook/docs/ops.md › Deploy v1  # Deploy
…
```

The path runs from the root of the git repository the file is in — a
submodule's being its superproject's — the root's own name first, so ingesting a tree and ingesting one file of it name that file
alike, and two repositories' `README.md` are two files. Outside a repository it
runs from the directory named, or for a file named on its own, from the
directory it is in. Two files that would still go by one name are refused
rather than recorded under one topic.

The topic is also what each section is embedded and reranked under, as
`{topic}: {content}`, so a section about "Rollback" in `docs/ops.md` is read
with both. The heading line stays in the section's content too, which is what
the two lexical channels index.

**A section is one memory however long it is.** Its size is left to the index,
which cuts every memory into spans and ranks a topic by its best span, so a
long section is found by any part of it and a hit shows the part that matched.
A section of fewer than 48 tokens joins a neighbour — a heading with nothing
under it the section after it, a short opening at the top of the file the
section after it, anything else the one before. Two sections under the same
headings are numbered `#2`, `#3`, and keep their numbers from one ingest to the
next: each is matched to the section it was by its text — the same text first,
then at least 80% of its three-word runs in common — and only the rest take
free numbers, in file order. So a section inserted above its twin is `#2`,
rather than taking the twin's name and making the twin's text a new version of
somebody else's section. YAML front matter is left out when its first line is
a `key:` and it holds no blank line.

```console
$ sed -i 's/ten minutes/fifteen minutes/' docs/ops.md
$ pamin ingest docs
Ingested 1 files as 2 sections: 1 written, 1 unchanged, 0 held in evidence only
$ sed -i 's/## Rollback/## Rolling back/' docs/ops.md
$ pamin ingest docs
Ingested 1 files as 2 sections: 1 written, 1 unchanged, 0 held in evidence only
1 sections were found under a new name and renamed, keeping their history:
  handbook/docs/ops.md › Deploy › Rollback -> handbook/docs/ops.md › Deploy › Rolling back
```

**Ingesting again records only what changed.** A section that says what its
topic already does is left as it is, with nothing written; a changed section is
a new version of its topic, as any write is, and only it is embedded again. An
edit to a section's text changes that section alone.

**A section found under a new name is renamed, not recorded afresh.** Renaming
a heading changes the name of every section under it, and moving a file or a
folder within a tree changes every name in it. A section an ingest finds gone
from one name is matched to one appearing under a new name — the same text
first, then at least 80% of its three-word runs in common, one to one — and its
topic takes the new name, keeping its versions, its evidence and the edges to
it; its new text, if it changed, is a new version. The old name stays as an
alias: `pamin read`, `pamin link` and `pamin neighbors` still find the topic by
it, and `read` answers with the name it goes by now. A renamed section is
owed to the index again, because its name is read into the header of every
span, and a span is re-embedded when its header changes. Only a section the
ingest could otherwise have taken off the retrieval surface is renamed, and
only to a name no topic already holds: a file moved out of the tree the command
walked is still `removed`, and one moved onto an old section's name is recorded
as new. Listed under `renamed`.

On this repository's own 105 Markdown and text files, an
unchanged re-ingest wrote nothing; the time it took was measured on an earlier
cut that also split long sections into parts, and is not re-measured for this
one ([measured](../benchmarks/results/ingest/ingest-cost-2026-10-23.md)).

**A section that is no longer there is taken off the retrieval surface**, and
listed under `removed`: its heading removed, or renamed with its text rewritten past matching, or its file deleted
from a directory the command walked, moved out of it, or now ignored by git.
Every version of its topic is soft deleted, so search no longer returns it, and
the edges its content derived are closed. Nothing is lost: the ledger keeps
every version and its evidence, `pamin grep` still finds the text, and
ingesting the section again brings it back as a new version — though until
then `pamin read` reports the topic has no live version, and `pamin topics`
still lists its name.

Only what the command could see is retired. A file it found and could not read
is not where a section went missing, and its sections stay. Under a directory
whose walk met an error — a subdirectory it could not open — a file it did not
find may only have been out of reach, so its sections are listed under
`missing` and stay; so are those of a file the walk skipped that is still on
disk, hidden or ignored by git and named on its own some earlier time. A section is retired only from the tree it was last
ingested from: each file's sections record that directory, so another tree
whose files are named alike, another clone or another `notes`, is not taken
for this one. And a section another ingest wrote again after this one looked —
found back in its file with new text — is left as it now is; one found back
unchanged is retired, and the next ingest brings it back.

`--keep-missing` only lists such sections, under `missing`, and leaves them
searchable. So does any section somebody also wrote by hand with `pamin write`,
whatever the flag: it is theirs as well as the file's.

**A section its file cut down to next to nothing is taken off too.** Text
below the filter's eight characters — a `TBD` where a paragraph was — is held,
counted under `held`, and would leave the section answering with the text the
file no longer has. So it is retired as a section the file lost is, and listed
under `removed`, `--keep-missing` or not: the file still has the section and
says next to nothing in it. Ingesting it again while it says the same writes
nothing, and once the file says something again the section is back. One
somebody also wrote by hand stays, held on every ingest and its evidence
recorded again each time.

Nothing outside what the command named is looked at. A repository whose
directory was renamed or checked out under another name is recorded afresh
under its new names, and its old sections stay searchable and are listed
nowhere. And two directories of the same name ingested by separate commands
name their files alike, so where both have a file of the same name, each
records that file's sections as new versions of the other's wherever the two
differ, and the sections remember whichever tree was ingested last.

Ingested topics are kept out of the name index the graph channel matches
memories against. A path and its headings are a name no memory spells out in
full, and a wide name makes every later write compare longer runs of its text.
A memory naming an ingested section therefore derives no edge to it, while a
section naming a topic written by hand does. A topic written by hand keeps its
name if it is later ingested, and one ingested first is named the first time
somebody writes it by hand.

`--json` carries every count, the `removed`, `missing`, `renamed` and `skipped` lists, and
the same `cascade` and `cascade_lagging` as `import`.

## `pamin read`

Reads a topic at a version. `--version-offset` counts back from the current one.

```console
$ pamin read deployment_pipeline
deployment_pipeline v2 (current, 0 of 2 versions, recorded 2026-10-17T02:23:17.362349Z)

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
  "recorded_at": "2026-10-17T02:22:59.777239Z",
  "observed_at": "2026-10-17T02:22:59.772516Z",
  "source_span": "01a147ab-6ec1-78e2-8ccf-08a6924aa0f0"
}
```

`valid_from` and `valid_to` are absent here rather than null: a memory written
without an interval carries none, and the fields appear only when it has one.
`source_span` is the evidence this version was promoted from, and `pamin grep`
is what reads it.

A long memory is indexed as spans (see [Long memories](#long-memories-spans)),
and `--span` reads one of them, numbered from 0 the way `pamin search` reports
them; `--around N` adds N spans either side. The `content` is then that run of
the memory and a `span` field says where it lies:

```console
$ pamin read incident_2026_03 --span 4 --around 1 --json
{
  "topic": "incident_2026_03",
  "version": 1,
  "content": "...",
  ...
  "span": { "ordinal": 4, "of": 12, "header": "incident_2026_03 > Timeline", "start": 2210, "end": 3391 }
}
```

`ordinal` is the span asked for, `of` how many the memory is cut into,
`header` the memory's name and the heading the run sits under, and `start` and
`end` the run's bytes in the memory's content. A historical version is cut the
same way, by the project's index.

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
(default 2). Two hidden flags go with them: `--rerank-depth`, how many fused
candidates the reranker reads (default 30), and `--search-effort`, how wide the
vector graph is searched (default 700). All four are per search: a server
answers each search at the depths that search asks for. Every one but
`--graph-depth`, where 0 turns the walk off, is refused at 0.

These exist for the evaluation harness, which is what settles them. They are
not a tuning surface for ordinary use: raising the depth costs latency for
recall you cannot measure from outside, and an agent that wants control over
retrieval should reach for `grep`, `read`, and `neighbors` rather than adjust
ranking internals it has no way to evaluate.

Every search reranks: bge-reranker-v2-m3, with int8 weights, reads the query
and each candidate together and reorders what fusion found. The weights are
PaminLabs' SmoothQuant re-export of `onnx-community`'s int8 file, which
moves activation outliers into the weights before they are quantized; see
[measured.md](measured.md) for what it changed. There is no
setting for it. Earlier versions chose a tier with `--rerank` and
`PAMIN_RERANK` -- `off`, `fast` (a 119 MB MiniLM cross-encoder) or `accurate`,
which is the one kept; see
[Upgrading from profiles and tiers](#upgrading-from-profiles-and-tiers).

The comparison that chose it is below. It is historical, measured before the
corrected XQuAD-R same-language key and the current full-head pass, and its
same-language and latency figures do not describe the current path. The
current comparison is tracked in [#121](https://github.com/pamin-labs/PaminMemory/pull/121).

| | what it loads | a search costs | cross-lingual nDCG@10 | same-language |
|---|---|---|---|---|
| `off` | nothing | 99 ms | 0.6114 | 0.7829 |
| `fast` | 119 MB | 359 ms | **+0.0397** | **−0.0060** |
| `accurate` | 571 MB | 1522 ms | **+0.0482** | +0.0006 |

All three rows are one run over the same 1,190 queries, taken when a tier
reranked twenty candidates; the reranker now sees thirty, which turned into
+0.0063 more cross-lingual (`p = 0.0001`) for half again as many model pairs.
At thirty, `pamin search` against a resident server over 3,230 memories, on a
four-core 2.8 GHz Xeon with no other measurement running, had a median of
3,207 ms (p95 5,128) when a pass took up to four pairs, and 2,675 ms (p95
4,359) with the one pair a pass that now ships on a multi-core CPU -- 0.834 of
it ([measured.md](measured.md) has the conditions). Read 1522 ms as the cost
at twenty candidates rather than as today's. Paired bootstrap against `off`,
10,000 resamples: cross-lingual `p = 0.0001` for both tiers; same-language
`p = 0.0008` for `fast` and not significant for `accurate`. `recall@50` is
0.8962 and 0.9571 in **every** arm, to four decimals — a reranker reorders a
shortlist and never changes what is in it. Query by query against `fast`,
`accurate` was ahead by 0.0086 cross-lingual (`p = 0.0015`) and 0.0066
same-language (`p = 0.0001`) on XQuAD-R, and by 0.0411 on MIRACL Swahili (83
queries better, 12 worse, `p = 0.0001`).

Measured on XQuAD-R's 13,014 sentences in eleven languages, through
`Engine::search_reranked` — the call this command makes, one layer below the
process it runs in. [measured.md](measured.md) reports the same corpus and
tiers as whole CLI invocations, 77/251/1241 ms, and the difference between the
two sets of figures is the invocation; neither is wrong and they are not
interchangeable.

What the `off` row is is worth knowing before reading the reranker as
overhead. Most of it is not retrieval either: the four channels, fusion and
reading the states back are about 16 ms of it, and the rest is the forward
pass that turns your query into a vector — 68 ms for the embedding model, on
four cores, for a query the server has not been asked before. A resident
server remembers a query's vector, so asking the same thing twice costs the
16 ms alone. [ADR 0001](adr/0001-tech-selection.md) divides all four stages.

The same-language column above comes from parallel text, where it is *the same
1,190 queries* as the cross-lingual column scored against a different answer
key — so every query in it still has correct answers in ten other languages
sitting in the index, which a real single-language workspace does not. On the
one genuinely single-language corpus measured, the pass **gains** 0.0496 nDCG@10
over fusion alone.

[ADR 0001](adr/0001-tech-selection.md) keeps the rows of the tiers removed
earlier, `balanced` and `noncommercial`. The reranker is permissively licensed;
[NOTICE](../NOTICE) lists what it downloads and the chain behind it.

The model is fetched the first time a search asks for one, into the same cache
as the embedding model.

It costs memory while it is loaded, and more than its download suggests. On a
13,014-document project a server that had not reranked was 1,625 MB resident,
and one search with the since-removed 130 MB `fast` model took it to 2,007 or
2,271 MB -- the difference being the inference runtime's arenas rather than
the weights. `accurate`'s model is 571 MB against 119. Its own resident cost
has since been taken: 271 MiB of live memory loaded from its mapped copy and
822 MiB from the download, and on MIRACL, loading it from the download added
911 MB of anonymous memory to a server (see [measured.md](measured.md)). `pamin serve` gives it back after
`--model-idle` seconds with nothing asking for it; the arena growth above that
stays.

A reranker reads the query and a memory together, which lets it correct an
order the channels got wrong and costs a forward pass for every candidate it
sees. This one scores the entire fused head, including lexical hits, then
blends its scores with fusion's. It can also see strong graph-only candidates
below the head.

On MIRACL Swahili dev's full 131,924 passages and 482 judged queries, the
recorded full-head pass scored nDCG@10 0.8193 and recall@50 0.9568 through
`search_reranked`; replay reproduced every returned order. The raw run's
complete code/device provenance is not yet archived, so this record is not a
verified current headline measurement.

The resident server caches logits for complete ordered batches within one
loaded model and tokenizer. With the INT8 export, a pair's score can depend on
its neighbours and padding when a pass holds more than one pair -- on an
accelerator, or on a CPU with an inference budget of one thread; on a
multi-core CPU a pass is one pair, so a score is the pair's own. The cache includes every query/memory pair in
order, including duplicates, and both the logical and physical batch shapes.
Changed contexts are scored again, so earlier searches do not change the
result of scoring the same final candidate list. An identical batch avoids a
model forward pass. The most recent complete input list also skips tokenization
and batch planning when its ordered pairs and effective limits match and all
referenced batches remain cached. Changed inputs or evicted references require
tokenizing every pair and planning complete batches before cache lookup. Hashing,
lookup and retrieval still cost work. The cache holds up to 4,096 logical
score slots across complete batches and goes when the model is released or the
server stops. The older pair-cache timings are historical, as recorded in the
[ADR](adr/0001-tech-selection.md), and do not measure this implementation.

The latencies are from four cores. Published figures for a reranker of this
size are a few milliseconds per candidate rather than the ten measured here,
and the difference is the core count.

`--graph-depth` accepts 0 to 4 and refuses anything larger. A topic's
neighbourhood grows multiplicatively with each hop and hub topics reach five
figures of degree, so a fifth hop is not a slower query but a differently sized
one. The walk also starts from at most 64 seeds, keeping the topics the query
named by name ahead of the ones the lexical and vector channels supplied.

```console
$ pamin search "how do we deploy" --limit 3
0.1289  deployment_pipeline v2  the deployment pipeline now runs on argo cd
        lexical_ngram#1 vector#1 graph#2 oncall_rota --depends_on-> deployment_pipeline (1hop) reranked
0.1214  rollback_plan v1  a rollback reverts the deployment pipeline to the previous tag
        lexical_ngram#2 vector#2 graph#3 rollback_plan --mentions-> deployment_pipeline (1hop) reranked
0.1138  oncall_rota v1  the oncall rota rotates every monday morning
        lexical_ngram#3 vector#3 graph#1 oncall_rota --depends_on-> deployment_pipeline (1hop) reranked
```

The JSON carries the same trace, shown here with `--pretty` because it is being
read by a person. Without it the same result is one line of 551 bytes against
934, which is what indentation costs on a hit this size -- about two fifths
off, and more on a fuller result because the saving is per line:

```console
$ pamin search "how do we deploy" --limit 1 --json --pretty
{
  "query": "how do we deploy",
  "hits": [
    {
      "topic": "deployment_pipeline",
      "version": 2,
      "content": "the deployment pipeline now runs on argo cd",
      "score": 0.12889996,
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
        },
        {
          "kind": "reranked"
        }
      ],
      "recorded_at": "2026-10-17T02:23:17.362349Z"
    }
  ]
}
```

### Long memories: spans

A memory is indexed as spans of at most 200 tokens, cut where its structure
allows -- a Markdown heading, a turn in a conversation, a paragraph -- so a
whole session or file can be written as one memory and still be found by a
passage deep inside it. Each span is found by every channel on its own, a
memory is ranked by its best span, and the reranker reads that span with as
many neighbours as fit its pair, nearer first, rather than the memory's
opening. A memory short enough to be one
span is indexed exactly as it always was.

A hit on a memory of more than one span returns that window, not the whole
memory, and says which spans it is:

```json
"span": { "ordinal": 4, "of": 12, "header": "incident_2026_03 > Timeline", "start": 1104, "end": 3391 }
```

The window is the best span and the neighbours, nearer first, that fit a
budget of 600 tokens with it, `start` and `end` its bytes in the memory. `pamin read <topic> --span 4 --around 2` reads further around
it, and `pamin read <topic>` reads all of it. A hit with no `span` field is a
memory of one span, and its `content` is the whole memory.

Measured against whole-memory indexing, spans rank the right memory higher on
every corpus tried -- nDCG@10 0.527 → 0.873 on LoCoMo with one memory per
session -- and return far fewer bytes per hit. On long conversation sessions
the window can miss the turn that answers the question when that turn is not
next to the best span; read around the hit when the answer is not in it.

An index keeps the grain it was built with. One built before spans holds one
document per memory and searches as it always did; `pamin reindex` rebuilds it
into spans.

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
that trade is declined; [ADR 0001](adr/0001-tech-selection.md) has both
tables.

The two lexical channels carry an eighth of a weight each because at full
weight the pair outvotes the other two on exactly the queries where the wording
matches and the meaning does not. They were also once described here as nearly
the same channel, and they are not: Kendall tau-b between their rankings is
0.2816, 0.3188 and 0.2973 on the three corpora this project measures, so they
agree about a third of the time. They share a field, not a ranking. The eighth
each is one number doing the work of two. A later independent 25-pair sweep
on XQuAD-R and MuSiQue found different preferred pairs across the corpora,
so the shipped pair has not changed. The n-gram channel is the weaker of the
two wherever either is measured alone. An eighth rather than the quarter that shipped
before because on MIRACL Swahili — 482 questions people asked, judged by
people — the quarter ranked *worse* than the vector channel by itself, and
because at the time the quarter had not been compared against anything smaller.
Three corpora and the sweep behind that are in
[ADR 0001](adr/0001-tech-selection.md).

The next two paragraphs describe the historical 2026-09-22 XQuAD-R run on a
content-only index with the old same-language answer key. Its 0.6114/0.7829
fusion baseline is not comparable with the current named-index, corrected-key
0.6372/0.8438 baseline; the [ADR](adr/0001-tech-selection.md) records both.

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

**`reranked`** — the cross-encoder scored this result. Its position comes
from its model score blended with fusion's. A result without this entry was not
shown to the model. The entry carries no score on the wire: a cross-encoder's
score is calibrated against nothing, so it
separates the candidates of one shortlist and means nothing between two
queries, and a number on the wire invites exactly the comparison it cannot
support.

The entry distinguishes scored from unscored results; it does not claim that
the model alone set the final order or that the position changed. A line
reading `vector#12 reranked` says the model scored that candidate. When fusion
returns two or more candidates, every result in the fused top thirty is scored,
whichever channel found it, so a line without `reranked` sat below that head
and was not among the strongest candidates only the graph found. When fusion
returns fewer than two, nothing is reranked and no hit carries `reranked`.

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

Showing 4 of 12 topics
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

Run it after changing `--vector-index`, on a project indexed under a retired
profile, or after deleting the index directory. It is also how an index built
before [spans](#long-memories-spans), one document per memory, is cut into
them: an index keeps the grain it was built with, and a rebuild uses the
current one. It rebuilds one project — the one named by `--project` — and
leaves the rest alone. An index built before the vector indexes existed stored
fp32 vectors under an in-memory graph; it is refused with a message naming
this command, and rebuilding it reuses every vector it holds.

It is also how a grown project resizes its vector segments at once, rather
than when the server gets to it. The index sizes them from the number of memories it holds when it is
created, which for a project starting from nothing is the smallest size; a
project that has since grown by orders of magnitude keeps that size until the
index is recreated. Rebuilding recomputes it from what the project holds now,
so a project that has outgrown its layout searches faster afterwards. Where the
old index was built the way a new one is, a memory whose text has not changed
keeps the vector it already has rather than being embedded again.

The server does this on its own. Once a project's index is spread over
more than twice the segments it should be, the server copies it into the right
shape in the background — reading it a batch at a time while it goes on
answering searches and writes, and building the copy's vector graph with the
index free — then swaps the copy in. Writes made during the copy are carried
over before the swap. Nothing is embedded, and for the length of the copy the
disk holds the index twice. `pamin cascade drain` reports a badly shaped index.

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

`pamin write` runs the queue before it returns, and the server runs whatever
is left, so ordinarily there is nothing here to do. These commands are for
making the index catch up at once rather than about ten seconds later, and for
jobs that failed often enough to be set aside.

```console
$ pamin cascade drain
Ran 3 jobs, 0 failed, 0 still owed
```

`drain` runs everything that is due and stops. It is how to make the index
catch up now — after a run of `--defer` writes, or a write that reported
`queued` — and like every other command it runs in the server, so it can be
called while other commands are running.

Without it the server gets there on its own. Every five seconds, if it is not
answering a request, it looks for projects owed work that changes what a search
finds — writes made with [`--defer`](#pamin-write), and work a server that was
killed left behind — and runs it in rounds of sixteen jobs, for up to five
seconds, stopping between rounds as soon as a request arrives. A search that
arrives during a round waits for the rest of it. It also opens a
project nobody has asked about, one per tick, when that project is owed work
that has never failed, so the first search after a server was killed finds
what the dead one had not indexed. What it opens it gives back like anything
else it holds, once nothing has wanted it for the idle window. A job that has
failed is retried when something opens its project, not on its own account.

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
deployment_pipeline v2 (current, 0 of 2 versions, recorded 2026-10-17T02:23:17.362349Z)

the deployment pipeline now runs on argo cd
```

What that buys is everything a short-lived process used to rebuild. Before, each
command connected to PostgreSQL, checked migrations, opened the index, and
loaded an embedding model before it did any work of its own.

`pamin serve` runs it in the foreground instead, which is useful when you want
to watch it. It takes the [server settings](#server-settings) as flags, reads
the rest from `config.toml`, logs each setting and where it came from at the
`info` level, and keeps them until it stops: a command asking for others is
refused with how to restart it, never answered with these. A server started in the background writes to
`$PAMIN_HOME/serve.log`; `PAMIN_LOG` sets its level, as everywhere else.

Between requests it brings indexes up to date with work a deferred write or a
killed server left owed, as [`pamin cascade`](#pamin-cascade) describes, and
looks after the indexes it holds open: it makes applied writes durable,
compacts an index spread over too many files, and reshapes one spread over too
many segments, as `pamin reindex` describes. A reshape logs
`reshaping the index` when it starts and `reshaped the index` with the segment
counts and its duration when it finishes, at the `info` level that
`PAMIN_LOG=info` shows; a reshape that fails logs a warning, which shows by
default, and leaves the index it was copying in service.

One server serves many projects, and it keeps the sixteen most recently used
indexes open; the seventeenth closes the one nobody has touched for longest.
Sixteen is a count of file descriptors, which is what the bound was built for,
and not a budget in bytes, which is what actually runs out: an open index costs
about 205 MB with BGE-M3, so sixteen of them is 3.3 GB. On a machine where that is too
much, `--open-indexes` sets a smaller number. Lowering it costs nothing but
a reopen when a query lands on a project that has fallen out.

Every command goes through the server except three. `serve` is the server,
`stop` is what shuts it down -- with no server running, `stop` stops the
database itself rather than starting a server to do it -- and `config` only
reads the settings a server would start with, so it starts none to show them.
There is no way to run a command
in the calling process instead: `PAMIN_NO_SERVER`, which used to, was removed,
and a command run with it set fails and says so rather than quietly going
through the server it was set to avoid.

The server is its own executable, `pamin-server`, installed beside `pamin`.
`pamin` is only the client: it parses the command, sends it, and prints the
answer, and it links none of the inference runtime, the vector engine, TLS or
the database driver, so a command does not pay to load them. It starts the
`pamin-server` in its own directory, after following a symlink to where `pamin`
really is, and fails with the path it looked for when that file is missing.
`pamin serve`, and `pamin stop` with no server running, hand over to that
`pamin-server`, which takes every command `pamin` does.

A client checks that the running server is the one it would start: the same
`pamin-server` file, by size and modification time, built from the same
protocol. A server from another build -- after a rebuild, or an upgrade -- is
stopped and replaced before the command is answered. Install `pamin` and
`pamin-server` together: a `pamin` from another build than the `pamin-server`
beside it fails, saying so, rather than replacing one server with another just
like it.

So the server is the only process the command line opens an index in. It holds
each project's index open for writing, and the index takes an exclusive lock on
its directory, so any other process opening the same project waits briefly and
then fails rather than sharing it. That is the lock doing its job: two processes
writing one index is what it exists to prevent.

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

### Accelerator plan recovery

Registered plugin libraries participate in automatic routing only for discovered NPU devices; arbitrary non-NPU plugins are not automatically selected. Runtime execution failure quarantines the affected qualified model/revision, runtime/build, target and configured shape scope for five minutes, requalifies the remaining accelerator/optimized CPU plans and retries the complete operation. A replacement becomes visible only after that operation succeeds. Resident models check the recovery deadline before query/score-cache lookups and requalify on the next request after expiry; failed revalidation retains the qualified fallback and reserves another five-minute interval. Successful replacements clear query/score entries while preserving lifetime counters, so cached results from different targets or export scales are not mixed. Completed all-plan measurements use the normal plan lifetime; only incomplete/transient qualification keeps the five-minute retry interval. A provider that cannot be registered in the process (its library or one it needs does not load, or it finds no device) is incompatible rather than transient: a CPU plan chosen without it takes the normal lifetime, so a machine without CUDA does not reload its models every five minutes.

Cold calibration is coordinated in one per-user cache location across workspace model caches. If coordination is unavailable, disk choice reuse/publication is disabled. A validated accelerator winner can be reused for a short interval while transiently failed alternatives are retried; malformed or nonfinite saved reference fixtures are cache misses.

Cold model reloads retain the earliest relevant execution-quarantine deadline.
Selection over temporarily excluded targets remains provisional; a completed
measurement can choose a new accelerator or CPU winner and clears temporary
recovery state. Transient or numerical qualification failures retain a bounded
retry interval; configured exclusions and incompatible plans do not by
themselves require repeated selection.

