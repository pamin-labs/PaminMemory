# Benchmarks

A harness for comparing this project against other agent-memory systems, with
the two things that usually differ between such comparisons held fixed: every
arm reads with the same model, and every arm that embeds uses the same weights.
What is left varying is the memory system, which is the only way a number here
says anything about memory systems.

## Layout

```
benchmarks/
  shim.py            an OpenAI-shaped endpoint over `claude -p`, chat and embeddings
  arms.py            one class per memory system
  resources.py       memory and disk accounting
  datasets/
    locomo.py        LOCOMO: conversations, and LLM-judged question answering
    longmemeval.py   LongMemEval: per-question haystacks, and retrieval metrics
  run.py             pick a dataset, pick arms, run
  results/           one directory per dataset; every row carries its machine
```

Adding a benchmark is adding a loader. It is deliberately not a directory per
benchmark holding its own copy of the arms: two copies of an arm drift, and
then two tables disagree and neither is wrong.

## What is held fixed, and how it is checked

Each of these has failed silently at least once, so each is an assertion rather
than a convention.

| Held fixed | What it looks like when it breaks | The assertion |
| --- | --- | --- |
| One model for every arm | An arm quietly uses its own default | Arms declare their endpoint; the shim counts calls and the run reports them per arm |
| One embedder for every arm | An arm keeps its own 384-dimension model and says nothing | The shim counts embedded texts; an arm whose count does not move during ingest fails |
| The corpus is actually there | Every query returns nothing, quickly, and a full latency table is the cost of searching an empty index | The project is discovered, its size asserted against the documents claimed, and one real query asserted to return hits |
| One server, not two | Two copies of a model resident, and a measurement that dies of memory | Wait on the socket, then assert the process count is exactly one |
| Cells do not share queries | A repeated query is answered from cache in microseconds and reported as search latency | Every cell draws a disjoint slice, and asserts its own p50 is above a floor no forward pass can beat |

## Running it

```sh
# 1. the shared endpoint
python3 benchmarks/shim.py 8088 &

# 2. a comparison
python3 benchmarks/run.py --dataset locomo \
    --arms bm25,pamin,pamin-wide,mem0,mempalace \
    --conversations 10 --questions 20

# 3. what it costs, without answering anything
python3 benchmarks/run.py --dataset locomo --mode cost --arms ...
```

Third-party systems each install into their own virtualenv. Installing one of
them into the shared environment moved `protobuf` past the ceiling another
declares, while that other one was being measured.

## Where you run this changes which numbers mean anything

Half of what this reports is a property of the memory systems and travels
between machines. The other half is a property of the machine and does not.
Mixing them is how a benchmark stops being one.

| Travels | Does not travel |
| --- | --- |
| Accuracy, and its split by question type | Latency, p50/p95, throughput |
| Recall@k, nDCG@k | Resident memory, peak and resting |
| LLM calls, and tokens per question | Wall-clock ingest time |
| Bytes stored | Anything measured while something else ran |

An accuracy figure from a slow box is the same accuracy. A latency figure from
one is only that box's latency. Say which you are quoting.

### The container the published figures came from

4 vCPU (Intel Xeon @ 2.80 GHz, no SMT), 15 GB RAM, Ubuntu 24.04. Four things
about it shaped this harness and would not shape a laptop:

- **It is reclaimed without warning.** Four times in two hours, mid-run, taking
  every background process with it -- including a watchdog written to restart
  the run. What survives is the filesystem, so the harness checkpoints after
  every conversation and resumes by skipping what it finds. Anything longer
  than about twenty minutes also needs a timer *outside* the container.
- **No API key.** Every model call goes through `claude -p` behind the shim,
  which is also what lets every arm share one model and one embedder. With a
  key, point the arms at the real endpoint -- and then check the embedder
  again, because that is the part that falls back quietly.
- **Everything runs as root, and PostgreSQL refuses to.** `initdb` will not run
  as root, so the workspace is provisioned as an unprivileged user and the
  harness derives that user from the data directory's owner.
- **The open-file limit is 1,024.** A 131,924-document index is 2,111 segment
  files and holds 2,733 descriptors at once.

### Moving to an Apple Silicon machine

- **Re-measure every latency and memory figure; none of them carry over.**
- **macOS defaults to 256 open files**, lower than Linux, so a server that
  raises its own soft limit matters more there, not less.
- **ONNX Runtime may select CoreML**, which is a different execution provider
  and therefore a different measurement at the same batch size.
- **Nothing above 131,924 documents has been measured anywhere.** That is the
  first thing worth running on a larger machine, and the descriptor count is
  the first thing likely to break.

Every file under `results/` records the machine it came from. A row without one
is not a measurement.
