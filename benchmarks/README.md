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
  embedder.py        the same embeddings from a process of their own, chat
                     forwarded on, so arms keep one base URL
  arms.py            one class per memory system
  resources.py       memory and disk accounting
  datasets/
    locomo.py        LOCOMO: conversations, and LLM-judged question answering
    longmemeval.py   LongMemEval: per-question haystacks, and retrieval metrics
    supersession.py  LongMemEval's knowledge-update questions, judged three
                     ways: the value that holds, the value it replaced, neither
  run.py             pick a dataset, pick arms, run
  summarise.py       raw rows in, the committed summaries out
  results/           one directory per dataset; summaries and scoped raw evidence are retained

  latency.py            query latency at the same layer on every arm, and what
                        the reader costs against how much context it is handed
  envelope.py           what a caller pays for a whole response, not its contents
  power.py              the supersession question again, five reads a question
  payload.py            whether trimming the JSON costs the reader any accuracy
  longmemeval_recall.py the BM25 baseline, re-ranked rather than stored
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
| Cells do not share queries | Repeated query/candidate pairs reuse scores and change the work being timed | Retain project/query text and order; assert query sets are disjoint across cells and from warmup. Record reranker offered/scored/batch counter deltas to distinguish cached calls from forward passes; a p50 floor does not prove isolation |
| The shortlist an arm reports | A library takes the size under a different keyword, drops the one you passed into `**kwargs`, and serves its own default to both the narrow arm and the wide one | Each arm counts the passages it received and fails if there are more than it asked for |

Reranker score-cache hits still tokenize each offered pair before lookup, so a
repeated query is not guaranteed to cost microseconds. Treat query-set checks
and cache/work counter deltas as separate premises: counters alone do not prove
query uniqueness, and elapsed time alone does not prove cache state. The
[Engine cache evidence](results/reranker/cache-engine-2026-09-30/README.md)
records that workload separately; its timings are not a portable latency floor.
These assertions are requirements for a new comparison, not a retroactive
certification of historical runs.

## Running it

```sh
# 1. the shared endpoint
python3 benchmarks/shim.py 8088 &

# 2. a comparison
python3 benchmarks/run.py --dataset locomo \
    --arms bm25,pamin,pamin-wide,mem0,mempalace \
    --units 10 --questions 20

# 3. what it costs, without answering anything
python3 benchmarks/run.py --dataset locomo --mode cost --arms ...

# 4. whether an answer is the fact that still holds
python3 benchmarks/run.py --dataset supersession \
    --arms bm25,pamin,pamin-dated,pamin-valid,pamin-valid-read \
    --units 70 --questions 1
```

`--units` is conversations on LOCOMO and questions on the other two, because a
LongMemEval question carries its own haystack. This file said
`--conversations` for a while; no such flag exists.

Two runs at once halve the wall clock -- the reader and the judge are network
waits, and the machine is idle through them -- and make four of the cost
columns describe the pair rather than the arm. Set `BENCH_SHARED_ENDPOINT=1`
when you do it. Every row then says so and the summary prints what survived
instead of a table that looks the same as a clean one. The arm that caught this
was one with no model on its write path reporting fourteen LLM calls, all of
them the other run's.

Third-party systems each install into their own virtualenv. Installing one of
them into the shared environment moved `protobuf` past the ceiling another
declares, while that other one was being measured.

**Install each one the way its own documentation does, optional extras
included.** `pip install mem0ai` leaves out `mem0ai[nlp]`, and without it
mem0's `lemmatize_for_bm25` returns its input unchanged -- so every memory is
stored with an unlemmatised keyword field and every query is matched against
one, which turns off half of a hybrid retriever. It reports this on a log line
and nowhere else. A default install is not the same as the system, and
measuring a competitor with part of it disabled is not a measurement of that
competitor.

The same channel has a second half-disabled trap: mem0's Qdrant store
lazy-imports `fastembed` to encode the BM25 sparse vector it stores beside
the dense one, and without it every write and every query falls back to
dense-only, again logged once and nowhere else (`pip install "mem0ai[extras]"`).
`Mem0.ingest` asserts both before trusting a run.

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

## What is committed under `results/`, and how to regenerate it

A run writes one row per arm per question, continuously, into
`results/<dataset>/`. Conversation/user-derived per-question rows stay out of git: they can contain
private content and a running experiment is still writing them. The scoped
`results/compute/` exception retains immutable, vetted public-fixture IDs,
relevance/exclusion labels, rankings and numeric timing/resource records, with
provenance and independent validation. It excludes personal conversation text,
provider credentials and live/incomplete runs. Those archives explicitly add
only their audited raw files; the default `*.jsonl` ignore remains.

For the conversation benchmarks, what is committed is one `summary-*.json` per table in
[docs/benchmarks.md](../docs/benchmarks.md), so that a reader of that page has
an artifact to check it against rather than a promise:

```
results/
  locomo/
    summary-accuracy.json          judged accuracy, by question type, McNemar
    summary-cost.json              write side, query side, resident and disk
    summary-latency.json           retrieval per arm per shortlist, and the reader
    summary-mempalace-no-llm.json  MemPalace's published mode against its default
  longmemeval/
    summary-session-retrieval.json   recall@k, no reader and no judge
    summary-supersession.json        current, stale or neither, over 70 questions
    summary-supersession-power.json  the same 70, read five times each
  retrieval/
    xquad-r-input.sha256             hashes of the pinned question, answer and passage files
    xquad-r-index-profile.txt        evaluated project's passage and storage marker
    xquad-r-model-artifacts.json      pinned hub revisions and model file hashes
    xquad-r-current-run.log          product-path XQuAD-R harness output
    summary-current-xquad.json       its four group metrics and run conditions
```

Regenerate them from a run's raw rows with:

```sh
python3 benchmarks/summarise.py /path/to/raw
```

It recomputes every figure from the rows rather than copying one from the page,
so a disagreement between the two shows up as a diff. It needs the LongMemEval
corpus as well, because the BM25 baseline in the retrieval table was never
stored per question and has to be re-ranked; `longmemeval_recall.py` does that
and `summarise.py` imports it rather than keeping a second copy of a retriever.

Each summary records the machine it came from, and a row without one is not a
measurement. The first LOCOMO and LongMemEval runs predate the field, so the
summaries built from them say `"machine": null` and say why in the same object:
an unverifiable row should not be dressed up as a verified one, and the run log
is not the row.

XQuAD-R uses the Rust `crosslingual` harness rather than `benchmarks/run.py`.
It prints group aggregates, not per-question rows. On a clean cache, run
`env -u HF_HOME PAMIN_EVAL_HOME=/path/to/eval-home cargo test -p pamin-engine --test crosslingual prepare_pinned_xquad_models -- --ignored`
before the recorded search command; this fetches the two archived model
revisions and makes the product loader use them. Its committed log is the
source for `python3 benchmarks/results/retrieval/derive_xquad.py`, which reads
the code commit, named passage encoding, indexed document count and forced CPU
reranker export printed at run time; the search command sets `PAMIN_DEVICE=cpu`.
the input manifest covers all eleven upstream JSON files, including the
question and answer keys, and the harness rejects stale cached copies. The
archived index marker is checked as `named` by the extractor. The model
manifest records both hub revisions, tokenizer files and prepared ONNX bytes;
`python3 benchmarks/results/retrieval/verify_xquad_models.py /path/to/eval-home/models xquad-run.log`
checks the cache before treating a run as the same configuration. The
concurrent rerun was used for accuracy only; its wall time is not a latency
measurement.

## Prototype compute evidence (2026-10-01)

See [dual-vector and shared compute results](results/compute/prototype-closeout-2026-10-01/README.md) for paired precision, independent process costs, resource tradeoffs and frozen revision limits; [normalization evidence](results/compute/i8-normalization-2026-10-01/README.md) is a kernel throughput diagnostic. Run `python3 benchmarks/verify_compute_evidence.py` to independently check the retained product rows.
