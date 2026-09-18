<img src="assets/icon.png" alt="Påmin Memory" width="96">

# Påmin Memory

Påmin Memory (Pamin Memory) is universal memory for AI agents, coding assistants, research tools, and knowledge-heavy applications.

It is designed to turn durable evidence into versioned knowledge that agents can retrieve through structure, meaning, relationships, and time. Instead of treating memory as a pile of extracted snippets, PaminMemory keeps the source trail intact, tracks how facts evolve, and explains why each piece of context was selected.

> **Early, and measured.** Retrieval, the version ledger, the relationship graph and the resident server all work and are benchmarked below. Source ingestion, page trees, curated notes and the MCP surface are not built. See [Scope](#scope).

## What It Does

- Preserves raw evidence and source spans as the authority behind memory.
- Tracks versioned memories so current, stale, contradicted, and historical facts can be separated.
- Combines lexical matching, semantic recall, relationship structure, and a reranking pass over what the lexical channels missed.
- Builds explainable context from the same evidence ledger rather than opaque one-off summaries.
- Prioritizes local-first operation so developers can inspect and control their memory stack.

## Where This Differs

Measured against mem0 and MemPalace on LOCOMO, every arm reading and judged by
the same model and embedding through the same endpoint. The full tables,
including the two conditions that were measured wrongly the first time, are
under [Measured](#measured).

**On answer accuracy the three systems tie, and this page will not claim
otherwise.** At a matched shortlist of thirty passages: 0.628 here, 0.623 for
MemPalace, 0.583 for mem0, and no pair of them separates statistically. Anyone
selling you a memory system on a few points of LOCOMO is selling you noise —
mem0 run twice at identical settings scores 0.603 and 0.598 while answering 41
of the same 199 questions differently.

The differences that are real are architectural, and they follow from one
choice: **no language model runs on the write path.**

| | this project | mem0 |
| --- | --- | --- |
| model calls to ingest 10 conversations | **0** | 272 |
| cost of that ingest | **$0** | $27.77 |
| time | **356 s** | 4,121 s |
| embedding requests to a service you must run | **0**, in-process | 6,335 |
| same corpus written twice | **byte-identical** | 41 of 199 answers change |
| questions whose answer is implied, not stated | **0.429** | 0.190 |

A system that asks a model to decide what a conversation *means* before storing
it pays for that on every ingest, cannot reproduce its own store, and cannot
return what the model chose not to write down. A system that stores the
evidence pays instead on every query, in the prompt it hands back. The two
cross at roughly **1,900 questions asked of a single conversation's memory**;
below that this is cheaper, and at LOCOMO's own density of twenty questions it
is cheaper by a factor of 31.

**Where this is behind.** mem0 leads temporal questions by about twenty points
(0.735 against 0.529) — reproducibly, across independent runs. Holding the
embedding model in-process costs about 2 GB resident where a system calling out
to an endpoint holds 177 MB and a bill. And at thirty passages this hands the
reader 1,511 prompt tokens against mem0's 1,017, because passages are longer
than rewritten facts.

## Quickstart

No Docker. No API key. No configuration. `init` provisions a local PostgreSQL for you, and the embedding model downloads the first time you search.

```bash
cargo install --path crates/pamin-cli

pamin init
pamin write --topic deployment_pipeline "deploys through the ci pipeline"
pamin search "how does deployment work"
```

Reading a topic's history, and what it looked like before:

```bash
pamin read deployment_pipeline
pamin read deployment_pipeline --version-offset 1
```

Every command takes `--json`, because the usual caller is an agent parsing output rather than a person reading it:

```bash
pamin search "deployment" --json
```

`pamin stop` shuts down the local database and the resident server. Both are deliberately left running between commands so an agent invoking the CLI repeatedly does not pay startup each time.

`pamin grep` searches the evidence itself — verbatim, unranked, and including content the filter held and no index ever saw.

Every command, its options, and its JSON shape are in [docs/cli.md](docs/cli.md).

## Any Language

Evidence is stored exactly as it arrives and is never translated. Translation would put a model on the write path, and it would break exact matching: after translation your own words no longer find your own memory.

Cross-language recall is handled by retrieval instead. Text is segmented with ICU before indexing, which covers languages that write without spaces, and a multilingual embedding model lets a query in one language reach a memory written in another.

```bash
pamin write --topic deploy "部署流水线运行在持续集成上面"
pamin search "how is the code deployed"   # finds it
```

## Architecture

The design rests on one separation: **an authority that is written to, and a
projection that is read from.** PostgreSQL holds every fact the system is
accountable for. The retrieval index holds nothing that PostgreSQL cannot
reproduce.

```text
                    write                              read
                      │                                  │
          ┌───────────▼───────────┐          ┌───────────▼───────────┐
          │  AUTHORITY            │          │  PROJECTION           │
          │  PostgreSQL           │          │  zvec, in-process     │
          │                       │          │                       │
          │  · raw evidence and   │  outbox  │  · segmented lexical  │
          │    source spans       │ ───────► │  · n-gram lexical     │
          │  · bi-temporal        │  cascade │  · dense vectors      │
          │    version ledger     │          │                       │
          │  · relationship graph │          │  derived: losing it   │
          │  · the outbox         │          │  costs a reindex,     │
          └───────────────────────┘          │  not a migration      │
                      │                      └───────────┬───────────┘
                      │  graph channel                   │  three channels
                      └──────────────┬───────────────────┘
                                     ▼
                        reciprocal rank fusion, in our layer
                                     ▼
                        optional cross-encoder rerank
                                     ▼
                        results, each carrying why it is here
```

**A bi-temporal version ledger, not a key-value store.** Every memory carries
both when a fact was true and when the system learned it — application time and
system time, the two period dimensions SQL:2011 names. Superseding a fact
writes a new version and closes the old one's validity rather than overwriting
it, and deletion is a closed interval rather than a `DELETE`. That is what lets
current, stale, contradicted and historical be distinguished instead of
conflated, and it is why a question about what was believed last March has an
answer.

**Evidence is preserved, never rewritten.** Nothing on the write path asks a
language model to decide what a conversation "means". Raw content and its
source spans stay as they arrived; the sensory filter records *why* something
was held back without discarding it. This is an architectural commitment with
measurable consequences, listed under
[Against the other memory systems](#measured): no cost and no external service
on ingest, a store that is byte-identical when the same corpus is written
twice, and answers that survive questions whose evidence was never stated
outright.

**A transactional outbox instead of dual writes.** A write records, in the same
transaction that stores the evidence, what the projection now owes it. A
cascade worker settles that debt afterwards. The index can therefore lag, fail
or be thrown away entirely without the authority ever being wrong — and
`pamin reindex` rebuilds it from PostgreSQL, which is also what keeps the
retrieval engine a replaceable component rather than a permanent commitment.

**Four recall channels, fused above the index rather than inside it.**
Segmented lexical, n-gram lexical, dense vector, and the relationship graph.
The first three come from the projection; the fourth lives in PostgreSQL, where
the index cannot see it. Letting the index pre-fuse its own three would produce
a list that then had to be fused again — weighting its members twice and losing
the rank each held in each channel. Fusing once, above both, is what makes the
result explainable:

```bash
$ pamin search "deployment pipeline" --json | jq '.hits[0].why'
[ { "kind": "channel", "channel": "lexical_ngram", "rank": 1, "weight": 0.25, ... },
  { "kind": "channel", "channel": "vector",        "rank": 2, "weight": 1.0,  ... },
  { "kind": "channel", "channel": "graph",         "rank": 1, "weight": 1.0,  ... },
  { "kind": "path", "from": "oncall_rota", "via": "oncall_rota", "hops": 1, ... } ]
```

Every hit reports the rank it held in each channel and the graph path that
reached it. There is no step at which a score becomes unattributable.

**A cross-encoder pass, tiered.** Over the fused shortlist, `off`, `fast`
(default) and `accurate` trade latency for quality on a curve that is measured
rather than assumed — the figures, including one that had to be corrected
twice, are under [Measured](#measured).

**Everything local.** Embeddings run in-process through ONNX Runtime;
PostgreSQL is bundled rather than something you install. A default install
makes no network call at query time and needs no API key.

### Crate layout

| crate | responsibility |
| --- | --- |
| `pamin-core` | domain model, ledger semantics, fusion. No heavy dependencies, because it is edited most and its rebuild cost sets the development loop. |
| `pamin-store` | the PostgreSQL authority: evidence, ledger, graph, outbox. |
| `pamin-index` | the projection: multilingual segmentation, lexical and vector channels. |
| `pamin-engine` | the only crate that holds both, and therefore the only place they can drift. Sits above `pamin-core` so index types never reach the domain layer. |
| `pamin-cli` | the command surface, and the resident server behind it. |

Design decisions, their trade-offs, and the ones that reversed when measured
are recorded in [docs/adr/](docs/adr/).

## Relationships

Memories are connected as well as ranked. Writing a memory that names another topic derives an edge to it, with no model in the path and nothing to configure:

```bash
pamin write --topic rollback_plan "a rollback reverts the deployment pipeline to the previous tag"
pamin neighbors rollback_plan          # finds deployment_pipeline
```

Derivation only finds relationships the text states, so anything else is asserted directly:

```bash
pamin link oncall_rota deployment_pipeline --kind depends_on
```

Edges are versioned the way memories are. Changing one closes the old version and appends a new one, `unlink` retracts a claim without erasing that it was made, and every edge carries its own validity interval — so "what did we think depended on this, back then" has an answer.

## Measured

Every figure below comes from `pamin search` and `pamin write` themselves, not
from the model or the index underneath them, because the gap between those two
is where this project's numbers have been wrong before.

**Retrieval quality**, at the shipped defaults, median of three runs:

| corpus | group | nDCG@10 | recall@50 |
| --- | --- | --- | --- |
| MIRACL Swahili dev — 131,924 real passages, 482 queries, 5,092 human judgements | one language throughout | 0.7359 | 0.9494 |
| XQuAD-R — 13,014 sentences in eleven languages, 1,190 queries | query and answer in **different** languages | 0.6097 | 0.8864 |
| XQuAD-R | query and answer in the same language | 0.7971 | 0.9630 |

Both corpora are fetched rather than vendored, and the harness that drives them
is in the repository: `cargo test -p pamin-engine --test crosslingual -- --ignored`.

**What those MIRACL figures are worth, against published results on the same
corpus, the same dev split and the same qrels:**

| MIRACL Swahili dev, 131,924 passages | nDCG@10 | what it is |
| --- | --- | --- |
| Pyserini BM25 baseline | 0.3826 | lexical only |
| this project, `--rerank off` | 0.7158 | four channels fused |
| this project, `fast` (default) | **0.7359** | fused, then a cross-encoder |
| this project, `accurate` | 0.7654 | fused, then a larger cross-encoder |
| BGE-M3, published | 0.787 | dense retrieval alone |

Read that last row carefully, because it is the honest reading: **a whole
retrieval stack here scores below a single dense retriever** — and it is the
same model, an int8 export of BGE-M3. Two differences are known and neither is
measured: the published figure is fp32, and MIRACL's own training split is in
BGE-M3's fine-tuning data, where this runs zero-shot. Neither excuses the gap;
they are where to look for it.

What the table does establish is the distance from the lexical baseline a
memory system would otherwise ship with, on a low-resource language, on four
CPU cores with no GPU anywhere.

Sources: [Pyserini MIRACL v1.0 regressions](https://github.com/castorini/pyserini/blob/master/docs/experiments-miracl-v1.0.md)
and [BGE-M3](https://arxiv.org/abs/2402.03216) Table 1 (v4 or later; v1–v3
report 0.786 and were corrected). The 0.7359 above was re-run and reproduced
exactly before being placed here.

**Retrieval on a memory benchmark.** LongMemEval-S, 59 of its 500 questions
drawn stratified by type, with the abstention questions dropped because their
correct answer is a refusal and the benchmark's own scorer drops them too. Each
question carries its own haystack of about fifty sessions and five hundred
turns; turns are indexed individually and rolled up to the session that
contains them. `recall_all@k` requires every gold session inside the top k;
`ndcg_any@k` counts any gold session as relevant. A plain BM25 over the same
turns, the same roll-up and the same cut runs beside it, because a retrieval
number without a lexical baseline says nothing about retrieval:

| LongMemEval-S, session level, 59 questions | BM25 | this project |
| --- | --- | --- |
| recall_all@5 | 0.7966 | 0.8983 |
| ndcg_any@10 | 0.8898 | 0.9202 |

The total is not the result. Split by question type it is:

| recall_all@5, by question type | n | BM25 | this project |
| --- | --- | --- | --- |
| multi-session | 15 | 0.467 | **0.867** |
| temporal-reasoning | 16 | 0.750 | 0.750 |
| knowledge-update | 9 | 1.000 | 1.000 |
| single-session-user | 8 | 1.000 | 1.000 |
| single-session-assistant | 7 | 1.000 | 1.000 |
| single-session-preference | 4 | 1.000 | 1.000 |

Four channels, rank fusion and a cross-encoder beat a plain lexical baseline on
one of the six question types. On four of the others BM25 already scores
perfectly, so those rows measure the benchmark and not any system. On the
sixth the two are not merely close: across all sixteen temporal-reasoning
questions they reach the same verdict question for question and fail on the
same four. Nothing in the stack bought anything there.

The win is real where it is real. multi-session is the type whose evidence is
spread over several sessions with no single one matching the question well, and
there this is forty points of recall@5 above lexical retrieval — with no
question anywhere in the set where it scores below BM25.

Three things this is not. It is not evidence about temporal reasoning: the
haystack was loaded as one memory per turn, so no topic ever had a second
version and the validity columns were never populated — the ledger this project
is built around was not in the measurement at all, and the retrieval that was
measured performs exactly as a lexical baseline does. It is not comparable to
the retrieval tables in the LongMemEval paper, which are computed on
LongMemEval-M, where each haystack holds roughly ten times as many sessions.
And recall@50 is omitted because the haystack holds about fifty sessions, so it
would be near one by construction.

Ingest ran at a median 112 s a question for about 480 turns, 29,170 turns in
all; search over one loaded haystack had a median of 0.24 s and a p95 of 0.47 s.

**Against the other memory systems.** LOCOMO, ten conversations and 5,882
turns, 199 questions drawn stratified and answered by every arm. One model
reads the retrieved passages and one judges the answer, the same model for
everyone; every arm embeds with the same BGE-M3 file, through one endpoint that
counts what each of them asks for. A comparison where the arms use different
models measures the models.

| LOCOMO, 199 questions | passages | accuracy | write: model calls | write: cost |
| --- | --- | --- | --- | --- |
| BM25, no memory system | 10 | 0.427 | 0 | $0 |
| this project | 10 | 0.518 | **0** | **$0** |
| mem0 | 10 | 0.538 | 272 | $27.77 |
| MemPalace | 10 | 0.558 | 20 | $0.38 – $1.15 |
| mem0 | 30 | 0.583 | 272 | $27.77 |
| MemPalace | 30 | 0.623 | 20 | $0.38 – $1.15 |
| this project, `--limit 30` | 30 | **0.628** | **0** | **$0** |

mem0's two rows share one ingest, so its write bill is paid once for both.
MemPalace's rows are two separate ingests of the same conversations, which cost
$1.15 and $0.38 — the same twenty calls, priced differently by prompt caching,
which is why its cell is a range and not a figure.

**On accuracy this is a tie, and reporting it as a win would be wrong.** At
thirty passages the three systems are 0.628, 0.623 and 0.583, and paired
McNemar separates no pair of them — p = 0.289, 1.000 and 0.396. At ten
passages, likewise. Being first by five thousandths is not being ahead.

There is a floor under those comparisons, and it is worth more than they are.
Running mem0 twice at the same settings on the same data gives 0.603 and 0.598
— but **41 of the 199 questions change answer between the two runs**. A
distillation performed by a model is not the same twice. This project has no
such floor on the write side, because there is no model there: the `--limit 30`
row above reads a store the `--limit 10` row built, byte for byte.

Two question types do clear that floor, and both reproduce across independent
runs. mem0 leads **temporal** questions by about twenty points, 0.735 against
0.529. This project and MemPalace lead **adversarial** questions — the ones
whose answer is implied rather than stated — by about the same, 0.429 against
0.190. That is what rewriting a conversation into facts costs: what was never
said is not in the rewrite to find. The totals tie because these cancel, which
is not the same as the systems performing alike.

**The difference is on the bill.** This project puts ten conversations in with
no model calls, no cost and no external service, in 356 seconds; mem0 takes 272
calls, $27.77 and 4,121 seconds, and 6,335 embedding requests to an endpoint it
does not host. What mem0 buys with that is a smaller prompt afterwards — about
1,017 tokens a question against 1,511 here, since it hands back rewritten facts
rather than passages. So one side pays once and the other pays forever, and
they cross at roughly **1,900 questions asked of a single conversation's
memory**. At LOCOMO's own density of twenty questions, the totals are $0.09
against $2.84.

Two findings here are negative and stay on the page: the version ledger this
project is built around bought nothing on this benchmark (p = 1.00), and a
prediction written down before re-running mem0 with its full retriever — that
its figures would rise — was wrong. The arms, what is held fixed and how each
condition is asserted are in [benchmarks/](benchmarks); the full tables,
including two conditions that were measured wrongly the first time and what
they invalidated, are in [docs/benchmarks.md](docs/benchmarks.md).

**Latency**, what one `pamin search` costs against a warm resident server at
the default `accuracy` profile. Each figure is a whole CLI invocation — fork,
exec, connect to the socket, and back — run serially over forty distinct
queries, reported as the median of them:

| corpus | `--rerank off` | `fast` (default) | `accurate` |
| --- | --- | --- | --- |
| XQuAD-R, 13,014 documents | 77 ms | 251 ms | 1241 ms |
| MIRACL Swahili dev, 131,924 documents | 142 ms | 472 ms | 1675 ms |

Seventeen to nineteen of those milliseconds are the invocation rather than the
search: `pamin --help` against the same workspace costs that much. It is
measured separately rather than subtracted, because a caller pays it either
way.

A write is 32 ms, most of it the `fsync` a durable append owes. That figure is
from the write-path measurement in the ADR and was not re-taken in this sweep.

Measured on 4 vCPU (Intel Xeon @ 2.80GHz, no SMT), 15 GB RAM, Ubuntu 24.04,
rustc 1.98.1, release build, embeddings on CPU through ONNX Runtime. The
queries in each cell are disjoint from every other cell's, because a repeated
query is answered from a cache in microseconds and would be reported here as
search latency.

`--rerank accurate` scores higher than the default on every corpus measured and
its pass costs about four and a half times `fast`'s; `fast` is the default on
that latency difference alone, which is a judgement rather than a result.

Two things these numbers are not. Four cores is where the embedding model and
the reranker contend, so a machine with cores to spare will not look like this
— published figures for a reranker of this size are a few milliseconds per
candidate against the ten measured here. And the write figure is for short
memories: a forward pass scales with length, so longer content costs more.

**Throughput, and where it stops.** The same sweep at one, eight and
thirty-two concurrent callers, `fast` being the default:

| corpus | tier | 1 | 8 | 32 | ceiling |
| --- | --- | --- | --- | --- | --- |
| XQuAD-R, 13,014 | `off` | 13.1 q/s | 19.5 | 20.7 | **~21 q/s** |
| | `fast` | 3.5 q/s | 4.7 | 4.2 | **~4.7 q/s at eight** |
| | `accurate` | 0.8 q/s | 0.9 | 1.0 | **~1 q/s** |
| MIRACL, 131,924 | `off` | 6.5 q/s | 10.3 | 10.6 | **~11 q/s** |
| | `fast` | 2.0 q/s | 2.4 | 2.4 | **~2.4 q/s** |
| | `accurate` | 0.6 q/s | 0.6 | 0.6 | **~0.6 q/s** |

Read it as a ceiling rather than a score. Four cores saturate at eight
concurrent callers and the rest is queueing: on the default tier, thirty-two
callers get *less* throughput than eight (4.2 against 4.7) and wait twenty-one
times longer than one does — p50 goes from 251 ms to 5.4 s. Nothing here
scales by adding callers. Adding cores is the lever; this measurement does not
say by how much.

**What a server holds.** Resident memory after all three tiers have run, which
is when the embedding model and both rerankers are loaded at once:

| corpus | server RSS | index on disk | workspace |
| --- | --- | --- | --- |
| XQuAD-R, 13,014 documents | 3.6 GB | 119 MB | 2.1 GB |
| MIRACL, 131,924 documents | 7.2 GB | 1.1 GB | 2.2 GB |

Seven gigabytes for a hundred and thirty thousand documents is the number to
plan around, and it is why two workspaces do not fit on a sixteen-gigabyte
machine at this corpus size. A workspace that never asks for `accurate` never
loads the 570 MB reranker; `--rerank off` never loads either.

**Above this, nothing is measured.** The largest corpus here is 131,924
documents. A million and beyond is untested — not projected, not extrapolated,
untested — and the descriptor count is the first thing that would break: this
index is 2,111 segment files and a search holds 2,733 descriptors open, which
already exceeds the 1,024 a Linux process is given by default.

What was measured, how, and the conclusions that reversed on measurement are in
[docs/adr/0001-tech-selection.md](docs/adr/0001-tech-selection.md), which is the
source of truth if it and this page ever disagree.

## Scope

Everything the architecture above describes is built; what has been measured,
and what has not, is stated in [Measured](#measured). Not built yet: source
ingestion and page trees; curated notes and the session brief; passive
optimization and forgetting; the MCP surface.

## Development

```bash
cargo test --workspace                  # fast; no database, no model
cargo test --workspace -- --ignored     # provisions postgres, downloads models
```

## Maintainer Notes

This repository can optionally include private maintainer notes through the `internal-docs` submodule. Public users do not need that submodule to use or follow the open-source project.

See [docs/internal-docs.md](docs/internal-docs.md) for the maintainer workflow.
