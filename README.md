<img src="assets/icon.png" alt="Påmin Memory" width="96">

# Påmin Memory

**[memory.paminlabs.com](https://memory.paminlabs.com)** · universal memory for
AI agents, coding assistants, research tools, and knowledge-heavy applications.

It is designed to turn durable evidence into versioned knowledge that agents can retrieve through structure, meaning, relationships, and time. Instead of treating memory as a pile of extracted snippets, Påmin Memory keeps the source trail intact, tracks how facts evolve, and explains why each piece of context was selected.

> **Early, and measured.** Retrieval, the version ledger, the relationship graph and the resident server all work and are benchmarked below. Source ingestion, page trees, curated notes and the MCP surface are not built. See [Scope](#scope).

## What It Does

- Preserves raw evidence and source spans as the authority behind memory.
- Tracks versioned memories so current, stale, contradicted, and historical facts can be separated.
- Combines lexical matching, semantic recall, relationship structure, and a reranking pass over what the lexical channels missed.
- Builds explainable context from the same evidence ledger rather than opaque one-off summaries.
- Prioritizes local-first operation so developers can inspect and control their memory stack.

## Where This Differs

**LOCOMO accuracy matching mem0 and MemPalace — with zero model calls on the
write path, zero cost to ingest, and a store that rebuilds itself byte for
byte.**

Measured head to head: every arm answering the same 199 questions, read and
judged by the same model, embedding through the same endpoint. Reproducible
from this repository with the commands in [benchmarks/](benchmarks).

| LOCOMO, thirty passages | accuracy | to ingest 10 conversations | retrieval |
| --- | --- | --- | --- |
| **Påmin Memory** | **0.628** | **0 calls, $0, 356 s** | **25.7 ms** |
| MemPalace | 0.623 | 20 calls, ~$1, 500–545 s | 54.7 ms |
| mem0 | 0.583 | 272 calls, $27.77, 4,121 s | 64.6 ms |

No pair of those accuracies separates statistically. That is the claim, and it
is deliberately a tie: **parity with the systems this category is named after,
from a design that spends nothing to reach it.**

### Why these numbers are lower than the ones on everyone's website

mem0 advertises 92.5 on LOCOMO. Under this harness it scores 0.583. Zep
advertises 94.7%. Neither is lying and neither figure is wrong — LOCOMO's score
is produced by a reader model turning passages into an answer and a judge
grading it, and changing either moves the result by tens of points before the
memory system is involved at all. Zep's own table shows 63.8% against 71.2% for
the same retrieval, read by a smaller and a larger model.

So an absolute LOCOMO number means nothing across harnesses, and this page does
not publish one to be compared against a website. It publishes three arms under
one reader, one judge and one embedder, and the noise floor beside them: mem0
run twice at identical settings scores 0.603 and 0.598 while answering 41 of
the same 199 questions differently. An independent analysis of the standard
LOCOMO harness found its judge accepting 63% of intentionally wrong answers.

A tie anyone can re-run is worth more than a lead nobody can check, and it is
the one claim here that survives someone checking it.

### The one absolute number, and why it is not the one they publish

LOCOMO scores move with the reader, so nothing above is quoted against a
website. LongMemEval's retrieval stage has no reader and no judge — it asks
whether the gold session is in the top k — so it is the one figure here that
can sit beside a published one. MemPalace publishes **96.6% R@5** on it.

| LongMemEval session retrieval, no model anywhere | BM25 | Påmin Memory |
| --- | --- | --- |
| R@5, as the field defines it — gold session in the top five | 96.6% | 98.3% |
| R@10 | 98.3% | **100%** |
| **R@5 strict — *every* gold session in the top five** | 79.7% | **89.8%** |

Read the first row and then discard it. **A plain BM25 keyword search, with no
memory system of any kind, scores 96.6% — the published headline, to the
digit.** Fifty candidate sessions and "is the gold one in the top five" does
not separate an architecture from `grep`; five of the benchmark's six question
types are at a perfect score for BM25 alone. Being 1.7 points above keyword
search there is not a product claim, and it is not made here.

The row that means something is the last one. Thirty-five of these 59 questions
have more than one gold session, and requiring all of them is the difference
between finding the evidence and finding *some* of it: **79.7% against 89.8%,
ten points over the lexical baseline, with no model called at any stage.**

### What the parity is bought with

Everything that separates these systems follows from one choice: **no language
model runs on the write path.** Of the seven memory systems surveyed in
[docs/benchmarks.md](docs/benchmarks.md), six run one by default. MemPalace can
be told not to, with `init --no-llm`, and measured that way it reaches the same
accuracy as Påmin Memory — so the property is not unique, and it does not buy
accuracy. What it buys is everything in the table below, and here it is the
architecture rather than a flag: there is no mode in which Påmin Memory puts a
model on its write path.

| | Påmin Memory | MemPalace | mem0 |
| --- | --- | --- | --- |
| texts embedded to ingest ten conversations | 5,882 | **1,668** | 6,335 |
| of those, requests to a separate process | **0**, in-process | 1,668 | 6,335 |
| same corpus written twice | **byte-identical** | LLM on the write path by default | 41 of 199 answers change |
| prompt tokens handed back, thirty passages | 1,511 | 5,133 | **~1,017** |

Those first two rows used to be one row reading "embedding requests to a
service you must run: 0". Both halves of that were wrong. **Påmin Memory does
not embed nothing** — it embeds every turn it stores, 5,882 of them, more than
MemPalace and about as many as mem0, because the other two distil first and it
does not. And **the service is not one you must run**: all three can point at a
local embedder, which is exactly what this benchmark does — the same BGE-M3,
through the same endpoint, for all three. So the cost of these embeddings is
CPU in every case, and what the second row measures is not a bill but where the
embedder lives: inside the process that holds the index, or across a socket.

Two of the headline figures need a sentence each. **Retrieval at 25.7 ms
against 64.6 ms is real and mostly invisible**: a model reading those passages
takes about five seconds and does not care whether it was handed five hundred
tokens or five thousand, so end to end the three are indistinguishable and
retrieval is around one per cent of the wait. A quarter of mem0's figure and
two fifths of MemPalace's is an embedding call over HTTP that this harness
serves and they pay inside every search, so read the gap as architecture rather
than as a stopwatch reading. These retrieval figures are also roughly half what
this page carried a day earlier: the earlier ones were measured beside a spin
loop of our own, and the correction runs against us —
[docs/benchmarks.md](docs/benchmarks.md) has both halves. Where it counts is a memory system feeding
an agent's own context, adding its latency to a call that was happening anyway.
**Ingest at 356 s against 4,121 s is the one nothing hides** — an hour of
difference is an hour.

A system that asks a model to decide what a conversation *means* before storing
it pays for that on every ingest, cannot reproduce its own store, and cannot
return what the model chose not to write down. A system that stores the
evidence pays instead on every query, in the prompt it hands back. The two
cross at roughly **1,900 questions asked of a single conversation's memory**;
below that this is cheaper, and at LOCOMO's own density of twenty questions it
is cheaper by a factor of 31.

**Where this is behind.** mem0 leads temporal questions by about twenty points
(0.735 against 0.529) — reproducibly, across independent runs. It also hands
the reader fewer tokens, 1,017 against 1,511, because rewritten facts are
shorter than the passages they came from; that costs nothing in time here but
it is real money at volume. And holding the embedding model in-process costs
about 2 GB resident where a system calling out to an endpoint holds 177 MB and
a bill.

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

## Teaching An Agent To Use It

The CLI is the whole interface, so an agent needs to know when reaching for it
beats answering from what it already has. That judgement ships as a skill:

```bash
npx skills add pamin-labs/PaminMemory --skill pamin-memory
```

It installs to `./.agents/skills/pamin-memory` and symlinks into the paths the
individual agents read, so one install covers Claude Code, Codex, Cline, Amp and
the rest. `npx skills add pamin-labs/PaminMemory --list` shows what is there
before you take it.

The skill is about judgement rather than syntax — which of `search`, `read`,
`grep` and `neighbors` answers which kind of question, how to read the `why`
trace on a result, and the traps around the evidence filter. It is the only
skill this repository publishes. There is a second one for people working on
Påmin Memory itself, about measurement discipline, and it is marked internal so
it stays out of the way; `INSTALL_INTERNAL_SKILLS=1` reveals it.

## Any Language

Evidence is stored exactly as it arrives and is never translated. Translation would put a model on the write path, and it would break exact matching: after translation your own words no longer find your own memory.

Cross-language recall is handled by retrieval instead. Text is segmented with ICU before indexing, which covers languages that write without spaces, and a multilingual embedding model lets a query in one language reach a memory written in another.

```bash
pamin write --topic deploy "部署流水线运行在持续集成上面"
pamin search "how is the code deployed"   # finds it
```

## Architecture

One separation carries the design: **an authority that is written to, and a
projection that is read from.** PostgreSQL holds every fact the system is
accountable for — raw evidence, the bi-temporal ledger, the relationship graph.
The retrieval index holds nothing PostgreSQL cannot reproduce, so losing it
costs a reindex rather than a migration.

[docs/architecture.md](docs/architecture.md) has the diagram and the pieces
behind it: the ledger, the transactional outbox, the four recall channels and
how they are fused, the tiered cross-encoder, and what runs locally. The
decisions, their trade-offs, and the ones that reversed when they were
measured, are in [docs/adr/](docs/adr/).

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

Every figure comes from `pamin search` and `pamin write` themselves rather than
from the model or the index underneath them, on four cores. **The numbers and
the conditions they were taken under are in
[docs/measured.md](docs/measured.md)**; the comparison against other memory
systems, and what it holds fixed, is in
[docs/benchmarks.md](docs/benchmarks.md); the committed evidence behind both is
under [benchmarks/results/](benchmarks/results).

| | | measured on |
| --- | --- | --- |
| retrieval, one language | nDCG@10 **0.7359** | MIRACL Swahili dev, 131,924 passages |
| retrieval, query and answer in different languages | nDCG@10 0.6097 | XQuAD-R, 13,014 sentences |
| one `pamin search` over a socket | **25.7 ms** | LOCOMO, `fast` reranking |
| one `pamin search` as a whole CLI invocation | 251 ms | XQuAD-R, `fast` reranking |
| one `pamin write` | 30.1 ms | 2,400 memories, most of it the `fsync` |
| resident, one project | 2,088 MB | model and index inside the server |

Latency is a corpus and a tier before it is a number, which is why every row
above names both and why the matrix is on the other page.

Four findings belong in the summary rather than only in the detail, because
each of them cuts against this project:

**It is a tie, and reporting it as a win would be wrong.** At thirty passages
the three systems are 0.628, 0.623 and 0.583, and paired McNemar separates no
pair of them — p = 0.289, 1.000 and 0.396. Being first by five thousandths is
not being ahead.

**There is a floor under those comparisons, and it is worth more than they
are.** Running mem0 twice at the same settings on the same data gives 0.603 and
0.598 — but **41 of the 199 questions change answer between the two runs**. A
distillation performed by a model is not the same twice. Påmin Memory has no
such floor on the write side, because there is no model there: its
thirty-passage row reads a store the ten-passage row built, byte for byte.

**One of the two categories where a gap appears is withdrawn rather than
claimed.** Påmin Memory and MemPalace lead **adversarial** questions, and 74% of
that category asks about the wrong speaker while the answer key rewards
replying with the other speaker's content. mem0 answers "no record of that",
which for the question as asked is better, and is marked wrong for it. Scoring
high there means ignoring who said what, which is a defect in a memory product.

Two more are negative and stay published. The version ledger this project is
built around bought nothing on LOCOMO (p = 1.00). Where it does win — cutting
answers-with-a-superseded-fact from 28.6% to 10.0% on LongMemEval's
knowledge-update questions, p = 0.0005 — it is **not established to beat
writing the date into the passage text**, a free alternative that needs no
columns: 0.900 against 0.814 is p = 0.0703, and re-running it with five reads a
question returned the same p.

**The MIRACL row above is older than the harness that will check it.** Every
other figure on this page is produced by a test in this repository. That one
was not: four pages quoted it and nothing in the tree could run it, because it
came from a program that was never committed. The harness now exists — `cargo
test -p pamin-engine --test monolingual -- --ignored` — and until it has
reproduced the figure, the figure is a claim about a past run rather than
something you can check. [measured.md](docs/measured.md) says which rows that
covers.

**Above this, nothing is measured.** The largest corpus here is 131,924
documents. A million and beyond is untested — not projected, not extrapolated,
untested.

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
