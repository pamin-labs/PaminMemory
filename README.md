<img src="assets/icon.png" alt="Påmin Memory" width="96">

# Påmin Memory

Påmin Memory (Pamin Memory) is universal memory for AI agents, coding assistants, research tools, and knowledge-heavy applications.

It is designed to turn durable evidence into versioned knowledge that agents can retrieve through structure, meaning, relationships, and time. Instead of treating memory as a pile of extracted snippets, PaminMemory keeps the source trail intact, tracks how facts evolve, and explains why each piece of context was selected.

> **Early days.** The foundation runs end to end and is worth trying, but most of the system described below is not built yet. See [Status](#status).

## What It Does

- Preserves raw evidence and source spans as the authority behind memory.
- Tracks versioned memories so current, stale, contradicted, and historical facts can be separated.
- Combines page/tree structure, semantic recall, lexical matching, temporal relationships, and reranking.
- Builds explainable context from the same evidence ledger rather than opaque one-off summaries.
- Prioritizes local-first operation so developers can inspect and control their memory stack.

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

`pamin stop` shuts the local database down. It is deliberately left running between commands so an agent invoking the CLI repeatedly does not pay startup each time.

Every command, its options, and its JSON shape are in [docs/cli.md](docs/cli.md).

## Any Language

Evidence is stored exactly as it arrives and is never translated. Translation would put a model on the write path, and it would break exact matching: after translation your own words no longer find your own memory.

Cross-language recall is handled by retrieval instead. Text is segmented with ICU before indexing, which covers languages that write without spaces, and a multilingual embedding model lets a query in one language reach a memory written in another.

```bash
pamin write --topic deploy "部署流水线运行在持续集成上面"
pamin search "how is the code deployed"   # finds it
```

## Stack

Two engines, each doing what it is best at:

```text
PostgreSQL   authority: evidence, the version ledger, bi-temporal validity,
             the outbox, transactions and concurrency control
zvec         projection: BM25 full-text and dense vectors, in-process,
             rebuildable from PostgreSQL at any time
```

PostgreSQL is bundled rather than something you install. The projection index holds nothing PostgreSQL cannot reproduce, so `pamin reindex` rebuilds it from scratch — which is also what keeps the retrieval engine replaceable.

Embeddings run locally through ONNX Runtime. The default install makes no network call at query time and needs no API key.

Retrieval draws on three channels — segmented lexical, n-gram lexical, and vector — and fuses them here rather than inside the index, so every result can report the rank it held in each channel:

```bash
$ pamin search "deployment pipeline" --json | jq '.hits[0].why'
[ { "kind": "channel", "channel": "lexical_segmented", "rank": 1, ... },
  { "kind": "channel", "channel": "vector",            "rank": 1, ... },
  { "kind": "modifier", "modifier": "importance",      "factor": 1.0 } ]
```

Design decisions and their trade-offs are recorded in [docs/adr/](docs/adr/).

## Status

This is an early foundation, not a finished product.

**Working:** the version ledger with bi-temporal fields and soft deletes; bundled PostgreSQL; the sensory filter, which records why content was held without ever discarding evidence; multilingual segmentation and language detection; both lexical channels plus the vector channel with reciprocal rank fusion and explainable results; rebuilding the index from PostgreSQL.

**Not built yet:** the relationship graph and its recall channel; the cascade worker and outbox consumption; source ingestion and page trees; curated notes and the session brief; passive optimization and forgetting; the REST and MCP surfaces; the evaluation harness that will settle the defaults this version guesses at.

## Development

```bash
cargo test --workspace                  # fast; no database, no model
cargo test --workspace -- --ignored     # provisions postgres, downloads models
```

## Maintainer Notes

This repository can optionally include private maintainer notes through the `internal-docs` submodule. Public users do not need that submodule to use or follow the open-source project.

See [docs/internal-docs.md](docs/internal-docs.md) for the maintainer workflow.
