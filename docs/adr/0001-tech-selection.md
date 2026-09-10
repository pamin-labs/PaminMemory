# 1. Technology selection

- Status: accepted
- Date: 2026-09-07

## Context

PaminMemory stores durable evidence, keeps a versioned ledger of how facts change, and retrieves context through several independent channels fused into one explainable result. Before writing code we had to settle the language, the authoritative store, the retrieval engine, how any human language gets tokenized, and how embeddings are produced.

Three criteria decided every question: **fast, cheap, accurate**. A decision that appears to improve all three without a cost has not been examined closely enough, so each choice below states which criterion pays.

One rule ran through all of it:

> **Delete components that overlap. Keep components that complement.**

## Decision

| Layer | Choice |
| --- | --- |
| Language | Rust `1.98.1`, 2024 edition |
| Async runtime | Tokio |
| Database driver | `sqlx` (pooled; no compile-time macros) |
| Migrations | `sqlx::migrate`, migrations listed in code |
| Authoritative store | PostgreSQL, bundled via `postgresql_embedded` |
| Retrieval engine | `zvec` (in-process, BM25 full-text and dense vectors) |
| Segmentation | `icu_segmenter` (ICU4X) |
| Language detection | `whatlang` |
| Embeddings | `fastembed` over ONNX Runtime, BGE-M3 with int8 weights by default |
| CLI | `clap` |

Nothing is hand-written where a mature crate already covers it. The migration runner comes from `sqlx` rather than being hand-rolled, and the same rule applies to argument parsing, configuration, and logging.

## Rationale

### Language: Rust

Three languages were considered seriously. C# was ruled out first: `zvec` publishes no C# SDK, and Lucene.NET is still at `4.8.0-beta` with its stable release stuck on Lucene 3.0.3, so both the retrieval engine and any local full-text fallback would have to be built.

Go became viable once `tantivy` left the design, because `zvec` publishes an official Go SDK. Rust still wins on three points:

- **Segmentation.** ICU4X is a pure Rust crate. Go has no equivalent and would need CGO into `icu4c`. Since segmentation is where multilingual accuracy comes from, this is not a peripheral dependency.
- **Embeddings.** `ort` is an ordinary crate. Go needs CGO, which costs easy cross-compilation and static linking, or `purego`, which its own documentation labels beta and which loads symbols at runtime, so the binary stops being self-contained.
- **Modelling.** An immutable version ledger, a `why[]` trace, and filter decisions are sum types with exhaustive matching. Go has no sum types, and this is the part of the codebase that carries the product's differentiator.

Rust's real costs are a cold build that compiles the whole dependency tree and a `target/` directory measured in gigabytes. Both are addressed by keeping the dependency tree deliberately thin, by splitting the workspace so the heavy crate rebuilds rarely, and by the budget gates described below — not by pretending they do not exist.

Speed did not decide this. The workload is I/O against PostgreSQL, index queries inside native code, and ONNX inference in C++; the host language is a small share of total latency. Distribution size did not decide it either: the bundled PostgreSQL, the ONNX Runtime, and the embedding model together dwarf the binary.

### Authoritative store: PostgreSQL

The authority layer is OLTP-shaped throughout: a `SKIP LOCKED` outbox, a write-contention protocol that claims topic locks in sorted order inside a transaction, a partial index that resolves the current topic state, foreign keys with cross-entity transactions, and point lookups by `(topic_id, version)`.

Columnar engines were considered and rejected for this layer. Apache DataFusion is a query engine and provides no storage authority; Lance and Parquet are columnar formats, and while Lance offers MVCC and an append-only transaction log, columnar systems write large immutable blocks, so a single-row update often rewrites a block. They would replace nothing — vectors still need `zvec`, transactions still need something else — while adding a very large dependency tree. Cost falls on **cheap** and **accurate**, both against.

PostgreSQL is also the shortest path to a distributed future rather than an obstacle to it, because the ecosystem of PostgreSQL-compatible distributed engines is the largest of any database. Four rules keep that path open from the first migration:

1. Every table carries `project_id` and is designed for sharding on it.
2. Only the portable SQL subset is used: `SKIP LOCKED`, `SELECT FOR UPDATE`, partial indexes, foreign keys, CTEs. A test rejects the rest.
3. No `LISTEN/NOTIFY`, no advisory locks, no PostgreSQL-only extensions.
4. UUID primary keys, never `SERIAL`.

Doing this now costs nothing. Doing it after a cloud tier exists is a migration.

PostgreSQL is bundled rather than brought by the user. `pamin init` provisions and hosts a local instance through `postgresql_embedded`, so there is no Docker and no configuration. This changes only the distribution mechanism; PostgreSQL remains the sole authority.

### Retrieval engine: one engine, `zvec`

`zvec` runs in-process and covers both channels we need from an index: BM25 full-text search and dense vectors, with write-ahead logging, per-field tokenizers, and index types that scale from memory to disk.

`tantivy` was evaluated for the lexical channel and rejected. It is a second BM25 inverted index next to the one `zvec` already provides — the overlap half of the rule. Choosing it would have meant a second index directory, a second rebuild path, and real compile cost for capability already present.

`zvec` is pre-1.0 and has made breaking changes between minor versions. Two mitigations make that acceptable, and both are executable rather than declared:

- It appears only in `pamin-index`, behind the `Projection` trait; `zvec` types must not reach `pamin-core`. Both halves are checked rather than reviewed: the trait is what `pamin-engine` holds, and `ci/budget.py` fails the build if the engine becomes reachable from a crate outside `pamin-index`, `pamin-engine`, and `pamin-cli`.
- The index is fully rebuildable from PostgreSQL, and `pamin reindex` is delivered and tested alongside it. A breaking upgrade is therefore a reindex, not a migration.

LanceDB and Qdrant Edge were also evaluated. LanceDB has the broadest tokenizer coverage available, but its Rust crate is pre-1.0 and pulls roughly sixty direct dependencies including Arrow and DataFusion, which is exactly the build cost this project is trying to avoid. Qdrant Edge runs in-process with on-device BM25, but its API is documented as beta.

### Fusion stays in our layer

`zvec` offers client-side hybrid search and reranking helpers. **We do not use them.**

The graph channel lives in PostgreSQL, where `zvec` cannot see it. Letting the engine pre-fuse the lexical and vector lists would produce an already-fused list that then has to be fused again with the graph list, double-weighting its members and destroying the contract that every result reports its rank in every channel it appeared in.

Recall engines return per-channel ranked lists. Reciprocal rank fusion runs in our layer, followed by post-fusion modifiers. This is a correctness requirement, not a preference.

`k = 10`, not the customary 60, and the two lexical channels carry half weight each. Both are measured on this project's evaluation corpus rather than taken from the literature: 60 came from fusing lists thousands of results deep, and each channel here proposes fifty, which the constant flattens to the point where rank barely counts. The lexical pair runs BM25 over the same text twice, so at equal weights their agreement with each other is counted as two votes against the vector and graph channels' one each. Correcting both takes cross-lingual nDCG@10 from 0.2041 to 0.3383 and costs nothing monolingual.

### Three recall channels, not seven

An earlier channel list had seven entries. Four were redundant, and two of those double-counted against modifiers the same design already applied after fusion:

- **Temporal** and **pinned/important** were already post-fusion modifiers. Running them as channels as well counted the same signal twice. "Facts valid at time T" is a filter over other channels, not an independent recall source.
- **Curated notes** and **page nodes** already enter the projection index. A separate channel queries the same data twice and splits one population into several, which dilutes results and forces the redundancy penalty to reason across populations.

What remains:

```text
recall channels (3)   lexical, vector, graph
document types        topic / span / page_node / note   (a filter)
post-fusion modifiers recency, importance and worth, source quality,
                      redundancy penalty
agentic primitives    grep, read by id, navigate, typed query
```

The projection holds one document per topic, carrying what that topic says now.
An earlier version of this decision held one per state, and that put a topic's
whole history into every channel's candidate budget: at the scale here -- a
million topics averaging a dozen or so versions -- a hundred million documents
stand in for seven million subjects, thirteen of every fourteen saying something
their topic no longer says. It also made the version-currentness modifier
necessary, to push down results the index should not have been returning. With
one document per topic both go away: history is read by version from the ledger
and is never ranked, so `search` returns current states only.

All three criteria improve: four fewer query groups per search, four fewer channels of code and index, and no double-weighted recency or importance.

### Segmentation: ICU4X in the application layer

Users write in any language in the world. `zvec` ships `standard` (UAX#29), `jieba`, `ngram`, ASCII folding, and a Snowball stemmer, which documents no handling for Japanese, Korean, Thai, Khmer, Lao, or Burmese. Those six fall back to `ngram`, which indexes them but matches across word boundaries and loses precision.

The fix is a segmenter, not a second search engine:

```text
source text (any language, stored verbatim)
  -> icu_segmenter        UAX#29 for most languages
                          dictionary for Chinese and Japanese
                          LSTM for Thai, Khmer, Lao, Burmese
  -> space-joined tokens  index input only; evidence is never rewritten
  -> zvec standard tokenizer -> BM25
queries pass through the same pipeline
```

This is the complement half of the rule: ICU4X segments, `zvec` retrieves, and neither does the other's job. It is pure Rust, so it adds no C++ build step, and it leaves the evidence layer untouched.

A second full-text field indexes the raw text with the `ngram` tokenizer, covering substrings that segmentation destroys: file paths, error codes, function names, configuration keys, and partial identifiers. Both fields are native `zvec` per-field configuration. The cost is roughly double the lexical index, paid by **cheap**, and whether it is worth paying is a question for the evaluation harness.

### Embeddings: profiles, and two different meanings of INT8

"INT8" names two operations whose costs differ by an order of magnitude, and conflating them is easy:

| | What it is | Measured cost | State |
| --- | --- | --- | --- |
| Model weight INT8 | ONNX weights quantized for CPU inference | 2.7–3.4x faster, under 0.5% MTEB | **On, by default** |
| Stored vector INT8 | Output embeddings stored as int8 rather than float32 | 1.5–3.5% loss, plus a calibration dataset | **Off, permanently** |

Weight quantization is a trade worth taking, and the default profile takes it. The registry publishes no quantized variant for multilingual E5, which is why the two E5 profiles still run full precision and why an earlier version of this decision recorded the trade as unavailable. It is available for BGE-M3, through a joint int8 export (`gpahal/bge-m3-onnx-int8`, MIT, exported from the MIT-licensed base model), and the difference is what makes that profile the default: 560 MB resident against the full-precision export's 2.2 GB, 35 ms a query, and 0.6550 cross-lingual nDCG@10 on this project's evaluation corpus against the full-precision 0.6720.

Stored vectors are float32. The original reasoning was about cost and benefit — a workspace of low millions of vectors makes the compression worth little, and the deterministic reranker has no cross-encoder to recover the accuracy it costs. At the scale this store now targets that reasoning would have expired, so the trade was measured rather than assumed.

It does not work in this engine. On 50,000 clustered 1024-dimensional vectors, an index built with `hnsw_with_quantize(..., Int8)` returns recall@10 of **0.000** against exact search, with or without the refiner — ten results per query, the right number, none of them the right ones. It does not error and nothing about the output looks wrong.

`enable_rotate`, which the engine's own benchmarks describe as what makes INT8 usable (Cohere-768 recall 92.87% unrotated against 94.01% rotated), is not exposed in the Rust binding at all. Whether that is the whole explanation is not established; what is established is that the configuration reachable from here is unusable. The refiner is likewise unavailable without quantization: on a full-precision index `is_using_refiner` fails the query outright rather than being ignored.

Revisit when the binding exposes rotation, or when the measurement above changes. Until then this is not a decision about compression being unworthy — it is that the compression on offer returns the wrong answers.

### The graph is the memory floor, and it just doubled

Quantizing stored vectors, if it worked, would shrink the payload and not the graph. The graph is the part that does not respond to it, which makes it the floor under everything else. At the size this store is built for — seven million documents in a project — an HNSW graph at `m = 16` is roughly 0.98 GiB per project, so a hundred projects is about **98 GiB of graph before a single vector is counted**.

Raising `m` to 32 for the recall measured above doubles that: roughly 196 GiB for the same hundred projects. That is a real cost and it is the right trade anyway, for a reason worth stating rather than assuming. Nothing puts hundreds of projects of this size in resident memory under *any* configuration — the fp32 payload alone is 2.7 TB, and the best case measured here, one-bit quantization that does not work in this engine, still leaves a floor in the hundreds of gigabytes. Protecting a factor of two on a budget already out of reach buys nothing, while a vector channel returning seven of every ten true neighbours is a live defect.

What it does change is when the disk-resident path stops being optional. Serving that many projects at that size means keeping cold indexes on disk and paging in the working set, and the graph doubling brings that forward rather than pushing it away. The engine exposes `IndexType::Diskann` and `IvfRabitq` for it, with two constraints to carry into that work: DiskANN is Linux x86-64 only, and `enable_mmap` is written into the manifest at creation and ignored when an existing collection is opened, so it cannot be turned on after the fact.

The embedding model is a profile, not a constant:

| Profile | Model | Dimensions | Resident | Per query | Cross-lingual nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| `speed` | `multilingual-e5-small` | 384 | 465 MB | 13 ms | — |
| `balanced` | `multilingual-e5-base` | 768 | 1.1 GB | 26 ms | 0.3383 |
| `accuracy` (default) | BGE-M3, int8 weights | 1024 | 560 MB | 35 ms | 0.6550 |

BGE-M3 is the default, reversing this decision's original position. That position rested on two claims, and the evaluation harness contradicted both. Its cost per query is not an order of magnitude higher — quantized weights put it at 35 ms against 26, and at 560 MB it is *smaller* resident than the model it replaces. And the sparse arm that was supposed to be its main increment is not: only the dense representation is kept, and the dense representation alone roughly doubles cross-lingual retrieval on our corpus while matching same-language retrieval exactly.

`multilingual-e5-small` is not the default because 384 dimensions is generally considered sufficient only when paired with a cross-encoder reranker, and our default reranker is deterministic and has none. EmbeddingGemma scores well and supports Matryoshka truncation, but is governed by the Gemma Terms of Use, whose restrictions must be passed to downstream users; that is not an acceptable burden to attach to an open-source default. The E5 family and BGE-M3 are Apache-2.0 or MIT, as is the int8 export.

Learned sparse retrieval such as SPLADE outperforms BM25 on most benchmarks but requires GPU inference, which is incompatible with a default install that needs no API key and no GPU. It stays a profile, not a default.

### No cross-encoder reranker, because it was measured and it did not help

A cross-encoder looked like the largest retrieval gain left. Published results put reranking at seven or eight points of nDCG@10, and the shape of our numbers seemed to invite it. Two permissively licensed multilingual rerankers were run against the evaluation corpus, and neither earned its cost:

| | Size | Per query | cross | mono | lexical |
| --- | --- | --- | --- | --- | --- |
| dense only, no reranking | — | — | **0.8299** | 0.9849 | **1.0000** |
| `gte-multilingual-reranker-base`, int8 | 325 MB | 245 ms | 0.7765 | **0.9908** | 0.9693 |
| `bge-reranker-v2-m3`, int8 | 544 MB | 403 ms | 0.8166 | 0.9821 | 0.9361 |

Reranking the dense top-30, nDCG@10, both models from ONNX exports of Apache-2.0 base models. Widening the shortlist to 50 made both worse and slower, not better: 0.7560 at 393 ms and 0.8136 at 683 ms.

The diagnostic matters more than the totals. On this corpus a reranker has nothing to recover: across all 137 queries, the relevant memory is already inside the top ten by dense retrieval alone — not one query has it sitting between rank 10 and rank 50 where reranking would pull it up. What is left is reordering inside the top ten, and on cross-lingual and lexical queries both cross-encoders order worse than BGE-M3's dense similarity does.

So this is not "reranking does not work". It is that a corpus of 210 memories does not put anything far enough down for a reranker to earn 245 ms, and the models cost accuracy in the groups this project cares most about. **Revisit when the opportunity is real** — a corpus where relevant memories fall below the retrieval cut, which is what millions of memories in one project would produce and what this one cannot simulate. The measurement to run first is the diagnostic above, not the nDCG: if nothing is below the cut, there is nothing to rerank.

Licensing is no longer the blocker it was. The embedding library's own four rerankers remain unusable — two English-only, one CC-BY-NC-4.0, and one carrying no licence at all — but its user-defined loader takes any ONNX, and permissively licensed multilingual exports exist. That path is open whenever the measurement turns.

### Engineering budgets

Retrieval quality is governed by numeric gates. Engineering cost gets the same treatment, because otherwise it drifts silently — and an earlier iteration of this decision would have added compile cost for capability the project already had.

CI measures binary size, dependency count, cold build time, incremental check time, and `target/` size. The first two are gated. Build times are reported rather than gated: shared runners vary enough that a wall-clock gate would fail at random, and a flaky gate teaches people to ignore gates.

Exceeding a budget is a trade to record in the pull request, not drift to accept.

## Consequences

- A cold build compiles the whole dependency tree, and `target/` is large. The dependency tree is kept thin deliberately: no web or metrics stack until something calls it, and no compile-time SQL macros.

  This originally read as a decision against `sqlx`, on the grounds of its compile-time macros. That was wrong twice over. The macros are one feature, off by default, and `sqlx` was already in the tree underneath `postgresql_embedded`, so refusing it bought a second driver rather than none: `tokio-postgres` and `refinery` alongside it, with a duplicate SHA-2 and a duplicate SCRAM implementation compiled into every binary. Moving the store onto `sqlx` and dropping both took the dependency count from 434 to 388. The `macros` feature stays off, which is the part of the original reasoning that survives.
- `zvec` is pre-1.0, so a breaking upgrade will require a reindex. `pamin reindex` exists from the first release precisely so this stays routine.
- Building `zvec` downloads a prebuilt native library, and ONNX Runtime does the same. Neither compiles C++ locally, but both require network access at build time and a measure of supply-chain trust.
- Two full-text fields roughly double the lexical index. This is a measured trade, revisited when the evaluation harness exists.
- Fusion in our own layer means more code than calling an engine helper. That code is the explainability contract, so it is the product rather than overhead.
