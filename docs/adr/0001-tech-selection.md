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

`k = 10`, not the customary 60, and the two lexical channels carry a quarter weight each. Both are measured rather than taken from the literature: 60 came from fusing lists thousands of results deep, and each channel here proposes fifty, which the constant flattens to the point where rank barely counts. The lexical pair runs BM25 over the same text twice, so at equal weights their agreement with each other is counted as two votes against the vector and graph channels' one each.

The weight was swept across both evaluation corpora — the one written for this project and XQuAD-R — at four values of `k`. Equal weighting is not a trade at any of them: it scores worse than half on every group of both corpora, cross-lingual and same-language alike. A quarter beats a half on seven of the eight measures the two corpora report, costing 0.033 of same-language ranking on the external corpus and buying 0.139 and 0.067 of cross-lingual nDCG@10 with the monolingual and lexical groups unmoved. Zero scores higher again cross-lingually and is refused: it takes the monolingual group off 0.9940 and the lexical group off its ceiling, which is the one thing the n-gram channel exists for, and it would leave both lexical channels contributing nothing.

What the sweep cannot settle is that the ideal weight is not the same for every query — near zero when a query and its answer are in different languages, and a half when they are not. One constant serves both by compromise. Making it a function of the query is recorded as an open question rather than guessed at here.

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

### Segment size is the whole vector-maintenance policy

A vector index needs a rule for when to build a graph, and the obvious form of that rule is a threshold: build once some number of documents are unindexed. This project shipped one — a hundred thousand — and it never fired, because a project reaching a hundred thousand unindexed documents is not the case that needs the graph. A threshold is a constant asked to be right at every size.

The engine makes one number do the job instead. Documents land in the segment being written and are searched by scanning it; the segment seals at a configured size, and only a sealed segment gets a graph. So the size decides both what a query scans and what a build costs, and there is no separate question of when to build — the answer is "whenever a segment sealed without one".

Scanning is not a fallback here, it is the faster thing to do while a segment is small. Measured on the default profile, one graph against an exhaustive scan:

| Documents | Scan | Graph | Build | Agreement |
| --- | --- | --- | --- | --- |
| 1,000 | 0.57 ms | 0.66 ms | 0.3 s | 1.0000 |
| 10,000 | 2.70 ms | 3.10 ms | 8.1 s | 1.0000 |
| 25,000 | 5.78 ms | 5.76 ms | 40.9 s | 0.9830 |
| 50,000 | 20.85 ms | 10.08 ms | 124.0 s | 0.9540 |
| 100,000 | 39.58 ms | 11.24 ms | 325.8 s | 0.8920 |

The crossover is near 25,000, and the build cost is superlinear where the query cost is not. So a segment holds a quarter of the collection, floored at 2,000 so a new project is one segment rather than a hundred tiny ones and capped at 250,000 so no single build is ever worth more than about twenty minutes. A project past a million documents therefore runs more than four segments rather than larger ones, which is the right way round.

The cost of segmenting at all is that BM25 statistics are per segment, so a term's rarity is measured against a segment rather than the project. It is small and it was measured, not assumed: six segments against one over the same 13,014 sentences moved same-language nDCG@10 from 0.8558 to 0.8517 and recall@50 from 0.9639 to 0.9655, while a query went from 208 ms to 63 ms.

**`pamin reindex` is the entry point for recomputing this.** The size is written into the collection's manifest when it is created, from the document count at that moment — which for a project that grows from nothing is the floor. A project that has since grown by orders of magnitude keeps the size it was created with until it is rebuilt, and rebuilding is what recomputes it. That is a deliberate consequence of the size living in the manifest rather than a gap: changing it in place would mean resealing every segment, which is a rebuild under another name.

The embedding model is a profile, not a constant:

| Profile | Model | Dimensions | Resident | Per query | Cross-lingual nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| `speed` | `multilingual-e5-small` | 384 | 465 MB | 13 ms | — |
| `balanced` | `multilingual-e5-base` | 768 | 1.1 GB | 26 ms | 0.3383 |
| `accuracy` (default) | BGE-M3, int8 weights | 1024 | 560 MB | 35 ms | 0.6550 |

BGE-M3 is the default, reversing this decision's original position. That position rested on two claims, and the evaluation harness contradicted both. Its cost per query is not an order of magnitude higher — quantized weights put it at 35 ms against 26, and at 560 MB it is *smaller* resident than the model it replaces. And the sparse arm that was supposed to be its main increment is not: only the dense representation is kept, and the dense representation alone roughly doubles cross-lingual retrieval on our corpus while matching same-language retrieval exactly.

`multilingual-e5-small` is not the default because 384 dimensions is generally considered sufficient only when paired with a cross-encoder reranker, and the one this project ships reorders only the candidates the lexical channels missed. It is not there to rescue a weaker embedding across the board, and cannot be relied on to. EmbeddingGemma scores well and supports Matryoshka truncation, but is governed by the Gemma Terms of Use, whose restrictions must be passed to downstream users; that is not an acceptable burden to attach to an open-source default. The E5 family and BGE-M3 are Apache-2.0 or MIT, as is the int8 export.

Learned sparse retrieval such as SPLADE outperforms BM25 on most benchmarks but requires GPU inference, which is incompatible with a default install that needs no API key and no GPU. It stays a profile, not a default.

### A cross-encoder reranker, once the opportunity was real

An earlier version of this decision recorded that reranking was measured and did not help. That measurement stands; its premise does not. It ran on this project's own 210-memory corpus, where the diagnostic said plainly that there was nothing to recover: across all 137 queries the relevant memory was already inside the top ten, so a second pass could only reorder what was already right, and both models reordered it worse. The conclusion drawn from it — *revisit when the opportunity is real* — named the measurement to run first, and an external corpus supplied it.

On 13,014 sentences in eleven languages, 3,849 relevant sentences sit between rank 10 and rank 50, across 1,090 of 1,190 queries. That is the opportunity the small corpus could not produce, and in it reranking is the largest single retrieval gain measured in this repository. (It was 4,845 across 1,149 queries when this was first run. Correcting the fusion weights moved several hundred of them up into the top ten, which is the right direction and leaves the point standing: the space a reranker works in is still most of the corpus.)

| Tier | Loads | Per query | Cross-lingual nDCG@10 | Same-language |
| --- | --- | --- | --- | --- |
| `off` | nothing | — | — | — |
| `fast` (default) | 113 MB | 151 ms | **+0.0595** | unchanged |
| `accurate` | 570 MB | 1795 ms | **+0.0852** | unchanged |

`fast` is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, a 21M-parameter distilled multilingual MiniLM; `accurate` is `onnx-community/bge-reranker-v2-m3-ONNX`, XLM-RoBERTa-large at 303M. Both are quantized ONNX behind the library's user-defined loader, fetched on first use into the same cache as the embedding model.

**The same-language column is unchanged by construction, not by luck.** Every cross-encoder tried improves cross-lingual ranking and damages same-language ranking by about as much — three models across two orders of magnitude of size, −0.0403 to −0.2109. Fusion is already good at placing a memory that shares words with the query, and a second pass reorders it worse. So the pass is confined to the candidates no lexical channel found, and they are written back into the positions they already held. Unconfined, two of those models score +0.1698/−0.2109 and +0.1678/−0.0806; confined, the same-language column cannot move at all.

The rule was a language comparison first, since "written in another language" is what the case really is. The two rules pick the same candidates — they agree on 93% of a shortlist and score within 0.002 — but the language test needs the query's language, and that is exactly what a detector will not commit to for a short query: `detect_language` returns nothing for "how does deployment work". A rule that quietly does nothing on the commonest shape of query is worse than a slightly different rule.

**Smaller is not worse.** `accurate` is fourteen times larger and twelve times slower than `fast` for 0.026 more. That reproduces *Shallow Cross-Encoders for Low-Latency Retrieval* (arXiv 2403.20222) without having read it first: under a latency budget the shallow model wins, because the budget buys more candidates. `fast` is therefore the default, and a workspace whose memories are all in one language should set `off` — the candidates the lexical channels miss are overwhelmingly the ones in another language.

A score depends on the query as well as the memory, so a resident process remembers the pairs it has computed: a repeated search measured 69.6 ms the first time and 0.0 ms the second, for the same ordering. Four thousand scores, about a quarter of a megabyte. It does nothing for a query never asked before, which is most of them; it is worth its quarter megabyte because agents retry, widen a limit, and ask again after writing. Without `pamin serve` there is no process to keep it in.

**There is no compilation trick left in the runtime.** Sorting candidates by length before batching and using batches of eight rather than sixteen took the same work from 191 ms to 151, because a batch is padded to its longest member. Against that, the export format is worth at most 1.45x on identical weights, fp16 is slower than fp32 on a CPU, and the session already runs every core at the highest graph optimization level. The measured 9.75 ms a pair is what twelve transformer layers on four cores cost.

The published answers to this latency all change the architecture instead, and both are deferred on a missing export rather than on a licence or a doubt:

| Route | Worth | Blocked on | Revisit when |
| --- | --- | --- | --- |
| Precomputed document layers (PreTTR, arXiv 2004.14255) | ~6x on this shape: twelve layers over a ten-token query and a hundred-token memory is 1320 layer-tokens; caching the memory's first eleven layers makes it 220. 151 ms becomes roughly 25. Storage is hot set x tokens x width: 384 MB for ten thousand memories | An export split into two halves. It is an export-time job, not runtime graph surgery — `ort` selects only among declared graph outputs and no ONNX graph-editing crate is in the tree | A split export of `mmarco-mMiniLMv2` exists **and** the score cache's hit rate shows the hot set is actually small, which is the same evidence that decides whether it is worth its storage |
| Late interaction (ColBERT) | Moves the cost to write time, where the cascade already runs a forward pass per memory; query time becomes MaxSim. At 64–128 dimensions int8 that is 6.4–12.8 KB a memory, the same order as the vector index | No usable model. `colbert-xm` is MIT and multilingual but has no ONNX export; the exports that exist are English-only; the multilingual one with an export is CC-BY-NC-4.0. BGE-M3's own ColBERT head is 1024-dimensional, 100 KB a memory int8, ten times the whole index | An ONNX export of `colbert-xm` exists and measures competitively cross-lingual |

Both routes rest on premises this project has not measured — that the hot set is small, that write-time cost is cheap — and the triggers are written to test the premise before the work.

Licensing was the blocker when this was first examined and is no longer. The embedding library's own four rerankers remain unusable — two English-only, one CC-BY-NC-4.0, and one carrying no licence at all — but its user-defined loader takes any ONNX, which is the path both tiers take.

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
