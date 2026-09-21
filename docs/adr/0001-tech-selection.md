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

LanceDB and Qdrant Edge were also evaluated. LanceDB has the broadest tokenizer coverage available, but its Rust crate is pre-1.0 and pulls roughly sixty direct dependencies including Arrow and DataFusion, which is exactly the build cost Påmin Memory is trying to avoid. Qdrant Edge runs in-process with on-device BM25, but its API is documented as beta.

### Fusion stays in our layer

`zvec` offers client-side hybrid search and reranking helpers. **We do not use them.**

The graph channel lives in PostgreSQL, where `zvec` cannot see it. Letting the engine pre-fuse the lexical and vector lists would produce an already-fused list that then has to be fused again with the graph list, double-weighting its members and destroying the contract that every result reports its rank in every channel it appeared in.

Recall engines return per-channel ranked lists. Reciprocal rank fusion runs in our layer, followed by post-fusion modifiers. This is a correctness requirement, not a preference.

`k = 10`, not the customary 60, and the two lexical channels carry a quarter weight each. Both are measured rather than taken from the literature: 60 came from fusing lists thousands of results deep, and each channel here proposes fifty, which the constant flattens to the point where rank barely counts. The lexical pair runs BM25 over the same text twice, so at equal weights their agreement with each other is counted as two votes against the vector and graph channels' one each.

The weight was swept across both evaluation corpora — the one written for Påmin Memory and XQuAD-R — at four values of `k`. Equal weighting is not a trade at any of them: it scores worse than half on every group of both corpora, cross-lingual and same-language alike. A quarter beats a half on seven of the eight measures the two corpora report, costing 0.033 of same-language ranking on the external corpus and buying 0.139 and 0.067 of cross-lingual nDCG@10 with the monolingual and lexical groups unmoved. Zero scores higher again cross-lingually and is refused: it takes the monolingual group off 0.9940 and the lexical group off its ceiling, which is the one thing the n-gram channel exists for, and it would leave both lexical channels contributing nothing.

The ideal weight is not the same for every query — near zero when a query and its answer are in different languages, and a half when they are not. One constant serves both by compromise, and making it a function of the query was tried.

The signal was the lexical channels themselves. They match on shared tokens, so whichever language they put the most of their score behind is, empirically, the language the query was asked in — no detector, which matters, because detection declines on "how does deployment work" and a rule that needed it would be absent on exactly the short queries an agent asks. A candidate in any other language then had its lexical contribution scaled down after fusion, by the fraction in the `xling` column below. `1.00` is the rule switched off.

| lexical | xling | ours: cross | ours: mono | XQuAD-R: cross | XQuAD-R: same |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 1.00 | 0.7223 | 0.9940 | 0.5722 | 0.8033 |
| 0.25 | 0.50 | 0.7234 | 0.9940 | 0.5750 | 0.8000 |
| 0.25 | 0.25 | 0.7201 | 0.9940 | 0.5760 | 0.7914 |
| 0.25 | 0.00 | 0.7181 | 0.9821 | 0.5759 | 0.7895 |
| 0.50 | 0.00 | 0.6444 | 0.9708 | 0.4649 | 0.8145 |

No setting clears the bar the reranker had to clear — cross-lingual up, same-language not down. The one cell that clears it on Påmin Memory's corpus, `0.25 / 0.50`, buys 0.0011 there, which on 137 queries is one of them, and on XQuAD-R the same setting costs 0.0033 of same-language for 0.0028 of cross-lingual. It fails in two separate ways, and both are worth recording.

**Cross-language lexical hits are not noise.** If they were, removing them could only help the cross-lingual group; on Påmin Memory's corpus it falls, 0.7223 to 0.7181. What a query shares with an answer in another language is proper nouns, numbers and borrowed technical terms — which is signal, and the only lexical signal that crosses a language boundary at all.

**A query's language cannot be read off its own lexical hits.** Same-language ranking falls at every setting on XQuAD-R, and it should not move at all if the rule only ever fired across a boundary. That corpus isolates the cause: every sentence in it carries the dataset's own language label, so the candidate side is ground truth and the inference is the only thing left to be wrong. It is wrong often enough to cost more than the rule buys, and it is worst exactly where the rule was aimed — eleven parallel translations of one passage give the ten wrong languages ten chances to outweigh the right one.

And the trade the constant exists to avoid does not open up. Half weight with cross-language contributions removed entirely scores 0.4649 cross-lingual on XQuAD-R, against 0.5722 for a quarter with the rule switched off.

So the weight stays a constant and none of this was kept. What would change the answer is a different signal for the query's language — one that does not come from the channel it is being used to correct.

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
| Stored vector INT8 | Output embeddings stored as int8 rather than float32 | 1.5–3.5% loss, plus a calibration dataset | **Off, until the binding exposes rotation** |

Weight quantization is a trade worth taking, and the default profile takes it. The registry publishes no quantized variant for multilingual E5, which is why the two E5 profiles still run full precision and why an earlier version of this decision recorded the trade as unavailable. It is available for BGE-M3, through a joint int8 export (`gpahal/bge-m3-onnx-int8`, MIT, exported from the MIT-licensed base model), and the difference is what makes that profile the default: 560 MB resident against the full-precision export's 2.2 GB, 35 ms a query, and 0.6550 cross-lingual nDCG@10 on Påmin Memory's evaluation corpus against the full-precision 0.6720.

**The joint export has a third cost, and it took a while to find.** On this export a text's vector depends on what else is in its batch. Against the same text embedded alone: cosine 0.9816 with a shorter neighbour in the batch, 0.9859 with a longer one, and the two neighbours disagree with each other at 0.9805. A batch of one is byte-identical to a single call, so it is the presence of a neighbour rather than the batching API, and it is not fastembed's Rust code either — the tokenizer pads to the batch's longest member, so a text that *is* the longest gets byte-identical ids and mask either way, and the mask is passed to the session. Only the batch dimension differs, which puts it in the export or the runtime's INT8 kernels. `speed` and `balanced` return byte-identical vectors batched or alone.

The consequence was not accuracy. It was reproducibility: `reindex` embedded in batches of 256 and the cascade embeds one document at a time, so a rebuild did not reproduce the index it replaced, and a document's vector depended on which other documents happened to be in flight beside it. So the joint export now runs one text at a time, which costs the batching win on this profile — thirty-two texts together take 190 ms against 409 ms one at a time, so `reindex` is roughly twice the wall clock here.

**Measured, because "it changes the vectors" and "it changes the answers" are different claims.** Re-running the model-alone arm of the cross-lingual harness on deterministic vectors returns 0.6335 cross-lingual and 0.6787 same-language nDCG@10, against 0.6351 and 0.6763 on the batch-perturbed ones — inside the run-to-run spread this harness already shows, and either side of the 0.6338 / 0.6748 recorded before. So the perturbation is systematic enough to leave the ranking alone, which is why it survived this long: nothing downstream looked wrong. It is fixed for determinism, not for quality, and the distinction belongs in the record.

Stored vectors are float32. The original reasoning was about cost and benefit — a workspace of low millions of vectors makes the compression worth little, and the deterministic reranker has no cross-encoder to recover the accuracy it costs. At the scale this store now targets that reasoning would have expired, so the trade was measured rather than assumed.

It does not work in this engine. On 50,000 clustered 1024-dimensional vectors, an index built with `hnsw_with_quantize(..., Int8)` returns recall@10 of **0.000** against exact search, with or without the refiner — ten results per query, the right number, none of them the right ones. It does not error and nothing about the output looks wrong.

`enable_rotate`, which the engine's own benchmarks describe as what makes INT8 usable (Cohere-768 recall 92.87% unrotated against 94.01% rotated), was not exposed in the Rust binding at all when this was written. **It is now, and that was this row's trigger.** `zvec-rust` 0.7.2 adds `IndexParams::quantizer_enable_rotate` and `set_quantizer_enable_rotate`, and `QuantizeType` carries `Rabitq` beside `Fp16`, `Int8` and `Int4` — verified against the crate source rather than a release note, and absent from 0.7.0, which is the version the 0.000 result below was taken on. Rotation defaults to off, so it has to be asked for. What that changes is the shape of the work: quantization is reachable on the same HNSW graph through `hnsw_with_quantize`, without moving to an IVF index and invalidating the parameter table and recall floor this section rests on. What it does not change is the gate — the 0.000 was silent, so a recall run and an end-to-end nDCG run on a real corpus both have to clear before any of it ships. Whether that is the whole explanation is not established; what is established is that the configuration reachable from here is unusable. The refiner is likewise unavailable without quantization: on a full-precision index `is_using_refiner` fails the query outright rather than being ignored.

Revisit when the binding exposes rotation, or when the measurement above changes. Until then this is not a decision about compression being unworthy — it is that the compression on offer returns the wrong answers.

### The graph is the memory floor, and it just doubled

Quantizing stored vectors, if it worked, would shrink the payload and not the graph. The graph is the part that does not respond to it, which makes it the floor under everything else. At the size this store is built for — seven million documents in a project — an HNSW graph at `m = 16` is roughly 0.98 GiB per project, so a hundred projects is about **98 GiB of graph before a single vector is counted**.

Raising `m` to 32 for the recall measured above doubles that: roughly 196 GiB for the same hundred projects. That is a real cost and it is the right trade anyway, for a reason worth stating rather than assuming. Nothing puts hundreds of projects of this size in resident memory under *any* configuration — the fp32 payload alone is 2.7 TB, and the best case measured here, one-bit quantization that does not work in this engine, still leaves a floor in the hundreds of gigabytes. Protecting a factor of two on a budget already out of reach buys nothing, while a vector channel returning seven of every ten true neighbours is a live defect.

What it does change is when the disk-resident path stops being optional. Serving that many projects at that size means keeping cold indexes on disk and paging in the working set, and the graph doubling brings that forward rather than pushing it away. The engine exposes `IndexType::Diskann` and `IvfRabitq` for it, with two constraints to carry into that work: DiskANN is Linux x86-64 only, and `enable_mmap` is written into the manifest at creation and ignored when an existing collection is opened, so it cannot be turned on after the fact.

### Segment size is the whole vector-maintenance policy

A vector index needs a rule for when to build a graph, and the obvious form of that rule is a threshold: build once some number of documents are unindexed. Påmin Memory shipped one — a hundred thousand — and it never fired, because a project reaching a hundred thousand unindexed documents is not the case that needs the graph. A threshold is a constant asked to be right at every size.

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

**And the rule above was never implemented.** "Whenever a segment has sealed without a graph" is the policy; what the cascade actually asked was whether the index had spread across more than 256 files, and it queued one job for both answers. That was sound while every write flushed the index — files grew about two a write, so the budget was reached every sixty or so and a sealed segment never waited long. The release that moved the flush to the server ended it: 136 files after three thousand writes, so the trigger moved from every sixty writes to roughly every 5,650, and `cascade.rs` kept the paragraph explaining why the graph needed no condition of its own.

Measured rather than reasoned, through `drain_cascade` on a fresh project of twelve memories with the budget lowered: **`vector_index_completeness` is 0.0 under the old condition and 1.0 under the new one.** Not a lagging graph — no graph at all, the vector channel answering entirely by exhaustive scan, with the document count confirming the twelve were indexed so the reading is not of an empty collection. That is the second time this project has shipped a vector channel with no graph beneath it, and the first time was also invisible because an exhaustive scan returns the right neighbours, only slower.

The graph now has its own condition, on the **unindexed remainder** rather than on the project: `vector_index_lags`, firing at 25,000 documents outside the graph. The value is the crossover in the table above, where a scan and a graph cost the same (5.78 ms against 5.76) — below it the scan is the faster of the two and a build would be work spent to go slower. A remainder is the right thing to bound because it is what a query scans; the threshold this project shipped before and never fired was on how much had ever been written, which says nothing about how much is outside the graph. Compaction keeps the file budget, which is its own resource and its own cost model.

**What this leaves open, and how to settle it.** Not "the benchmarks measured a graph and the product did not", which was the first guess and is not what the harnesses do: neither `crosslingual.rs` nor the `pamin` benchmark arm calls `reindex`: both ingest and then drain the cascade, the same path a user takes. What separates them is volume. `MAX_FILES`' own note names an import as one of the cases that still reaches the file budget, so an import of thirteen thousand sentences crosses 256 files and gets a graph from the old condition after all, while a workspace written into a few hundred times through a server never came close. So the corpora behind the published figures were probably indexed, and the case that certainly was not is the one a person actually has.

Probably is not measured, and the gap is per figure rather than per page: each published latency, throughput and resident-memory number has to be checked against how much its corpus wrote before it is quoted again. What can be said without checking is that a graph built by crossing a file budget is built at whatever moment the crossing happened, so the documents written after the last crossing were outside it — the figures describe a partly-indexed collection of unknown proportion, which is the quantity `vector_index_completeness` now reports and every arm should record.

The assertion is in the tree (`crates/pamin-engine/tests/graphupkeep.rs`), and it fails on the code that preceded it — verified by reverting the condition and re-running rather than by argument.


The embedding model is a profile, not a constant:

| Profile | Model | Dimensions | Resident | Per query | Cross-lingual nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| `speed` | `multilingual-e5-small` | 384 | 465 MB | 13 ms | — |
| `balanced` | `multilingual-e5-base` | 768 | 1.1 GB | 26 ms | 0.3383 |
| `accuracy` (default) | BGE-M3, int8 weights | 1024 | 560 MB | 35 ms | 0.6550 |

BGE-M3 is the default, reversing this decision's original position. That position rested on two claims, and the evaluation harness contradicted both. Its cost per query is not an order of magnitude higher — quantized weights put it at 35 ms against 26, and at 560 MB it is *smaller* resident than the model it replaces. And the sparse arm that was supposed to be its main increment is not: only the dense representation is kept, and the dense representation alone roughly doubles cross-lingual retrieval on our corpus while matching same-language retrieval exactly.

`multilingual-e5-small` is not the default because 384 dimensions is generally considered sufficient only when paired with a cross-encoder reranker, and the one Påmin Memory ships reorders only the candidates the lexical channels missed. It is not there to rescue a weaker embedding across the board, and cannot be relied on to. EmbeddingGemma scores well and supports Matryoshka truncation, but is governed by the Gemma Terms of Use, whose restrictions must be passed to downstream users; that is not an acceptable burden to attach to an open-source default. The E5 family and BGE-M3 are Apache-2.0 or MIT, as is the int8 export.

Learned sparse retrieval such as SPLADE outperforms BM25 on most benchmarks but requires GPU inference, which is incompatible with a default install that needs no API key and no GPU. It stays a profile, not a default.

### A cross-encoder reranker, once the opportunity was real

An earlier version of this decision recorded that reranking was measured and did not help. That measurement stands; its premise does not. It ran on Påmin Memory's own 210-memory corpus, where the diagnostic said plainly that there was nothing to recover: across all 137 queries the relevant memory was already inside the top ten, so a second pass could only reorder what was already right, and both models reordered it worse. The conclusion drawn from it — *revisit when the opportunity is real* — named the measurement to run first, and an external corpus supplied it.

On 13,014 sentences in eleven languages, 3,813 relevant sentences sit between rank 10 and rank 50, across 1,088 of 1,190 queries. That is the opportunity the small corpus could not produce, and in it reranking is the largest single retrieval gain measured in this repository. (It was 4,845 across 1,149 queries when this was first run. Correcting the fusion weights moved several hundred of them up into the top ten, which is the right direction and leaves the point standing: the space a reranker works in is still most of the corpus.)

| Tier | Loads | A search costs | Of which reranking | Cross-lingual nDCG@10 | Same-language |
| --- | --- | --- | --- | --- | --- |
| `off` | nothing | 53 ms | — | — | — |
| `fast` (default) | 113 MB | 264 ms | 211 ms | **+0.0381** | **−0.0053** |
| `accurate` | 570 MB | 1001 ms | 948 ms | **+0.0448** | **+0.0017** |

Measured through `Engine::search_reranked`, the entry point `pamin search`
calls, with `TIERS=1` on the cross-lingual harness over all 1,190 queries. The
first version of this table came from a scratch program that reordered a dumped
shortlist with its own copy of the pipeline, and overstated both gains by about
half — which is the argument for measuring the product rather than a model of
it.

**The latency in this row is a correction, and the figure it corrects was
itself a correction.** This table recorded 204 ms and 508 ms, the second of
those having replaced an earlier 1795 ms. Re-running the same harness on the
same machine, the same binary and the same workspace returns 264 ms and 1001
ms: `accurate` costs about twice what was recorded, and the reranking pass
948 ms rather than 469. So the first figure was too high, the correction was
too low, and neither was checked by running it again.

Where to look, offered as a lead rather than a cause: the pass is confined to
the candidates no lexical channel found, so its cost moves with the fusion that
decides which those are, and the lexical pair's weight was changed from 0.5 to
0.25 in the same round. A figure that was not re-derived after that change
would be a figure for a different shortlist.

The quality columns reproduce and the latency does not, which is worth saying
separately. Cross-lingual `off` returns 0.5705 exactly; the gains come back as
+0.0381 and +0.0448 against the +0.0375 and +0.0458 recorded, and the counts
below rank ten as 3,813 against 3,849. A thousandth is not nothing: it means
these numbers are not bit-identical across runs the way the MIRACL figures are,
and the index has been served and maintained between them.

`fast` is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, a 21M-parameter distilled multilingual MiniLM; `accurate` is `onnx-community/bge-reranker-v2-m3-ONNX`, XLM-RoBERTa-large at 303M. Both are quantized ONNX behind the library's user-defined loader, fetched on first use into the same cache as the embedding model.

**The same-language column is nearly, but not exactly, unchanged.** Every cross-encoder tried improves cross-lingual ranking and damages same-language ranking by about as much — three models across two orders of magnitude of size, −0.0403 to −0.2109. Fusion is already good at placing a memory that shares words with the query, and a second pass reorders it worse. So the pass is confined to the candidates no lexical channel found, and they are written back into the positions they already held. Unconfined, two of those models score +0.1698/−0.2109 and +0.1678/−0.0806.

Confining it was previously recorded here as making the same-language column *unable* to move. That was wrong, and measuring the shipped path is what found it. The confinement holds movement to the candidates the lexical channels missed — but a same-language answer they missed is one of those candidates, and reordering can carry it down. Sixty-one same-language queries leave a relevant sentence below rank ten with reranking off, and seventy-one with `fast` on. The residue is −0.0062, small enough that the design still works and large enough that "cannot move" was a claim about the code rather than about the corpus.

The rule was a language comparison first, since "written in another language" is what the case really is. The two rules pick the same candidates — they agree on 93% of a shortlist and score within 0.002 — but the language test needs the query's language, and that is exactly what a detector will not commit to for a short query: `detect_language` returns nothing for "how does deployment work". A rule that quietly does nothing on the commonest shape of query is worse than a slightly different rule.

**`fast` is the default on latency, not on quality.** It is fourteen times smaller than `accurate` and its pass costs 211 ms against 948 — 4.5x. For that it gives up 0.0067 cross-lingual, and it gives up the 0.0070 of same-language that `accurate` gains: `accurate` is the only tier that costs nothing on either group.

The appeal to *Shallow Cross-Encoders for Low-Latency Retrieval* (arXiv 2403.20222) turns on a latency budget, and the budget is what this section got wrong twice. At the figures recorded here the gap was 2.5x and the shallow model's case looked thin; re-measured it is 3.8x end to end, 264 ms against 1001. `fast` stays the default, and the reason is unchanged and now better supported: a search that takes a second is a different product from one that takes a quarter, and the difference it buys is in the third decimal. That is a judgement about the budget rather than a result, and `--rerank accurate` is there for a workspace that judges differently.

**On a corpus that is not parallel text the two tiers separate much further.** MIRACL's Swahili dev split is 131,924 real passages averaging 229 characters with human judgements, one language throughout — the shape XQuAD-R is not:

| Tier | nDCG@10 | Gain | A search | Of which reranking |
| --- | --- | --- | --- | --- |
| `off` | 0.7158 | — | 142 ms | — |
| `fast` | 0.7359 | **+0.0201** | 474 ms | 332 ms |
| `accurate` | 0.7654 | **+0.0496** | 1867 ms | 1725 ms |

`fast` is worth half what it is worth on sentences, which was expected: the pass reorders only what the lexical channels missed, and within one language that is a fraction of the shortlist rather than nearly all of it — 84 of 482 queries leave a relevant passage below rank ten here against 1,042 of 1,190 there. `accurate` was expected to shrink with it and does the opposite, gaining more here than there, so the ratio between the tiers goes from 1.2 to 2.5. Long varied passages are where twenty-one million parameters start to tell against three hundred million, and that is the case the shallow-cross-encoder argument does not cover.

These MIRACL rows came from a program that was never committed, which is worth saying here because this section is where they are used to argue for a default. `crates/pamin-engine/tests/monolingual.rs` is the harness for them now, with the model-alone arm this corpus never had, and until it has reproduced them the numbers above are a record of a past run rather than something a reader can re-take. What the argument rests on -- that the tiers separate further on real passages than on parallel sentences -- is the part to re-check first.

It costs accordingly: 1725 ms against 332. Two seconds a search is not an interactive budget, so the default does not move — but on real passages choosing `fast` gives up three fifths of the available gain rather than a fifth, and that is worth knowing before accepting it. recall@50 is 0.9494 for all three tiers, which is the same invariant the other corpus shows. A workspace whose memories are all in one language should set `off` — the candidates the lexical channels miss are overwhelmingly the ones in another language.

A score depends on the query as well as the memory, so a resident process remembers the pairs it has computed: a repeated search measured 69.6 ms the first time and 0.0 ms the second, for the same ordering. Four thousand scores, about a quarter of a megabyte. It does nothing for a query never asked before, which is most of them; it is worth its quarter megabyte because agents retry, widen a limit, and ask again after writing. Without `pamin serve` there is no process to keep it in.

**There is no compilation trick left in the runtime.** Sorting candidates by length before batching and using batches of eight rather than sixteen took the same work from 191 ms to 151, because a batch is padded to its longest member. Against that, the export format is worth at most 1.45x on identical weights, fp16 is slower than fp32 on a CPU, and the session already runs every core at the highest graph optimization level. The measured 9.75 ms a pair is what twelve transformer layers on four cores cost.

The published answers to this latency all change the architecture instead, and both are deferred on a missing export rather than on a licence or a doubt:

| Route | Worth | Blocked on | Revisit when |
| --- | --- | --- | --- |
| Precomputed document layers (PreTTR, arXiv 2004.14255) | ~6x on this shape: twelve layers over a ten-token query and a hundred-token memory is 1320 layer-tokens; caching the memory's first eleven layers makes it 220. 165 ms becomes roughly 28 — the base was 151 ms until the tier figures were re-measured through the engine, and it is the ratio that carries the estimate rather than the base. Storage is hot set x tokens x width: 384 MB for ten thousand memories | An export split into two halves. It is an export-time job, not runtime graph surgery: `ort` selects only among declared graph outputs, and while `ort` 2.0.0-rc.13 — the version in this lockfile — does now carry an `editor` module behind feature `api-22`, it builds a graph from scratch and cannot load an existing export to cut one. Nothing on crates.io can | A split export of `mmarco-mMiniLMv2` exists **and** the score cache's hit rate shows the hot set is actually small, which is the same evidence that decides whether it is worth its storage |
| Late interaction (ColBERT) | Moves the cost to write time, where the cascade already runs a forward pass per memory; query time becomes MaxSim. At 64–128 dimensions int8 that is 6.4–12.8 KB a memory, the same order as the vector index | **Was** no usable model, and that is no longer true. `colbert-xm` is still MIT with no ONNX export, and the English-only exports are still English-only — but `lightonai/mLateOn` is Apache-2.0, multilingual over nine languages, 128 dimensions, on mmBERT-base, and its own repository carries `model.onnx` and `model_int8.onnx`. Verified against the Hugging Face API on 2026-09-20, not from a card or an announcement. 128 dimensions int8 is 12.8 KB for a hundred-token memory, the top of the range this row already budgeted for. What is not verified is whether the projection head is folded into the export or applied after it, which is a ten-minute check at load time. Rejected on licence while looking: `LiquidAI/LFM2-ColBERT-350M` ships under LFM Open License v1.0, which this project cannot take while it distributes the database itself. BGE-M3's own ColBERT head is 1024-dimensional, 100 KB a memory int8, ten times the whole index | Superseded. The trigger named `colbert-xm` because it was the only permissive multilingual candidate in 2026-03; `mLateOn` now satisfies what the trigger was standing in for, so what remains is the premise the row itself flags — measure it cross-lingual on the XQuAD-R and MIRACL harnesses against the `fast` and `accurate` tiers, and measure what a forward pass per memory costs the cascade, before taking the storage |

Both routes rest on premises Påmin Memory has not measured — that the hot set is small, that write-time cost is cheap — and the triggers are written to test the premise before the work. Re-checked 2026-09-20: one route's blocker dissolved and the other's did not, which is the reason to re-check a deferral rather than trust the note that created it.

Licensing was the blocker when this was first examined and is no longer. The embedding library's own four rerankers remain unusable — two English-only, one CC-BY-NC-4.0, and one carrying no licence at all — but its user-defined loader takes any ONNX, which is the path both tiers take.

### Where a search's milliseconds go

The tier table above prices reranking against a search with it off, and that
left the other 53 ms undivided. Naming it mattered because the optimisation
list was ordered against a figure that turned out not to exist: an earlier
note in this repository put "about 50 ms of a CLI search that is not
retrieval", derived by subtracting a socket measurement on one corpus from a
whole-CLI measurement on another at a different rerank tier. Two corpora and
two tiers cannot be subtracted, so that number was an artifact and is
withdrawn.

Measured instead with every arm on one corpus -- the 13,014-sentence XQuAD-R
workspace, `profile accuracy`, one resident server, four cores and 15 GB:

| Stage | Costs | Of a `fast` search |
| --- | --- | --- |
| The cross-encoder pass (`fast`) | 226 ms | 70% |
| The query's own embedding | 68 ms | 21% |
| Four channels, fusion, and reading the states back | 16 ms | 5% |
| Being a process rather than a socket call | 13 ms | 4% |

Each row is a difference between two arms that differ by one thing, and each
is four independent samples of sixty queries, or three of thirty-six for the
last. The reranking row reproduces the tier table's 211 ms from a different
direction, which is the check on the method.

Two of the arms need saying, because both are ways this measurement could
have lied. The server remembers a query's vector and remembers each
query-document score it has computed, so asking the same query twice measures
a different thing from asking it once: a query the server has never seen costs
85 ms with reranking off and 310 ms with `fast`, and the same query asked
again costs 16 ms either way. Every "unseen" row here is unseen by
construction -- disjoint halves of query sets drawn fresh from the corpus, no
half reused across arms -- because the first attempt at this table reported
16 ms for a cold query and was measuring its own warm-up. And the whole-CLI
arm runs behind `sudo -u`, since the workspace's PostgreSQL directory is mode
700; that costs 8 ms of somebody else's fork and exec, and it is subtracted
rather than charged to the binary.

These rows are not interchangeable with [measured.md](../measured.md)'s
sweep, which reports 77 ms for a whole CLI search at `off` where this reports
85 ms over a socket, and the difference goes the wrong way for a process to
explain it. Two things differ and both are stated rather than resolved: this
ran against a supplied PostgreSQL 17.11 instead of the pinned build, which
[cli.md](../cli.md) already says disqualifies a figure from that table, and
its queries are sentences drawn from the corpus at 40 to 160 characters where
that sweep's are XQuAD-R's questions. The second is the likelier of the two,
and it is the table's own point: if the query's embedding is most of an `off`
search then an `off` search costs what the query is long, which is a property
of the caller rather than of the corpus.

**The ordering this gives is not the one the optimisation list had.** Cutting
allocations out of the retrieval path -- tokenizing a query once instead of
three times, handing out a query vector behind an `Arc` instead of cloning
4 KB of it -- works on the 16 ms row, which is 5% of a search and already the
smallest of the four. The 13 ms of process is smaller still, and 3.3 ms of it
is the dynamic loader, so unlinking the 36.9 MB index library from a client
that never touches the index would buy a few milliseconds for a second
executable that both engineering budgets would stop seeing.

What the table says instead is that a search is two forward passes and a
little bookkeeping. Reranking is the largest, and the fusion section records
that on this corpus its entire cross-lingual gain is paying back what fusing
four channels gave away -- so the same change that stops the dilution is also
the one that makes 70% of the latency optional, and accuracy and latency point
the same way for once. The query's own embedding is the floor under
`--rerank off`: 68 ms of XLM-RoBERTa-large on four cores, which no amount of
work in this repository will make cheaper, and which a smaller profile would.
That trade has not been measured and should not be guessed at here.

### The index lock, and what would actually lift it

Every call into the projection goes through one exclusive lock. The engine
declares `Sync` and does not honour it — a reader takes an unsynchronized
snapshot of the segments a writer is changing, reported upstream as
alibaba/zvec#714 — and without the lock, searches fail inside a minute under
sustained concurrent traffic.

Two things about that lock are worth writing down, because both are easy to get
wrong from the outside.

**It is a mutex rather than a read-write lock, and that was decided by a hang
nobody upstream has reported.** With readers allowed to share, twenty writers
and twenty readers wedged the process inside the engine's own code: forty-six of
its threads asleep on futexes, no caller of ours above them, and no processor
time used by any of them for thirty-five minutes. The same run with writers
alone passes; the same run with readers made exclusive passes for the full five
minutes. Upstream #714 reports SIGSEGV, and the error we also saw —
`Read next record batch failed (fill_result): fetch table failed` — but **no
upstream issue reports a hang**. So the two are consistent with each other and
have not been shown to be the same defect.

**Which means the fix landing upstream is not, by itself, permission to remove
the lock.** #714 closed with #715 (merged as `515c11a`, giving the segment locks
shared readers) and the sibling data-loss report #724 closed with #731 (as
`31d88ea`). Neither is in a published version — 0.7.0 predates both — and
neither touched a public header, so adopting them is a version bump rather than
a binding change.

The trigger is therefore two-part, and the second part is the one that matters:

- **When** a `zvec-rust` release contains `515c11a`, take it.
- **Then** run `readers_and_writers_share_one_index_without_bringing_it_down`
  for its full five minutes with the lock relaxed to a read-write lock, on
  Linux, before believing anything. On macOS the race is latent and a green run
  says nothing. If it wedges again, the hang is a second defect, the lock stays
  a mutex, and *that* is the point at which it is worth reporting upstream with
  the stack and the three-way bisect above.

Compaction is already outside this lock, on the strength of #614, which did ship
in 0.7.0: Optimize is a brief exclusive seal, a long phase holding no schema
lock, and a brief exclusive commit. That is the one part of the engine's
concurrency Påmin Memory relies on today.

**And on four cores the lock is not what is stopping concurrent search anyway.**
That was measured before planning anything around it, because the cost of the
lock had been asserted and never established.

A search takes two exclusive guards in sequence: the embedder, for one forward
pass, and then the index, for the three recalls. Varying how many requests hit
the query cache separates them without instrumenting the source: a hit skips the
model, so what is left is the index guard. Thirteen thousand XQuAD-R sentences,
accuracy profile, the resident pool the server uses, throughput in queries a
second:

| N | 0% cached | 50% | 100% |
| --- | --- | --- | --- |
| 1 | 13.1 | 21.9 | 47.4 |
| 2 | 15.3 | 22.1 | 70.0 |
| 4 | 14.9 | 24.2 | 75.2 |
| 8 | 15.5 | 23.8 | 77.9 |

**Concurrency never costs throughput here.** An earlier version of this section
reported that it did — that two cached readers got less than one, 65.7 q/s down
to 35.7 — and that was an artefact of the harness rather than a property of the
engine. The query cache holds 256 entries and evicts first-in, and the sweep
warmed it once before measuring every configuration in turn, so each
cache-miss configuration flushed the entries the next cache-hit configuration
depended on. Only the very first row was measuring what it claimed. Re-warming
before each configuration, and asserting that a fully-cached run is at least
twice as fast as an uncached one, reverses the finding: cache hits scale from
47.4 to 77.9 and misses stay flat.

Flat is the real result. Cache-miss throughput barely moves from one caller to
eight because the forward pass is compute-bound on four cores, and the control
below says the locks are not why:

| N | embeddings/s | per pass |
| --- | --- | --- |
| 1 | 182.3 | 5.5 ms |
| 2 | 92.3 | 10.8 |
| 4 | 66.9 | 14.9 |
| 8 | 53.5 | 18.2 |

**Throughput halves at N=2 with no lock in the picture, and keeps falling.** ONNX
Runtime's own intra-op pool already uses all four cores for a single forward
pass, so a second caller does not find an idle core to run on — it finds the
first caller's threads. Against that control the cached arm degrades *less* than
lock-free work does, and the fresh arm gains a third rather than losing
anything. Neither guard is the binding constraint here; the machine is.

So there is nothing for removing the index mutex to buy on this hardware, and
nothing for splitting the embedder's cache guard from its model guard either —
a cache hit is already three times the throughput of a miss at every N.

**No admission control either.** Throughput never falls as callers are added at
the shipped settings, so there is no concurrency limit for a semaphore in front
of the server to recover. That question was worth asking only while the cached
column appeared to say the opposite.

**What the cores are divided between is worth setting, and is not worth
changing by default.** ONNX Runtime splits one forward pass across every core
unless told otherwise, and `PAMIN_INFERENCE_THREADS` is the other way to divide
them — fewer threads per pass, more passes at once. Throughput in queries a
second, same sweep:

| | 0% cached | | | 50% | | | 100% | | |
| N | 1 thread | 2 | 4 | 1 | 2 | 4 | 1 | 2 | 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 9.7 | 12.1 | **13.1** | 15.2 | 19.8 | **21.9** | 46.0 | **47.7** | 47.4 |
| 2 | 11.6 | **18.3** | 15.3 | 18.6 | **24.2** | 22.1 | 67.9 | 68.4 | **70.0** |
| 4 | 10.9 | **16.7** | 14.9 | 22.1 | **30.2** | 24.2 | 69.6 | 71.9 | **75.2** |
| 8 | 11.0 | **16.4** | 15.5 | 19.4 | **29.0** | 23.8 | 70.7 | 72.9 | **77.9** |

Two threads is worth up to a quarter more throughput on concurrent traffic that
misses the cache, and it loses on the two cases either side: a single caller,
where four threads finish one pass sooner, and fully-cached traffic, where the
model does not run and the split is pure overhead. One thread is worst
everywhere — the per-pass cost of splitting a small model's tensors four ways is
smaller than the cost of not splitting them at all.

The rule set before the sweep was that a setting has to beat the shipped one at
every mix to become the default. Two threads does not, so the default stays as
it was and the setting is documented instead: a deployment that knows it serves
concurrent, mostly-distinct queries can take the quarter, and one serving a
single agent should not.

**This says nothing about a machine with cores to spare.** On sixteen or
thirty-two, one forward pass would not saturate the box, callers would not be
fighting for the same cores, and the guards could well become exactly the
ceiling this measurement failed to find. The sweep is `conc-harness.sh`, kept
out of the repository with the rest of the measurement harnesses; re-run it
there before concluding anything about a larger machine, and treat the two-part
trigger above as unchanged until then.

One methodological note, because it nearly went the other way: `Engine::open`
takes `Connections::PerCommand`, which caps the pool at four, and a search uses
several connections. The first run of this sweep went through it, so eight
concurrent searches were partly queueing on connections rather than on anything
being measured. Re-running against `Connections::Resident` — what `pamin serve`
actually uses — moved no number outside run-to-run noise, so the pool was not
the confound it looked like. A measurement of a lock has to be a measurement of
that lock.

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
