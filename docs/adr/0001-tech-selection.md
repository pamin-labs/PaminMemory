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
| Vector index | Half-precision vectors under a DiskANN graph (`disk`, the default) or an HNSW graph (`memory`), candidates ranked again by an exact f32 cosine |
| Segmentation | `icu_segmenter` (ICU4X) |
| Language detection | `whatlang` |
| Embeddings | ONNX Runtime through `ort` and `tokenizers`, BGE-M3 with int8 weights by default; the E5 profiles through `fastembed` |
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

**The bundled cluster runs with `fsync` on and `synchronous_commit` off.** `postgresql_embedded` 0.21 starts every cluster with `-F`, which is `fsync=off`, and until this was noticed every workspace ran that way: nothing PostgreSQL wrote was forced to disk, so a power cut could leave the data directory corrupt, not merely behind. For the one store here that cannot be rebuilt from anything else that is the wrong trade at any price, and the store now passes `fsync=on` after the flag, which overrides it. What it gives up instead is the last moments: with `synchronous_commit` off a commit returns before its WAL is flushed, the WAL writer flushes within a few hundred milliseconds, and a crash can lose the commits of that window but cannot leave the cluster inconsistent. A memory lost that way is one whose write had returned, which is a real cost; its evidence is the agent's own recent output, which is the cheapest thing in the system to say again. Waiting for the flush on every commit was measured through `Engine::write` on a synthetic project of 152,000 topics, three runs alternating the two settings with `fsync` on, 100 writes each: p50 3.0-5.9 ms a write with synchronous commit against 2.3-2.7 ms without, on this machine's disk, in a debug build at a load average near 12. A laptop's flush can cost more or less than this container's; the direction is the same. A workspace that is already running keeps what it was started with until `pamin stop`.

### The driver stays `sqlx`: pipelining measured

`sqlx` sends a statement and waits for its result before it sends the next.
The PostgreSQL protocol allows pipelining, which means sending several
statements before reading any result, and `tokio-postgres` does it on one
connection. So the question was whether the store pays for `sqlx` in round
trips. Upstream, as of 2026-09-24, the `launchbadge/sqlx` issues this touches
(#408 and #2798) are open, pull request #3891 was closed on 2026-09-14, and the
pool redesign in #3582 is open.

The harness is a scratch program outside the tree. It runs the statements of
the write path verbatim, copied from `repository.rs` and `jobs.rs`: the twelve
that rewriting an existing topic issues inside `Engine::write`'s transaction.
It also runs the read that `current_states_of` makes, for 64 topics. The data
is 20,000 seeded topics in a local PostgreSQL. `sqlx` 0.9.0 uses the
product's pool options, and `tokio-postgres` 0.7 is the control. Each iteration
runs every arm once, in an order rotated per iteration, for 7 rounds of 300
iterations. Ratios are taken per iteration and then the median is reported.
Each arm asserts its premise before it is timed:

- a write arm must have produced the product's work, meaning that many new
  states, each superseding its predecessor, with spans, queued jobs, and
  pointers on the newest state;
- a read arm must return exactly the reference rows;
- both drivers must report the `synchronous_commit` the run asked for;
- 64 pipelined `SELECT 1` must take under 0.7 of their sequential time. They
  measured 0.16 to 0.24.

An earlier run stopped on the premise that its pipelined reads overlap, and no
figure here comes from it. The machine was the shared four-core one, at a load
average of 8 to 13.

| paired ratio | multi-thread runtime, `synchronous_commit=off` | multi-thread, `on` | current-thread, `off` |
| --- | --- | --- | --- |
| write, `tokio-postgres` pipelined along its dependencies (4 round trips) ÷ the same 12 statements one at a time | 0.92 | 1.03 | 0.83 |
| write, merged into writable CTEs and pipelined (2 round trips) ÷ 12 one at a time, `tokio-postgres` | 0.82 | 1.03 | 0.65 |
| write, `sqlx`, merged into writable CTEs (6 statements) ÷ `sqlx`'s 12 | 0.86 | 0.87 | 0.92 |
| write, `sqlx`, one PL/pgSQL function ÷ `sqlx`'s 12 | 0.54 | 0.74 | 0.68 |
| write, `tokio-postgres`'s 12 ÷ `sqlx`'s 12 | 0.59 | 0.64 | 0.91 |
| 64 reads, `sqlx`, one `= ANY($1)` ÷ 64 statements on the pool | 0.07 | 0.07 | 0.08 |
| 64 reads, `tokio-postgres` pipelined ÷ 64 `sqlx` statements on the pool | 0.14 | 0.15 | 0.25 |
| 64 reads, `sqlx` on one held connection ÷ on the pool | 0.50 | 0.62 | 0.42 |
| one `SELECT 1`, `sqlx` on a held connection ÷ on the pool | 0.64 | 0.65 | 0.69 |

**Pipelining the write transaction is worth 0.83 to 1.03.** Its statements
depend on each other: the locks need the ids the lookups return, the new state
needs the previous one, and the pointer needs the new version. So twelve
statements pipeline into four round trips at best. On a local socket a round
trip is not the cost either. A `SELECT 1` takes 0.058 to 0.090 ms on
`tokio-postgres`, and the whole write transaction on `sqlx` takes 10.7 to 13.7
ms (medians). With commits flushed, which is PostgreSQL's default and the
product does not change it, pipelining measured 1.03.

**The reads are already batched, and batching beats pipelining.** The store
reads current states in one `= ANY($2)` statement. The same shape on `sqlx`
takes 0.915 ms against 1.831 ms for 64 reads pipelined on `tokio-postgres`
(multi-thread runtime, `off`).

**Part of the gap between the drivers is the pool, and part is not
explained.** Every statement run on `&PgPool` acquires a connection and
releases it, and `sqlx-core` 0.9.0 pings the connection on every release
(`return_to_pool` in `pool/connection.rs`). `test_before_acquire(false)` does
not turn that off. Holding one connection for 64 reads costs 0.42 to 0.62 of
running them on the pool, so where a path runs several statements back to
back outside a transaction, holding one connection is the fix, and `sqlx`
already provides it. The write transaction already holds one connection, so the
ping is not what makes `tokio-postgres` 0.59 to 0.91 of `sqlx` there. What
does was not isolated.

So the store stays on `sqlx`. Pipelining is worth close to nothing on the path
that could use it. Switching drivers for the per-statement gap would bring back
the second driver the Consequences below record removing. The largest single
lever measured on `sqlx`, one server-side function at 0.54 to 0.74, is PL/pgSQL, and
PL/pgSQL is not in the portable subset listed above. Writable CTEs are in that
subset and are worth 0.86 to 0.92 on `sqlx` with no driver change, which makes
them the lever to reach for if the write transaction's round trips ever matter.
Revisit this when the database stops being local: every figure above has a
round trip under a fifth of a millisecond. Revisit it too if `sqlx` ships
pipelining or stops pinging on release.

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

Recall engines return per-channel ranked lists. Reciprocal rank fusion runs in our layer. This is a correctness requirement, not a preference.

`k = 10`, not the customary 60, and the two lexical channels carry an eighth weight each — a quarter each until a third corpus was measured; the paragraphs below record the quarter as it was argued, and the section after them is what replaced it. Both are measured rather than taken from the literature: 60 came from fusing lists thousands of results deep, and each channel here proposes fifty, which the constant flattens to the point where rank barely counts. The lexical pair runs BM25 over the same text twice, so at equal weights the two of them cast two votes against the vector and graph channels' one each. That much holds. The stronger claim this record used to make alongside it — that the two are near enough one channel to share a weight — does not: Kendall tau-b between their rankings is 0.2816 on this project's own corpus, 0.3188 on XQuAD-R and 0.2973 on MIRACL. They are two channels that agree about a third of the time, sharing a field rather than a ranking, and the single constant they share has never been swept apart.

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

### A third corpus, and the value the sweep never tried

Everything above was settled on two corpora: the one written for Påmin Memory
and XQuAD-R. Both have a property worth naming. Påmin Memory's own corpus has
two of its three groups pinned at 1.000, so nothing can be observed to cost
anything there; XQuAD-R is parallel translation whose questions are SQuAD's,
built from the answer sentence's own words. Neither is a query anybody asked.

MIRACL's Swahili dev split is: 131,924 Wikipedia passages, 482 questions people
asked, 5,092 human relevance judgements, one language throughout. Swept the same
way (`speed` profile, fusion only, no reranking, 482 queries a row):

| lexical | k=5 | k=10 | k=20 | k=60 |
| --- | --- | --- | --- | --- |
| 0.000 | 0.6848 | 0.6848 | 0.6848 | 0.6848 |
| **0.125** | **0.6885** | **0.6882** | 0.6821 | 0.6604 |
| 0.250 | 0.6854 | 0.6826 | 0.6641 | 0.6282 |
| 0.500 | 0.6713 | 0.6606 | 0.6414 | 0.5950 |
| 1.000 | 0.5750 | 0.5676 | 0.5545 | 0.5263 |

The `0.000` row is flat because `k` has nothing to do when one channel is left:
the graph channel returns nothing on this corpus, which the harness asserts
rather than assumes. That row is also **0.6848 to four decimals against the
embedder compared exhaustively to every passage** — so the HNSW index over
131,924 documents reproduces the brute-force scan, which is the check on the
rest of the table.

Two things follow, and the first is uncomfortable.

**At the shipped setting, fusing four channels is net negative here.** `k=10`,
lexical 0.25 scores 0.6826 against 0.6848 for the vector channel alone: −0.0022.
Not large, and in the wrong direction for a layer whose purpose is to improve
the ranking.

**And a quarter was never compared against half of itself.** The sweep that
settled it ran `1.00 / 0.50 / 0.25 / 0.00` — a quarter was the smallest non-zero
value ever tried, and every argument above for it is an argument against zero.
An eighth beats it on every group of all three corpora except one: +0.0056 here,
+0.0550 on Påmin Memory's cross-lingual group with the other two unmoved at
their ceilings, +0.0377 on XQuAD-R cross-lingual, and −0.0501 on XQuAD-R
same-language.

That exception is the one measurement worth arguing with, because the same
quantity is now measured twice on same-language queries and the two disagree by
a factor of twenty: what the lexical pair is worth on questions asked in the
answer's own language is **+0.0769 on XQuAD-R and +0.0034 on MIRACL**. The
difference between those two corpora is that one's questions were written from
the answer's own words. A benchmark can overstate a channel, and this is what it
looks like when one does.

**The bias in this run points the other way, which is why it is worth acting
on.** These figures are the `speed` profile — multilingual-e5-small, 384
dimensions — because `accuracy` is ten and a half hours of indexing for this
corpus. A weaker dense channel leaves the lexical pair more to contribute, so
fusion looks *better* here than it would on the shipped model. It is still net
negative at the shipped weight.

**The per-query rule is still not the lever.** `adapt 0.50-1.00` scores 0.6885,
the best cell in the table — and the constant hiding inside it, an eighth held
at every query, scores 0.6882. Three corpora now agree that the rule is worth
between nothing and 0.009 over the constant it reduces to, while the constant
itself is worth ten times that. It stays off.

**The default is an eighth**, and what moved with it. Measured on each corpus's
shipped path — `search_reranked` at the default tier, which is what `pamin
search` calls — except where the row says otherwise:

| | quarter | eighth |
| --- | --- | --- |
| Påmin Memory's corpus, cross-lingual | 0.7223 | **0.7773** |
| Påmin Memory's corpus, monolingual / lexical | 0.9940 / 1.0000 | 0.9940 / 1.0000 |
| XQuAD-R, cross-lingual | 0.6097 | **0.6480** |
| XQuAD-R, same-language | 0.7971 | 0.7495 |
| MIRACL sw, `speed`, fusion alone | 0.6826 | 0.6882 |
| MIRACL sw, `speed`, default tier | not taken | 0.6730 |

The XQuAD-R cross-lingual row is the one that answers the question this project
started with. The embedding model alone scores 0.6335 there; at a quarter the
whole stack scored 0.6097, **below the model it is built on**, and the
cross-encoder was spending 226 ms a query buying back damage the fusion layer
had done. At an eighth the stack scores 0.6480, above the model, with the same
reranker.

#### Re-taken query by query, and one leg of the argument above is withdrawn

Every figure in that table is a difference of means. Re-taken as paired
comparisons — the eighth against the quarter directly, on the same queries in
the same order, fusion only at `k = 10`:

| | mean | wins / losses / ties | p |
| --- | --- | --- | --- |
| Påmin Memory's corpus, cross-lingual (43 queries) | **+0.0589** | 21 / 2 / 20 | 0.0004 |
| Påmin Memory's corpus, monolingual and lexical | unmoved | — | at 0.9940 and 1.0000 |
| XQuAD-R, cross-lingual (1,190) | **+0.0377** | 698 / 26 / 466 | 0.0001 |
| XQuAD-R, same-language (1,190) | **−0.0500** | 16 / 252 / 922 | 0.0001 |
| MIRACL Swahili (482) | +0.0056 | 77 / 72 / 333 | **0.2300** |

**The decision stands and the reason given for it does not.** Seventy-seven
wins against seventy-two losses is not a result, so the +0.0056 that this
section led with — the eighth as "the best of the five on MIRACL" — is noise.
So is the −0.0022 by which the quarter was said to rank below the vector
channel on its own: paired, zero weight against an eighth on MIRACL is 30 wins
to 59 losses at p = 0.2972, and the quarter against an eighth is the row above.
Both sentences are withdrawn.

What does hold is the pair of cross-lingual groups, and it holds hard. On
XQuAD-R the eighth beats the quarter on 698 queries and loses on 26. On this
project's own corpus it wins 21 of 43 and loses 2, with the other two groups
sitting still on their ceilings. Those are the legs the change actually rests
on, and neither was the one being led with.

**And the cost is as real as the gain, which the original note undersold.**
XQuAD-R's same-language group loses on 252 queries and wins on 16. That is a
two-sided trade with both sides significant, not a gain with a footnote. It is
recorded here as a trade this project chose, for the reason the next paragraph
gives — not as something small.

**What MIRACL does say, once it is allowed to say only what it can.** Nothing
in the range from zero to a quarter is distinguishable on 482 real
single-language queries. Half weight is worse (p = 0.0003) and full weight much
worse (p = 0.0001), so the corpus is not insensitive — it can separate the
settings that matter and cannot separate these. Read plainly: **on real
monolingual retrieval the two lexical channels' contribution is not measurable
at any weight this project would consider.** They earn their place on
cross-language queries, where the measurement is unambiguous, and on the
file-path and error-code matching the n-gram channel exists for, which no
corpus here tests.

**That last sentence was wrong, and the instrument is why.** A weight sweep
moves both lexical channels at once. They are two channels, agreeing only about
a third of the time, and moving them together lets one cancel the other -- so a
sweep that separates nothing is not evidence that neither matters. Taking the
segmented channel away on its own, which the leave-one-out diagnostic does,
costs MIRACL significantly at p = 0.0125. The channel is measurable there. What
was not measurable was a one-dimensional slice through a two-dimensional
question, and the sentence above read a null result off the wrong instrument.

**The adaptive rule is now visibly just a weaker constant.** Against the
quarter on XQuAD-R, `adapt 0.00-1.00` scores +0.0618 cross-lingual and zero
weight scores +0.0635; the four adaptive rows interpolate monotonically between
the constants they are built from, group for group, with the same wins and
losses. It was already off by default on the grounds that it bought almost
nothing; the counts show there is no separate thing there to buy.

**The rule has been removed, and what it disproves is load-bearing.** Its
signal was cross-channel agreement: how much of the vector channel's list the
lexical channels also returned. That is the cheap way to ask the question
fusion actually needs answered — *is this channel worth listening to on this
query* — and it is the only way that needs nothing plumbed, because ranks are
all it reads. Measured across three corpora it is worth between +0.0003 and
+0.0085, which is to say nothing. So the cheap route is closed, not untried,
and the remaining route is the channels' own scores: how far a channel's best
candidate stands above its own field, which is legible in the score
distribution and nowhere in the ranks. Those scores existed at the index layer
all along and were being discarded before anything could read them.

The MIRACL shipped-path cell at the quarter says `not taken` because it never
was: the run that would have produced it died on an assertion the harness makes
about its own graph channel, and by the time the harness was fixed the weight
was the thing under test. The pair of MIRACL rows that do exist are both at the
eighth, and they are enough for what follows.

**And the reranker was servicing a debt, which those two rows price.** Once
fusion stops diluting, the cross-encoder *costs* 0.0152 on MIRACL: 0.6730 with
it against 0.6882 without. That corpus has one language and no parallel
translations, so there is no cross-language confusion for a reranker to
resolve; what it did on XQuAD-R was largely undo the lexical pair's
misdirection there. The tier stays on by default because XQuAD-R still pays
+0.0403 for it and the LOCOMO and LongMemEval figures were taken with it, but
the case for `--rerank off` on single-language corpora is now measured rather
than speculative. The depth constant it uses was swept against the quarter's
baseline and has not been re-swept — see `pamin_index::reranking::DEPTH`.

### Rank fusion was never compared against score fusion

Everything above argues about `k` and about the lexical weight. Both are
parameters *of* reciprocal rank fusion, and the choice to fuse ranks at all —
rather than to normalise each channel's scores and combine those — was never
measured here. It was inherited. That is the larger of the two questions and
the 2025–2026 literature is close to unanimous on it.

| | year | what it found |
| --- | --- | --- |
| [Ranking-based Fusion Algorithms for XMTC](https://arxiv.org/html/2507.03761v1) | 2025 | Ten fusion algorithms against six normalisations on four corpora. **z-score normalisation with CombMNZ was highest on every corpus**, and the rank-based family (ISR, Log-ISR) and the voting family (Borda, Condorcet) lost to score fusion on every one |
| [Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval](https://arxiv.org/html/2606.04194) | 2026 | On LoCoMo and LongMemEval-S — two of this project's own corpora — on CPU with no training: **z-score weighted fusion Hit@1 0.752 against RRF's 0.718**. The weight has a wide plateau, 0.25 to 0.50 all above 0.73 |
| [From BM25 to Corrective RAG](https://arxiv.org/html/2604.01733v1) | 2026 | 23,088 questions. RRF at `k=60` Recall@5 0.695, RRF at `k=10` 0.716, **a convex combination at an untuned α=0.5 0.726** |
| [Calibrated Fusion for Heterogeneous Graph-Vector Retrieval](https://arxiv.org/html/2603.28886v1) | 2026 | The only published work that fuses a graph channel with a vector channel, which is this project's shape. Its ablation's conclusion is that **"normalization appears to be the dominant empirical factor"** — it mattered more than the combination rule |

**And the parameter this ADR spends the most words on turns out to be the
cheap one.** The only real `k` sweep published in the window tested 10, 30, 60
and 100 and found 10 the best of them; a SIGIR 2025 paper sets `k = 0`
outright. No paper in 2025 or 2026 derives `k` from anything. So the spread
across sensible values of `k` is small, this project's `k = 10` is on the
favoured side of it, and the fusion function itself is where the difference
lives.

**Why this matters more here than in the two-channel papers.** Almost all of
that work fuses one sparse channel with one dense one. This project fuses four,
and one of them is a graph walk in PostgreSQL whose scores are on no comparable
scale at all. Rank fusion hides that — which is its appeal — but hiding it is
not the same as handling it. A graph result ranked first out of four contributes
exactly what a vector result ranked first out of fifty thousand contributes,
because rank is all that survives. There is no way to express "this channel
returned four things and is not confident about any of them".

[Balancing the Blend](https://arxiv.org/abs/2508.01405) (2025) is the only
published four-channel analysis, over eleven corpora and eleven channel
combinations, and it names the failure this sets up as the **weakest link**: on
one configuration, rank-fusing full-text search with dense vector search scored
**0.604 nDCG@10 where dense search alone scored 0.784** — fusion destroyed
eighteen points. Its stated mechanism is that RRF read the high ranks coming
from both channels as agreement and promoted irrelevant documents on the
strength of it. Its prescription is cheap and needs no training: **set each
channel's weight to that channel's own standalone nDCG@10**.

Which is a measurement this project has never taken. There is no figure
anywhere for what any single channel is worth on its own. Every number here is
of the four fused.

**What the diagnostic found.** Measuring each channel alone needed no new runs:
`fuse` writes a trace line for every candidate of every channel at any weight,
so one pass carries the whole matrix of where each channel ranked what, and now
of what it scored it too. Three findings, on three corpora:

| channel, alone | ours cross | XQuAD-R cross | XQuAD-R same |
| --- | --- | --- | --- |
| `lexical_segmented` | 0.1569 | 0.0366 | **0.7299** |
| `lexical_ngram` | 0.0528 | 0.0106 | 0.5337 |
| `vector` | **0.8268** | **0.6335** | 0.6787 |
| `graph` | premise absent | premise absent | premise absent |
| all four fused | 0.7985 | 0.6114 | 0.7829 |

**Fusing four channels ranks below one of them on cross-lingual queries**, and
on the same-language queries of the same corpus the lexical channels earn their
place outright. Leave-one-out, paired against all four: taking
`lexical_segmented` away is **+0.0148** on this project's cross-lingual group
(11 wins, 1 loss, p = 0.0148) and **+0.0128** on XQuAD-R's (397 wins, 25
losses, p = 0.0001), while `lexical_ngram` is +0.0179 (15 / 1, p = 0.0189) and
+0.0094 (269 / 16, p = 0.0001). The same removals cost **−0.0523** and
**−0.0361** on XQuAD-R's same-language group (16 wins to 231 and 9 to 155, both
p = 0.0001). The channels are not weak. A global constant cannot tell the two
cases apart — this is the weakest link the four-channel paper above names,
arrived at independently.

The figures in this paragraph and the table above it are the banded combiner's,
retaken; see *The weight table was taken under rank fusion and never retaken*
below for the rank-fusion figures these replace and for why the trade is better
in both directions than it used to read.

**The two lexical channels are not one channel.** Kendall tau-b between their
rankings, over the candidates they share: 0.2816 on this project's own corpus,
0.3188 on XQuAD-R, 0.2973 on MIRACL. They agree about a third of the time. The
premise that justified one shared weight is refuted, and every sweep this
project ever ran moved both together, so no measurement distinguishes the two
numbers at all.

**The graph channel contributes exactly 0.0000** — in every group of all three
corpora, so removing it changes no ranking anywhere. **It is not a measurement
of the channel, and the earlier version of this paragraph got how far that goes
wrong.** It said two of the corpora have no relationships to walk and that this
project's own corpus "does have edges and still reports zero", leaving the
latter as the interesting unexplained case. The own corpus has no edges either.

The only edge kind the engine derives is `Mentions`, asserted where one memory's
content contains another topic's *name* as a contiguous token run. Topic names
in the own corpus are identifiers of the form `<subject>_<language>` —
`deploy_pipeline_en` — which `name_sequence` opens into the three-token run
`deploy pipeline en`, and no memory's prose contains that run. The harness now
counts them: **`live edges in this project: 0`**, and the `Graph` row reads
"returned nothing on any query" in all three groups, with leave-one-out at
0W/0L across 43, 32 and 62 queries. The two external corpora name their topics
deliberately unlike their own text and say so in their harness, so theirs are
empty by design.

**The experiment that would give the row a premise is sitting in the corpus
unused.** Every one of the 43 cross-lingual queries has four or five relevant
topics and they are always the same subject in different languages —
`vpn_access_de`, `vpn_access_ja`, `vpn_access_th`, `vpn_access_zh`. That is a
`same_as` relationship the dataset genuinely asserts, on the one group where
fusion still ranks below the vector channel alone (0.7985 against 0.8268), with
the vector channel's own `recall@50` at 0.9605 — so there is room for a hop to
pull a missed sibling up.

It will be reported as an **upper bound, not an estimate**, and the reason is
worth stating before anybody runs it: those siblings *are* the judgement
structure, so `same_as` edges over them restate the answer key as edges. What
such an arm can establish is whether the mechanism works at all when the graph
agrees with the judgements — which would catch a walk that reaches nothing, or
one that ranks its arrivals below every other channel's candidates. It cannot
say what a graph channel is worth on a real workspace, where somebody writing
one policy in four languages would write one memory. The monolingual and lexical
groups, whose relevant lists are single topics, should *lose* from the same
edges, and that asymmetry is the case for deciding per query whether to walk at
all.

So all three zeros are one fact stated three times: **the graph channel has
never been measured with a graph.** The harness now prints the live-edge census
before these rows and labels the graph cell as premise-absent rather than
letting a zero read as a figure. The nearest thing to real evidence is
elsewhere: the LOCOMO `pamin-ledger` arm, where consecutive turns are linked
explicitly, scored 0.523 against `pamin`'s 0.518 — twenty-one discordant
questions against twenty, `p = 1.000`. One small sample, and it says the channel
did not pay there. Nothing says whether it can.

**And the combiner itself was never a recorded choice.** This record argues at
length about `k` and about the channel weights. Both are parameters *of*
reciprocal rank fusion. That fusing ranks rather than normalised scores was a
decision at all appears nowhere, and it is the load-bearing one. Every
2025–2026 result found goes the other way:

| | what it measured | result |
| --- | --- | --- |
| [2606.04194](https://arxiv.org/html/2606.04194) (2026) | LoCoMo and LongMemEval-S, CPU, no training — this project's own benchmarks and constraints | z-score weighted fusion **Hit@1 0.752 against RRF's 0.718**, with a wide plateau in the mixing weight |
| [2507.03761](https://arxiv.org/html/2507.03761v1) (2025) | ten fusion algorithms × six normalisers, four corpora | standardised scores with **CombMNZ highest on all four**; every rank-based and vote-based method below every score-based one |
| [2603.28886](https://arxiv.org/html/2603.28886v1) (2026) | graph and vector channels, multi-hop QA | RRF's mean gain **not significant** where calibrated fusion's *smaller* mean gain is; ablation names normalisation the dominant factor |
| [2604.01733](https://arxiv.org/html/2604.01733v1) (2026) | 23,088 queries, the only real sweep of `k` in the window | `k = 10` Recall@5 0.716 against `k = 60`'s 0.695 — the low constant wins, and nothing published derives one |

Read together they say the knob this project tuned is the low-leverage one:
`k` is worth one to three points and normalisation three to eight. So
`Combine` implemented the score combiners beside `Reciprocal`, which shipped
then, and the offline grid priced each of them with and without the confidence
rule, because the two mechanisms answer different halves and a row that moved
both could not say which half moved it. One prediction was written down in a
unit test before the run: CombMNZ multiplies by agreement, and agreement
between a confident channel and a worthless one is exactly the failure measured
on both cross-lingual groups above, so the combiner the systematic comparison
ranks first should rank last on these corpora.

**What that argues for, and what was built.** Not a normaliser: standardising a
channel's candidates removes the units and not the quality, so a channel whose
fifty candidates are all worthless still maps its best one to about `z = +2`.
Not cross-channel agreement either — that is the adaptive rule, measured and
removed. What is left is how far a channel's best candidate stands above its
own field, `(best - mean) / deviation` over that channel's own candidates:
dimensionless, so a BM25 score and a cosine similarity become comparable, and
readable from the candidates already in hand. That last part is the binding
constraint. A normaliser against a corpus-wide distribution — what the
graph-channel paper did — costs a histogram that has to be maintained as
memories are written, which that paper never had to pay because its corpus was
static, and which would make the mechanism wrong for a user the moment they
wrote something.

`Fusion::with_confidence` implemented it and `Combine` the score combiners,
**both off by default**, and both have since been removed — see
*Fusion designs measured and removed* below. The sweep runs offline from one
pass over each corpus — `channels::as_if` replays a run's trace through the
shipped `fuse`, so a grid that used to cost thirteen minutes a row costs
microseconds a row.

**Both mechanisms have now been measured on all three corpora, and neither
ships.** The combiner is the near miss and it is worth stating exactly, because
it was made the default before the gates caught it.

On nDCG@10 a weighted sum of standardised scores is better in three of four
groups and significantly so — **+0.0281 on this project's own cross-lingual
group (15 wins, 1 loss, p = 0.0037), +0.0134 on XQuAD-R's (448 / 120,
p = 0.0001), +0.0177 on XQuAD-R's same-language group (142 / 106, p = 0.0001)**
— and not significantly worse anywhere, MIRACL being −0.0043 at p = 0.1632.
That is the first setting in this project's history that is not a two-sided
trade, and it takes this project's own cross-lingual group from 0.7910 to
0.8192 against the vector channel's own 0.8268.

**Then XQuAD-R's cross-lingual `recall@50` fell from 0.8960 to 0.7765, through
a floor of 0.8000, and the default went back.** The mechanism is the sign:
every reciprocal-rank contribution is positive, so a candidate one channel
ranked fiftieth still helps it stay in the list, whereas a standardised score
is centred and a candidate below its channel's own mean contributes a
*negative* number. On a cross-lingual query, where a lexical channel scores
0.0366 alone, that channel's confident top hit at `+2` outranks a genuine deep
hit from the vector channel at `-1`: the top ten improves because strong vector
hits dominate it, and the sentences that sat between ranks ten and fifty fall
past fifty. A precision-for-recall trade is one a search feeding a reranker
cannot take — nothing recovers a memory that was never returned. A floor under
each contribution, or normalising to `[0, 1]` rather than centring, would
change that; neither is what the published work measured, so neither is
implemented.

**The offline grid that nearly let it through printed nDCG@10 and nothing
else**, while the gates assert nDCG *and* recall. Forty variants were priced on
half the criterion. It prints both now — and with both columns the finding is
sharper than it was with one: **the recall cost belongs to the combiner and to
nothing else.** Every variant built on rank fusion, at any pair of lexical
weights and any confidence setting, holds XQuAD-R's recall at 0.896
cross-lingual and 0.958 same-language. Every standardised variant sits at
0.7765 and 0.9403 whatever else is set, and confidence does not rescue it. That
is a structural consequence of centring rather than a constant chosen badly,
and −0.12 against −0.018 is the same mechanism seen where the weak channel is
garbage and where it is good.

**And reading that as a property of the combiner said what to fix.** Rank
fusion's narrow range is not incidental: it is what makes the channel weights
mean anything. Any normaliser onto `[0, 1]` spans a factor of infinity inside
one channel, so position beats weight; rank fusion spans 5.45, so weight beats
position. Keep the band and change only what orders the candidates inside it —
min-max each channel's scores onto `[(k + 1) / (k + n), 1]`, which is exactly
the range rank fusion would have used over the same candidates.

| group | nDCG@10 | p | recall@50 |
| --- | --- | --- | --- |
| XQuAD-R cross-lingual | **+0.0037** | 0.0003 | 0.8960 → 0.8962 |
| XQuAD-R same-language | **+0.0273** | 0.0001 | 0.9580 → 0.9571 |
| MIRACL Swahili | −0.0003 | 0.9210 | 0.9314 → 0.9309 |
| this project, cross-lingual | +0.0075 | 0.3731 | 0.9605 → 0.9605 |

Two groups significantly better, none significantly worse, recall moving by at
most 0.0009 — **so `Combine::Banded` is what ships.** Every accuracy gate
passes and two published figures improve: on the shipped path XQuAD-R goes from
0.6480 to 0.6511 cross-lingual and from 0.7495 to 0.7769 same-language. The
same-language gain is the notable half: every weight this project ever changed
took something from that group to pay for the cross-lingual one, and this is
the first change that improves it.

The rest of the grid ships nothing, and that part of the question is closed.
Confidence buys +0.0324 same-language for −0.0174 cross-lingual; zeroing the
lexical pair buys +0.0258 cross-lingual for −0.0769 same-language; the
standardised sum buys nDCG everywhere and −0.12 of cross-lingual recall. Each
is a trade this project declines, and each is declined against a number rather
than a preference.
Per-channel confidence on top of rank fusion is the weaker mechanism: +0.0118
at best (8 wins, 0 losses, p = 0.0381) with a narrow plateau, which by
*Balancing the Blend*'s own reading is what fitting a development set looks
like. The `floor` moves almost nothing, so the effect is discounting a channel
rather than silencing it.

**And CombMNZ is significantly worse, as predicted before the run**: −0.0392 at
3 wins to 21, p = 0.0008. The systematic comparison ranks it first of ten on
four corpora; the four-channel analysis names multiplying by agreement as the
weakest-link mechanism. This corpus says the second one governs here, which is
the same finding as the channel table above reached from the other direction.

**MIRACL disagrees about the combiners and refutes the confidence rule
outright.** On 482 single-language queries every combiner is indistinguishable
from what ships — standardised scores −0.0043 at p = 0.1632, CombMNZ −0.0059 at
p = 0.2675 — which is consistent rather than contradictory: adding scores
instead of ranks buys a great deal where fusion was hurting and nothing where
it was not.

Confidence is a different matter. At the two lowest spreads the change on
MIRACL is 0.0000 across all 482 queries, 0 wins and 0 losses, and the largest
effect anywhere in that grid is −0.0008. The arithmetic was written down before
the run: a standardised top score cannot exceed `sqrt(n - 1)` = 7.00 over fifty
candidates, so a spread of two or less clamps every channel to full weight and
a spread grid read without that in mind shows a ceiling as a plateau. But the
substantive finding is worse than a badly chosen constant. On this corpus the
lexical channels *are* mildly harmful, and their score distributions still look
confident: **the measure cannot see, here, the thing it was built to see.** It
never shipped, and has since been removed; that is measured rather than
cautious. Percentile normalisation against a corpus-wide distribution remains
the alternative the literature supports, and remains refused for the reason
above — a memory store's distribution moves on every write.

**What MIRACL does support is the lexical split**, and it is the one result
here that a one-dimensional sweep could not have produced. The best row of the
whole grid is asymmetric — segmented 0.250 with the n-gram channel at zero,
0.6958, +0.0076 at 79 wins to 51, p = 0.0580 — while moving both together puts
0.250/0.250 at −0.0056. The two channels want opposite directions and every
sweep before this one averaged them. The direction matches the standalone
figures and matches MIRACL's own authors naming Swahili a language where a BM25
hybrid is the strongest zero-shot baseline. At p = 0.0580 it is not a result
and is not taken as one.

`k` is closed. `k = 5` is +0.0220 at p = 0.0039 on this project's own corpus and
+0.0003 at p = 0.9057 on MIRACL — two corpora, opposite readings — so ten
stays, which is what the literature predicts for a constant worth one to three
points.

Nothing changes default until XQuAD-R reports, because it is the corpus that
separates cross-lingual from same-language queries over the same 1,190
questions and the split above is the whole question.

**The first attempt at this sweep measured standardised fusion at 0.0099** — 0
wins, 43 losses — because zvec reports cosine *distance* for a cosine index, so
the vector channel was being summed backwards and its confidence read off its
worst candidate. A figure that implausible is a bug, not a result. The test that
should have caught it passed: it asserted every channel orders its candidates by
the score it reports, and wrote every document with the same stub embedding, so
the vector channel reported one constant and ordering by a constant asserts
nothing. `collect_scored` now takes the orientation as a parameter, and no
channel may report one score for every candidate. Nothing published before that
fix had ever read a score.

### One frontier, and the number that explains it

Ninety-six fusion settings, scored offline from the traces of one XQuAD-R run
over 1,190 queries, against the shipped weights. The point of reading them
together rather than one sweep at a time is that **every single dial moves the
two groups in opposite directions**, so no row can be read as an improvement
and the only question that means anything is whether one dial trades better
than another.

**Why the trade is not a defect in this fusion.** The lexical channels' heads
are in the query's own language:

| channel | of its top ten, in the query's own language |
| --- | --- |
| `LexicalSegmented` | **90.7%** (10,705 of 11,805) |
| `LexicalNgram` | **67.7%** (8,052 of 11,890) |
| `Vector` | 18.8% (2,235 of 11,900) |

Eleven languages, so 9.1% is what no preference at all would look like. On the
cross-lingual group the query's own language **cannot be the answer** — the one
gold sentence in it is deliberately removed from the ranking — and on the
same-language group it is the *only* answer. So a dial that strengthens lexical
matching must help one group by exactly the mechanism that hurts the other.
That is a property of a parallel corpus scored two ways, not a property of rank
fusion, and it is why the same trade appears in every mechanism tried:

| dial | cross-lingual | same-language |
| --- | --- | --- |
| lexical weight 0 → 0.5 | 0.6335 → 0.5125 | 0.6787 → 0.8453 |
| `k` 0 → 60 | 0.6185 → 0.5638 | 0.7710 → 0.8152 |
| confidence spread 1 → 7 | 0.6114 → 0.6005 | 0.7829 → 0.8051 |
| `Combine` rrf / band / zsum / zmnz | 0.6077 / 0.6114 / 0.6211 / 0.5727 | 0.7556 / 0.7829 / 0.7733 / 0.8257 |

**What this says about the generic advice to raise `k`.** Published guidance on
hybrid search suggests punishing a weak channel's mid-ranks by raising its `k`.
Measured here, globally, `k` is monotone *downward* on cross-lingual nDCG@10:
0.6185 at `k = 0`, 0.6114 at 10, 0.5638 at 60, the last at −0.0476 against the
shipped setting with 33 wins to 658 losses. And a per-channel `k` would be the
same dial as the weight rather than a new one — both scale what a weak
channel's rank is worth — which the two curves confirm: plotted against each
other in the two groups they lie on top of one another to within 0.015, and
`k = 0` is the only point anywhere above the weight's own curve.

**And the default is on the frontier.** Of the ninety-six settings, twenty-six
are Pareto-optimal in the two groups. The shipped weights are not among them on
nDCG alone — the standardised sum at a confidence spread of 5 beats them on
both groups, 0.6134 against 0.6114 and 0.7925 against 0.7829 — but its
cross-lingual `recall@50` is **0.7772 against 0.8962**, and nothing that
survives the recall floor beats the shipped weights on both. That is the first
time this project has been able to say the default is not merely untested.

**Corroboration: it works, and it is the same suppression as the weight.** A
rule was built and deleted here. `Fusion::needing_support` gave a channel's own
last place, rather than what its rank was worth, to a candidate that no other
channel had returned — a per-candidate condition rather than a per-query one,
which was supposed to escape the trade above. At full lexical weight it is
worth a great deal: cross-lingual nDCG 0.2648 → 0.4634 and `recall@50` 0.7919
→ 0.8755, with the same-language group better as well. **At the shipped eighth
weight it is a bit-identical no-op on both groups**, 0 wins and 0 losses over
1,190 queries.

The arithmetic says why. At an eighth an uncorroborated candidate is worth at
most 0.0114 and this rule floors it at 0.0069 — a difference of 0.0045, which
does not reorder a top ten. Corroboration and the weight are the same
suppression applied to overlapping sets, and an eighth weight has already
applied it to everything. Read as a frontier, every corroboration setting lies
on the weight's own curve to within 0.0017. The code was removed; this is the
finding. It came back later for the graph channel alone, measured as a no-op
at the graph weight that ships, and was removed again — see *Fusion designs
measured and removed*.

That also closes the published form of the idea — dropping a lexical candidate
whose dense similarity falls below a threshold. It is the same suppression with
a tuned constant in front of it, so it has the same ceiling, and it needs a
score the vector channel does not compute for a candidate it did not return.

**Two things are premise-absent rather than measured.** The graph channel's
weight is identical in its effect from 0.0 to 1.0 — 0 wins, 0 losses, the same
ranking — because this corpus has no edges, which the edge census prints
directly. So the 0.15 the literature suggests is untested here, not rejected.
And the per-language deficit is uniform rather than concentrated: all eleven
languages are negative cross-lingually, from −0.0066 (de) to −0.0481 (zh), and
all eleven positive on same-language, from +0.0486 (zh) to +0.1669 (th). There
is no script family to write a rule about — including Thai and Hindi, which
share a script with nothing else in the set — so the mechanism is not
look-alike collision between related languages. It is that lexical matching
retrieves the query's own language, and this benchmark defines that as wrong.

### The weight table was taken under rank fusion and never retaken

The sweep that justifies the eighth weight, reproduced in three documents and
in `Fusion::default`'s own comment, was measured when `Combine::Reciprocal`
shipped. `Combine::Banded` ships now, and the figures move. Retaken, same
corpora, one run each:

| | own cross-lingual | XQuAD-R cross-lingual | XQuAD-R same-language |
| --- | --- | --- | --- |
| vector channel alone | 0.8268 | 0.6335 | 0.6787 |
| all four, rank fusion | 0.7910 | 0.6077 | 0.7556 |
| all four, banded — **ships** | **0.7985** | **0.6114** | **0.7829** |

So the deficit against the vector channel alone is −0.0283 on this project's
own corpus (20 wins to 1, p = 0.0046) and **−0.0221** on XQuAD-R (481 wins to
35, p = 0.0001), not the −0.0358 and −0.0258 the rank-fusion figures imply. And
what the lexical channels are worth on the same-language group is **+0.1042**,
not +0.0769. The trade is better in both directions than the documents say.

Every quotation of 0.6077 as "all four fused" is therefore a rank-fusion figure
labelled as the current one, and is corrected wherever it appears.

### What the 2025-2026 fusion literature says, and which of it applies here

Four findings from a survey of the fusion work since the RRF-and-BM25 advice
this project's design started from. They are recorded together because two of
them name this project's measured defect and two of them close off routes it was
about to take.

**The defect has a name and a published numeric twin.** `arXiv:2508.01405`
(VLDB 2026) calls it the *weakest-link* effect and reports a case with the same
shape and nearly the same numbers as this project's: full-text search 0.744,
dense vector search 0.830, the two fused **0.816** — below the dense channel
alone. Its mechanism is the one this project derived from its own arithmetic:
rank fusion is "susceptible to high ranks from a weak path, irrespective of
relevance", and a *small* `k` aggravates it, because a small `k` is what makes
the head of a weak channel's list worth a lot. This project runs `k = 10`.

So the generic advice to raise `k` for the weak channel is directionally right
and insufficient. Raising `k_lexical` shrinks the term a lexical first place
contributes; it does not stop that term being *added* to the vector channel's
own. The addition is the mechanism — at `k = 10` and an eighth weight a lexical
first place is 0.0114, which cannot reach the head alone, but 0.04 + 0.0114
moves a vector-fifteenth candidate to about eighth. Only a zero weight or a
non-additive rule removes an addition, which is what `Fusion::needing_support`
was built to be; it measured as the same suppression as the weight and was
removed — see *Fusion designs measured and removed* below.

**This project's normaliser is the least stable variant in the canonical
taxonomy.** `arXiv:2210.11934` (ACM TOIS 41(4), 2023, and still the systematic
reference) separates fusion functions by what they normalise with, and finds
**TM2C2** — a convex combination of *theoretically* min-max-normalised scores —
beats RRF at p < 0.01 on nearly every dataset it tests (MS MARCO nDCG@1000: RRF
0.425, TM2C2 0.454, semantic alone 0.441). Two of its secondary findings land
directly on decisions recorded above: an *unbounded* normalisation degrades
badly, which is the standardised sum's recall loss arrived at
independently; and a tuned RRF `k` **reverses its own ordering out of domain**,
which is an argument against ever quoting this project's `k = 10` as a
transferable choice. What it costs this project is that `Combine::Banded`
normalises with the *empirical* min and max over the query's own candidates and
then maps onto a band whose width depends on `n` — two query-dependent
statistics where TM2C2 has none.

That was a real finding, and it has since been measured rather than argued.
A theoretical maximum exists for a cosine similarity and not for a BM25 score,
so the version built reads each channel from its score's *infimum* up to the
query's best — TM2C2 itself, and the band read that way. Both lost badly, for a
reason the taxonomy's corpora could not show: this engine's embedder is
anisotropic, so read from minus one its fifty candidates are nearly flat. See
*Fusion designs measured and removed* below.

**The per-query-weight route is closed by measurement, not by argument.**
`arXiv:2608.00183` builds exactly the oracle this project built — a per-query
best mixing weight, worth +21.8% relative — and then tries three ways to predict
it: a random forest over hand-crafted query features at **−0.00004**, a ridge
regression over query embeddings at +0.0022 (p = 0.111), and a confidence
heuristic at **−0.0161**. The best method in their study is training-free RRF at
+0.0090 (p = 0.0046). The conclusion is worth stating as a rule, because this
project had an oracle and was one step from building the predictor: **an oracle
gap is not evidence that a predictor can close it.**

This is why the remedy measured here conditioned on the *candidate* rather
than on the query. "Is this query cross-lingual" is not answerable from a query
— on XQuAD-R because both groups are the same 1,190 queries scored against
different answer keys, and in production because a user asking a question does
not know what language the answer was written in. "Did any other channel also
return this candidate" is answerable from data already in hand.

**And the graph channel's weight has no support anywhere.** The one comparable
published system (`arXiv:2609.01617`) weights its graph channel at **0.15**
against a dense 0.50. This project weights it at **1.0**, equal to the vector
channel, and that number was arrived at by nothing — it is the default for an
unnamed channel. It is now in the offline sweep.

**Two questions this project cares about are genuinely unpublished**, which is
worth recording so they are not researched a fourth time: there is no
2025-2026 comparison of convex combination against RRF *on a cross-lingual
benchmark*, and no evaluation of language- or script-conditional fusion weights
at all. The +0.0283 (p = 0.0046) this project measured for zero lexical weight
on its own cross-lingual group is therefore its own evidence rather than a
confirmation of anyone else's.

### Fusion designs measured and removed

Each of these was built, measured against what ships, and deleted once the
measurement was in. The code, its tests and its sweep rows are gone; what each
was worth is recorded here so the question is not reopened without new
evidence. `Combine::Banded` ships, and `Combine::Reciprocal` is kept, reachable
through `Fusion::with`, as the baseline every figure in this record is quoted
against.

| design | what it was | measured | why it lost |
| --- | --- | --- | --- |
| Standardised sum (`Combine::Standardised`) | each channel's scores centred on their own mean and divided by their own deviation, summed by weight | nDCG@10 better in three of four groups (own cross-lingual +0.0281, p = 0.0037), but XQuAD-R cross-lingual `recall@50` **0.8960 → 0.7765**, through a floor of 0.8000 | A centred score is negative below its channel's mean, so a weak lexical channel's confident hit at `+2` outranks the vector channel's genuine deep hit at `−1`. The head improves and the relevant candidates at ranks ten to fifty fall out, where no reranker can recover them. |
| CombMNZ (`Combine::StandardisedTimesVotes`) | the standardised sum, multiplied by how many channels returned the candidate | own cross-lingual **−0.0392**, 3 wins to 21, p = 0.0008; XQuAD-R −0.0350 cross-lingual and +0.0700 same-language, recall 0.7764 | It multiplies by agreement, and agreement between a confident channel and a worthless one is the weakest-link failure. It pays only where the agreeing channels are independently right, and it inherits the standardised sum's recall loss. |
| TM2C2 (`Combine::Convex`, `arXiv:2210.11934`) | each channel on theoretical min-max — from its score's infimum, 0 for BM25 and −1 for a cosine, to the query's best — convexly weighted, no band | own cross-lingual **0.7791 → 0.5744**, 0 wins to 37 losses, at the paper's own alpha, and no lexical or graph weight in the sweep recovered it; XQuAD-R cross-lingual **−0.1480** | Anisotropy of the embedder. A query's fifty vector candidates sit in the top ~17% of the distance from the cosine infimum to the best of them (median), where BM25's spread across about three quarters of theirs. Read from −1, the vector channel's own ordering is flattened to a few hundredths, and the lexical and graph channels decide among its candidates. |
| Band on theoretical min-max (`Combine::BandedTheoretical`) | `Banded`, with a candidate's place in its channel's band read from the infimum rather than from the channel's worst candidate | own cross-lingual **0.4214** against the shipped 0.7791 | The same anisotropy: every vector candidate lands at the top of its band, so the band keeps the range and loses the channel's ordering. |
| Per-channel confidence (`Fusion::with_confidence`) | each channel's weight scaled by how far its best candidate stands above its own field, `(best − mean) / deviation` over a `spread`, clamped to a `floor` | on rank fusion +0.0118 at best (8 / 0, p = 0.0381) on a narrow plateau; 0.0000 on MIRACL at low spreads. Cross-validated over the whole sweep of about a hundred settings, nothing beats what ships on held-out queries (own p = 0.57, XQuAD-R p = 0.10), and the XQuAD-R choice, spread 5 floor 0.5, is a net loss on the own corpus | A standardised top score cannot exceed `sqrt(n − 1)`, 7.00 over fifty candidates, so a low spread clamps every channel to full weight. Above that, on MIRACL the mildly harmful lexical channels still look confident, so the measure cannot see what it was built to see; everywhere else its gains were trades between groups. |
| Support rule (`Fusion::needing_support`) | a named channel's own last place, instead of what its rank was worth, for any candidate no unnamed channel returned; shipped naming the graph channel | at the shipped graph weight 0.30, 0.0000 on all four groups of the own corpus; bit-identical for the lexical channels at their eighth over 1,190 XQuAD-R queries. On MuSiQue — 10,785 memories, 12,840 live `mentions` edges, 1,000 two-hop questions through `search_reranked_with` with the `accurate` reranker — nDCG@10 0.6834 and `recall@50` 0.8435 with it and without it, **0 wins, 0 losses, 1,000 ties**; the own corpus through the same path, all 157 questions tied | It was kept as insurance for graphs denser than any corpus here, and the dense graph did not need it: three tenths already quiets the channel as far as the rule would. Its one measured benefit was at a graph weight of 1.0 (own cross-lingual 0.5109 → 0.5606), a weight that is itself refuted. It was a no-op by measurement, not by construction — it lowered every candidate only the graph returned, and when the graph returns fewer than about forty candidates that can change which make the top fifty, in principle the top ten — so removing it moved the scores of those candidates and no measured result. |

The figures were taken while the code existed, by the harnesses of the time;
nothing in the tree today can reproduce them, and that is the point of writing
them down.

### What the closest published system does differently, and what that explains

*Jev-Mem* (arXiv 2609.23986, September 2026, CC-BY-4.0, code MIT) is an
agentic-memory system built on the same four ingredients as this one — a vector
index, a lexical index, a typed relation graph, and reciprocal-rank fusion to
pick the anchors. It reports 0.777 on LoCoMo against the strongest baseline's
0.700, with memory construction in 158 seconds and 0.93 seconds a query. It is
recorded here because two of its differences explain results this project
measured and could not account for, and one of its numbers should not be read
the way the paper's headline reads it.

**Its control plane is a hosted API, which is what makes the latency
incomparable.** Every decision — relation typing, query routing, retrieval
budget, candidate relevance, evidence sufficiency — is a call to a hosted typed
decision model, up to sixteen of them within a fifteen-second budget. So 0.93
seconds a query is a figure that includes sixteen possible network round trips
and is not the same measurement as the 359 ms this project spends on four local
cores. The code is MIT and the decisions need a paid key, so the published
result is not reproducible offline.

**Read the adversarial column before reading the headline.** The 11.0% relative
gain is not spread across the benchmark. Against the strongest baseline the
adversarial split moves 0.742 to 0.962, +0.220, where multi-hop is +0.095,
open-domain +0.101, single-hop +0.026 and **temporal is −0.013**. LoCoMo's
adversarial split is the set where the right answer is to abstain, and their
pipeline has an explicit evidence-sufficiency decision with adaptive stopping —
which wins that column close to by construction. The paper reports no
ablations, so nothing in it separates a retrieval gain from an abstention gain,
and the per-column decomposition says most of it is the second. This project
has no abstention at all: `pamin search` returns its best candidates whatever
the evidence looks like. That is a gap worth naming, and it is a different gap
from retrieval quality. Closing it with the calibrated reranker score was
measured and does not ship; see "The transfer test, taken as an abstention
decision" below.

**Why their weighted sum works where this project's did not.** They combine
five terms as a plain normalised weighted sum: embedding similarity, query
relevance, a graph-need probability times a relation weight, information
novelty, and edge weight plus evidence support. A plain weighted sum of
heterogeneous signals is exactly what failed here — the standardised sum
lost cross-lingual recall through the floor, for the reason set out above. The
difference is not the arithmetic. **Four of their five terms are probabilities
on `[0, 1]` emitted by one calibrated model, so they are commensurable by
construction.** This project's terms are a cosine similarity, two BM25-family
scores and a hop-decayed confidence, which are not, and no rescaling makes them
so within one query. That is the same conclusion `Combine::Banded` was derived
from, arrived at from the other direction, and it is the clearest available
argument that a calibrated scorer is an enabling piece rather than a
refinement.

**Why their graph channel pays where this project's has never been shown to.**
Be careful with the comparison, because the obvious version of it is wrong in
this project's favour and in its own. The `0.0000` in the leave-one-out table
above is **not** a measurement of this design: two of the three corpora are
sentence collections, and the harness names their topics *deliberately* unlike
anything in their text, so mention derivation finds nothing and the channel is
handed an empty graph. An arm whose premise is false by construction measures
the premise, not the channel.

**And a sentence that used to stand here is withdrawn.** It read "this
project's own corpus has edges and still contributes `0.0000`", which is false:
the edge census prints zero edges for that corpus too, `retrieval.rs` says it
"can derive no edges at all", and a paragraph of this document three hundred
lines above already said so. That sentence was the one place a premise failure
was laundered into a verdict about the design, and every later reading of the
graph channel in this file rested on it.

**Measured now, on a corpus built to give it a premise.** The `relational`
group adds ten pairs of memories whose answering half is named by a phrase the
other half's prose contains, so mention derivation fires and that project holds
eleven live edges — the first non-empty graph anything here has ever fused.
Fusion alone, nDCG@10, sweeping the weight:

| weight | cross-lingual | lexical | monolingual | relational |
| --- | --- | --- | --- | --- |
| 0.00 | **0.7903** | 1.0000 | 0.9940 | 0.5237 |
| 0.15 | 0.7853 | 1.0000 | 0.9940 | 0.5517 |
| 0.30 | 0.7746 | 1.0000 | 0.9940 | 0.6295 |
| 0.50 | 0.7415 | 1.0000 | 0.9821 | 0.6583 |
| 1.00 — *was the default* | 0.5109 | 0.9885 | 0.9246 | **0.6910** |

**So the channel pays, and the weight it was paying at was catastrophic.** At
1.0 removing it is worth **+0.2794** on the cross-lingual group — forty wins to
nothing, `p = 0.0001`, the largest single effect measured anywhere in this
project — and +0.0694 on the monolingual group, against the 0.1673 it earns on
the twenty queries written to favour it. The whole search path at 1.0 fails
this repository's own guard: 0.9246 on the monolingual group against a 0.9400
floor. Three tenths clears every floor; 1.0 and 0.5 are significantly worse
than it once the whole sweep is priced as one family, 0.15 cannot be told
apart from it at this sample size, and
`pamin_core::fusion` carries the arithmetic.

The weight was never chosen. Every unnamed channel defaults to 1.0 and this one
was simply never named, so it voted as loudly as the dense channel on the
strength of a single derived mention — and nothing could see that, because
every corpus handed it an empty graph. **That is what a premise failure costs.**
It does not produce a wrong number; it produces no number, for as long as
nobody notices that the zero is the corpus answering.

The other sample still stands and still says nothing: the LOCOMO `pamin-ledger`
arm — built expressly so the graph could reach the rest of an exchange — scored
0.523 against 0.518, twenty-one discordant questions against twenty,
`p = 1.000`.

Read the `relational` figures as what they are. Twenty queries is a collapse
detector, not a regression detector, and they were written in this repository
*to make this channel look useful* — which is why the weight was chosen from
which rows the family correction can separate rather than from a four-group
mean that would have let
the purpose-built group pick its own weight.

Against that, three differences in their design are specific enough to test:

- **Relation type decides traversal, not just provenance.** Nine `EdgeKind`s
  exist here and `graph::expand` can filter on them, but nothing in the search
  path ever does: every kind is walked identically, and the only kind the engine
  ever derives is `Mentions`. Everything else needs an explicit `pamin link`.
  They insert a typed edge only when its relation probability clears 0.60 and
  then route per type.
- **Traversal is gated on predicted need.** Depth is derived per query from a
  predicted multi-hop requirement, and budget is allocated across relation views
  by a predicted usefulness. Here `Depths::graph` is 2 for every query and
  `HOP_DECAY` is 0.5 for every arrival, so the walk costs the same on a query
  that cannot use it as on one that can.
- **Expansion stops adaptively**, on sufficiency, novelty and contradiction,
  inside hard caps on nodes, edges, decisions and wall time. Here the caps exist
  — `MAX_SEEDS`, `MAX_FRONTIER`, `MAX_DEPTH` — and the stopping rule does not.

What all three share is a per-query decision about whether to do the work at
all, which is the same calibrated judgement the fusion and the abstention gap
both want, reached a third time. **And the first thing to do is not a
refactor**: it is a diagnostic that says what fraction of queries have a
relevant memory reachable across an edge and *not* already found by another
channel. If that fraction is near zero on a corpus with a real graph, no
traversal policy can recover it, and the channel's cost should be removed rather
than tuned.

**One constant is corpus-dependent, and both values are right.** They seed
expansion with reciprocal-rank fusion at `k = 60`, the field's convention. This
project measured `k = 10` better on its corpora and `k` a two-sided trade
rather than a plateau. Neither figure generalises, which is the useful finding:
`fusion::DEFAULT_K` is a property of a corpus's channel agreement, not a
constant to inherit.

### How a constant here is chosen, and what that changed

Every fusion parameter in this repository was chosen the same way: run one
search pass per query, replay eighty to ninety-six fusion settings offline from
the trace, score each on *all* of a corpus's queries, pick one by eye --
usually "the knee", where one group's gain stops exceeding another's loss --
and report the chosen setting with the numbers it was chosen by. That is a fit
with ninety degrees of freedom scored on its own training data, and it had four
defects, each with a name in the literature and each now fixed.

**No split.** Fuhr (*Some Common Mistakes in IR Evaluation*, SIGIR Forum
51(3), 2017) states that the tuning set must be disjoint from the test set;
Cawley and Talbot (JMLR 11, 2010) measure what skipping it costs and show it
can reorder competing methods. This repository's calibration arm already split
by query and said why. The fusion sweeps never did. They now run the selection
as a procedure: five folds stratified by group, the rule sees four and is
scored on the fifth.

**A rule chosen after looking.** The knee is an expected-utility rule under an
unstated uniform prior, and it depends on how densely the grid was sampled. The
rule is now fixed in code before any table is read and stated so it can be
argued with: maximise the mean over groups of nDCG@10, every group equal, ties
to what ships.

**Ninety uncorrected tests.** Each row reported its own p against the shipped
setting, so about four rows a table cleared 0.05 by chance. Every table now
carries a Westfall-Young family-adjusted p, the correlation-aware procedure
evaluated for exactly this design by Boytsov, Belova and Westfall (SIGIR 2013),
and the smallest difference each row's queries could detect at 80% power
(Sakai, IRJ 19, 2016). The twenty-query relational group can see about 0.08.

**An anti-conservative test.** The paired bootstrap-shift test every p here came
from is measured by Urbano, Lima and Hanjalic (`arXiv:1905.11096`, 500 million
simulated p values) at 0.059 actual against 0.050 nominal. It is now the exact
paired sign-flip randomisation. The difference is not subtle: three queries all
improving is `p = 0.25` under it and was `p = 0.0001` under the bootstrap, and
this document has carried rows such as `4W/0L/16T p = 0.0618` where the
smallest attainable p on four untied queries is 0.125.

**What the honest procedure says, which is the part worth keeping:**

| | own corpus | XQuAD-R |
| --- | --- | --- |
| best row, scored on the queries that chose it | 0.8562 | 0.7033 |
| the procedure, on queries it did not choose on | 0.8534 | 0.6998 |
| what ships, on the same queries | 0.8495 | 0.6972 |
| procedure against what ships | +0.0025, 19W/3L, `p = 0.57` | +0.0026, 231W/**320L**, `p = 0.10` |
| rows chosen across the five folds | two different | three different |

**Chosen honestly, nothing in the sweep beats what ships on either corpus, and
the folds do not agree on what they would choose.** The in-sample optimism is
small here -- about 0.003 -- because the grid's rows are close to one another;
what the procedure changes is not the size of the number but whether it is a
result. A previous reading of this same sweep found two asymmetric lexical
weightings "no worse than shipped on all four groups" of the own corpus and
nearly moved the default on it; that is exactly what choosing the best of
eighty-five on the queries that grade them produces.

One decision survives the correction, and it is the large one: the graph
channel's weight moving from 1.0 to 0.30, significant after family adjustment
on the cross-lingual and monolingual groups. Its stated reason -- the knee --
did not survive, and `pamin_core::fusion` now gives the one that does.

One criticism the literature made of this procedure does not apply, and is
recorded so it is not re-litigated. Offline replay was suspected of *support
deficiency* -- that the candidate set is authored by the shipped setting, so a
challenger's documents are invisible. It is not: the replay requests more than
four times the channel depth and asserts on every query that the fused list did
not reach that limit, so every candidate every channel returned is in the trace
and a replayed setting is an exact simulation, not an approximation.

And one limit no procedure here can remove. Bruch, Gai and Ingber
(`arXiv:2210.11934`) tuned per-channel rank constants on proper validation
splits and watched them lose 8 to 10% out of domain, the optimal direction
reversing between corpora. Cross-validation over these corpora estimates
performance on these corpora. The cheapest thing that speaks to transfer is to
choose on one corpus and test on another, and **neither direction transfers**:

| chosen on | the row the rule picks | on the other corpus |
| --- | --- | --- |
| own corpus | drop the n-gram channel | XQuAD-R cross-lingual +0.0094, same-language **−0.0361** (9 wins to 155, family `p = 0.0001`): a net loss |
| XQuAD-R | confidence, spread 5, floor 0.5 | own corpus cross-lingual +0.0117 and relational −0.0559, neither significant: a net loss |

Each choice improves the corpus that made it and costs the one that did not,
which is Bruch, Gai and Ingber's result reproduced on this system. So the
lexical and combiner settings stay where they are, and not for want of looking:
every candidate the sweep can offer either fails to beat them where it was
chosen or fails to carry to the next corpus. The graph weight's move is the
exception because it is neutral wherever the graph is empty and significantly
positive where it is not -- there is no corpus here on which it costs anything.

### Every accuracy figure here is a difference of means

Stated as its own section because it applies to all of them, including the
ones this decision record treats as settled.

Not one comparison in this project has ever been tested. The fusion weight
moved on +0.0056 on MIRACL; the reranker is priced at −0.0152 on the same
corpus; the segmentation verdict rests on a recall column; the reranker's depth
was settled on +0.0369 against +0.0110. All of those are differences between
two averages, and an average cannot distinguish every query moving slightly
from one query moving a great deal.

The graph-vector fusion paper above is what makes this concrete rather than
pedantic. It reports rank fusion beating vector-only by +1.7 points, and then
reports the same comparison as **15 wins against 6 losses at p = 0.078** —
while its own method's *smaller* mean gain is **8 wins against 1 loss at
p = 0.039**. Read as means, the wrong method wins.

`crates/pamin-engine/tests/statistics/mod.rs` now reports wins, losses, ties
and a paired randomisation p alongside every mean -- a bootstrap until the
section below found it anti-conservative -- and the cross-lingual harness
fails if reranking's gain is not significant rather than merely small. Until
each figure below has been re-taken through it, **a small difference in this
document is a difference of means and nothing more**.

The first two re-taken were the two that most needed it, and **both survive**.
Reranking on XQuAD-R is +0.0403 cross-lingual at 547 wins against 206 losses,
p = 0.0001, and −0.0061 same-language at 3 wins against 19 losses, p = 0.0007.
On MIRACL it is −0.0152 at 37 wins against 57 losses over 482 queries,
p = 0.0129. The reranker's cost on single-language retrieval was acted on
before it was tested; testing it did not take it away.

What the test adds is the shape the means hid. The same-language damage is not
diffuse — the pass reaches twenty-two of XQuAD-R's 1,190 same-language queries
and makes nineteen of them worse. It looks small in the mean only because
confining the pass to unlexical candidates keeps it away from almost every
query, which is that confinement working exactly as its own note claims. On
MIRACL, where there is no other language for a candidate to be in, it reaches
ninety-four of 482.

The third was the fusion weight's +0.0056 on MIRACL, and it did **not**
survive: 77 wins against 72 losses, p = 0.2300. The default it was used to
justify is supported by two other corpora and not by that one; the section on
the fusion weight above now says so.

### Three recall channels, not seven

An earlier channel list had seven entries. Four were redundant, and two of those double-counted against modifiers the same design already applied after fusion:

- **Temporal** and **pinned/important** were to be expressed as post-fusion modifiers instead. Running them as channels as well would count the same signal twice. "Facts valid at time T" is a filter over other channels, not an independent recall source.
- **Curated notes** and **page nodes** already enter the projection index. A separate channel queries the same data twice and splits one population into several, which dilutes results and forces the redundancy penalty to reason across populations.

What remains:

```text
recall channels (3)   lexical, vector, graph
document types        topic / span / page_node / note   (a filter)
agentic primitives    grep, read by id, navigate, typed query
```

The post-fusion modifiers this list used to carry — recency, importance and
worth, source quality, a redundancy penalty — are gone, and the reason is worth
recording because it is not the reason the list was shortened. Importance and
worth were implemented: `Modifiers::apply` multiplied every result by
`1 + 0.2 * importance` and by `1 + 0.2 * worth`. Both were columns the
repository read and **no code path anywhere wrote**, so both factors were
exactly 1.0 on every search this project has ever run, and the trace lines for
them were already suppressed on the grounds that they said nothing. A modifier
over a constant is not a ranking signal; it is a multiplication. With the
modifiers gone nothing read them either, so `RetrievalSignals`, which carried
them and two access counters -- equally never written -- onto every state a
search loaded, went too, and migration V11 drops the five columns -- after
checking that every row still holds the default it was inserted with, and
refusing with the state named if one does not. Restoring the feature starts
with a write path, which can add back the column it writes, not with a
multiplier.

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

The field indexes `zvec`'s default of two-character grams, and that was measured against three-character grams and against two and three together. Each is one tokenizer parameter (`{"ngram_min":3,"ngram_max":3}` or `{"ngram_min":2,"ngram_max":3}`) and a rebuild. Measured through `search_reranked` against a rule written before the runs, neither ships. Both are significantly worse on XQuAD-R's cross-lingual group (−0.0055 and −0.0035, p = 0.0001 each), neither is significantly better on any corpus as a whole, and three-character grams also lose Thai cross-lingual. Two and three together is not the cheap hedge it looks like: on XQuAD-R and MuSiQue it makes the index 1.50 to 1.67 times larger and the rebuild 1.60 to 1.67 times slower. The figures are in [what the project measures](../measured.md).

### Embeddings: profiles, and two different meanings of INT8

"INT8" names two operations whose costs differ by an order of magnitude, and conflating them is easy:

| | What it is | Measured cost | State |
| --- | --- | --- | --- |
| Model weight INT8 | ONNX weights quantized for CPU inference | 2.7–3.4x faster, under 0.5% MTEB | **On, by default** |
| Stored vector INT8 | Output embeddings stored as int8 rather than float32 | No recall loss at all, half the query and half the build, **30% more disk** | **Off, because it is a disk loss** — see below |

Weight quantization is a trade worth taking, and the default profile takes it. The registry publishes no quantized variant for multilingual E5, which is why the two E5 profiles still run full precision and why an earlier version of this decision recorded the trade as unavailable. It is available for BGE-M3, through a joint int8 export (`gpahal/bge-m3-onnx-int8`, MIT, exported from the MIT-licensed base model), and the difference is what makes that profile the default: 560 MB resident against the full-precision export's 2.2 GB, 35 ms a query, and 0.6550 cross-lingual nDCG@10 on Påmin Memory's evaluation corpus against the full-precision 0.6720 — both at the lexical weight of that day, a half.

### The embedder, surveyed again: one candidate, and the leaderboard would have picked wrong

Surveyed in September 2026 against the models released since BGE-M3, with the
licence read from each card's metadata. Scores are recomputed from the
per-task files in `embeddings-benchmark/results`; "cross" averages the subsets
whose query and document languages differ.

| model | licence | MMTEB retrieval | MIRACL-HN | MLQA cross | Belebele cross |
| --- | --- | --- | --- | --- | --- |
| BGE-M3 (shipped) | MIT | 54.6 | **69.6** | 74.7 | 77.0 |
| Harrier-0.6B (Microsoft, 2026) | MIT | **70.8** | 66.4 | 72.7 | 77.0 |
| pplx-embed-v1-0.6b (Perplexity, 2026) | MIT | 65.4 | 68.6 | **79.1** | 72.7 |
| granite-embedding-311m-multilingual-r2 | Apache-2.0 | 65.2 | 59.8 | 66.9 | 64.8 |
| Qwen3-Embedding-0.6B | Apache-2.0 | 64.6 | 61.2 | 72.8 | 67.6 |
| multilingual-e5-large-instruct | MIT | 57.1 | 57.7 | 76.0 | **79.9** |

EmbeddingGemma (Gemma terms) and jina v3/v5 (CC-BY-NC-4.0) were excluded on
licence. BGE-M3's low retrieval average is reasoning, English and long-context
tasks; on the four multilingual Wikipedia tasks it is still the best under a
billion parameters.

Then measured offline, vector channel alone, exact cosine, one text per call
as the product embeds, against BGE-M3's int8 export, nDCG@10, paired:

| model | XQuAD-R cross | XQuAD-R same | MuSiQue | own, cross (43) |
| --- | --- | --- | --- | --- |
| BGE-M3 int8 | 0.6348 | 0.6725 | 0.6262 | 0.8275 |
| pplx-embed-v1-0.6b, own int8 | **+0.0251**, p = 0.0002 | **+0.0457**, p = 0.0001 | **+0.0647**, p = 0.0001 | +0.050, p = 0.054 |
| Harrier-0.6B, own int8 | **−0.378** | +0.185 | | −0.163 |
| multilingual-e5-large-instruct | **−0.435** | +0.200 | −0.030 | −0.289 |
| granite-311m-r2, IBM's int8 | −0.145 | −0.050 | −0.002 | +0.023 |

The pipeline reproduces the engine's own BGE-M3 figures within 0.0013 on three
arms and 0.0062 on XQuAD-R same-language, where the int8 export itself moves by
that much with the runtime's optimisation level (cosine 0.985 between builds).

**The two highest-ranked models collapse across languages**, and MTEB cannot
see it: it scores each language pair against a corpus in one language, while a
memory store holds all its languages in one pool. Harrier and mE5 rank a
same-language non-answer above the answer in another language -- two thirds
and three quarters of their cross-lingual top ten are in the query's own
language, where a language-blind ranking would put one in eleven -- and
removing their query instructions does not change it.

**pplx-embed-v1-0.6b is the only candidate worth an end-to-end trial.** Its
gains hold on every corpus, but four things stand between that and a default:
it was measured on the vector channel alone, and fusion and the reranker
already recover part of what a better vector buys; its published 8-bit export
runs 8-10x slower on this CPU, so shipping it means shipping a quantization of
our own (dynamic int8 on every layer but `down_proj`, cosine 0.995 to fp32,
against the shipped BGE-M3 export's 0.980); Greek queries are worse by 0.071
(p = 0.002, surviving correction over eleven languages); and a query costs
about 2.2 times BGE-M3's, a passage 2-3 times, and every workspace would have
to be re-embedded.

### BGE-M3's other outputs, measured: none of them ships

One forward pass of BGE-M3 returns three things — a dense vector, a sparse
vector of per-token lexical weights, and one multi-vector (ColBERT) embedding
per token — and the model was trained on passages up to 8,192 tokens. The
product keeps the dense vector and truncates at 512 (`JOINT_MAX_TOKENS`). Each
of the other three was measured in September 2026 against what ships, under
selection rules written down before any of it was computed.

Every comparison is paired per question: sign-flip randomisation, 10,000
draws, two-sided, so 0.0001 is the floor. "Significantly worse" means a
negative mean at p < 0.05 without correction, which is deliberately strict
against the change. The groups are this project's own four, XQuAD-R's two and
MuSiQue's 1,000 two-hop questions.

**This was measured below the entry point, and the reason is the tunable.** A
channel weight and a reordering of the shortlist are not settings `pamin
search` accepts, so a replay stands in for the last two stages: `Banded`
fusion, `engine::rerankable` and `engine::place` over the engine's own
per-channel candidates, with the `accurate` cross-encoder scoring the same
shown set at 256 tokens in length-sorted batches of eight. On XQuAD-R the
replay reproduces the product, 0.6608 / 0.7842 against 0.6613 / 0.7838 at a
rerank depth of twenty. On MuSiQue it reads 0.6634 against the product's
0.6834: the replay leaves out the graph seed text the engine shows the
cross-encoder, and MuSiQue is the only corpus of the two with a graph. So its
MuSiQue figures are differences between arms of the same replay, not product
figures. Everything below is at the rerank depth of thirty that ships; for the
sparse channel, twenty gives the same signs.

**The sparse channel helps within a language and hurts across one.** Added as
a fifth channel, banded like the two BM25 channels, nDCG@10 after the rerank
against what ships:

| sparse weight | own cross-lingual (43) | XQuAD-R cross-lingual (1,190) | XQuAD-R same-language (1,190) | MuSiQue two-hop (1,000) |
| --- | --- | --- | --- | --- |
| 0.0625 | **−0.0078**, p = 0.0045 | **−0.0114**, p = 0.0001, 25 wins / 373 losses | **+0.0141**, p = 0.0001, 91 / 9 | **+0.0035**, p = 0.0028 |
| 0.125 | **−0.0248**, p = 0.0002 | **−0.0213**, p = 0.0001 | **+0.0229**, p = 0.0001 | **+0.0051**, p = 0.0018 |
| 0.25 | **−0.0549**, p = 0.0001 | **−0.0415**, p = 0.0001 | **+0.0340**, p = 0.0001 | **+0.0097**, p = 0.0001 |

The relational, lexical and monolingual groups do not move significantly at
any weight in the table. The cause is visible in the channel alone: it scores
0.7684 nDCG@10 on XQuAD-R's same-language group and 0.0864 on its cross-lingual
one.
It is a third lexical channel, it matches tokens, and a token does not cross a
language. Every weight trades one group for another, and the smallest weight
still costs the cross-lingual group 373 queries against 25.

The rule chose on what the reranker is handed, `recall@30` of the fused list,
and there the sparse channel is significantly worse on XQuAD-R cross-lingual at
every weight (−0.0021 at 0.0625, p < 0.001). Using it to replace one of the
two BM25 channels instead of adding it is significantly worse there too, fused
nDCG@10 −0.0081 in place of the segmented channel and −0.0128 in place of the
n-gram one. No weight was admissible on any two corpora, so leave-one-corpus-out
had nothing to carry to the third, and the rule says not to build it.

**A sparse model that does cross languages exists, and its licence rules it
out.** MILCO (`omai-research/milco-650m` and `-300m`, arXiv 2510.00671) maps
every language into one English lexical space and reports MKQA `recall@100` of
76.6 against BGE-M3 sparse's 45.3, which is the gap measured above. Its cards
say `apache-2.0`, but both released checkpoints carry `naver/splade-v3`'s MLM
head (24M parameters, CC-BY-NC-SA-4.0; its bias correlates with splade-v3's at
1.0000 read from the released tensors), its second training stage distils
scores from a `license: gemma` reranker, and its training code carries no
licence. By the leaf-and-base rule above that is the same refusal as
`bge-reranker-v2-gemma` and the splade-v3 family. It also has no ONNX export,
and it would add a second XLM-R-large forward to every query and every write.

Cost is not the reason. The sparse vector falls out of the forward pass the
product already runs, and stores at 129 to 533 bytes a memory (a 32-bit id and
weight per non-zero, on this project's corpus and MuSiQue) against 4,096 for
the dense vector. The index could not hold it as things stand anyway:
`zvec-rust` 0.7.2 declares sparse field types and accepts a sparse sub-query,
but its `Doc` has no setter for a sparse field, so writing one would go
through the raw FFI.

**The multi-vector output loses to the cross-encoder it would replace.**
Reordering the same shown set, nDCG@10 against the shipped `accurate` tier:

| reordered by | own cross-lingual | XQuAD-R cross-lingual | XQuAD-R same-language | MuSiQue two-hop |
| --- | --- | --- | --- | --- |
| `accurate` cross-encoder, shipped | 0.8162 | 0.6673 | 0.7844 | 0.6596 |
| M3 "All", the card's 0.4 dense + 0.2 sparse + 0.4 multi-vector | −0.0033, p = 0.8344 | **−0.0527**, p = 0.0001 | +0.0006, p = 0.7164 | **−0.0167**, p = 0.0001 |
| M3 multi-vector alone | −0.0030, p = 0.8337 | **−0.0531**, p = 0.0001 | −0.0025, p = 0.1758 | **−0.0200**, p = 0.0001 |
| cross-encoder and "All", rank-fused, weight 0.25 | −0.0030, p = 0.4786 | +0.0011, p = 0.1294 | −0.0000, p = 1.0000 | **−0.0022**, p = 0.0101 |
| the same at 0.5 | −0.0006, p = 0.8988 | **−0.0030**, p = 0.0177 | +0.0004, p = 0.5800 | −0.0025, p = 0.0812 |
| the same at 1.0 | +0.0121, p = 0.2832 | **−0.0110**, p = 0.0001 | +0.0009, p = 0.3549 | **−0.0080**, p = 0.0001 |

The paper's weights (1, 0.3, 1) give −0.0501 and −0.0172 on the two groups
that move. Against fusion with no rerank at all, "All" gains nothing
significant on XQuAD-R cross-lingual (+0.0032, p = 0.1553) and loses MuSiQue
(−0.0070, p = 0.0001). That is the `mLateOn` result recorded below, from a
second model: late interaction reorders this pipeline's candidates no better
than the fusion order it replaces. The rank-fused weight was chosen
leave-one-corpus-out, and the choice does not hold out: no weight was
admissible on the other two corpora with this project's own held out, and the
two folds that chose one lose on the corpus they did not see — XQuAD-R
cross-lingual −0.0030 (p = 0.0177) at 0.5, MuSiQue −0.0022 (p = 0.0101) at
0.25.

**And it is not cheaper either way it could be built.** Stored, the per-token
vectors at int8 are 19.3 KiB a memory on this project's corpus, 42.9 KiB on
XQuAD-R and 117.7 KiB on MuSiQue — 545 MiB and 1,240 MiB for the two external
corpora, against 4 KiB of dense vector a memory. Computed at query time
instead, each candidate not cached needs a BGE-M3 forward pass, which costs
about what the cross-encoder's pair does: 130.8 ms against 134.8 on XQuAD-R
sentences and 376.2 against 299.0 on MuSiQue paragraphs, medians at batch one
from a single run. A cache of those vectors over the harness's stream of
questions hits 46% of reranked candidates on XQuAD-R and 38% on MuSiQue at 500
entries, and 64% and 77% with no bound.

**The longer window buys nothing these corpora can see.** None of this
project's 230 memories or XQuAD-R's 13,014 sentences exceeds 512 tokens. On
MuSiQue 32 of 10,785 paragraphs do (0.3%), the longest at 597. On the 46
questions whose relevant paragraph is one of those, embedding it whole moves
the vector channel alone, exact search, from 0.5323 to 0.5348 nDCG@10 (+0.0025,
p = 0.820, 4 wins, 3 losses), with `recall@50` unchanged; the sparse output
moves by +0.0010 (p = 1.000). The forward pass grows faster than the text: 223
ms at 128 tokens, 1,458 at 512, 2,874 at 1,024, 10,670 at 2,048 and 24,459 at
4,096, the fastest of three calls on one text (of two at 4,096). The window
stays at 512.

### pplx-embed-v1-0.6b on the shipped path: measured, not adopted

**Status: rejected.** BGE-M3 remains the default. The accuracy gain held up
with the export a product would ship, but the maintainer weighed it against a
3.4× slower write path and chose BGE-M3; the decision and the figures it rests
on close this section.

The survey above named one candidate and four reasons it was not yet a
default, the first being that it had been measured on the vector channel
alone. The end-to-end trial ran it through `search_reranked` at the `accurate`
tier on the XQuAD-R and MuSiQue harnesses. It used an experimental `pplx`
profile over our own int8 export (52bc0d9) and a harness option that pairs one
profile's saved per-question scores with another's (783551a), and neither is on
the default branch. It ran at the rerank depth of twenty that shipped at the
time. Both indexes embed `name: content`:

| | BGE-M3 | pplx | difference | wins / losses | p |
| --- | --- | --- | --- | --- | --- |
| MuSiQue two-hop (1,000), nDCG@10 | 0.7129 | 0.7451 | **+0.0322** | 306 / 218 | 0.0001 |
| MuSiQue two-hop, `recall@50` | 0.8570 | 0.8865 | **+0.0295** | 94 / 39 | 0.0001 |
| XQuAD-R same-language (1,190), nDCG@10 | 0.8030 | 0.8193 | **+0.0162** | 168 / 118 | 0.0076 |
| XQuAD-R same-language, `recall@50` | 0.9605 | 0.9580 | −0.0025 | 13 / 16 | 0.7137 |
| XQuAD-R cross-lingual (1,190), nDCG@10 | 0.6714 | 0.6643 | −0.0072 | 421 / 486 | 0.0986 |
| XQuAD-R cross-lingual, `recall@50` | 0.9005 | 0.8946 | −0.0059 | 175 / 180 | 0.1934 |

**What survived fusion and the reranker is a third to a half of the model-alone
gain within a language, and none of it across languages.** Alone, the vector
channel had gained +0.0647 on MuSiQue, +0.0457 same-language and +0.0251
cross-lingual. The prediction written before this run was +0.015 on MuSiQue (a
range of −0.005 to +0.035), +0.035 same-language, and +0.005 cross-lingual, not
significant. MuSiQue came in at the top of its range and same-language at half
the prediction. Cross-lingual was not significant, as predicted, but its sign
is negative.

**The first figures this trial printed were wrong, and what was wrong was the
baseline's text, not its model.** They were +0.0029 cross-lingual (p = 0.5351)
and +0.0355 same-language (p = 0.0001) on XQuAD-R. Those paired pplx on a fresh
index against BGE-M3 indexes built before 0c2ff9f, which embed content alone
and keep doing so until `pamin reindex` rebuilds them. Measured alone on the
same harness, the encoding moves BGE-M3 from content to `name: content` by
+0.0101 cross-lingual (444 wins / 321 losses), +0.0193 same-language
(141 / 80) and +0.0294 on MuSiQue (249 / 166), each at p = 0.0001. So the
+0.0355 was +0.0193 of encoding and +0.0162 of model, and the +0.0029 was an
encoding gain covering a model loss. The harness now refuses to pair runs over
different encodings (783551a). The prediction for the encoding was wrong in
sign on XQuAD-R (−0.003 and −0.002, on the reasoning that names like `de:12:3`
are noise to the model) and low on MuSiQue (+0.012). It also separates two
decisions: a BGE-M3 workspace built before 0c2ff9f gains the encoding figures
from `pamin reindex` with no change of model.

What it costs:

| | BGE-M3, shipped | pplx, own int8 export |
| --- | --- | --- |
| query embedding | — | 2.24× to 3.02× BGE-M3's: the median per-query ratio in each of four runs of 375 queries |
| resident after loading and 20 queries | 628 MiB (299 anonymous, 329 file-backed) | 886 MiB (167 anonymous, 720 file-backed) |
| model on disk | 560 MiB | 850 MiB |
| passage embedding | — | 2 to 3 times BGE-M3's, from the survey |

The latency is the embedding call through the crate's own encoder, not a whole
search. It was taken on the shared four-core machine at a load average of 17 to
20, so only the per-query ratio is quoted: BGE-M3's own median moved between
40.8 and 90.8 ms across the four runs. The resident figures are the median of
three alternating rounds, one model per fresh process, which agree to within
1.5 MiB.

**Combining the two models was measured, and it is not proposed.** It used the
same replay as the section above, with its MuSiQue caveat, under rules written
before any combined result. Every arm keeps the two BM25 channels at 0.125 and
the graph at 0.30:

- **A** is what ships: BGE-M3 dense at 1.0, over an index of content, as the
  benchmark workspaces were built. **A′** is A over a fresh BGE-M3 index of
  `name: content`, which is what a new install builds.
- **B** is A plus BGE-M3's sparse channel at 0.0625.
- **C** is pplx's dense vector in place of BGE-M3's, at 1.0.
- **D** is C plus BGE-M3's sparse channel at 0.0625.
- **E** is BGE-M3 at 1.0, pplx at 0.5 and the sparse channel at 0.0625, and
  **E′** is E over the fresh BGE-M3 index.

D's and E's weights are what the rule chose on all three corpora. The
leave-one-corpus-out folds disagreed, choosing a sparse weight of 0.125 for D
and 0.25 for E with this project's corpus held out, so by the same rule neither
arm has an established sparse weight. nDCG@10 after the rerank, at the depth of
thirty that ships, paired against A:

| arm | own cross-lingual (43) | own relational (20) | XQuAD-R cross-lingual | XQuAD-R same-language | MuSiQue two-hop |
| --- | --- | --- | --- | --- | --- |
| A | 0.8162 | 0.6480 | 0.6673 | 0.7844 | 0.6596 |
| A′ | +0.0201, p = 0.0995 | +0.0353, p = 0.3065 | **+0.0051**, p = 0.0161 | **+0.0160**, p = 0.0001 | **+0.0218**, p = 0.0001 |
| B | **−0.0078**, p = 0.0045 | −0.0020, p = 1.0000 | **−0.0114**, p = 0.0001 | **+0.0141**, p = 0.0001 | **+0.0035**, p = 0.0028 |
| C | **+0.0644**, p = 0.0001 | −0.0135, p = 0.8611 | +0.0049, p = 0.2404 | **+0.0230**, p = 0.0005 | **+0.0679**, p = 0.0001 |
| D | **+0.0471**, p = 0.0046 | −0.0320, p = 0.6664 | −0.0073, p = 0.0765 | **+0.0351**, p = 0.0001 | **+0.0695**, p = 0.0001 |
| E | **+0.0347**, p = 0.0014 | −0.0054, p = 0.7534 | **+0.0085**, p = 0.0001 | **+0.0243**, p = 0.0001 | **+0.0359**, p = 0.0001 |
| E′ | **+0.0478**, p = 0.0001 | +0.0312, p = 0.6124 | **+0.0135**, p = 0.0001 | **+0.0332**, p = 0.0001 | **+0.0458**, p = 0.0001 |

The lexical group is 1.0000 in every arm, and the monolingual group is 0.9940
in every arm except C, which is 0.9881 (one query, p = 1.0000). Against A′,
the baseline a new install gets, C still has no group significantly worse:
+0.0443 own cross-lingual (p = 0.0023), −0.0002 and +0.0069 on XQuAD-R's two
groups (p = 0.9557 and 0.2542), and +0.0461 on MuSiQue (p = 0.0001). D loses
XQuAD-R cross-lingual to A′, −0.0124 (p = 0.0013). E and E′ have no group
significantly worse than A or A′ either. But they need both models, 628 + 886
MiB resident and 560 + 850 MiB on disk, and they score below C on MuSiQue
(0.6955 and 0.7054 against 0.7275). D needs both as well, because its sparse
channel is BGE-M3's, and B loses XQuAD-R cross-lingual. So among the arms that
load one model, C is the only one with no group significantly worse than A or
A′, and it was the configuration taken to a decision.

**The replay and the product agree on direction, not to the third decimal.**
Paired against a named BGE-M3 index at depth twenty, the replay gives C
+0.0450 on MuSiQue where the product gave +0.0322, and +0.0069 (not
significant) same-language on XQuAD-R where the product gave +0.0162. The
decision rests on the product-path tables in this section. The replay only
ranks the combinations against one another.

**The four open items were settled before the decision** (the adoption
branch, `claude/perf-33-pplx-default-7gafc1`, and its PR #98 were closed
unmerged and hold the code and logs):

1. **The export.** Perplexity's own 8-bit ONNX export at a pinned revision
   runs fast once each of its 196 `MatMulNBits` nodes is told to compute in
   int8 (accuracy level 4), set at load time with no Python. Against the
   full-precision export over 400 texts its mean cosine is 0.99925, above our
   own export's 0.99571, and on the XQuAD-R path the two exports do not differ
   (+0.0019 cross-lingual, p = 0.097; +0.0009 same-language, p = 0.50).
2. **Re-embedding.** A replaced model's index can be rebuilt beside the old
   one in the background while the old one serves.
3. **Greek.** Per query language on the shipped path, Holm-corrected across
   eleven languages, Greek cross-lingual is −0.0512 (30 wins / 59 losses,
   corrected p = 0.006) and Chinese cross-lingual +0.0685 (corrected p = 0.001);
   nothing else moves. Greek alone accounts for the whole cross-lingual −0.0045.
4. **Fusion weights.** A leave-one-corpus-out re-sweep written before it ran
   kept every weight: the three folds each chose a different setting, and the
   procedure lost −0.0020 pooled over 3,537 held-out questions (p = 0.0023).

**The decision, on the export that would ship, at the shipped depth of
thirty.** Accuracy, paired per question:

| | BGE-M3 | pplx | difference | wins / losses | p |
| --- | --- | --- | --- | --- | --- |
| MuSiQue two-hop (1,000), nDCG@10 | 0.7131 | 0.7470 | **+0.0339** | 300 / 211 | 0.0001 |
| MuSiQue two-hop, `recall@50` | 0.8570 | 0.8855 | **+0.0285** | 93 / 41 | 0.0001 |
| XQuAD-R same-language (1,190), nDCG@10 | 0.8041 | 0.8199 | **+0.0157** | 162 / 111 | 0.010 |
| XQuAD-R cross-lingual (1,190), nDCG@10 | 0.6745 | 0.6699 | −0.0045 | 412 / 493 | 0.24 |

Cost, in two long-lived processes opened as `pamin serve` opens them, the two
models alternated per query on 100 MuSiQue and 100 XQuAD-R queries at the CLI
defaults (load average 5.6 to 7.2 on four cores):

| | BGE-M3 | pplx | ratio |
| --- | --- | --- | --- |
| query embedding, median | 46–47 ms | 119–138 ms | 2.5–3.1×, p = 0.0001 |
| whole search, both corpora, median | 2,062 ms | 2,112 ms | 1.02× (geometric mean of per-query ratios), p = 0.83 |
| passage embedding, a write | 135 ms | 523 ms; 459 ms with the passes of a batch run side by side | 3.4–4.1× |
| resident | 628 MiB | 797 MiB | |

**Why the whole search did not slow down, and why that did not decide it.**
The query embedding is 1 to 7% of a search that reranks, and pplx changed how
often the reranker runs at all (57 of 100 MuSiQue queries against 66). But
every memory written pays the passage cost before it becomes searchable, and
the cost sits in the kernel: this export's `MatMulNBits` runs on MLAS's
AVX512-VNNI path, while the AMX path serves only the `MatMulInteger` that
BGE-M3's export uses. Batching does not recover it, since a pass's cost per
token is flat and each passage already runs alone. Only a per-column int8
export of our own reaches AMX, which means hosting weights. With accuracy and
latency weighed about equally, a 4.8% relative multi-hop gain did not buy a
3.4× slower write path, and BGE-M3 stays.

### Quantizing the stored vectors: measured, and it is the wrong lever

This decision recorded stored-vector quantization as deferred "until the binding exposes rotation", and expected it to be a disk saving — vectors are 55% of a real index's bytes. Both halves turned out wrong, and one of them was a defect this project shipped.

`PAMIN_VECTOR_STORAGE` built the projection's vector field five ways (it has since been removed, with every storage below; see "Two vector indexes, both half precision"). Over 50,000 clustered 1024-dimensional vectors in four segments, everything else held equal, on an otherwise idle machine:

| storage | recall@10 | per query | build | whole index |
| --- | --- | --- | --- | --- |
| **fp32, the default** | **0.9980** | **12.0 ms** | **88 s** | **225.9 MB** |
| fp16 | 0.9980 | 10.9 ms | 82 s | 337.0 MB |
| int8 | 0.9980 | 7.3 ms | 43 s | 292.6 MB |
| int4 | 0.9990 | 9.3 ms | 45 s | 269.4 MB |
| rabitq | — | — | — | refused when the graph is built: HNSW with RaBitQ is not in the engine's C API |

**Quantizing cannot save disk here, because the refiner keeps the full-precision vectors.** Every quantized row carries the same 206.4 MB of raw vectors — 50,000 × 1024 × 4 bytes is 204.8 MB, so that column is the raw copy — and adds its codes on top. So the trade is not "smaller index for slightly worse recall"; it is **19% to 49% more disk for half the query time and half the build**, at no measured recall cost. That is a real trade and it is the opposite of the one this decision went looking for, so the default stayed fp32 and the mechanism stayed available to a workspace that would rather spend disk than milliseconds -- until the owner's decision below replaced both.

The measurement could not be made by reading the code. Whether a refiner stores a second copy beside the codes is not visible from the binding's surface, and it is the whole answer.

**And rotation, enabled here on reasoning, was destroying the index.** The parameter was set for every quantized storage because spreading the bits across dimensions that carry comparable information must help the coarse storages and could not hurt the others. The engine accepts it only for int8 and int4 — for anything else it refuses when the *segment* opens its vector field rather than when the parameters are built, so fp16 presented as a segment that would not take writes rather than as a rejected setting. And on the two storages that accept it, it is ruinous:

| | recall@10 with rotation | without |
| --- | --- | --- |
| int8 | 0.0530 | 0.9980 |
| int4 | 0.0580 | 0.9990 |

Same bytes, same build time, same query time. This is the failure shape recorded above from the previous quantization attempt — an index returning plausible neighbours that are not the nearest ones, with no error anywhere — and it was reproduced here only because the harness reports recall rather than whether the calls succeeded. Why is not established. It is not a missing fitted transform, which is what this paragraph used to say: the engine's FHT rotator is random, seeded from `std::random_device`, so there is nothing to fit or to supply, and the likely cause is upstream. Nor is the rabitq row's refusal about a `raw_vector_provider` the binding does not expose, though that is what the engine's message names: the index that pairs a graph with RaBitQ, `HNSW_RABITQ`, is not in the engine's C API at all — `zvec_index_params_create(4)` hands back Flat parameters without an error — so no binding of that API can reach it. Rotation was off from then on, and with the quantized storages gone there is nothing left to turn it on for; `each_vector_index_returns_the_nearest_documents` in `crates/pamin-index/tests/projection.rs` is what would notice a collapse of this shape now.

**The joint export has a third cost, and it took a while to find.** On this export a text's vector depends on what else is in its batch. Against the same text embedded alone: cosine 0.9816 with a shorter neighbour in the batch, 0.9859 with a longer one, and the two neighbours disagree with each other at 0.9805. A batch of one is byte-identical to a single call, so it is the presence of a neighbour rather than the batching API, and it is not the tokenization either — the tokenizer pads to the batch's longest member, so a text that *is* the longest gets byte-identical ids and mask either way, and the mask is passed to the session. Only the batch dimension differs, which puts it in the export or the runtime's INT8 kernels. `speed` and `balanced` return byte-identical vectors batched or alone.

The consequence was not accuracy. It was reproducibility: `reindex` embedded in batches of 256 and the cascade embeds one document at a time, so a rebuild did not reproduce the index it replaced, and a document's vector depended on which other documents happened to be in flight beside it. So the joint export now runs one text at a time, which costs the batching win on this profile — thirty-two texts together take 190 ms against 409 ms one at a time, so `reindex` is roughly twice the wall clock here.

**Measured, because "it changes the vectors" and "it changes the answers" are different claims.** Re-running the model-alone arm of the cross-lingual harness on deterministic vectors returns 0.6335 cross-lingual and 0.6787 same-language nDCG@10, against 0.6351 and 0.6763 on the batch-perturbed ones — inside the run-to-run spread this harness already shows, and either side of the 0.6338 / 0.6748 recorded before. So the perturbation is systematic enough to leave the ranking alone, which is why it survived this long: nothing downstream looked wrong. It is fixed for determinism, not for quality, and the distinction belongs in the record.

**What this section said before 0.7.2, kept because it was wrong in an instructive way.** On `zvec-rust` 0.7.0 an index built with `hnsw_with_quantize(..., Int8)` returned recall@10 of 0.000 against exact search over the same 50,000 clustered vectors, with or without the refiner and with no error, and the decision was deferred "until the binding exposes rotation", which the engine's own benchmarks describe as what makes int8 usable. 0.7.2 exposed it (`IndexParams::set_quantizer_enable_rotate`), and the tables above are what that trigger found: int8 recalls 0.9980 *without* rotation, and rotation is what now collapses it. So rotation was never what int8 lacked, and the 0.000 did not reproduce on 0.7.2. What stood is the gate that result set — a recall run and an end-to-end run on real corpora before any storage ships — and the next section is that gate, run.

### What else zvec-rust 0.7.2 offers, measured

The binding exposes more than rotation: a half-precision vector field, IVF-RaBitQ and DiskANN as alternatives to the graph, an explicit memory limit, and a document iterator. Each was measured against a rule written down before its first number, under this project's order of priorities — accuracy, then latency, then memory, then disk — and each rule asked for a win measured where the product runs: `search_reranked` at the default tier on the `accuracy` profile, with the reranker's and the embedder's caches bypassed and the arms taken in rotated order. That ran over all 157 own-corpus queries, every second of XQuAD-R's 1,190 (595, scored in both groups) and every fifth of MuSiQue's first thousand (200), on four cores that two other evaluations were sharing (load about 9) — so the times below are ratios within a question, never absolute figures. The index-level figures are over the MIRACL-sw passages the evaluation workspace holds — 131,924 real BGE-M3 vectors, the 482 real queries embedded, exact search as the truth — and over the 50,000 synthetic clustered vectors `recall.rs` uses, in the four-segment shape `pamin reindex` produces.

| | recall@10, MIRACL | recall@10, synthetic | index query, MIRACL | disk, MIRACL vectors | resident, MIRACL vectors | end-to-end nDCG@10 | end-to-end time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **fp32 HNSW (shipped then)** | **1.0000** | **0.9980** | **12.3 ms (5.95–12.55)** | **595.5 MB** | **565 MB** | — | **1** |
| fp16 field (`VectorFp16`) | 0.9985–0.9988 | 0.9650 | 9.2 ms (8.62–9.39) | 327.8 MB | 310 MB | −0.0004 / −0.0002 / 0.0000, 1 and 5 questions worse, 4 better | 0.997 (p = 0.39) |
| int8 codes and refiner | 1.0000 | 0.9980 | 6.5 ms (4.69–6.71) | 758.7 MB | 704 MB | identical on every question | 0.998 (p = 0.47) |
| IVF-RaBitQ, 7 bits, nprobe of 363 lists: 45 / 90 / 181 | 0.9693 / 0.9921 / 0.9981 | 0.786 / 0.951 at 55 / 111 of 223 | 2.90 / 3.49 / 4.81 ms | 703.1 MB | 630–633 MB | not run | not run |
| DiskANN, degree 64, build list 100, search list 300 | 1.0000 | 1.0000 at 5,000 | 30.79 ms | 1,160.5 MB | 22 MB | not run | not run |

End-to-end nDCG@10 is own corpus / XQuAD-R / MuSiQue, each against fp32 on the same questions; time is the geometric mean of each question's ratio to fp32, pooled over the three (952 questions). Resident is what opening the vector-only collection and answering the 482 queries added to the process. Index query times are the median of four runs, one each for IVF-RaBitQ and DiskANN, on a machine whose load moved between 1.4 and 6 — fp32's own ranged from 5.95 to 12.55 ms — so they give a direction and nothing below rests on more than that.

**The half-precision field halves what it should and costs recall the engine's arithmetic loses.** The binding has no way to write one: `Doc::add_vector_f32` tags its bytes as fp32 and the engine refuses them for an fp16 field ("type mismatch, expected VECTOR_FP16 but got VECTOR_FP32"), and a query handed over as fp32 is refused as 2,048 dimensions. Written through the C API the binding re-exports, and queried with fp16 codes, it works: the vector bytes halve (−45% on MIRACL, and −25% and −19% of the whole XQuAD-R and MuSiQue indexes, 83.0 against 110.2 MB and 95.6 against 118.6), and so does their resident set. What it costs is recall, and not where expected. Rounding a vector to fp16 moves a component by at most 2^-11 of itself; exact search over the rounded vectors recalls 0.9970 of the synthetic truth. The engine's own scores are off by 3.7 × 10^-4 against exact cosine (fp32's by 1.5 × 10^-6), and its recall is 0.9650 — and the graph is not the cause, because a linear scan through the same field loses the same (0.9820 against 1.0000 at 5,000). On real embeddings, whose neighbours are further apart than synthetic clusters, the loss is 0.0012 to 0.0015, and end to end it moved nine of 1,190 XQuAD-R scores (four better, five worse) and one own-corpus query. Its index query time is not settled either way: 0.75 of fp32's in three rounds under load and 1.45 in the one quieter round. The rule allowed 0.002 of synthetic recall and it lost 0.033, so it did not ship. fp16 cannot be combined with a quantizer either: "dense_vector's index_params of VECTOR_FP16 do not support quantize". The next section finds the loss in the engine's arithmetic rather than in the rounding, takes it back with a rescore in f32, and ships the field.

**Int8 is the trade the section above describes, and at the whole search it buys nothing.** Recall and every ranking were identical — not one of 952 questions scored differently — and the index answers in 0.53 to 0.79 of fp32's time on MIRACL, but a search spends a few milliseconds in the vector index out of 1.1 s (own corpus) to 6 s (MuSiQue), almost all of it in the reranker. Measured there, it is 0.993, 0.998 and 1.002 of fp32's time on the three corpora (pooled 0.998, p = 0.47), against a rule that asked for at least 2% at p < 0.05. What it costs is real: 16% more disk on MuSiQue, 22% on XQuAD-R and 27% on MIRACL's vectors, and 25% more resident. So fp32 stays, now with the end-to-end number behind it rather than the index one.

**IVF-RaBitQ saves nothing at this size; DiskANN saves resident memory and pays for it in everything else.** Both were the expected path for a large or cold project. IVF-RaBitQ is faster than the graph at every probe count tried and less accurate at all of them — within 0.002 only when probing half the lists — and it holds more, not less: 703 MB on disk against 596, and 630 MB resident against 565, which is what a full-precision copy beside the codes would look like. On the synthetic clusters it is worse again, 0.951 at half the lists, with a resident peak of 1.14 GB against 0.77. DiskANN is the one real memory lever here: it recalls everything fp32 does and opening it and answering 482 queries added 22 MB to the process, against 565 MB for the graph, because it reads the vectors from disk per query rather than holding them — so what it touches sits in the page cache, which the kernel can take back, rather than in the process. Everything else goes the other way: 2.5 to 5 times the index query time (30.8 ms in one run, against fp32's 5.95 to 12.55 ms in four), twice the disk (1,160.5 MB), fourteen times the build (757 s against 55) with a 3.2 GB peak while building, and a 5,000-document build that took 70 s against the graph's 3. Under this project's order latency outranks memory, and the rule asked for no slower than 1.03 at the index, so it is not the default. It is the path for a project whose resident set is the constraint and whose searches can afford some 20 ms more in the index — not measured end to end, where that would be roughly 0.3 to 2% of a search at the times above — and, like `enable_mmap`, it is fixed when a collection is created, so reaching it is `pamin reindex` -- which, with `--vector-index`, is now how a project moves between it and the in-memory graph.

**The memory limit is not read by an index opened this way.** `initialize(Some(..))` with `ConfigBuilder::memory_limit` sets the size of the pool the engine reads vectors through when a collection was created with mmap off; left unset, the engine takes 80% of the cgroup's or the host's memory. Every collection here is created with the default, mmap on, and the pool is then not consulted. Measured rather than read, over XQuAD-R's 13,014 documents, each arm in its own process and two rounds: 256 MiB, 2 GiB and no limit opened at the same resident set (243 MB, to the kilobyte's noise), returned identical results for 400 vector and 400 lexical queries, answered in the same time within noise (4.6 to 5.5 ms a pair, overlapping across arms), and built the same index in 25.2 to 26.4 s with a peak of 826 to 873 MB — no limit lowest in both rounds. The engine is still initialized with `None`.

**The document iterator reads faster and saves little a rebuild can see, and it ships anyway.** A rebuild lent its vectors by reading the set-aside index twice through keyed fetches of 256 — once to count what it can lend, once to take it. Over MIRACL's 131,924 documents, warm, one pass of fetches took 1.46 to 1.59 s and one pass of the iterator 0.28 s, the resident set after each was the same 1.53 GB — the index, mapped — and the two read identical vectors and text. So the iterator takes about 2.5 s off a rebuild of that size; XQuAD-R's rebuild takes 30.0 s, of which reads at that rate are about 0.3 s. The rule asked for 10% of the rebuild's time or its peak memory, and the best case is 1%, so by the rule it did not ship. The rule was then overruled: the owner's position is that an optimization that costs nothing in accuracy is taken however small, and this one costs nothing — the same documents, the same text and bit-for-bit the same vectors, which `a_rebuild_lends_through_the_iterator_what_keyed_reads_return` in `crates/pamin-index/tests/projection.rs` asserts against keyed reads of the same index. `Previous::lends` and `Previous::lend` now make one pass each, and the rebuild writes what is lent in the order the old index holds it, then embeds the rest. Its three constraints are met by where it is used rather than by care: the set-aside index is opened read-only, so there is no segment being written for the iterator to seal; nothing optimizes a set-aside index, so none is blocked; and the vector field is not nullable, so no document fails the iteration for lacking one. The reshape still reads by key, because it copies the *served* index, which goes on taking writes and being optimized, and afterwards has to read again exactly the topics written meanwhile.

The runs are scratch builds (a storage label read at open and the two caches bypassed) and cannot be re-run from the repository. The fp16 field needed about a hundred lines around the C API, which the tree now carries in `crates/pamin-index/src/half.rs`.

### Two vector indexes, both half precision: `disk` by default

The section above measured each storage against a rule and shipped none of them. The owner then set the rule aside for the vector index and decided the shape directly: vectors are stored in half precision, the index is one of two, `disk` (DiskANN) or `memory` (HNSW), chosen with `--vector-index` / `PAMIN_VECTOR_INDEX`, and `disk` is the default because this project ranks resident memory above query time and disk, and `disk` is the index that holds almost nothing resident. Everything else — fp32 as a way to write, int8, int4, RaBitQ, IVF-RaBitQ, the quantizer's rotation, `PAMIN_VECTOR_STORAGE` — is gone from the code. What was measured to hold that decision to the accuracy bar, and what it costs, follows.

**Half precision is accurate once the engine's scores are not trusted.** The paragraph above found the fp16 field losing 0.033 of synthetic recall, and a later probe found why: on a CPU with AVX-512 FP16 the engine's cosine and inner-product kernels multiply and accumulate in half precision (`inner_product_distance_batch_impl_fp16_avx512fp16.cc` upstream), so its scores are off by about 4e-4 where rounding the vectors alone moves a cosine by about 1e-5. The fix is on this side of the boundary: ask either index for twice the candidates with their stored vectors, and rank those again by an exact f32 cosine against the query as the model produced it. That leaves only the rounding's error, which is what exact search over the rounded vectors recalls. The stored vectors are read back through the C API — the binding's f32 getter returns nothing for an fp16 field, which is why the section above concluded they could not be — and `crates/pamin-index/src/half.rs` carries the three crossings the binding lacks.

Measured against exact search, recall@10 and recall@50, with the rescore; fp32 HNSW is what shipped until now. Synthetic is the 50,000 clustered 1024-dimensional vectors `recall.rs` uses in four segments, 200 queries; MIRACL is its 131,924 real BGE-M3 passages and 482 real queries, in four segments. Times are the median index query on four cores that other evaluations were sharing at load 9 to 13, so they are directions within a row group, not figures:

| | recall@10 / @50, synthetic | recall@10 / @50, MIRACL | index query, MIRACL, k = 10 / 50 | build, MIRACL | peak while building | disk, MIRACL | resident, MIRACL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 HNSW (what shipped) | 0.9980 / 0.9974 | 1.0000 / 0.9999 | 9.5 / 9.4 ms | 171 s | +570 MB | 595.5 MB | 575 MB |
| **`memory`**: fp16 HNSW, width 700 | 0.9965 / 0.9965 | 1.0000 / 0.9997 | 5.9 / 10.0 ms | 84 s | +721 MB | 327.8 MB | 320 MB |
| **`disk`** (default): fp16 DiskANN, width 1,200 | 0.9985 / 0.9975 | 1.0000 / 0.9996 | 68.6 / 79.2 ms | 1,379 s | +1,353 MB | 618.1 MB | 34 MB |

Resident is what opening the vector-only collection and answering the 482 queries added to the process. Build and peak are one `optimize` over every document, the way `pamin reindex` builds. The synthetic set shows the same ordering: `memory` built in 58 to 64 s, fp32 in 98 s and `disk` in 922 to 1,024 s; they held 130, 224 and 27.5 MB resident and took 132.1, 232.0 and 235.6 MB of disk.

**`memory` meets the bar at the width it already had.** Its recall is inside 0.002 of fp32's on the synthetic set and inside 0.001 on MIRACL, and it is faster to query, to build and smaller on disk and resident than the fp32 graph, all of it from half the bytes a vector. The query width was measured again for it, because a narrower one is where its remaining milliseconds are: at 200, 300 and 500 it saves about a millisecond a query and misses the bar at fifty on both sets (0.9458, 0.9769 and 0.9927 on the synthetic set; 0.9980, 0.9988 and 0.9993 on MIRACL), so it stays at 700.

**`disk` meets it only by searching wide, and pays for that in query time and above all in build.** The on-disk graph is what limits its recall, not the arithmetic — the rescore adds at most 0.0015 at any width — and at DiskANN's own default width of 300 it recalls 0.981 of the synthetic top ten. It needs a width of 1,200 to be within 0.002 of fp32 at both depths, and the query then takes roughly ten times what the in-memory graph takes. On MIRACL's real embeddings the graph is easier to search -- 0.9996 at ten and 0.9986 at fifty from a width of 300, 0.9995 at fifty from 800 -- but the synthetic clusters are what set the width, as they set the in-memory graph's, and 1,200 is also what MIRACL was measured at. A build is the heaviest cost: over the synthetic 50,000, 922 to 1,024 s against 58 s for `memory`, with 946 MB more resident while it runs; a build list of 200 rather than 100 took 1,416 s and recalled no more, and 64 product-quantization chunks for navigation answered in a third of the time but recalled 0.9200 at ten and 0.8629 at fifty -- the rescore cannot find again what navigating by codes lost -- and took 1,859 s. What it buys is the resident set: the vectors and the graph stay on disk and are read per query, so opening the index and answering the queries added 34 MB to the process against 320 MB for `memory` and 575 MB for fp32; what it touches sits in the page cache, which the kernel can take back.

**What `disk` does to the write path is its largest cost, and it is not settled.** Measured through the product's own index on 25,000 synthetic vectors written the way a grown project holds them (segments of 10,000), with searches and writes running beside the `optimize` that upkeep issues: both indexes keep a freshly written memory searchable before any build -- every one of fifty found itself, by exhaustive scan of the unbuilt segment -- and neither holds a search up while it builds (`disk`: 15,423 searches during the build, median 13.5 ms, p95 26 ms; `memory`: median 20.5 ms). But `disk`'s first build took 567 s where `memory`'s took 81, and its index queries afterwards ran at a median of 166 ms and a p95 of 537 ms on a machine at load 12 to 15 against `memory`'s 6.8 and 11. And then five increments of 64 documents, each written, flushed and followed by an `optimize` -- what a working drain produces -- took **114, 114, 174, 196 and 310 s** of `optimize` each under `disk`, against 1.1 to 1.6 s under `memory`. Whatever the engine rebuilds when a few documents join an on-disk graph, it rebuilds a great deal of it, and the cost grew with each round. Upkeep runs `optimize` whenever the file budget, the unmerged blocks or the unindexed remainder asks for it, so how often a working project asks is what decides this cost: each time is minutes of two or more cores under `disk` and about a second under `memory`. That has to be settled before `disk` is the default in fact as well as in the code.

**End to end, neither index moves a ranking, and `disk` costs a few percent of a search.** Measured through `search_reranked` at the shipped defaults (`accuracy`, `accurate` reranking), each project rebuilt from its fp32 index the way `pamin reindex` rebuilds, lending every vector: the own corpus's 157 questions, and every second XQuAD-R question, 595 of them scored in two groups. Against fp32 on the same questions (the fp32 arm of the section above), nDCG@10 moved by +0.0003 on the own corpus (3 better, 2 worse, p = 0.76 by paired sign-flip) and −0.0003 on XQuAD-R (55 better, 55 worse, p = 0.68) under either index, because `disk` and `memory` scored identically on every question: their vector top ten agreed on 100 of 100 probe queries before a question was timed. In the same run, `disk` took 1.033 of `memory`'s time on the own corpus (median 1,371 ms against 1,287, p = 0.34) and 1.054 on XQuAD-R (2,339 against 2,236 ms, p < 0.001). Against fp32 the ratios are 1.23 and 1.13 for `disk` and 1.20 and 1.08 for `memory`, but fp32 was timed in an earlier run at a mean load of 8.9 and 9.0 against 15.1 and 10.4 here, so those carry the machine as well as the index and are not a figure. The rebuild that moved XQuAD-R's 13,014 documents onto each index took 424 s under `disk` and 30 s under `memory`, and left 108.1 and 80.6 MB of index where fp32's was 123.1.

**The two end-to-end tests the write path depends on pass under both, and one of them is flaky before this change as well.** `a_deferred_write_is_found_without_anything_else_being_run` passed in every run: three under `disk`, three under `memory` and three on the fp32 build this branches from, a deferred write searchable after 5.2 to 32.1 s, 5.5 to 34.5 s and 5.5 to 34.7 s against a limit of 60. `catching_up_does_not_hold_a_search_up` passed 2 of 3 times under `disk`, 1 of 3 under `memory` and 0 of 3 on the fp32 build. Its failures are of the same two shapes on every build: the server caught up only once while the thirty searches ran, so the test refused to time them, or one search during catching up took 21.7 to 24.4 s against a settled slowest under 0.1 s. Both shapes appear on the fp32 build this branches from, so neither is this change's. In every run, on every build, two logged rounds of catching up were 28 to 35 s apart where the loop sleeps five seconds between them; what holds the loop, and whether it is what the long searches waited for, is not established and is not fixed here.

**An index built before this is refused rather than searched.** Its marker's storage line says `fp32`, or nothing, which reads as `fp32`; it names neither index, so opening it fails with the message the profile check already gives — run `pamin reindex` — and the rebuild lends every vector it holds, reading them as fp32 and storing them as half precision would have stored the model's. Changing `--vector-index` on a built project goes the same way: refused, then rebuilt without embedding anything. Refusing is what the marker already did for a profile change, and it keeps one way of searching: an index is only ever searched as what it was built as. `each_vector_index_reads_back_what_it_stored_and_is_opened_only_as_itself`, `an_fp32_index_from_before_is_refused_and_lends_to_its_rebuild` and `each_vector_index_returns_the_nearest_documents` in `crates/pamin-index/tests/projection.rs` hold the three properties, and the last fails at 0.01 recall when the query is scrambled.

### The graph is the memory floor, and it just doubled

Quantizing stored vectors, if it worked, would shrink the payload and not the graph. The graph is the part that does not respond to it, which makes it the floor under everything else. At the size this store is built for — seven million documents in a project — an HNSW graph at `m = 16` is roughly 0.98 GiB per project, so a hundred projects is about **98 GiB of graph before a single vector is counted**.

Raising `m` to 32 for the recall measured above doubles that: roughly 196 GiB for the same hundred projects. That is a real cost and it is the right trade anyway, for a reason worth stating rather than assuming. Nothing puts hundreds of projects of this size in resident memory under *any* configuration — the fp32 payload alone is 2.7 TB, and the best case measured here, one-bit quantization that does not work in this engine, still leaves a floor in the hundreds of gigabytes. Protecting a factor of two on a budget already out of reach buys nothing, while a vector channel returning seven of every ten true neighbours is a live defect.

What it does change is when the disk-resident path stops being optional. Serving that many projects at that size means keeping cold indexes on disk and paging in the working set, and the graph doubling brings that forward rather than pushing it away. The engine exposes `IndexType::Diskann` and `IvfRabitq` for it, with two constraints to carry into that work: DiskANN is available on Linux x86-64 and ARM64, macOS ARM64 and Windows x86-64 but IVF-RaBitQ only on Linux x86-64, and `enable_mmap` is written into the manifest at creation and ignored when an existing collection is opened, so it cannot be turned on after the fact.

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

The last column is a comparison between models and is frozen at the fusion of
the day it was taken, `k = 10` with the lexical pair at half. The weight has
been halved twice since; the default profile's own corpus figure is 0.7773
today. The ordering the column exists to show does not move with it, because
the weight applies to every row alike.

BGE-M3 is the default, reversing this decision's original position. That position rested on two claims, and the evaluation harness contradicted both. Its cost per query is not an order of magnitude higher — quantized weights put it at 35 ms against 26, and at 560 MB it is *smaller* resident than the model it replaces. And the sparse arm that was supposed to be its main increment is not: only the dense representation is kept, and the dense representation alone roughly doubles cross-lingual retrieval on our corpus while matching same-language retrieval exactly.

`multilingual-e5-small` is not the default because 384 dimensions is generally considered sufficient only when paired with a cross-encoder reranker, and the one Påmin Memory ships reorders only the candidates the lexical channels missed. It is not there to rescue a weaker embedding across the board, and cannot be relied on to. EmbeddingGemma scores well and supports Matryoshka truncation, but is governed by the Gemma Terms of Use, whose restrictions must be passed to downstream users; that is not an acceptable burden to attach to an open-source default. The E5 family and BGE-M3 are Apache-2.0 or MIT, as is the int8 export.

Learned sparse retrieval such as SPLADE outperforms BM25 on most benchmarks but requires GPU inference, which is incompatible with a default install that needs no API key and no GPU. It stays a profile, not a default.

### The floor is the segment size, for every workspace that grows

The section above ends by noting that a collection records its segment size at
creation, so a project that has grown keeps the size it was created with until
`pamin reindex` rebuilds it. That is true and it understates the case, because
of what a collection is created *with*: nothing. A workspace is created before
anything has been written to it, so `segment_documents` is asked about zero
documents, is clamped to `SMALLEST_SEGMENT`, and records 2,000.

**So for every workspace that grows from empty -- which is every workspace a
user has -- the division by `TARGET_SEGMENTS` never runs.** The floor is the
segment size, at every scale, and the four-segment target the table above was
measured to support is reachable only through `reindex`, which is created
knowing the count. What a project actually gets:

| documents | segments, grown from empty | segments, after `reindex` |
| --- | --- | --- |
| 50,000 | 25 | 4 |
| 131,924 | 66 | 4 |
| 1,000,000 | 500 | 4 |

And the floor's own comment says what it was chosen for -- *so a new and nearly
empty project is one segment rather than a hundred tiny ones* -- which is a
reason about the small case that silently became the value for the large one.

Measured over the same 50,000 documents on this machine, four segment sizes:

| a segment holds | segments | recall@10 | a query | build |
| --- | --- | --- | --- | --- |
| 2,000 (what a grown project records) | 25 | 1.0000 | 39.8 ms | 52 s |
| 12,500 (what `reindex` chooses) | 4 | 0.9980 | 16.9 ms | 129 s |
| 25,000 | 2 | 0.9880 | 26.4 ms | 221 s |
| 50,000 | 1 | 0.9510 | 16.6 ms | 415 s |

**Twenty-five segments cost 2.4x a query against four, and recall does not pay
for it** -- it is slightly better with the smaller graphs, which is the
direction `recall.rs`'s floor comment records.

**And the last two rows killed the obvious fix.** Raising `SMALLEST_SEGMENT`
follows directly from the 0.36 ms-a-segment figure above, and it is wrong:
recall falls monotonically as the segments grow, reaching 0.9510 at one
segment, which is below `recall.rs`'s own 0.97 floor. Aiming at fewer segments
than the policy wants trades accuracy away. A floor of 25,000 would have moved
a fifty-thousand-document project from four segments to two and cost 0.010 of
recall for nothing.

One caveat on the latency column, because it matters for how much weight it
carries: 26.4 ms for two segments sits above both one and four, which is not a
shape anything physical would produce. These arms ran while a
131,924-passage index build had the machine. The recall column is unaffected by
that and the 25-segment latency reproduced across two runs; the middle rows'
milliseconds should not be leaned on.

It is also where the descriptors go. A segment is 79 files here, so 66
segments is about 5,200 against the 1,024 a Linux process starts with -- which
is how indexing MIRACL's Swahili split died with `Too many open files` 65
minutes in, and why `raise_open_file_limit` moved out of the server and into
every process.

**Changing it on an open collection is not available, and the way it is not
available is worth recording.** `CollectionSchema::set_max_doc_count_per_segment`
exists, and on a schema read back from an open collection it returns `Ok`,
reports the new value from that handle, and changes nothing: reopening the
collection reports 2,000 again. That is the same shape as `enable_mmap` above
-- a setter that is a no-op after creation and says so only by being ignored --
so resegmenting means recreating the collection, which is what `pamin reindex`
does.

What to do about it is a live question rather than a decision recorded here.
Raising the floor sets the de-facto segment size for a grown project and is one
constant; it cannot help a workspace that already exists, and the `reindex`
those need is hours on the default profile -- 131,924 passages measured at 576
cascade jobs a minute, which is ten and a half of them.

### A cross-encoder reranker, once the opportunity was real

An earlier version of this decision recorded that reranking was measured and did not help. That measurement stands; its premise does not. It ran on Påmin Memory's own 210-memory corpus, where the diagnostic said plainly that there was nothing to recover: across all 137 queries the relevant memory was already inside the top ten, so a second pass could only reorder what was already right, and both models reordered it worse. The conclusion drawn from it — *revisit when the opportunity is real* — named the measurement to run first, and an external corpus supplied it.

On 13,014 sentences in eleven languages, 3,813 relevant sentences sit between rank 10 and rank 50, across 1,088 of 1,190 queries. That is the opportunity the small corpus could not produce, and in it reranking is the largest single retrieval gain measured in this repository. (It was 4,845 across 1,149 queries when this was first run. Correcting the fusion weights moved several hundred of them up into the top ten, which is the right direction and leaves the point standing: the space a reranker works in is still most of the corpus.)

| Tier | Loads | A search costs | Of which reranking | Cross-lingual nDCG@10 | Same-language |
| --- | --- | --- | --- | --- | --- |
| `off` | nothing | 99 ms | — | 0.6114 | 0.7829 |
| `fast` | 119 MB | 359 ms | 260 ms | **+0.0397** `p=0.0001` | **−0.0060** `p=0.0008` |
| `balanced`, removed | 341 MB | 821 ms | 722 ms | +0.0094 `p=0.0146` | +0.0029 `p=0.0293` |
| `accurate` (default) | 571 MB | 1522 ms | 1423 ms | **+0.0482** `p=0.0001` | +0.0006 *ns* |
| `noncommercial`, removed | 280 MB | 905 ms | 806 ms | +0.0279 `p=0.0001` | +0.0003 *ns* |

Five arms, one run, the same 1,190 queries, paired bootstrap at 10,000
resamples. Nothing in this table may be read against a figure published before
it: the two rows that existed were taken at the previous fusion weights and both
moved. `recall@50` is 0.8962 cross-lingual and 0.9571 same-language in **every**
arm to four decimals, which is the invariant worth stating — a reranker reorders
a shortlist and never changes what is in it.

**Three results the two-row version could not show.** `fast` is the only tier
that measurably damages same-language ranking, and the damage is significant
rather than incidental: nineteen queries worse against three better. So the
default's cross-lingual gain is partly bought from the other group, and the
three larger models decline that trade. **Parameter count does not order the
table**: 21.2M `fast` beats both 84.9M models cross-lingual, and `accurate` at
302M beats everything — so the ordering is neither monotone in size nor
explainable by capacity. And `balanced` and `noncommercial` are *the same
architecture at the same size*, twelve layers of width 768, yet differ by 0.0185
cross-lingual, which is twice `balanced`'s entire gain. What is being chosen at
this size is the training, not the model's shape.

**Relaxing the licence bought nothing, and the experiment is recorded for
that.** `noncommercial` existed to answer one question — whether accepting
CC-BY-NC buys accuracy that a permissive licence cannot — and the answer is no.
The prediction written before the run was that it would be indistinguishable
from `balanced`; it is better than `balanced` and still worse than `fast` at a
quarter of its size. Half the prediction held and the more interesting half did
not.

**`balanced` and `noncommercial` were removed on these rows.** They were
`onnx-community/gte-multilingual-reranker-base` and
`jinaai/jina-reranker-v2-base-multilingual`, and neither has a place on the
trade a tier is chosen on: `fast` beats each of them cross-lingual at under half
the download and under half the latency, and `accurate` beats everything
cross-lingual. `balanced`'s one distinction, the only significant *positive* on
same-language, is +0.0029 and not worth 821 ms. They were kept for a while so
the product's own tier table could say so; the numbers are here instead, and
with the non-commercial tier went the licence notice and the
`PAMIN_ACCEPT_NONCOMMERCIAL` variable that existed only for it. Asking for
either tier by name is now the unknown-tier error.

Measured through `Engine::search_reranked`, the entry point `pamin search`
calls, with `TIERS=1` on the cross-lingual harness over all 1,190 queries. The
first version of this table came from a scratch program that reordered a dumped
shortlist with its own copy of the pipeline, and overstated both gains by about
half — which is the argument for measuring the product rather than a model of
it.

**The latency in this row is a correction, and it is the third one.** This
table recorded 204 ms and 508 ms, the second of those having replaced an earlier
1795 ms; re-running the same harness returned 264 ms and 1001 ms. The five-tier
run above returns **359 ms and 1522 ms** for the same two tiers, on the same
machine and binary, with the fusion layer changed in between — which is the
lead the paragraph below names, now acted on rather than offered. Four figures
for one measurement, each taken once. The rule the sequence earns is the one
this project now follows: a latency figure is re-taken in the same run as
everything it will be compared against, and a figure from another run is not a
baseline.

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

**`fast` was the default on latency, not on quality.** It is fourteen times smaller than `accurate` and its pass costs 260 ms against 1423 — 5.5x. For that it gives up 0.0085 cross-lingual, and it gives up the 0.0066 of same-language that `accurate` holds on to: `accurate` is the only tier that costs nothing on either group, and `fast` is the only one that costs something.

The appeal to *Shallow Cross-Encoders for Low-Latency Retrieval* (arXiv 2403.20222) turns on a latency budget, and the budget is what this section got wrong twice. At the figures recorded here the gap was 2.5x and the shallow model's case looked thin; re-measured it is 4.2x end to end, 359 ms against 1522. `fast` stays the default, and the reason is unchanged and now better supported: a search that takes a second is a different product from one that takes a quarter, and the difference it buys is in the third decimal. That is a judgement about the budget rather than a result, and `--rerank accurate` is there for a workspace that judges differently.

**On a corpus that is not parallel text the two tiers separate much further.** MIRACL's Swahili dev split is 131,924 real passages averaging 229 characters with human judgements, one language throughout — the shape XQuAD-R is not:

| Tier | nDCG@10 | Gain | A search | Of which reranking |
| --- | --- | --- | --- | --- |
| `off` | 0.7158 | — | 142 ms | — |
| `fast` | 0.7359 | **+0.0201** | 474 ms | 332 ms |
| `accurate` | 0.7654 | **+0.0496** | 1867 ms | 1725 ms |

`fast` is worth half what it is worth on sentences, which was expected: the pass reorders only what the lexical channels missed, and within one language that is a fraction of the shortlist rather than nearly all of it — 84 of 482 queries leave a relevant passage below rank ten here against 1,042 of 1,190 there. `accurate` was expected to shrink with it and does the opposite, gaining more here than there, so the ratio between the tiers goes from 1.2 to 2.5. Long varied passages are where twenty-one million parameters start to tell against three hundred million, and that is the case the shallow-cross-encoder argument does not cover.

These MIRACL rows came from a program that was never committed, which is worth saying here because this section is where they are used to argue for a default. `crates/pamin-engine/tests/monolingual.rs` is the harness for them now, with the model-alone arm this corpus never had, and until it has reproduced them the numbers above are a record of a past run rather than something a reader can re-take. What the argument rests on -- that the tiers separate further on real passages than on parallel sentences -- is the part to re-check first.

It costs accordingly: 1725 ms against 332. Two seconds a search is not an interactive budget, so the default does not move — but on real passages choosing `fast` gives up three fifths of the available gain rather than a fifth, and that is worth knowing before accepting it. recall@50 is 0.9494 for all three tiers, which is the same invariant the other corpus shows.

**This section used to end by advising a single-language workspace to set `off`, and the table four paragraphs above it says the opposite.** On the one genuinely single-language corpus measured here, at the profile that ships, `fast` gains **+0.0201** and `accurate` **+0.0496**. The advice came from reading XQuAD-R's same-language column as a model of a monolingual workspace, and the gate sweep in `crates/pamin-engine/tests/crosslingual.rs` shows why that reading fails: on parallel text the cross-lingual and same-language groups are *the same 1,190 queries* scored against different answer keys, so every query in the "same-language" group still has correct answers in ten other languages sitting in the index. A real single-language workspace has none. The column measures a parallel corpus, not a monolingual one.

So the recommendation is withdrawn rather than softened: **on a single-language corpus the pass pays**, by half of what it pays on parallel sentences, and the tier to choose is a latency question like any other. What remains true is the mechanism the advice was reaching for — the pass reorders only what the lexical channels missed, and within one language that is a smaller fraction of the shortlist, 84 of 482 queries here against 1,042 of 1,190 there. A smaller fraction is not zero.

One figure elsewhere looks like a contradiction and is not. [measured.md](../measured.md) prices `fast` on this same corpus at **−0.0152** (37 wins, 57 losses, p = 0.0129). That arm is the **`speed` profile**; this table is the **`accuracy`** profile, the default. Two embedding models produce two different shortlists and the pass is worth different things on each, which is a result rather than a discrepancy — and the sign flip between them is the strongest evidence on this page that a reranker's value is a property of what fusion hands it rather than of the model.

**The default moved to `accurate`, and what moved it was the ordering rather than a new number.** Every paragraph above that kept `fast` did so on latency, and each one said it was a judgement about the budget. The project's ordering is accuracy, then latency, then memory, then disk, and under that ordering the judgement goes the other way. Put query by query through `search_reranked`, `accurate` against `fast`:

| corpus, group | `accurate` − `fast` | wins / losses / ties | p |
| --- | --- | --- | --- |
| XQuAD-R, cross-lingual | **+0.0086** | — | 0.0015 |
| XQuAD-R, same-language | **+0.0066** | — | 0.0001 |
| MIRACL Swahili, `speed` profile | **+0.0411** | 83 / 12 / 387 | 0.0001 |
| own corpus, cross-lingual | +0.0151 | 16 / 11 / 16 | 0.37 |
| own corpus, relational | +0.0096 | 3 / 1 / 16 | 0.63 |

`accurate` is never behind, and significantly ahead wherever there are queries enough to say. `fast` is below no reranking at all on both same-language measurements — XQuAD-R −0.0060 (p = 0.0002) and MIRACL at the `speed` profile −0.0154 (35 / 58, p = 0.014) — so the tier that shipped was, for a query whose answer shares its language, worse than asking for nothing. The cost is paid in the order the project ranks things: 1522 ms against 359, a quarter of the throughput, 571 MB loaded against 119. `--rerank fast` and `--rerank off` buy it back for a workspace that has to.

**Exempting graph-reached candidates from the pass was measured and dropped.** The relational group scores lower through the shipped path than through fusion alone, and the mechanism is specific: such an answer is relevant because *another* memory mentions it, which a cross-encoder reading the query and that one memory cannot see. The engine already exempts candidates with lexical evidence, so exempting candidates with graph evidence on the same terms was the obvious rule, priced at the `fast` tier from one shipped run by replaying both. It recovered relational (+0.0259, 4 / 0, p = 0.12) and cost cross-lingual (−0.0108, 0 / 10, p = 0.0018): it moved the loss rather than removing it.

A score depends on the query as well as the memory, so a resident process remembers the pairs it has computed: a repeated search measured 69.6 ms the first time and 0.0 ms the second, for the same ordering. Four thousand scores, about a quarter of a megabyte. It does nothing for a query never asked before, which is most of them; it is worth its quarter megabyte because agents retry, widen a limit, and ask again after writing. Without `pamin serve` there is no process to keep it in.

**There is no compilation trick left in the runtime.** Batching was the lever inside it, and it has been pulled twice. Sorting candidates by length before batching and using batches of eight rather than sixteen took the same work from 191 ms to 151, because a batch is padded to its longest member. Grouping pairs by their real length in tokens, each pass within 512 padded tokens and four pairs, then took a whole default search at a depth of thirty to 0.81 of what chunks of eight cost, paired over every query of XQuAD-R, MIRACL and MuSiQue, because on four cores a pair also costs more the more tokens share its pass; no group's nDCG@10 moved significantly, which had to be measured rather than assumed, since the int8 export quantizes a batch's activations together and so scores a pair by its company. The rule it was chosen by and the tables are at `BATCH_TOKENS` in `crates/pamin-index/src/reranking.rs`. Against that, the export format is worth at most 1.45x on identical weights, fp16 is slower than fp32 on a CPU, and the session already runs every core at the highest graph optimization level. The measured 9.75 ms a pair is what twelve transformer layers on four cores cost.

The published answers to this latency all change the architecture instead, and both are deferred on a missing export rather than on a licence or a doubt:

| Route | Worth | Blocked on | Revisit when |
| --- | --- | --- | --- |
| Precomputed document layers (PreTTR, arXiv 2004.14255) | ~6x on this shape: twelve layers over a ten-token query and a hundred-token memory is 1320 layer-tokens; caching the memory's first eleven layers makes it 220. 165 ms becomes roughly 28 — the base was 151 ms until the tier figures were re-measured through the engine, and it is the ratio that carries the estimate rather than the base. Storage is hot set x tokens x width: 384 MB for ten thousand memories | An export split into two halves. It is an export-time job, not runtime graph surgery: `ort` selects only among declared graph outputs, and while `ort` 2.0.0-rc.13 — the version in this lockfile — does now carry an `editor` module behind feature `api-22`, it builds a graph from scratch and cannot load an existing export to cut one. Nothing on crates.io can | A split export of `mmarco-mMiniLMv2` exists **and** the score cache's hit rate shows the hot set is actually small, which is the same evidence that decides whether it is worth its storage |
| Late interaction (ColBERT) | Moves the cost to write time, where the cascade already runs a forward pass per memory; query time becomes MaxSim. At 64–128 dimensions int8 that is 6.4–12.8 KB a memory, the same order as the vector index | **Was** no usable model, and that is no longer true. `colbert-xm` is still MIT with no ONNX export, and the English-only exports are still English-only — but `lightonai/mLateOn` is Apache-2.0, multilingual over nine languages, 128 dimensions, on mmBERT-base, and its own repository carries `model.onnx` and `model_int8.onnx`. Verified against the Hugging Face API on 2026-09-20, not from a card or an announcement. 128 dimensions int8 is 12.8 KB for a hundred-token memory, the top of the range this row already budgeted for. What is not verified is whether the projection head is folded into the export or applied after it, which is a ten-minute check at load time. Rejected on licence while looking: `LiquidAI/LFM2-ColBERT-350M` ships under LFM Open License v1.0, which this project cannot take while it distributes the database itself. BGE-M3's own ColBERT head is 1024-dimensional, 100 KB a memory int8, ten times the whole index | Superseded. The trigger named `colbert-xm` because it was the only permissive multilingual candidate in 2026-03; `mLateOn` now satisfies what the trigger was standing in for, so what remains is the premise the row itself flags — measure it cross-lingual on the XQuAD-R and MIRACL harnesses against the `fast` and `accurate` tiers, and measure what a forward pass per memory costs the cascade, before taking the storage |

Both routes rest on premises Påmin Memory has not measured — that the hot set is small, that write-time cost is cheap — and the triggers are written to test the premise before the work. Re-checked 2026-09-20: one route's blocker dissolved and the other's did not, which is the reason to re-check a deferral rather than trust the note that created it.

Licensing was the blocker when this was first examined and is no longer. The embedding library's own four rerankers remain unusable — two English-only, one CC-BY-NC-4.0, and one carrying no licence at all — but its user-defined loader takes any ONNX, which is the path every tier here takes.

### What each shipped tier is licensed under, and what was surveyed against them

Licences were checked at the leaf and at the base, because a fine-tune's card can declare a licence its base does not permit.

| tier | model | licence | base |
| --- | --- | --- | --- |
| `fast` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `apache-2.0`, declared on the card | `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`, **no licence tag**; MiniLMv2 originates in `microsoft/unilm`, MIT |
| `accurate` | `onnx-community/bge-reranker-v2-m3-ONNX` | **no licence tag** — its front matter is `library_name` and `base_model` and nothing else | `BAAI/bge-reranker-v2-m3`, `apache-2.0` |

Both chains are defensible and **neither states its licence where it is shipped from** — `accurate`'s export carries no front matter but `library_name` and `base_model`, and `fast`'s base is itself untagged. A re-export with no tag is usable when the chain to a licensed source is readable, which is the rule this project settled on, and every chain above is given in [`NOTICE`](../../NOTICE) so that a reader does not have to re-derive it. It is still worth an upstream request or a self-controlled export, and it is recorded here rather than left to be rediscovered.

`balanced` and `noncommercial` shipped for a while and were removed on their measurements; both are in the survey below. `balanced`'s export carried no tag over an `apache-2.0` base. `noncommercial` was `cc-by-nc-4.0` on its own card, the only tier whose licence restricted what may be done with the *output* rather than how the weights may be redistributed, and it printed the terms once and then ran rather than refusing. Every tier that ships now is permissive, so nothing is left to warn about.

**The survey against them, and the reason none of it changed the default.** Every candidate below was checked for a readable permissive licence first, because a model that cannot be shipped does not need measuring.

| candidate | licence | why not |
| --- | --- | --- |
| `jinaai/jina-reranker-v2-base-multilingual`, `-v3`, `jina-reranker-m0`, `jina-colbert-v2` | **CC-BY-NC-4.0**, the whole line | Non-commercial, so never a default. `v2` was shipped as the opt-in `noncommercial` tier to measure it, and removed; see below for what it was worth. `v3` and `m0` are not runnable here at all |
| `BAAI/bge-reranker-v2-gemma` | card says `apache-2.0`; base `google/gemma-2b` is `license: gemma`, gated | The Gemma rider follows the derivative — the same reason EmbeddingGemma was refused above |
| `BAAI/bge-reranker-v2-minicpm-layerwise` | card says `apache-2.0`; base MiniCPM weights carry the General Model License with a commercial-authorization requirement | Painful, because its 8–40 selectable output layers are exactly the early-exit mechanism the latency problem wants |
| `naver/splade-v3` family | CC-BY-NC-SA-4.0 | Non-commercial and share-alike. `Splade_PP_en_v1` is Apache-2.0 and English |
| `Qwen/Qwen3-Reranker-0.6B` | `apache-2.0` — the cleanest licence and the best multilingual quality in the field | A decoder at roughly twenty times the compute-relevant parameters of `fast`. Estimated seconds a query on four cores; three to six times the `accurate` tier, which is already not an interactive budget |
| `mixedbread-ai/mxbai-rerank-base-v2` | `apache-2.0` | MIRACL 28.56. Not a multilingual reranker in the sense this product needs, whatever the language count says |
| `Alibaba-NLP/gte-multilingual-reranker-base` | `apache-2.0`, with an int8 ONNX re-export | Four times `fast`'s compute for a 12-layer model. Shipped as `balanced` to settle it, and **measured worse than `fast` cross-lingual** at 2.3 times its latency — the "plausible middle tier" this row predicted is not one, and it was removed |
| `nreimers/mmarco-mMiniLMv2-L6-H384-v1` | **no licence tag at all** | The obvious "halve the layers" move, unavailable for the reason this project's rules anticipate |

**What the non-commercial licence actually buys, now that it has been paid.**
The survey above ruled the whole Jina line out as non-commercial and left it
there. The rule was then relaxed — CC-BY-NC acceptable as a named, opt-in,
non-default tier — so the question became answerable and was answered rather
than argued: `jina-reranker-v2-base-multilingual` shipped as `noncommercial` and
its figures are in the tier table above. **It scores +0.0279 cross-lingual where
`fast` scores +0.0397, at four times the parameters and 2.5
times the latency.** Accepting the licence bought nothing.

And it is the *best case* for the hypothesis, not a weak instance of it. It is
the most-downloaded non-commercial reranker there is, it is a genuine
XLM-RoBERTa cross-encoder in the shape this project can load, and it carries a
full ONNX suite — fp32, fp16, int8, q4, bnb4. Everything newer in that line is
further away rather than closer:

| | licence | base / architecture | ONNX exports | vs `fast` |
| --- | --- | --- | --- | --- |
| `jina-reranker-v2-base-multilingual` | `cc-by-nc-4.0` | XLM-R cross-encoder | **the full suite** | 4.0x |
| `jina-reranker-v3` | `cc-by-nc-4.0` | **`Qwen/Qwen3-0.6B`**, `JinaForRanking` | **none** | ~19x |
| `jina-reranker-m0` | `cc-by-nc-4.0` | **`Qwen2-VL-2B-Instruct`**, `JinaVLForRanking` | **none** | ~60x |
| `openjev/openjev` | `cc-by-nc-4.0` | `Qwen3_5ForConditionalGeneration` | **none** | not a ranker head |

`v3` and `m0` are the decoder class this record already priced out, and neither
publishes a single `.onnx` file, so there is nothing for a session to load, there is no
quantized export to fall back on, and adopting one means both a raw `ort` path
*and* an export nobody has made. `openjev` has 159 downloads and is a
conditional-generation model rather than a ranking head.

**So the non-commercial licence does not correlate with accuracy here. It
correlates with size and with a hosted-API business model** — the line moved to
0.6B and 2B decoders, which are out of an interactive budget on four CPU cores
whatever their terms say. That is the generalisable finding, and it is why the
`noncommercial` tier's result is recorded here now that the tier is gone rather
than dropped along with it.

One caveat, stated because it is the arm that was not run: the measured export
is `onnx/model_int8.onnx`. `fast` and `accurate` are quantized too, so the
comparison is consistent, but no fp16 arm exists. For it to change the verdict
that arm would have to close 0.0118 of nDCG *and* get faster, and dequantizing
does neither.

**Jev, and the shape of its claim.** TypeSafe's Jev is a decision model: text in, a number out, no token generation. It is API-only at $0.042 per million input tokens, so it cannot be part of an offline product whatever its quality. The open recreations do not rescue it. `openjev/openjev` is CC-BY-NC-4.0 and 27B parameters — 54 GB in fp16, and its own card measures about 80 ms **for one short decision on an H100 in fp8**, where this product scores twenty pairs in 226 ms on four CPU cores. `jaredpalmer/kev-0.8b` is Apache-2.0 at the adapter and base, tagged `language: en`, and its declared training data includes `Yelp/yelp_review_full`, whose terms grant academic use only.

The speed claim is real and does not apply here. Parallel, evaluating Jev independently, states it plainly: *"The headline cost and speed comparisons are against autoregressive LLMs, not dedicated classifiers. Specialized classifiers will still often outperform Jev on cost and speed."* A cross-encoder is a dedicated classifier — one forward pass, no generation — so the thing Jev's design removes is a cost this product never paid. An independent multilingual evaluation (9,831 graded pairs, 164 Chinese and English queries) puts Jev's rerank at **+0.012 over BGE-M3 under its own labels and −0.028 under judge-independent labels**, and measures the circularity that separates them.

What is worth taking from it is not the model. `jev-reranker`, a library wrapping the API, reports that a relevance *threshold* — dropping candidates rather than reordering all of them — removed 92.38% of the candidates and scored higher than reordering them, 0.975 against 0.969 nDCG@10. That is a gate, and a gate needs no weights at all.

**That idea has now been measured here and it lost — but read what was measured before reading the result as a refutation.** The `GATE` arm of the cross-lingual harness swept it over 1,190 queries with paired significance: every relative cut is negative cross-lingual and monotone in how much it drops, the gentlest cuts move nDCG@10 by exactly zero while `recall@50` falls, and `best − 2.0` drops 69.1% for **−0.1642** with recall going 0.8962 to 0.5613. Nothing in the sweep is a win.

Three differences separate that from the published experiment, and each one is load-bearing:

- **Their cut is on a calibrated probability.** The threshold is `0.2` on a `[0, 1]` scale, because Jev returns probabilities. The sweep's absolute family cut a cross-encoder *logit*, which this record says in three places is calibrated against nothing — so that family tested a different claim and, by losing systematically rather than randomly, confirmed the warning it was included to test.
- **Their candidate set was unfiltered.** The top 100 from hybrid BM25 and dense retrieval, nothing removed. This project's pass is already confined to the candidates no lexical channel found — about fifteen of fifty-one — so a threshold here removes signal where theirs removed junk. The prediction written before the run said exactly this.
- **Scale and evidence.** Fifty queries on one NanoBEIR split, a hand-set threshold with no account of how it was chosen, no significance test and no latency figure, for +0.006. Against 1,190 queries and a paired bootstrap. And the library's own advice is *"evaluate the cutoff on your own data"*, which is what happened.

So the honest conclusion is narrower than "thresholding does not work": **what has been refuted is thresholding an uncalibrated score over a pre-confined candidate set.** The published idea needs a score that means the same thing across queries, and nothing here has one — which makes it one of the things the calibrated-scorer work below is for, rather than a closed question.

### The model map, one year on: ModernBERT, mmBERT, Laya, and the decoder rerankers

The survey above was run against a size budget. That constraint was lifted —
accuracy, not download size, decides which model may be offered — and the
survey was re-run against the field as it stands, because a rejection whose
reason has expired is not a decision any more.

**Nothing in it changes a default, and the reasons are now different reasons.**
That is the point of recording it: the two families rejected for size are still
rejected, on grounds that a larger budget does not touch.

| candidate | licence | languages | architecture | non-embedding vs `fast` | why not |
| --- | --- | --- | --- | --- | --- |
| `mixedbread-ai/mxbai-rerank-base-v2` | `apache-2.0` | 109 | **`Qwen2ForCausalLM`** | ~17x | A decoder, and seconds a query |
| `Qwen/Qwen3-Reranker-0.6B` | `apache-2.0` | multi | **`Qwen3ForCausalLM`** | ~19x | Same, one size up |
| `Alibaba-NLP/gte-reranker-modernbert-base` | `apache-2.0` | **English only** | cross-encoder | 4.0x | Full int8 ONNX, 2.7M downloads, and monolingual |
| `Antix5/product-reranker-mmBERT-small` | `mit` | 13 | cross-encoder | 2.0x | Trained on product similarity, 82 downloads |
| `convaiinnovations/laya` | `apache-2.0` | 1811 (multilingual variant) | **typed-decision RL agent** | 5.2x | Not a cross-encoder; see below |

**The two `apache-2.0` decoders are blocked on latency, and on nothing else.**
The earlier reading here put shape first. Shape is a real difference — both
rerank by prompting and comparing the logits of a "yes" and a "no" token, so the
graph returns a vocabulary-sized tensor where the reranker's encoder reads one
logit from a sequence classifier — but it costs much less than this record assumed, because
the two things it was thought to cost are already paid:

- **`ort` 2.0.0-rc.13 and `tokenizers` 0.23.2 are already in the lockfile**,
  and every reranker already runs on a raw session over them
  (`crates/pamin-index/src/encoder.rs`). Another shape is a module, not a new
  dependency, and it does not move the size budget.
- **The export already exists, permissively licensed.**
  `onnx-community/Qwen3-Reranker-0.6B-ONNX` is `apache-2.0` with single-file
  graphs — `model_quantized.onnx` at 1219 MB, `model_q4.onnx` at 995 MB — and
  all four tokenizer files.

So this is named work rather than a closed door, and if it is opened it should
be opened on the permissive model. `jina-reranker-v3` is the same size and the
same architecture class, publishes no ONNX at all, and wraps it in a custom
`JinaForRanking` head: strictly more work for a worse licence at equal size.
Build the path on Qwen3 and Jina v3 becomes a one-line addition for anyone who
wants the comparison.

What remains is the latency, which the size ruling did not repeal.
595,776,512 total parameters less a tied 151,936 × 1024 embedding is about 440M
non-embedding, roughly twenty-one times the default tier, where `accurate` at
302M already spends 1423 ms of a 1522 ms search. **The prediction, recorded
before the run: about two seconds of reranking, a search around 2.2 s** —
coherent as an opt-in tier at 1.4 times `accurate`, and not a default at any
accuracy.

**A cost model, calibrated against a measurement, so the next size question is
arithmetic.** Every rejection above rests on an estimate of what a model would
cost here, and the estimates were made by ratio against `fast` without anything
anchoring them. One measurement anchors them: `accurate` is 302M non-embedding
parameters over a shortlist of about 770 tokens in 1423 ms, which is roughly
**330 effective GFLOPS** on four cores at int8. A transformer's forward pass is
about `2 × parameters × tokens`, and a reranker does one pass — it reads a score
or a last-position logit and generates nothing — so the whole table follows:

| | non-embedding | predicted reranking pass | |
| --- | --- | --- | --- |
| `fast` | 21.2M | 260 ms | **measured** |
| Laya, mmBERT-base | 110.3M | ~510 ms at int8, ~0.5–1.5 s at its float16 export | |
| `accurate` | 302M | 1423 ms | **measured; the anchor** |
| `Qwen3-Reranker-0.6B` | ~440M | ~2.05 s | |
| `openjev/openjev` | **27.4B** | **~128 s** | |

The last row is why "just read a yes/no token" does not rescue a large decoder.
Reading one position's logits still costs one full prefill over every weight, so
27.4 billion of them is two orders of magnitude outside an interactive budget
rather than a slow tier. That is a separate objection from its licence, its
absence of any ONNX export, its 159 downloads, and the independent evaluation
recorded above that puts the Jev line's reranking at **−0.028 under
judge-independent labels**.

**The same path makes Laya measurable, and the earlier rejection here was
narrower than it read.** What `head_max_len` of 256 tokens rules out is the
*listwise* framing — candidates as options, one softmax over the shortlist.
*Pointwise* is a different shape and is the one the model natively has: the
document goes in the state, within `max_len` of 1024, and only two short option
labels go in the head budget. Its `noul` question type returns the probability of
one of two options, with a per-option-bucket temperature and a confidence over a
bounded answer space. The reranker's encoder does not supply the marker
positions and query type that graph wants; a session built for it can.

**And the port is bounded rather than a reverse engineering job, because both
halves are published.** The export is `mizchi/laya-multilingual-onnx` —
Apache-2.0, 646.9 MB, **float16 weights with a float32 decision tail** (an
earlier version of this section said fp32 only, which was wrong), opset 18,
**standard operators only**, and its card reports validation against the
reference runtime at **63/63 answers agreeing, maximum probability error 5.1e-4
on the ONNX Runtime CPU provider**. Being float16 halves the download and the
resident set and buys no speed, because this record already measures fp16 as
slower than fp32 on a CPU where the runtime converts per operation.

The prompt format is given exactly by the published `rl_common.py`:

```text
[CLS] "<qtype> question: <instructions>" [SEP]
  [MASK]" false: no, the statement does not hold"
  [MASK]" true: yes, the statement holds"          [SEP]
  <state>                                          [SEP]
```

markers at the two `[MASK]` indices, `max_len` 1024, `head_max_len` 256, each
option capped at forty-eight tokens and the head truncated to what the options
leave. `render_options` fixes those two labels for a `noul` question and its own
comment records the property that makes this usable: *"Noul is always
`[false, true]` so `p[1] == noul`"*. So a relevance judgement is `qtype = 2`,
the query and document in the **state**, and `p[1]` read straight off as
`P(relevant)`.

That also settles why `head_max_len` was never a constraint on the document, as
the paragraphs above first assumed: the document goes in the state, inside
`max_len`, and only two short labels go in the head budget. What 256 tokens
rules out is the listwise framing, and only that.

One gap is named rather than discovered later. The published multilingual
checkpoint ships **no calibration**: its `rl_agent_config.json` carries
`temperature: [1.0, 1.0, 1.0]` and an empty `temperature_by_options`, which is
the identity — the over-confident state its own card puts at mean ECE 0.466. So
an arm has to fit the temperature on our own held-out judgements, which is the
same machinery the calibration work needs anyway. The two share a dependency
rather than competing for the slot.

So Laya becomes an arm rather than a dismissal, and **what it is being measured
for is not its ranking.** It is the calibrated, cross-query-comparable score
nothing else here produces, which is the piece the fusion, abstention and
reranker-gating questions all rest on. The risk to measure rather than assume is
that it is an RL agent trained on typed schema decisions and has never been
trained on query-document relevance, so any arm must report calibration —
a reliability curve and expected calibration error — and not only nDCG. A
well-calibrated mediocre judge is worth more here than an uncalibrated better
one.

**Laya is genuinely open and genuinely fast, and is the wrong shape twice
over.** Its multilingual encoder is `jhu-clsp/mmBERT-base` — 22 layers at width
768 with an `intermediate_size` of 1152, 110.3M non-embedding parameters, 5.2
times the default tier. Its head takes `marker_pos`, `marker_mask` and a
`qtype`, plants markers in a prompt and chooses among them; a reranker needs
one `(query, document)` pair in and one relevance logit out. The published 33
ms is one GPU, and "faster than Jev" is measured against a hosted API's network
round trip rather than against a matrix multiply.

Its card also states that *every question in a call is answered in one single
forward pass*, which would be a different cost model if it meant shared
encoding. It does not. The released `rl_agent_api.py` builds one sequence per
question and stacks them into a batch, so the state is re-encoded for every
question — one *launch*, not one *encode*, which is what this project already
gets from its reranker for a shortlist. The published latencies say the same
thing: 39.5 ms for one question, 158.6 ms for ten and 771 ms for fifty is 5.0
times the questions for 4.86 times the time between the last two, linear once
the GPU is full. And its `head_max_len` of 256 tokens is smaller than the
shortlist this project reranks — about 770 tokens of candidate text — so the
listwise shape, one pass and one softmax over the whole shortlist, is not
something that head can express.

One property of it is worth wanting and is recorded rather than dismissed:
**calibrated probabilities over a bounded answer space.** Every score in this
system is comparable only within one query — which is the constraint
`Combine::Banded` was derived from and normalises around. Nothing here can say
*how* relevant a result is in terms that mean the same thing for the next
query, and three separate wants need exactly that: a weighted-sum fusion, an
abstention gate, and the published threshold idea above.

**And the claim is evidenced rather than asserted, which is rare in this
field.** The model is trained with reinforcement learning against strictly
proper scoring rules, so its card can say that "reporting honest probabilities
is the only way to maximise reward", and it reports an expected calibration
error of **0.081** against its base checkpoint's 0.144 and the hosted Jev
model's 0.246. That is a number, measured, in the one dimension this project
has no instrument for at all.

**With one operational cost the card states plainly and nobody should discover
later.** It *ships over-confident*: mean ECE is **0.466** out of the box, and
reaching 0.081 requires refitting one temperature per `(question type, option
count)`. So adopting the calibration means fitting temperatures on our own
held-out data, per question shape, and maintaining them — which is real work
rather than a flag, and is the first thing to cost if that arm is ever run.

The risk on the other side is equally plain: its training workflows are invoice
processing, security incidents, customer service and agent-trace
observability. **Relevance ranking is not among them, and its card carries no
retrieval benchmark of any kind.** So an arm must measure calibration *and*
ranking, and a well-calibrated mediocre judge would still be the more useful
result of the two.

**What the field is missing is the model, not the architecture.** mmBERT is
multilingual ModernBERT, MIT, 1811 languages, and `mmBERT-small` is 42.2M
non-embedding — twice the default tier, not five times, with a narrow 1152-wide
MLP where the convention is four times the hidden size. That is the most
interesting encoder available for this slot. What does not exist is a
well-trained multilingual retrieval reranker on it: the one model with both the
architecture and the `text-ranking` shape is trained on product similarity,
which is the training-distribution mismatch this project has already measured
as the reason an off-the-shelf reranker loses. So it is a thing to watch for,
not a thing to adopt, and it goes into the sweep the day one appears
permissively licensed.

### What a calibrated score would restructure, and why the cheap version comes first

Everything above about a calibrated relevance probability is scattered through
three sections as a thing that would be nice to have. It is worth stating once
what it would actually change, because the answer is larger than a better
reranker and the route to it turns out not to need a new model at all.

**One constraint holds up the whole fusion design.** The four channels' scores
are not commensurable — a BM25 score, a cosine similarity and a hop-decayed
confidence are different quantities — so only their *ranks* can be combined.
Everything else follows from that: reciprocal rank fusion's contribution range
is narrow enough that a channel's *weight* decides against another channel's
*position*, `Combine::Banded` exists to preserve exactly that band, its floor is
`(k + 1) / (k + n)` for that reason, and the standardised sum lost
cross-lingual recall through the floor when it broke the property. One
constraint, one design.

A calibrated `P(relevant | query, document)` removes the constraint rather than
working around it, because such a number is comparable across channels, across
queries and across corpora:

| | today | with a calibrated score |
| --- | --- | --- |
| ordering | rank fusion combines incomparable evidence | sort by probability — **fusion collapses to recall**, and the weights and `k` leave the ordering path |
| thresholding | not expressible, which is how the borrowed idea failed here | an absolute cut means something |
| abstention | none at all | `max P < τ` and return nothing |
| channel weights | hand-tuned, and fusion still ranks *below* the vector channel alone on cross-lingual queries | fit offline against the scorer's own labels, at **zero runtime cost** |
| the graph channel | its `0.0000` cannot be interpreted | ask whether a graph-reached candidate scores above what its rank implies |

**With one counterweight, and it is load-bearing.** A system can only score what
it can afford to score: 16.9 ms a pair over fifty-one candidates is 860 ms, so
any real pipeline scores a subset — and the moment it scores a subset it needs a
rule for placing the unscored candidates among the scored ones. That is not a
detail to settle later; it is the same question as whether the reranker's score
should *join* the channels' evidence or *replace* it, and it is therefore the
necessary shape of any partial-scoring architecture, calibrated or not.

**And the cheap version comes first, because a cross-encoder can be calibrated
too.** A temperature, Platt or isotonic fit on the tier already running —
against held-out judgements — yields a calibrated probability from the model
this project already pays 260 ms for. No new model, no new dependency, no
647 MB download, and none of the per-question-shape temperature maintenance the
alternative's own card describes.

It is measurable offline from data already on disk, which it was not before
`Why::Reranked` existed: **18,353 `(score, relevant?)` pairs over 1,190
queries** come straight out of the trace, and Platt is a two-parameter fit. So
the thing blocking every row of the table above was never the model. It was
that the score was computed and thrown away.

That ordering also improves the model question either way. If calibrating the
shipped tier opens those doors, a purpose-trained calibrated judge becomes a
candidate for a *better* one, measured against a calibrated baseline rather
than against nothing. If it does not open them, a purpose-trained one very
likely will not either — calibration is notoriously corpus-specific, and
cross-corpus comparability is the exact property being bought. So the fit is
made on one corpus and **the calibration error is reported on another**. A fit
that does not transfer is a negative result worth publishing, and it would
predict the same failure for anything calibrated per question shape.

**Measured, and the shape of the fit is the finding.** `CALIBRATE=1` on
XQuAD-R, 9,147 pairs fitted against 9,175 held out, split so no query is on both
sides, 43.8% of the cross-lingual pairs relevant. Expected calibration error on
the held-out half:

| | cross-lingual |
| --- | --- |
| the raw sigmoid of the logit | 0.0905 |
| Platt, with label smoothing | 0.2146 (**+0.1241**) |
| isotonic regression | **0.0172** (−0.0733) |

So a calibrated cross-query-comparable probability *is* available from the tier
already running, at zero new download and zero new runtime cost — but only from
the monotone fit. The reliability table under isotonic is diagonal across ten
bins: 0.045 predicted against 0.042 observed on 1,669 candidates, 0.237 against
0.258, 0.633 against 0.643, 0.943 against 0.918.

**Platt fails here, and not for want of regularisation.** The first attempt was
read as a missing-regulariser bug in this repository and it was not: with
Platt's own `(n + 1) / (n + 2)` target smoothing in place the coefficient is
still −81, which is separability divergence — a two-parameter sigmoid pushed
towards a step function by data it can nearly separate. The prediction that
smoothing would fix it was wrong and is withdrawn. What fixes it is choosing a
fit that cannot diverge: isotonic regression is a monotone step function and has
no coefficient to run away.

**The same-language row of that run is not a result.** Its positive rate reads
0.2%, isotonic collapses to a single bin predicting 0.002 everywhere, and its
ECE of 0.0001 is what a constant predictor always scores — a degenerate fit, not
a calibrated one. That rate is also the wrong number on its face: 1,190 queries
with one gold sentence each over 18,353 candidates is 6.5%, not 0.2%, so what
the arm is labelling in that group has to be explained before any figure from it
is quoted.

And the caveat the harness prints itself still stands: this fit is made and
tested on one corpus, so it bounds the within-corpus case and says nothing about
another. The transfer test is the one that decides whether the fusion rows of
the table above can be believed, and it belongs on a corpus this was not fitted
on.

### The transfer test, taken as an abstention decision: measured, not shipped

The decision that would use a calibrated score first is abstention, and it has
a column to be scored on: LoCoMo's adversarial questions, where upstream's own
evaluation counts only "not mentioned" as correct. So the transfer test was
run as that decision, through `pamin search` itself, with the rule written
down before any LoCoMo or LongMemEval search ran.

**The signal.** The `accurate` tier's logit for the top hit `pamin search`
returns -- reused when the reranker already scored it, scored as one more pair
after the ranking is final when it did not, so the order returned is
unchanged -- mapped through an isotonic fit, with the verdict `weak` below a
probability of 0.5. The half is the equal-cost decision on a calibrated
probability, not a fitted cut. The results were still returned; the verdict
was advice beside them.

**The fit, on a third corpus.** 484 MuSiQue answerable dev questions, every
fifth, their 5,964 paragraphs pooled into one project, labelled by whether the
top hit is a supporting paragraph (74.4% were). Held-out expected calibration
error, fitted on one half by question and scored on the other: **0.0590**
isotonic against 0.1804 for the raw sigmoid. Within a corpus the fit works
again, as it did on XQuAD-R.

**The rule.** Ship the verdict as a field only if, paired per question with an
exact two-sided McNemar test: adversarial improves at p < 0.05, none of the
four answerable columns falls at p < 0.05, and LongMemEval-S does not either.
An answerable question counts as served when an evidence turn is in the top ten
*and* the verdict is `sufficient`; an adversarial one when the verdict is
`weak`.

**Measured on all 1,986 LoCoMo questions**, one project per conversation, the
turns written as the benchmark harness writes them:

| column | n | never abstains | with the verdict | wins | losses | p | `weak` rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi-hop | 282 | 0.791 | 0.592 | 0 | 56 | 3e-17 | 0.277 |
| temporal | 321 | 0.826 | 0.670 | 0 | 50 | 2e-15 | 0.227 |
| open-domain | 96 | 0.521 | 0.292 | 0 | 22 | 5e-7 | 0.479 |
| single-hop | 841 | 0.810 | 0.718 | 0 | 77 | 1e-23 | 0.127 |
| adversarial | 446 | 0 | **0.314** | 140 | 0 | 1e-42 | 0.314 |
| all five | 1,986 | 0.614 | 0.581 | 140 | 205 | 0.0006 | |

On LongMemEval-S, the 59-question sample the published figures use,
recall_any@10 falls from 0.983 to 0.627 (no wins, 21 losses, p = 1e-6), the
verdict calling a third of the top hits `weak`. On its 30 false-premise `_abs`
questions, where abstaining is the benchmark's answer, it is `weak` on 22
(p = 5e-7), but the same threshold withdrew 21 of the 58 answers the sample had
retrieved: the two sets separate at AUROC 0.716.

**The rule fails, on every guard at once.** The verdict abstains on a third
of the adversarial questions, and to do it withdraws between 11% (single-hop)
and 44% (open-domain) of the answers every other column had retrieved; pooled
over all five, LoCoMo gets significantly worse. It does not ship.

**Why: the signal barely separates the two, and the fit does not travel.**
The probability ranks an answerable question above an adversarial one with
AUROC **0.606** (0.640 counting only answerable questions whose evidence was
retrieved) -- median 0.762 against 0.600. That was predicted before the run
(0.55 -- 0.65), for the reason given then, which the run is consistent with
but does not isolate: 74% of its
questions have a turn that matches exactly and was only said by the other
speaker, and a relevance model is not trained to care who said something. The
calibration did not transfer either. On LoCoMo's answerable top hits the
MuSiQue map scores an ECE of **0.3100**, against 0.0590 on its own held-out
half: it says 0.372 where 11.4% are relevant (297 hits) and 0.977 where 80.6%
are (391). A conversational turn is short and indirect beside a Wikipedia
paragraph, and 42.1% of LoCoMo's answerable top hits are an evidence turn against 74.4%
of MuSiQue's, so the same logit means something different. LongMemEval says
the same: ECE 0.2951 on its top hits.

So the caveat above resolves the way it was feared: the isotonic fit is a
within-corpus result. What the cross-query-comparable score still offers --
the rows of the table in the section above -- needs a fit per corpus, or a
judge trained to be calibrated across them, and an abstention decision needs a
signal that sees attribution, which a relevance score does not.

Conditions: a release build of the verdict (`2a72188`, `5d7adf0`) with an
empty table, so each row recorded the raw sigmoid of the logit; the table was
fitted afterwards and applied to those logits offline, which is the same
lookup the build with the table performs. `pamin search --limit 10 --json` at
the shipped tier, four cores at a load average near 18. The latency of the
extra pair was not measured, because the rule failed on accuracy first; the
top hit arrived without a reranker score on 1,789 of the 1,986 LoCoMo
searches. Reverted in `8dcfd14`.

### Where the decision-model field is, and which of it a CPU can reach

An independent leaderboard now exists for this class of model — the Jev
Decision Index, 132,422 requests over 19 benchmarks in five areas, 31 open
reproductions against the hosted target. It is the best external evidence this
project has found on the question, and it is worth recording what it does and
does not settle.

**What it does not settle: whether this project's retrieval is better or worse
than theirs.** Nothing has been run on both. The figures here are XQuAD-R and
MIRACL; the figures there are BRIGHT and Amazon ESCI. Those are different
corpora, so no comparison exists in either direction, and the honest position
is that the question is open until one of their benchmarks is run here. That is
why two of them are now on the backlog as corpora rather than as competitors.

**What it does settle is the shape of the field, and it is narrower than it
looks.** The category called *Retrieval & Classification* is four
classification benchmarks — BANKING77, CLINC150, SGD and Amazon ESCI, all
macro-F1 — and one retrieval benchmark, BRIGHT at nDCG@10. On the one that
ranks documents, the whole field tops out at **0.1933** against a random
baseline of 0.0448, with the hosted target at 0.1869. So a headline category
score near 37 is mostly intent classification, and nothing in it establishes
reranking quality for passage retrieval.

**Every latency in it is one GPU.** Their methodology states it: "1 x NVIDIA
RTX PRO 6000 (Blackwell Server Edition, 96 GB), one GPU per run", with the
target itself over a hosted HTTPS API. There is not one CPU number in the
suite. And the per-benchmark view shows what the aggregate medians are medians
*of*: a single classification decision is 76 ms on ESCI and 267 ms on CLINC150
— where the option count is 4 and 150 — while **BRIGHT, the one benchmark that
ranks a candidate pool, is 1.95 s** for the fastest entrant and 502 ms for the
quickest technique. Milliseconds per decision, seconds per ranked pool.

#### Why none of the techniques transfer, enumerated rather than asserted

Eight of the entrants are techniques rather than weights. Filtered to the
retrieval category they are: autoregressive label-scoring (Jevfire, openvons,
SemIf, mini-jev), text diffusion in a decision mode (both diffusiongemma
ports), a GLiFormer encoder (`jeff`), and one RLCD fine-tune.

**They all make token generation cheaper or more bounded, and this project
generates no tokens.** Jevfire's own README is explicit about the baseline it
beats: "Score labels; assemble the object in Python" against "Autoregressively
emit the object", for 10.3x on a 27B model with 28 fields. A cross-encoder is
already the first of those — one forward pass, a score head, the ranking
assembled by the caller. The saving is against a cost this project never paid,
which is the same structural point this record already makes about Jev against
dedicated classifiers.

**One of them points somewhere useful, which is worth more than the technique
itself.** Jevfire's other saving is prefix reuse: independent fields sharing a
prompt prefix are batched with the instruction cached. This project
re-encodes the query 15.4 times per search, once per candidate, so prefix reuse
would be exactly the right saving — and **a cross-encoder cannot have it.**
Prefix caching needs causal attention, where the prefix's state does not depend
on the suffix; a cross-encoder is bidirectional, and the query's representation
depending on the document is the whole reason it beats a bi-encoder. So the
saving is unreachable by construction in this architecture, and the
architecture that does have it is **late interaction**, where query and
document are encoded separately and meet only at scoring.

#### What a CPU can actually reach, with the arithmetic shown

The constraint is four cores and 15 GB, and it has two halves that this record
previously conflated.

**Memory is not the blocker for a 4B model, and an earlier claim here that it
was is withdrawn.** At int8 a 4.02B model is about 4 GB of weights and at q4
about 2 GB, against a server already resident at 3.6–7.2 GB. That fits on the
smaller workspace. What genuinely does not fit is the mixture-of-experts
entrants: 26B and 36B of weights must all be resident even though only 3–4B are
active per token, so their compute is cheap and their footprint is two to three
times the machine.

**Latency is the blocker, and two independent estimates agree.** Against the
anchor this record measures — 302M non-embedding parameters over 770 tokens in
1423 ms, roughly 330 effective GFLOPS — a 4B pointwise reranker over 15.4
candidates of about 100 tokens each is `2 × 4.02e9 × 1500 / 330e9`, about 36
seconds. Independently, CPU prompt-processing throughput for a 4B model at q4
on four cores is in the tens of tokens a second, which puts 1,500 tokens at 19
to 38 seconds. Two methods, one order of magnitude. Not "cannot" in any
physical sense; "can, at half a minute", which for an interactive memory search
is the same answer.

**Two things in this class are CPU-reachable, and both are measured rather than
assumed to be good.** A purpose-trained 0.6B reranker — not a Jev-style
decision model, which is why the leaderboard's sub-1B entrants failing at
retrieval says nothing about it — is about 440M non-embedding, predicted near
2 s, permissively licensed with a single-file quantized export already
published. And late interaction moves the pass to write time entirely, where
the cascade already runs a forward pass per memory, leaving query time as a
dot product.

#### Late interaction: the one candidate that removes the cost rather than moving it

**Measured, and it is worse than not reranking at all.** The arm reconstructs
the exact subset the `accurate` tier is offered — the tier's own rule, read off
the trace, with the tier's counter asserted to agree — and reorders it by MaxSim
instead. XQuAD-R, 1,190 queries, nDCG@10:

| | cross-lingual | against fusion alone | same-language |
| --- | --- | --- | --- |
| fusion alone | 0.6114 | — | 0.7829 |
| **mLateOn** | **0.5857** | **−0.0257**, 381 wins / 530 losses, `p = 0.0001` | 0.7808 (n.s.) |
| `accurate` | 0.6597 | +0.0482, 560 / 192 | 0.7835 (n.s.) |

Head to head it loses to `accurate` by 0.0739, 144 queries to 729. The
prediction written before the run was that it would lose to `accurate` and
might beat `fast`; it lost to every tier including `off`. The query-side cost
was about 82 ms, which was the whole case for it, and it does not matter at
this accuracy. It is the same shape of result the typed judge gave — a model
whose card is strong on the benchmarks it was trained toward and whose
reordering of *this* pipeline's candidates is worse than the fusion order it
replaces — and for the same practical reason the storage never had to be
built. The implementation and the arm were deleted; this paragraph is the
record, and the commit that removed them is where to reproduce it.

`lightonai/mLateOn` is Apache-2.0 on ModernBERT, and the export was checked
rather than taken on trust. `model_int8.onnx` is 312 MB — smaller than the
`accurate` tier's 571 MB — and the projection head is **not** folded into it:
three `Dense` modules ship as separate weights, 768→1536, 1536→768 and
768→128, all with identity activation and no bias, the first two residual. So
using it means three matrix multiplies after the session, about 2.5M
multiply-accumulates a token, which is nothing. Its `onnx_config.json` supplies
the rest: `[Q] ` and `[D] ` prefixes at token ids 256000 and 256001, 128
embedding dimensions, and no query expansion.

The trade is the first one on this page that is genuinely four-axis:

| | change |
| --- | --- |
| query latency | the 260 ms cross-encoder pass becomes **a dot product** — MaxSim over stored token embeddings, no model at query time |
| write | one forward pass per memory, which the cascade already runs — but it cannot replace the pooled vector the HNSW index needs, so it is a second head or a second pass |
| disk | 128 dimensions at int8 is 128 bytes a token, so 12.8 KB for a hundred-token memory: 167 MB on the evaluation corpus and **about 5.5 GB on a 425,916-row workspace, three times the whole database** |
| licence | Apache-2.0, nine languages, on the 2025 architecture |

**And the disk cost has a published answer, which is what makes the trade worth
taking seriously**: ColBERTv2 and PLAID compress these embeddings to a centroid
plus one or two bit residuals for roughly 20 to 30 times, which would put 5.5 GB
at 200 to 400 MB — about the size of the whole `topic_states` table on that
workspace (239 MB), and two to four times the duplicated content column (94 MB)
the store has since stopped keeping. The compression is part of the same piece
of work as the measurement, not a later optimisation.

#### The two product rulings that narrow all of this

**A hosted API is out**, on cost and on unpredictable latency, which is the
same position the offline design already took for different reasons.

**An optional local accelerator is in, and it is the revisit this record
named.** The GPU section above rejected *linking* an execution provider and
said plainly that what to reconsider first is loading one at runtime. That is
what an optional, auto-detected accelerator is, and the size budget does not
move because the provider is never bundled.

What changes is the target rather than the argument. The old rejection measured
accelerating the *small* models already shipped and found the win marginal or
negative — 27 ms becoming 42 on CoreML, DirectML often slower than the CPU on
integrated graphics. Those measurements stand. What they do not cover is a
model class that cannot exist on the CPU at all.

**The binding constraint is correctness, and it comes from this record's own
strongest objection to GPUs: DirectML is reported to return numerically
divergent results on Intel integrated graphics.** A path that orders results
differently would mean two users getting different answers from one workspace,
every accuracy gate passing on CPU while saying nothing about the other path,
and bug reports that do not reproduce. So a provider is accepted only after a
**startup self-check** — a committed fixture through both paths, compared, with
any disagreement past a stated tolerance falling back to CPU permanently and
saying so. That converts "quality depends on hardware" into "speed depends on
hardware, output verified identical", and it is the answer to the objection
rather than an override of it.

Unmoved, and stated so it cannot drift: the default path is CPU-only, every
gate is measured on CPU, and **every published figure stays a CPU figure** —
otherwise the numbers on this page stop being product claims and become
hardware claims.

### Optional GPU: measured against, not deferred

Accelerating the reranker on a GPU was considered and is not being built, and the reason is not the size budget alone.

- **CUDA is the only execution provider with a large win, and it cannot be shipped.** `libonnxruntime_providers_cuda.so` is 220–340 MB in official releases, against a 92 MB distribution, and it additionally requires the user to have installed a matching CUDA and cuDNN. Raising the budget does not fix the second half.
- **On the machine a user actually has, the accelerator loses.** A comparable Rust stack on the same runtime measured CoreML turning 27 ms into 42 ms, because op coverage excludes the embedding lookup and mask arithmetic and the graph partitions fourteen ways. Apple's neural engine is fp16, so the int8 exports this project ships fall back to CPU per operation, silently, and a second fp16 export of every model would be needed to use it at all. Per-dispatch overhead is about 2.3 ms, against three dispatches per query at batch 8.
- **DirectML on integrated graphics is documented as often slower than the CPU**, compiles shaders per input shape on first use, and is reported to return numerically divergent results on Intel integrated GPUs — which would make this project's own quality gates hardware-dependent.
- **The work is the wrong shape.** Twenty short pairs and one query embedding are tiny batches, and a GPU's advantage is throughput. This project's bottleneck is 226 ms spent on queries that, by its own measurement, mostly do not need it: 84 of MIRACL's 482 queries have a relevant passage below rank 10 before reranking, and on XQuAD-R's same-language group reranking makes the ordering *worse*. Removing work beats moving it.

Execution providers stay available as non-default cargo features for anyone building from source, and the shipped distribution stays CPU-only. If that is revisited, the thing to revisit first is loading a provider at runtime rather than linking one, which is what keeps a single binary inside a size budget.

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
last. The reranking row reproduced the tier table's then-current 211 ms from a
different direction, which was the check on the method. The tier table now
reads 260 ms for the same pass at the new fusion weights, so **this division is
a stage breakdown of a search that no longer exists at these absolute
figures**; what it still establishes is the shape, and it is not a baseline for
anything in the five-tier table.

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

**Idle inference threads block instead of spinning.** ONNX Runtime lets an
intra-op thread that runs out of work spin before it sleeps, and nothing here
had said otherwise, so every table above was taken with it spinning. The
embedder and the reranker each own a pool of one thread per core, so on a
machine with other work -- the upkeep drain's embedding, another caller, or
another process -- the spinning threads hold cores the working threads need.
`inference::session` now turns it off for every model it loads. The rule was
written before anything was timed: ship if rankings and scores are
bit-identical, the paired search-time ratio at the machine's own load is at
most 1.05, under a fixed contention it is at most 0.80 at `p < 0.01`, and eight
concurrent callers get no less throughput. Through `pamin serve` and `pamin
search` at the defaults, one hundred queries (sixty from the own corpus, forty
from an XQuAD-R subset of three languages), a fresh server per arm and round so
no query is a cache hit, three rounds in rotated order, on four cores:

| | ambient load (2.5 to 13) | two busy loops beside it (load 9 to 13) |
| --- | --- | --- |
| search, spin off / spin on | 1.033, faster on 39/100, `p = 0.068` | **0.754**, faster on 86/100, `p = 0.0001` |
| own corpus / XQuAD-R | 1.027 / 1.042 | 0.720 / 0.809 |
| eight callers, throughput ratio | 1.17 (1.22, 1.46, 0.89) | 1.13 (1.04, 1.55, 0.90) |
| one global pool, spin off / spin on | 0.964, `p = 0.037` | 0.886, `p = 0.0002` |

Ratios are geometric means of per-query ratios, each query's time the median of
its three rounds; `p` is a paired sign-flip test. Every ranking of all two
hundred queries was identical across arms and rounds, and in process forty
query embeddings, sixteen passage embeddings and 256 `accurate` scores were
bit-identical. What it costs where nothing competes was not measured -- this
machine was never quiet -- and the ambient column, at 1.033 and not
significant, is the nearest thing to it. One pool shared by both models, through
`ort`'s global thread pool, was the other candidate: it passes the same rule
but gives up most of the gain under contention, so it does not ship. The
`speed` and `balanced` profiles' embedders load through `fastembed`, whose
options do not reach this setting, and still spin; both rerankers and the
default profile's embedder do not.

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
