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
| `lexical_ngram` | 0.0528 | 0.0106 | — |
| `vector` | **0.8268** | **0.6335** | 0.6787 |
| `graph` | 0.0000 | 0.0000 | 0.0000 |
| all four fused | 0.7910 | 0.6077 | 0.7556 |

**Fusing four channels ranks below one of them on cross-lingual queries**, and
on the same-language queries of the same corpus the lexical channels earn their
place outright. Leave-one-out, paired against all four: taking
`lexical_segmented` away is **+0.0310** on this project's cross-lingual group
(16 wins, 1 loss, p = 0.0011) and **+0.0158** on XQuAD-R's (496 wins, 30
losses, p = 0.0001), while `lexical_ngram` is +0.0223 (13 / 1, p = 0.0173) and
+0.0110 (333 / 20, p = 0.0001). The same removal costs **−0.0442** on XQuAD-R's
same-language group (14 wins, 275 losses) and −0.0073 on MIRACL (p = 0.0125). The channels
are not weak. A global constant cannot tell the two cases apart — this is the
weakest link the four-channel paper above names, arrived at independently.

**The two lexical channels are not one channel.** Kendall tau-b between their
rankings, over the candidates they share: 0.2816 on this project's own corpus,
0.3188 on XQuAD-R, 0.2973 on MIRACL. They agree about a third of the time. The
premise that justified one shared weight is refuted, and every sweep this
project ever ran moved both together, so no measurement distinguishes the two
numbers at all.

**The graph channel contributes exactly 0.0000** — in every group of all three
corpora, so removing it changes no ranking anywhere. That is a weaker statement
than it looks and is recorded as an observation rather than a verdict. Two of
the corpora are sentence collections with no relationships to walk, so the zero
there is a property of the corpus. This project's own corpus does have edges
and still reports zero, and that part is unexplained: nothing here has
established whether the walk reaches nothing relevant or reaches it at a rank
that `weight / (k + rank)` places below the other channels' candidates.

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
`k` is worth one to three points and normalisation three to eight. `Combine`
implements all three combiners, `Reciprocal` still ships, and the offline grid
prices each of them with and without the confidence rule, because the two
mechanisms answer different halves and a row that moved both could not say
which half moved it. One prediction is written down in a unit test rather than
here: CombMNZ multiplies by agreement, and agreement between a confident
channel and a worthless one is exactly the failure measured on both
cross-lingual groups above, so the combiner the systematic comparison ranks
first should rank last on these corpora.

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

`Fusion::with_confidence` implements it and **is off by default because it has
not been measured**. The three corpora's accuracy floors stand on constant
weights. Its two constants, and the two lexical weights now that they are
separable, are swept offline from one pass over each corpus — `channels::as_if`
replays a run's trace through the shipped `fuse`, so a grid that used to cost
thirteen minutes a row costs microseconds a row. Until that sweep is run and
reported with per-query wins, losses and a bootstrap p, the mechanism is a
hypothesis with an argument behind it and no number.

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
and a paired bootstrap p alongside every mean, and the cross-lingual harness
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
`1 + 0.2 * importance` and by `1 + 0.2 * worth`. Both are columns the
repository reads and **no code path anywhere writes**, so both factors were
exactly 1.0 on every search this project has ever run, and the trace lines for
them were already suppressed on the grounds that they said nothing. A modifier
over a constant is not a ranking signal; it is a multiplication. The columns
stay, because they are the authority store's schema, and `RetrievalSignals` now
says outright that nothing writes them — restoring the feature starts with a
write path, not with a multiplier.

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
| Stored vector INT8 | Output embeddings stored as int8 rather than float32 | No recall loss at all, half the query and half the build, **30% more disk** | **Off, because it is a disk loss** — see below |

Weight quantization is a trade worth taking, and the default profile takes it. The registry publishes no quantized variant for multilingual E5, which is why the two E5 profiles still run full precision and why an earlier version of this decision recorded the trade as unavailable. It is available for BGE-M3, through a joint int8 export (`gpahal/bge-m3-onnx-int8`, MIT, exported from the MIT-licensed base model), and the difference is what makes that profile the default: 560 MB resident against the full-precision export's 2.2 GB, 35 ms a query, and 0.6550 cross-lingual nDCG@10 on Påmin Memory's evaluation corpus against the full-precision 0.6720 — both at the lexical weight of that day, a half.

### Quantizing the stored vectors: measured, and it is the wrong lever

This decision recorded stored-vector quantization as deferred "until the binding exposes rotation", and expected it to be a disk saving — vectors are 55% of a real index's bytes. Both halves turned out wrong, and one of them was a defect this project shipped.

`PAMIN_VECTOR_STORAGE` builds the projection's vector field five ways. Over 50,000 clustered 1024-dimensional vectors in four segments, everything else held equal, on an otherwise idle machine:

| storage | recall@10 | per query | build | whole index |
| --- | --- | --- | --- | --- |
| **fp32, the default** | **0.9980** | **12.0 ms** | **88 s** | **225.9 MB** |
| fp16 | 0.9980 | 10.9 ms | 82 s | 337.0 MB |
| int8 | 0.9980 | 7.3 ms | 43 s | 292.6 MB |
| int4 | 0.9990 | 9.3 ms | 45 s | 269.4 MB |
| rabitq | — | — | — | refuses to train without a `raw_vector_provider` |

**Quantizing cannot save disk here, because the refiner keeps the full-precision vectors.** Every quantized row carries the same 206.4 MB of raw vectors — 50,000 × 1024 × 4 bytes is 204.8 MB, so that column is the raw copy — and adds its codes on top. So the trade is not "smaller index for slightly worse recall"; it is **19% to 49% more disk for half the query time and half the build**, at no measured recall cost. That is a real trade and it is the opposite of the one this decision went looking for, so the default stays fp32 and the mechanism stays available to a workspace that would rather spend disk than milliseconds.

The measurement could not be made by reading the code. Whether a refiner stores a second copy beside the codes is not visible from the binding's surface, and it is the whole answer.

**And rotation, enabled here on reasoning, was destroying the index.** The parameter was set for every quantized storage because spreading the bits across dimensions that carry comparable information must help the coarse storages and could not hurt the others. The engine accepts it only for int8 and int4 — for anything else it refuses when the *segment* opens its vector field rather than when the parameters are built, so fp16 presented as a segment that would not take writes rather than as a rejected setting. And on the two storages that accept it, it is ruinous:

| | recall@10 with rotation | without |
| --- | --- | --- |
| int8 | 0.0530 | 0.9980 |
| int4 | 0.0580 | 0.9990 |

Same bytes, same build time, same query time. This is the failure shape recorded above from the previous quantization attempt — an index returning plausible neighbours that are not the nearest ones, with no error anywhere — and it was reproduced here only because the harness reports recall rather than whether the calls succeeded. Rotation needs a fitted transform and nothing supplies one; the binding's RaBitQ path is explicit about it, refusing to train without a `raw_vector_provider` the binding does not expose. It is off, and `crates/pamin-index/tests/scratch_quantize.rs` is what would notice if it came back.

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

### What the two shipped tiers are licensed under, and what was surveyed against them

Licences were checked at the leaf and at the base, because a fine-tune's card can declare a licence its base does not permit.

| tier | model | licence | base |
| --- | --- | --- | --- |
| `fast` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `apache-2.0`, declared on the card | `nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large`, **no licence tag**; MiniLMv2 originates in `microsoft/unilm`, MIT |
| `accurate` | `onnx-community/bge-reranker-v2-m3-ONNX` | **no licence tag** — its front matter is `library_name` and `base_model` and nothing else | `BAAI/bge-reranker-v2-m3`, `apache-2.0` |

Both chains are defensible and neither is stated where it is shipped from. That is worth an upstream request or a self-controlled export; it is recorded here rather than left to be rediscovered.

**The survey against them, and the reason none of it changed the default.** Every candidate below was checked for a readable permissive licence first, because a model that cannot be shipped does not need measuring.

| candidate | licence | why not |
| --- | --- | --- |
| `jinaai/jina-reranker-v2-base-multilingual`, `-v3`, `-v3.5`, `jina-colbert-v2` | **CC-BY-NC-4.0**, the whole line | Non-commercial. Not measurable for a default |
| `BAAI/bge-reranker-v2-gemma` | card says `apache-2.0`; base `google/gemma-2b` is `license: gemma`, gated | The Gemma rider follows the derivative — the same reason EmbeddingGemma was refused above |
| `BAAI/bge-reranker-v2-minicpm-layerwise` | card says `apache-2.0`; base MiniCPM weights carry the General Model License with a commercial-authorization requirement | Painful, because its 8–40 selectable output layers are exactly the early-exit mechanism the latency problem wants |
| `naver/splade-v3` family | CC-BY-NC-SA-4.0 | Non-commercial and share-alike. `Splade_PP_en_v1` is Apache-2.0 and English |
| `Qwen/Qwen3-Reranker-0.6B` | `apache-2.0` — the cleanest licence and the best multilingual quality in the field | A decoder at roughly twenty times the compute-relevant parameters of `fast`. Estimated seconds a query on four cores; three to six times the `accurate` tier, which is already not an interactive budget |
| `mixedbread-ai/mxbai-rerank-base-v2` | `apache-2.0` | MIRACL 28.56. Not a multilingual reranker in the sense this product needs, whatever the language count says |
| `Alibaba-NLP/gte-multilingual-reranker-base` | `apache-2.0`, with an int8 ONNX re-export | Four times `fast`'s compute for a 12-layer model. A plausible middle tier on long documents; not a path below 226 ms |
| `nreimers/mmarco-mMiniLMv2-L6-H384-v1` | **no licence tag at all** | The obvious "halve the layers" move, unavailable for the reason this project's rules anticipate |

**Jev, and the shape of its claim.** TypeSafe's Jev is a decision model: text in, a number out, no token generation. It is API-only at $0.042 per million input tokens, so it cannot be part of an offline product whatever its quality. The open recreations do not rescue it. `openjev/openjev` is CC-BY-NC-4.0 and 27B parameters — 54 GB in fp16, and its own card measures about 80 ms **for one short decision on an H100 in fp8**, where this product scores twenty pairs in 226 ms on four CPU cores. `jaredpalmer/kev-0.8b` is Apache-2.0 at the adapter and base, tagged `language: en`, and its declared training data includes `Yelp/yelp_review_full`, whose terms grant academic use only.

The speed claim is real and does not apply here. Parallel, evaluating Jev independently, states it plainly: *"The headline cost and speed comparisons are against autoregressive LLMs, not dedicated classifiers. Specialized classifiers will still often outperform Jev on cost and speed."* A cross-encoder is a dedicated classifier — one forward pass, no generation — so the thing Jev's design removes is a cost this product never paid. An independent multilingual evaluation (9,831 graded pairs, 164 Chinese and English queries) puts Jev's rerank at **+0.012 over BGE-M3 under its own labels and −0.028 under judge-independent labels**, and measures the circularity that separates them.

What is worth taking from it is not the model. `jev-reranker`, a library wrapping the API, reports that a relevance *threshold* — dropping candidates rather than reordering all of them — removed about 92% of the candidates and scored higher than reordering them, 0.975 against 0.969 nDCG@10. That is a gate, and a gate needs no weights at all.

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
