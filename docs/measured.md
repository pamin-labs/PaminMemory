# What Påmin Memory measures about itself

Every figure here comes from `pamin search` and `pamin write` themselves, not
from the model or the index underneath them, because the gap between those two
is where this project's numbers have been wrong before. The README carries the
summary; this page carries the numbers and the conditions they were taken
under.

The comparison against other memory systems is a different question and lives
in [benchmarks.md](benchmarks.md), along with what that comparison holds fixed
and how each condition is asserted. The committed evidence behind both pages is
under [benchmarks/results/](../benchmarks/results).

**Retrieval quality**, at the shipped defaults, median of three runs:

| corpus | group | nDCG@10 | recall@50 |
| --- | --- | --- | --- |
| MIRACL Swahili dev — 131,924 real passages, 482 queries, 5,092 human judgements | one language throughout | 0.7359 | 0.9494 |
| XQuAD-R — 13,014 sentences in eleven languages, 1,190 queries | query and answer in **different** languages | 0.6097 | 0.8864 |
| XQuAD-R | query and answer in the same language | 0.7971 | 0.9630 |

Both corpora are fetched rather than vendored, and each has a harness in the
repository: `cargo test -p pamin-engine --test crosslingual -- --ignored` for
XQuAD-R and `--test monolingual` for MIRACL.

**The MIRACL rows on this page are older than the harness named beside them,
and that has to be said rather than tidied away.** This page previously named
one harness for both corpora, which was true of the XQuAD-R rows and false of
the MIRACL ones: they came from a program that was never committed, so nothing
in the repository could produce them, break them, or be trusted to notice.
`monolingual.rs` exists to close that, and every MIRACL figure below is a
target for it to reproduce until it has — at which point this paragraph goes
and the figures carry floors, which they also do not have today.

**What those MIRACL figures are worth, against published results on the same
corpus, the same dev split and the same qrels:**

| MIRACL Swahili dev, 131,924 passages | nDCG@10 | what it is |
| --- | --- | --- |
| Pyserini BM25 baseline | 0.3826 | lexical only |
| Påmin Memory, `--rerank off` | 0.7158 | four channels fused |
| Påmin Memory, `fast` (default) | **0.7359** | fused, then a cross-encoder |
| Påmin Memory, `accurate` | 0.7654 | fused, then a larger cross-encoder |
| BGE-M3, published | 0.787 | dense retrieval alone |

Read the last row carefully, because it is the honest reading: **a whole
retrieval stack here scores below a single dense retriever** — the same model,
as an int8 export. Two differences are known and neither is measured: the
published figure is fp32, and MIRACL's training split is in BGE-M3's
fine-tuning data where this runs zero-shot. Neither excuses the gap; they are
where to look for it.

**Since this was written, the other corpus answered where to look, and it is
neither of those.** On XQuAD-R the embedder alone and the product run side by
side in one harness, so their difference is the fusion layer's own effect, and
it is −0.0635 on the cross-lingual group: fusing four channels dilutes a dense
ranking, and the cross-encoder's +0.0374 is buying that back rather than adding
to it. −0.071 here is the same order. That reading is an extrapolation from a
corpus of parallel sentences until `monolingual.rs`'s model-alone arm runs
here, which is the one arm this corpus has never had — see
[ADR 0001](adr/0001-tech-selection.md).

What the table does establish is the distance from the lexical baseline a
memory system would otherwise ship with, on a low-resource language, on four
CPU cores with no GPU anywhere.

Sources: [Pyserini MIRACL v1.0 regressions](https://github.com/castorini/pyserini/blob/master/docs/experiments-miracl-v1.0.md)
and [BGE-M3](https://arxiv.org/abs/2402.03216) Table 1 (v4 or later). The
0.7359 was re-run and reproduced exactly before being placed here -- by the
program that is not in the repository, which is what the note above the first
MIRACL table is about: reproduced twice by the same uncommitted thing is not
the same as reproducible.

**Retrieval on a memory benchmark.** LongMemEval-S, 59 of its 500 questions
drawn stratified by type, scored at the session level against a plain BM25 over
the same turns — because a retrieval number without a lexical baseline says
nothing about retrieval. Method and definitions in
[benchmarks.md](benchmarks.md):

| LongMemEval-S, session level, 59 questions | BM25 | Påmin Memory |
| --- | --- | --- |
| recall_all@5 | 0.7966 | 0.8983 |
| ndcg_any@10 | 0.8898 | 0.9202 |

The total is not the result. Split by question type it is:

| recall_all@5, by question type | n | BM25 | Påmin Memory |
| --- | --- | --- | --- |
| multi-session | 15 | 0.467 | **0.867** |
| temporal-reasoning | 16 | 0.750 | 0.750 |
| knowledge-update | 9 | 1.000 | 1.000 |
| single-session-user | 8 | 1.000 | 1.000 |
| single-session-assistant | 7 | 1.000 | 1.000 |
| single-session-preference | 4 | 1.000 | 1.000 |

Four channels, rank fusion and a cross-encoder beat a plain lexical baseline on
**one** of the six types. Four of the others are already perfect for BM25, so
those rows measure the benchmark and not any system; on the sixth the two agree
question for question and fail on the same four. The win is real where it is
real: multi-session is the type whose evidence is spread across sessions with
no single one matching the question well, and there this is forty points of
recall@5 above lexical retrieval, with no question anywhere in the set where it
scores below BM25.

Three things this is not. It is not evidence about temporal reasoning: the
haystack was loaded as one memory per turn, so no topic ever had a second
version and the validity columns were never populated — the ledger Påmin Memory
is built around was not in the measurement at all, and the retrieval that was
measured performs exactly as a lexical baseline does. It is not comparable to
the retrieval tables in the LongMemEval paper, which are computed on
LongMemEval-M, where each haystack holds roughly ten times as many sessions.
And recall@50 is omitted because the haystack holds about fifty sessions, so it
would be near one by construction.

Ingest ran at a median 112 s a question for about 480 turns, 29,170 turns in
all; search over one loaded haystack had a median of 0.24 s and a p95 of 0.47 s.

**Latency**, what one `pamin search` costs against a warm resident server at
the default `accuracy` profile. Each figure is a whole CLI invocation — fork,
exec, connect to the socket, and back — run serially over forty distinct
queries, reported as the median of them:

| corpus | `--rerank off` | `fast` (default) | `accurate` |
| --- | --- | --- | --- |
| XQuAD-R, 13,014 documents | 77 ms | 251 ms | 1241 ms |
| MIRACL Swahili dev, 131,924 documents | 142 ms | 472 ms | 1675 ms |

Seventeen to nineteen of those milliseconds are the invocation rather than the
search — `pamin --help` against the same workspace costs that much — and it is
measured rather than subtracted, because a caller pays it either way.

**What the rest of an `off` search is spent on is not retrieval either**, and
[ADR 0001](adr/0001-tech-selection.md) divides all four stages: on the XQuAD-R
workspace the cross-encoder is 226 ms of a `fast` search, the query's own
embedding 68, the four channels and fusion and reading the states back 16, and
being a process rather than a socket call 13. That division was taken against a
supplied PostgreSQL rather than the pinned build, so its absolute figures are
not interchangeable with this table's -- and it disagrees with this table by
more than a process, in the direction its queries were longer. What carries
across is the shape: a search is two forward passes and a little bookkeeping,
and the smallest of the four is the one the process costs. A write
is 30.1 ms, most of it the `fsync` a durable append owes — measured over 2,400
memories and published in [cli.md](cli.md), not re-taken in this
sweep.

Measured on 4 vCPU (Intel Xeon @ 2.80 GHz, no SMT), 15 GB RAM, release build,
embeddings on CPU through ONNX Runtime, with every cell's queries disjoint from
every other's so that no figure is a cache hit. `accurate` scores higher on
every corpus measured and costs 3.5 to 4.9 times `fast` on the two corpora in
that table; `fast` is the default on that difference alone, which is a
judgement and not a result.
Four cores is where the embedding model and the reranker contend, so a machine
with cores to spare will not look like this.

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

Read it as a ceiling, not a score. Four cores saturate at eight concurrent
callers and the rest is queueing: at thirty-two the default tier gets *less*
throughput than at eight (4.2 against 4.7) and waits twenty-one times longer —
p50 from 251 ms to 5.4 s. Nothing here scales by adding callers; adding cores
is the lever, and this measurement does not say by how much.

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
[the ADR](adr/0001-tech-selection.md), which is the
source of truth if it and this page ever disagree.
