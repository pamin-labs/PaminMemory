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

**Retrieval quality**, at the shipped defaults:

| corpus | group | nDCG@10 | recall@50 |
| --- | --- | --- | --- |
| MIRACL Swahili dev — 131,924 real passages, 482 queries, 5,092 human judgements | one language throughout | 0.7359 | 0.9494 |
| XQuAD-R — 13,014 sentences in eleven languages, 1,190 queries | query and answer in **different** languages | 0.6480 | 0.8960 |
| XQuAD-R | query and answer in the same language | 0.7495 | 0.9580 |

Both corpora are fetched rather than vendored, and each has a harness in the
repository: `cargo test -p pamin-engine --test crosslingual -- --ignored` for
XQuAD-R and `--test monolingual` for MIRACL.

**The MIRACL row is older than the harness named beside it and older than the
fusion the product now ships**, and both have to be said rather than tidied
away. The fusion weight that produced 0.7359 was halved after a sweep on this
very corpus (below), so that row is not the current default's score either; the
default profile needs about ten hours of index building on four cores before it
can be re-taken, and until it is, the XQuAD-R rows are the only two on this
page taken at the fusion that ships. On the `speed` profile, which can be built
in an hour, the same corpus moved from 0.6826 to 0.6882 fused when the weight
was halved. The same applies to the LOCOMO and LongMemEval figures further down
and to every latency figure on this page: all were taken at the quarter, and
the weight reorders results without changing which ones are retrieved, so the
latency rows are unaffected and the quality rows are not yet re-taken.

Each XQuAD-R row was reproduced identically to four decimals by a second run
before being placed here, which is what this harness does: fixed corpus, fixed
index, fixed model, a greedy pass.

**But reproducible is not the same as significant, and until recently nothing
here could tell the difference.** Every comparison on this page is between two
averages, and an average cannot distinguish every query moving slightly from
one query moving a great deal. That matters at the sizes being reported: the
fusion weight moved on +0.0056 and the reranker is priced at −0.0152, and both
are small enough that a handful of queries decides them. The harnesses now
report per-query wins, losses and a paired bootstrap p beside every mean — see
`crates/pamin-engine/tests/statistics/mod.rs` and the section in
[ADR 0001](adr/0001-tech-selection.md). Until a figure below carries a win/loss
count, read it as a difference of means and nothing stronger.

**The first two re-taken are the two the reranker rests on, and both survive:**

| | mean | wins / losses / ties | p |
| --- | --- | --- | --- |
| XQuAD-R, cross-lingual, `fast` against `off` | +0.0403 | 547 / 206 / 437 | 0.0001 |
| XQuAD-R, same-language, `fast` against `off` | −0.0061 | 3 / 19 / 1,168 | 0.0007 |
| MIRACL Swahili (`speed`), `fast` against fusion alone | −0.0152 | 37 / 57 / 388 | 0.0129 |

And the fusion weight, the eighth against the quarter it replaced, fusion only:

| | mean | wins / losses / ties | p |
| --- | --- | --- | --- |
| Påmin Memory's corpus, cross-lingual (43) | +0.0589 | 21 / 2 / 20 | 0.0004 |
| XQuAD-R, cross-lingual (1,190) | +0.0377 | 698 / 26 / 466 | 0.0001 |
| XQuAD-R, same-language (1,190) | −0.0500 | 16 / 252 / 922 | 0.0001 |
| MIRACL Swahili (482) | +0.0056 | 77 / 72 / 333 | **0.2300** |

**One of those was the figure the change was announced on, and it did not
survive.** The MIRACL row is 77 wins against 72 losses — noise. The default is
supported by the two cross-lingual groups, which are unambiguous, and costs
XQuAD-R's same-language group, which is equally unambiguous. On 482 real
single-language queries no weight between zero and a quarter is
distinguishable. [ADR 0001](adr/0001-tech-selection.md) carries the full
correction.

So the reranker is not a wash in either direction. It is strongly positive
where a query and its answer are in different languages and significantly
negative where they are not — and the second finding is sharper than its mean
suggests. On XQuAD-R it reaches twenty-two same-language queries out of 1,190
and makes nineteen of them worse; on MIRACL, where every query is
same-language, it reaches ninety-four of 482 and loses on fifty-seven.

As for the harness named beside the MIRACL row: this page previously named
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
retrieval stack here scored below a single dense retriever** — the same model,
as an int8 export. Two differences are known and neither is measured: the
published figure is fp32, and MIRACL's training split is in BGE-M3's
fine-tuning data where this runs zero-shot. Neither excuses the gap; they are
where to look for it.

**Since this was written the cause was found, and it was none of those: the
fusion layer was diluting the dense ranking.** On XQuAD-R the embedder alone
and the product run side by side in one harness, so their difference is the
fusion layer's own effect, and at the weight of the day it was −0.0616 on the
cross-lingual group, with the cross-encoder's +0.0375 buying that back rather
than adding to it.

This corpus then arbitrated it. `monolingual.rs`'s model-alone arm is the arm
MIRACL never had; it now runs, and on the `speed` profile the embedding space
alone scores 0.6848 against 0.6826 for the four channels fused at the weight
that used to ship — **the fused answer was worse than one of the things
fused**. Halving the weight puts fusion ahead, 0.6882, and the sweep behind
that number is in [ADR 0001](adr/0001-tech-selection.md): the quarter had never
been compared against anything smaller than itself.

What that bought, where both arms are measured at the shipped default: on
XQuAD-R **the whole stack now outranks the model it is built on**, 0.6480
against the embedder's 0.6335 cross-lingual and 0.7495 against 0.6787
same-language. It did not before.

**And the dilution is still there underneath, now located.** The channel
diagnostic reads every channel's own ranking out of one fused run's trace, so
it costs nothing to take and had never been taken. Fusion alone, at the weight
that ships:

| channel, alone | ours cross | XQuAD-R cross | XQuAD-R same |
| --- | --- | --- | --- |
| `lexical_segmented` | 0.1569 | 0.0366 | **0.7299** |
| `lexical_ngram` | 0.0528 | 0.0106 | — |
| `vector` | **0.8268** | **0.6335** | 0.6787 |
| `graph` | 0.0000 | 0.0000 | 0.0000 |
| all four fused | 0.7910 | 0.6077 | 0.7556 |

And leave-one-out, paired against all four: taking `lexical_segmented` away is
**+0.0310** on this project's cross-lingual group (16 wins, 1 loss, p = 0.0011)
and **+0.0158** on XQuAD-R's (496 wins, 30 losses, p = 0.0001); taking
`lexical_ngram` away is +0.0223 (13 / 1, p = 0.0173) and +0.0110 (333 / 20,
p = 0.0001). The same removal costs **−0.0442** on XQuAD-R's same-language
group (14 wins, 275 losses) and −0.0073 on MIRACL (p = 0.0125).

Fusing four channels ranks below one of them on cross-lingual queries, on both
corpora that have such a group — and on the same-language queries of the same
corpus the lexical channels are worth having outright. Reranking buys the
cross-lingual loss back, which is most of what the cross-encoder is being paid
for. The channels are not weak; one global weight cannot tell the two cases
apart.

Two other things the diagnostic settled. The two lexical channels agree at
Kendall tau-b 0.2816, 0.3188 and 0.2973 on the three corpora, so **they are not
the near-duplicate pair this project described them as** and the single weight
they share has never been swept apart. And the graph channel contributes
**exactly 0.0000 in every group of all three corpora** — two of them have no
relationships to walk, so that much is the corpus, but this project's own
corpus does have edges and also reports zero, and that is unexplained rather
than understood.

### What the fusion function itself is worth

Two mechanisms were built for the finding above and swept offline from one pass
over each corpus. On this project's own corpus, cross-lingual group, 43 queries,
against the reciprocal rank fusion that ships:

| | nDCG@10 | against what ships |
| --- | --- | --- |
| reciprocal rank fusion (ships) | 0.7910 | — |
| **weighted sum of standardised scores** | **0.8192** | **+0.0281, 15 wins / 1 loss, p = 0.0037** |
| the same, times the number of channels that found it (CombMNZ) | 0.7519 | −0.0392, 3 / 21, p = 0.0008 |
| *the vector channel alone, for reference* | *0.8268* | — |

**Adding scores rather than ranks is worth a significant +0.0281 here, and it
closes most of the gap the section above opens** — fusion was 0.0358 behind the
vector channel alone and is now 0.0076 behind it. A rank cannot express a
margin: a candidate its channel put a long way clear of the field and one that
merely came first in a flat list both contribute `weight / (k + 1)`.

**CombMNZ is significantly worse, and that was predicted before the run.** The
systematic comparison of ten combiners ranks it first on all four of its
corpora; the four-channel analysis names multiplying by agreement as the
weakest-link mechanism. Those cannot both hold here, and this corpus says the
second one does — which is the same finding as the table above, arrived at from
the other direction.

Per-channel confidence, on top of rank fusion, is the weaker of the two:
+0.0118 at its best setting (8 wins, 0 losses, p = 0.0381), +0.0070 and +0.0133
either side of it, one of those not significant. The direction is consistent and
the plateau is narrow, which by the same paper's own reading is what overfitting
a development set looks like. The `floor` — how much a channel with no opinion
keeps — moves almost nothing, so what little is there comes from discounting a
channel rather than silencing it.

The other two groups of this corpus sit on their ceilings (1.0000 and 0.9940)
and separate nothing.

**MIRACL disagrees, and the disagreement is the finding.** On 482 real
single-language queries, `speed` profile, every combiner is indistinguishable
from what ships: standardised scores −0.0043 (45 wins, 49 losses, p = 0.1632)
and CombMNZ −0.0059 (p = 0.2675). Adding scores instead of ranks is worth a
great deal on cross-lingual queries and nothing measurable on single-language
ones, which is the same split the channel table above reports — the mechanism
helps exactly where fusion was hurting.

**And per-channel confidence does literally nothing there.** At the two lowest
spreads the change is 0.0000 across all 482 queries, 0 wins and 0 losses; the
largest effect anywhere in the grid is −0.0008. The reason is arithmetic and
was written down before the run: a standardised top score cannot exceed
`sqrt(n - 1)`, which is 7.00 over the fifty candidates each channel proposes,
so any spread of two or less clamps every channel to full weight. What it means
is worse than a badly chosen constant. On this corpus the lexical channels *are*
mildly harmful — removing the n-gram channel is +0.0024 and removing the
segmented one is −0.0073 at p = 0.0125 — and their score distributions
nonetheless look confident. **The measure cannot see the thing it was built to
see here.** It stays off, and that is now a measured decision rather than a
cautious one.

**What MIRACL does say is that splitting the two lexical weights was the right
move**, and it is the one thing on this page that a one-dimensional sweep could
not have found:

| segmented / n-gram | nDCG@10 | against what ships |
| --- | --- | --- |
| 0.125 / 0.125 (ships) | 0.6882 | — |
| **0.250 / 0.000** | **0.6958** | **+0.0076, 79 wins / 51 losses, p = 0.0580** |
| 0.250 / 0.250 | 0.6826 | −0.0056, not significant |
| 0.000 / 0.125 | 0.6809 | −0.0073, p = 0.0125 |

The best row of the whole grid is asymmetric. Moving both channels together —
which is what every sweep before this one did — puts 0.25/0.25 at −0.0056 and
hides that 0.25/0.00 is +0.0076, because the two channels want opposite
directions and the diagonal cancels them. The direction matches the standalone
figures (segmented 0.3113 against n-gram 0.0974) and matches MIRACL's own
authors naming Swahili a language where a BM25 hybrid is the strongest
zero-shot baseline. At p = 0.0580 it is not yet a result, and it is not being
taken as one.

`k` is settled and closed: `k = 5` is +0.0220 at p = 0.0039 on this project's
own corpus and +0.0003 at p = 0.9057 on MIRACL. Two corpora, opposite readings,
so ten stays — which is what the literature predicts for a constant worth one
to three points against a normalisation worth three to eight.

**Nothing has changed default.** Reciprocal rank fusion still ships at the
weights it shipped at. XQuAD-R is the corpus that separates cross-lingual from
same-language queries on the same 1,190 questions, and it has to report before
any of this moves a default.

One thing the first attempt at this sweep is worth recording. Standardised
fusion measured **0.0099 against 0.7910** — 0 wins, 43 losses — because zvec
reports cosine *distance* for a cosine index and the vector channel was being
summed backwards. A number that implausible is a bug rather than a result. What
let it through is worse: the test asserting every channel orders its candidates
by the score it reports wrote every document with the same stub embedding, so
the vector channel reported one constant and ordering by a constant asserts
nothing. Both are fixed, and no figure published before this had ever read a
score — every ranking that ships, and every number in the table above the fix,
reads ranks. The MIRACL comparison above is not re-taken
yet, for the ten hours named earlier, and on `speed` it carries a second
finding worth stating early: once fusion stops diluting, the cross-encoder
*costs* 0.0152 there — 0.6730 with it against 0.6882 without, for 226 ms a
query. Part of what the reranker was worth was undoing the fusion layer's own
damage.

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
