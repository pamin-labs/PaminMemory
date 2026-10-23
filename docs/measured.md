# What Påmin Memory measures about itself

The headline search figures here run through the product's `search_reranked`
entry point or the CLI; the write figures run through its write path. They are not
model-only or index-only benchmarks, because the gap to a whole search is
where this project's numbers have been wrong before. The README carries the
summary; this page carries the numbers and their conditions.

The comparison against other memory systems is a different question and lives
in [benchmarks.md](benchmarks.md), along with what that comparison holds fixed
and how each condition is asserted. The committed evidence behind both pages is
under [benchmarks/results/](../benchmarks/results).

Every search now embeds with BGE-M3 and reranks with bge-reranker-v2-m3: what
the figures below call the `accuracy` profile and the `accurate` tier. The
other profiles (`speed`, `balanced`, `dual_accuracy`) and the `fast` tier were
removed, and rows measured under them are kept as history; they describe no
path a current build runs. An `off` row is fusion alone, what
`Engine::search_fused` returns, which the evaluations still measure as the
comparison for what reranking is worth; `pamin search` no longer offers it.

**Complete XQuAD-R product-path comparison (2026-09-30)**: all 1,190 queries
through `search_reranked`, accuracy profile, accurate reranker, head 30, limit
60, same complete revision-bound 13,014-sentence memory index.

| group | Main CPU int8 nDCG@10 | Main default auto nDCG@10 | Optimized CoreML nDCG@10 | recall@50 (all) |
| --- | ---: | ---: | ---: | ---: |
| cross-language | 0.726808 | 0.727362 | 0.727194 | 0.903193 |
| same-language | 0.868201 | 0.869148 | 0.869446 | 0.964706 |

| Whole-search / process metric | Main default auto | Optimized CoreML | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| p50 | 5.085653 s | 0.483738 s | −4.601915 s | −90.49% |
| p95 | 7.374466 s | 0.814224 s | −6.560242 s | −88.96% |
| Search-call wall total | 6177.679 s | 723.069 s | −5454.610 s | −88.30% |
| Sampled peak process RSS | 4436525056 B | 3068149760 B | −1368375296 B | −30.84% |
| Whole accurate-plus-off harness CPU user + system | 26080.018 s | 402.627 s | −25677.390 s | −98.46% |
| Added three-bucket compiled cache, logical bytes | No equivalent bucket cache | 6836411689 B | +6836411689 B | N/A; zero baseline |
| Matched total persistent model/cache disk | Not measured | Not measured | N/A | N/A |

Both default arms actually selected CoreML; main used legacy NeuralNetwork,
while the optimized reranker used MLProgram + ALL with FP16 encoder / FP32
classifier and static buckets. Embedding stayed CPU int8. ALL permits CPU/GPU/
ANE cooperation; the run did not capture their internal operation placement.
CPU time covers the whole accurate-plus-off harness, including startup; search-only
CPU was not measured. It excludes external CoreML services and device work; RSS includes model
startup/compilation, not isolated steady residency. Compared with the CPU
control instead, optimized CoreML RSS increased 104.44%; baseline matters.

The [default-main evidence](../benchmarks/results/inference/coreml-main-auto-search-2026-09-30.md)
and [CPU-control evidence](../benchmarks/results/inference/coreml-search-full-2026-09-30.md)
archive pinned artifacts, complete rankings, judgments and paired tests. The
[generating wrappers and controllers](../benchmarks/harnesses/product-search-2026-09-30/README.md)
retain the reproduction source and commands.
Recall is identical per query. Default-main nDCG deltas −0.000168 cross /
+0.000298 same have paragraph-cluster, four-metric adjusted p 0.1532 / 1.0.
These are small nonsignificant differences, not proof of numerical equivalence.
Timing is one sequential shared-host process per arm; repeated rotated timings
and other corpora remain pending. The combined-stack result cannot establish
any individual PR's speed contribution.

The [older summary](../benchmarks/results/retrieval/summary-current-xquad.json)
retains its original incomplete index provenance. The fresh revision-bound run
above supersedes its pending-rerun status without retroactively validating it.

**Recorded MIRACL Swahili dev full-head run**, through the
`search_reranked` path on 131,924 passages and 482 judged queries: nDCG@10
**0.8193**, recall@50 **0.9568**. The conditions and replay check are in
[cli.md](cli.md). Its complete code/device provenance has not been archived,
so it is not part of the verified current headline.

The figures below preserve earlier runs for their design history. They were
not rerun under every later change to passage encoding, the answer key or the
full-head rerank. The older MIRACL rows below are not the current full-head
result above.

**Historical retrieval quality**, at the `fast` reranking tier, which was the default when
these were taken:

| corpus | group | nDCG@10 | recall@50 |
| --- | --- | --- | --- |
| MIRACL Swahili dev — 131,924 real passages, 482 queries, 5,092 human judgements | one language throughout | 0.7359 | 0.9494 |
| XQuAD-R — 13,014 sentences in eleven languages, 1,190 queries | query and answer in **different** languages | 0.6480 | 0.8960 |
| XQuAD-R | query and answer in the same language | 0.7495 | 0.9580 |

The default later became `accurate`, chosen on the paired comparison in
[cli.md](cli.md). Those historical runs measured MIRACL at 0.7654 and XQuAD-R
at 0.6597 cross-lingual and 0.7835 same-language. A reranker reorders a
shortlist and leaves its full-depth recall unchanged. The current XQuAD-R
result is in the table above; its gain over these older rows cannot be assigned
to reranking alone because the passage encoding and answer key also changed.

Both corpora are fetched rather than vendored. For XQuAD-R, prepare a fresh
persistent cache, then run only the search test:

```sh
env -u HF_HOME PAMIN_EVAL_HOME=/path/to/eval-home cargo test -p pamin-engine --test crosslingual prepare_pinned_xquad_models -- --exact --ignored
env -u HF_HOME PAMIN_EVAL_HOME=/path/to/eval-home PAMIN_EVAL_DEVICE=cpu cargo test -p pamin-engine --test crosslingual search_reaches_across_languages -- --exact --ignored --nocapture > xquad-run.log 2>&1
python3 benchmarks/results/retrieval/verify_xquad_models.py /path/to/eval-home/models xquad-run.log
```

The harness reads its settings from `PAMIN_EVAL_*` variables, never the
product's `PAMIN_*` names, and refuses a `PAMIN_*` variable it would ignore.
The search also refuses every `PAMIN_EVAL_*` setting except
`PAMIN_EVAL_DEVICE=cpu` and `PAMIN_EVAL_POSTGRES_DIR`, so the baseline runs the
shipped settings whatever the shell exports. The PostgreSQL directory is
allowed because a host without a downloadable PostgreSQL needs it, and no
ranking reads it. Another installation can still move the timings, so a run
that sets it records the path, absolute, among its settings; leave it unset to
use the installation `pamin` downloads. That includes `PAMIN_EVAL_INFERENCE_THREADS`: the thread count
decides how many pairs a CPU reranker pass takes (one where passes run side by
side, up to four at one thread), and so the rankings.

Preparation is needed once per fresh cache. Repeating preparation can fetch
source weights the product has released after making mapped copies. The
verifier reads the actual selected graph paths from this run's log; a prepared
key from the archived machine cannot identify another host's active copy.
Prepared graph hashes remain an exact archived-artifact check: a different
runtime or CPU can require fresh evidence rather than matching those bytes.
MIRACL uses `--test monolingual`.

**The historical MIRACL row predates the current harness and fusion.** The
fusion weight that produced 0.7359 was later halved; that row remains a record
of its original path. The current full-head run is described above, but its
complete code/device provenance has not yet been archived beside XQuAD-R.
The LoCoMo and LongMemEval rows below also retain their original configurations
rather than claiming that every later retrieval change was rerun on them.

The older XQuAD-R tables reported a second run agreeing to four decimals.
The complete CPU and automatic-provider rows above retain their separate
model/runtime provenance; different exports are not interchangeable repeats.

**But reproducible is not the same as significant, and until recently nothing
here could tell the difference.** Every comparison on this page is between two
averages, and an average cannot distinguish every query moving slightly from
one query moving a great deal. That matters at the sizes being reported: the
fusion weight moved on +0.0056 and the reranker is priced at −0.0152, and both
are small enough that a handful of queries decides them. The harnesses now
report per-query wins, losses and a paired randomisation p beside every mean — see
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

The MIRACL table below preserves an earlier run from a program that was not
committed. `monolingual.rs` later measured the current full-head path reported
above; the figures in this table describe the older path.

**What those MIRACL figures are worth, against published results on the same
corpus, the same dev split and the same qrels:**

| MIRACL Swahili dev, 131,924 passages | nDCG@10 | what it is |
| --- | --- | --- |
| Pyserini BM25 baseline | 0.3826 | lexical only |
| Påmin Memory, `--rerank off` | 0.7158 | four channels fused |
| Påmin Memory, `fast` | 0.7359 | fused, then a cross-encoder |
| Påmin Memory, earlier `accurate` path | **0.7654** | fused, then a larger cross-encoder |
| BGE-M3, published | 0.787 | dense retrieval alone |

Read that historical `accurate` row carefully: **a whole retrieval stack then
scored below a single dense retriever** — the same model,
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
| `lexical_ngram` | 0.0528 | 0.0106 | 0.5337 |
| `vector` | **0.8268** | **0.6335** | 0.6787 |
| `graph` | premise absent, since supplied — see below | premise absent | premise absent |
| all four fused | 0.7985 | 0.6114 | 0.7829 |

And leave-one-out, paired against all four: taking `lexical_segmented` away is
**+0.0148** on this project's cross-lingual group (11 wins, 1 loss, p = 0.0148)
and **+0.0128** on XQuAD-R's (397 wins, 25 losses, p = 0.0001); taking
`lexical_ngram` away is +0.0179 (15 / 1, p = 0.0189) and +0.0094 (269 / 16,
p = 0.0001). The same removals cost **−0.0523** and **−0.0361** on XQuAD-R's
same-language group (16 wins to 231 and 9 to 155, both p = 0.0001).

Every figure here is the banded combiner's, which is what ships. The
rank-fusion figures this table used to carry are 0.7910 and 0.6077 fused, and
they understate the trade in both directions.

Fusing four channels ranks below one of them on cross-lingual queries, on both
corpora that have such a group — and on the same-language queries of the same
corpus the lexical channels are worth having outright. Reranking buys the
cross-lingual loss back, which is most of what the cross-encoder is being paid
for. The channels are not weak; one global weight cannot tell the two cases
apart.

Two other things the diagnostic settled. The two lexical channels agree at
Kendall tau-b 0.2816, 0.3188 and 0.2973 on the three corpora, so **they are not
the near-duplicate pair this project described them as**. A later independent
weight grid on XQuAD-R and MuSiQue found no transferable replacement for their
equal eighths (ADR 0001). And the graph channel contributes
**exactly 0.0000 in every group of all three corpora**, which is not a
measurement of the channel: **none of the three corpora has any edges.** The two
external ones name their topics deliberately unlike their own text, which their
harness states outright. The own corpus was recorded here as having edges, and
does not: the only kind the engine derives is `Mentions`, asserted where one
memory's content contains another topic's name as a contiguous token run, and
this corpus names topics `<subject>_<language>` — a two-to-four token run that no
memory's prose contains. Simulated over all 210 memories against all 210 names
it yields zero. The harness now prints the edge census and marks the cell
premise-absent instead of printing a zero that reads like a figure.

**And the premise has since been supplied, which changed a default.** The own
corpus gained a `relational` group: ten pairs of memories whose answering half
is named by a phrase the other half's prose contains, so derivation fires and
the project holds eleven live edges. With something to walk, the graph channel
at its old weight of 1.0 costs **0.2794** nDCG@10 on the cross-lingual group —
forty wins to nothing for removing it, `p = 0.0001` — and 0.0694 on the
monolingual group, where the whole search path then measured 0.9246 against
this repository's own 0.9400 floor and failed it. Against that it earns 0.1673
on the twenty queries built to need it.

The weight was never chosen: every unnamed channel defaults to 1.0 and this one
was never named. It is three tenths now -- 1.0 and 0.5 are significantly worse
once the sweep is priced as one family, and 0.15 cannot be told apart -- and every
floor clears. The sweep is in `pamin_core::fusion`.

**Twenty queries written by the mechanism's authors are not evidence on their
own, so the graph was then measured on a corpus nobody here wrote.** MuSiQue's
answerable dev split (CC BY 4.0) stores each paragraph under its Wikipedia
title, so the edges are the engine's own `Mentions` derivation — 12,840 of them
over 10,785 memories. On its first 1,000 two-hop questions, through
`search_reranked` itself:

| | nDCG@10 | recall@50 |
| --- | --- | --- |
| the search as shipped then, showing ten graph finds | **0.6834** | **0.8435** |
| the same search without the graph | −0.0406 (114 wins, 238 losses, p = 0.0001) | |

Fusion alone gains 0.0159 from the graph there, and the weight of three tenths
chosen on the own corpus is again the best of the sweep — the first external
confirmation of a graph setting. The rest of the gain is the reranker being
shown the graph's ten strongest finds below its head: of 153 supporting titles
the graph alone found, none had reached the head (median fused rank 99, taken
while a support rule, since measured as a no-op on these questions and
removed, also held them down). The
harness is `pamin-engine/tests/multihop.rs`, and it asserts the graph keeps
paying.

**Showing more graph finds buys nothing, and the first attempt to choose them
by score failed its time bound.** Replaying the fused lists found one cut that another setting would
move: 29 supporting titles only the graph found sat below the ten shown, and
twenty brings 19 of them into the reranker's view (19 questions gain, none
lose, p = 0.0001). No fusion change (k, RRF, graph weight) moved that view by
a significant amount. Four arms then went through `search_reranked` at the
`accurate` tier and the `accuracy` profile, on the named-passage project the
evaluation workspace holds now (where ten scores 0.7131), under a rule written
before the runs: ship the simplest arm that is significantly worse nowhere,
better somewhere, costs at most 1.5 times the search time, and, where accuracy
and speed disagree, wins on the effect with the larger standardized size.
Every find is in the table for scale only; it was not a candidate.

| arm | MuSiQue nDCG@10, 1,000 questions | own corpus, 157 | search time, MuSiQue | search time, own |
| --- | --- | --- | --- | --- |
| **ten** | **0.7131** | **unchanged** | **1** | **1** |
| twenty | +0.0010 (9 better, 25 worse, p = 0.41) | identical on every query | 1.41 | 0.99 (n.s.) |
| a graph score of at least τ, at most thirty | −0.0004 (9 better, 35 worse, p = 0.74) at τ = 0 | −0.0002 (1 worse) at τ = 0.5 | 1.78 | 0.82 |
| every graph find | −0.0002 (11 better, 41 worse, p = 0.90) | identical on every query | 2.26 | 0.99 (n.s.) |

Times are the geometric mean of each question's ratio to ten, from searches
with the reranker's and the embedder's caches bypassed and the arms taken in
rotated order: 100 MuSiQue questions (every tenth) and all 157 own-corpus
queries, on four cores that other work was sharing. XQuAD-R has no edges, so
every arm is ten there. τ was chosen leave-one-corpus-out: on the own corpus
every τ scores the same, down to showing no graph find at all, so MuSiQue got
τ = 0, which is thirty; MuSiQue chose τ = 0.5 for the own corpus. So the
threshold is faster on the own corpus and fails the time bound on MuSiQue,
and twenty wins nothing. Neither shipped.

Two findings came out of it. The recall the reranker is shown does not
predict what it returns: twenty adds 19 relevant titles to its view and
+0.0010 to its top ten. And the own corpus pays for graph finds it does not
use: ten shows about four of them a query, none of them ever changes a top
ten there, and not showing them is 18% of a search. On 100 MuSiQue questions,
rebuilt from the reranker's own scores, τ = 0.5 (about nine finds a question)
scored 0.6920 against ten's 0.6902, and a five-fold split of those questions
picked it in every fold. That is a hypothesis for a threshold measured on its
own, not a result: it was not the arm this comparison chose.

MuSiQue's twenty and thirty come from a run that took ten first and then
reused the pairs it had scored, so their shared pairs were scored in other
batches; the int8 export makes a score depend on its neighbours. The 100
questions searched fresh match that run's twenty and thirty on 97 each. The
arms change a count and a threshold that the tree does not expose, so they
were measured on a scratch build and cannot be re-run from the repository.

**A graph score of at least 0.5, at most thirty, ships in place of the fixed
ten, on a held-out test.** The hypothesis above came from 100 questions it was
then fitted on, so those were set aside (every tenth, from the first), and a
second rule was written before anything ran: compare ten with τ = 0.5 capped at
thirty on 600 other MuSiQue questions (the six in every ten after each set-aside
one), the own corpus and XQuAD-R; ship the threshold only if it is
significantly worse on no corpus and no group (sign-flip, p < 0.05), at most
1.1 times ten's search time on MuSiQue, and significantly faster or better on
MuSiQue or the own corpus (p < 0.0125, four tests). Searches went through
`search_reranked` at the `accurate` tier and the `accuracy` profile, both arms
of a question in one process in alternating order, with the reranker's and the
embedder's caches bypassed and the shown set asserted against an independent
recomputation from the trace on every search.

| | MuSiQue, 600 held-out questions | own corpus, 157 | XQuAD-R, 119 |
| --- | --- | --- | --- |
| nDCG@10, ten | 0.7138 | 0.9103 | |
| threshold − ten | +0.0017 (16 better, 5 worse, p = 0.057) | −0.00005 (1 worse, p = 1.0) | identical lists |
| search time, threshold / ten | **0.87** (p = 0.0001) | **0.81** (p = 0.0001) | 1.00 (n.s.) |
| graph finds shown, ten → threshold | 10 → 8.4 | 3.95 → 0.01 | 0 → 0 |
| reranker pairs a search | 28.4 → 26.9 | 24.3 → 20.3 | 24.2 → 24.2 |

No group is worse; the own corpus's one loss is a cross-lingual query that
drops by 0.0078. Every clause holds, so the threshold ships
(`GRAPH_SHOWN_FROM` and `GRAPH_SHOWN_AT_MOST` in `pamin-engine`). Times are
geometric means of per-question ratios on four cores that other evaluations
were sharing. The 0.6834 and 0.7131 above were measured with ten.

What it trades, read after the verdict and not part of it: on MuSiQue the
threshold shows more than ten finds on 142 questions, and those searches take
1.60 times as long, with 2 better and 3 worse; on the 458 where it shows ten
or fewer they take 0.72 times as long, with 14 better and 2 worse. So the
saving and the gain both come from dropping weak finds, and the cap of thirty
costs time it was not seen to earn. A cap of ten under the same threshold is
the next hypothesis. These 600 questions shaped it, so it was tested on
questions of its own, below.

**A cap of ten passed a second held-out test and does not ship.** With
more than ten finds at the threshold the strongest ten all reach it, so a cap
of ten shows exactly the old fixed ten there and exactly what thirty shows
everywhere else: the two differ only on the questions where thirty shows more
than ten. The rule was written before anything ran: compare the two caps on
the 300 MuSiQue questions neither earlier test had seen (the last three in
every ten, asserted on every search) and the own corpus as a replication;
ship ten only if it is significantly worse on neither corpus and no
own-corpus group (sign-flip, p < 0.05) and significantly faster on
MuSiQue (geometric mean of per-question ratios, p < 0.025). Same method as
above: `search_reranked`, `accurate` tier, `accuracy` profile, both arms of a
question in one process in alternating order, both caches bypassed, the shown
set checked against an independent recomputation on every search.

| | MuSiQue, 300 held-out questions | own corpus, 157 |
| --- | --- | --- |
| nDCG@10, thirty | 0.7186 | 0.9102 |
| ten − thirty | −0.0031 (2 better, 5 worse, p = 0.17) | identical on every query |
| search time, ten / thirty | **0.91** (p = 0.0001) | 1.01 (p = 0.49) |
| graph finds shown, thirty → ten | 8.32 → 5.05 | 0.01 → 0.01 |
| reranker pairs a search | 26.8 → 23.6 | 20.3 → 20.3 |

Both clauses hold, and thirty still shipped then (`GRAPH_SHOWN_AT_MOST`; see below for four): accuracy
outranks latency in this project, and the accuracy estimate below points the
wrong way on exactly the questions the cap changes. The caps differed on
72 of the 300 questions, as the 142 of 600 above predicted; there ten takes
0.66 of thirty's time and 43 pairs become 29, and on the other 228 the ratio
is 1.00. The nDCG@10 difference is not significant, but it points the wrong
way, and read after the verdict it sits where thirty showed the most: the
five losses are on questions where thirty showed 16 to 30 finds, the two
gains where it showed 11 and 13. This harness ran from the build the
evaluation workspace was made with, which predates changes elsewhere in the
stack; the two arms differ only in `rerankable`'s cap.

**A cap of four, ranked by strength shared across a hub, ships.** A hub one
hop from a seed hands the graph channel dozens of neighbours with similar
scores, and ranked by score alone they fill any cap with one topic's
surroundings. `rerankable` now divides each eligible find's graph score by
the square root of how many eligible finds came through the same `via`
topic, and shows the strongest four. Replayed on recorded searches first,
against the cap of thirty: MuSiQue 2-hop (600) −0.0005 (7 better, 7 worse;
p_fw = 0.79 in a four-arm replay, 1.0 in an eight-arm one), every LoCoMo category level or better (multi-hop +0.0055,
p_fw = 0.066), XQuAD identical (no edges). Then at the product entry point
(`pamin serve` and one `pamin search --limit 10` process a query, CPU
execution provider, four Xeon cores at 2.8 GHz with AVX-512 VNNI, ONNX
Runtime 1.28, `gpahal/bge-m3-onnx-int8` @ `2b34e84` and
`onnx-community/bge-reranker-v2-m3-ONNX` (`onnx/model_int8.onnx`) @
`6f5ff65`, LoCoMo's five
conversations with their adjacency links, 120 timed queries after 20 warm
ones, four rotated runs an arm, no run shared), main `010c4d5` against the
branch at `fe30d05`:

| metric | cap 30 | cap 4, fan-out ranked | abs diff | % change | ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries | 0.7189 | 0.7173 | −0.0017 | −0.2% | 0.998 [0.980, 1.016] |
| search p50 ms | 4,426 | 2,442 | −1,985 | −44.8% | 0.552 [0.531, 0.576] |
| search p95 ms | 5,431 | 3,070 | −2,361 | −43.5% | 0.565 [0.533, 0.604] |
| server CPU user s a query | 16.43 | 8.86 | −7.57 | −46.1% | 0.539 [0.508, 0.565] |
| search peak RSS MB (server process) | 1,323 | 1,329 | +6 | +0.5% | 1.005 [0.991, 1.014] |
| ingest peak RSS MB | 1,101 | 1,132 | +31 | +2.8% | 1.028 [0.988, 1.072] |
| index MB, 5 projects | 54.6 | 55.6 | +1.1 | +1.9% | 1.019 [0.999, 1.035] |
| graph-only finds shown a query | 27.7 | 4.0 | −23.7 | −85.6% | N/A (diagnostic) |

The nDCG@10 difference is not significant (paired bootstrap over queries
[−0.0147, +0.0114]; 7 better, 5 worse, 108 equal; queries of one
conversation share a graph, so that interval is optimistic): multi-hop
+0.0152, temporal level, single-hop −0.0068, open-domain −0.0127 on 7
queries. Search memory, ingest and the index do not move beyond noise; the
peak-RSS rows are the server process only, with no accelerator services,
and the index row excludes postgres and the model caches. Times are wall
time of the search process, warm, prepared copies in the page cache.

### What the fusion function itself is worth

Two mechanisms were built for the finding above and swept offline from one pass
over each corpus. On this project's own corpus, cross-lingual group, 43 queries,
against the reciprocal rank fusion that shipped when this was measured — the
banded combiner ships now, and these figures are not retaken against it. Both
score combiners below and the confidence rule have since been removed from the
code; [ADR 0001](adr/0001-tech-selection.md) records each under *Fusion designs
measured and removed*:

| | nDCG@10 | against rank fusion |
| --- | --- | --- |
| reciprocal rank fusion | 0.7910 | — |
| **weighted sum of standardised scores** | **0.8192** | **+0.0281, 15 wins / 1 loss, p = 0.0037** |
| the same, times the number of channels that found it (CombMNZ) | 0.7519 | −0.0392, 3 / 21, p = 0.0008 |
| *the vector channel alone, for reference* | *0.8268* | — |

**Adding scores rather than ranks is worth a significant +0.0281 here on
nDCG@10** — fusion was 0.0358 behind the vector channel alone and is 0.0076
behind it that way. A rank cannot express a margin: a candidate its channel put
a long way clear of the field and one that merely came first in a flat list both
contribute `weight / (k + 1)`.

**It still does not ship, and the reason is recall rather than nDCG.** Read
across all three corpora the standardised sum is better on nDCG@10 in three of
four groups, significantly, and worse in none — the first setting in this
project's history that is not a trade. It was made the default on that reading
and the accuracy gates rejected it: **XQuAD-R's cross-lingual `recall@50` fell
from 0.8960 to 0.7765, through a floor of 0.8000.**

The mechanism is the sign. Every reciprocal-rank contribution is positive, so a
candidate one channel ranked fiftieth still helps it stay in the list. A
standardised score is centred, so a candidate below its channel's own mean
contributes a *negative* number — and on a cross-lingual query, where a lexical
channel scores 0.0366 on its own, that channel's confident top hit at `+2`
outranks a genuine deep hit from the vector channel at `-1`. The top ten
improves because strong vector hits dominate it; the tail fills with lexical
noise and the relevant sentences that sat between ranks ten and fifty fall past
fifty.

So it is a precision-for-recall trade, and a search that hands its results to a
reranker cannot afford one: nothing recovers a memory that was never returned.
What would make it shippable is a floor under each contribution, or normalising
to `[0, 1]` instead of centring — neither of which is what the published work
measured, so neither is implemented.

**And the grid that nearly let it through printed nDCG and nothing else.** The
gates assert both metrics; forty variants were being priced on one. It prints
recall@50 now, and with both columns the answer is cleaner than it was with
one. On XQuAD-R:

| setting | cross-lingual nDCG / recall | same-language nDCG / recall |
| --- | --- | --- |
| banded — **ships** | 0.6114 / **0.8962** | 0.7829 / **0.9571** |
| rank fusion | 0.6077 / 0.8960 | 0.7556 / 0.9580 |
| + confidence, spread 5 | 0.6018 / 0.8961 | **0.8041** / 0.9588 |
| both lexical weights at zero | **0.6335** / 0.8966 | 0.6787 / 0.9529 |
| standardised sum | **0.6211** / **0.7765** | 0.7733 / 0.9403 |
| + CombMNZ | 0.5727 / **0.7764** | **0.8257** / 0.9403 |

**The recall cost belongs to the combiner and to nothing else.** Every variant
built on rank fusion — any pair of lexical weights, any confidence setting —
holds recall at 0.896 and 0.958. Every standardised variant sits at 0.7765 and
0.9403 regardless of what else is set, and confidence does not rescue it
(0.7772 at its best). That is what a structural consequence looks like as
opposed to a badly chosen constant: centring the scores is what costs the
tail, and −0.12 cross-lingual against −0.018 same-language is the same
mechanism seen where the weak channel is garbage and where it is good.

### What that mechanism, once stated, made shippable

Reading the recall column as a property of the *combiner* rather than of the
settings says what to fix. Rank fusion's contributions span a narrow range —
over fifty candidates at `k = 10`, `1/11` down to `1/60`, a factor of 5.45 —
and that narrowness is what makes the channel weights mean anything: a lexical
channel's top hit at an eighth weight scores 0.0114 while the vector channel's
fiftieth scores 0.0167, so the good channel's marginal candidate still wins and
the tail stays full of things a reranker could recover. **Any normaliser onto
`[0, 1]` spans a factor of infinity inside one channel**, so position beats
weight and a worthless channel's confident hit displaces a good channel's deep
one. Centring adds a sign on top of that; min-max alone would do the same
thing.

So: min-max each channel's scores and map them onto the band rank fusion would
have spanned over the same candidates, `[(k + 1) / (k + n), 1]`. The band is
derived, not tuned, and the only thing that changes is whether a channel's
candidates are ordered inside it by score or by rank — which is the question
this whole line of work was trying to ask and could not ask cleanly while the
range was moving too.

| group | nDCG@10 | p | recall@50 |
| --- | --- | --- | --- |
| XQuAD-R cross-lingual | **+0.0037** | 0.0003 | 0.8960 → 0.8962 |
| XQuAD-R same-language | **+0.0273** | 0.0001 | 0.9580 → 0.9571 |
| MIRACL Swahili | −0.0003 | 0.9210 | 0.9314 → 0.9309 |
| this project, cross-lingual | +0.0075 | 0.3731 | 0.9605 → 0.9605 |

**Two groups significantly better, none significantly worse, recall moving by
at most 0.0009 — and this is what ships now.** Every accuracy gate passes and
two published figures improve: on the shipped path XQuAD-R goes from 0.6480 to
**0.6511** cross-lingual and from 0.7495 to **0.7769** same-language.

The same-language gain is the part worth noticing. Every weight this project
ever changed took something from that group to pay for the cross-lingual one —
the section above is a table of exactly that. This is the first change that
improves it, and it does so by letting the channel that is genuinely good there
say *how much* better its top candidates are rather than only that they came
first.

The rest of the grid still ships nothing, and that is now a complete statement:
confidence buys +0.0324 same-language for −0.0174 cross-lingual, zeroing the
lexical pair buys +0.0258 cross-lingual for −0.0769 same-language, and the
standardised sum buys nDCG everywhere for a recall collapse.

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

**MIRACL and XQuAD-R fill the picture in.** On 482 real single-language
queries, `speed` profile, every combiner is indistinguishable from what ships:
standardised scores −0.0043 (45 wins, 49 losses, p = 0.1632) and CombMNZ
−0.0059 (p = 0.2675). Adding scores instead of ranks is worth a great deal on
cross-lingual queries and nothing measurable on single-language ones, which is
the same split the channel table above reports — the mechanism helps exactly
where fusion was hurting.

**And CombMNZ is the reverse of itself between the two XQuAD-R groups**, which
is the cleanest statement of the whole problem this page describes. Over the
same 1,190 questions it is −0.0350 cross-lingual (122 wins, 601 losses,
p = 0.0001) and **+0.0700 same-language** (265 / 26, p = 0.0001) — the largest
single gain measured anywhere here. CombMNZ multiplies by how many channels
found a candidate, so it is a bet that agreement is evidence. Same-language,
the channels that agree are independently good — segmented BM25 alone scores
0.7299 against the vector channel's 0.6787 — and the bet pays. Cross-lingual,
the lexical channels score 0.0366 and 0.0106 alone, so their agreement with the
vector channel is coincidence and the bet fails by the same mechanism that
makes fusion worse than one of its channels there. **Agreement is evidence only
when the channels agreeing are independently right**, which is why the
systematic comparison that ranks CombMNZ first of ten combiners is not wrong
about its own corpora and not transferable to these.

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
see here.** It never shipped and has since been removed, and that is a
measured decision rather than a cautious one.

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

**The n-gram field keeps two-character grams. Three were measured and lose
the cross-lingual group.** A scratch audit of the n-gram channel alone had
three-character grams far ahead of `zvec`'s default of two: +0.0694 nDCG@10 on
XQuAD-R and +0.1875 on MuSiQue. After fusion, almost none of that was left:
+0.0016 (p = 0.36) and +0.0047 (p = 0.007). So two arms went through
`search_reranked` at the `accurate` tier and the `accuracy` profile: three
alone (`{"ngram_min":3,"ngram_max":3}`), and two and three together
(`{"ngram_min":2,"ngram_max":3}`). The second was the hedge for two-character
Chinese and Thai words. The rule was written before the runs. It shipped an arm
only if all of these held:

- no aggregate cell significantly worse;
- none of Chinese or Thai, same-language or cross-lingual, significantly worse;
- no other XQuAD-R language significantly worse after Holm correction;
- one aggregate cell better at p < 0.025;
- at most 1.10 times the median search time, 1.40 times the index on disk, and
  1.50 times the rebuild.

Neither arm met it:

| arm | XQuAD-R same-language, 1,190 | XQuAD-R cross-lingual, 1,190 | MuSiQue, 300 | own corpus, 157 | index on disk | rebuild |
| --- | --- | --- | --- | --- | --- | --- |
| **two (ships)** | **0.8118** | **0.6703** | **0.7161** | **unchanged** | **1** | **1** |
| three | +0.0034 (46 better, 26 worse, p = 0.052) | **−0.0055** (151 better, 266 worse, p = 0.0001) | −0.0001 (33 / 35, p = 0.98) | no group moves significantly | 1.07–1.30 | 1.16–1.25 |
| two and three | +0.0016 (32 / 16, p = 0.22) | **−0.0035** (114 / 194, p = 0.0001) | −0.0013 (27 / 30, p = 0.74) | no group moves significantly | 1.16–1.67 | 1.60–1.67 |

Three-character grams have a lower cross-lingual mean in all eleven query
languages, by under 0.001 in Chinese and Vietnamese, and two of those losses
fail the rule on their own. Thai loses −0.0049 (8 better,
17 worse, p = 0.018), and English loses −0.0126 (p = 0.036 after Holm). The
risk the rule named in advance did not appear: Chinese same-language moves by
−0.0002 and Thai same-language by +0.0031, neither significant. Recall@50 of
the final list does not move on XQuAD-R (−0.0001), so the loss is in the order
of the top ten, not in what is found. The MuSiQue gain the fused list showed
does not survive the reranker, although recall@50 there rises by 0.0050 in both
arms. Two and three together is not a cheap hedge. It stores both gram sizes,
so it is the largest index and the slowest rebuild of the three arms, and it
still loses the cross-lingual group.

Median search time is within 2.3% of what ships on every corpus, faster on
some and slower on others. It was measured with the reranker's and the embedder's caches
bypassed and the arms taken in rotated order, on four cores that two other
evaluation runs were sharing. Disk is the whole project index directory after a
rebuild and an optimise. Two gram sizes make it 1.50 times as large on XQuAD-R
and 1.67 times on MuSiQue, which is 110 MB against 166 MB and 119 MB against
198 MB. Rebuild is the reindex wall clock with every vector lent from the index
it replaces, so it is the index's own work: 21.9 s, 27.4 s and 35.0 s on
XQuAD-R. The own corpus rebuilds in under half a second, too little to give a
ratio. MuSiQue is its first 300 questions, not 1,000. At about twenty seconds a
question on the shared machine, the full set would have taken five and a half
hours, and XQuAD-R had already decided the verdict. The addendum saying so
was written before those rows were analysed. The
arms were built on the tree from before the graph threshold, when the reranker
was shown a fixed ten graph finds. XQuAD-R has no edges, and each comparison is
paired on one base. The
gram size is a scratch-build switch, not something the tree exposes, so the
arms cannot be re-run from the repository. With no arm eligible, the rule did
not re-sweep the n-gram weight, and it stays at an eighth.

**A learned fusion head was fitted and does not generalise.** The `FEATURES`
arm (`pamin-engine/tests/features`, since replaced by the `REPLAY_OUT` record
in `pamin-engine/tests/reranking/record.rs`, which holds the same per-candidate
channel ranks and scores and, for the superset it scores, the model's) dumped every fused candidate with each
channel's rank and score, and the shipped combiner rebuilt from the dump alone
matches the engine bit for bit on all 3,537 question-groups of the own corpus,
XQuAD-R and MuSiQue. The rule, written before any table was read: a head ships
only if, trained on two corpora, it beats the shipped combiner on recall@20 --
what the reranker is handed -- in the third, every time, and loses no group
significantly. Held out by corpus, recall@20 against the shipped combiner:

| head | own | XQuAD-R | MuSiQue |
| --- | --- | --- | --- |
| per-group weights, fitted in sample (an upper bound) | +0.0172 | +0.0117 | +0.0050 |
| the shipped weights re-tuned on the other two | +0.0115 | −0.0013 | ±0 |
| logistic regression | +0.0013 | −0.0131 | **−0.2375** |
| pairwise linear | +0.0096 | −0.0037 | **−0.1180** |
| gradient-boosted LambdaRank, monotone | +0.0054 | −0.0070 | +0.0200 |

None passes. Three reasons, each visible in the data: only MuSiQue has a graph
worth weighting, so any head trained without it gets the graph wrong; XQuAD-R's
two groups are the same queries with opposite optima for the lexical weight, so
any head can only pick a point on that trade, and a script feature buys
cross-lingual +0.028 for same-language −0.050; and even the in-sample oracle is
worth under two points of recall@20. A per-query gate choosing the lexical
weight from query features -- the shape of DAT and MoR -- turns lexical off on
3,506 of 3,537 queries and loses on both external corpora. Published learned
fusion (Bruch et al., TOIS 2023; DAT and MoR, 2025) reports gains inside one
benchmark mix; the held-out-corpus test is the one that fails here.

**The one lead the fit left was tested on the shipped path and refuted.**
Offline, the graph at half weight without its support rule moved MuSiQue's
recall@20 by +0.0175. Through `search_reranked_with`, every question paired
against what ships (the `GRAPH_VARIANTS` arm this was measured with is
deleted):

| | own, cross-lingual | own, relational | MuSiQue, 1,000 two-hop |
| --- | --- | --- | --- |
| graph 0.50, no support rule | −0.0060, 0W/6L, p = 0.035 | +0.0201, n.s. | −0.0026, 86W/85L, n.s. |
| graph 0.50 | −0.0030, 0W/6L, p = 0.035 | +0.0201, n.s. | −0.0023, 85W/83L, n.s. |
| graph 0.30, no support rule | ±0, every query tied | ±0 | ±0, every query tied |

The recall the fit found in the fused list does not survive the reranker, and
the last row says something else: at the shipped weight the support rule does
nothing on 1,157 questions, including MuSiQue's 12,840-edge graph -- the dense
case it was kept for.

**The combiner has changed default since this was written.** XQuAD-R, the
corpus that separates cross-lingual from same-language queries on the same
1,190 questions, had to report before any of this moved a default; once it
had, the banded combiner replaced reciprocal rank fusion — see *What that
mechanism, once stated, made shippable* above.

One thing the first attempt at this sweep is worth recording. Standardised
fusion measured **0.0099 against 0.7910** — 0 wins, 43 losses — because zvec
reports cosine *distance* for a cosine index and the vector channel was being
summed backwards. A number that implausible is a bug rather than a result. What
let it through is worse: the test asserting every channel orders its candidates
by the score it reports wrote every document with the same stub embedding, so
the vector channel reported one constant and ordering by a constant asserts
nothing. Both are fixed, and no figure published before this had ever read a
score — every ranking that shipped then, and every number in the table above
the fix, read ranks. The MIRACL comparison above is not re-taken
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

**Historical abstention experiment: invalid for product calibration.** No
abstention verdict ships. The historical isotonic map was fitted on batched
MuSiQue top-hit scores (484 questions, reported held-out ECE 0.0590), but the
LoCoMo/LongMemEval evaluation reused individual pair logits or generated
singleton scores; 1,789 of 1,986 LoCoMo top hits used the singleton path. INT8
neighbours and padding affect the score, so these are incomparable model inputs.
The ECE, AUROC, `weak` rates, paired outcomes and decision claims are withdrawn
as product-calibration evidence. The following table preserves historical
numbers only and must not be used to accept or reject an abstention strategy:

| LoCoMo, evidence in the top ten and not `weak` | n | never abstains | with the verdict | p |
| --- | --- | --- | --- | --- |
| multi-hop | 282 | 0.791 | 0.592 | 3e-17 |
| temporal | 321 | 0.826 | 0.670 | 2e-15 |
| open-domain | 96 | 0.521 | 0.292 | 5e-7 |
| single-hop | 841 | 0.810 | 0.718 | 1e-23 |
| adversarial, `weak` | 446 | 0 | 0.314 | 1e-42 |

Other retained historical observations were LongMemEval-S recall_any@10 0.983
versus 0.627 (p = 1e-6), LoCoMo AUROC 0.606, and LoCoMo ECE 0.3100. They do
not establish calibration transfer failure or valid product effects. Current
product calibration metrics are N/A. The [ADR validity correction](adr/0001-tech-selection.md#the-transfer-test-taken-as-an-abstention-decision-invalid-for-product-calibration)
records the incomparable scoring procedure, retained observations and complete
product-batch requirements. No corrected experiment was run; valid product-calibration metrics remain N/A.

**Latency**, what one `pamin search` costs against a warm resident server at
the default `accuracy` profile. Each figure is a whole CLI invocation — fork,
exec, connect to the socket, and back — run serially over forty distinct
queries, reported as the median of them:

| corpus | `--rerank off` | `fast` | `accurate` (default) |
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
that table; `fast` was the default on that difference alone, which was a
judgement and not a result, and the default is now `accurate` on the
project's own ordering of accuracy before latency.
Four cores is where the embedding model and the reranker contend, so a machine
with cores to spare will not look like this.

**The `accurate` pass at the depth that ships, and batched by tokens.** The
table above was taken when a tier reranked twenty candidates. It now reranks
thirty, and the pairs reach the model in passes of at most 512 padded tokens
and four pairs, grouped by their real length in tokens, where they went in
chunks of eight sorted by characters. Through `Engine::search_reranked` at the
`accuracy` profile, every query of each corpus through both batchings in one
process, in rotated order, the median search:

| corpus | queries reranked | chunks of eight | 512 tokens, four pairs | paired ratio, median | faster on |
| --- | --- | --- | --- | --- | --- |
| XQuAD-R, 13,014 sentences | 1,187 | 2,006 ms | **1,633 ms** | 0.833 | 1,053 |
| MIRACL Swahili dev | 482 | 2,324 ms | **1,758 ms** | 0.800 | 461 |
| MuSiQue, 1,000 2-hop questions | 997 | 6,112 ms | **4,949 ms** | 0.810 | 944 |

These are not quiet-machine figures, and they are from a different machine
from the table above: 4 vCPU (Intel Xeon @ 2.10 GHz, AMX and AVX-512 VNNI, no
SMT), 15 GB RAM, release build, shared with two or three other measurements —
the load average's median over the three runs was 8.1, 6.2 and 6.4. Load
inflates both columns, so the milliseconds say what a search at thirty costs
on a busy four-core machine and the paired ratio is what carries. Rerank
nDCG@10 moved significantly in no group: XQuAD-R cross-lingual −0.0004,
same-language −0.0002, MIRACL −0.0011, MuSiQue +0.0005, every p above 0.4.
The rule the batching was chosen by, written down before anything was timed,
and the pilot of seven candidates are at `BATCH_TOKENS` in
`crates/pamin-index/src/reranking.rs`.

Both columns run a call's passes one after another, each pass with every core
and up to four pairs. On a multi-core CPU the passes of one rerank now run
side by side, one thread and one pair each, so these milliseconds are the
predecessor's and describe no search that ships; the paired ratio between the
two batchings still stands. The entry-point figures for what ships are below.

**On the CPU, one pair a pass.** Once the passes of one rerank ran side by
side, one thread each, on one shared session, grouping pairs into a pass only
padded them. Against four pairs on the same build, `pamin search` through a
resident server (4 vCPU Xeon, three rotated runs of 120 warm queries an arm)
took a p50 of 0.834 [0.757, 0.931] and a p95 of 0.850 [0.715, 0.997] as long,
and through `Engine::search_reranked` over copies of one index no significant
nDCG@10 change was detected in any group (Holm-corrected p = 1.00 in each):
XQuAD-R cross-lingual −0.0001 [−0.0020, +0.0016], same-language −0.0012
[−0.0037, +0.0011], single-target −0.0013 [−0.0052, +0.0026] (1,190 queries
each), MuSiQue 2-hop +0.0040 [−0.0029, +0.0113] (its first 200 questions). The table is at `CPU_BATCH` in
`crates/pamin-index/src/reranking.rs`. Accelerators, and a CPU with an
inference budget of one thread, where passes cannot overlap, still group up to
four pairs within 512 padded tokens. The write-ups, raw rows, per-query dumps
and harnesses are retained for maintainers in the private submodule (see
[internal-docs.md](internal-docs.md)): the timing at
`internal-docs/pamin-memory/research/2026-10-16/q42-shared-session-b1-and-review-status.md`
and `internal-docs/pamin-memory/research/2026-10-16/raw/q42/`, the accuracy at
`internal-docs/pamin-memory/research/2026-10-16/q44-b1-entry-point-accuracy.md`
and `internal-docs/pamin-memory/research/2026-10-16/raw/q44/`.

**The shared session, and one pair a pass, step by step at the entry point.**
The paragraph above gives the second step's ratios; these are both steps with
their absolute figures. Each step is measured against its immediate
predecessor on the same machine, in the same rotated runs:

- **Builds.** Before the shared session: `ddf0800`, where each side-by-side
  pass had a session of its own, with up to four pairs a pass. Shared session:
  `710ceec`, one shared single-threaded session per model, still four pairs.
  One pair a pass: `710ceec` with its rerank batch set to one pair, which is
  what `b972ead` made the default on a multi-core CPU; `b972ead` itself was not
  timed.
- **Host.** 4 vCPU Xeon @ 2.8 GHz (AVX-512 VNNI, no AMX). ONNX Runtime 1.28,
  statically linked, CPU execution provider, four inference threads.
- **Models.** `gpahal/bge-m3-onnx-int8` @ `2b34e84d` and the int8
  bge-reranker-v2-m3 @ `6f5ff652`, both loaded from their mapped copies.
- **Workload.** `pamin import` of the same 3,230 texts in a fixed order, then
  20 warm-up and 120 timed `pamin search` queries, against a fresh home and
  server for each phase.
- **Runs.** Twelve counted runs, three an arm, in rotated order (A S SB AB,
  SB AB A S, S A AB SB, where A is before the shared session, S the shared
  session, SB one pair a pass, and AB one pair a pass without the shared
  session, a fourth arm not reported here). Each run asserted its binary,
  environment, execution provider, session counts, and that no model or plan
  changed; two runs that failed one of those, and one with load from another
  process, were left out.
- **Scope.** Memory is the `pamin serve` process alone: no PostgreSQL, no
  accelerator service. Disk is the project's index directory and the model
  cache the server loaded. CIs are 95% bootstrap over runs.
- **Known exposure.** This host has the CUDA provider built in but cannot
  register it, which a resident server retries every five minutes by
  reloading its models (see [cli.md](cli.md#accelerator-plan-recovery)). Every
  search phase had one such reload, in every arm, and the medians excluding
  the queries that overlapped one moved the ratios by at most 0.004.

The shared session, before `ddf0800` and after `710ceec`:

| metric | before | after | absolute difference | % change | after/before, or difference [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| nDCG@10 | N/A: not measured | N/A | N/A | N/A | N/A |
| top-5 results, 120 queries | reference | identical order on 120 of 120 | N/A: identity, not a difference | N/A | N/A |
| search p50 ms (pooled) | 3,072.7 | 3,206.7 | +134.1 | +4.4% | 1.044 [0.937, 1.157] |
| search p95 ms (pooled) | 5,135.0 | 5,128.4 | −6.6 | −0.1% | 0.999 [0.880, 1.191] |
| search server CPU user s, 120 queries | 1,295.0 | 1,328.1 | +33.0 | +2.6% | 1.026 [0.951, 1.114] |
| search server CPU system s, 120 queries | 38.3 | 42.5 | +4.3 | +11.1% | 1.111 [0.983, 1.253] |
| search server peak RSS MB (VmHWM) | 5,656 | 3,021 | −2,636 | −46.6% | 0.534 [0.509, 0.556] |
| search server Pss_Anon MB after 120 queries | 1,827 | 1,488 | −338 | −18.5% | 0.815 [0.627, 0.991] |
| ingest wall s (import and drain) | 294.7 | 295.8 | +1.2 | +0.4% | 1.004 [0.941, 1.072] |
| ingest server peak RSS MB (VmHWM) | 3,795 | 2,158 | −1,637 | −43.1% | 0.569 [0.480, 0.647] |
| index directory MB allocated | 27.4 | 28.9 | +1.5 | +5.6% | 1.056 [0.946, 1.195] |
| model cache in use MB | 2,788 | 2,788 | 0 | 0.0% | N/A: same model files |

**For the shared session, no significant latency change was detected; its
measured gain is memory.** The p50 ratio was 1.044 [0.937, 1.157] and the p95
0.999 [0.880, 1.191], both including 1.0. A component measurement of the same change, a rerank timed alone, had
come out at 0.81 of its predecessor; at `pamin search` that did not carry
over. What it does carry is memory: one session per model in place of one a
pass took the search server's peak from 5.7 GB to 3.0 GB.

One pair a pass, before `710ceec` at four pairs and after at one:

| metric | before | after | absolute difference | % change | after/before, or difference [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: |
| nDCG@10, XQuAD-R cross-lingual, 1,190 queries | 0.7231 | 0.7229 | −0.0001 | −0.01% | [−0.0020, +0.0016], Holm p = 1.00 |
| nDCG@10, XQuAD-R same-language, 1,190 queries | 0.8679 | 0.8667 | −0.0012 | −0.14% | [−0.0037, +0.0011], Holm p = 1.00 |
| nDCG@10, XQuAD-R single-target, 1,190 queries | 0.6662 | 0.6649 | −0.0013 | −0.20% | [−0.0052, +0.0026], Holm p = 1.00 |
| nDCG@10, MuSiQue 2-hop, first 200 | 0.7669 | 0.7709 | +0.0040 | +0.52% | [−0.0029, +0.0113], Holm p = 1.00 |
| top-5 results, 120 queries | reference | same set on 82 of 120, same order on 38 | N/A: identity, not a difference | N/A | N/A |
| search p50 ms (pooled) | 3,206.7 | 2,675.3 | −531.4 | −16.6% | 0.834 [0.757, 0.931] |
| search p95 ms (pooled) | 5,128.4 | 4,359.3 | −769.1 | −15.0% | 0.850 [0.715, 0.997] |
| search server CPU user s, 120 queries | 1,328.1 | 1,214.1 | −113.9 | −8.6% | 0.914 [0.842, 0.991] |
| search server CPU system s, 120 queries | 42.5 | 19.3 | −23.2 | −54.6% | 0.454 [0.377, 0.540] |
| search server peak RSS MB (VmHWM) | 3,021 | 2,671 | −350 | −11.6% | 0.884 [0.841, 0.930] |
| search server Pss_Anon MB after 120 queries | 1,488 | 1,308 | −180 | −12.1% | 0.879 [0.726, 1.132] |
| ingest wall s (import and drain) | 295.8 | 283.3 | −12.6 | −4.2% | 0.958 [0.904, 1.014] |
| ingest server peak RSS MB (VmHWM) | 2,158 | 1,942 | −216 | −10.0% | 0.900 [0.876, 0.920] |
| index directory MB allocated | 28.9 | 27.4 | −1.5 | −5.3% | 0.947 [0.837, 1.057] |
| model cache in use MB | 2,788 | 2,788 | 0 | 0.0% | N/A: same model files |

Against `ddf0800`, the two steps together took the search p50 to 0.871
[0.796, 0.963] and the search server's peak to 0.472 of what it was. A CPU
with an inference budget of one thread keeps four pairs a pass (`4ca4c6f`),
and that path was not measured in either step. The nDCG@10 rows were taken
through `Engine::search_reranked` over byte-identical copies of one index, as
the paragraph above describes, not at `pamin search`; recall@50 was identical
in every group. CPU time counts the server's worker threads, so it exceeds
wall time; no accelerator ran. The raw rows, per-run premises and harnesses
are in the private q42 and q44 folders named above.

**The longest pass first.** The reranker already handed its passes to the
side-by-side sessions longest first, by reversing its shortest-first plan.
The embedder did not: a store's texts went out in the order they came, so a
long text could start last while the other sessions sat idle.
`Encoder::run_each` now orders every caller's passes longest first, by
padded tokens, and the reranker's own reversal is gone. No pass changes, so
no score does. What moved is ingest, which is where the embedder's order
changed; search reranks as before. Measured at the entry point (`pamin serve`
and one `pamin search --limit 10` process a query, CPU execution provider,
four Xeon cores at 2.8 GHz with AVX-512 VNNI, ONNX Runtime 1.28,
`gpahal/bge-m3-onnx-int8` @ `2b34e84` and
`onnx-community/bge-reranker-v2-m3-ONNX` (`onnx/model_int8.onnx`) @
`6f5ff65`, 3,230 texts ingested, 120 timed queries asked twice after 20
warm ones, four rotated runs an arm, none shared), main `010c4d5` against
the branch at `77f032c`. These rows are of the pass order alone, measured
before main took the graph cap of four (`fdc5782`); the branch merged on top
of it was not re-timed, so they do not cover the two together:

| metric | before | after | abs diff | % change | ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries | 0.7381 | 0.7381 | 0 | 0.0% | top-10 identical in all 8 runs |
| search p50 ms, round 1 (per run) | 2,173.3 | 2,081.9 | −91.5 | −4.2% | 0.958 [0.878, 1.035] |
| search p95 ms, round 1 (per run) | 3,419.6 | 3,110.2 | −309.4 | −9.0% | 0.910 [0.865, 1.034] |
| search server CPU user ms/query | 4,067.0 | 3,853.6 | −213.4 | −5.2% | 0.948 [0.876, 1.029] |
| search server CPU sys ms/query | 23.8 | 21.6 | −2.2 | −9.4% | 0.906 [0.784, 0.984] |
| search VmHWM MB (process RSS) | 1,026 | 1,035 | +9 | +0.9% | 1.009 [0.993, 1.029] |
| ingest wall s | 228.2 | 209.3 | −18.9 | −8.3% | **0.917 [0.867, 0.983]** |
| ingest server CPU user s | 788.0 | 739.8 | −48.3 | −6.1% | 0.939 [0.881, 1.019] |
| ingest server CPU sys s | 10.1 | 9.0 | −1.1 | −10.9% | 0.891 [0.565, 1.514] |
| ingest VmHWM MB (process RSS) | 955 | 1,015 | +60 | +6.2% | **1.062 [1.011, 1.143]** |
| index dir MB | 27.2 | 27.9 | +0.7 | N/A | allocation noise; same index |

Indexing is faster, and its peak memory is higher: the longest passes now run
at the same time, and their intermediates are alive together. Search latency
moved in the same direction but within noise, and search peak memory did not
move. Memory is the `pamin serve` process alone, with no accelerator
services; the prepared model copies (1,668 MB) are the same in both arms and
are not in the index figure. The raw rows and harness are in the private
`internal-docs/pamin-memory/research/2026-10-19/raw/q65/` folder.

**Which searches the latency rows time.** This harness, and the ones below
that ask their 120 timed queries twice, serve the second round entirely from
the query-embedding and reranker score caches: about 30 ms a search, with no
model pass. An earlier version of these tables reported a "per run" p50 and
p95 over both rounds, so its p50 was the fastest round-1 search and its p95
about the round-1 p90. The rows now report round 1 alone, the 120 searches
that run the models. The round-2 ratios all include 1. The pooled rows of
harnesses that ask each query once, above and below, are unaffected, and so
is the table further down that reports the two rounds separately.

**A fixed glibc mapping threshold, trading indexing speed for memory.** glibc
maps a request of 128 KiB or more and returns it on free, but raises that
threshold to the size of each mapped block it frees, up to 32 MiB. After the
first few passes, the passes' intermediate buffers of a few megabytes come
from the heap, which keeps what it frees. `tune_heap` now fixes the threshold
at 1 MiB with `mallopt`, unless the process was started with its own
threshold. Measured at the entry point, with the same setup as the
longest-pass-first rows above, main `1d141db` against the branch at
`d569c57`, four rotated runs an arm, none shared:

| metric | before | after | abs diff | % change | ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries | 0.7381 | 0.7381 | 0 | 0.0% | 120 of 120 queries equal |
| search p50 ms, round 1 (per run) | 1,818.3 | 1,868.9 | +50.6 | +2.8% | **1.028 [1.003, 1.079]**, borderline; pooled 1.038 [0.975, 1.108] |
| search p95 ms, round 1 (per run) | 2,791.2 | 2,879.1 | +88.0 | +3.2% | 1.032 [0.947, 1.093] |
| search server CPU sys ms/query | 18.7 | 38.3 | +19.6 | +105.1% | **2.051 [1.590, 2.383]** |
| search VmHWM MB (process RSS) | 1,019 | 993 | −26 | −2.5% | **0.975 [0.946, 0.991]** |
| search peak RssAnon MB | 368 | 338 | −30 | −8.1% | **0.919 [0.858, 0.963]** |
| ingest wall s | 189.9 | 201.5 | +11.6 | +6.1% | **1.061 [1.014, 1.133]** |
| ingest server CPU sys s | 7.8 | 44.5 | +36.7 | +472.6% | **5.726 [4.742, 9.991]** |
| ingest VmHWM MB (process RSS) | 986 | 789 | −198 | −20.0% | **0.800 [0.707, 0.816]** |
| ingest peak RssAnon MB | 630 | 397 | −232 | −36.9% | **0.631 [0.581, 0.654]** |
| index dir MB | 26.1 | 26.1 | 0.0 | N/A | allocation noise; same index |

**The cost is indexing time.** Every pass now maps its buffers and faults
them in afresh, so ingest system time rises 5.7× and ingest wall time
rises 6%. The gain is memory: indexing peaks 198 MB lower, and search
peaks under a gigabyte. The trade was made deliberately, because the search
peak is held to about a gigabyte.

**Alternatives measured and rejected:**

- **Trimming the heap after every group of passes, in the same runs.** It
  kept none of the memory: search −9 MB and ingest −37 MB, neither
  significant. The blocks are retained within a group of passes, not
  between groups.
- **A 4 MiB threshold.** The search gain disappeared at the entry point.
- **A 128 KiB threshold.** Mapping every small buffer cost ingest 1.19×
  its wall time.

Memory is the `pamin serve` process alone, with no accelerator services.
CPU time counts the server's worker threads. The setting has an effect only
under glibc on Linux; macOS and Windows are unaffected and unmeasured. The
raw rows and harnesses are in the private
`internal-docs/pamin-memory/research/2026-10-20/raw/q69/` folder, with
q64 and q67 beside it.

**An idle model's weights handed back to the page cache.** A CPU session
reads its weights from a mapped prepared copy, and every page a pass touches
stays in the process until the mapping goes. A search runs the embedder and
then the reranker, so the process holds both models' weights. A background
thread now hands a model's clean mappings back with `madvise(MADV_DONTNEED)`
once that model has run no pass for 30 s. The first table below measured an
earlier form that released them after every model's passes in every search;
why that changed, and the idle release's own measurement, follow it.
- The pages stay in the page cache, and the next pass maps them back with
  minor faults.
- A mapping with any written page is skipped.
- Only the mappings an encoder's own sessions created are released. They are
  found once, as each session loads: no two loads from the same model
  directory run at once (loads from different directories do), and that
  directory's mappings in `/proc/self/maps` before are compared with those
  in `/proc/self/smaps` after. A
  released range is therefore never the address of another session's
  mapping, which may be unmapped and its addresses reused.
- Finding them once, not after every pass, matters because reading
  `/proc/self/smaps` walks every page table.

Measured at the entry point with the same setup as the rows above, four
rotated runs an arm. The release is `1e97b9e` against main's code `d569c57`.
The finding-once change is `7e78d51` against `1e97b9e`, with one shared run
excluded and re-run:

| metric | main | release | release, ranges found once | abs diff vs main | % vs main |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries | 0.7381 | 0.7381 | 0.7381 | 0 | 0.0% |
| search p50 ms, round 1 (per run) | 2,384.8 | 2,415.6 (1.013 [0.919, 1.084]) | 0.975 [0.890, 1.038] of release | N/A | N/A, not measured against main |
| search p95 ms, round 1 (per run) | 3,638.4 | 3,695.7 (1.016 [0.923, 1.155]) | 0.912 [0.884, 1.030] of release | N/A | N/A, not measured against main |
| search server CPU sys ms/query | 82.4 | 124.3 | 89.8 (**0.744 [0.481, 0.919]** of release) | N/A | N/A, not measured against main |
| search server reads kB/query | 1,132.6 | 2,743.2 | 1,132.6 | 0 | 0.0% |
| **search VmHWM MB (process RSS)** | **973** | **689 (0.708 [0.693, 0.718])** | 693 (1.001 [0.973, 1.038] of release) | **−284** | **−29.2%** |
| search peak RssAnon MB | 312 | 325 (**1.044 [1.012, 1.075]**) | 323 | +14 | +4.4% |
| ingest VmHWM MB (process RSS) | 790 | 759 (**0.960 [0.916, 0.984]**) | 757 | −31 | −4.0% |
| ingest wall s | 251.4 | 255.7 (1.017 [0.971, 1.088]) | 0.979 [0.932, 1.047] of release | N/A | N/A |
| index dir MB | 27.3 | 27.2 | 28.4 | N/A | allocation noise; same index |

**The search peak falls by about one model's mapped weights, and no latency
change is claimed.**
- At the end of a search, neither model's data file holds a resident page,
  against 291 and 294 MB before.
- Reading smaps after every pass cost reads and system time. Finding the
  ranges once removed both: reads are back to main's exact figure.
- No latency figure moved significantly, either way. The branch head was
  not measured directly against main, so its latency cells against main
  are N/A.
- The measured `7e78d51` found the ranges once per set of loaded sessions by
  looking at the model's directory. Finding them per session load instead
  reads the same files no more often per search, but that form is not in
  this table.
- These runs' absolute latencies are higher than the rows above, because
  they interleaved with an unrelated accuracy replay that never overlapped a
  timed run. The comparisons within each experiment hold.

**Releasing after every search was not shown to cost latency, but its
point estimates lean slower, so the release now waits for 30 s of
idleness.** Two later entry-point experiments measured the per-search
release directly against main, four rotated runs an arm:
- q74 (`740778d` against main `48448b6`): round-1 search p50 +4.9%, 1.049
  [0.966, 1.116]; p95 +0.4%, 1.004 [0.951, 1.117]; search VmHWM −29.4%.
- q78 (`3bc51c2` against main `b2b3bc7`): round-1 search p50 +2.1%, 1.021
  [0.984, 1.133]; p95 +2.8%, 1.028 [0.955, 1.161]; search VmHWM −31.0%.
- No latency change is significant in either; the point estimates are
  about +2% to +5%.
- If the release costs anything, it is not the refaults: it added about
  4,300 minor faults a search, on the order of milliseconds.
- Mapping the weights back in one call before the passes
  (`MADV_POPULATE_READ`, `d7ed3aa`) did not measurably change latency and
  returned the search peak to main's, because the mapped data file includes
  the word-embedding table the product reads with `pread`. It was reverted
  (`003152b`).

Latency outranks memory. An agent's queries come in bursts a few seconds
apart, while a memory tool is idle most of the time. Releasing only after
30 s without a pass should keep a burst's latency at main's and still hand
the weights back while idle. It gives up the lower peak during a burst.

q80 measured the idle release (`c3c6260`) directly against main
(`b2b3bc7`), which is also its immediate predecessor, at the entry point:
four rotated runs an arm (`A I I A I A A I`), none shared. Each run timed the
same 240 back-to-back queries, then waited 45 s with no query and read the
server's memory:

| metric | main | idle release | abs diff | % change | ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries | 0.7436 | 0.7436 | 0 | 0.0% | top-10 identical across all runs |
| search p50 ms, round 1 (per run) | 2,224.2 | 2,325.9 | +101.7 | +4.6% | 1.046 [0.964, 1.130] |
| search p95 ms, round 1 (per run) | 3,453.3 | 3,622.8 | +169.4 | +4.9% | 1.049 [0.941, 1.125] |
| search server CPU user ms/query | 4,201.8 | 4,350.9 | +149.0 | +3.5% | 1.035 [0.978, 1.119] |
| search VmHWM MB (process RSS) | 971 | 972 | +1 | +0.1% | 1.001 [0.987, 1.008] |
| **VmRSS MB after 45 s idle** | **926** | **336** | **−590** | **−63.8%** | **0.362 [0.349, 0.390]** |
| model data files resident after 45 s idle, MB | 585.1 | 0.0 | −585.1 | −100% | released in every run |
| first query after idle | N/A | N/A | N/A | N/A | not measured; see below |
| ingest VmHWM MB | 798 | 782 | −17 | −2.1% | 0.979 [0.975, 0.999] |
| ingest wall s | 258.2 | 265.3 | +7.1 | +2.8% | 1.028 [0.956, 1.121] |

- **Accuracy and latency.** Accuracy is identical. Neither latency change
  is significant; both point estimates are slower.
- **No release during a burst.** At the end of the timed loop, both arms
  still held both models' data files (585 MB), so back-to-back queries
  never released anything. The search peak is therefore main's.
- **Memory while idle.** After 45 s, the idle release held none of the
  model data and its process RSS was 0.36 of main's.
- **The first query after idle is not measured.** The harness re-ran
  queries it had already scored. Those were served from the reranker score
  cache without a pass, so no weights were mapped back. The per-search
  release, which mapped the weights back on every search, is proxy evidence
  only: its round-1 p50 was +2.1% (q78) to +4.9% (q74), not significant,
  with CIs reaching +13.3%. Neither bounds the latency of one query after
  idle, or its tail.

Memory is the `pamin serve` process alone, with no accelerator services.
Disk is the index directory; the prepared copies (1,668 MB) are unchanged.
The release runs only on Linux; macOS and Windows are unaffected and
unmeasured. The raw rows and harnesses are in the private
`internal-docs/pamin-memory/research/2026-10-20/raw/q72b/`, `raw/q72c/` and
`raw/q74/` folders, and in `2026-10-21/raw/q78/` and `raw/q80/`.

**The reranker's int8 weights, re-exported with SmoothQuant.** The upstream
int8 file quantizes each activation with one scale per tensor, at run time.
The inputs to the q/k/v projections and the first feed-forward layer have a few
outlier channels, and those set that scale. PaminLabs' re-export
(`pamin-labs/bge-reranker-v2-m3-sq-int8-onnx` @ `eb7411f`) divides each such
input by a per-channel factor and multiplies the weights by it before
quantizing them (SmoothQuant, alpha 0.75). That re-quantizes 96 weights and
adds 48 elementwise `Mul`s; the rest of the file, the fp16 file and the
tokenizer are upstream's. The recipe and a sync script are in the repository.

q81 measured it (`6743e5b`) against main (`558ff94`), its immediate
predecessor, at the entry point. The setup is the q80 one above: four rotated
runs an arm (`A S S A S A A S`), none shared. Its 120-query nDCG@10 was too
small to settle accuracy, so q82 ran the complete XQuAD-R harness,
`search_reaches_across_languages`, on both (main, and `6743e5b` with the
benchmark's reranker pin moved to the re-export). All 1,190 questions were
scored, on the CPU, each arm in its own fresh evaluation home:

| metric | main | re-export | abs diff | % change | ratio or diff [95% CI] |
| --- | --- | --- | --- | --- | --- |
| **XQuAD-R cross-lingual nDCG@10, 1,190 queries** | **0.7225** | **0.7252** | **+0.0027** | **+0.37%** | **[+0.0008, +0.0045], sign-flip p = 0.004** |
| XQuAD-R same-language nDCG@10 | 0.8667 | 0.8678 | +0.0011 | +0.13% | [−0.0014, +0.0037], p = 0.40 |
| XQuAD-R single-target nDCG@10 | 0.6665 | 0.6673 | +0.0008 | +0.12% | [−0.0028, +0.0046], p = 0.67 |
| XQuAD-R recall@50, every group | 0.9005 / 0.9613 / 0.9017 | the same | 0 | 0.0% | the reranker reorders only |
| nDCG@10, the 120 timed queries | 0.7436 | 0.7414 | −0.0022 | −0.3% | [−0.0083, +0.0041]; 13 better, 18 worse |
| search p50 ms, round 1 (per run) | 2,128.5 | 2,062.5 | −66.0 | −3.1% | 0.969 [0.919, 1.104] |
| search p95 ms, round 1 (per run) | 3,125.4 | 3,157.4 | +32.0 | +1.0% | 1.010 [0.964, 1.185] |
| search server CPU user ms/query | 3,918.4 | 3,829.8 | −88.6 | −2.3% | 0.977 [0.947, 1.094] |
| search VmHWM MB (process RSS) | 967 | 978 | +11 | +1.1% | 1.011 [0.984, 1.020] |
| first query s (loads both models) | 11.69 | 12.18 | +0.49 | +4.1% | 1.041 [0.791, 1.261] |
| ingest wall s | 227.0 | 233.3 | +6.2 | +2.8% | 1.028 [0.989, 1.060] |
| ingest server CPU sys s | 45.6 | 55.4 | +9.8 | +21.5% | **1.215 [1.055, 1.393]** |
| reranker download (int8 file + tokenizer) MB | 560.6 | 560.8 | +0.2 | +0.04% | the 48 scale vectors |
| prepared copies MB | 1,668.9 | 1,668.8 | −0.1 | 0.0% | |

- **Accuracy.** Cross-lingual nDCG@10 rose by 0.0027 over all 1,190
  questions, and the other two groups moved up within noise. A component
  replay on held-out shortlists had measured +0.0024. The 120 timed queries
  moved the other way (XQuAD's 74 of them −0.0051, MuSiQue's 46 +0.0025), but
  within noise, and the full harness is the measurement with the power to
  decide. With reranking off, 1,185 of 1,190 top-10 lists are identical and
  no group moves significantly.
- **Latency and memory.** No significant change in either.
- **The one significant change is a cost.** Ingest's system CPU rose by
  21.5%, while ingest wall time did not change significantly. Ingest does not
  rerank, but a server warms the reranker once its weights are on disk. The
  cause was not isolated.
- **Disk.** The download and the prepared copy are the same size. An
  upgraded install also keeps the old download until it is removed by hand;
  see [cli.md](cli.md#upgrading-from-profiles-and-tiers).

Memory is the `pamin serve` process alone, with no accelerator services. Disk
is the index directory, the prepared copies and the reranker's download, as
named. The XQuAD-R arms ran one after the other on the same host, for
accuracy only; their wall times are not a latency measurement. Accelerators
load the fp16 file, which is upstream's, so they are unaffected and
unmeasured. The raw rows and harnesses are in the private
`internal-docs/pamin-memory/research/2026-10-21/raw/q81/` and `raw/q82/`
folders.

**Truncating at 192 or 384 tokens rather than 256 was measured, and neither
ships.** In the same runs, each limit paired against 256 with the new batching:
192 made a search 0.80 of what it was on MuSiQue, where half the pairs reach
the limit, and cost 0.0045 of its nDCG@10 (`p = 0.010`); 384 cost a third more
a search there (1.32) and moved its nDCG@10 by −0.0016 (`p = 0.31`). On XQuAD-R
and MIRACL neither limit moved ranking by more than 0.0006, significant nowhere.
The rule, and the table, are at `MAX_TOKENS` in the same file.

**The reranker's last layer is computed for its first token alone, and a
batch-1 pass runs its attention without a mask.** Only the classification
token's row of the last layer reaches the classifier. So in that layer, the
attention's query and output projections and both feed-forward layers now run
on one row; the keys and values still run over every token. A batch-1 pass has
no padding, so its fused attention takes no key mask. The rewritten graph is
written beside the prepared copy (`attention.cls1.onnx`, 0.66 MB with its
split twin) and loaded only by computed batch-1 CPU passes; a copy that can
pad keeps the masked graph. Neither change is bit-identical, because the int8
matrix products quantize activations with one scale over the rows they see.

Measured at the entry point with the same setup as the rows above, four
rotated runs an arm, main `48448b6` against `19eaa4a`:

| metric | before | after | abs diff | % change | ratio [95% CI] |
| --- | --- | --- | --- | --- | --- |
| nDCG@10, 120 queries (paired) | 0.7381 | 0.7436 | +0.0054 | +0.7% | diff [−0.0009, +0.0116] |
| **search p50 ms, round 1 (per run)** | **2,305.4** | **2,133.9** | **−171.4** | **−7.4%** | **0.926 [0.872, 0.971]** |
| **search p95 ms, round 1 (per run)** | **3,523.1** | **3,267.6** | **−255.5** | **−7.3%** | **0.927 [0.867, 0.985]** |
| **search server CPU user ms/query** | **4,275.2** | **3,964.8** | **−310.4** | **−7.3%** | **0.927 [0.887, 0.968]** |
| search server CPU sys ms/query | 50.2 | 33.5 | −16.7 | −33.3% | **0.667 [0.404, 0.897]** |
| search VmHWM MB (process RSS) | 988 | 971 | −17 | −1.7% | 0.983 [0.973, 1.004] |
| ingest VmHWM MB (process RSS) | 790 | 787 | −4 | −0.5% | 0.995 [0.969, 1.027] |
| ingest wall s | 244.1 | 250.8 | +6.7 | +2.8% | 1.028 [0.936, 1.120] |
| **search first query s (loads both models)** | **11.29** | **9.92** | **−1.37** | **−12.2%** | **0.878 [0.749, 0.993]** |
| index dir MB | 26.1 | 27.2 | +1.1 | N/A | allocation noise; same index |
| prepared copies MB | 1,668.3 | 1,668.9 | +0.66 | +0.04% | the trimmed graphs |

**Searches are about 7% faster, typical and slowest alike, and ranking
does not move beyond noise.**
- p50, p95 and server CPU fall, and their CIs exclude 1. Pooled over runs,
  round-1 p50 is 0.929 [0.855, 0.983] and p95 0.914 [0.840, 0.993].
- nDCG@10 moves +0.0054, with 23 queries better and 9 worse. Its CI includes
  zero, so no accuracy gain is claimed.
- In an offline replay of every XQuAD-R shortlist, no key moved
  significantly. The logits moved about as much as they do between two builds
  of the runtime.
- The first query is faster too. Its cause was not profiled, so no mechanism
  is claimed.
- Memory is the `pamin serve` process alone, with no accelerator services.
  The prepared-copy figure includes the trimmed graphs; the index figure does
  not.
- The raw rows and harness are in the private
  `internal-docs/pamin-memory/research/2026-10-20/raw/q73/` folder, with the
  replay in `raw/q71/`.

**Reranking only the queries that need it was measured, and does not ship.**
No reranking is the cheapest sufficient choice for 43.4% of XQuAD-R's
cross-lingual queries, so an oracle would save a great deal. The `ROUTES` arm
prices every way of acting on that from one shipped run: a cascade where the
`fast` reranker picks what `accurate` orders, five rules on the fused list, and
a ridge regression of the pass's gain on features of the list before it,
cross-validated by question so no asking of a question is scored by a model
that saw another. Against reranking every query, nDCG@10:

| route | XQuAD-R cross-lingual | MIRACL | MuSiQue |
| --- | --- | --- | --- |
| `fast` picks ten, `accurate` orders them (historical: `fast` has been removed) | −0.0284, p = 0.0001 | | −0.0047, p = 0.0015 |
| best rule (skip when the top is first in both lexical channels) | −0.0133, p = 0.0001 | | ±0 (never fires) |
| learned, reranking half the queries | −0.0085, p = 0.0001 | −0.0036, p = 0.014 | −0.0023, n.s. |
| learned, reranking four in five | −0.0020, p = 0.0001 | +0.0001, n.s. | −0.0000, n.s. |

The learned route beats a random choice at the same rate everywhere -- half the
queries keep 66% of the pass's gain on XQuAD-R, 87% on MIRACL and 93% on
MuSiQue, where chance keeps 50% -- and it is still significantly worse than
reranking everything on the cross-lingual group at any rate that saves much.
Every cross-validated fold on every corpus chose to rerank every query. The
literature says the same thing from the other side: query-performance
predictors transfer badly between collections, and routers recover 60-80% of
an oracle's saving at best. Accuracy is the axis this project will not trade,
so every query is still reranked, and the learned route does not ship.

**Every latency table on this page was taken while idle inference threads
spun, and they no longer do.** ONNX Runtime lets an intra-op thread that runs
out of work spin before it sleeps; the embedder and the reranker each own a
pool of one thread per core, so on a busy machine the spinning threads hold
cores the working ones need. Turned off, through `pamin serve` and `pamin
search` at the defaults on one hundred queries of the own corpus and an
XQuAD-R subset, with rankings and scores bit-identical:

| | ambient load (2.5 to 13) | two busy loops beside the server |
| --- | --- | --- |
| search time, off / on (geometric mean of per-query ratios) | 1.033, `p = 0.068` | **0.754**, `p = 0.0001` |
| eight concurrent callers, throughput off / on | 1.17 | 1.13 |

The latency tables on this page were taken with spinning on, several of them
on four cores at load averages of 6 to 9, so some of what they report is the
spinning rather than the search, and their milliseconds are likely high by an
amount this measurement cannot say per row. They have not been re-taken; the paired ratios in them compare arms
that spun alike, and are what to read. The rule and the method are in the
ADR, under the thread settings.

**Throughput, and where it stops.** The same sweep at one, eight and
thirty-two concurrent callers, taken while `fast` was the default:

| corpus | tier | 1 | 8 | 32 | ceiling |
| --- | --- | --- | --- | --- | --- |
| XQuAD-R, 13,014 | `off` | 13.1 q/s | 19.5 | 20.7 | **~21 q/s** |
| | `fast` | 3.5 q/s | 4.7 | 4.2 | **~4.7 q/s at eight** |
| | `accurate` | 0.8 q/s | 0.9 | 1.0 | **~1 q/s** |
| MIRACL, 131,924 | `off` | 6.5 q/s | 10.3 | 10.6 | **~11 q/s** |
| | `fast` | 2.0 q/s | 2.4 | 2.4 | **~2.4 q/s** |
| | `accurate` | 0.6 q/s | 0.6 | 0.6 | **~0.6 q/s** |

Read it as a ceiling, not a score. Four cores saturate at eight concurrent
callers and the rest is queueing: at thirty-two `fast` gets *less*
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
machine at this corpus size. Those figures include the since-removed `fast`
reranker loaded beside the current one; a current server loads only the
570 MB reranker.

**And where those seven gigabytes are.** The `MEMORY` arm of
`pamin-engine/tests/monolingual.rs` reads `/proc/self/smaps` after each thing a
server loads, on MIRACL at the shipped configuration. Anonymous memory -- the
part only the process can give back:

| stage | anonymous memory added |
| --- | --- |
| opening the index, 66 segments | **3,283 MB** |
| the embedder, and a first search | 1,080 MB |
| the `fast` reranker | 674 MB |
| the `accurate` reranker, whose weights are 545 MB | 911 MB |
| a hundred searches after that | 160 MB |
| freed afterwards by `malloc_trim` | −457 MB |

The index was the largest single item, and not because of its size on disk: a
project grown from empty recorded a segment of 2,000 documents, and every
segment holds its own full-text stores resident. Fifty thousand documents open
at 1,292 MB in 25 segments and at 308 MB in 4. The floor is now 10,000 -- the
largest size whose graph still agrees with an exhaustive scan exactly -- and
`pamin reindex` reshapes an existing index reusing every vector it holds, so it
needs no model: fifty thousand documents reused all fifty thousand. ONNX
Runtime's memory arena is off, which took a hundred `accurate` searches from
6,208 MB to 5,914 MB anonymous with bit-identical scores.

Both tables above were taken with every model loaded from the file the hub
serves, which ONNX Runtime copies onto the heap. They have not been re-run
since the change below.

**The weights are mapped now, not copied.** On the CPU each model loads from a
copy the runtime maps from disk -- written once from the download; see
`crates/pamin-index/src/prepared.rs` -- and
`crates/pamin-index/tests/prepared.rs` measures it through `Reranker::load`
and `Embedder::load`. Each load runs in a fresh process, and what is counted
is live anonymous memory with the model loaded, after `malloc_trim`, in MiB as
the test prints it:

| model | from the download | from the copy | the copy's data file |
| --- | --- | --- | --- |
| `accurate` reranker | 822 | **271** | 833 |
| BGE-M3, the `accuracy` embedder | 824 | **272** | 832 |
| `fast` reranker | 385 | **268** | 133 |

Every score and every vector is bit-identical, and the figures repeated to the
megabyte across two runs. Each load is one session over the copy; the shared
single-threaded session side-by-side passes run on is not in them, and the
shared-session table above has what a whole server holds with it. The
reranker now loads that session with the model rather than at its first
rerank, so this test's `loaded` figure for it includes the session. Measured
through the test's own child process on host A (4 vCPU, the `accurate` copy
of the current export, 12 fresh processes an arm, scores bit-identical), that
added nothing measurable at load -- 67.6 MiB live anonymous memory with the
session loaded and without, because its weights are the same mapped copy --
nor after scoring (108.3 MiB in both); after `malloc_trim` the process held a
median of 58.5 MiB (54.3 to 62.1) rather than 54.4. The table's figures were
taken before then and have not been re-run. What the copy leaves is mostly not the weights: a
bare runtime session adds 139 MB loading the `fast` reranker's download and
12 MB loading its copy, so most of the 268 is what a load holds besides its
session. The data file is the price, on disk rather than in memory -- larger
than the model it came from, because the packed weights are stored beside the
originals.

**And the download goes once its copy has loaded**, so a model is on disk once
rather than twice. Measured on a model directory holding BGE-M3 and the
`accurate` reranker as the code before this left them after first use -- each
download beside its copy -- seeded by hard links from the evaluation
workspace's directory so that nothing was downloaded, and read with `du` before
and after one load of each through `Embedder::load` and `Reranker::load`:

| | bytes on disk |
| --- | --- |
| each download beside its copy | 2,923 MB |
| after one load of each | **1,783 MB** |

Each load mapped the copy that was already there (read from
`/proc/self/maps`), so the 1,141 MB is the two int8 exports, 570 MB each; what
is left is the two copies and the two 17 MB tokenizers. The same lifecycle on a
fresh directory through the hub, with the `fast` tier to spare the disk: 157 MB
after its first load, where its 119 MB download used to stay beside that. With
its copy renamed to another key, which is what a runtime upgrade leaves, and
the hub unreachable, the next load failed naming the model and the network;
with the hub back it fetched the file again, wrote the copy under the same key
as before -- a re-download is given the removed file's modification time, so
it keys identically -- and removed the download again. All five loads scored
bit-identically, the load from the download (then `PAMIN_PREPARED=off`, now
`--prepared false`) among them. What this gives up is
offline use across a key change, and [cli.md](cli.md) says so where the
setting is described. Disuse is not a key change: the collection that removes
copies nothing has loaded for two weeks never removes the one a model's record
points at for the running version, so a tier left unused does not need the
network again.

**And one vocabulary between them, not one each.** What a copy leaves is mostly
tokenizer. BGE-M3 and every reranker tier use the same 250,002-piece Unigram
vocabulary, and loading it adds 280 MiB anonymous each time -- measured by
loading BGE-M3's `tokenizer.json` and then the `accurate` reranker's in one
process, 280 MiB for each. The `accurate`
reranker describes exactly BGE-M3's model, every piece and score bit for bit, and `fast`
differs only in leaving a default flag unstated, so a loaded model now finds
one already built rather than building its own; see
`crates/pamin-index/src/tokenizer.rs`.
`crates/pamin-index/tests/shared_vocabulary.rs` measures it with BGE-M3 and
the `accurate` reranker both loaded, as a server searching at the defaults
holds them, through `Embedder::load` and `Reranker::load` against `fastembed`
loading the same prepared copies the way the product did before. Each arm is a
fresh process; nine rounds across two runs, on four cores shared with another
measurement, in MiB:

| | `fastembed` | shared vocabulary |
| --- | --- | --- |
| live, after `malloc_trim` | 539 -- 582, median 582 | **292 -- 330, median 330** |
| before the trim | 778 in every round | 339 -- 346 |
| seconds to load both | 3.30 -- 3.66 | 1.77 -- 1.95 |

**Provisional and unretained:** the raw rows and the commands behind this
table were not kept, so it cannot be re-checked, and it stands only until it
is measured again.

Every vector and every score is bit-identical: 18,432 embedding values over
eighteen texts and 65 `accurate` scores, over eight scripts, runs of spaces,
trailing and leading spaces, empty texts and one past both length limits --
and 65 `fast` scores, from its download. Before the trim the gap is wider than one vocabulary; `fastembed`'s
loader clones each tokenizer it configures and drops the original, which would
leave that much freed and not yet returned, but that was not measured
separately. Twenty rerank passes of sixteen pairs of about 200 tokens showed no speed
change: the per-round median moved between -22% and +9% against `fastembed`,
median -2.5%, while one arm's own rounds moved by up to 40%.

The 280 MiB is `tokenizers`' own Unigram model, a trie with a hash map in every
node. The shared vocabulary has since been held in a double-array trie instead
(see `crates/pamin-index/src/tokenizer.rs`), with ids tested equal to that
model's. The figures above are for the layout before it. The same measuring
child, run before (`ad33d2d`) and after (`c176020`) in the order A B, B A,
A B on a shared 4-vCPU Xeon with a warm page cache, gave these medians of
three:

| metric | before | after | change |
| --- | ---: | ---: | ---: |
| anonymous MiB after loading both models | 345.0 | 95.8 | −249.2 (−72.2%) |
| anonymous MiB held after use and trim | 314.4 | 47.1 | −267.3 (−85.0%) |
| peak RSS MiB | 991.2 | 744.2 | −247.0 (−24.9%) |
| release `pamin` binary, bytes | 38,494,728 | 38,523,368 | +28,640 (+0.074%) |
| new cache or index files | none | none | N/A: no file write added |
| model cache | unchanged | unchanged | N/A: no model or prepared-copy path touched |

The disk rows compare the release binary built at `perf/rescore-simd`
(`67fd1f3`) and at this change (`21d249d`). The change's product code is the
tokenizer and its `yada` dependency, and it adds no file write.

At the search entry point (`pamin import` and `pamin search` against a fresh
`pamin serve`), the same two commits gave the rows below. Before is
`67fd1f3` (`perf/rescore-simd`) and after is `21d249d`.

**Scope.** These rows compare the tokenizer change with its parent on the
stack as it stood then. Both commits are still ancestors of this branch, but
the branch has since been merged forward onto a newer `perf/rescore-simd`
(`d1643db`), which brought in other product changes, including changes to the
engine, embedding, inference and reranking code. The rows therefore do not
measure the branch as it now stands. A measurement of the current head against
`d1643db`, its current parent, is pending; until it is retained, no figure
here describes that pair.

**Conditions.**

- Host: a shared 4-vCPU Xeon @ 2.10 GHz (AVX-512 VNNI, AMX). ONNX Runtime
  1.28 on the CPU execution provider, with 4 threads.
- Models: profile `accuracy`, which is BGE-M3 int8 with the `accurate`
  reranker. The resolved Hugging Face revisions were
  `gpahal/bge-m3-onnx-int8` at `2b34e84df040034d4b9eabb62383a87c18955822`
  and `onnx-community/bge-reranker-v2-m3-ONNX` (`onnx/model_int8.onnx`) at
  `6f5ff65298512715a1e669753bc754d2bc8f367b`. These come from the model-cache
  listing taken after the untimed seed run, which held one snapshot of each
  repository. Every timed run checked that the cache still matched that
  listing, before and after the run.
- Workload: the corpus is 3,230 texts in a fixed order. The search phase is
  120 timed queries, after 20 warm-up queries, on a warm server for each
  phase.
- Runs: 4 runs per arm in the order ABBA BAAB. Every arm first waited for a
  load average under 1.0 for 60 s. No run was marked shared.
- Memory: server process only, with no accelerator services.
- Disk: the timed project's index directory and the model cache in use.
- CIs: 95% percentile bootstrap.

**No speed change is claimed: every CI includes 1.0.** The gain is memory.

| metric | before | after | change | ratio [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| search p50 ms (pooled) | 2,199.84 | 2,155.41 | −44.44 (−2.0%) | 0.980 [0.925, 1.041] |
| search p95 ms (pooled) | 3,336.45 | 3,213.47 | −122.98 (−3.7%) | 0.963 [0.901, 1.074] |
| ingest wall s (import and drain) | 285.17 | 286.70 | +1.53 (+0.5%) | 1.005 [0.968, 1.039] |
| search server peak RSS MB (VmHWM) | 2,082 | 1,922 | −160 (−7.7%) | N/A |
| search server Pss_Anon MB | 853 | 659 | −194 (−22.7%) | N/A |
| ingest server peak RSS MB (VmHWM) | 1,750 | 1,487 | −263 (−15.0%) | N/A |
| index directory MB allocated | 54.97 | 55.64 | +0.66 (+1.2%) | N/A |
| model cache in use MB | 2,788 | 2,788 | 0 (0.0%) | N/A |

q38 gives no CI for the memory and disk rows. The change
writes no new files.

Top-5 results at the entry point fall into two clusters that mix both arms.
This is the segment-layout run-to-run variation, not the tokenizer. The
write-up, raw rows and harness are at
`internal-docs/pamin-memory/research/2026-10-14/q38-compact-tokenizer-timing.md`
and `internal-docs/pamin-memory/research/2026-10-14/raw/q38/`.

Outputs were bit-identical in all six runs. Encoding time was not measured,
because the host was busy; load time moved by less than its run-to-run spread.
These are whole-process figures, so they bound what the new layout holds but
do not isolate it. The raw output of all six runs, the commands that drove them
and the conditions are retained for maintainers in the private submodule (see
[internal-docs.md](internal-docs.md)), at
`internal-docs/pamin-memory/research/2026-10-13/raw/t1-tokenizer-memory/`.

**Where the rest of a search's memory was: glibc's heap arenas.** Since
reranker passes run on threads of their own, glibc gave each contended thread
an arena of its own (up to eight per core) and each arena kept what it freed.
On four cores a server ended a search holding 31 to 35 arenas of 64 MiB and
about a gigabyte of free heap. `pamin` now caps them at two
(`mallopt(M_ARENA_MAX, 2)` in `main`, before the runtime starts its threads;
glibc only). A process started with `MALLOC_ARENA_MAX` or
`glibc.malloc.arena_max` in `GLIBC_TUNABLES` keeps that limit instead,
higher or lower, and the figures below do not describe it. At the search entry point, before `d0b6b89` and after `7977b78`,
same conditions as the tokenizer rows above but 4 runs per arm in the order
CPPC PCCP, `inference_threads = 4` from `config.toml`, the resolved model
revisions unchanged:

| metric | before | after | change | ratio [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| top-5 order, 120 queries, every run | identical | identical | none | N/A |
| search p50 ms (pooled) | 1,203.1 | 1,159.5 | −43.6 (−3.6%) | 0.964 [0.893, 1.043] |
| search p95 ms (pooled) | 1,913.2 | 1,734.4 | −178.9 (−9.3%) | 0.907 [0.845, 1.034] |
| ingest wall s (import and drain) | 140.5 | 134.8 | −5.7 (−4.0%) | 0.960 [0.894, 1.065] |
| ingest import wall s | 139.56 | 134.11 | −5.46 (−3.9%) | 0.961 [0.894, 1.063] |
| ingest import server CPU user s | 471.93 | 455.30 | −16.63 (−3.5%) | 0.965 [0.887, 1.072] |
| ingest import server CPU sys s | 6.19 | 4.76 | −1.43 (−23.0%) | 0.770 [0.644, 0.976] |
| ingest drain wall s | 0.913 | 0.712 | −0.202 (−22.1%) | 0.779 [0.497, 1.654] |
| ingest drain server CPU user s | 1.635 | 1.245 | −0.390 (−23.9%) | 0.761 [0.232, 3.458] |
| ingest drain server CPU sys s | 0.130 | 0.075 | −0.055 (−42.3%) | 0.577 [0.231, 1.500] |
| search server CPU user s per query | 4.422 | 4.257 | −0.166 (−3.7%) | 0.963 [0.890, 1.062] |
| search server CPU sys s per query | 0.085 | 0.025 | −0.060 (−70.3%) | 0.297 [0.257, 0.335] |
| search server peak RSS MB (VmHWM) | 2,451 | 1,322 | −1,129 (−46.1%) | 0.539 [0.523, 0.555] |
| search server peak anonymous RSS MB | 1,527 | 418 | −1,109 (−72.6%) | 0.274 [0.262, 0.286] |
| search server peak file-backed RSS MB | 915 | 915 | 0 (0.0%) | 1.000 [0.998, 1.003] |
| ingest server peak RSS MB (VmHWM) | 1,918 | 1,129 | −789 (−41.1%) | 0.589 [0.557, 0.625] |
| index directory MB allocated | 28.3 | 26.7 | −1.6 (−5.8%) | 0.942 [0.912, 1.044] |
| model cache in use | unchanged | unchanged | N/A: no model path touched | N/A |

**No speed change is claimed: every latency CI includes 1.0.** Import is
`pamin import` writing the texts and indexing them as they arrive; drain is
`pamin cascade drain` finishing the jobs left after it, under a second, whose
CIs are wide. CPU is the server process's own; postgres and the clients are
excluded. Memory is the server process only, sampled every 0.5 s, with no
accelerator services. Disk is the project's index directory: postgres data,
model caches and temporary files are not in it, and temporary files were not
measured (N/A). The
file-backed part is almost all the two prepared model copies (849 MB
resident, each mapped once), which is what remains between this server and
1 GB. The same cap set through `GLIBC_TUNABLES` on `d0b6b89` measured the
same (search VmHWM 2,354 to 1,306 MB). The summaries, raw rows and harness
are retained for maintainers in the private submodule (see
[internal-docs.md](internal-docs.md)):
[q52-arena-cap-in-code.md](../internal-docs/pamin-memory/research/2026-10-18/q52-arena-cap-in-code.md)
with [raw/q52/](../internal-docs/pamin-memory/research/2026-10-18/raw/q52/), and
[q51-memory-split-and-arena-cap.md](../internal-docs/pamin-memory/research/2026-10-18/q51-memory-split-and-arena-cap.md)
with [raw/q51/](../internal-docs/pamin-memory/research/2026-10-18/raw/q51/).

**The word-embedding tables, read rather than mapped.** Each prepared CPU
copy's uint8 word-embedding table (250,002 rows of 1,024) was gathered by
ONNX Runtime from the mapped file, and the kernel's read-around left much of
it resident. The prepared copy now drops that `Gather` and feeds the rows a
batch needs, read with `pread` and dequantized as `DequantizeLinear` does,
bit for bit. At the search entry point, one binary (`f34cec6`) with the
hidden `--split-embeddings` flag off (before) and on (after), 4 runs per arm
in the order ABBA BAAB, on a 4-vCPU Xeon @ 2.80 GHz (AVX-512 VNNI, no AMX),
CPU execution provider, `inference_threads = 4`, the same 3,230 texts, 20
warm-up and 120 timed queries:

| metric | before | after | change | ratio [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| nDCG@10, 120 queries | 0.7381 | 0.7381 | 0.0000 (0.0%) | 120 of 120 equal |
| top-10 order, 120 queries, every run | identical | identical | none | N/A |
| search p50 ms (pooled) | 2,040.6 | 2,000.5 | −40.1 (−2.0%) | 0.980 [0.894, 1.073] |
| search p95 ms (pooled) | 3,301.2 | 3,107.5 | −193.7 (−5.9%) | 0.941 [0.859, 1.043] |
| search server CPU user ms per query | 7,765.0 | 7,593.0 | −172.0 (−2.2%) | 0.978 [0.902, 1.026] |
| search server peak RSS MB (VmHWM, median of runs) | 1,425 | 1,171 | −254 (−17.8%) | 0.822 [0.650, 1.039] |
| search server peak file-backed RSS MB | 1,063 | 798 | −265 (−24.9%) | 0.751 [0.530, 1.126] |
| search server peak file-backed PSS MB | 906 | 647 | −258 (−28.5%) | 0.715 [0.676, 0.853] |
| ingest server peak RSS MB (VmHWM) | 1,109 | 926 | −184 (−16.5%) | 0.835 [0.772, 0.935] |
| ingest wall s (import and drain) | 222.8 | 210.2 | −12.6 (−5.6%) | 0.944 [0.915, 1.046] |
| search server read syscalls per query | 584 | 2,836 | +2,252 (+386%) | the row reads |
| index directory MB allocated | 27.2 | 26.1 | −1.1 (−4.1%) | 0.959 [0.843, 1.087] |
| prepared copies MB (both, split files included) | 1,668.3 | 1,668.3 | 0 (0.0%) | written in both arms |

**No speed change is claimed: every latency and CPU CI includes 1.0.** Memory
is the server process only, with no accelerator services; disk is the index
directory and the two prepared copies, without postgres or temporary files
(not measured, N/A). Half the runs in each arm crossed a five-minute model
reload this host triggered (its CUDA provider cannot register; fixed in the
next paragraph), which mapped both models twice for a moment; without it the after arm peaked at 1,022 and
1,049 MB. The summary, raw rows and harness:
[q54-split-word-embeddings.md](../internal-docs/pamin-memory/research/2026-10-19/q54-split-word-embeddings.md)
with [raw/q54/](../internal-docs/pamin-memory/research/2026-10-19/raw/q54/).

**A provider that cannot register no longer reloads the models every five
minutes.** Every x86-64 Linux build offers CUDA to the plan measurement, and on
a machine where CUDA cannot register (here its `libcublasLt.so.13` is missing)
that failure was treated as transient. So the CPU plan it fell back to kept the
five-minute retry interval rather than the normal lifetime. Each time the
interval ran out, both models were reloaded beside the resident ones -- mapping
the prepared copies twice -- and the reranker's pair-score cache, which lives
in the replaced reranker, was emptied. A registration failure is now
incompatible. At the search entry point, main (`83f5442`, before) against the
fix (`0f31715`, after), 4 runs per arm rotated A B B A B A A B (a fifth A run
was excluded because another process shared the host). Hardware and settings:
a 4-vCPU Xeon @ 2.80 GHz (AVX-512 VNNI, no AMX), the CPU execution provider,
`inference_threads = 4`, split embeddings on, and the same 3,230 texts. The 120
timed queries were asked twice each, 240 searches, so the search phase outlasts
the deadline. The corpus is `corpus.jsonl` (SHA-256 `5e41ac21…`), and the timed
queries are `queries_timed.jsonl` (SHA-256 `4012c98a…`), asked in file order, the
whole list once and then again, after 20 warm-up queries on a fresh home per
run. The models are `gpahal/bge-m3-onnx-int8` at `2b34e84` and
`onnx-community/bge-reranker-v2-m3-ONNX` at `6f5ff65`, both int8, loaded from
their prepared copies with split word embeddings. Every A run reloaded once
and no B run did:

| metric | before | after | change | ratio [95% CI] |
| --- | ---: | ---: | ---: | ---: |
| nDCG@10, 120 queries | 0.7381 | 0.7381 | 0.0000 (0.0%) | identical top-10s |
| search p50 ms, first time a query is asked | 2,220 | 2,152 | −68 (−3.1%) | N/A (within run spread) |
| search p95 ms, first time a query is asked | 3,443 | 3,419 | −24 (−0.7%) | N/A |
| search p50 ms, the same query again | 2,085 | 32 | −2,053 (−98.5%) | the cached scores survive |
| search p95 ms, the same query again | 3,293 | 43 | −3,250 (−98.7%) | the cached scores survive |
| search server CPU user ms per query, both passes | 7,782 | 4,036 | −3,747 (−48.1%) | 0.519 [0.485, 0.563] |
| search server CPU system ms per query, both passes | 44.5 | 23.5 | −21.0 (−47.1%) | 0.529 [0.430, 0.591] |
| **search server peak RSS MB (VmHWM, median of runs)** | **1,295** | **1,032** | **−263 (−20.3%)** | **0.797 [0.785, 0.821]** |
| search server peak file-backed RSS MB | 945 | 652 | −293 (−31.0%) | 0.690 [0.688, 0.690] |
| search server peak file-backed PSS MB | 646 | 647 | 0 (0.0%) | 1.000 [0.999, 1.001] |
| ingest server peak RSS MB (VmHWM) | 925 | 962 | +37 (+4.0%) | 1.040 [0.940, 1.079] |
| ingest wall s (import and drain) | 230.4 | 228.7 | −1.7 (−0.7%) | 0.993 [0.933, 1.047] |
| index directory MB allocated | 27.2 | 26.1 | −1.1 (−4.0%) | 0.960 [0.922, 1.043] |

**No first-pass speed change is claimed.** The repeated-query rows are an
upper bound: they time a query whose answer is still cached, which a workload
that rarely repeats a query seldom sees. The CPU row is specific to this run,
which asks every query exactly twice: a workload that repeats queries across
several reload intervals would miss the cache again after each reload before
the fix, so its saving could be larger, and one that never repeats a query
would save less. The first-pass rows were computed from the raw query
rows, and CPU was sampled over the whole timed loop, so a first-pass CPU figure
is N/A. Memory is the server process only, with no accelerator services; disk
is the index directory, without postgres, the prepared copies (unchanged, 1,668
MB) or temporary files (not measured, N/A). The summary, raw-row checksums and
harness:
[q56-provider-reload.md](../internal-docs/pamin-memory/research/2026-10-19/q56-provider-reload.md)
with [raw/q56/](../internal-docs/pamin-memory/research/2026-10-19/raw/q56/).

**What the database is made of**, which is a figure this page has never carried
and which turned out to be worth carrying. Broken down by table on the
evaluation workspace, 1.7 GB across its projects, before either change below:

| table | total | share |
| --- | --- | --- |
| **`index_jobs`** | **632 MB** | **38.7%** |
| `source_versions` | 273 MB | 16.7% |
| `topic_states` | 239 MB | 14.6% |
| `topics` | 176 MB | 10.8% |
| `topic_name_tokens` | 139 MB | 8.5% |
| `sources` | 94 MB | 5.8% |
| `source_spans` | 81 MB | 4.9% |

**The largest table was the work queue, and the corpus had finished indexing.**
`index_jobs` held 651,128 settled rows out of 1,054,646, completed the day
before and pruned by nothing since — because nothing had been written since.
`jobs::prune` existed and worked; its call sat behind `drained.completed > 0`,
so a drain cleaned the queue only when it had found work, and skipped it in
exactly the state that needs it.

That was fixed, and then the question behind it was asked: **why keep a settled
row at all?** Nothing read one. Every statement over the table asks about work
still owed, and `enqueue` inserting a fresh row leaves it in the same state as
reviving a settled one did. So a job is now deleted when it completes, and
migration V10 deletes the settled rows an earlier build kept.

The indexes were the other half: payloads were 57 MB of the 632, and the rest
was row overhead and six indexes. `EXPLAIN ANALYZE` of every statement in
`jobs.rs`, on a synthetic queue shaped like that one (1,054,646 rows, 651,128
settled, five projects), found one of the six used by nothing and two
answering questions a single index answers:

| index | used by | now |
| --- | --- | --- |
| primary key | claim, complete, fail | kept |
| `(project_id, idempotency_key)` unique | the enqueue conflict | kept |
| `by_priority (priority, available_at, project_id)` | the claim, leading with the wrong column since claims became per-project | replaced |
| `claimable (project_id, available_at)` | `pending`, `failed`, `replay`, `discard`, through `project_id` alone | replaced |
| `exhausted` partial on `last_error IS NOT NULL` | nothing — no statement says that, so no plan can use it | dropped |
| the TOAST table's | PostgreSQL | kept |

`index_jobs_claim_order (project_id, priority, available_at)` replaces the
middle three: the claim reads it in order and the per-project statements by its
prefix. On the synthetic queue the three were 55 MB, and the one index is
20 MB over the 403,518 rows still owed.

**`topic_states` kept a second copy of every memory.** This page used to say
it duplicated `topics.content`, which has never existed; the column was
`topic_states.content`, and what it duplicated was the span it points at.
Commit 39b60c5 checked all 425,916 states of the evaluation workspace and found
every one equal to its span's slice of `source_versions.content`, 94 MB of
content bytes. A state's content is now read from the evidence through the
span, cut at its byte offsets, and migration V9 drops the column — after
checking every row, and refusing with the offending state named if any
disagrees.

Both measured on fresh temporary workspaces, as an unprivileged user on four
cores shared with other measurements, by a scratch harness that drives
`Engine::remember` and `drain_cascade` at the default profile: XQuAD-R's 2,640
distinct paragraphs in eleven languages, one topic each, before the change
and after it, sizes from `pg_total_relation_size`:

| | before | after |
| --- | --- | --- |
| `topic_states`, 2,640 states | 3.92 MB | **1.06 MB** |
| `index_jobs`, 7,920 jobs owed | 4.09 MB | 3.91 MB |
| `index_jobs`, once drained | 5.45 MB, every row kept | 4.01 MB, **no rows** |
| `index_jobs`, drained and vacuumed | 5.45 MB | 2.14 MB |
| whole database, drained and vacuumed | 16.6 MB | **10.4 MB** |

`topic_states` falls by 73% on this corpus because its paragraphs are long —
1,237 bytes on average — and the table now holds a row's bookkeeping and no
text. The evaluation workspace's memories are shorter, so there the column
was 94 MB of a 239 MB table rather than most of it.

What it costs, paired and alternated between the two builds, five rounds of
2,640 writes each and three read rounds per round:

| | before | after |
| --- | --- | --- |
| one `remember`, median of five per-round medians | 1.93 ms (1.55–2.20) | 1.88 ms (1.67–2.25) |
| `topic_states_by_id`, 150 states, since deleted: nothing called it | 1.58 ms (1.39–1.82) | **1.89 ms** (1.71–2.07) |
| `current_states_of`, 150 topics | 1.70 ms (1.49–1.97) | **2.06 ms** (1.84–2.15) |
| `current_content`, the write path's lookup | 0.100 ms | 0.111 ms, inside both ranges |

The write is unchanged within the noise. **The state fetch is not free**: one
more primary-key join, to `source_versions`, costs about 0.3 ms at the
150-candidate ceiling, which is the price of storing each memory once. A
search pays it in up to three lookups — the channels' candidates, any topic the
query names that no channel returned, and the graph's arrivals, at most fifty
at the default depth — so under a millisecond, against the historical 99 ms
[the CLI reference](cli.md) gives a fusion-only search and 1,522 ms for one
reranking twenty candidates; a search today reranks thirty. The results
themselves did not move: the before workspace, migrated by the after build,
returned byte-identical top tens for forty questions through
`Engine::search_reranked` at its defaults, scores to six decimals, and read
back every state identically; so did a workspace the after build filled from
scratch.

**Every state also carried five columns nothing ever wrote.** `importance`,
`worth_positive`, `worth_negative`, `access_count` and `last_accessed_at` held
their defaults in every row any build produced, and were read into every state
the store loaded for a field nothing read. The reads went first, and migration
V11 drops the columns, refusing with the state named if any row holds anything
but the default. On XQuAD-R's 2,640 paragraphs, written through the store's
append functions into fresh workspaces by a scratch harness, then
`VACUUM ANALYZE`:

| | before | after V11 |
| --- | --- | --- |
| a `topic_states` row, `pg_column_size` | 136 bytes | **120 bytes** |
| `topic_states` heap | 393,216 bytes | **352,256 bytes** |
| `topic_states` with its indexes | 1,081,344 bytes | 1,040,384 bytes |
| `current_states_of`, 150 topics, p50 | 3.95 ms (3.78–4.03) | **3.59 ms** (3.48–3.75) |

The fetch figures are six paired processes, alternating which build went
first, three rounds of 300 calls each; the build after was faster in six of
six, by a median 0.32 ms. Almost all of that is the reads, not the smaller
rows: the build that stops reading the columns without dropping them,
alternated against the build before on the same workspace, was 0.37 ms
faster, also six of six. Both harness and store were debug builds, which
inflates exactly the per-column decoding removed here, on four cores at a
load average near 13, and these fetches are slower than the table above for
both reasons -- read the difference, not the level. Nothing ranked on the
columns, so no result can move.

**A migration does not make an existing file smaller.** Dropping a column marks
it dropped, and a `DELETE` frees space for PostgreSQL to reuse; neither returns
anything to the filesystem, and `VACUUM FULL` cannot run inside the
transaction a migration runs in. So V9, V10 and V11 stop the growth — every state
written afterwards is smaller, and the queue stops at the work owed — and the
bytes already on disk stay until something rewrites the tables. Measured on the
migrated before workspace with 15,840 states and 39,600 jobs owed: `topic_states`
22.5 MB as migrated and 4.9 MB after `VACUUM FULL topic_states`; `index_jobs`
21.5 MB and 16.3 MB. Nothing in the product runs one.

What is still unattributed on this axis: `source_versions` at 16.7% has not been
looked at, and it is now the only copy of every memory's text.

**The queue said each job's subject three times.** Once in `payload` as JSON,
once inside `idempotency_key` as `<kind>:<subject>` text, and the kind again in
`job_type` — 59 MB of payloads and 57 MB of keys on the synthetic queue, where
the unique index over the key was the largest index at 126 MB. Migration V12
gives the subject a column and states the uniqueness on
`(project_id, job_type, subject)`, nulls not distinct so project-wide work
still coalesces. Filled through `jobs::enqueue_all` on a fresh cluster with
300,001 owed rows (100,000 subjects, three kinds each), two runs a build:
the table 67.0 MB → 38.4 MB, the unique index 35.7–35.9 MB → 27.7–28.0 MB,
the table with its indexes 135.7–135.9 MB → 99.5–99.7 MB, and an enqueue of
three kinds p50 0.42–0.46 ms → 0.39–0.40 ms.

**The first search on a new connection is slower, and nothing that can be
moved off it pays.** A search's statements -- the states of sixty candidates,
the names in the query, their states, a two-hop walk keeping fifty, the states
it reached, the names at the ends of its paths -- asked on a freshly opened
resident pool, then again on the same connection: 29 rounds a run, three runs
a row, a project of 320 memories that name each other, debug build, four cores
at a load average near 20. The figures are p10, because the medians of the
same runs moved by up to two thirds:

| | first | second | the warm-up itself |
| --- | --- | --- | --- |
| a new backend, nothing run on it | 9.8 -- 12.7 ms | 6.2 -- 7.5 ms | |
| after the statements ran once, matching nothing | 7.1 ms | 6.1 -- 6.2 ms | 4.4 -- 4.6 ms |

The server's statement log (`log_min_duration_statement = 0`) puts about three
of the four to five milliseconds in the backend. Planning took 3.4 -- 3.6 ms
against 1.5 -- 1.7 (17.7 in one of three traced rounds), parsing 0.6 -- 0.8 ms
against 0.2 -- 0.3 for the same statements parsed again on the warm backend,
and executing 1.3 -- 1.5 against 1.1 -- 1.2. Nearly all of it lands on the first
statement to touch each table: `EXPLAIN (ANALYZE, SUMMARY)` of the state lookup
in a new session plans in 1.8 -- 2.4 ms and in 0.3 -- 0.8 ms the next time. So
it is the backend filling its catalog and relation caches, which also shows as
423 minor page faults in the first run against 8 in the second. `sqlx`'s own
share is the four statements it prepares, one round trip each, and clearing its
cache on the warm backend puts that at 0.5 -- 1.0 ms. No index or model work is
in these figures, because the harness asks only the store.

Running the statements first does remove the cost from the search, and it
costs as much as it removes: the caches have to be filled once per backend,
whoever fills them. So the only question is where the cost lands. A resident
server opens its first connection at start, and its first search in a project
also opens that project's index and loads its model, which costs far more than
five milliseconds. Every other connection is opened by a request that finds the
pool busy. That request would pay a warm-up at the same moment it now pays for
the cold statements, and if it were a write it would pay for a search's
statements. Connections are kept for the server's life (`idle_timeout` and
`max_lifetime` are off), so each backend is cold once. The server therefore
does not warm its connections. One lead for a later change: every search plans
each of its statements again, and planning is 1.5 of the 2.6 ms the warm
backend spends.

**What else the index engine offers was measured.**
`zvec-rust` 0.7.2 adds a half-precision vector field, IVF-RaBitQ and DiskANN
beside the graph, a memory limit and a document iterator. Each was held to a
rule written before its first number, in the order accuracy, latency, memory,
disk, and the storages were run through `search_reranked` at the default tier
and profile with both caches bypassed and the arms in rotated order: all 157
own-corpus queries, 595 of XQuAD-R's and 200 of MuSiQue's, on four cores two
other evaluations were sharing, so times are ratios within a question.

| against fp32 | nDCG@10: own / XQuAD-R / MuSiQue | search time | vectors on disk, MIRACL | resident, MIRACL | recall@10, MIRACL / synthetic |
| --- | --- | --- | --- | --- | --- |
| int8 codes and refiner | identical on all 952 questions | 0.998 (p = 0.47) | +27% | +25% | 1.0000 / 0.9980, as fp32 |
| fp16 vector field | −0.0004 / −0.0002 / 0.0000, none significant | 0.997 (p = 0.39) | −45% | −45% | 0.9985 / **0.9650**, against 1.0000 / 0.9980 |
| IVF-RaBitQ, half the lists probed | not run | not run | +18% | +12% | 0.9981 / 0.951 |
| DiskANN | not run | not run | +95% | **−96%** | 1.0000 / — |

Int8 makes the index itself answer in about half to four-fifths of the time,
and that does not reach the search: the index is a few milliseconds of a search
the reranker spends one to six seconds on. The fp16 field halves the vector
bytes, but the engine's fp16 arithmetic loses recall on clustered vectors that
rounding alone does not, and the rule allowed 0.002 (the next paragraph takes
that loss back with a rescore). DiskANN is the only
memory lever, holding 22 MB where the graph holds 565, at 2.5 to 5 times the
index query time, twice the disk and fourteen times the build, so it was left
as the path for a project whose resident set is the constraint -- until the
decision below made it the default.
An explicit memory limit changed nothing, because it sizes a pool only an index
created with mmap off reads. The iterator reads 131,924 documents in 0.28 s
against 1.5 s of keyed fetches, which is about 1% of a rebuild; it failed its
rule and ships anyway, because it changes nothing a rebuild lends and the owner
takes every optimization that costs no accuracy. The tables and the rules are
in [the ADR](adr/0001-tech-selection.md#what-else-zvec-rust-072-offers-measured);
the runs were scratch builds and cannot be re-run from the repository.

**Vectors are now half precision, under `disk` or `memory`.** The owner set
the storage directly: fp16 vectors, the HNSW graph in memory by default
(`memory`) or a DiskANN graph on disk (`disk`), both ranking twice the
candidates again by an exact f32 cosine, because the engine's own fp16 scores
lose recall in their arithmetic rather than in the rounding. Against exact
search, with fp32 HNSW -- what shipped before -- as the bar:

| | recall@10 / @50, 50,000 synthetic | recall@10 / @50, MIRACL | vector query, MIRACL | full build, MIRACL | resident, MIRACL | disk, MIRACL |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 HNSW (before) | 0.9980 / 0.9974 | 1.0000 / 0.9999 | 9.5 ms | 171 s | 575 MB | 595.5 MB |
| `memory` (default) | 0.9965 / 0.9965 | 1.0000 / 0.9997 | 5.9 ms | 84 s | 320 MB | 327.8 MB |
| `disk` | 0.9985 / 0.9975 | 1.0000 / 0.9996 | 68.6 ms | 1,379 s | 34 MB | 618.1 MB |

Query times are medians at k = 10 on four cores other evaluations were
sharing at load 9 to 13, so they are directions. `disk` reaches the bar only
at a search width of 1,200 (0.981 at its default of 300 on the synthetic set),
and its cost on the write path is the open question: through the product's
index, five `optimize` calls after 64 new documents each took 114 to 310 s
under `disk` and 1.1 to 1.6 s under `memory`. Both keep a new memory
searchable before any build and neither holds a search up while one runs.
`disk` was the default at first, because this project ranks resident memory
above disk; once that write-path cost was measured the owner made `memory` the
default, and `disk` stays for a project whose memory is scarce.
Through `search_reranked` at the shipped defaults neither index moved a
ranking against fp32: nDCG@10 +0.0003 on the own corpus's 157 questions
(p = 0.76) and −0.0003 on 595 XQuAD-R questions (p = 0.68), the two indexes
identical on every question. A whole search took 1.03 and 1.05 of `memory`'s
time under `disk` (medians 1,371 against 1,287 ms and 2,339 against 2,236),
and rebuilding XQuAD-R's 13,014 documents took 424 s under `disk` and 30 s
under `memory`.
The table, the tuning and the write-path measurement are in
[the ADR](adr/0001-tech-selection.md#two-vector-indexes-both-half-precision-memory-by-default).

**Re-ingesting an unchanged tree appends no evidence, and costs seconds.** This
repository's 105 tracked Markdown and text files, cut into 4,090 sections,
through the release `pamin ingest` at `cf4a771` on four shared cores: six
unchanged re-ingests across two runs took 4.20 to 6.72 s each and appended no
evidence, and a one-word edit to prose wrote one to three sections, three when
it fell in a section split into parts. Re-importing the same 4,090 memories
with `pamin import` appended 4,090 evidence rows, 4 MB.

| | `import` | `ingest` | difference | change |
| --- | ---: | ---: | ---: | ---: |
| evidence rows an unchanged re-run appended | 4,090 | 0 | −4,090 | −100% |
| database growth from that re-run | +4.0 MiB | +0.0 MiB | −4.0 MiB | −100% |
| that re-run, wall (n = 1 each) | 2.52 s | 4.20 s | +1.68 s | +66.4%, inside ingest's own 4.20–6.72 s spread |
| cold, wall (n = 1 each) | 505.2 s | 565.1 s | +59.9 s | +11.9%, inside the 351.5–671.9 s spread |

Only the first two rows are a difference that holds: they are counts, the same
on every run. The timings are one sample each, and a cold ingest of this tree
ran 351.5, 671.9 and 565.1 s on three runs of the day. Nor is the cold pair the
same work: ingest keeps its topics out of the name index and owes them no
backfill, where import files 4,090 path-and-heading names and backfills each.
Server CPU excludes PostgreSQL's backends and includes whatever index upkeep
the server ran during an arm. Accuracy over ingested documents is not measured
-- no benchmark here holds long documents with questions about them -- and
nothing on the search path changed. Conditions, every arm and the raw rows are
in [benchmarks/results/ingest/](../benchmarks/results/ingest/ingest-cost-2026-10-23.md).

**Above this, nothing is measured.** The largest corpus here is 131,924
documents. A million and beyond is untested — not projected, not extrapolated,
untested — and the descriptor count is the first thing that would break: this
index is 2,111 segment files and a search holds 2,733 descriptors open, which
already exceeds the 1,024 a Linux process is given by default.

What was measured, how, and the conclusions that reversed on measurement are in
[the ADR](adr/0001-tech-selection.md), which is the
source of truth if it and this page ever disagree.

The old `PASSAGES` content-only versus named diagnostic is disabled: its retained
content-only project has no recorded embedding revision, so a newer model cache
could silently compare different vector spaces. Historical rows stay historical;
the named product index remains revision-bound. Reintroduce the diagnostic only
with a newly built and verified revision-bound content-only project.

`TIERS=1` and `ROUTES` need nothing beyond the default preparation, which
fetches only the embedder and the reranker. The Fast tier's preparation and
its two export checks were removed with that tier in `9cccff4`; its
[manifest](../benchmarks/results/retrieval/xquad-r-fast-model-artifacts.json)
stays as the record of what its historical rows ran.
