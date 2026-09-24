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

**Retrieval quality**, at the `fast` reranking tier, which was the default when
these were taken:

| corpus | group | nDCG@10 | recall@50 |
| --- | --- | --- | --- |
| MIRACL Swahili dev — 131,924 real passages, 482 queries, 5,092 human judgements | one language throughout | 0.7359 | 0.9494 |
| XQuAD-R — 13,014 sentences in eleven languages, 1,190 queries | query and answer in **different** languages | 0.6480 | 0.8960 |
| XQuAD-R | query and answer in the same language | 0.7495 | 0.9580 |

The default is now `accurate`, chosen on the paired comparison in
[cli.md](cli.md). The runs behind this table measured it too: MIRACL 0.7654,
XQuAD-R 0.6597 cross-lingual and 0.7835 same-language, recall unchanged because
a reranker reorders a shortlist and never changes it. The XQuAD-R pair is from
the later run under the fusion that ships, so it sits against 0.6114 with no
reranking rather than against the rows above.

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
| Påmin Memory, `fast` | 0.7359 | fused, then a cross-encoder |
| Påmin Memory, `accurate` (default) | **0.7654** | fused, then a larger cross-encoder |
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
the near-duplicate pair this project described them as** and the single weight
they share has never been swept apart. And the graph channel contributes
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
| the shipped search | **0.6834** | **0.8435** |
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

**A learned fusion head was fitted and does not generalise.** The `FEATURES`
arm (`pamin-engine/tests/features`) dumps every fused candidate with each
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
| `fast` picks ten, `accurate` orders them | −0.0284, p = 0.0001 | | −0.0047, p = 0.0015 |
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
so every query is still reranked; the learned route is a candidate for the
`speed` profile, not the default.

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
machine at this corpus size. A workspace that never asks for `accurate` never
loads the 570 MB reranker; `--rerank off` never loads either.

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
copy the runtime maps from disk -- written once beside the download; see
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
megabyte across two runs. What the copy leaves is mostly not the weights: a
bare runtime session adds 139 MB loading the `fast` reranker's download and
12 MB loading its copy, so most of the 268 is what a load holds besides its
session. The data file is the price, on disk rather than in memory -- larger
than the model it came from, because the packed weights are stored beside the
originals.

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

Every vector and every score is bit-identical: 18,432 embedding values over
eighteen texts and 65 `accurate` scores, over eight scripts, runs of spaces,
trailing and leading spaces, empty texts and one past both length limits --
and 65 `fast` scores, from its download. Before the trim the gap is wider than one vocabulary; `fastembed`'s
loader clones each tokenizer it configures and drops the original, which would
leave that much freed and not yet returned, but that was not measured
separately. Twenty rerank passes of sixteen pairs of about 200 tokens showed no speed
change: the per-round median moved between -22% and +9% against `fastembed`,
median -2.5%, while one arm's own rounds moved by up to 40%.

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
at the default depth — so under a millisecond, against the 99 ms
[the CLI reference](cli.md) gives a search with reranking off and 1,522 ms at
the default tier. The results
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
looked at, and it is now the only copy of every memory's text. The queue's rows
also still carry their subject twice, once in `payload` and once inside
`idempotency_key` — 59 MB and 57 MB on the synthetic queue, where the unique
index over the key was the largest index at 126 MB — which is the next thing
to narrow.

**Above this, nothing is measured.** The largest corpus here is 131,924
documents. A million and beyond is untested — not projected, not extrapolated,
untested — and the descriptor count is the first thing that would break: this
index is 2,111 segment files and a search holds 2,733 descriptors open, which
already exceeds the 1,024 a Linux process is given by default.

What was measured, how, and the conclusions that reversed on measurement are in
[the ADR](adr/0001-tech-selection.md), which is the
source of truth if it and this page ever disagree.
