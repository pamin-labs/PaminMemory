# Benchmarking against the field

How to produce a number that someone outside this project recognises, and why
the obvious comparison does not work.

Everything below was gathered from primary sources in September 2026. Each
figure carries its source so it can be rechecked; this field moves fast and
several of these numbers have already changed meaning once.

## What the field publishes

Agent-memory projects report **LLM-judge accuracy on conversational QA**:
LOCOMO, LongMemEval, DMR. An LLM answers questions from the memory system's
output and a second LLM grades the answers.

| Project | Benchmark | Claimed | LLM on write path | Local or API |
| --- | --- | --- | --- | --- |
| mem0 ([paper][mem0-paper], ECAI 2025) | LOCOMO | J 66.88%; graph variant 68.44% | Yes — extraction + update decision | API |
| mem0 ([site][mem0-site], 2026-09-16) | LOCOMO / LongMemEval | 92.5 / 94.4 | Yes | API |
| Zep / Graphiti ([paper][zep-paper]) | DMR; LongMemEval | DMR 94.8%; LongMemEval 71.2% (gpt-4o) | Yes — entity/edge extraction | API + Neo4j |
| Letta ([blog][letta]) | LoCoMo | 74.0% with plain files + grep | No extraction step | Framework |
| Memobase ([repo][memobase]) | LOCOMO | 75.78% | Yes — profile extraction | API |
| Cognee ([site][cognee]) | BEAM | 0.79 against 0.73 prior | Yes — LLM graph construction | API |
| MemPalace ([repo][mempalace]) | LongMemEval | R@5 96.6%, **NDCG@10 0.889** | No | Local |
| Supermemory ([site][supermemory]) | — | "#1", no figures published | Yes | API |

Every one of these is self-reported.

## Why these are not comparable to our numbers

Five reasons, any one of which is disqualifying:

1. **Different stage of the pipeline.** Those headline scores measure retrieval
   *plus an LLM reader plus an LLM judge*, compounded. `nDCG@10` measures
   ranking alone. Zep's own table shows how much the reader contributes: the
   same Zep retrieval scores 63.8% with `gpt-4o-mini` and 71.2% with `gpt-4o` —
   7.4 points of "memory quality" that is entirely the reader model.
2. **The judge is an instrument with known error.** An independent analysis of
   the LOCOMO harness ([gde03][locomo-critique]) reports the judge accepting
   **63% of intentionally wrong answers**.
3. **One gold document against many.** Where these projects do report recall,
   it is "is the one gold session in the top k" over roughly fifty candidates.
   MIRACL's Swahili dev split carries 5,092 judgements over 482 queries, 1.89
   relevant passages each, pooled over 131,924; XQuAD-R has eleven parallel
   relevant sentences per query. (Those judgements are binary, not graded -- an
   earlier draft of this page said graded, and the qrels file holds only 0 and
   1.)
4. **Haystack size differs by orders of magnitude.** ~50 sessions, or LOCOMO's
   16k–26k tokens, against MIRACL's 131,924-passage Swahili dev split.
5. **The benchmarks are contaminated.** The same analysis finds 6.4% of LOCOMO's
   answer key wrong and 56% of per-category comparisons inside noise, and
   reports that the standard harness silently drops its adversarial category --
   446 questions, 22.5% of the set -- while instructing the model never to
   abstain.

   An earlier version of this page said those 446 questions have a refusal for
   their only correct answer. Reading the file says otherwise: all 446 carry an
   `adversarial_answer` rather than an `answer`, and the value is a substantive
   answer -- "Sweden", "researching adoption agencies" -- in every case but two.
   What makes them adversarial is that the answer is implied rather than stated,
   not that it is absent. Dropping them still removes a fifth of the benchmark
   and the hardest fifth; it does not remove an abstention test, because there
   is not one to remove.

So: we cannot say this project beats mem0, and we cannot say it loses. The
quantities do not overlap. What is sayable is architectural — no LLM on the
write path, none at query time, local-first — plus one observation: **nobody in
this category publishes a retrieval-quality number that can be checked**, and
nobody publishes a cross-language result at all.

## The disputes are the point

Do not quote a competitor's headline without reading its audit. As of this
writing:

- **Zep audited mem0** ([blog][zep-rebuttal]): alleges misconfiguration and
  sequential rather than parallel search inflating Zep's reported latency.
- **mem0 audited Zep** ([issue #5][mem0-issue]): alleges an excluded adversarial
  category counted in the numerator but not the denominator, worth about 25
  points.
- The same Zep system is claimed at **84%, 75.14%, 65.99% and 58.44%** depending
  on who ran it. Neither party's figure survives the other's audit.
- **MemPalace's 96.6%** was shown by an independent teardown
  ([lhl/agentic-memory][mempalace-teardown]) to measure ChromaDB's default
  embeddings — the benchmark path imports almost nothing from the library it
  is named after. An independent reimplementation with none of its code scored
  93.8% R@5. Where the architecture is used it scores *worse*.
- **Letta's result is the one to take seriously**: a plain filesystem agent with
  grep and semantic search scored 74.0%, beating mem0's graph variant at 68.5%.
  If a filesystem wins, the benchmark is measuring context management rather
  than any memory mechanism.

## The one honest comparison available

**LongMemEval has a retrieval-only stage**, scored with Recall@k and NDCG@k by
`print_retrieval_metrics.py`, with no LLM anywhere. That is the same metric
family this project already reports.

Reference points to compare against, from the LongMemEval paper (Stella V5
1.5B, K=V+fact):

| granularity | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |
| --- | --- | --- | --- | --- |
| session | 0.732 | 0.620 | 0.862 | **0.652** |
| round | 0.644 | 0.498 | 0.784 | **0.536** |

And MemPalace's 0.889–0.938 NDCG@10, with the caveat above about what it
actually measured.

### Running it

Not yet run here. The shape it would take:

1. Fetch LongMemEval-S from [its repository][longmemeval]. Fetch rather than
   vendor, the way `crosslingual.rs` fetches XQuAD-R — licence, and size.
2. Ingest each question's session haystack with `pamin import` (about 10× cheaper
   per memory than looping `pamin write`), one project per question so the
   haystacks stay isolated.
3. Query with `Engine::search_reranked` — the entry point `pamin search` calls.
   Measuring the layer underneath it is the mistake this repository already
   made once; see the `pamin-dev` skill.
4. Emit results in LongMemEval's expected format and score with its own script,
   rather than reimplementing the metric. Reimplementing it is how a number
   ends up describing the harness instead of the product.

Cost: zero API calls, zero dollars. Compute is ingest plus search.

**What to expect, stated in advance so the result can falsify it**: the haystack
is about fifty sessions per question, so scores saturate — MemPalace's hybrid
already sits at R@10 99.8%. A high number here is a credential ("we can be
scored on your benchmark"), not evidence of quality. MIRACL and XQuAD-R remain
the numbers that discriminate.

### What we deliberately do not run

**LOCOMO QA accuracy.** It needs an LLM reader on the read path and an LLM judge
on the scoring path — 1,540 questions, averaged over ten runs in the mem0 paper,
so roughly 15,400 reader calls and as many judge calls. Tens of dollars a sweep,
low hundreds if baselines are re-run rather than quoted.

The dollar cost is not the reason to decline. The reason is that the resulting
number would depend on which reader model was picked, would be graded by an
instrument with a documented 63% false-accept rate against an answer key that is
6.4% wrong, and would require instructing the model never to say "I don't know"
— which is the opposite of what this project is for.

## Reference points on MIRACL, and the traps around them

This project reports nDCG@10 on MIRACL's Swahili dev split, so the numbers a
reader will reach for to interpret it are collected here, each traced to a
primary source. Every figure below is over the same 131,924-passage corpus and
the same `miracl-v1.0-sw` dev qrels, verified rather than assumed: the corpus
file decompresses to exactly 131,924 lines, and the qrels hold 5,092 judgements
over 482 queries, labels in {0, 1}, 910 positives, 1.89 per query.

| | nDCG@10 | source |
| --- | --- | --- |
| Pyserini BM25 baseline | 0.3826 | [Pyserini MIRACL v1.0 regressions][pyserini] |
| MIRACL paper, same baseline | 0.383 | [arXiv 2210.09984][miracl] Table 2 |
| BGE-M3 dense | 0.787 | [arXiv 2402.03216][bgem3] Table 1 |
| BGE-M3 sparse | 0.579 | same |
| BGE-M3 multi-vector (ColBERT) | 0.791 | same |
| BGE-M3 dense + sparse | 0.785 | same |
| BGE-M3 all three | 0.796 | same |

Four things that will bite anyone quoting these:

- **0.787 is not 0.786.** arXiv v1–v3 report 0.786; the model card records a
  2024-07-01 correction ("we mistakenly removed the passages that have the same
  id as the query"), and v4, v5 and the ACL camera-ready report 0.787. Cite v4
  or later.
- **The hybrid premium on Swahili is small.** Dense 0.787 to all-three 0.796 is
  +0.009. Anyone reaching for the hybrid number to make a gap look smaller is
  reaching for nine thousandths.
- **BGE-M3's own BM25 row says 0.351, not 0.3826.** They re-ran BM25 with the
  XLM-R tokenizer instead of the Lucene analyzer (their Appendix C.2). Both are
  legitimate; putting them in one table without saying which is not.
- **A MIRACL figure from the MTEB leaderboard is probably not comparable.** MTEB
  carries a `MIRACLRetrievalHardNegatives` variant whose corpus is the top 250
  documents per query from three retrievers — a pooled subset, not 131,924
  passages. Confirm the variant before using the number.

And one ceiling that applies to every row including this project's: MIRACL's
judgements come from pooling a 2022 ensemble's top ten. Anything relevant that
no pooled system surfaced counts as a miss for every system scored afterwards.
It does not break comparisons between the rows; it caps all of them together.

**What is not established**: the quality cost of the int8 export this project
runs against the fp32 weights the published figure used. No measurement of that
delta was found, and it is the single experiment that would explain part of the
gap rather than gesturing at it.

## What running it actually showed

LOCOMO, 199 questions drawn stratified from ten conversations, every arm
answering the same questions with the same Sonnet and embedding with the same
BGE-M3 ONNX file. The harness is in [benchmarks/](../benchmarks); the
conditions it holds fixed, and how each is asserted, are in its README.

Three of the arms are this project at different settings, because the first
question about any gain is whether it came from the thing you changed:

- **`pamin`** — `pamin import` and `pamin search` at the default `--limit 10`.
- **`pamin-wide`** — the same, at `--limit 30`. Nothing else differs, so what
  separates it from `pamin` is the size of the shortlist and nothing else.
- **`pamin-ledger`** — each session imported under its own `--valid-from` and
  consecutive turns linked, so the graph channel can reach the rest of an
  exchange. Still no model on the write path.

mem0 and MemPalace each get two arms for the same reason, so that neither side
is alone in being allowed a wider shortlist.

### The shortlist every arm actually got

One column has to come before the results, because getting it wrong invalidates
a comparison silently. mem0's search keyword is `top_k`, not `limit`, and it
defaults to 20; unknown keywords go into `**kwargs` and are dropped without an
error. The first run of these arms passed `limit`, so **both mem0 arms ran at
20** — the narrow one wider than it claimed, the wide one narrower. Every one of
the 199 questions returned exactly 20 passages, which is how it was caught. The
arm now passes `top_k` and checks the count it gets back, the way the MemPalace
arm checks that it embedded through the shared endpoint.

The figures below are from that first run and are labelled with the shortlist
each arm was actually at, not the one it was asked for.

| arm | shortlist asked | shortlist received |
| --- | --- | --- |
| `pamin`, `pamin-ledger`, MemPalace, BM25 | 10 | 10 |
| `pamin-wide`, MemPalace at 30 | 30 | 30 |
| mem0 | 10 | **20** |
| mem0 at `limit 30` | 30 | **20** |

### Accuracy

| arm | shortlist | accuracy | against `pamin` |
| --- | --- | --- | --- |
| BM25, no memory system | 10 | 0.427 | |
| `pamin` | 10 | 0.518 | |
| `pamin-ledger` | 10 | 0.523 | +0.005, **p = 1.00** |
| MemPalace | 10 | 0.558 | |
| mem0 | 20 | 0.603 | |
| mem0, second run | 20 | 0.598 | |
| MemPalace at 30 | 30 | 0.623 | |
| `pamin-wide` | 30 | **0.628** | +0.110, **p = 0.002** |

Paired McNemar on the discordant questions, which is the test that matters
when every arm answers the same set:

| | this one right | that one right | p |
| --- | --- | --- | --- |
| `pamin-wide` vs BM25 | 52 | 12 | **0.0000** |
| MemPalace at 30 vs BM25 | 56 | 17 | **0.0000** |
| mem0 vs BM25 | 54 | 19 | **0.0001** |
| `pamin-wide` vs `pamin` | 35 | 13 | **0.0021** |
| MemPalace at 30 vs `pamin` | 39 | 18 | **0.0075** |
| `pamin` vs BM25 | 42 | 24 | **0.0356** |
| mem0 vs `pamin` | 40 | 23 | **0.0430** |
| MemPalace at 30 vs MemPalace | 26 | 13 | 0.053 |
| `pamin-wide` vs MemPalace | 35 | 21 | 0.081 |
| `pamin-wide` vs mem0 | 31 | 26 | 0.597 |
| `pamin-wide` vs MemPalace at 30 | 31 | 30 | 1.000 |
| `pamin-ledger` vs `pamin` | 21 | 20 | 1.000 |
| mem0 second run vs mem0 | 20 | 21 | 1.000 |

**Four findings, and two of them are negative.**

**There is a noise floor, and it is large.** The last row is not a comparison
between two configurations. Both mem0 arms ran at the same shortlist, on the
same data, with the same settings — it is the same experiment twice. The totals
differ by 0.005, which looks like reassuring stability, but **41 of the 199
questions changed answer between the two runs**. About a fifth of the benchmark
is not repeatable for an arm whose write path distils conversations with a
model, because the distillation is not the same twice. That number is the scale
against which every other gap on this page has to be read, and two of them do
not survive it.

This project has no such floor on the write side: there is no model there, so
`pamin-wide` reads a store that `pamin` built, byte for byte. That is a
property worth having, but it is a property of reproducibility, not of accuracy.

Two things are being separated here and only one of them is mem0's. Identical
mem0 runs flip 41 questions, which is run-to-run variance. But `pamin` against
`pamin-ledger` — both deterministic — also flips 41, from a configuration change
worth 0.005. So LOCOMO at this size churns about a fifth of its questions under
almost any perturbation, and a net difference of five or ten questions is inside
that whoever produces it.

**The ledger bought nothing.** Twenty-one questions it got right that the flat
arm missed, twenty the other way. Not "a small gain" — no gain. It does not
rescue the category it was built for either: temporal goes 0.412 to 0.500,
which sounds like something until the wide arm reaches 0.529 with no validity
intervals and no edges at all. Either this benchmark does not ask the question
a ledger answers — LOCOMO asks when something happened, not whether a fact was
revised — or the idea is weaker than the design assumes. Those two are not
distinguished by anything measured here, and distinguishing them needs a
benchmark that asks what changed rather than what happened.

**The shortlist was worth eleven points, and the default was leaving them on
the floor.** That is the largest single retrieval gain in this repository and
it is a configuration change.

**Against both other systems the result is a tie.** `pamin-wide` against mem0
is 31 to 26, p = 0.597; against MemPalace at the same shortlist it is 31 to 30,
p = 1.000. Fifty-odd questions disagree and the net is five, then one — the
churn described above with no direction in it. Neither is evidence of anything.

Two earlier readings of this comparison were wrong and are withdrawn. The first
was a prediction, that mem0 would gain from a wider shortlist and pull ahead;
the second was the conclusion drawn when it did not move, that the widening was
worthless to it. Both were about an arm that never widened. What the pair
actually measures is repeatability, and it is the more useful of the two.

The one comparison that was never confounded is `pamin` against mem0 at what
mem0 really ran: 23 to 40 against us, p = 0.043, with mem0 holding twice the
shortlist. Read as written that is a real loss; read as a like-for-like it is
not a comparison at all. It is the reason the wide arms exist.

By question type the three systems are not close anywhere; they are opposite:

| | BM25 | `pamin` | `pamin-ledger` | MemPalace | mem0 | mem0 (2nd) | MP 30 | `pamin-wide` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-hop | 0.241 | 0.517 | 0.483 | **0.724** | 0.690 | 0.621 | 0.690 | 0.586 |
| temporal | 0.324 | 0.412 | 0.500 | 0.382 | 0.676 | **0.765** | 0.382 | 0.529 |
| open-domain | 0.222 | **0.333** | 0.222 | 0.111 | **0.333** | 0.222 | 0.222 | 0.222 |
| single-hop | 0.682 | 0.718 | 0.706 | 0.753 | 0.765 | 0.765 | **0.835** | 0.824 |
| adversarial | 0.167 | 0.238 | 0.262 | 0.286 | 0.214 | 0.190 | **0.429** | **0.429** |

Two splits are wide enough to clear the floor. mem0 leads **temporal** by
fifteen to twenty-four points over every arm here; this project and MemPalace
lead **adversarial** — questions whose answer is implied rather than stated — by
about the same over mem0, which is what distilling a conversation into rewritten
facts costs you when the question is about what was never said. The totals tie
because those cancel, which is a different fact from "the systems perform alike"
and should not be reported as one.

Nine open-domain questions is too few to say anything, and it is listed only so
the column is not quietly dropped.

### What each arm spends

Accuracy is half of a claim that says "less". The other half:

| arm | write: LLM calls | write: seconds | query: prompt tokens | passages |
| --- | --- | --- | --- | --- |
| BM25 | 0 | 0 | 547 | 10 |
| `pamin` | **0** | 292 | 557 | 10 |
| `pamin-wide` | **0** | — | 1,511 | 30 |
| MemPalace | 21 | 462 | 1,756 | 10 |
| MemPalace at 30 | 20 | 476 | 5,133 | 30 |
| mem0 | **272** | 3,999 | **676** | 20 |

Ten conversations, 5,882 turns. Prompt tokens are counted with
`cl100k_base` — not the model's own tokenizer, so the absolute figures are
approximate, but every arm is counted the same way and what this needs is the
ratio.

**The two halves point in opposite directions, and that is the whole result.**
mem0 spends 272 model calls and an hour of model time putting ten
conversations in, which `pamin` does in five minutes with none. But mem0
stores rewritten facts, so what it hands the reader afterwards is compact —
676 tokens a question against `pamin-wide`'s 1,511. One arm pays once; the
other pays on every question, forever.

So there is a crossing point, and it is worth stating rather than leaving each
side to quote its favourite half. `pamin-wide` costs 835 more prompt tokens a
question than mem0. Against mem0's write side — $27.83 as reported, less
roughly $9 of per-call overhead this environment adds to every call, so about
$19 of marginal cost — and Sonnet's list input price, the two meet at roughly
seven and a half thousand questions across those ten conversations, or about
750 questions asked of one conversation's memory.

That estimate rests on two assumptions, both stated so they can be attacked:
the overhead subtraction, and list pricing. What does not rest on either is
the shape — a one-time cost against a per-question one — and which side each
system is on.

### Memory and disk

Resident memory as PSS over each arm's own process tree, one conversation:

| arm | resident | on disk | where the embedder lives |
| --- | --- | --- | --- |
| BM25 | 16 MB | — | nowhere; no model |
| `pamin` | **2,088 MB** | 13 MB | inside its own server, so inside this figure |
| MemPalace | 29 MB | 1 MB | **outside**, 1,145 MB wherever it runs |
| mem0 | 177 MB | 2 MB | **outside**, 1,145 MB wherever it runs |

One conversation means `conv-26`, 419 turns — the second smallest of the ten,
so these are not the figures for a mean-sized one.

PSS rather than RSS because PostgreSQL's backends share one pool of buffers
and RSS charges it to each of them — 326 MB summed as RSS against 78 MB as
PSS, for the same ten processes. The `pamin` figure includes that cluster.

The embedder column is an architectural difference, not an accounting one, and
folding it into a single number would hide it. `pamin` loads BGE-M3 into its
own process. mem0 calls out for embeddings — here to the shared endpoint, in
its default deployment to a hosted API. So mem0 running its embedder locally
is about 1,322 MB against `pamin`'s 2,088, and mem0 using a hosted one is
177 MB locally plus a bill.

### What this does not establish

- **Nothing about the ledger.** The only arm that used versions and validity
  intervals gained nothing, and that is a fact about this benchmark as much as
  about the feature. It is not evidence the feature works, and it is not
  evidence it does not.
- **Nothing about mem0 at a shortlist of its own choosing.** Both its arms
  ran at 20 because of the dropped keyword above. Re-running them at 10 and 30
  costs another $27.83 and about an hour of model time, because the arm clears
  its store before each conversation and so cannot reuse the one it built.
- **Nothing about any difference smaller than the noise floor.** Forty-one of
  199 questions moved between two identical mem0 runs. Anything at that scale
  here is unmeasured, not measured-as-equal — the two are different claims and
  only the first is supported.
- **Nothing about scale.** Ten conversations, 5,882 turns, 588 on average.
- **Nothing that travels between machines except accuracy, calls and tokens.**
  Latency, resident memory and wall-clock are properties of four shared cores.

## Keeping this current

This page goes stale faster than anything else in the repository: mem0's own
LOCOMO figure moved from 66.88 to 92.5 within a year with no published bridging
methodology. When refreshing it:

- Re-fetch each figure from the primary source and update the date at the top.
- Check whether an audit has appeared since. The audits are more informative
  than the claims.
- Do not add a number to the README from this page. The README carries figures
  this project measured itself; these are other people's claims about other
  people's software, and they belong behind this explanation of why they do not
  line up.

[mem0-paper]: https://arxiv.org/abs/2504.19413
[mem0-site]: https://mem0.ai/research
[mem0-issue]: https://github.com/getzep/zep-papers/issues/5
[zep-paper]: https://arxiv.org/abs/2501.13956
[zep-rebuttal]: https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/
[letta]: https://www.letta.com/blog/benchmarking-ai-agent-memory/
[memobase]: https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md
[cognee]: https://www.cognee.ai/ai-memory-evals-0825
[mempalace]: https://github.com/MemPalace/mempalace/blob/develop/benchmarks/BENCHMARKS.md
[mempalace-teardown]: https://github.com/lhl/agentic-memory/blob/main/ANALYSIS-mempalace.md
[supermemory]: https://supermemory.ai/research/longmembench/
[locomo-critique]: https://dev.to/gde03/the-ai-memory-benchmark-everyone-quotes-forbids-saying-i-dont-know-o1n
[longmemeval]: https://github.com/xiaowu0162/LongMemEval
[pyserini]: https://github.com/castorini/pyserini/blob/master/docs/experiments-miracl-v1.0.md
[miracl]: https://arxiv.org/abs/2210.09984
[bgem3]: https://arxiv.org/abs/2402.03216
