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
5. **The benchmarks are contaminated.** The same analysis finds 446 LOCOMO
   questions (22.5%) whose only correct answer is a refusal, silently dropped by
   the standard harness while the model is instructed never to abstain; 6.4% of
   the answer key wrong; and 56% of per-category comparisons inside noise.

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
