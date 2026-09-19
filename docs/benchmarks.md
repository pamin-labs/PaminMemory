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
| MemPalace ([repo][mempalace]) | LongMemEval | R@5 96.6%, **NDCG@10 0.889** | Not in the mode it publishes; **yes by default** | Local |
| Supermemory ([site][supermemory]) | — | "#1", no figures published | Yes | API |

Every one of these is self-reported.

**One of those "No"s needs a footnote, because this page measured the opposite.**
MemPalace's headline is "96.6% R@5 **raw** — zero API calls", and raw is real:
`mempalace init --no-llm` runs heuristics only. But it is not the default. As of
3.10.0 the CLI says so itself — `--llm` is "DEPRECATED — LLM-assisted entity
refinement is now ON by default ... pass `--no-llm` to opt out". The arm here
ran the default and measured 20 model calls and $1.15 to ingest ten
conversations. Both statements are true of different configurations, and a
table with one column for "LLM on write path" cannot hold that, so the column
now says which.

Worth stating what this is not: the arm did not switch an LLM on. It passed
`--accept-external-llm`, which only waives the consent prompt that fires when
one is already configured.

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

   This page has described that category wrongly twice. It first said the 446
   questions have a refusal for their only correct answer; all 446 carry an
   `adversarial_answer` rather than an `answer`, and the value is substantive
   in every case but two. It then said the answer is implied rather than
   stated, which is also wrong.

   Following each question to the turn its own `evidence` field names: **332 of
   the 446, 74%, attribute to one speaker something the other speaker said**,
   and the key gives that other speaker's content as correct.

   | question | answer key | the turn it cites |
   | --- | --- | --- |
   | What country is **Melanie's** grandma from? | Sweden | *Caroline*: "a gift from my grandma in my home country, Sweden" |
   | What instrument does **Caroline** play? | clarinet and violin | *Melanie*: "Yeah, I play clarinet!" |
   | Did **Caroline** make the black and white bowl? | Yes | *Melanie*: "I made this bowl in my class" |

   So the category rewards a pipeline that ignores attribution and penalises
   one that answers "no record of that" -- which, for the question as asked, is
   the better answer. Dropping it still removes a fifth of the benchmark, but
   what it removes is not an abstention test and not a hardness test; it is a
   test of whether a system will answer about the wrong person.

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

  There is a simpler problem with that figure, measured here rather than cited.
  MemPalace's own benchmark doc defines R@5 as whether the labelled session is
  inside the top five. Run a plain BM25 keyword search over the same haystacks,
  with no memory system of any kind, and it scores **96.6%** — the published
  headline, to the digit. The metric is saturated: about fifty candidate
  sessions, and "is the gold one in the top five" is not a question that
  separates anything. See [what running it showed](#longmemeval-the-published-metric-is-saturated).
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

**A supersession test built out of LongMemEval's `knowledge-update` category.**
This one was designed and then abandoned on its own premise check, which is
worth recording because the category looks like exactly the benchmark this
project's ledger needs.

All 78 of those questions carry exactly two gold sessions, dated, and the shape
is right: one session says "a personal best of 27:12", a later one says
"hoping to beat my personal best of 25:50", and the answer is 25:50. That
invites a metric costing nothing to run -- does the system rank the revision
above the value it superseded? -- and no reader or judge is needed for it.

It needs a label saying which of the two sessions holds the current value, and
the data does not support deriving one. Taking the answer's distinguishing
tokens, its numbers where it has them:

| where the current value appears | of 78 |
| --- | --- |
| only in the newer gold session | 26 (33%) |
| **in both gold sessions** | **39 (50%)** |
| only in the older gold session | 5 (6%) |
| in neither | 8 (10%) |

Half the time the current value is stated in both sessions, so "the newer one
is the revision" is not a label, it is a guess that would be wrong often enough
to produce whatever result was wanted. Constructing the labels needs a model
reading each pair, which makes the metric neither free nor independent of an
instrument -- and a labelling pass would itself need validating before anything
measured against it meant anything.

So the ledger still has no benchmark, and this is now a specific gap rather
than a vague one: what is missing is a corpus where supersession is annotated,
not inferred.

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

### What the mem0 arms were first measured with

Two conditions were wrong in the first run of the mem0 arms, both silent, both
found while writing the results up rather than while running them. mem0 has
since been re-measured with both corrected; this section says what was wrong so
the withdrawn figures are not quietly replaced.

**The shortlist.** mem0's search keyword is `top_k`, not `limit`, and it
defaults to 20; unknown keywords go into `**kwargs` and are dropped without an
error. Both mem0 arms therefore ran at 20 — the narrow one wider than it
claimed, the wide one narrower. Every one of the 199 questions returned exactly
20 passages, which is how it was caught. The arm now passes `top_k` and checks
the count it gets back.

**The retriever.** mem0's BM25 channel lemmatises on both sides: every memory
is stored with a `text_lemmatized` field and every query is lemmatised before
the keyword search. That path needs the `mem0ai[nlp]` extra, which a plain
`pip install mem0ai` does not bring, and without it `lemmatize_for_bm25`
returns its input unchanged. So the first mem0 figures were taken with half of
a hybrid retriever turned off, announced on one log line and nowhere else.

Both are the same rule: an arm asserts every premise it reports, and opening
the other side's budget covers the extras its own documentation installs, not
only its settings.

MemPalace was then audited the same way, and comes back clean. Its nine
optional extras are hardware backends (`coreml`, `gpu`, `dml`), alternative
vector stores (`milvus`, `pgvector`), binary document readers (`extract`, and
this corpus is Markdown), dev tooling, and `spellcheck`. The last was the only
candidate -- `autocorrect` is indeed not installed -- but the module that uses
it is `normalize`, which converts message transcripts, and the searcher never
imports it. Nothing MemPalace ships behind an extra gates a retrieval channel
the way mem0's `[nlp]` gated its keyword side.

The re-measurement ingests once per conversation and queries that store at both
shortlists. mem0 clears its store before each conversation, so running two arms
separately pays its $27.77 write bill twice; sharing the ingest pays it once and
has the better property besides — the two arms then read identical memories, so
what separates them is the shortlist alone.

### Accuracy

Every arm at the shortlist it actually received, which for every arm here is
now the one it asked for:

| arm | shortlist | accuracy |
| --- | --- | --- |
| BM25, no memory system | 10 | 0.427 |
| `pamin` | 10 | 0.518 |
| `pamin-ledger` | 10 | 0.523 |
| mem0 | 10 | 0.538 |
| MemPalace | 10 | 0.558 |
| mem0 | 30 | 0.583 |
| MemPalace | 30 | 0.623 |
| `pamin-wide` | 30 | **0.628** |

Paired McNemar on the discordant questions, which is the test that matters when
every arm answers the same set. Comparisons across different shortlists are
listed but decide nothing:

| | this one right | that one right | p |
| --- | --- | --- | --- |
| `pamin-wide` vs BM25 | 52 | 12 | **0.0000** |
| MemPalace at 30 vs BM25 | 56 | 17 | **0.0000** |
| mem0 at 30 vs BM25 | 50 | 19 | **0.0002** |
| `pamin-wide` vs `pamin` | 35 | 13 | **0.0021** |
| MemPalace at 30 vs `pamin` | 39 | 18 | **0.0075** |
| `pamin` vs BM25 | 42 | 24 | **0.0356** |
| *`pamin-wide` vs mem0 at 10* | *40* | *22* | *0.030, different shortlists* |
| **`pamin-wide` vs mem0 at 30** | 33 | 24 | **0.289** |
| **`pamin-wide` vs MemPalace at 30** | 31 | 30 | **1.000** |
| **MemPalace at 30 vs mem0 at 30** | 38 | 30 | **0.396** |
| **`pamin` vs mem0 at 10** | 30 | 34 | **0.708** |
| **`pamin` vs MemPalace at 10** | 28 | 36 | **0.382** |
| `pamin-ledger` vs `pamin` | 21 | 20 | 1.000 |
| mem0 at 20, run twice | 21 | 20 | 1.000 |

**Five findings, and three of them are negative.**

**There is a noise floor, and it is large.** The last row is not a comparison
between two configurations. The shortlist bug left both mem0 arms at the same
setting, so it is one experiment run twice: totals 0.603 and 0.598, and **41 of
199 questions answered differently**. A deterministic configuration change worth
0.005 — `pamin` against `pamin-ledger` — moved the same 41, so the churn is
LOCOMO's at this size as much as it is mem0's nondeterminism. A net of five or
ten questions is inside it whoever produces it. That accident is the most useful
pair on this page, and the rule from it is to run one arm twice before comparing
two.

This project has no such floor on the write side: there is no model there, so
`pamin-wide` reads a store that `pamin` built, byte for byte. That is a property
of reproducibility, not of accuracy, but it is the reason only one side of this
comparison has to be run twice to be believed.

**Giving mem0 its missing retriever changed nothing measurable, and the
prediction that it would was wrong.** Written down before the re-run: restoring
the lemmatiser would raise mem0's figures, so the tie might become a loss. At
thirty passages *and* with the full retriever, mem0 reaches 0.583 against the
0.603 it scored at twenty passages with the retriever half off — slightly lower,
14 against 18 discordant, p = 0.597. The extra was worth nothing here. It was
still wrong to have measured without it, and the disclosure stands whichever way
the number moved.

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

**At a matched shortlist, all three systems tie.** At thirty passages:
`pamin-wide` 0.628, MemPalace 0.623, mem0 0.583, and no pair separates —
33 to 24 against mem0 (p = 0.289), 31 to 30 against MemPalace (p = 1.000), and
MemPalace against mem0 38 to 30 (p = 0.396). At ten passages: MemPalace 0.558,
mem0 0.538, `pamin` 0.518, and again nothing separates. Fifty-odd questions
disagree in each pair and the nets are inside the floor above. This project is
numerically first at thirty passages, which is not the same as being ahead and
is not reported as one.

Two earlier readings are withdrawn outright. The first was a prediction, that
mem0 would gain from a wider shortlist and pull ahead; the second was the
conclusion drawn when it appeared not to, that widening was worthless to it.
Both were about an arm that never widened. The unconfounded version of that
question now exists, and mem0 does gain from the wider shortlist: 0.538 to
0.583, 21 questions to 12, though at p = 0.163 even that is not established.

By question type the three systems are not close anywhere; they are opposite:

| | BM25 | `pamin` | `pamin-ledger` | mem0 10 | MemPalace 10 | mem0 30 | MemPalace 30 | `pamin-wide` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi-hop | 0.241 | 0.517 | 0.483 | 0.483 | **0.724** | 0.655 | 0.690 | 0.586 |
| temporal | 0.324 | 0.412 | 0.500 | 0.706 | 0.382 | **0.735** | 0.382 | 0.529 |
| open-domain | 0.222 | **0.333** | 0.222 | 0.222 | 0.111 | 0.222 | 0.222 | 0.222 |
| single-hop | 0.682 | 0.718 | 0.706 | 0.718 | 0.753 | 0.729 | **0.835** | 0.824 |
| adversarial | 0.167 | 0.238 | 0.262 | 0.143 | 0.286 | 0.190 | **0.429** | **0.429** |

Two splits are wide enough to clear the floor, and both reproduce across mem0's
two independent runs, which is what makes them worth stating at all.

**Temporal is mem0's**, by about twenty points at either shortlist: 0.706 and
0.735 against this project's best of 0.529. It scored 0.676 and 0.765 in the
earlier pair too, so this is not the churn.

**Adversarial goes to this project and MemPalace, 0.429 against mem0's 0.143
and 0.190 — and it should not be counted as a win.** Seventy-four per cent of
that category asks about the wrong speaker, as set out at the top of this page,
and the answer key rewards replying with the other speaker's content anyway.
Splitting the forty-two sampled questions on exactly that:

| | n | `pamin-wide` | MemPalace 30 | mem0 30 |
| --- | --- | --- | --- | --- |
| the question names the wrong speaker | 33 | 0.485 | 0.485 | 0.242 |
| the question names the right speaker | 9 | 0.222 | 0.222 | 0.000 |

Four fifths of the category is the first row, and what separates the arms there
is that mem0 answers "no record of that" and is marked wrong for it. mem0
distils facts against a `user_id` and filters by it, so a question about
Melanie does not reach Caroline's memories. That is the behaviour a memory
product should have. Returning raw turns and letting the reader answer from
whichever one matched is the behaviour this project has, and here it scores
higher.

The nine questions that name the right speaker are the ones that would have
said something, and nine is too few to say it. So this category is reported and
then set aside: it is not evidence for this project, and the earlier reading of
it — that losing it is what distilling a conversation costs — is withdrawn.

That leaves one split that survives scrutiny, and it is mem0's. The totals tie
because temporal and adversarial cancel — but only one of those two is a real
difference between the systems, and it is not this project's.

Nine open-domain questions is too few to say anything, and it is listed only so
the column is not quietly dropped.

### What each arm spends

Accuracy is half of a claim that says "less". The other half, over the same ten
conversations and 5,882 turns.

**The write side, paid once:**

| arm | LLM calls | model seconds | cost | texts embedded | on disk |
| --- | --- | --- | --- | --- | --- |
| BM25 | 0 | 0 | $0 | 0 | 0 MB |
| `pamin` | **0** | **0** | **$0** | 0, in-process | 13 MB |
| `pamin-ledger` | **0** | **0** | **$0** | 0, in-process | — |
| MemPalace | 20 | 286 | $1.15 | 1,668 | 1 MB |
| mem0 | **272** | **3,913** | **$27.77** | 6,335 | 5 MB |

Ingest wall-clock: 356 s for `pamin`, 520 s for MemPalace, 4,121 s for mem0.
mem0's figure covers one ingest serving both of its shortlists — it clears its
store before each conversation, so measuring its two arms separately would have
paid that bill twice.

**The query side, paid on every question:**

| arm | passages | context bytes | prompt tokens |
| --- | --- | --- | --- |
| BM25 | 10 | 1,728 | 547 |
| `pamin` | 10 | 1,736 | 557 |
| `pamin-ledger` | 10 | 1,741 | ~557 |
| mem0 | 10 | 1,403 | ~384 |
| MemPalace | 10 | 7,475 | 1,756 |
| `pamin-wide` | 30 | 5,154 | **1,511** |
| mem0 | 30 | 4,224 | **~1,017** |
| MemPalace | 30 | 22,350 | 5,133 |

Context bytes are measured. Prompt tokens are counted with `cl100k_base` — not
the model's own tokenizer, so absolute figures are approximate, and every arm is
counted the same way because what this needs is the ratio. The four marked `~`
are not counted at all: they are derived from the measured bytes at mem0's own
ratio of 4.46 bytes per token, and `pamin-ledger`'s from `pamin`'s.

Latency is deliberately absent from that table. The accuracy run records a
`recall_seconds`, and it is not a comparison: it timed this project through
`su ubuntu -c "pamin ... search ..."` — two process spawns and a socket round
trip — and timed mem0 as an in-process library call, while ten arms, an
embedding endpoint and a PostgreSQL cluster shared four cores. It reported
170 ms against mem0's 106 and the obvious reading of that is wrong.

### Latency, timed at the same layer

Re-measured with [benchmarks/latency.py](../benchmarks/latency.py): every arm at
the boundary an application actually calls, every unit warmed first, nothing
else on the machine, and each arm run a second time last to show the order is
not in the number. Same 199 questions, same corpus.

| retrieval call | p50 @10 | p50 @30 | p95 @30 |
| --- | --- | --- | --- |
| this project, socket round trip | **29 ms** | **28 ms** | 46 ms |
| MemPalace, `search_memories` | 41 ms | 63 ms | 84 ms |
| mem0, `search` | 94 ms | 90 ms | 114 ms |
| this project, one CLI invocation | 43 ms | 42 ms | 61 ms |
| this project, that CLI behind `su` — *what the accuracy run timed* | 48 ms | 48 ms | 66 ms |
| *of which* mem0's embedding HTTP call | *20 ms* | *20 ms* | *22 ms* |

Two of the three figures the accuracy run produced were wrong, and in opposite
directions. It reported 170 ms here and 830 ms for MemPalace: this project was
timed behind a `su` and a CLI process, MemPalace behind a Python interpreter
starting and a package importing per query. Correcting both narrows this
project's lead over MemPalace from twenty-eight fold to 1.4. `su` is 5 ms of
the original figure and the process spawn 13; the rest was contention and cold
indexes.

One property does survive the correction and is worth naming: widening from
ten passages to thirty costs this project nothing measurable (29 to 28 ms)
where it costs MemPalace half as much again (41 to 63 ms).

### But retrieval is not what a caller waits for

Retrieval latency answers the wrong question on its own. What an agent waits
for is retrieval **plus** the model call that reads the passages, and that is
the term the arms differ on most — 557 tokens here at ten passages against
5,133 for MemPalace at thirty. So the reader was measured across that whole
range, twelve calls at each size:

| context handed to the reader | reader call, median |
| --- | --- |
| 541 tokens | 5.62 s |
| 1,001 tokens | 5.28 s |
| 1,495 tokens | 5.20 s |
| 5,117 tokens | 4.86 s |

It does not move. Nine times the context, and the median falls rather than
rises — which is noise, and the point: across the range these systems actually
produce, the reader costs about five seconds regardless.

**So end to end, with a model reading the results, the three systems are
indistinguishable.** Retrieval is roughly one per cent of the wait, and a
sixty-millisecond difference is not something a user experiences:

| arm, thirty passages | retrieval | reader | total | retrieval's share |
| --- | --- | --- | --- | --- |
| this project | 28 ms | ~5.2 s | ~5.2 s | 0.5% |
| MemPalace | 63 ms | ~4.9 s | ~5.0 s | 1.3% |
| mem0 | 90 ms | ~5.3 s | ~5.4 s | 1.7% |

That does not make the retrieval figures pointless; it says where they count.
A memory system feeding an agent's own context adds its retrieval latency to a
call that was happening anyway, and there 29 ms against 94 ms is the entire
marginal cost. Where a separate reader stands between the memory and the
answer, it is not.

Two limits on the reader figure. It is a CLI against a hosted API, so its five
second floor is process start and network rather than prefill — a local reader
would shift the balance back toward retrieval. And it says nothing about
prompts an order of magnitude larger, where prefill does dominate.

The one latency that is not swamped is on the other side: 356 s to ingest ten
conversations here against 4,121 s for mem0. Nothing hides a difference of an
hour.

**The two halves point in opposite directions, and that is the whole result.**
mem0 spends 272 model calls and an hour of model time putting ten conversations
in, which `pamin` does in six minutes with none and at no cost. But mem0 stores
rewritten facts, so what it hands the reader afterwards is compact — 1,017
tokens a question against `pamin-wide`'s 1,511 at the same thirty passages. One
arm pays once; the other pays on every question, forever.

So there is a crossing point. At thirty passages each, `pamin-wide` costs 494
more prompt tokens a question. Against mem0's write side — $27.77 as reported,
less roughly $9 of per-call overhead this environment adds to every call, so
about $19 of marginal cost — and Sonnet's list input price:

| | break-even |
| --- | --- |
| thirty passages each, against mem0's full $27.77 | ~1,900 questions per conversation |
| thirty passages each, against ~$19 marginal | ~1,270 questions per conversation |
| ten passages each | 3,600 – 5,400 questions per conversation |

An earlier version of this section put the crossing at about 750 questions. That
figure compared this project's thirty passages against mem0's twenty, which was
the shortlist bug, and so charged this side for context the other was not
carrying.

At LOCOMO's own density — twenty questions asked of one conversation — the
totals are $0.09 for `pamin-wide` against $2.84 for mem0, a factor of 31.

That estimate rests on two assumptions, both stated so they can be attacked: the
overhead subtraction, and list pricing. What does not rest on either is the
shape — a one-time cost against a per-question one — and which side each system
is on.

### MemPalace with its LLM off, which is the mode it publishes

MemPalace's default refines entities with a model; `init --no-llm`, the mode
its 96.6% is measured in, does not. That makes the second one the only
third-party arm here with the same architectural commitment as this project --
nothing on the write path decides what a conversation means -- and the only
thing that can say whether the 20 calls the default spends buy anything.

Both modes, both shortlists, the same 198 paired questions. The arm asserts its
own premise: zero model calls during ingest, because an arm named `raw` that
quietly used one would be the default measured twice.

| MemPalace | accuracy @10 | accuracy @30 | ingest | model calls | cost |
| --- | --- | --- | --- | --- | --- |
| default, LLM refinement on | 0.561 | **0.626** | 545 s | 20 | $1.15 |
| `--no-llm`, as published | **0.571** | 0.606 | **219 s** | **0** | **$0** |

**The 20 calls buy nothing measurable.** At ten passages the raw mode is
*ahead* — 8 questions to 10 discordant, p = 0.81. At thirty the default is
ahead by 17 to 13, p = 0.58. Neither separates, the two modes embed the same
1,668 texts and hand the reader the same number of tokens, and turning the
model off makes ingest two and a half times faster and free.

**And it is bad news for one of this project's claims.** With its model off,
MemPalace is the like-for-like peer, and it ties:

| same shortlist | this project | MemPalace `--no-llm` | discordant | p |
| --- | --- | --- | --- | --- |
| ten passages | 0.520 | **0.571** | 25 / 35 | 0.245 |
| thirty passages | **0.631** | 0.606 | 32 / 27 | 0.603 |

So "no language model on the write path" is an architecture this project
shares rather than owns, and at equal shortlists it buys equal accuracy in
someone else's implementation too. What remains specific to this project,
measured against that peer, is narrower and should be stated as such: a store
that reproduces itself byte for byte, a prompt 3.4x more compact at thirty
passages (1,511 tokens against 5,138), and retrieval at 28 ms against 63 ms.

One question of the 199 is missing from the wide raw arm — MemPalace's
re-ingest of the largest conversation stopped responding on the fill-in pass —
so every figure in this section is over the 198 both arms answered.

### LongMemEval: the published metric is saturated

LOCOMO scores depend on which model reads the passages and which grades them,
which is why nothing on this page compares an absolute LOCOMO number with a
competitor's. LongMemEval's retrieval stage has no such problem: it asks
whether the gold session is in the top k, and no model generates or judges
anything. It is the one figure here that could sit beside a published one.

So it was measured at the definition the field publishes — MemPalace's
benchmark doc defines R@5 as whether the labelled session is inside the top
five — and at the stricter one this page has been reporting. 59 questions drawn
stratified from LongMemEval-S's 500, a proportional sample of the same six
types; 35 of them carry more than one gold session, which is why the two
metrics differ at all.

| session retrieval, no model anywhere | BM25 | this project |
| --- | --- | --- |
| `recall_any@5` — **the published metric** | **0.9661** | 0.9831 |
| `recall_any@10` | 0.9831 | 1.0000 |
| `recall_all@5` — every gold session in the top five | 0.7966 | **0.8983** |

**The headline metric is saturated, and this is the finding.** A plain BM25
keyword search with no memory system of any kind reaches 0.9661 — MemPalace's
published 96.6%, to the digit. Fifty candidate sessions and "is the gold one in
the top five" does not separate a memory architecture from `grep`. This
project's 0.9831 is one question better than keyword search out of 59, and
quoting it as a competitive number would be quoting the benchmark.

What separates them is the strict metric, where the question is whether *all*
the evidence was found: 0.7966 against 0.8983, ten points. Split by type, the
loose metric is at a ceiling everywhere except one:

| `recall_any@5` | n | BM25 | this project |
| --- | --- | --- | --- |
| knowledge-update | 9 | 1.000 | 1.000 |
| multi-session | 15 | 1.000 | 1.000 |
| single-session-user | 8 | 1.000 | 1.000 |
| single-session-assistant | 7 | 1.000 | 1.000 |
| single-session-preference | 4 | 1.000 | 1.000 |
| temporal-reasoning | 16 | 0.875 | 0.938 |

Five of six types are solved by keyword search. Any system reporting a single
R@5 over all six is reporting five ceilings and one real number.

Reproduce with `python3 benchmarks/longmemeval_recall.py`. It rebuilds the
corpus from the LongMemEval file rather than the per-question ndjson the
original run wrote, and the check on that rebuild is BM25's `recall_all@5`,
which comes back 0.7966 exactly as first recorded.

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
- **Nothing about mem0's own defaults.** It is measured here with the shortlist
  and the extras this comparison chose for it, not the configuration its authors
  would pick. The re-run corrects two ways it was handicapped; there may be a
  third nobody has looked for, and the way to find one is to read its
  documentation rather than its behaviour.
- **Nothing about a third system with no model on its write path.** MemPalace
  answers that question for itself, below, and answers it against this project
  too. Nothing else here does.
- **Nothing about any difference smaller than the noise floor.** Forty-one of
  199 questions moved between two identical mem0 runs. Anything at that scale
  here is unmeasured, not measured-as-equal — the two are different claims and
  only the first is supported.
- **Nothing about scale.** Ten conversations, 5,882 turns, 588 on average.
- **Nothing that travels between machines except accuracy, calls and tokens.**
  Latency, resident memory and wall-clock are properties of four shared cores.
  The latency figures above are a ratio measured under one condition, not a
  number to quote on other hardware.

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
