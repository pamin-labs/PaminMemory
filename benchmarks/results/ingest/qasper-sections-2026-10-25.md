# Qasper: heading sections against sized sections, 2026-10-25

`pamin ingest` used to cut a file into sections sized so the reranker read each
whole beside its name, about two hundred tokens, splitting a long section into
numbered parts. It now cuts at headings only, one topic per heading section,
and leaves size to the index, which cuts every memory into spans of at most 200
tokens. This compares the two cuts, and recording each file whole, on
documents nobody here wrote.

**Conditions.**

- **Hardware and build.** One 4-vCPU Intel Xeon @ 2.10 GHz, 15 GB RAM, in a cloud
  container on a host we do not control. Release build. No build or other
  measurement of ours ran during the timed rounds; one short summary script
  ran between round 0's last arm and round 1's first.
- **Models and database.** CPU execution through ONNX Runtime. BGE-M3 int8
  (`gpahal/bge-m3-onnx-int8@2b34e84`) for embedding and bge-reranker-v2-m3
  SmoothQuant int8 (`pamin-labs/bge-reranker-v2-m3-sq-int8-onnx@eb7411f`) for
  reranking. PostgreSQL 17.6, the published build of the version the product
  bundles, supplied as an installation.
- **Settings.** Default settings, `memory` vector index, spans of 200 tokens.
- **Corpus.** Qasper's test split (Dasigi et al., NAACL 2021, CC BY 4.0). The
  harness's default draw of 100 papers (seed 20261012) has 308 questions, and
  98 of the papers have at least one question. Every question is asked
  through `Engine::search_reranked`, the function `pamin search` calls, at
  the product's depths, for 30 hits.
- **Arms.** Each arm is a fresh project:
  - `whole`: each paper is one memory.
  - `sized`: `sections` at `a2b6fb3`, sized to the reranker's window, with
    long sections split into parts. That gives 5,147 memories, 3,655 of them
    parts.
  - `headings`: `sections` at `cf6ebe0`, one memory per heading section,
    1,492 memories.
- **Rounds.** Three rounds, rotating the order of the arms. Accuracy was
  identical on every question in every round, so it is read from round 0.
  Timings are the median of the three rounds.

Harness and commands:
[`benchmarks/harnesses/qasper-sections-2026-10-25`](../../harnesses/qasper-sections-2026-10-25/README.md).
Raw rows: [`qasper-sections-rows-cf6ebe0-2026-10-25.json.gz`](qasper-sections-rows-cf6ebe0-2026-10-25.json.gz),
one per question, arm and round; and
[`qasper-sections-runs-cf6ebe0-2026-10-25.csv`](qasper-sections-runs-cf6ebe0-2026-10-25.csv),
one per arm and round.

**Metrics.**

- **covered@k.** The share of the best-served annotator's evidence
  paragraphs found in the text the first k hits *return*: a hit's span
  window, or the whole memory when it is one span. A paragraph counts as
  found when at least half of its distinct content words are read, and only
  from the question's own paper.
- **Recall at a fixed read.** The same share, read from the hits' returned
  text in rank order until 1,000 tokens have been taken. This holds the
  amount read fixed.
- **nDCG@10 and recall@10.** Scored over memories. A memory is relevant when
  it covers an evidence paragraph by the same rule.
- **Statistics.** Paired over papers: each paper contributes its questions'
  scores, `statistics::cluster_macro_contributions`. p is the harness's
  sign-flip test. The CI is a 95% bootstrap over papers from `summarise.py`.

## Heading sections against sized sections

| metric | `sized` | `headings` | difference | change | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| covered@1 | 0.1537 | 0.2372 | +0.0834 | +54.3% | [+0.051, +0.119] | 0.0001 |
| covered@5 | 0.4197 | 0.4564 | +0.0367 | +8.7% | [+0.007, +0.067] | 0.019 |
| covered@10 | 0.4761 | 0.5420 | +0.0658 | +13.8% | [+0.036, +0.096] | 0.0001 |
| text returned by the first 10 hits | 5,260 B | 12,433 B | +7,174 B | +136.4% | [+6,405, +7,987] | 0.0001 |
| **recall in the first 1,000 tokens read** | 0.4584 | 0.4091 | −0.0494 | **−10.8%** | [−0.087, −0.013] | 0.010 |
| nDCG@10 (memories) | 0.2665 | 0.3491 | +0.0827 | +31.0% | [+0.055, +0.111] | 0.0001 |
| recall@10 (memories) | 0.3442 | 0.4443 | +0.1001 | +29.1% | [+0.066, +0.136] | 0.0001 |
| search p50, ms | 1,786 | 1,918 | +132 | +7.4% | N/A, 3 rounds | |
| search p95, ms | 2,133 | 2,266 | +133 | +6.2% | N/A, 3 rounds | |
| memory (RSS) | N/A | N/A | N/A | N/A | not recorded by this harness | |
| index on disk | 453.8 MiB | 165.5 MiB | −288.3 MiB | −63.5% | same every round | |
| documents (spans) indexed | 5,147 | 4,351 | −796 | −15.5% | counts | |
| ingest wall, record and drain | 293.8 s | 261.8 s | −32.0 s | −10.9% | N/A, 3 rounds | |

**What holds, and what does not.**

- **More evidence at every rank, because more text per hit.** Heading
  sections find more evidence in the first 1, 5 and 10 hits. Each hit also
  returns more text: the best span and one either side, up to about 600
  tokens, against a sized section of at most about 200.
- **Less evidence when the amount read is held fixed.** Read to the same
  1,000 tokens, the sized sections find more, by −10.8% for heading sections
  (p = 0.010). So the gain at rank k is not separable from the extra text.
  - Both arms index units of at most 200 tokens. What differs is how much a
    hit shows.
  - That is a property of the hit view, `VIEW_TOKENS` on
    `feat/span-view-budget`, rather than of the cut. A comparison at a fixed
    byte budget, as the span work requires for that change, is the one that
    can say whether a narrower view recovers it.
- **nDCG@10 and recall@10 over memories do not compare across these arms.**
  `sized` has 3.4 times as many memories per paper, so its relevant set is
  larger and each member smaller. They are reported because the review asked
  for them. covered@k and the fixed read are the measures that compare.
- **Cost favours heading sections.** The index is 63.5% smaller: every
  memory carries a fixed cost there, and `sized` has 3.4 times as many. The
  index cuts heading sections into 15.5% fewer spans, and ingest is 10.9%
  faster. Search is 6–7% slower at p50 and p95. Three rounds cannot
  separate that from the machine: round 2 ran slower for `whole` and
  `headings`, at 2,504 and 2,671 ms p95, with no build running.

## Heading sections against whole files

| metric | `whole` | `headings` | difference | change | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| covered@1 | 0.2427 | 0.2372 | −0.0055 | −2.3% | [−0.043, +0.032] | 0.79 (not significant) |
| covered@5 | 0.3507 | 0.4564 | +0.1057 | +30.1% | [+0.058, +0.154] | 0.0001 |
| covered@10 | 0.3859 | 0.5420 | +0.1561 | +40.5% | [+0.106, +0.207] | 0.0001 |
| text returned by the first 10 hits | 17,372 B | 12,433 B | −4,939 B | −28.4% | [−5,476, −4,421] | 0.0001 |
| recall in the first 1,000 tokens read | 0.2977 | 0.4091 | +0.1114 | +37.4% | [+0.068, +0.156] | 0.0001 |
| search p50, ms | 1,941 | 1,918 | −23 | −1.2% | N/A, 3 rounds | |
| search p95, ms | 2,266 | 2,266 | 0 | 0.0% | N/A, 3 rounds | |
| index on disk | 59.6 MiB | 165.5 MiB | +105.9 MiB | +177.7% | same every round | |
| documents (spans) indexed | 4,360 | 4,351 | −9 | −0.2% | counts | |
| ingest wall, record and drain | 230.5 s | 261.8 s | +31.3 s | +13.6% | N/A, 3 rounds | |

Against whole papers, heading sections return less text and find more, both
at rank k and in a fixed read. That is because a paper's sections stop
competing as one topic and each brings its own window. nDCG@10 and recall@10
over memories are left out here for the same reason as above: one memory per
paper makes every relevant set a single memory.

## Not measured

Memory, for any arm. A docs tree, and any corpus where a question spans
files. Both cuts before span grain. The review's request to report these on
Qasper is met by the tables above. The docs tree is the next corpus, with the
outline-edge comparison.
