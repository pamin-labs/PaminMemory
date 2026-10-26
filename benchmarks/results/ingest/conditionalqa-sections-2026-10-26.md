# ConditionalQA: sections, `part_of` and parent context, 2026-10-26

What cutting a document at its headings is worth when a question's evidence is
spread over its sections, and what relating the sections adds.

The Qasper comparison
([qasper-sections-2026-10-25.md](qasper-sections-2026-10-25.md)) could not
answer that. 78% of its questions are answered by one section, and only 2% need
a section together with the one above it.

ConditionalQA (Sun, Cohen and Salakhutdinov, ACL 2022) is built the other way:

- **Documents:** gov.uk guidance pages with their heading structure.
- **Questions:** each answer holds only under conditions stated elsewhere on the
  page, often under the heading above.
- **Prior use:** DAPR (Wang, Reimers and Gurevych, ACL 2024) uses it to test
  document-aware passage retrieval. It traced 53.5% of strong retrievers'
  errors to missing document context.

HiCBench, built to judge chunking with multi-section evidence, was considered
first. Its data is not published.

## Conditions

- **Corpus.** All 652 pages of ConditionalQA v1.0, rendered to Markdown:
  - each page's title becomes `#`, its `h1`–`h4` become `##`–`#####`;
  - list items become `- `, table rows `| … |`;
  - every line's text is kept verbatim.

  Ingested as `pamin ingest` records files, one memory per heading section. The
  pages are Crown copyright under the Open Government Licence, fetched from the
  dataset's repository, not vendored.
- **Questions.** All 285 dev and 600 train questions, drawn by seed 20261012.
  Of those, 871 have evidence; all 3,634 of their evidence sentences are found
  verbatim in their page. Each is asked as its scenario followed by its question.
- **Spread.** Each question is classed by where its evidence sentences sit in
  its page's outline:
  - one section: 430;
  - a section and one above it: 267;
  - sibling sections: 102;
  - otherwise: 72.
- **Search.** Every question goes through `Engine::search_reranked`, the
  function `pamin search` calls, at the product's depths, for 30 hits.
- **Scores.** A sentence counts as found when it appears exactly in the text a
  hit returns, from the question's own page.
  - **All evidence @k:** every sentence found in the first k hits.
  - **covered@k:** the share of sentences found in the first k hits.
  - **Fixed read:** the hits read in order until 1,000 tokens are taken.
  - **Statistics:** paired over pages. p is a sign-flip test; the CI is a 95%
    bootstrap over pages.
- **Build and hardware.** Release builds on one 4-vCPU Intel Xeon @ 2.10 GHz
  with 15 GB, in a cloud container. CPU execution through ONNX Runtime:
  - embedding: BGE-M3 int8 (`gpahal/bge-m3-onnx-int8@2b34e84`);
  - reranking: bge-reranker-v2-m3 SmoothQuant int8
    (`pamin-labs/bge-reranker-v2-m3-sq-int8-onnx@eb7411f`);
  - database: PostgreSQL 17.6, the published build of the bundled version.
- **Commits.**
  - The `whole`, `sections`, both-links and `part_of`-only passes ran at
    `920af63`.
  - The parent-context arms ran on that work merged with `main` at
    `cd30adf`. That merge includes #305, the hit-view budget, which changed
    what 14 of 871 questions' hits return. Each comparison below is between
    passes of one binary.
- **Timing.** Not measured. Builds and tests ran on the machine during these
  passes, so every latency is N/A, and only accuracy and returned text are
  reported.

Harness, commands and summarisers:
[`benchmarks/harnesses/conditionalqa-sections-2026-10-26`](../../harnesses/conditionalqa-sections-2026-10-26/README.md).
Raw rows, one per question and arm:
[`conditionalqa-sections-rows-2026-10-26.json.gz`](conditionalqa-sections-rows-2026-10-26.json.gz)
for the cut and the edges, and
[`conditionalqa-parents-rows-2026-10-26.json.gz`](conditionalqa-parents-rows-2026-10-26.json.gz)
for parent context.

## Heading sections against whole pages

| metric | whole pages | heading sections | difference | change | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| all evidence @5 | 0.428 | 0.690 | +0.262 | +61.1% | [+0.222, +0.303] | 0.0001 |
| all evidence @10 | 0.432 | 0.805 | +0.373 | +86.4% | [+0.326, +0.421] | 0.0001 |
| covered@1 | 0.519 | 0.530 | +0.011 | +2.2% | [−0.016, +0.040] | 0.44 |
| covered@10 | 0.608 | 0.896 | +0.288 | +47.3% | [+0.252, +0.327] | 0.0001 |
| all evidence in a fixed 1,000-token read | 0.425 | 0.755 | +0.331 | +77.8% | [+0.286, +0.375] | 0.0001 |
| text the first 10 hits return | 11,429 B | 5,544 B | −5,885 B | −51.5% | [−6,320, −5,474] | 0.0001 |
| section + one above: all evidence @10 | 0.146 | 0.596 | +0.449 | +307.7% | [+0.367, +0.532] | 0.0001 |
| siblings: all evidence @10 | 0.157 | 0.794 | +0.637 | +406.3% | [+0.539, +0.735] | 0.0001 |
| documents indexed (spans) | 13,522 | 11,358 | −2,164 | −16.0% | counts | |
| search latency, memory, disk | N/A | N/A | N/A | N/A | not measured | |

A whole page returns its best span and neighbours, and the evidence of a
question about two parts of a page is rarely in one window. Sections let each
part be found on its own, with half the text returned.

## `part_of` between sections

Measured on the sections project: first with no edges, then with each
section's `part_of` the section above it **and** a `related_to` to the next
sibling, then with `part_of` alone. The project and the index stayed the same
throughout.

| metric | no edges | `part_of` | difference | change | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| all evidence @5 | 0.690 | 0.719 | +0.029 | +4.2% | [+0.011, +0.047] | 0.0031 |
| all evidence @10 | 0.805 | 0.823 | +0.018 | +2.3% | [+0.003, +0.034] | 0.031 |
| all evidence in a fixed 1,000-token read | 0.755 | 0.792 | +0.037 | +4.9% | [+0.022, +0.053] | 0.0001 |
| text the first 10 hits return | 5,544 B | 5,312 B | −232 B | −4.2% | [−270, −196] | 0.0001 |
| **section + one above (267): all evidence @5** | 0.378 | 0.476 | +0.097 | +25.7% | [+0.052, +0.142] | 0.0001 |
| section + one above: fixed read, all evidence | 0.506 | 0.618 | +0.112 | +22.2% | [+0.075, +0.154] | 0.0001 |
| one section (430): covered@1 | 0.708 | 0.685 | −0.023 | −3.2% | [−0.045, +0.000] | 0.069 |
| one section: fixed read, all evidence | 0.940 | 0.930 | −0.009 | −1.0% | [−0.021, +0.000] | 0.22 |
| hits only the graph reached, in the first 10 | 0 | 0.139 | +0.139 | n/a | [+0.108, +0.173] | 0.0001 |

**`related_to` between siblings adds nothing `part_of` does not.** The graph
walks two edges, so siblings already reach each other through their parent.

| `part_of` alone against both edges | both | `part_of` | difference | p |
| --- | ---: | ---: | ---: | ---: |
| all evidence @5 | 0.700 | 0.719 | +0.018 | 0.005 |
| all evidence in a fixed read | 0.791 | 0.792 | +0.001 | 1.0 |
| siblings (102): all evidence @10 | 0.794 | 0.804 | +0.010 | 1.0 |
| hits only the graph reached, in the first 10 | 0.258 | 0.139 | −0.119 | 0.0001 |

So the product states `part_of` only.

## Parent context in the hit

One search pass with `part_of`. Every question had a hit with a parent. The
same ranking was shown three ways:

- **off:** the hits as they are.
- **every hit:** each hit preceded by its parent section's opening (up to 200
  tokens), unless that parent was already read.
- **shared:** a parent's opening shown once, before the first of two or more
  of the first 10 hits under it.

The parent's text counts against every budget.

| metric | off | every hit | shared |
| --- | ---: | ---: | ---: |
| covered@1 | 0.520 | 0.595 (+14.5%, p 0.0001) | 0.520 (0.0%) |
| all evidence @5 | 0.719 | 0.760 (+5.8%, p 0.0001) | 0.738 (+2.7%, p 0.0001) |
| section + one above: covered@1 | 0.337 | 0.477 (+41.6%, p 0.0001) | 0.337 (0.0%) |
| text the first 10 return | 5,465 B | 7,654 B (+40.1%) | 6,116 B (+11.9%) |
| **all evidence in a fixed 1,000-token read** | 0.786 | 0.772 (−1.9%, p 0.11) | 0.785 (−0.1%, p 1.0) |
| one section: fixed read, all evidence | 0.930 | 0.912 (−2.0%, p 0.022) | 0.930 (0.0%) |

A hit's parent puts its context nearer the top, but only by giving more to
read. Read to the same budget, it finds no more evidence, and a question
answered by one section reads less of what it needs. Search therefore carries
only the parent's name on each hit, as `parent`. The hit's own text is
unchanged, and a caller can read the parent when it wants to.

What the name is worth depends on a caller choosing to read it, which no
retrieval score measures.

A parent replacing its children, the auto-merge of HiChunk, does not apply
here. A section holds the text under its own heading and not its subsections',
so a parent replacing its children would drop the text that matched.
