# `pamin ingest` cost, 2026-10-23

What ingesting a tree costs, what re-ingesting it costs, and how both compare
with `pamin import` of the same memories -- the bulk write path ingest records
its sections through, and the closest thing it has to a predecessor.

**Conditions.** One 4-vCPU Intel Xeon @ 2.80 GHz, no SMT, 15 GB RAM, in a cloud
container on a host we do not control. No build or other measurement of ours
ran during an arm, but the one-minute load average was 2.1 and 2.7 when the two
runs began, as the first line of each CSV records. Release build of commit `cf4a771`; CPU
execution through ONNX Runtime; BGE-M3 int8 (`gpahal/bge-m3-onnx-int8`)
embedding and bge-reranker-v2-m3 SmoothQuant int8 reranking, both loaded by a
warm-up search before the first arm. PostgreSQL 16 from the system packages.
Default settings, `memory` vector index. The corpus is this repository's 105
tracked `.md` and `.txt` files outside `internal-docs` and dot-directories, 1.4
MB, cut into 4,090 sections. Server CPU is the server process's own user and
system time: it excludes PostgreSQL's backends and the client, and includes
whatever index upkeep the server ran during an arm -- the 10.6 s outlier below
is an arm during which the index directory shrank from 120.8 to 33.5 MiB.
Database and index sizes are MiB, the index's as `du` allocates it, neither
counting the model cache. Harness and commands:
[`benchmarks/harnesses/ingest-cost-2026-10-23`](../../harnesses/ingest-cost-2026-10-23/README.md).
Raw rows: [`ingest-arms-cf4a771-2026-10-23.csv`](ingest-arms-cf4a771-2026-10-23.csv),
[`ingest-vs-import-cf4a771-2026-10-23.csv`](ingest-vs-import-cf4a771-2026-10-23.csv).

## Against `import` of the same sections

The sections the ingest recorded, exported verbatim and imported into a fresh
workspace. One sample of each arm, in the order ingest, import.

| metric | `import` | `ingest` | difference | change |
| --- | ---: | ---: | ---: | ---: |
| accuracy | N/A | N/A | N/A | N/A — see below |
| evidence rows an unchanged re-run appended | 4,090 | 0 | −4,090 | −100% |
| database growth from that re-run | +4.0 MiB | +0.0 MiB | −4.0 MiB | −100% |
| index after that re-run | 31.6 MiB | 31.6 MiB | 0 | 0% |
| server peak RSS | 2,352.6 MB | 2,352.4 MB | −0.2 MB | −0.0% |
| unchanged re-run, wall (n = 1) | 2.52 s | 4.20 s | +1.68 s | +66.4% |
| unchanged re-run, server CPU (n = 1) | 4.97 s | 7.22 s | +2.25 s | +45.3% |
| cold, wall (n = 1) | 505.2 s | 565.1 s | +59.9 s | +11.9% |
| cold, server CPU (n = 1) | 1,870.4 s | 2,118.5 s | +248.1 s | +13.3% |

**The counts are the difference that holds.** Importing an unchanged file again
holds every memory as a restatement and records each one as evidence: 4,090 new
`source_versions` rows and 4 MiB here. Ingesting an unchanged tree again writes
nothing. Those are the same on every run.

**The timings are not findings.** Each is one sample. Ingest's own six
unchanged re-ingests at `cf4a771` ran 4.20 to 6.72 s, a spread wider than the
1.68 s between the two re-run arms. The same cold-ingest arm ran 671.9 s and
565.1 s at `cf4a771`, and 351.5 s at `88bc9e6` before the container was
restarted, with server CPU moving in step (2,503 s, 2,119 s, 1,277 s): nearly
two to one between runs of one arm, against 12% between ingest and import.
What ingest adds on a re-run -- reading the files, loading the tokenizer,
cutting them -- was not timed apart from the rest.

**Nor is the cold pair the same work.** Both embed the same 4,090 texts, but an
ingested topic is kept out of the name index and owes no backfill, while
`import` files 4,090 path-and-heading names there and backfills each one. Which
way that moves the cold figure was not measured.

**Accuracy is not measured.** No benchmark here holds long documents with
questions about them, so whether sections sized for the reranker retrieve
better than other sizes is unmeasured. Search itself is unchanged: nothing on
the search path was modified, so the retrieval figures on
[measured.md](../../../docs/measured.md) describe manual memories exactly as
before. Search latency over an ingested corpus is not measured either.

## Re-ingesting, repeated

From [`ingest-arms-cf4a771-2026-10-23.csv`](ingest-arms-cf4a771-2026-10-23.csv),
one workspace, the arms in this order:

| arm | n | wall, median (range) | server CPU, median (range) | sections written |
| --- | ---: | ---: | ---: | --- |
| unchanged | 5 | 5.03 s (4.64–6.72) | 5.23 s (4.48–10.6) | 0 every time |
| one-word prose edit | 5 | 5.35 s (5.11–6.76) | 5.70 s (5.50–7.07) | 3, 3, 1, 1, 2 |
| a `##` heading renamed | 1 | 5.71 s | 5.97 s | 1, and 1 listed missing |

An edit writes more than one section when it falls in a section long enough to
have been split: the parts after it shift. The renamed heading had no sections
nested under it; renaming one that does rewrites each of them, as the run below
shows.

## An earlier run, not reproducible from the committed harness

[`ingest-arms-88bc9e6-2026-10-23.csv`](ingest-arms-88bc9e6-2026-10-23.csv) was
taken at `88bc9e6`, a revision of this branch before its last two review fixes
that no branch now holds, with an earlier version of the harness whose edit
arms prefixed a word to the first line after the first blank line. It is kept
for what it shows, not as a published figure. Its five unchanged re-ingests
took 2.96 to 3.68 s and wrote nothing. Three of its edits landed on heading
lines and so unmade the heading: the sections under it were renamed, and one
such edit, to the README's title, rewrote 56 sections and listed 57 more as
missing. That is the cost of renaming a heading near the top of a file, and why
the committed harness edits prose and renames one heading as an arm of its own.
