# File records and re-ingest cost, 2026-10-27

What recording each ingested file saves a re-ingest, measured through `pamin
ingest` against the stack it builds on.

**Conditions.** One 4-vCPU Intel Xeon in a cloud container on a host we do not
control, 15 GB RAM; the one-minute load average was 2.06 when the run began and
1.85 when it ended, as the CSV's first and last lines record. No build of ours
ran during an arm. Release builds of `0539965b` (`base`) and `9505cf8`
(`head`); CPU execution through ONNX Runtime; BGE-M3 int8
(`gpahal/bge-m3-onnx-int8`) embedding and bge-reranker-v2-m3 SmoothQuant int8
reranking, loaded by a warm-up search before the first arm. PostgreSQL 17.6.
Default settings, `memory` vector index. The corpus is this repository's 112
tracked `.md` and `.txt` files outside `internal-docs` and dot-directories, cut
into 388 sections by both builds. Server CPU is the server process's own user
and system time: it excludes PostgreSQL's backends and the client. Harness and
commands: [`benchmarks/harnesses/ingest-sync-2026-10-27`](../../harnesses/ingest-sync-2026-10-27/README.md).
Raw rows: [`ingest-sync-arms-9505cf8-2026-10-27.csv`](ingest-sync-arms-9505cf8-2026-10-27.csv).

`head` is `9505cf8`, the tip of the work this change is the first part of:
file records, the ingest core split, file sync and the search re-check
together, before `--keep-missing` was removed. It ingested with `--no-sync`, so
the watcher took no part, and nothing after the records changes an unchanged
or a one-file re-ingest; the figures below are what this change does to one.

## Re-ingesting

Five arms each, the build order alternated each round, every arm 3 s after the
one before.

| metric | `base` | `head` | difference | change |
| --- | ---: | ---: | ---: | ---: |
| unchanged re-ingest, wall, median (range) | 0.484 s (0.473–0.529) | 0.017 s (0.017–0.019) | −0.467 s | −96.5% |
| unchanged re-ingest, server CPU, median | 0.47 s | 0.01 s | −0.46 s | −97.9% |
| one-word edit, wall, median (range) | 0.790 s (0.694–0.874) | 0.301 s (0.218–0.454) | −0.489 s | −61.9% |
| one-word edit, server CPU, median | 1.11 s | 0.59 s | −0.52 s | −46.8% |
| sections written per edit | 1 | 1 | 0 | 0% |
| cold ingest, wall (n = 1) | 189.0 s | 183.8 s | −5.2 s | −2.8%, one sample each |
| memory, disk | N/A | N/A | N/A | not measured |

The ranges do not overlap. An unchanged `head` re-ingest stats each file
against its record and reads none of them; `base` reads, cuts and compares
every section, one query each. An edit costs `head` the one file. The
individual commits -- the tokenizer kept loaded, records, the scoped section
lookup -- were not measured apart: this is the stack against `base`.

The harness also has search arms. They measure the search re-check of a
later change and are reported with it.
