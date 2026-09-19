---
name: pamin-dev
description: How to measure and how to keep claims true when working on PaminMemory itself. Use this skill whenever you are about to benchmark something, quote a latency or an accuracy figure, tune or justify a constant, add or change an evaluation harness, or write a number into a README, ADR, code comment or PR description. Also use it before opening a PR that touches docs, and whenever you notice documentation that might describe behaviour the code has since changed. Reach for it on phrases like measure, benchmark, how fast is, how much better, tune, sweep, is it worth it, or regression — and when a repository claim and the code appear to disagree.
---

# Measuring PaminMemory, and keeping its claims true

This repository has been wrong about its own numbers more than once, and each
time the mistake was the same shape: a number was measured somewhere other than
where the product runs, and nothing caught it because documentation has no
tests. What follows is what those mistakes cost and how to avoid repeating them.

## Measure the entry point the product calls

Both evaluation harnesses drove `Engine::search_fused`. `pamin search` calls
`Engine::search_reranked`. Every retrieval figure this repository published
therefore described a pipeline one stage shorter than the one that ships. When
it was re-measured through the real entry point:

| | published | measured |
| --- | --- | --- |
| `fast` reranker gain | +0.0595 | +0.0375 |
| `accurate` reranker gain | +0.0852 | +0.0458 |
| `accurate` latency | 1795 ms | 469 ms |
| same-language effect | "unchanged, by construction" | −0.0062 and +0.0022 |

Gains overstated by about 1.6×, latency by 3.8×, and one claim that was not
merely imprecise but false. A separate figure was taken at `PAMIN_PROFILE=speed`
while the product defaults to `accuracy` — 21.8 ms against the real 32.1.

So, before running anything, answer one question: **what does the user's command
actually call?** Trace it. `pamin search` → `Engine::search_reranked`;
`pamin write` → the full write path including the filter and the transaction.
Measure that function, with the defaults the product ships, on the profile the
product defaults to.

The exception is deliberate and worth stating when you take it: a *sweep* over a
tunable has to call the layer that accepts the tunable. Sweeping fusion weights
through a pass that reorders the top twenty afterwards would credit the
reranker's work to the weight. When you measure below the shipped entry point,
say in the same breath why, and do not let that figure escape into a README.

A rewritten harness is not a measurement of the product. If a scratch program
re-implements the pipeline to measure it, you are measuring the scratch program.

## Every measurement arm must assert its own premise

A concurrency harness had a "cached" arm and a "fresh" arm. It warmed the query
cache once before the whole sweep, and each cache-miss configuration then
evicted it — 256 entries, FIFO. So the cached arm was measuring misses. It
produced "two concurrent readers get less throughput than one", which was
published in the ADR and later retracted. The truth was the opposite:
throughput rises with concurrency, 47.4 → 77.9 q/s.

The guard that would have caught this existed in the first version of the
harness and was deleted during a rewrite.

So: **each arm carries an assertion that fails if the arm is not measuring what
it claims.** The cached arm now asserts that its fully-cached p50 is under half
the uncached p50; an evicted cache fails the arm instead of reporting a number.
Ask of every arm: what would it look like if this arm were silently measuring
the wrong thing, and what assertion distinguishes the two?

A second sweep, on the same harness, was pointed at a project that was empty.
The corpus lived in `xquad-accuracy-24ad7f1862182925`; the sweep asked
`default`, got no hits, and answered every query in 53 ms. It would have
produced a full p50/p95/p99 table for two corpora and three reranking tiers,
every cell of it the cost of searching nothing — and the cache guard that
sweep already carried would have passed it, because 53 ms is comfortably above
a cache hit.

That is the shape to watch for: **a guard only catches the failure it was
written for.** The arm now discovers which project holds the corpus, asserts
its topic count equals the document count the sweep claims, and asserts that a
real query returns hits, before a single cell is measured.

The same question applies to any test. "Would this assertion have failed before
the fix?" If not, it is decoration. Known trap: verifying "after drain the index
has it" also passes on code with no queue at all — proving the outbox is worth
anything requires interrupting between commit and execution.

## Where a harness lives

Two kinds, two homes, and the rule is about what the harness asserts:

- **It asserts an invariant** → it belongs in the repository, as a normal test.
  `crates/pamin-store/tests/hubcost.rs` is in the tree because it asserts that a
  bounded graph walk returns the same top fifty as an unbounded one *and* reaches
  fewer rows. Removing the bound fails it.
- **It only produces numbers** → it stays in the scratchpad, and the numbers go
  into the ADR with the conditions they were taken under. A test that asserts
  nothing costs CI time and rots.

Measurement harnesses **never modify tracked files**. Two of them used to append
code into `crates/pamin-engine/tests/crosslingual.rs` and restore it on exit,
and every failure mode of that arrived: 216 lines of scratch committed and
pushed because `git add -A` ran while a background harness held the file; a
completion hook reporting uncommitted changes a dozen times across a three-hour
run; and a killed run leaving the file modified because the trap never fired.

The pattern that replaced it — `.gitignore` covers `crates/*/tests/scratch_*.rs`,
and the harness generates `crates/pamin-engine/tests/scratch_<name>.rs` as a
copy of the base test plus its measurement block. Cargo discovers `tests/*.rs`
as its own integration target with no `Cargo.toml` entry:

```sh
SCRATCH=$REPO/crates/pamin-engine/tests/scratch_conc.rs
rm -f "$REPO"/crates/pamin-engine/tests/scratch_*.rs   # sweep at START too
trap 'rm -f "$SCRATCH"' EXIT
cat "$BASE" "$SP/conccost.rs.append" > "$SCRATCH"
cargo test --test scratch_conc -- --ignored --nocapture
```

Sweeping at startup is the part people leave out. An ignored file a killed run
leaves behind is invisible to `git status`, and the next `clippy --all-targets`
will compile it and may fail the `-D warnings` gate on scratch code.

## Running a measurement without losing it

- **Write raw output to a file first, filter when reading.** An hour of
  measurement vanished to a grep filter that matched `^  [0-9]` against rows
  indented by five spaces. The run is expensive; the filter is free to redo.
- **Do not touch git while a background harness runs.** `git add -A` picks up
  whatever it is holding.
- **Three runs, take the median.** If the spread exceeds about 10%, add rounds
  rather than picking a number.
- **Check for shared state between configurations.** The cache-eviction bug was
  configuration *N* poisoning configuration *N+1*.
- **Write the prediction down before you look.** Predictions in this repository
  have been wrong often enough to be worth recording: a write-profile delta
  predicted at +31 ms measured +10.2; the reranker was predicted to be nearly
  useless on a monolingual corpus and the `accurate` tier gained *more* there
  (+0.0496) than on the parallel corpus (+0.0458). A prediction that survives is
  cheap; one that fails is the most informative output of the run.

## The environment is part of the measurement

Five runs in one session died or lied, and not one of them was wrong about
retrieval. They were wrong about the box.

- **Memory, not just CPU.** A retrieval run was killed at question 6 of 59 by
  starting a second measurement beside it. The reasoning was "quality scores do
  not depend on CPU contention, so these can share the machine", which is true
  and beside the point: each server holds a model and an index, 3.6 GB at
  13,014 documents and 7.2 GB at 131,924, and two of them do not fit. Before
  running anything beside a measurement, price it in memory.
- **Count the servers, and count them again after.** `pamin search` starts a
  server when none is listening. A harness that spawns `pamin serve` and then
  probes it with a search races the two: the spawned one has not bound the
  socket yet, the probe starts a second, and two copies of a large model are
  resident. Wait on the socket, then assert the count is exactly one.
- **The user.** These workspaces keep PostgreSQL's data directory at mode 700.
  Run as anyone else and the server never starts, which surfaces as a startup
  timeout rather than a permission error. Derive the user from the directory
  rather than hard-coding one.
- **The file-descriptor limit.** A 131,924-document index is 2,111 segment
  files and a search holds 2,733 descriptors, against the 1,024 a Linux process
  gets by default. This was a real defect and is fixed in the server, but the
  general form stands: a measurement that only ever ran under a raised limit
  was measuring the shell, not the product.
- **The measurement process itself.** A harness run through a pipe that the
  caller then closed exited with status 0 having printed one tier of three, its
  remaining output still in a block buffer. Detach it, force line buffering,
  and write to a file — `setsid`, `stdbuf -oL`, `> log 2>&1` — so a death shows
  where it died instead of looking like a short run.
- **The machine, which may not stay.** This container is reclaimed without
  warning; six times in three hours during one comparison, each time taking
  every background process with it including the watchdog written to restart
  the run. Two things together survive that and neither alone does: a
  checkpoint after every unit of work, so a restart costs the unit in flight
  rather than the run, and a timer *outside* the container to start it again.
  A laptop has neither problem, which is the argument for moving long runs to
  one — but keep the checkpointing, because a run that cannot resume is a run
  nobody repeats.
- **The package manager.** Installing a third-party system into the shared
  environment moved `protobuf` past the ceiling another one declares, while
  that other one was being measured. It survived, by luck, because it had
  already imported what it needed. Each third-party system gets its own
  virtualenv, and nothing is installed while a measurement is running.

## A number without its corpus is a claim, not a measurement

Every figure carries the corpus it came from, the profile, and the hardware when
that matters. `0.7359 nDCG@10` means nothing; `0.7359 nDCG@10 on MIRACL's
Swahili dev split, 131,924 passages, at the shipped defaults` can be checked
and can be reproduced.

Two properties that are easy to state and easy to forget: every latency here is
a **four-core** result, and the write figure is for **short** content because a
forward pass scales with length.

When a constant changes because of a measurement, the comment next to it cites
what was measured, not what was assumed. Re-deriving `DEPTH`, `BATCH` and
`MAX_TOKENS` through the engine found that every value survived and **every
justification written for them was wrong** — the values were right by luck, and
the comments were teaching the next reader something false.

Make constants settable when you need to sweep them (`PAMIN_RERANK_DEPTH` and
friends) rather than editing the tree for each configuration. Undocumented on
purpose: they exist so a sweep does not dirty the working tree, not as a
supported surface.

## Auditing claims the code no longer backs

Documentation has no tests, so it goes stale silently. One pass over this
repository after a large merge found six:

| where | said | actually |
| --- | --- | --- |
| README `why` example | ends on `"modifier": "importance", "factor": 1.0` | the ranker stopped emitting modifiers that changed nothing |
| README Quickstart | `pamin stop` shuts the database down | it stops the resident server too |
| README "What It Does" | leads with page/tree structure | the same file lists it under "Not built yet" |
| `docs/cli.md` weight table | lexical weight `0.5` | `0.25` since the sweep settled it |
| `docs/cli.md` reranker prose | same-language ranking "unchanged" | the table three lines above says −0.0062 |
| ADR quantization table | "Off, permanently" | four paragraphs below: "Revisit when the binding exposes rotation" |

The pattern in all six: a later commit corrected the mechanism, or the table,
and left the prose that explained the old behaviour standing.

A seventh is worse than stale, because it was introduced by a correction:

| where | said | actually |
| --- | --- | --- |
| ADR reranker table, `docs/cli.md`, `reranking.rs` | `accurate` costs 508 ms, its pass 469 ms | 1001 ms and 948 ms, from the same harness on the same machine and workspace |

That 469 ms had itself replaced an earlier 1795 ms. The first figure was too
high, the correction was too low, and the argument for the default tier was
rewritten around the wrong ratio twice. Neither number was checked by running
it again.

So: **a figure you correct is a figure you re-run.** A correction inherits all
the trust of the thing it replaces and none of the scrutiny, which makes it the
easiest place in a repository for a wrong number to live. Re-run it, and say in
the commit message that you did.

Worth a sweep whenever a behaviour changes, and before any PR that touches docs.
Places to look, in rough order of how often they are wrong:

1. **Worked examples and sample output** — these encode behaviour precisely, so
   they break precisely. Recompute derived values rather than eyeballing them:
   a fusion contribution is `weight / (10 + rank)`, so a weight change moves
   every contribution and the score.
2. **Prose next to a table you edited.** Correcting a number and leaving the
   paragraph that justified the old one is the single most common failure here.
3. **Verdict cells** — "permanently", "never", "unavailable". Check them against
   the prose that follows; a deferral with a trigger is not a permanent decision.
4. **PR descriptions of open PRs.** They are read during review and they go
   stale as the branch moves.
5. **Comments citing a measurement.** The value can survive while its
   justification does not.

When you find one, fix the claim rather than deleting it, and say in the commit
message which commit made it stale. That is the record of how it happened, and
it is what stops the next person re-deriving the same wrong thing.

## The gate

Before pushing anything:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace
python3 ci/budget.py
```

For anything touching a database, an index or a model:

```bash
cargo test --workspace -- --ignored    # provisions postgres, downloads models
```

If a constant that affects scores changed, re-run both corpora, re-set the
floors, and verify the floors by reverting: a floor that still passes with the
feature disabled is not guarding it. A tenth of margin is wider than the whole
reranker contribution, which is why the reranker needed its own assertion
(`RERANK_IS_WORTH`) rather than a floor.

## Comparing against other projects

Before quoting anyone else's number, read
[docs/benchmarks.md](../../../docs/benchmarks.md). The short version: this
category publishes LLM-judge accuracy on conversational QA, which measures a
retrieval stage plus a reader model plus a judge, compounded — not the same
quantity as `nDCG@10`, and not placeable on the same axis. Several headline
figures in that field have been audited by competitors and did not survive.

The harness that runs those comparisons is in [benchmarks/](../../../benchmarks),
along with what it holds fixed and how each of those is asserted. Nine rules
come out of building it, and each cost a run to learn.

**Open the other side's budget before claiming a win.** Raising `--limit` from
ten to thirty moved this project's LOCOMO accuracy by eleven points — a real
and significant gain, and for a while it read as beating mem0. It was not, and
the reason is that only one arm had been widened. Any knob you turn for your
own arm, turn for theirs, and report what happened when you did.

**Turning the knob is not the same as the knob turning.** The wide mem0 arm was
built to obey that rule and did not: mem0's keyword is `top_k`, not `limit`, it
defaults to 20, and unknown keywords land in `**kwargs` and are discarded
without an error. Both mem0 arms therefore ran at 20 — the narrow one wider
than it claimed, the wide one narrower — and the conclusion drawn from them,
that a wider shortlist was worth nothing to mem0, was about an arm that never
widened. Nothing failed; the numbers were plausible; only counting what came
back caught it. So an arm asserts the setting it reports, from the result and
not from the call: the shortlist arm checks how many passages it received, the
same way the MemPalace arm checks that it embedded through the shared endpoint.
A comparison is a set of premises, and an unasserted premise is a guess.

**Run one arm twice before comparing two.** The accident above left two
identical mem0 runs, which turned out to be the most informative pair on the
page: totals 0.603 and 0.598, but 41 of 199 questions answered differently.
Any arm whose write path puts a model in the loop disagrees with itself, and
LOCOMO at this size churns about a fifth of its questions under almost any
perturbation — a deterministic configuration change worth 0.005 moved the same
41. A net of five or ten questions is inside that. Spend one run on the same
arm twice, before spending ten on arms whose differences you cannot read.

**Install the other side the way its own documentation does.** mem0's BM25
channel lemmatises on both sides -- every memory stores a `text_lemmatized`
field, every query is lemmatised before the keyword search -- and that path
needs the `mem0ai[nlp]` extra, which a plain install does not bring. Without
it `lemmatize_for_bm25` returns its input unchanged and half of a hybrid
retriever is off, announced on one log line and nowhere else. Every mem0
figure taken before this was found was taken in that state. A default install
is not the same as the system: read what extras the other side's own
instructions install, and have the arm refuse to run when one is missing.

**Time the same layer on both sides, or do not report time.** The accuracy run
recorded a `recall_seconds` per question and it looked like a latency column.
It was not one: this project was timed through `su ubuntu -c "pamin ... search
..."` -- two process spawns and a socket round trip -- and mem0 as an
in-process library call, while ten arms, an embedding endpoint and a PostgreSQL
cluster shared four cores. It read 170 ms against 106 and the obvious
conclusion was the opposite of the truth. Timed at the boundary each system's
callers actually use, with nothing else running, it is 29 ms against 94. A
latency number needs its own harness, because the three things it depends on --
the layer, warmth and quiet -- are exactly the three an accuracy run cannot
hold still. And warm every unit before timing anything, not the first one: an
early draft of that harness warmed one conversation of ten, timed the socket
arm first, and reported it at three times the CLI it is faster than. Run each
arm again last; two passes that disagree mean the order is in the number.

**A category you win is the one to read the items of.** LOCOMO's adversarial
split went to this project and MemPalace, 0.429 against mem0's 0.190, and it
was written up twice as a property worth having before anyone followed a
question to the turn its own `evidence` field names. Doing that: 332 of the 446
attribute to one speaker something the *other* speaker said, and the key gives
that other speaker's content as correct. "What country is Melanie's grandma
from?" is keyed to Sweden; the evidence is Caroline saying "my grandma in my
home country, Sweden". The category rewards ignoring attribution. mem0 answers
"no record of that" and is marked wrong for the better answer.

The lesson is not about LOCOMO. A split where this project leads is the one
where the temptation to stop reading is strongest, and this category was
described wrongly twice in this repository before it was described wrongly in
its favour. So: open the items, follow the evidence field, and ask what
behaviour a high score is actually rewarding. If the answer is a behaviour you
would call a bug in a bug report, the score is not a result.

**Cost is half the claim.** A project whose pitch is "less" cannot check that
pitch with an accuracy table. Measure what each arm spends: calls to a model
on the write path, tokens in the prompt it hands the reader, bytes it stores,
memory it holds. The two halves point opposite ways here — one arm pays a
model once at write time and hands the reader less afterwards; the other pays
nothing at write time and hands the reader more, every question, forever — so
a comparison that reports one half reports the wrong system as cheaper.

**Count tokens with a tokenizer you name, not with what the runtime reports.**
This environment's CLI reports a prompt of any size as a constant plus
`cache_creation_input_tokens`, and its delta for a 3,623-character string was
1.9x what any tokenizer gives for it. Absolute token figures here are
therefore approximate and say so; the ratio between arms, counted the same way
on both sides, is what the comparison actually needs.

**An arm that rewrites what it stores cannot be scored on retrieval.** mem0
extracts facts, so its memories belong to no turn and cannot be mapped back to
gold sessions. The harness refuses rather than inventing a mapping. Say which
metric an arm is eligible for before running it, not after the numbers look
strange.

That page also records the one comparison that is honest and cheap
(LongMemEval's retrieval-only stage, scored with Recall@k and NDCG@k, no LLM
anywhere), what it would take to run, and what result to expect before running
it so the outcome can falsify the expectation.

## Standing up a live workspace for a measurement

`initdb` refuses to run as root, which is what stops an `--ignored` suite or an
ad-hoc harness in a root container. Run as an unprivileged user, and reuse the
model cache rather than downloading 1.7 GB again:

```sh
W=/tmp/scratch-ws
mkdir -p "$W/postgres"
ln -s /tmp/pamin-eval5/models "$W/models"       # whatever cache already exists
cp -a /tmp/pamin-eval5/postgres/install "$W/postgres/"
chown -R ubuntu:ubuntu "$W"
su ubuntu -c "PAMIN_HOME=$W .../pamin init"
```

Once the server is up, other users can reach it through the socket, so only the
provisioning needs the unprivileged user.

Decisions and the measurements behind them live in
[docs/adr/0001-tech-selection.md](../../../docs/adr/0001-tech-selection.md),
which is the source of truth when it and any other page disagree.
