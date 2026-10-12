# LongMemEval dual-metric run — handoff (2026-10-12)

Status note for whoever picks up task #5 on branch
`bench/mem0-mempalace-miracl-head-to-head`. The run was stopped for handoff,
not because anything is broken. This file is operational status, not a
published result — nothing here belongs in `docs/benchmarks.md` until the run
finishes and goes through `summarise.py`.

## What this run is

Two `benchmarks/run.py` invocations sharing one `claude -p` shim endpoint
(`BENCH_SHARED_ENDPOINT=1`, model `haiku`), on the full LongMemEval corpus
(no sampling):

- **Retrieval-only** — arms `bm25,pamin,pamin-wide,mempalace-raw`, 470 units,
  scoring recall/nDCG against gold session identity. No reader, no judge.
- **Answer-accuracy (qa)** — arms `bm25,pamin,pamin-wide,mem0,mempalace-raw`,
  500 units, reader + judge scoring free-text answers. `mem0`'s ingest is
  slow by design on this corpus (~845s/haystack) so this arm alone was
  expected to take ~5 days; the others finish in hours.

Both write one row per arm per question as checkpointed JSONL, so either can
be killed and resumed by just running the same command again — the harness
skips rows already on disk. See `benchmarks/README.md` for what this harness
holds fixed generally; this note only covers what's specific to this run.

## State as of last confirmed checkpoint

As of **2026-10-12 02:35 UTC** (last cycle both processes were confirmed
alive and advancing):

| file | rows | target | phase |
| --- | --- | --- | --- |
| `benchmarks/results/longmemeval/retrieval.jsonl` | 631 | 1880 (470 units × 4 arms) | still in the `pamin` arm |
| `benchmarks/results/longmemeval_qa/quality.jsonl` | 619 | 2500 (500 units × 5 arms) | still in the `pamin` arm |

Neither arm had reached `pamin-wide` or `mempalace-raw` yet; `qa` had not
reached `mem0` (the slow one) yet either.

**These checkpoint files are not in this commit and will not be in any
commit.** `benchmarks/README.md` is explicit that conversation/user-derived
per-question rows stay out of git because they can contain private content
and the run is still writing them — that rule applies here same as anywhere
else in this harness, mid-run or not. If your session gets the same
container/filesystem this one ran in, the files above are still on disk at
those paths and `run.py` resumes from them automatically. If you're in a
**fresh** container (a new cloud session normally is), they're gone and both
arms restart from row 0 — that's a real cost (hours of compute, already-run
`pamin`/`bm25` rows), not a formality, so say so to whoever is waiting on
this if it happens.

## Why it was stopped

Asked to hand off to another team member/session; nothing was failing. The
container had just been reclaimed and rebooted (fresh PID 1 at 02:45 UTC,
~10 minutes after the last confirmed-healthy check), which coincidentally
left shim/pamin/both run.py processes already down — a clean point to stop
rather than restart them. The two scheduled check-in triggers that were
driving this (a 20-minute self-re-arming `send_later` chain and an hourly
safety-net cron) were both deleted so this session's history doesn't keep
firing into empty air after handoff.

## Incident history (so you don't re-diagnose it)

**2026-10-12 01:01–01:30 UTC: stale postgres lock files after a reclaim.**
A container reclaim killed postgres mid-spawn and left two stale files in
`$PAMIN_HOME` (`/tmp/bench/pamin-home` in this run):
`postgres/data/postmaster.pid` (referencing a dead PID) and `pamin.spawning`
(an empty marker). The first made `pg_ctl` fail with "lock file already
exists"; once that was removed, retries *hung indefinitely with no output*
instead of failing, because the second stale marker made pamin wait forever
on a spawn that would never finish. Fixed by confirming (via `ps -p <pid>`)
neither referenced process was actually running, then removing both files
and retrying — pamin came back in ~4s. If `pamin server FAILED to come back
up after 3 attempts` shows up again, check for these two files before
assuming anything else is wrong.

No other incidents. Every other reclaim during this run cleared on restart
with zero data loss (checkpoint-resume duplicates — the same unit logged
twice, once with real elapsed time and once ~1s — are the normal signature
of that and not an error).

## Reproducing the environment in a new container

None of this is committed anywhere (per the harness's own venv-isolation and
no-committed-raw-data rules), so a fresh container needs it rebuilt. This
is what was actually on disk in this one, confirmed at handoff time:

```sh
# pamin itself
cd /home/user/PaminMemory && cargo build --release

# mem0, in its own venv -- the plain extras are NOT enough, see
# benchmarks/README.md's "half-disabled trap" section
python3 -m venv /tmp/bench/mem0-venv
/tmp/bench/mem0-venv/bin/pip install 'mem0ai[nlp,extras]' qdrant-client fastembed openai
# installed versions at handoff: mem0ai 2.2.1, qdrant-client 1.19.1,
# fastembed 0.8.1, openai 3.26.0 (openai client is only used for its
# request/response shapes against the shim, no key needed)

# MemPalace, in its own venv -- plain PyPI install, nothing custom
python3 -m venv /tmp/bench/mp-venv
/tmp/bench/mp-venv/bin/pip install mempalace==3.10.0

# the shared shim -- claude -p behind an OpenAI-shaped endpoint, no API key
SHIM_MODEL=haiku SHIM_LOG=/tmp/bench/shim.log \
  python3 /home/user/PaminMemory/benchmarks/shim.py 8088 &

# pamin's resident server needs the system postgres, not whatever else may
# be on PATH, and needs to run as the unprivileged owner of its data dir
# (postgres refuses to run as root) -- PAMIN_HOME derives that user
PAMIN_HOME=/tmp/bench/pamin-home PAMIN_POSTGRES_DIR=/usr/lib/postgresql/16 \
  su ubuntu -c "/home/user/PaminMemory/target/release/pamin --project smoketest search ping --limit 1 --json"
```

### The two run commands

```sh
cd /home/user/PaminMemory/benchmarks

PAMIN_BIN=/home/user/PaminMemory/target/release/pamin \
BENCH_USER=ubuntu BENCH_WORK=/tmp/bench BENCH_PAMIN_HOME=/tmp/bench/pamin-home \
MEMPALACE_PYTHON=/tmp/bench/mp-venv/bin/python RUN_TAG=lmer \
BENCH_SHARED_ENDPOINT=1 /tmp/bench/mem0-venv/bin/python3 -u run.py \
  --dataset longmemeval --mode quality \
  --arms bm25,pamin,pamin-wide,mempalace-raw --units 470 --questions 1 \
  --out /home/user/PaminMemory/benchmarks/results/longmemeval/retrieval.jsonl \
  >> /tmp/bench/lme-retrieval.log 2>&1 &

PAMIN_BIN=/home/user/PaminMemory/target/release/pamin \
BENCH_USER=ubuntu BENCH_WORK=/tmp/bench BENCH_PAMIN_HOME=/tmp/bench/pamin-home \
MEMPALACE_PYTHON=/tmp/bench/mp-venv/bin/python RUN_TAG=lmeqa \
BENCH_SHARED_ENDPOINT=1 /tmp/bench/mem0-venv/bin/python3 -u run.py \
  --dataset longmemeval_qa --mode quality \
  --arms bm25,pamin,pamin-wide,mem0,mempalace-raw --units 500 --questions 1 \
  --out /home/user/PaminMemory/benchmarks/results/longmemeval_qa/quality.jsonl \
  >> /tmp/bench/lme-qa.log 2>&1 &
```

Both are safe to re-run as-is any number of times: `run.py` checkpoints and
skips rows already in `--out`.

### A resilience/check-in loop, if you want one

This run was being kept alive across container reclaims and usage-limit
resets by re-running an idempotent restart check every ~20 minutes (via this
session's own `send_later`), plus an hourly cron-style safety net as a second
independent path. Neither survives this handoff — they were tied to this
session and both were deleted. If you want the same resilience, recreate the
equivalent of: check the shim (`curl :8088/stats`), check pamin via the
smoketest search above with up to 3 retries 10s apart, check whether each
`run.py` is still running (`pgrep -f "run.py --dataset longmemeval"`) and
re-launch with the exact commands above if not, then reschedule. Watch for
the stale-lock-file failure mode described above before assuming a repeated
pamin-startup failure is something new.

## Still open

- **Scope decision, unanswered.** The revised full-scope timeline was
  reported as roughly 1.5–2 weeks total (retrieval-only in hours, qa mostly
  gated by `mem0`'s slow ingest). Whether to keep full scope or trim it
  (drop `pamin-wide`, sample fewer units) was never decided — nobody
  responded. Don't restart this run assuming either answer; ask.
- **Task list**, from the broader plan this run is part of:
  - #1–4, #6: done (mem0/MemPalace venvs, shared shim, LOCOMO re-run, MIRACL
    loader).
  - #5 (this run): in progress, stopped for handoff as above.
  - #7: direct MIRACL full-corpus comparison (Haiku write path) — not
    started.
  - #8: update `docs/benchmarks.md`, `README.md`, `benchmarks/results/` with
    whatever #5 and #7 produce — not started. Follow `.agents/skills/pamin-dev`
    before writing any numbers (measure the real entry point, assert each
    arm's premise, every published figure needs a committed summary behind
    it).
  - #9: open a PR — not started. No PR exists for this branch yet; it is
    pushed to `origin/bench/mem0-mempalace-miracl-head-to-head` through
    commit `211a18f`.
  - There's also an older, separate 59-question `longmemeval.py` table
    already on the `docs/benchmarks.md` page; whether this new 470/500
    question run supersedes it or sits beside it as a different scope was
    never decided either — flagged but not resolved before this handoff.
