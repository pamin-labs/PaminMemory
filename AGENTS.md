# Repository Instructions

You must follow `.agents/skills/pamin-engineering/references/core.md`: @.agents/skills/pamin-engineering/references/core.md

## pamin-memory Workflow

- Read the relevant `pamin-memory/` materials in InternalDocs before implementing pamin-memory changes. InternalDocs is a sibling clone at `$PAMIN_LABS_DEV_INTERNAL_DOCS` (default `../internal-docs`); pull it first. See [docs/internal-docs.md](docs/internal-docs.md).
- Keep private planning, monetization, strategy, and sensitive internal notes out of public files.
- When internal documentation changes are needed, open a PR in the private `internal-docs` repository. This repository keeps no pointer to it; cite InternalDocs evidence with SHA permalinks (`https://github.com/pamin-labs/internal-docs/blob/<sha>/<path>`).
- Do not recreate local private planning folders in this repository.
- Before starting a multi-PR feature, record it in the internal roadmap or
  backlog so parallel work on the same concern is visible before code exists.
- External contributors follow [CONTRIBUTING.md](CONTRIBUTING.md) instead of
  the pamin-labs-dev workflow.
- PR descriptions must not include private internal-doc details.

## Code Quality

- Shared logic lives in the crate that owns the concern; other crates call its
  public interface instead of copying it. Frozen benchmark harnesses and
  results are records and stay as they are.
- A new dependency passes `cargo deny check licenses`.
- Hot paths here are search, ingest and model load (`performance.md`); any
  speed claim follows [Measuring Every Optimization](#measuring-every-optimization).
- Before adding logic that cuts, scores, fuses, ranks, caches or stores, search
  `main` for an existing implementation of the same concern (`core.md`, "One
  implementation per concern").
- Model caches and indexes are built files, never committed; measurement
  evidence is kept as the `pamin-dev` skill says, not as stray build output.
- Extending memory means one more operation, setting, model, recall channel or
  language: `core.md`'s extensibility check applies to each.

## How the shared rules apply here

- The coordination tool is pamin-labs-dev: its `protocol` skill, its `pamin-labs-dev:commit` skill for commits and PRs, and `pamin-labs-dev:peer-review` for the review before every PR, which also checks the change against the `pamin-engineering` skill.
- The roadmap and decision records are in InternalDocs; a published library's caret minimums are InternalDocs `pamin-platform/decisions/2026-11-01-library-dependency-requirements.md`.
- Prefer `rtk <command>` for shell commands when `rtk` is available.
- Provenance: the layer and domain rules came from an audit of Tala's server on 2026-11-03, and the errors, data, settings and calls rules from Tala's decisions of 2026-11-01 to 2026-11-05; the owner set the reuse ladder and the review before every PR on 2026-11-04 and the proven-final-commit merge rule on 2026-11-05.
- A module served by the Pamin platform is metered by the platform, never by its own code.

## Areas

Each area has its own `AGENTS.md` with its stack, its rules and its gates;
read it before changing code there.

| Folder | What | Instructions |
| --- | --- | --- |
| `crates/` | The Rust workspace: the engine, its store and index, the contract, the CLI | `crates/AGENTS.md` |

Measurement harnesses in `benchmarks/`, `maintenance/` and `ci/` follow the
`pamin-dev` skill.

## Gates Before Every Push

Run the gates of every area a change touches; each area's `AGENTS.md` lists
them.

## Skills

Longer-form guidance lives in `.agents/skills/`, one directory per skill, each
with a `SKILL.md`. Read the one that matches what you are doing:

- **`pamin-engineering`** — the engineering bar every Pamin Labs repository
  shares: read it before writing or changing code, adding a dependency,
  claiming a speed-up, or reviewing.
- **`pamin-memory`** — using the `pamin-memory` CLI as memory: which of `search`,
  `read`, `grep` and `neighbors` answers which question, how to read the `why`
  trace, and the traps around the evidence filter.
- **`pamin-dev`** — measuring this project and keeping its claims true: measure
  the entry point the product calls, make every measurement arm assert its own
  premise, keep harnesses out of the tracked tree, and audit documentation for
  claims the code stopped backing.

`.claude/skills/` holds symlinks to the same directories, the way `CLAUDE.md`
symlinks to this file, so Claude Code discovers them without a second copy
going stale.

## Commits And Pull Requests

- Use the `git-commit` skill whenever possible when creating commits.
- The scope of a commit or PR title is the crate or area the change is about.
- A PR in a stack says in its description which PR it builds on.
- `.claude/settings.json` turns agent attribution off.

## Measuring Every Optimization

- Read the `pamin-dev` skill before measuring or claiming an optimization.
- Measure every optimization against its immediate predecessor under the same
  conditions. Also measure the combined stack against current `main`; a stack
  comparison does not establish each individual change's contribution.
- Record accuracy, latency (p50/p95), memory, and disk usage, in that priority
  order. Include regressions and tradeoffs, not just the metric that improved.
- Record elapsed (wall) time separately from CPU user/system time. For aggregate
  time comparisons, hold the amount of work fixed and separate startup, index
  preparation, search calls, and diagnostics. CPU time includes worker threads
  and can exceed elapsed time; declare whether child processes, CoreML services,
  and GPU/NPU device execution are included. Process CPU time alone is not total
  accelerator cost. Retain missing historical measurements as `N/A`.
- Every optimization PR must include a table with **metric, before, after,
  absolute difference, and percentage change**, plus links to retained evidence.
  Mark unmeasured or inapplicable cells `N/A` and explain why; never imply zero
  regression or claim an unmeasured gain.
- Identify the compared commits, hardware, actual execution providers, model
  revisions/precision, corpus/query order, settings, and cold/warm conditions.
  State whether memory is process RSS or includes accelerator services, and
  whether disk figures include model caches, indexes, and temporary files.
- Measure the actual product entry point for product claims. Label component
  measurements separately; they cannot establish whole-search improvements.
- Repeat and rotate timing arms to distinguish improvements from noise. Record
  shared-machine interference and keep our builds and other model experiments
  out of timing runs. A single pass is provisional, not proof of a speedup.
- Retain reproducible commands, raw results, and a durable summary following
  `pamin-dev`'s public/private and harness-location rules. Publish supported
  results in PRs and relevant benchmark/ADR documentation; do not mark an
  optimization validated while its claimed improvement remains unmeasured.

## Behavioral Guidelines

The guidance below is adapted from Forrest Chang's Karpathy-style coding-agent guidelines:
https://github.com/forrestchang/andrej-karpathy-skills/blob/main/CLAUDE.md

These rules bias toward caution over speed. For trivial tasks, use judgment and keep the process lightweight.

### Think Before Coding

- State assumptions when they affect the implementation.
- Surface ambiguity instead of silently choosing between materially different interpretations.
- Name tradeoffs when a simpler or safer path exists.
- Ask for clarification when the uncertainty cannot be resolved from the repository.

### Keep Changes Simple

- Implement the minimum code or documentation needed to satisfy the request.
- Do not add speculative features, configurability, or abstractions.
- Avoid defensive handling for impossible states unless the repository already follows that pattern.
- If a solution is becoming large, look for a simpler design before continuing.

### Make Surgical Edits

- Touch only files and lines that connect directly to the request.
- Match existing style even when another style would also be reasonable.
- Do not refactor, reformat, or clean adjacent code unless the task requires it.
- Remove imports, variables, functions, or text made unused by your own change.
- Mention unrelated dead code or cleanup opportunities instead of deleting them.

### Work Against Verifiable Goals

- Convert requests into concrete success criteria before finishing.
- For bug fixes, prefer a reproduction check before the fix when practical.
- For refactors, verify behavior before and after when practical.
- For documentation changes, check links, public/private boundaries, and formatting.
- Keep iterating until the agreed verification passes or report the exact blocker.
