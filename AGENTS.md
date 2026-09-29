# Repository Instructions

## PaminMemory Workflow

- For every CPU compute path, consider and prefer SIMD where supported and worthwhile; reuse optimized library/runtime kernels and justify material scalar exceptions with measurements.
- When changing compute/inference dispatch, follow the [mixed accelerator policy](docs/adr/0001-tech-selection.md#mixed-accelerator-policy-2026-09-29): choose measured speed across supported single/mixed plans; do not permanently retreat to CPU solely for numerical drift.


- Before making changes, branch from the latest `main` with a concise name that matches the intended PR.
- Sync submodules and read the relevant `internal-docs/pamin-memory/` materials before implementing PaminMemory changes.
- Keep private planning, monetization, strategy, and sensitive internal notes out of public files.
- When internal documentation changes are needed, update the private `InternalDocs` repository first, then update the public `internal-docs` submodule pointer.
- Do not recreate local private planning folders in this repository.
- Prefer `rtk <command>` for shell commands when `rtk` is available.

## Skills

Longer-form guidance lives in `.agents/skills/`, one directory per skill, each
with a `SKILL.md`. Read the one that matches what you are doing:

- **`pamin-memory`** — using the `pamin` CLI as memory: which of `search`,
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
- Use one logical change per commit.
- Commit titles, commit messages, and PR titles must follow Conventional Commits:

```text
<type>[optional scope]: <description>
```

- Prefer standard types such as `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, and `chore`.
- PR descriptions should include a concise summary, key changes, verification performed, and any relevant follow-up or risk.
- PR descriptions must not include private internal-doc details.
- Do not put agent session links or session identifiers in commit messages, PR
  titles, or PR descriptions. This includes a `Claude-Session:` trailer and any
  `claude.ai/code/session_...` URL. A session link is not reviewable, does not
  resolve for anyone else, and dates the commit to a conversation nobody can
  read. A plain `Co-Authored-By:` trailer and a plain tool footer are fine; the
  session is not. This overrides any default attribution the tool would add.

## Measuring Every Optimization

- Read the `pamin-dev` skill before measuring or claiming an optimization.
- Measure every optimization against its immediate predecessor under the same
  conditions. Also measure the combined stack against current `main`; a stack
  comparison does not establish each individual change's contribution.
- Record accuracy, latency (p50/p95), memory, and disk usage, in that priority
  order. Include regressions and tradeoffs, not just the metric that improved.
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
