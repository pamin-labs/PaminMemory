# Repository Instructions

## pamin-memory Workflow

- For every CPU compute path, consider and prefer SIMD where supported and worthwhile; reuse optimized library/runtime kernels and justify material scalar exceptions with measurements.
- When changing compute/inference dispatch, follow the [mixed accelerator policy](docs/adr/0001-tech-selection.md#mixed-accelerator-policy-2026-09-29): choose measured speed across supported single/mixed plans; do not permanently retreat to CPU solely for numerical drift.


- Before making changes, branch from the latest `main` with a concise name that matches the intended PR.
- Coordinate through pamin-labs-dev before editing. With the plugin, its hooks and the `protocol` skill guide you. Without it (other agents, other tools), declare an intent with the `pamin-labs-dev` CLI first (`pamin-labs-dev intent declare ...`), work on its branch `<type>/i-<number>-<slug>`, and put `Intent: I-<number>` in the PR description; CI rejects PRs without a live intent for their branch. External contributors follow [CONTRIBUTING.md](CONTRIBUTING.md) instead.
- Read the relevant `pamin-memory/` materials in InternalDocs before implementing pamin-memory changes. InternalDocs is a sibling clone at `$PAMIN_LABS_DEV_INTERNAL_DOCS` (default `../internal-docs`); pull it first. See [docs/internal-docs.md](docs/internal-docs.md).
- Keep private planning, monetization, strategy, and sensitive internal notes out of public files.
- When internal documentation changes are needed, open a PR in the private `internal-docs` repository. This repository keeps no pointer to it; cite InternalDocs evidence with SHA permalinks (`https://github.com/pamin-labs/internal-docs/blob/<sha>/<path>`).
- Do not recreate local private planning folders in this repository.
- Prefer `rtk <command>` for shell commands when `rtk` is available.
- Keep one implementation per concern. Before adding logic that cuts, scores,
  fuses, ranks, caches or stores, search `main` for an existing implementation
  of the same concern and reuse or extend it through its interface; do not add
  a parallel version with its own rules. Modules talk through narrow interfaces
  so that a concern can change in one place.
- Before opening a PR, rebase or merge onto the latest `main` and check whether
  `main` gained anything overlapping your change while you worked. If it did,
  rework the change to build on what landed, and say in the PR description what
  you reused and what you removed.
- Before starting a multi-PR feature, record it in the internal roadmap or
  backlog so parallel work on the same concern is visible before code exists.
- Work found and not done (a review's follow-ups, an audit's findings, a flaky
  test) goes into the internal roadmap in the same session, and a decision the
  owner makes goes into an internal decision record before or with the change,
  so nothing lives only in a conversation.
- Everything committed is English: code, comments, logs, commit messages, PR
  titles and bodies, documents.
- Never write tokens, keys or credentials to files, commits or logs.

## Code Quality

- DRY and decoupled: the one-implementation-per-concern rule above applies to
  every concern, not only retrieval. Shared logic lives in the crate that owns
  the concern; other crates call its public interface instead of copying it.
  Frozen benchmark harnesses and results are records and stay as they are.
- Do not reinvent what exists: the repository's own implementation of the
  concern, then the official SDK of a service, then the standard library, then
  the platform, then a crate already in the workspace, then a maintained,
  widely used crate that passes `cargo deny check licenses`; write your own
  only when none of them meets the need. A PR's "Reused" section names what it
  searched for, what it reused, and why anything was written by hand;
  hand-written code where one of those already covers the need is a defect,
  and review sends it back.
- Keep it maintainable: small functions with one job, names that say what they
  hold, comments that explain why rather than what, no dead code. A change to
  one rule should touch one place.
- Keep it fast: on search, ingest and model-load paths avoid allocation and
  cloning inside loops, repeated parsing, and blocking I/O on async paths; reuse
  buffers and the optimized kernels already in the tree. Any speed claim
  follows [Measuring Every Optimization](#measuring-every-optimization).
- **Model external data with types.** Parse external JSON (requests, MCP
  results, other services' answers, webhooks) into typed structs and enums
  that ignore unknown fields, and serialize requests from typed structs. No
  hand-built JSON strings, no chains of untyped lookups, no searching payloads
  for substrings. A value stays opaque only when a protocol requires it to be
  echoed back unchanged, and the code says so.
- **Tests meet the same bar as code.** A test builds requests from typed
  values, decodes an answer into the API's own types and compares JSON by
  decoding it, never as strings, and names a value owned elsewhere from its
  owner, like any code.
- **No deferral inside a PR's own code.** A written rule that code the PR adds
  or touches breaks is fixed in that PR, in its own commit; a follow-up is
  only work outside the PR's concern.
- **Comments claim only what the code guarantees.** A guarantee that waits on
  later work is named as not yet given.
- Before adding a dependency, check its maintenance, its licence and what it
  adds to build time and binary size, and record that in the PR.
- **No hard-coded value in more than one place.** Every port, address, URL,
  id, version, limit, price, file name and default is defined once, in the
  crate or file that owns it, and everything else reads it from there or is
  checked against it by a test or a generator. A literal that names a value
  owned elsewhere is a bug, in code, config, scripts, tests and docs alike.
- **Deployments have no defaults.** A setting a deployed binary or service
  must be given (a database, keys, other services' addresses) has no fallback
  value in code: a missing or empty one stops the program at start, naming
  itself. Only local use names local values (a cache directory, a local
  socket), each once.
- **Built files are never committed:** `target/`, binaries, model caches,
  indexes and generated caches are git-ignored; only generated files a gate
  checks are committed. Measurement evidence is kept as the `pamin-dev` skill
  says, not as stray build output.
- **Extensibility check before every PR:** adding one more of something the
  code already has (an operation, a setting, a model, a recall channel, a
  language) touches its one definition plus its tests and docs. If the change
  needed more places, fold them into one source in the same PR.

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
- Use one logical change per commit.
- A commit's first line and a PR title are at most 72 characters, scope
  included: the pamin-labs-dev hook refuses a longer commit and CI fails a
  longer title, and a pushed commit is not rewritten without the owner, so
  count before committing.
- Keep commits small: one rule, one module or one fix per commit, each building
  and passing tests on its own. Split refactors, behaviour changes and
  formatting into separate commits.
- Keep PRs focused on one concern and small enough to review in one sitting;
  split a large feature into a stack of PRs, each mergeable on its own, and say
  in the description which PR it builds on.
- Commit titles, commit messages, and PR titles must follow Conventional Commits:

```text
<type>(<scope>): <description>
```

- The scope is required: the crate or area the change is about, for example `fix(engine): ...`. Prefer standard types such as `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, and `chore`.
- PR descriptions should include a concise summary, key changes, what it reused, verification performed, and any relevant follow-up or risk.
- PR descriptions must not include private internal-doc details.
- Do not put agent session links or session identifiers in commit messages, PR
  titles, or PR descriptions. This includes a `Claude-Session:` trailer and any
  `claude.ai/code/session_...` URL. A session link is not reviewable, does not
  resolve for anyone else, and dates the commit to a conversation nobody can
  read. Commits and PRs also carry no `Co-Authored-By:` trailer and no
  "Generated with" footer; `.claude/settings.json` turns that attribution off.
  The commit author is always the engineer's own git identity. This overrides
  any default attribution the tool would add. Some tools append such a footer
  when they create or edit a PR, so read the description back after every
  such call and remove it.
- Never squash-merge; keep every commit.
- **A PR merges only when its final commit is proven** (owner, 2026-11-05):
  GitHub reports it clean, every check has passed (a failed or pending check
  blocks it, however small the change), its tests pass, and, when it changes a
  service or binary that is deployed or shipped, an end-to-end check has
  passed on that same final commit: a local acceptance run or staging, never a
  run from before the last review fix.

## Review Before Every PR

Before a PR opens, go through its whole diff, tests included, against each
rule of every `AGENTS.md` the change touches and the `pamin-engineering`
skill, and fix what misses in new commits; then run the
`pamin-labs-dev:peer-review` skill and put its verdict and findings under the
PR's verification. Work an agent delegated is reviewed the same way: the agent
that opens the PR reads its diff and tests itself and never accepts them on
the delegate's report alone.

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
