# Repository Instructions

## PaminMemory Workflow

- Before making changes, branch from the latest `main` with a concise name that matches the intended PR.
- Sync submodules and read the relevant `internal-docs/pamin-memory/` materials before implementing PaminMemory changes.
- Keep private planning, monetization, strategy, and sensitive internal notes out of public files.
- When internal documentation changes are needed, update the private `InternalDocs` repository first, then update the public `internal-docs` submodule pointer.
- Do not recreate local private planning folders in this repository.
- Prefer `rtk <command>` for shell commands when `rtk` is available.

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
