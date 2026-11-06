---
name: pamin-engineering
description: The engineering bar for code in this repository - fast, small, quick to build, clean, DRY, decoupled and easy to extend. Use it before writing or changing any code, before adding a dependency, before claiming something is faster or smaller, and when reviewing a change, even when the change looks small. Reach for it on words like implement, refactor, optimise, add a crate or package, abstraction, interface, layer, handler, migration, error code, setting, performance, memory, binary size, bundle size, build time, or "is this over-engineered".
---

# The engineering bar

Every repository that carries this skill holds code to one bar: high performance, low resource use, fast builds, and clean, DRY, decoupled code that is easy to maintain and extend. The rules live once, in `references/`, and each `AGENTS.md` imports the ones its code needs. This file is how to use them while you work; it states no rule of its own.

## Decide before you write

Walk these in order, and stop at the first that answers.

1. **Does it need to exist?** Is a real caller asking for it now? If it is for a second use that has not appeared (an option, a layer, an interface, a setting), do not build it.
2. **Does it exist already?** Search this repository, then the shared libraries it depends on, for the same concern. If an owner exists, extend it through its interface.
3. **Can something else do it?** Climb the reuse ladder in `core.md` ("Do not reinvent what exists"): the official SDK, the standard library, the platform, a dependency already in the tree, then a new maintained one. Writing it yourself is the last rung, and the PR's "Reused" section says why.
4. **Where does it go?** Find its layer and directory in the `AGENTS.md` of the area you are in ("Layers here", the layout or module list). If no directory fits, the layout changes in the same PR.
5. **Will it be faster or smaller?** Then measure before and after, on the entry point the product calls, and put both numbers in the PR. No number, no claim.

## Which reference to read

Read every reference the change touches, before writing and again before the PR opens.

| Reference | Read it when |
| --- | --- |
| `references/core.md` | Always: shape of code, layers and domains, errors, data, settings, calls to other services, resources, review, commits and PRs |
| `references/rust.md` | Writing or reviewing Rust: toolchain, errors, panics, `unsafe`, async, tasks, SQL, tests |
| `references/go.md` | Writing or reviewing Go: tools, file layout, errors, context, concurrency, types, tests |
| `references/performance.md` | Touching a request's hot path, a pool, start-up, or a benchmark |
| `references/web.md` | A React, Vite and TypeScript web client or operator console |
| `references/workers.md` | A Cloudflare Worker: routing, containers, config, deploys |

`AGENTS.md` files import these with `@` lines, so Claude Code loads them already; other tools open the paths they name.

## When the reference and the repository disagree

- The repository's `AGENTS.md` says how it meets each rule and what it has not built yet; it never weakens a rule. A gap it names is not a licence to add to it: new code follows the rule.
- A rule that seems wrong for a case is raised with the owner and recorded in a decision record, then changed in `references/` in every repository together. Do not work around it in code.

## Reviewing a change

1. Read the whole diff, tests included.
2. Hold it against every rule of each reference its files import, and each rule of the `AGENTS.md` files it touches.
3. Check the extensibility test in `core.md`: would one more of the same thing touch one place?
4. Fix what misses in new commits, then run the peer-review skill, as `core.md`'s "Review before every PR" says.

## Before the PR opens

1. Rebase on the latest `main`, and build on anything overlapping that landed.
2. Run the gates of every `AGENTS.md` the change touches, locally.
3. Review it as above, then write the description and commits as `core.md`'s "Commits and PRs" asks; the coordination tool's commit skill checks the mechanics.
4. Put measurements, with how they were taken, under the PR's verification.

## Where a question goes

- A rule for every language: `core.md`.
- How one language meets it: that language's reference.
- How this repository meets it, its layers, its crates or packages, its gates: its own `AGENTS.md`.
- Why a rule exists, and who decided it: the decision records, cited with a SHA permalink.

## Changing these rules

`references/` is identical in every repository that carries this skill. A change to a reference is made in all of them together, cites its decision record, and adds no repository-specific fact: those stay in the repository's own `AGENTS.md`.
