---
name: pamin-engineering
description: The Pamin Labs engineering bar for code in this repository - fast, small, quick to build, clean, DRY, decoupled and easy to extend. Use it before writing or changing any code, before adding a dependency, before claiming something is faster or smaller, and when reviewing a change. Reach for it on words like implement, refactor, optimise, add a crate or package, abstraction, interface, performance, memory, binary size, build time, or "is this over-engineered".
---

# The Pamin Labs engineering bar

Every Pamin Labs repository holds code to the same bar: high performance, low resource use, fast builds, clean code that is easy to maintain and extend. The rules below are how that bar turns into decisions while you work. They apply to Rust, Zig and TypeScript alike; the language notes at the end only add to them.

## Before writing anything

1. **Does it need to exist?** Build what a real caller needs now. No options, layers, traits or config for a second use that has not appeared; add them when it does.
2. **Does it already exist here?** Search the repository (and the shared crates `pamin-runtime` and `pamin-kit`) for the same concern. Extend the owner of a concern through its interface; never add a parallel version with its own rules.
3. **Does the official SDK of the service, then the standard library, then the platform, then a dependency already in the tree, then a widely used maintained dependency cover it?** Write custom code only when none fits, and say why in the PR's "Reused" section. Lock files pin every dependency to an exact version; a published library requires caret minimums instead, so its consumers can unify, and pins exactly only with a stated reason.

## Shape of the code

- **One rule, one place.** A change to a rule touches one function or module. If you need to edit the same logic in two places, the logic is in the wrong place.
- **Decoupled through narrow interfaces.** Components talk through small, explicit interfaces; dependencies point one way (adapters → contract → engine → runtime). An engine never knows about transports, auth or billing.
- **Small functions with one job; names that say what they hold; comments that say why, not what; no dead code.**
- **Errors carry context** and are handled where something can be done about them; nothing is swallowed.
- **Trust boundaries are explicit.** Identity, tenant and project come from authentication, never from a request body, a tool argument or a model.

## Performance and resources

- **Hot paths:** no allocation or cloning inside loops, no repeated parsing, no blocking I/O on async paths, static dispatch. Reuse buffers and the optimised kernels that already exist.
- **Resources:** idle processes hold little memory; caches are bounded and evict; work that can wait runs in the background with a budget.
- **Measure before and after** any change that claims to be faster, smaller or cheaper, on the entry point the product actually calls, and put both numbers and how they were taken in the PR. A claim without a measurement is removed.

## Build speed and footprint

- Keep dependency trees shallow; prefer one well-chosen crate or package over several small ones, and turn off default features you do not use.
- Split code so that an edit rebuilds as little as possible; keep heavy dependencies (inference, TLS) behind the crates or features that need them.
- Release builds are as small as the language allows without losing safety checks the product relies on.

## Before opening a PR

- Branch `<type>/i-<number>-<slug>` from your pamin-labs-dev intent, Conventional Commits, one logical change per commit, each building and passing on its own.
- Run the gates of every `AGENTS.md` the change touches, locally.
- Rebase on the latest `main`; if `main` gained something overlapping, build on it and say what you reused.
- Do the review before every PR that `AGENTS.md` describes: the whole diff against every rule, then the `pamin-labs-dev:peer-review` skill.
- The PR says what changed, what it reused, how it was verified (with numbers when performance is involved), and any risk or follow-up.

## Language notes

- **Rust:** edition and toolchain pinned in `rust-toolchain.toml`; `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo test`; `cargo deny` for licences and bans; a shared `CARGO_TARGET_DIR` keeps disk use down; `debug = "line-tables-only"` in dev profiles. The `## Rust` and `## Performance` rules in the `AGENTS.md` that covers the repository's Rust code add to these.
- **Zig:** compiler version pinned in the repository; `ReleaseSafe` for production; one arena allocator per request; `std.testing.allocator` in tests to catch leaks; fuzz parsers.
- **TypeScript:** strict mode; Bun for installs and scripts; measure the shipped JavaScript and keep a budget; no dependency for what the platform already does (CSS over JavaScript, native elements over widgets).
