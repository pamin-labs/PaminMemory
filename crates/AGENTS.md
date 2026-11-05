# Rust Workspace Instructions

The crates of pamin-memory, one Cargo workspace. The repository's
`AGENTS.md` applies too; this file adds what is particular to Rust code.

## Stack

- Rust with the edition and toolchain pinned in `rust-toolchain.toml`; every
  dependency is declared once in the root `Cargo.toml`'s
  `[workspace.dependencies]`, with a comment saying why and which features.
- MCP only through `pamin-kit`; instruction-set kernels, ONNX Runtime devices,
  thread budgets and prepared-model keys through `pamin-runtime`;
  PostgreSQL through `sqlx`, and `postgresql_embedded` for local and test
  databases; logs through `tracing`.
- The budgets workflow checks binary and distribution size and dependency
  count against `ci/budgets.json`, and reports build time; exceeding a budget
  is a trade recorded in the PR.

## Crates

This list names every crate by its path, with what it owns. A new crate goes
under `crates/` and is described here in the PR that adds it; no test reads
this list yet, so review checks it.

- `crates/core/` (`memory-core`): the domain model, version ledger semantics
  and retrieval fusion, the parts that stay stable whatever the store or
  index.
- `crates/store/` (`memory-store`): the PostgreSQL authority store: evidence,
  the version ledger and the cascade outbox, with its migrations.
- `crates/index/` (`memory-index`): the projection index, derived data:
  multilingual segmentation and the lexical and vector recall channels.
- `crates/engine/` (`memory-engine`): composes the store, the index and the
  embedder.
- `crates/contract/` (`memory-contract`): the operation contract, the request
  and result types the CLI, the socket and MCP share.
- `crates/client/` (`memory-client`): the command line's client half: its
  arguments, the wire protocol to the workspace's server, and rendering.
- `crates/cli/` (`memory-cli`): the `pamin-memory` and `pamin-memory-server`
  executables.

Business logic lives in `core`, composed by `engine`; `store` and `index` are
storage, and `contract`, `client` and `cli` are transport: they parse, call
the engine and render, and a rule found there is a bug. `memory-core` and
`memory-client` are meant for other repositories to depend on, so the
published-crate rules below apply to them.

## Rust

These rules are the same in every Pamin Labs Rust repository; change them in all of them together.

- **Toolchain and lints.** The edition and toolchain are pinned in `rust-toolchain.toml` and upgraded as their own change; code is `rustfmt`-clean and `clippy`-clean with warnings denied. A lint beyond clippy's defaults (a `clippy::pedantic` lint, `missing_docs`) is enabled in `[lints]` only after its findings on the current tree have been counted and found few and worth fixing, and the PR that enables it fixes them all.
- **Errors.** A library declares typed errors with `thiserror`, one enum per boundary with variants a caller can act on; `anyhow` or a boxed error only in binaries and tests. Errors travel with `?`, each layer adding what it was doing (a variant's field, or `.context(...)` in a binary), and are handled once: returned or logged, never both.
- **No panic reachable from input.** `unwrap`, `expect`, `panic!`, `unreachable!`, indexing or slicing that can go out of bounds, and arithmetic that can overflow stay out of every path a request, a file or another service's answer reaches. They are allowed in tests, at start-up before serving, and where an invariant rules the failure out, with an `expect` message that states the invariant.
- **`unsafe`** only where no safe API gives the result (FFI, a measured SIMD kernel), in the smallest block, with a `// SAFETY:` comment naming every invariant it relies on and a test that exercises it. A new crate that needs none starts with `#![forbid(unsafe_code)]`.
- **Borrow before cloning.** A `.clone()` added to satisfy the borrow checker on a hot path is a defect: restructure, borrow, or share an `Arc` and say why. Parameters take `&str`, `&[T]` or `impl AsRef<_>`; functions return owned values.
- **Types over strings and integers.** Ids, units and amounts are newtypes or standard types (`struct TenantId(Uuid)`, `Duration`), never a bare `String`, `i64` or `u64`; a closed set of values is an enum, never a string compared in code. Input is parsed into these types at the boundary (`serde`, `FromStr`, `TryFrom`), so inner code only sees valid values.
- **Typed JSON.** Bodies are `serde` types; walking a `serde_json::Value` by hand, `json!` for a body whose shape is known and `format!` building JSON are defects. A value a protocol requires to be echoed unchanged is a `Box<RawValue>`, and the code says why.
- **Published crates** (one another repository depends on) keep semver: a public struct or enum that may grow is `#[non_exhaustive]` or has private fields and a constructor, and a breaking change is called out in the PR and released as one. Features are additive: enabling one never removes or changes what another relies on, and the default features are the smallest set a typical user needs.
- **Dependency features.** Each dependency is declared once (in `[workspace.dependencies]` where there is a workspace) with only the features the code calls; `tokio`, `axum` and other large crates never take `full` or unused defaults. A new dependency's added clean build time (`cargo build --timings`) and release binary size are measured and recorded in the PR.
- **No blocking in async.** No blocking file or network call, `std::thread::sleep`, long CPU-bound loop or contended `std::sync` lock on an async task: use the async API, `spawn_blocking` or a dedicated thread. No lock guard is held across `.await` (clippy's `await_holding_lock` stays on); a `tokio::sync` lock only when the guard must cross one.
- **Everything queued is bounded.** Channels, queues, caches and per-key maps have a bound named once as a constant or a setting, and the code says what happens when it is reached (wait, refuse or evict); an unbounded channel is a defect outside tests.
- **Tasks are owned.** Every spawned task's `JoinHandle` is kept (or the task lives in a `JoinSet` or a `TaskTracker`) and awaited or aborted by its owner, so its panic or error is seen. Every branch of a `tokio::select!` is cancellation-safe, or a comment says why the lost work does not matter. A long-running service shuts down gracefully on SIGTERM: it stops accepting, finishes or hands over work in flight within a deadline, flushes what it buffers, then exits.
- **Shared state** is `Send + Sync`, owned by one type that keeps each lock next to the fields it guards; on a hot path prefer atomics or message passing to a lock, and never take two locks in different orders.
- **Logs** go through `tracing` with structured fields: ids, counts and durations, never a secret, a token, a request body or user content. A type that holds a secret redacts it in `Debug`.
- **SQL** (`sqlx`) binds every value (`$1`, `.bind`); a query built with `format!` from input is a defect, and what cannot be bound (an identifier) comes from a closed list in code. Migrations are append-only: a merged migration is never edited, renamed or removed; a change is a new migration.
- **Tests.** Unit tests sit beside the code in `#[cfg(test)] mod tests`; tests of a crate's behaviour through its public API go in its `tests/` directory. Tests are deterministic and independent: no `sleep` to wait for something (await the event, poll with a deadline, or pause tokio's clock), no dependence on test order, the wall clock, the network or state another test left; anything random uses a fixed seed. A fix comes with a test that fails without it.

## Performance

- **Hot paths allocate once.** Reuse buffers (`clear()` and refill, `with_capacity` from a known size); no `collect()` into a `Vec` only to iterate it once; no `to_string()` or `format!` for a value that is only compared or written out.
- **Static dispatch on hot paths** (generics, `impl Trait`); `dyn` at boundaries crossed once per request or less.
- **Connections:** a database pool's size is set explicitly, and a transaction holds its connection only for SQL, never across a call to another service.
- **Start-up** does not wait on other services: connect, fetch keys and load what can wait on first use or in the background, unless the program can answer nothing without it.

## Compute

- For every CPU compute path, consider and prefer SIMD where supported and worthwhile; reuse optimized library/runtime kernels and justify material scalar exceptions with measurements.
- When changing compute/inference dispatch, follow the [mixed accelerator policy](../docs/adr/0001-tech-selection.md#mixed-accelerator-policy-2026-09-29): choose measured speed across supported single/mixed plans; do not permanently retreat to CPU solely for numerical drift.

## Gates

`cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --all-targets
--all-features`, `cargo test --workspace`, `cargo deny check licenses`; the
ignored tests that need PostgreSQL or model weights, for the crates a change
touches, run locally with `-- --ignored` and are reported in the PR.
