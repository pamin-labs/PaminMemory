# Rust Workspace Instructions

The crates of pamin-memory, one Cargo workspace. The repository's
`AGENTS.md` applies too; this file adds what is particular to Rust code.

You must follow `../.agents/skills/pamin-engineering/references/rust.md`: @../.agents/skills/pamin-engineering/references/rust.md
You must follow `../.agents/skills/pamin-engineering/references/performance.md`: @../.agents/skills/pamin-engineering/references/performance.md

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

## Layers here

The crates under "Crates" above take these layers, and their dependencies point one way: `memory-cli` → `memory-client` and `memory-contract` → `memory-engine` → `memory-store` and `memory-index` → `memory-core`.

- **Entry and wiring:** `crates/cli/src/main.rs` and `pamin_memory_server.rs` read their arguments; `server.rs` and `session.rs` build the pool, the engines and the models once, and `watch.rs` is the server's background sync.
- **Transport:** `memory-client` (arguments, the socket protocol, MCP and rendering), `memory-contract` (each operation's request and result types) and the server's handlers in `crates/cli/src/command/`.
- **Domain:** `memory-engine`, which composes the store, the index and the embedder, and the rules of `memory-core`.
- **Store:** `memory-store`, the only crate that names `sqlx` and the tables, on any PostgreSQL its user names; "Per-transaction state only" binds it: a project's transaction sets its role and project with `SET LOCAL`.
- **Integrations:** `memory-index`, the projection index and local inference through `pamin-runtime`.
- **Values:** `memory-core`'s types, which every crate shares; it depends on no other crate of the workspace.

Checks: `crates/client/tests/light.rs` reads `cargo metadata` and fails when the engine reaches an adapter or the contract, or the client reaches what only the server needs; a test of every layer's direction is not yet written; on the roadmap. `migrations_stay_within_the_portable_sql_subset` (`crates/store/src/migrate.rs`) refuses `LISTEN`, `NOTIFY` and advisory locks in migrations. The test that each store function is called from one domain only is not yet written; on the roadmap. Today the handlers in `crates/cli/src/command/` still call `memory-store` and `memory-index` directly; new code does not add to that, and moving it behind the engine is its own change.

## How this repository meets the errors, data, settings and calls rules

Where pamin-memory differs from a rule above, or has not built it yet:

- **Values:** `memory-core` imports no other crate of the workspace, and `memory-store` imports only `memory-core`.
- **Errors:** the store and the index declare `StoreError` and `IndexError`; a failure crosses the socket and MCP as a message (`memory_client::protocol::Response`), not yet as a kind and a stable code.
- **API documents:** the operations' types are `memory-contract`'s; there is no API document beyond the MCP tools' schemas that `pamin-kit` builds from them.
- **Isolated APIs:** one API, the operation contract, served on the local socket and as MCP; no operator API.
- **Transactions and charged writes:** the store is the authority and its cascade outbox records, in the write's transaction, what the index must still catch up on; nothing here charges, so no write takes a client id.
- **Queries and migrations:** migrations in `crates/store/migrations/` are written by hand, numbered and append-only, and `migrations_stay_within_the_portable_sql_subset` checks them; the check for locking and destructive changes is not yet written; on the roadmap. `Database`'s project transaction still builds its `SET LOCAL` statement with `format!`.
- **Settings:** settings are resolved once by `memory_client::settings` (flags over `config.toml` over defaults, with the `PAMIN_MEMORY_*` variables) and handed to the engine by `crates/cli/src/settings.rs`; there is no `.env.example`, product settings table, kill switch or operator API: those have no subject in a local tool.
- **Calls and models:** the engine's own inference (embedding and reranking) goes only through `memory-index`, on `pamin-runtime`; there is no external service on a request's path. Metering and cost alerts belong to whoever hosts memory, not to this repository.
- **Stubs and logs:** the test databases come from `postgresql_embedded`; logs go through `tracing` with ids, counts and durations.

## Compute

- For every CPU compute path, consider and prefer SIMD where supported and worthwhile; reuse optimized library/runtime kernels and justify material scalar exceptions with measurements.
- When changing compute/inference dispatch, follow the [mixed accelerator policy](../docs/adr/0001-tech-selection.md#mixed-accelerator-policy-2026-09-29): choose measured speed across supported single/mixed plans; do not permanently retreat to CPU solely for numerical drift.

## Gates

`cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --all-targets
--all-features`, `cargo test --workspace`, `cargo deny check licenses`; the
ignored tests that need PostgreSQL or model weights, for the crates a change
touches, run locally with `-- --ignored` and are reported in the PR.
