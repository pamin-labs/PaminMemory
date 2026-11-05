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

## Compute

- For every CPU compute path, consider and prefer SIMD where supported and worthwhile; reuse optimized library/runtime kernels and justify material scalar exceptions with measurements.
- When changing compute/inference dispatch, follow the [mixed accelerator policy](../docs/adr/0001-tech-selection.md#mixed-accelerator-policy-2026-09-29): choose measured speed across supported single/mixed plans; do not permanently retreat to CPU solely for numerical drift.

## Gates

`cargo fmt --all --check`, `RUSTFLAGS="-D warnings" cargo clippy --all-targets
--all-features`, `cargo test --workspace`, `cargo deny check licenses`; the
ignored tests that need PostgreSQL or model weights, for the crates a change
touches, run locally with `-- --ignored` and are reported in the PR.
