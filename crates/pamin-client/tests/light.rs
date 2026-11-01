//! The client crate stays light: nothing it depends on, on any platform,
//! reaches what `pamin-server` alone should pay for.
//!
//! `pamin` is built from this crate alone, and what a process links is what
//! it costs at every start. The inference runtime is the expensive one: any
//! binary that links the `ort` crate at all runs its static initialisers and
//! needs the C++ runtime, whatever its `main` reaches, so no amount of dead
//! code elimination takes it back out. One innocent dependency on a library
//! crate would put it, the vector engine, TLS and the database driver back
//! into every command an agent runs, with nothing failing to say so.
//!
//! The same graph keeps dependencies pointing one way: adapters (the CLI, the
//! socket, MCP) over the contract over the engine. The engine reaching an
//! adapter or the contract would let a transport's types into the domain, so
//! that fails too.
//!
//! Read from `cargo metadata`'s resolved graph, over every target platform
//! rather than the host's, following normal dependencies only: a build script
//! or a test may use what it likes. Not `--offline`: every platform's graph
//! needs manifests a host build never downloads, such as Windows-only crates.

use std::collections::{BTreeSet, HashMap};
use std::process::Command;

/// What only the server may reach, by package name.
const HEAVY: &[&str] = &[
    "ort",
    "ort-sys",
    "zvec-rust",
    "zvec-rust-sys",
    "openssl-sys",
    "native-tls",
    "sqlx",
    "sqlx-core",
    "tokenizers",
    "postgresql_embedded",
    "pamin-index",
    "pamin-store",
    "pamin-engine",
];

/// What the engine, and everything under it, may not reach.
const ADAPTERS: &[&str] = &["pamin-client", "pamin-contract", "pamin-kit", "rmcp"];

#[test]
fn the_client_reaches_nothing_only_the_server_needs() {
    let heavy = reached_from("pamin-client", HEAVY);
    assert!(
        heavy.is_empty(),
        "the client crate reaches what only the server should link:\n{}",
        heavy.into_iter().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn the_engine_reaches_no_adapter() {
    let adapters = reached_from("pamin-engine", ADAPTERS);
    assert!(
        adapters.is_empty(),
        "the engine reaches an adapter or the contract:\n{}",
        adapters.into_iter().collect::<Vec<_>>().join("\n")
    );
}

/// The path to each package in `forbidden` that `package` links.
fn reached_from(package: &str, forbidden: &[&str]) -> BTreeSet<String> {
    let output = Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1", "--locked"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("running cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata prints JSON");

    let names: HashMap<&str, &str> = metadata["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .map(|package| {
            (
                package["id"].as_str().expect("an id"),
                package["name"].as_str().expect("a name"),
            )
        })
        .collect();
    let nodes: HashMap<&str, &serde_json::Value> = metadata["resolve"]["nodes"]
        .as_array()
        .expect("a resolved graph")
        .iter()
        .map(|node| (node["id"].as_str().expect("an id"), node))
        .collect();

    let root = names
        .iter()
        .find(|(_, name)| **name == package)
        .map(|(id, _)| *id)
        .unwrap_or_else(|| panic!("{package} is in the workspace"));

    // Every package the root links, and the path that reached each.
    let mut reached: HashMap<&str, Vec<&str>> = HashMap::from([(root, vec![package])]);
    let mut pending = vec![root];
    while let Some(id) = pending.pop() {
        for dependency in nodes[id]["deps"].as_array().expect("deps") {
            let normal = dependency["dep_kinds"]
                .as_array()
                .expect("dep kinds")
                .iter()
                .any(|kind| kind["kind"].is_null());
            let target = dependency["pkg"].as_str().expect("a package id");
            if normal && !reached.contains_key(target) {
                let mut path = reached[id].clone();
                path.push(names[target]);
                reached.insert(target, path);
                pending.push(target);
            }
        }
    }

    assert!(
        reached.len() > 10,
        "only {} packages reached from {package}, so this is not reading the graph",
        reached.len()
    );
    reached
        .iter()
        .filter(|(id, _)| forbidden.contains(&names[*id]))
        .map(|(_, path)| path.join(" -> "))
        .collect()
}
