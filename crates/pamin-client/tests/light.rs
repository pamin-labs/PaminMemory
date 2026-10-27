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

#[test]
fn the_client_reaches_nothing_only_the_server_needs() {
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

    let client = names
        .iter()
        .find(|(_, name)| **name == "pamin-client")
        .map(|(id, _)| *id)
        .expect("pamin-client is in the workspace");

    // Every package the client links, and the path that reached each.
    let mut reached: HashMap<&str, Vec<&str>> = HashMap::from([(client, vec!["pamin-client"])]);
    let mut pending = vec![client];
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
        "only {} packages reached, so this is not reading the graph",
        reached.len()
    );
    let heavy: BTreeSet<String> = reached
        .iter()
        .filter(|(id, _)| HEAVY.contains(&names[*id]))
        .map(|(_, path)| path.join(" -> "))
        .collect();
    assert!(
        heavy.is_empty(),
        "the client crate reaches what only the server should link:\n{}",
        heavy.into_iter().collect::<Vec<_>>().join("\n")
    );
}
