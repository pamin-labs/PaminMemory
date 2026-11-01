//! An argument a tool call leaves out means what it means on the command line.
//!
//! MCP clients send JSON with only what they chose to set, so every optional
//! argument needs a serde default as well as clap's. Two defaults can drift;
//! this parses each operation both ways, from only its required arguments,
//! and compares what the server would receive.

use clap::{Args, FromArgMatches};
use pamin_contract::{grep, link, neighbors, read, search, topics, unlink, write};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

fn same<A: Args + FromArgMatches + Serialize + DeserializeOwned>(cli: &[&str], tool: Value) {
    let command = A::augment_args(clap::Command::new("pamin").no_binary_name(true));
    let matches = command
        .try_get_matches_from(cli)
        .expect("the CLI parses it");
    let parsed = A::from_arg_matches(&matches).expect("clap builds the arguments");
    let called: A = serde_json::from_value(tool.clone()).expect("the tool call parses");
    assert_eq!(
        serde_json::to_value(&parsed).unwrap(),
        serde_json::to_value(&called).unwrap(),
        "{cli:?} and {tool}"
    );
}

#[test]
fn a_tool_call_and_the_cli_agree_on_every_default() {
    same::<search::Args>(&["q"], json!({"query": "q"}));
    same::<read::Args>(&["t"], json!({"topic": "t"}));
    same::<grep::Args>(&["s"], json!({"literal": "s"}));
    same::<neighbors::Args>(&["t"], json!({"topic": "t"}));
    same::<topics::Args>(&[], json!({}));
    same::<write::Args>(
        &["--topic", "t", "c"],
        json!({"topic": "t", "content": "c"}),
    );
    same::<link::Args>(&["a", "b"], json!({"from": "a", "to": "b"}));
    same::<unlink::Args>(&["a", "b"], json!({"from": "a", "to": "b"}));
}
