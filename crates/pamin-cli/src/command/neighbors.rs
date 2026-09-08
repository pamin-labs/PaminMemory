//! `pamin neighbors` — walk the graph directly, without ranking anything.
//!
//! Search fuses the graph with three other channels and returns what it thinks
//! is relevant. This returns what is actually connected, which is a different
//! question and the one to ask when the ranking is what you doubt.

use anyhow::{Result, bail};
use pamin_core::EdgeKind;
use pamin_store::graph::Expansion;
use pamin_store::{Database, Workspace, graph, repository};
use serde::Serialize;

use crate::command::validity;
use crate::output::Format;

#[derive(clap::Args)]
pub struct Args {
    /// The topic to walk out from.
    pub topic: String,

    /// How many edges to traverse.
    #[arg(
        long,
        default_value_t = 2,
        value_parser = clap::value_parser!(u8).range(0..=graph::MAX_DEPTH as i64)
    )]
    pub depth: u8,

    /// Restrict traversal to one relationship kind. Repeatable.
    #[arg(long = "kind")]
    pub kinds: Vec<String>,

    /// Only follow edges asserted to hold at this RFC 3339 instant.
    #[arg(long)]
    pub at: Option<String>,
}

#[derive(Serialize)]
struct Neighbor {
    topic: String,
    hops: u8,
    /// The topic on the other end of the final edge.
    via: String,
    edge: String,
    /// Whether the edge was asserted by a caller or derived by the engine.
    derivation: String,
    confidence: f32,
}

#[derive(Serialize)]
struct Neighborhood {
    topic: String,
    depth: u8,
    neighbors: Vec<Neighbor>,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format, args: Args) -> Result<()> {
    let kinds = args
        .kinds
        .iter()
        .map(|name| {
            EdgeKind::parse(name)
                .ok_or_else(|| anyhow::anyhow!("unknown relationship kind {name:?}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.client(), project).await?;

    let Some(topic) = repository::find_topic(database.client(), project.id, &args.topic).await?
    else {
        bail!("no topic named {}", args.topic);
    };

    let at = validity::parse(args.at.as_deref(), "--at")?;
    let neighbors = graph::expand(
        database.client(),
        project.id,
        &[topic.id],
        &Expansion {
            depth: args.depth,
            kinds: (!kinds.is_empty()).then_some(kinds.as_slice()),
            at,
        },
    )
    .await?;

    // Names are resolved in one pass rather than per neighbour, since the walk
    // can return every topic in a well-connected project.
    let names: std::collections::HashMap<_, _> =
        repository::all_topics(database.client(), project.id)
            .await?
            .into_iter()
            .map(|topic| (topic.id, topic.name))
            .collect();
    let name_of = |id: &pamin_core::TopicId| {
        names
            .get(id)
            .cloned()
            .unwrap_or_else(|| "<unknown topic>".to_string())
    };

    let result = Neighborhood {
        topic: args.topic,
        depth: args.depth,
        neighbors: neighbors
            .iter()
            .map(|neighbor| Neighbor {
                topic: name_of(&neighbor.topic),
                hops: neighbor.hops,
                via: name_of(&neighbor.via),
                edge: neighbor.kind.as_str().to_string(),
                derivation: format!("{:?}", neighbor.derivation).to_lowercase(),
                confidence: neighbor.confidence,
            })
            .collect(),
    };

    format.emit(&result, || {
        if result.neighbors.is_empty() {
            return format!(
                "{} is connected to nothing within {} hops",
                result.topic, result.depth
            );
        }
        result
            .neighbors
            .iter()
            .map(|neighbor| {
                format!(
                    "{}  {} hop  via {} --{}--> ({}, {:.2})",
                    neighbor.topic,
                    neighbor.hops,
                    neighbor.via,
                    neighbor.edge,
                    neighbor.derivation,
                    neighbor.confidence
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(())
}
