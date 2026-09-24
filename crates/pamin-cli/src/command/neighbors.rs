//! `pamin neighbors` — walk the graph directly, without ranking anything.
//!
//! Search fuses the graph with three other channels and returns what it thinks
//! is relevant. This returns what is actually connected, which is a different
//! question and the one to ask when the ranking is what you doubt.

use anyhow::Result;
use pamin_store::graph::Expansion;
use pamin_store::{graph, repository};
use serde::{Deserialize, Serialize};

use crate::command::{resolve, validity};

use crate::session::Session;

#[derive(clap::Args, Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
struct Neighbor {
    topic: String,
    hops: u8,
    /// The topic on the other end of the final edge.
    via: String,
    edge: String,
    /// The end the final edge was asserted from, and the end it points at.
    ///
    /// The walk is undirected, so `via` says how this topic was reached and
    /// not what was claimed. For `depends_on`, `supersedes`, `contradicts`,
    /// `derived_from` and `part_of` the direction is the claim, and reading it
    /// off `via` gives the opposite answer depending on which end the walk
    /// started from. These two say it outright.
    asserted_from: String,
    asserted_to: String,
    /// Whether the edge was asserted by a caller or derived by the engine.
    derivation: String,
    confidence: f32,
}

#[derive(Serialize, Deserialize)]
pub struct Neighborhood {
    topic: String,
    depth: u8,
    neighbors: Vec<Neighbor>,
}

pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Neighborhood> {
    let kinds = resolve::edge_kinds(&args.kinds)?;

    let database = session.database();
    let project = session.project(project).await?;

    let topic = resolve::topic(database, project, &args.topic).await?;

    let at = validity::parse(args.at.as_deref(), "--at")?;
    let neighbors = graph::expand(
        &mut *database.pool().acquire().await?,
        project,
        &[topic.id],
        &Expansion {
            depth: args.depth,
            kinds: (!kinds.is_empty()).then_some(kinds.as_slice()),
            at,
            keep: None,
        },
    )
    .await?;

    // Names are resolved in one pass rather than per neighbour, since the walk
    // can return every topic in a well-connected project.
    let names: std::collections::HashMap<_, _> = repository::all_topics(database.pool(), project)
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
                asserted_from: name_of(if neighbor.outbound {
                    &neighbor.via
                } else {
                    &neighbor.topic
                }),
                asserted_to: name_of(if neighbor.outbound {
                    &neighbor.topic
                } else {
                    &neighbor.via
                }),
                derivation: format!("{:?}", neighbor.derivation).to_lowercase(),
                confidence: neighbor.confidence,
            })
            .collect(),
    };

    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(result: &Neighborhood) -> String {
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
            // The arrow is drawn in the direction the edge was asserted, not
            // the direction the walk took, so the same edge reads the same way
            // from either end. `via` is not printed because at the final edge
            // it is always whichever of the two endpoints is not this topic.
            format!(
                "{}  {} hop{}  {} --{}--> {} ({}, {:.2})",
                neighbor.topic,
                neighbor.hops,
                if neighbor.hops == 1 { "" } else { "s" },
                neighbor.asserted_from,
                neighbor.edge,
                neighbor.asserted_to,
                neighbor.derivation,
                neighbor.confidence
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
