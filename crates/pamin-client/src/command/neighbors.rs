//! `pamin neighbors` — walk the graph directly, without ranking anything.

use pamin_core::graph::MAX_DEPTH;
use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic to walk out from.
    pub topic: String,

    /// How many edges to traverse.
    #[arg(
        long,
        default_value_t = 2,
        value_parser = clap::value_parser!(u8).range(0..=MAX_DEPTH as i64)
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
pub struct Neighbor {
    pub topic: String,
    pub hops: u8,
    /// The topic on the other end of the final edge.
    pub via: String,
    pub edge: String,
    /// The end the final edge was asserted from, and the end it points at.
    ///
    /// The walk is undirected, so `via` says how this topic was reached and
    /// not what was claimed. For `depends_on`, `supersedes`, `contradicts`,
    /// `derived_from` and `part_of` the direction is the claim, and reading it
    /// off `via` gives the opposite answer depending on which end the walk
    /// started from. These two say it outright.
    pub asserted_from: String,
    pub asserted_to: String,
    /// Whether the edge was asserted by a caller or derived by the engine.
    pub derivation: String,
    pub confidence: f32,
}

#[derive(Serialize, Deserialize)]
pub struct Neighborhood {
    pub topic: String,
    pub depth: u8,
    pub neighbors: Vec<Neighbor>,
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
