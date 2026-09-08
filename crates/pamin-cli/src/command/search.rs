//! `pamin search` — retrieve memories, with the reasoning attached.

use anyhow::Result;
use pamin_core::Why;
use pamin_index::Profile;
use pamin_store::Workspace;
use serde::Serialize;

use crate::output::Format;
use pamin_engine::{Depths, Engine};

#[derive(clap::Args)]
pub struct Args {
    /// What to search for, in any language.
    pub query: String,

    /// How many results to return.
    #[arg(long, default_value_t = 5)]
    pub limit: u32,

    /// How many candidates each channel contributes before fusion.
    ///
    /// For the evaluation harness, which the architecture names as the thing
    /// that tunes this. An agent wanting control over retrieval should reach
    /// for the primitives — `grep`, `read`, `neighbors` — rather than adjust
    /// ranking internals it has no way to evaluate.
    #[arg(long, env = "PAMIN_CHANNEL_DEPTH", default_value_t = Depths::default().channel)]
    pub channel_depth: u32,

    /// How many edges the graph channel walks out from its seeds.
    #[arg(
        long,
        env = "PAMIN_GRAPH_DEPTH",
        default_value_t = Depths::default().graph,
        value_parser = clap::value_parser!(u8).range(0..=pamin_store::graph::MAX_DEPTH as i64)
    )]
    pub graph_depth: u8,
}

#[derive(Serialize)]
struct Hit {
    /// What to pass to `pamin read` to see this topic's other versions.
    topic: String,
    topic_state: String,
    version: u32,
    is_current: bool,
    content: String,
    score: f32,
    /// The rank this result held in each channel it appeared in, and every
    /// modifier applied afterwards. An agent can audit its own retrieval from
    /// this without trusting the ranking.
    why: Vec<Why>,
    /// The byte range in the source this state came from.
    source_span: String,
}

#[derive(Serialize)]
struct Results {
    query: String,
    hits: Vec<Hit>,
}

pub async fn run(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
    args: Args,
) -> Result<()> {
    let mut engine = Engine::open(workspace, project, profile).await?;
    let depths = Depths {
        channel: args.channel_depth,
        graph: args.graph_depth,
    };
    let hits = engine.search(&args.query, args.limit, depths).await?;

    let results = Results {
        query: args.query,
        hits: hits
            .into_iter()
            .map(|hit| Hit {
                topic: hit.topic,
                topic_state: hit.state.id.to_string(),
                version: hit.state.version,
                is_current: hit.is_current,
                content: hit.state.content,
                score: hit.result.score,
                why: hit.result.why,
                source_span: hit.state.source_span_id.to_string(),
            })
            .collect(),
    };

    format.emit(&results, || {
        if results.hits.is_empty() {
            return format!("No memories matched {:?}", results.query);
        }
        results
            .hits
            .iter()
            .map(|hit| {
                let marker = if hit.is_current {
                    "current"
                } else {
                    "historical"
                };
                format!(
                    "{:.4}  {} v{} ({marker})  {}\n        {}",
                    hit.score,
                    hit.topic,
                    hit.version,
                    hit.content,
                    describe(&hit.why)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(())
}

/// Renders the trace as one line, so the reason a result is here is visible
/// without asking for JSON.
fn describe(why: &[Why]) -> String {
    why.iter()
        .map(|entry| match entry {
            Why::Channel { channel, rank, .. } => format!("{}#{rank}", channel.as_str()),
            Why::Modifier { modifier, factor } => format!("{modifier:?}x{factor:.2}"),
            Why::Path {
                from,
                via,
                hops,
                edge,
                ..
            } => {
                if from == via {
                    format!("from {from} --{}-> ({hops}hop)", edge.as_str())
                } else {
                    format!("from {from} via {via} --{}-> ({hops}hop)", edge.as_str())
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
