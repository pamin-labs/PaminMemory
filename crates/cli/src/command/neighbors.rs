//! `pamin-memory neighbors` — walk the graph directly, without ranking anything.
//!
//! Search fuses the graph with three other channels and returns what it thinks
//! is relevant. This returns what is actually connected, which is a different
//! question and the one to ask when the ranking is what you doubt.

use anyhow::Result;
use memory_contract::neighbors::{Args, Neighbor, Neighborhood};
use memory_contract::validity;
use memory_store::graph::Expansion;
use memory_store::{graph, repository};

use crate::command::resolve;
use crate::session::Session;

pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Neighborhood> {
    let kinds = resolve::edge_kinds(&args.kinds)?;

    let database = session.database();
    let project = session.project(project).await?;

    let topic = resolve::topic(database, project, &args.topic).await?;

    let at = validity::parse(args.at.as_deref(), "--at")?;
    let neighbors = graph::expand(
        // Read only: dropped at the end of the walk, which rolls it back.
        &mut *database.scoped(project).await?,
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
    let names: std::collections::HashMap<_, _> = database
        .within(project, async |connection| {
            anyhow::Ok(repository::all_topics(connection, project).await?)
        })
        .await?
        .into_iter()
        .map(|topic| (topic.id, topic.name))
        .collect();
    let name_of = |id: &memory_core::TopicId| {
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
