//! `pamin topics` — what is already here, so a writer does not invent a name
//! for something that already has one.
//!
//! Every other read command needs a topic name to start from, and an agent
//! resuming work does not have one. Without this it writes to whatever name
//! occurs to it, and `deployment_pipeline` and `deploy_pipeline` become two
//! memories that never meet again. That failure is silent and permanent, which
//! is the worst combination a memory store can offer.
//!
//! With a query this answers by two routes and says which found what, because
//! they fail differently. The name index is exact on whole words and reaches a
//! topic nobody has written much about; the content channels are forgiving and
//! reach a topic whose name says nothing about what it holds. A caller shown
//! only one route cannot tell an absence from a miss.

use anyhow::Result;
use pamin_client::command::topics::{Args, Found, Topics};
use pamin_index::VectorIndex;

use crate::session::Session;
use pamin_core::Fusion;
use pamin_engine::Depths;
use pamin_store::repository;

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Topics> {
    let project_id = session.project(project).await?;
    let total = repository::topic_count(session.database().pool(), project_id).await?;

    let Some(query) = args.query else {
        let recent =
            repository::recent_topics(session.database().pool(), project_id, args.limit).await?;
        return Ok(Topics {
            query: None,
            total,
            topics: recent
                .into_iter()
                .map(|topic| Found {
                    topic: topic.name,
                    how: "recent".to_string(),
                })
                .collect(),
        });
    };

    let engine = session.engine(project, vector_index).await?;
    let named = engine.topics_named_like(&query, args.limit).await?;

    // The content route is the ordinary search without the reranker: this
    // returns names, so reordering the ones that got through would change
    // which twenty came back and not which one is right.
    let hits = engine
        .search_fused(&query, args.limit, Depths::default(), Fusion::default())
        .await?;

    // Name matches first: a caller looking for what to call something wants
    // the topic that is called that, before the topic that mentions it.
    let mut found: Vec<Found> = Vec::new();
    for name in &named {
        let also = hits.iter().any(|hit| &hit.topic == name);
        found.push(Found {
            topic: name.clone(),
            how: if also { "both" } else { "name" }.to_string(),
        });
    }
    for hit in hits {
        if !named.contains(&hit.topic) {
            found.push(Found {
                topic: hit.topic,
                how: "content".to_string(),
            });
        }
    }
    found.truncate(args.limit as usize);

    Ok(Topics {
        query: Some(query),
        total,
        topics: found,
    })
}
