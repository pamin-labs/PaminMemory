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
use pamin_index::{Profile, Rerank};
use serde::{Deserialize, Serialize};

use crate::session::Session;
use pamin_engine::Depths;
use pamin_store::repository;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// What the topic might be about, or be called. Omit to list recent ones.
    ///
    /// Answered twice over. The content route is forgiving and is what finds a
    /// topic from a half-remembered description; the name route is exact on
    /// the segmenter's tokens, so `deployment pipeline` finds
    /// `deployment_pipeline` and `deploy pipeline` does not, and it is there
    /// for the topic nobody has written enough about for content to reach.
    pub query: Option<String>,

    /// How many to return.
    #[arg(long, default_value_t = 20)]
    pub limit: u32,
}

#[derive(Serialize, Deserialize)]
struct Found {
    topic: String,
    /// `name` when the name index matched, `content` when a memory under it
    /// did, `both` when each did. Which one it was is the difference between
    /// "this topic is about that" and "this topic is called that".
    how: String,
}

#[derive(Serialize, Deserialize)]
pub struct Topics {
    /// Absent when nothing was asked for, so the list is simply what is here.
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<String>,
    /// How many topics the project holds, which is the context the page needs:
    /// twenty of twenty-two is the whole story, twenty of nine thousand is a
    /// sample and the caller should narrow instead.
    total: u64,
    topics: Vec<Found>,
}

pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
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

    let engine = session.engine(project, profile).await?;
    let named = engine.topics_named_like(&query, args.limit).await?;

    // The content route is the ordinary search, at the tier that costs nothing
    // to run: this returns names, so reordering the ones that got through
    // would change which twenty came back and not which one is right.
    let hits = engine
        .search_reranked(&query, args.limit, Depths::DEFAULT, Rerank::Off)
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

/// Renders the result for a person reading it.
pub fn render(result: &Topics) -> String {
    if result.topics.is_empty() {
        return match &result.query {
            Some(query) => format!("No topic matched {query:?}, of {} here", result.total),
            None => "No topics yet".to_string(),
        };
    }
    let mut out: Vec<String> = result
        .topics
        .iter()
        .map(|found| format!("{:<10}{}", found.how, found.topic))
        .collect();
    out.push(format!(
        "\nShowing {} of {} topics",
        result.topics.len(),
        result.total
    ));
    out.join("\n")
}
