//! `pamin topics` — what is already here, so a writer does not invent a name
//! for something that already has one.

use serde::{Deserialize, Serialize};

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
pub struct Found {
    pub topic: String,
    /// `name` when the name index matched, `content` when a memory under it
    /// did, `both` when each did. Which one it was is the difference between
    /// "this topic is about that" and "this topic is called that".
    pub how: String,
}

#[derive(Serialize, Deserialize)]
pub struct Topics {
    /// Absent when nothing was asked for, so the list is simply what is here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// How many topics the project holds, which is the context the page needs:
    /// twenty of twenty-two is the whole story, twenty of nine thousand is a
    /// sample and the caller should narrow instead.
    pub total: u64,
    pub topics: Vec<Found>,
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
