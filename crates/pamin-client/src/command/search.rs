//! `pamin search` — retrieve memories, with the reasoning attached.

use pamin_core::{Channel, Depths, Derivation, EdgeKind, Why};
use serde::{Deserialize, Serialize};

pub use super::read::Span;

#[derive(clap::Args, Serialize, Deserialize)]
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
    #[arg(long, default_value_t = Depths::DEFAULT.channel)]
    pub channel_depth: std::num::NonZeroU32,

    /// How many edges the graph channel walks out from its seeds.
    #[arg(
        long,
        default_value_t = Depths::DEFAULT.graph,
        value_parser = clap::value_parser!(u8).range(0..=pamin_core::graph::MAX_DEPTH as i64)
    )]
    pub graph_depth: u8,

    /// How many fused candidates the reranker reads. For measuring the
    /// reranker's depth against its cost; see `reranking::DEPTH`.
    #[arg(
        long,
        hide = true,
        default_value_t = Depths::DEFAULT.rerank
    )]
    #[serde(default = "default_rerank_depth")]
    pub rerank_depth: std::num::NonZeroUsize,

    /// How wide the vector graph is searched. For measuring recall against
    /// time; see `projection::SEARCH_EFFORT`.
    #[arg(
        long,
        hide = true,
        default_value_t = Depths::DEFAULT.effort
    )]
    #[serde(default = "default_search_effort")]
    pub search_effort: std::num::NonZeroU32,

    /// Always `accurate`, and read by nothing in this build.
    ///
    /// It named the reranking tier until there was one. A server from before
    /// then refuses a search without it as unreadable rather than answering
    /// with its version, so a client that stopped sending it could neither
    /// use such a server nor replace it -- the reason `Request::profile` is
    /// still sent.
    #[arg(skip = String::from("accurate"))]
    #[serde(default)]
    pub rerank: String,
}

fn default_rerank_depth() -> std::num::NonZeroUsize {
    Depths::DEFAULT.rerank
}

fn default_search_effort() -> std::num::NonZeroU32 {
    Depths::DEFAULT.effort
}

/// One entry of the trace, as a caller sees it.
///
/// [`Why`] also carries `score`, `weight` and `contribution`, and `docs/cli.md`
/// prints the last two as things the reader works out: weight is a constant per
/// channel, and contribution is `weight / (10 + rank)`. Ten hits of them cost
/// about seven hundred tokens of somebody's context window to restate what they
/// already know, so the command layer leaves them out.
///
/// `score` is left out for a different reason. It is the channel's own quantity
/// in the channel's own units, so a reader comparing a BM25 score against a
/// vector similarity would be comparing nothing. Fusion reads it to judge how
/// confident a channel is *against that channel's other candidates*, and that
/// judgement already reaches the caller as the rank the fused list gives.
///
/// A separate type rather than `#[serde(skip)]` on the core one. Skipping
/// would make the field deserialize as zero on the far side of the socket,
/// and a number that is quietly wrong is a worse price than the tokens.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trace {
    Channel {
        channel: Channel,
        rank: u32,
    },
    Path {
        from: String,
        via: String,
        hops: u8,
        asserted_from: String,
        asserted_to: String,
        edge: EdgeKind,
        derivation: Derivation,
    },
    /// The cross-encoder scored this result, and its position blends that
    /// score with fusion's.
    ///
    /// Carried without its score, for the reason the channel score is left out
    /// above and one more. A cross-encoder's logit is calibrated against
    /// nothing, so it is comparable within one shortlist and meaningless
    /// between two -- and a number on the wire invites exactly the comparison
    /// it cannot support. What a reader can act on is the *fact*: a result
    /// carrying this entry was scored by the model, whatever channel found
    /// it, and one without it holds the place fusion gave it: either fusion
    /// returned fewer than two candidates, so nothing was reranked, or it sat
    /// below the reranker's head and was not among the strongest finds only
    /// the graph made. That distinction costs about four tokens and is the
    /// difference between auditing a ranking and guessing at it.
    Reranked {},
}

impl From<Why> for Trace {
    fn from(why: Why) -> Self {
        match why {
            Why::Channel { channel, rank, .. } => Self::Channel { channel, rank },
            Why::Path {
                from,
                via,
                hops,
                asserted_from,
                asserted_to,
                edge,
                derivation,
            } => Self::Path {
                from,
                via,
                hops,
                asserted_from,
                asserted_to,
                edge,
                derivation,
            },
            Why::Reranked { .. } => Self::Reranked {},
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Hit {
    /// What to pass to `pamin read` to see this topic's other versions.
    ///
    /// The name, and not the state's identifier. Search ranks topics and
    /// returns the current state of each, and no command anywhere takes a
    /// state id as an argument -- so one on every hit was two hundred and
    /// eighty tokens of a context window that nothing could spend.
    pub topic: String,
    pub version: u32,
    pub content: String,
    pub score: f32,
    /// The rank this result held in each channel it appeared in, and every
    /// modifier applied afterwards. An agent can audit its own retrieval from
    /// this without trusting the ranking.
    pub why: Vec<Trace>,
    /// When this state was recorded, RFC 3339.
    ///
    /// Ranking says how well a memory matches, not how current it is, and an
    /// agent assembling context needs both. Carried here so judging staleness
    /// does not cost a `read` per hit.
    pub recorded_at: String,
    /// When the claim starts holding, RFC 3339, absent when open.
    ///
    /// The other half of what `recorded_at` is here for, and the half that
    /// answers the question actually asked of a memory: not when we were told
    /// a thing, but when it was true. The two disagree whenever a source is
    /// backdated, which is every import that carries `--valid-from`, and an
    /// agent handed only the recording time reads the import order instead of
    /// the timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    /// When it stops holding, RFC 3339, absent when open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    /// For a memory the index cut into spans, which of them `content` is: the
    /// span the search matched best and one either side. Absent for a memory
    /// of one span, whose `content` is all of it. `pamin read --span` reads
    /// more of the memory around it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

#[derive(Serialize, Deserialize)]
pub struct Results {
    pub query: String,
    pub hits: Vec<Hit>,
}

/// Renders the result for a person reading it.
pub fn render(results: &Results) -> String {
    if results.hits.is_empty() {
        return format!("No memories matched {:?}", results.query);
    }
    results
        .hits
        .iter()
        .map(|hit| {
            // The span's own flag, so the line says what to read more of.
            let span = hit.span.as_ref().map_or(String::new(), |span| {
                format!(" --span {} (of {}) {}:", span.ordinal, span.of, span.header)
            });
            format!(
                "{:.4}  {} v{}{span}  {}\n        {}",
                hit.score,
                hit.topic,
                hit.version,
                hit.content,
                describe(&hit.why)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Renders the trace as one line, so the reason a result is here is visible
/// without asking for JSON.
fn describe(why: &[Trace]) -> String {
    why.iter()
        .map(|entry| match entry {
            Trace::Channel { channel, rank } => format!("{}#{rank}", channel.as_str()),
            // The arrow is drawn the way the edge was asserted, so it reads
            // the same whichever end the walk reached it from. `from` is the
            // seed the walk began at, which is a different fact and is kept.
            Trace::Path {
                from,
                via,
                hops,
                asserted_from,
                asserted_to,
                edge,
                ..
            } => {
                let arrow = format!("{asserted_from} --{}-> {asserted_to}", edge.as_str());
                if from == via {
                    format!("{arrow} ({hops}hop)")
                } else {
                    format!("from {from} via {via}: {arrow} ({hops}hop)")
                }
            }
            // One word, because the fact is the whole content. A line reading
            // `vector#12 reranked` says the fused list had this twelfth and
            // the model scored it; its place also uses fusion's score.
            Trace::Reranked {} => "reranked".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        args: Args,
    }

    /// A search request is still readable by a server from before the tier
    /// was removed, which required the field and knew `accurate`, so that
    /// server answers with its version and the client replaces it.
    #[test]
    fn a_search_is_readable_by_a_server_that_required_a_tier() {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct Before {
            query: String,
            limit: u32,
            channel_depth: u32,
            graph_depth: u8,
            rerank: String,
        }
        let args = Command::parse_from(["pamin", "deployment rollback"]).args;
        let wire = serde_json::to_string(&args).unwrap();
        let before: Before = serde_json::from_str(&wire).expect("an older server reads it");
        assert_eq!(before.rerank, "accurate");
    }
}
