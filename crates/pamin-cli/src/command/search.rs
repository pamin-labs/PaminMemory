//! `pamin search` — retrieve memories, with the reasoning attached.

use anyhow::Result;
use pamin_core::{Channel, Derivation, EdgeKind, Why};
use pamin_index::{Profile, Rerank};

use serde::{Deserialize, Serialize};

use crate::command::validity;
use crate::session::Session;
use pamin_engine::Depths;

#[derive(Clone, clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// What to search for, in any language.
    pub query: String,

    /// How many results to return.
    #[arg(long, default_value_t = 5)]
    pub limit: u32,

    /// How much to spend reordering the results: off, fast, or accurate.
    ///
    /// A cross-encoder reads the query and a memory together, which is what
    /// lets it correct an order the channels got wrong and what makes it cost
    /// a forward pass per candidate. Only the candidates no lexical channel
    /// found are reordered -- not, as this used to say, the ones in another
    /// language; the rule is the absence of a lexical hit rather than a
    /// language test, because a language detector is absent on exactly the
    /// short queries an agent asks.
    ///
    /// The default is `accurate`, the tier that ranks best on every corpus
    /// measured and costs about a second and a half a search on four cores;
    /// `fast` and `off` buy that time back at a measured price. See
    /// `docs/cli.md`.
    #[arg(long, env = "PAMIN_RERANK", default_value = "accurate", value_parser = tier)]
    pub rerank: Rerank,
}

/// Reads a tier where the command line is parsed, which is the one place it
/// is checked: before a server is started or a database provisioned, so a
/// misspelled tier is an error the caller can act on at once rather than one
/// that arrives after a PostgreSQL install. The socket carries the parsed
/// tier, so the server never sees a name it would have to check again.
fn tier(name: &str) -> Result<Rerank, String> {
    Rerank::parse(name).ok_or_else(|| format!("unknown rerank tier {name:?}"))
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
enum Trace {
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
    /// The cross-encoder decided this position, and fusion did not.
    ///
    /// Carried without its score, for the reason the channel score is left out
    /// above and one more. A cross-encoder's logit is calibrated against
    /// nothing, so it is comparable within one shortlist and meaningless
    /// between two -- and a number on the wire invites exactly the comparison
    /// it cannot support. What a reader can act on is the *fact*, which
    /// nothing before this exposed: a result carrying this entry was
    /// reordered by the model, and one without it holds the place fusion gave
    /// it, either because a lexical channel found it or because it sat below
    /// the tier's depth. That distinction costs about four tokens and is the
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
struct Hit {
    /// What to pass to `pamin read` to see this topic's other versions.
    ///
    /// The name, and not the state's identifier. Search ranks topics and
    /// returns the current state of each, and no command anywhere takes a
    /// state id as an argument -- so one on every hit was two hundred and
    /// eighty tokens of a context window that nothing could spend.
    topic: String,
    version: u32,
    content: String,
    score: f32,
    /// The rank this result held in each channel it appeared in, and every
    /// modifier applied afterwards. An agent can audit its own retrieval from
    /// this without trusting the ranking.
    why: Vec<Trace>,
    /// When this state was recorded, RFC 3339.
    ///
    /// Ranking says how well a memory matches, not how current it is, and an
    /// agent assembling context needs both. Carried here so judging staleness
    /// does not cost a `read` per hit.
    recorded_at: String,
    /// When the claim starts holding, RFC 3339, absent when open.
    ///
    /// The other half of what `recorded_at` is here for, and the half that
    /// answers the question actually asked of a memory: not when we were told
    /// a thing, but when it was true. The two disagree whenever a source is
    /// backdated, which is every import that carries `--valid-from`, and an
    /// agent handed only the recording time reads the import order instead of
    /// the timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    valid_from: Option<String>,
    /// When it stops holding, RFC 3339, absent when open.
    #[serde(skip_serializing_if = "Option::is_none")]
    valid_to: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Results {
    query: String,
    hits: Vec<Hit>,
}

pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
    args: Args,
) -> Result<Results> {
    let engine = session.engine(project, profile).await?;
    let hits = engine
        .search_reranked(&args.query, args.limit, Depths::DEFAULT, args.rerank)
        .await?;

    let results = Results {
        query: args.query,
        hits: hits
            .into_iter()
            .map(|hit| Hit {
                topic: hit.topic,
                version: hit.state.version,
                content: hit.state.content,
                score: hit.result.score,
                why: hit.result.why.into_iter().map(Trace::from).collect(),
                recorded_at: validity::render(hit.state.recorded_at),
                valid_from: hit.state.validity.from.map(validity::render),
                valid_to: hit.state.validity.to.map(validity::render),
            })
            .collect(),
    };

    Ok(results)
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
            format!(
                "{:.4}  {} v{}  {}\n        {}",
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
            // the model moved it, which is what a reader auditing a ranking
            // wants and could not previously get from anywhere.
            Trace::Reranked {} => "reranked".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}
