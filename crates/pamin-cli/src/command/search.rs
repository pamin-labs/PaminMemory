//! `pamin search` — retrieve memories, with the reasoning attached.

use anyhow::Result;
use pamin_core::{Channel, Derivation, EdgeKind, Why};
use pamin_index::{Licence, Profile, Rerank};

use serde::{Deserialize, Serialize};

use crate::command::validity;
use crate::session::Session;
use pamin_engine::Depths;

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

    /// How much to spend reordering the results: off, fast, balanced,
    /// accurate, or noncommercial.
    ///
    /// `noncommercial` needs `PAMIN_ACCEPT_NONCOMMERCIAL` as well, and says so
    /// when it does not have it. Its weights are CC-BY-NC-4.0.
    ///
    /// A cross-encoder reads the query and a memory together, which is what
    /// lets it correct an order the channels got wrong and what makes it cost
    /// a forward pass per candidate. Only the candidates no lexical channel
    /// found are reordered -- not, as this used to say, the ones in another
    /// language; the rule is the absence of a lexical hit rather than a
    /// language test, because a language detector is absent on exactly the
    /// short queries an agent asks. In practice that is mostly the same set,
    /// so a workspace in one language gains little from this and, on a corpus
    /// with one language throughout, measurably loses: see `docs/cli.md`.
    #[arg(long, env = "PAMIN_RERANK", default_value = "fast")]
    pub rerank: String,
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
    let rerank = Rerank::parse(&args.rerank)
        .ok_or_else(|| anyhow::anyhow!("unknown rerank tier {:?}", args.rerank))?;

    let engine = session.engine(project, profile).await?;
    let depths = Depths {
        channel: args.channel_depth,
        graph: args.graph_depth,
    };
    let hits = engine
        .search_reranked(&args.query, args.limit, depths, rerank)
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

/// The environment variable that accepts a non-commercial tier's terms.
const ACCEPT: &str = "PAMIN_ACCEPT_NONCOMMERCIAL";

/// What to tell a caller about a tier's licence before its weights are
/// fetched, if anything.
///
/// A tier whose weights are not free for commercial use is offered rather than
/// withheld, and this is the notice that goes with offering it. It does not
/// refuse. Naming the tier is already a deliberate act -- nothing reaches it
/// by default and the default is permissive -- so the job here is to make sure
/// nobody arrives at those terms without being told, not to decide on their
/// behalf whether their use is within them. Only the caller knows that.
///
/// The wording is about an obligation rather than a hazard, and the difference
/// is not cosmetic. Nothing here is going to break: the model loads, scores,
/// and ranks like any other. What the licence does is restrict *what the
/// output may be used for*, which is a question about the caller's situation
/// and one this program has no way to answer. So the notice says what the
/// terms are and whose responsibility it is to stay inside them.
///
/// Returned rather than printed, so the caller decides where it goes. That
/// matters: it belongs where a person is reading and nowhere else, and a
/// library that wrote to stderr on its own would put it in a resident server's
/// log file, which is the one place it is certain to inform nobody.
///
/// `None` for every permissive tier and for `off`, which loads nothing.
pub(crate) fn caution(tier: Rerank) -> Option<String> {
    if tier.licence() != Some(Licence::NonCommercial) || std::env::var_os(ACCEPT).is_some() {
        return None;
    }

    Some(format!(
        "notice: the {tier} reranker tier downloads weights licensed CC-BY-NC-4.0. They \
         permit research and personal use and do not permit commercial use. Nothing about \
         the model is less reliable for it -- what the licence restricts is what you may \
         use the results for, and whether your use falls inside those terms is yours to \
         determine and yours to comply with.\n\
         Set {ACCEPT}=1 once you have, to record that and stop showing this. Every other \
         tier is permissively licensed: `off`, `fast`, `balanced`, `accurate`. See NOTICE \
         for what each one downloads.",
        tier = tier.name()
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A permissive tier says nothing, which is most of the point.
    ///
    /// Asserted alongside the notice because a `caution` that spoke about
    /// every tier would pass the test below and make the notice worthless by
    /// making it ordinary.
    #[test]
    fn a_permissively_licensed_tier_carries_no_notice() {
        for tier in [
            Rerank::Off,
            Rerank::Fast,
            Rerank::Balanced,
            Rerank::Accurate,
        ] {
            assert_eq!(
                caution(tier),
                None,
                "{} is permissively licensed and carries a notice",
                tier.name()
            );
        }
    }

    /// The non-commercial tier carries one, and it states the terms.
    ///
    /// This is the whole notice -- nothing refuses and nothing else mentions
    /// the licence at the moment of use -- so the test reads its contents
    /// rather than only checking that some string came back. It has to name
    /// the licence, say what it forbids, and say who is responsible for
    /// staying inside it; a notice missing any of those informs nobody.
    ///
    /// It must also not read as a warning about reliability. The model is not
    /// less trustworthy for its licence, and a notice that implied otherwise
    /// would be inaccurate in the direction that makes people ignore notices.
    ///
    /// The environment is read rather than injected, so this can only assert
    /// the notice when the variable is unset. It panics rather than passing
    /// quietly in that case: a developer who set it in their own shell would
    /// otherwise see this test assert nothing at all.
    #[test]
    fn the_non_commercial_tier_carries_a_notice_about_the_obligation() {
        if std::env::var_os(ACCEPT).is_some() {
            panic!(
                "{ACCEPT} is set in this environment, so this test cannot check the notice. \
                 Unset it and run again."
            );
        }

        let notice = caution(Rerank::Noncommercial).expect("no notice for a CC-BY-NC tier");

        for expected in [
            "noncommercial",
            "CC-BY-NC-4.0",
            "do not permit commercial use",
            "yours to comply with",
            ACCEPT,
            "NOTICE",
        ] {
            assert!(
                notice.contains(expected),
                "the notice does not mention {expected:?}: {notice}"
            );
        }
    }

    /// And it is a notice rather than a refusal: the tier still runs.
    ///
    /// Worth its own test because the first version of this refused, and the
    /// difference between the two is the entire product decision. A tier
    /// nobody can reach was not offered.
    #[test]
    fn carrying_a_notice_does_not_stop_the_tier_from_running() {
        assert_eq!(
            Rerank::parse(Rerank::Noncommercial.name()),
            Some(Rerank::Noncommercial)
        );
    }
}
