//! `pamin import` — record many memories in one call.
//!
//! `pamin write` costs a process, a socket round trip, and a drain per memory,
//! and for one memory that is the right shape: an agent writes when something
//! happened, and what it wants back is whether that memory is findable. An
//! import wants none of that. Measured here, two and a half thousand deferred
//! writes spent 67 seconds, of which the ledger's share was a few milliseconds
//! each and the rest was paid one invocation at a time.
//!
//! So this is the bulk half, in the shape every comparable store has one:
//! Qdrant asks for batches of 64 to 256 points, Weaviate batches server-side,
//! Milvus reads files. It takes a file of memories, records them all against
//! one open engine, and lets the projection catch up in rounds rather than per
//! memory.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use pamin_index::Profile;
use serde::{Deserialize, Serialize};

use crate::command::validity;
use crate::command::write::pays_for_upkeep;
use crate::session::Session;

/// How many memories to record between checks on the queue.
///
/// The queue is what stops an import running away: past the depth that reports
/// the projection behind, the importer pays it down rather than going on
/// queueing. Asking how deep it is costs a query, so it is asked once per
/// thousand memories rather than per memory -- which at three jobs a memory
/// puts the queue no more than three thousand past the ceiling before anyone
/// looks, against a ceiling of ten.
const BETWEEN_CHECKS: usize = 1_000;

/// How many topics to record at once.
///
/// Every memory is a read of the topic's current content followed by one
/// transaction, and both are round trips this process spends waiting on the
/// server. Recorded one after another they do not overlap, so an import of
/// *n* memories costs *n* times the latency of one whatever the server's
/// capacity.
///
/// Bounded by the pool rather than by a number of its own: `Connections`
/// gives a command four connections, and concurrency past what the pool can
/// serve is not concurrency -- the extra tasks queue on the pool instead of on
/// the server, and the bound stops meaning anything. Derived from the pool at
/// run time for that reason, with this as the floor for a pool that reports
/// something unusable.
const LEAST_AT_ONCE: usize = 2;

#[derive(Clone, clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// A file of memories, one JSON object per line: `{"topic": …, "content": …}`.
    ///
    /// Read by whichever process holds the workspace, which is the server when
    /// one is running. Both are on this machine and run as you.
    #[arg(long)]
    pub from: PathBuf,

    #[command(flatten)]
    pub validity: validity::Flags,
}

/// One line of the file.
#[derive(Deserialize)]
struct Memory {
    topic: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
pub struct Imported {
    /// Memories read from the file.
    memories: usize,
    /// Of those, the ones the filter put on the retrieval surface.
    promoted: usize,
    /// The rest: recorded as evidence, not indexed. Re-importing the same file
    /// is how this fills up, and it is the right answer rather than an error.
    held: usize,
    /// Whether the projection caught up before this returned.
    cascade: String,
    /// Set when the projection was behind at any point during the import.
    cascade_lagging: bool,
}

/// The file's memories, grouped by topic, each group in the file's own order.
///
/// Extracted from `execute` so the invariant has a test that does not need a
/// database: a group's order *is* the correctness condition, and the end-to-end
/// check of it costs a PostgreSQL cluster and a model download.
fn by_topic(memories: &[Memory]) -> Vec<Vec<&Memory>> {
    let mut grouped: BTreeMap<&str, Vec<&Memory>> = BTreeMap::new();
    for memory in memories {
        grouped.entry(&memory.topic).or_default().push(memory);
    }
    grouped.into_values().collect()
}

pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
    args: Args,
) -> Result<Imported> {
    // Parsed before anything is provisioned, so a malformed interval fails
    // without having started a database.
    let validity = args.validity.parse()?;

    let file = std::fs::read_to_string(&args.from)
        .with_context(|| format!("reading {}", args.from.display()))?;

    // Read and parsed before the first write, so a typo on the last line is a
    // refusal rather than half an import. The file is memories, which are small
    // and already sitting on this machine's disk.
    let memories: Vec<Memory> = file
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("line {} of {}", index + 1, args.from.display()))
        })
        .collect::<Result<_>>()?;

    let engine = session.engine(project, profile).await?;
    let upkeep = pays_for_upkeep(&engine);

    // Grouped by topic, and **the grouping is what makes concurrency correct
    // rather than faster**. `Engine::remember` reads the topic's current
    // content and judges the new memory against it, and one of the verdicts it
    // can return is `Restatement`. Two writes to one topic running at once
    // both read the content from before either of them, so the second is
    // judged against the wrong text and a restatement is promoted as though it
    // said something new. That is not a torn row a transaction would catch --
    // both transactions are serialisable and both commit -- it is two correct
    // writes of a decision that was made on stale input.
    //
    // Within a group the order is the file's, because the same read makes each
    // memory's verdict depend on the one before it. Across groups there is
    // nothing shared: the read is keyed by topic, the source row is
    // `manual:{topic}`, and the name and state rows are the topic's own. So
    // topics run at once and memories do not.
    let groups = by_topic(&memories);

    let at_once = usize::try_from(engine.database.pool().options().get_max_connections())
        .unwrap_or(LEAST_AT_ONCE)
        .max(LEAST_AT_ONCE);

    let mut promoted = 0;
    let mut lagging = false;
    let mut since_check = 0;

    // Chunked, so the queue check below still happens *between* writes rather
    // than beside them. Draining the cascade while writes are in flight would
    // have the importer paying down a queue the same import is still filling,
    // and the check would report on a moment that never existed.
    for chunk in groups.chunks(at_once) {
        // Built eagerly and awaited together rather than streamed, because
        // the chunk is already the bound: `try_join_all` over a chunk of
        // `at_once` groups runs exactly `at_once` of them at a time, and the
        // first error cancels the rest.
        let counted = futures::future::try_join_all(chunk.iter().map(|group| async {
            let mut promoted = 0;
            for memory in group.iter() {
                let (verdict, _) = engine
                    .remember(&memory.topic, &memory.content, validity)
                    .await?;
                if verdict.is_promoted() {
                    promoted += 1;
                }
            }
            Ok::<usize, anyhow::Error>(promoted)
        }))
        .await?;

        promoted += counted.iter().sum::<usize>();
        since_check += chunk.iter().map(Vec::len).sum::<usize>();

        if since_check >= BETWEEN_CHECKS {
            since_check = 0;
            let behind = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
            if !pamin_core::may_defer(behind) {
                lagging = true;
                engine.drain_cascade(upkeep).await?;
            }
        }
    }

    // Once at the end rather than per memory, which is the whole point: the
    // cascade claims sixty-four jobs a round, so an import that drained per
    // memory would pay a round's fixed costs for every line of the file.
    engine.drain_cascade(upkeep).await?;

    // And the import is on disk when it returns. A write leaves its flush to
    // the server and reports the memory findable, which it is; an import is a
    // batch somebody is waiting on the end of, and it has just accumulated the
    // largest number of applied writes anything here produces. Flushing them
    // together is what the batching was for.
    engine.flush_what_is_applied().await?;
    let owed = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;

    Ok(Imported {
        memories: memories.len(),
        promoted,
        held: memories.len() - promoted,
        cascade: if owed == 0 { "applied" } else { "queued" }.to_string(),
        cascade_lagging: lagging,
    })
}

/// Renders the result for a person reading it.
pub fn render(result: &Imported) -> String {
    format!(
        "Imported {} memories: {} written, {} held in evidence only",
        result.memories, result.promoted, result.held
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(topic: &str, content: &str) -> Memory {
        Memory {
            topic: topic.to_string(),
            content: content.to_string(),
        }
    }

    /// A topic's memories stay in the file's order, and none of them moves
    /// between topics.
    ///
    /// The whole correctness condition for recording several topics at once.
    /// `Engine::remember` judges a memory against the topic's current content,
    /// so the second memory under a topic depends on the first having been
    /// recorded -- two of them at once are both judged against the text from
    /// before either, and a restatement is promoted as though it said
    /// something new. Nothing downstream can see that: both transactions
    /// commit, and both are correct writes of a decision made on stale input.
    ///
    /// The file interleaves its repeats, so an implementation that grouped by
    /// topic but sorted within a group, or that used a hash map's iteration
    /// order for the group, would fail this.
    #[test]
    fn grouping_by_topic_keeps_each_topics_order() {
        let memories = vec![
            memory("deploy", "first thing about deploying"),
            memory("oncall", "first thing about oncall"),
            memory("deploy", "second thing about deploying"),
            memory("rota", "the only thing about the rota"),
            memory("deploy", "third thing about deploying"),
            memory("oncall", "second thing about oncall"),
        ];

        let groups = by_topic(&memories);

        assert_eq!(groups.len(), 3, "one group per distinct topic");
        assert_eq!(
            groups.iter().map(Vec::len).sum::<usize>(),
            memories.len(),
            "every memory is in exactly one group"
        );
        for group in &groups {
            let topic = &group[0].topic;
            assert!(
                group.iter().all(|memory| &memory.topic == topic),
                "a group mixed topics"
            );
            let contents: Vec<&str> = group.iter().map(|memory| memory.content.as_str()).collect();
            let expected: Vec<&str> = memories
                .iter()
                .filter(|memory| &memory.topic == topic)
                .map(|memory| memory.content.as_str())
                .collect();
            assert_eq!(
                contents, expected,
                "{topic} was not in the order the file gave it"
            );
        }
    }

    /// An empty file groups into nothing rather than into one empty group.
    ///
    /// `by_topic`'s result is chunked and each chunk awaited together, and a
    /// group with no memories would be a task that does nothing -- harmless,
    /// and a sign the grouping had invented a topic.
    #[test]
    fn nothing_groups_into_nothing() {
        assert!(by_topic(&[]).is_empty());
    }
}
