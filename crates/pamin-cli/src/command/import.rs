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

use std::path::PathBuf;

use anyhow::{Context, Result};
use pamin_index::Profile;
use serde::{Deserialize, Serialize};

use crate::command::validity;
use crate::command::write::{pays_for_upkeep, record};
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

#[derive(clap::Args, Serialize, Deserialize)]
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

    let mut promoted = 0;
    let mut lagging = false;

    for (index, memory) in memories.iter().enumerate() {
        let (verdict, _) = record(&engine, &memory.topic, &memory.content, validity).await?;
        if verdict.is_promoted() {
            promoted += 1;
        }

        if index > 0 && index.is_multiple_of(BETWEEN_CHECKS) {
            let behind = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
            if !pamin_core::may_defer(behind) {
                lagging = true;
                engine.drain_cascade(upkeep).await?;
            }
        }
    }

    // Once at the end rather than per memory, which is the whole point: the
    // cascade claims sixty-four jobs a round and flushes the index once a
    // round, so an import that drained per memory would pay one flush each.
    let owed = engine.drain_cascade(upkeep).await?.pending;

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
