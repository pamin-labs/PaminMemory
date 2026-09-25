//! `pamin write` — record a memory.

use anyhow::{Context, Result};
use pamin_index::Profile;
use serde::{Deserialize, Serialize};

use crate::command::validity;
use crate::session::Session;
use pamin_engine::Owed;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic this memory belongs to.
    #[arg(long)]
    pub topic: String,

    /// The memory content. Reads standard input when omitted.
    pub content: Option<String>,

    /// Record the memory without waiting for the index to catch up.
    ///
    /// The work is queued rather than skipped. The server runs it once no
    /// request is being answered, and `pamin cascade drain` runs it at once.
    #[arg(long)]
    pub defer: bool,

    #[command(flatten)]
    pub validity: validity::Flags,
}

#[derive(Serialize, Deserialize)]
pub struct Written {
    topic: String,
    /// Absent when the filter held the content in the evidence layer.
    version: Option<u32>,
    promoted: bool,
    /// Why the filter decided as it did, promoted or not.
    reason: String,
    /// Always set: evidence is recorded whatever the filter decides.
    source_version: u32,
    /// Whether the projection caught up before this command returned, or the
    /// work is still owed. Either way the memory is recorded.
    cascade: String,
    /// Set when the projection has fallen far enough behind to say so. The
    /// queue is unbounded on purpose -- a write must not fail because the
    /// index is slow -- but a backlog nobody reports looks like searches
    /// quietly missing the newest memories.
    cascade_lagging: bool,
    /// The truth interval this state was asserted for, if one was given.
    valid_from: Option<String>,
    valid_to: Option<String>,
}

pub async fn execute(
    session: &Session,
    project: &str,
    profile: Profile,
    args: Args,
) -> Result<Written> {
    // Parsed before anything is provisioned, so a malformed interval fails
    // without having started a database.
    let validity = args.validity.parse()?;

    // Standard input is read by the front end, before dispatch, because the
    // server has none -- `main::fill_from_stdin`. So this is not a fallback:
    // the content is already here, and reading standard input would block the
    // server on a descriptor nobody is going to write to.
    let content = args
        .content
        .context("no content: pass it as an argument or on standard input")?;

    let engine = session.engine(project, profile).await?;
    let (verdict, recorded) = engine.remember(&args.topic, &content, validity).await?;

    // The projection catches up from the outbox rather than here. Draining now
    // keeps `write` then `search` working the way it reads, without the write
    // transaction having depended on the index at all: if the index is
    // unreachable the memory is still recorded and the work is still owed.
    //
    // `--defer` is that separation made visible. The memory is committed either
    // way; what changes is whether this write is the one that pays for the
    // index -- and, past the ceiling, it is, because deferring is the only way
    // the queue grows without bound.
    //
    // The two numbers are two different facts, and reporting one of them for
    // both was the gap. What the queue owed when this write looked at it is
    // about the writer's rate and stays true whatever is done about it; what it
    // owes on the way out is about whether this memory is searchable yet.
    //
    // Either drain stops at what the memory needs. The index's upkeep is left
    // to the server, which is still here after the write returns and runs it
    // between requests.
    let (behind, owed) = if args.defer {
        // Counted only as far as the bound it is compared with.
        let behind = pamin_store::jobs::pending_up_to(
            engine.database.pool(),
            engine.project,
            pamin_core::LAGGING_AT,
        )
        .await?;
        let owed = if pamin_core::may_defer(behind) {
            behind
        } else {
            still_owed(engine.drain_cascade(Owed::WhatAMemoryNeeds).await?)
        };
        (behind, owed)
    } else {
        let owed = still_owed(engine.drain_cascade(Owed::WhatAMemoryNeeds).await?);
        (owed, owed)
    };

    let result = Written {
        topic: args.topic,
        version: recorded.state.as_ref().map(|state| state.version),
        promoted: verdict.is_promoted(),
        reason: verdict.reason().to_string(),
        source_version: recorded.source_version,
        cascade: if owed == 0 { "applied" } else { "queued" }.to_string(),
        cascade_lagging: !pamin_core::may_defer(behind),
        valid_from: validity.from.map(validity::render),
        valid_to: validity.to.map(validity::render),
    };

    Ok(result)
}

/// What the projection still owes that would change what a search finds.
///
/// Not the same as what the queue still holds. A write whose document the index
/// has but whose flush has not happened yet is findable now -- the projection
/// buffers it in memory and a query reads that buffer -- and its job stays
/// claimed only so that a power cut replays it rather than losing it. Reporting
/// those as owed would say a memory is not searchable when it is, on every
/// single write, which is the opposite of what this field is for.
fn still_owed(drained: pamin_engine::Drained) -> i64 {
    // Never below nothing: the two numbers are read a moment apart while the
    // server's flusher is retiring rows between them, so the applied count can
    // outrun the queue's. Both readings mean the same thing here -- there is
    // nothing left that a search would miss.
    (drained.pending - drained.applied as i64).max(0)
}

/// Renders the result for a person reading it.
pub fn render(result: &Written) -> String {
    match result.version {
        Some(version) => format!("Wrote {} v{}", result.topic, version),
        None => format!(
            "Held in evidence only: {}\nStored as {} source version {}",
            result.reason, result.topic, result.source_version
        ),
    }
}
