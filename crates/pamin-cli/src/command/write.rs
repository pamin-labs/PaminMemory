//! `pamin write` — record a memory.

use anyhow::{Context, Result};
use pamin_core::SensoryFilter;
use pamin_index::Profile;
use pamin_store::repository;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::command::validity;
use crate::session::Session;
use pamin_engine::{Engine, Write};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic this memory belongs to.
    #[arg(long)]
    pub topic: String,

    /// The memory content. Reads standard input when omitted.
    pub content: Option<String>,

    /// Record the memory without waiting for the index to catch up.
    ///
    /// The work is queued rather than skipped, and `pamin cascade drain` runs
    /// it. Importing in bulk is what this is for: one rebuild of the vector
    /// graph at the end instead of the queue being drained after every write.
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

    let content = match args.content {
        Some(content) => content,
        None => std::io::read_to_string(std::io::stdin()).context("reading content from stdin")?,
    };

    let engine = session.engine(project, profile).await?;

    // Looked up rather than created: a write the filter holds should leave no
    // trace on the retrieval surface, and an empty topic is a trace. Promotion
    // is what creates one, inside the write transaction.
    let current = current_content(&engine, &args.topic).await?;
    let verdict = SensoryFilter::default().judge(&content, current.as_deref());

    let (language, confidence) = match pamin_index::detect_language(&content) {
        Some((language, confidence)) => (Some(language), Some(confidence)),
        None => (None, None),
    };

    let recorded = engine
        .write(&Write {
            topic: &args.topic,
            content: &content,
            content_hash: &hash(&content),
            verdict: verdict.decision,
            reason: verdict.reason(),
            promoted: verdict.is_promoted(),
            language: language.as_deref(),
            language_confidence: confidence,
            observed_at: OffsetDateTime::now_utc(),
            validity,
        })
        .await?;

    // The projection catches up from the outbox rather than here. Draining now
    // keeps `write` then `search` working the way it reads, without the write
    // transaction having depended on the index at all: if the index is
    // unreachable the memory is still recorded and the work is still owed.
    //
    // `--defer` is that separation made visible. The memory is committed either
    // way; what changes is whether this process is the one that pays for the
    // index -- and, past the ceiling, it is, because deferring is the only way
    // the queue grows without bound.
    //
    // The two numbers are two different facts, and reporting one of them for
    // both was the gap. What the queue owed when this write looked at it is
    // about the writer's rate and stays true whatever is done about it; what it
    // owes on the way out is about whether this memory is searchable yet.
    let (behind, owed) = if args.defer {
        let behind = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
        let owed = if pamin_core::may_defer(behind) {
            behind
        } else {
            engine.drain_cascade().await?.pending
        };
        (behind, owed)
    } else {
        let owed = engine.drain_cascade().await?.pending;
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

/// The content the topic currently resolves to, if it exists at all.
async fn current_content(engine: &Engine, topic: &str) -> Result<Option<String>> {
    let Some(topic) = repository::find_topic(engine.database.pool(), engine.project, topic).await?
    else {
        return Ok(None);
    };

    let versions = repository::topic_versions(engine.database.pool(), topic.id).await?;
    let Some(resolved) = pamin_core::resolve(&versions, pamin_core::VersionOffset::LATEST) else {
        return Ok(None);
    };

    Ok(
        repository::topic_state(engine.database.pool(), topic.id, resolved.version)
            .await?
            .map(|state| state.content),
    )
}

fn hash(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}
