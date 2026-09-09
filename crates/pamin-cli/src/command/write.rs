//! `pamin write` — record a memory.

use anyhow::{Context, Result};
use pamin_core::SensoryFilter;
use pamin_index::{Access, Profile};
use pamin_store::{Workspace, repository};
use serde::Serialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::command::validity;
use crate::output::Format;
use pamin_engine::{Engine, Write};

#[derive(clap::Args)]
pub struct Args {
    /// The topic this memory belongs to.
    #[arg(long)]
    pub topic: String,

    /// The memory content. Reads standard input when omitted.
    pub content: Option<String>,

    #[command(flatten)]
    pub validity: validity::Flags,
}

#[derive(Serialize)]
struct Written {
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
    cascade: &'static str,
    /// The truth interval this state was asserted for, if one was given.
    valid_from: Option<String>,
    valid_to: Option<String>,
}

pub async fn run(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
    args: Args,
) -> Result<()> {
    // Parsed before anything is provisioned, so a malformed interval fails
    // without having started a database.
    let validity = args.validity.parse()?;

    let content = match args.content {
        Some(content) => content,
        None => std::io::read_to_string(std::io::stdin()).context("reading content from stdin")?,
    };

    let mut engine = Engine::open(workspace, project, profile, Access::ReadWrite).await?;

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
    let cascade = engine.drain_cascade().await?;

    let result = Written {
        topic: args.topic,
        version: recorded.state.as_ref().map(|state| state.version),
        promoted: verdict.is_promoted(),
        reason: verdict.reason().to_string(),
        source_version: recorded.source_version,
        cascade: if cascade.pending == 0 {
            "applied"
        } else {
            "queued"
        },
        valid_from: validity.from.map(validity::render),
        valid_to: validity.to.map(validity::render),
    };

    format.emit(&result, || match result.version {
        Some(version) => format!("Wrote {} v{}", result.topic, version),
        None => format!(
            "Held in evidence only: {}\nStored as {} source version {}",
            result.reason, result.topic, result.source_version
        ),
    });
    Ok(())
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
