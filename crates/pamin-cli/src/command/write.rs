//! `pamin write` — record a memory.

use anyhow::{Context, Result};
use pamin_core::{SensoryFilter, SourceKind};
use pamin_index::Profile;
use pamin_store::{Workspace, repository};
use serde::Serialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::engine::Engine;
use crate::output::Format;

#[derive(clap::Args)]
pub struct Args {
    /// The topic this memory belongs to.
    #[arg(long)]
    pub topic: String,

    /// The memory content. Reads standard input when omitted.
    pub content: Option<String>,
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
}

pub async fn run(
    workspace: &Workspace,
    project: &str,
    profile: Profile,
    format: Format,
    args: Args,
) -> Result<()> {
    let content = match args.content {
        Some(content) => content,
        None => std::io::read_to_string(std::io::stdin()).context("reading content from stdin")?,
    };

    let mut engine = Engine::open(workspace, project, profile).await?;
    let project = engine.project;

    // Manual writes to one topic share a source, so their evidence forms a
    // single chain rather than a new source per write.
    let source = repository::ensure_source(
        engine.database.client(),
        project,
        SourceKind::Manual,
        &format!("manual:{}", args.topic),
    )
    .await?;

    let topic = engine.ensure_topic(&args.topic).await?;
    let current = current_content(&engine.database, topic.id).await?;

    let verdict = SensoryFilter::default().judge(&content, current.as_deref());

    // Evidence first, always, and before the filter's verdict is acted on. That
    // ordering is what makes a rejection recoverable instead of a loss.
    let source_version = repository::append_source_version(
        engine.database.client(),
        project,
        source,
        &content,
        &hash(&content),
        verdict.decision,
        verdict.reason(),
    )
    .await?;

    // Detected per span, not per deployment: one workspace holds many
    // languages, and this is what the note-language rule reads later.
    let (language, confidence) = match pamin_index::detect_language(&content) {
        Some((language, confidence)) => (Some(language), Some(confidence)),
        None => (None, None),
    };

    let span = repository::append_source_span(
        engine.database.client(),
        project,
        source_version.id,
        0,
        content.len() as u32,
        language.as_deref(),
        confidence,
    )
    .await?;

    let state = if verdict.is_promoted() {
        let state = repository::append_topic_state(
            engine.database.client_mut(),
            project,
            topic.id,
            &content,
            span.id,
            OffsetDateTime::now_utc(),
        )
        .await?;

        // Index only what was promoted. Filtered content stays in the evidence
        // layer, reachable and replayable, but off the retrieval surface, which
        // is the whole point of filtering after persistence rather than before.
        engine.index_state(&state)?;

        // Derived writes run here rather than in a worker because there is no
        // worker yet. Both this and the index update move into the cascade
        // pipeline together, which is where derived writes belong.
        engine.derive_mentions(&state).await?;
        Some(state)
    } else {
        None
    };

    let result = Written {
        topic: args.topic,
        version: state.as_ref().map(|state| state.version),
        promoted: verdict.is_promoted(),
        reason: verdict.reason().to_string(),
        source_version: source_version.version,
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

async fn current_content(
    database: &pamin_store::Database,
    topic: pamin_core::TopicId,
) -> Result<Option<String>> {
    let versions = repository::topic_versions(database.client(), topic).await?;
    let Some(resolved) = pamin_core::resolve(&versions, pamin_core::VersionOffset::LATEST) else {
        return Ok(None);
    };
    Ok(
        repository::topic_state(database.client(), topic, resolved.version)
            .await?
            .map(|state| state.content),
    )
}

fn hash(content: &str) -> String {
    format!("{:x}", Sha256::digest(content.as_bytes()))
}
