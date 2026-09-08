//! `pamin write` — record a memory.

use anyhow::{Context, Result};
use pamin_core::{SensoryFilter, SourceKind};
use pamin_index::{Access, Profile};
use pamin_store::{Workspace, repository};
use serde::Serialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use crate::command::validity;
use crate::output::Format;
use pamin_engine::Engine;

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

    // Looked up rather than created: a write the filter holds should leave no
    // trace on the retrieval surface, and an empty topic is a trace. Promotion
    // is what creates one, further down.
    let existing = repository::find_topic(engine.database.client(), project, &args.topic).await?;
    let current = match &existing {
        Some(topic) => current_content(&engine.database, topic.id).await?,
        None => None,
    };

    let verdict = SensoryFilter::default().judge(&content, current.as_deref());

    // Evidence first, always, and before the filter's verdict is acted on. That
    // ordering is what makes a rejection recoverable instead of a loss.
    let source_version = repository::append_source_version(
        engine.database.client_mut(),
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
        // Creating it here also backfills the edges memories written before it
        // already carry, which is why promotion goes through the engine rather
        // than straight to the repository.
        let topic = match existing {
            Some(topic) => topic,
            None => engine.ensure_topic(&args.topic).await?,
        };
        let state = repository::append_topic_state(
            engine.database.client_mut(),
            project,
            topic.id,
            &content,
            span.id,
            OffsetDateTime::now_utc(),
            validity,
        )
        .await?;

        // Index only what was promoted. Filtered content stays in the evidence
        // layer, reachable and replayable, but off the retrieval surface, which
        // is the whole point of filtering after persistence rather than before.
        engine.index_state(&state).await?;

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
