//! `pamin read` — read a topic's current or historical state.

use anyhow::{Result, bail};
use pamin_core::VersionOffset;
use pamin_store::{Database, Workspace, repository};
use serde::Serialize;

#[derive(clap::Args)]
pub struct Args {
    /// The topic to read.
    pub topic: String,

    /// How many versions back from the current one. Zero is current.
    #[arg(long, default_value_t = 0)]
    pub version_offset: u32,
}

#[derive(Serialize)]
pub struct Read {
    topic: String,
    version: u32,
    content: String,
    /// Whether this is the current state.
    is_current: bool,
    /// How far back the read actually reached, which differs from the request
    /// when it ran past the oldest surviving version.
    actual_version_offset: u32,
    oldest_version: u32,
    latest_version: u32,
    available_versions: u32,
}

/// Reads the topic and returns what was found, rendering nothing.
///
/// Separated from rendering because the caller that prints it is no longer the
/// only one: a resident server computes this and hands it back to a client
/// that does the printing.
pub async fn execute(workspace: &Workspace, project: &str, args: Args) -> Result<Read> {
    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;

    let Some(topic) = repository::find_topic(database.pool(), project.id, &args.topic).await?
    else {
        bail!("no topic named {}", args.topic);
    };

    let versions = repository::topic_versions(database.pool(), topic.id).await?;
    let Some(resolved) = pamin_core::resolve(&versions, VersionOffset(args.version_offset)) else {
        bail!("topic {} has no live versions", args.topic);
    };

    let Some(state) = repository::topic_state(database.pool(), topic.id, resolved.version).await?
    else {
        bail!("version {} of {} is missing", resolved.version, args.topic);
    };

    let result = Read {
        topic: args.topic,
        version: resolved.version,
        content: state.content,
        is_current: resolved.is_current,
        actual_version_offset: resolved.actual_offset.0,
        oldest_version: resolved.oldest_version,
        latest_version: resolved.latest_version,
        available_versions: resolved.available_versions,
    };

    Ok(result)
}

/// Renders the result for a person reading it.
pub fn render(result: &Read) -> String {
    let marker = if result.is_current {
        "current"
    } else {
        "historical"
    };
    format!(
        "{} v{} ({marker}, {} of {} versions)\n\n{}",
        result.topic,
        result.version,
        result.actual_version_offset,
        result.available_versions,
        result.content
    )
}
