//! `pamin read` — read a topic's current or historical state.

use anyhow::{Result, bail};
use pamin_core::VersionOffset;
use pamin_store::repository;
use serde::{Deserialize, Serialize};

use crate::command::{resolve, validity};
use crate::session::Session;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic to read.
    pub topic: String,

    /// How many versions back from the current one. Zero is current.
    #[arg(long, default_value_t = 0)]
    pub version_offset: u32,
}

#[derive(Serialize, Deserialize)]
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
    /// When this version was recorded, RFC 3339.
    ///
    /// Comparing it across versions is the only way to answer "when did this
    /// change", which the ledger has always known and no read surface said.
    recorded_at: String,
    /// When the source claims the fact was true, RFC 3339.
    ///
    /// The other timeline. Equal to `recorded_at` unless a writer stated
    /// otherwise, and the pair is what keeps "when we believed it" separate
    /// from "when it held" -- see Two kinds of time in docs/cli.md.
    observed_at: String,
    /// The asserted truth interval. Both open unless a writer bounded them.
    #[serde(skip_serializing_if = "Option::is_none")]
    valid_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    valid_to: Option<String>,
    /// The byte range in the source this state was derived from.
    ///
    /// Moved here from `pamin search`, which put one on every hit. Ten of them
    /// cost a caller two hundred and eighty tokens to carry an identifier no
    /// command accepts -- and `read` did not carry it at all, so the claim
    /// that every state traces back to bytes in a source had no surface on the
    /// command line. One per read is where it is affordable and where somebody
    /// auditing a single claim is already looking.
    source_span: String,
}

/// Reads the topic and returns what was found, rendering nothing.
///
/// Separated from rendering because the caller that prints it is no longer the
/// only one: a resident server computes this and hands it back to a client
/// that does the printing.
pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Read> {
    let database = session.database();
    let project = session.project(project).await?;

    let topic = resolve::topic(database, project, &args.topic).await?;

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
        recorded_at: validity::render(state.recorded_at),
        observed_at: validity::render(state.observed_at),
        valid_from: state.validity.from.map(validity::render),
        valid_to: state.validity.to.map(validity::render),
        source_span: state.source_span_id.to_string(),
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
        "{} v{} ({marker}, {} of {} versions, recorded {})\n\n{}",
        result.topic,
        result.version,
        result.actual_version_offset,
        result.available_versions,
        result.recorded_at,
        result.content
    )
}
