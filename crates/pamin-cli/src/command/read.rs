//! `pamin read` — read a topic's current or historical state.

use anyhow::{Result, bail};
use pamin_client::command::read::{Args, Read};
use pamin_core::VersionOffset;
use pamin_store::repository;

use pamin_client::command::validity;

use crate::command::resolve;
use crate::session::Session;

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

    let Some(state) =
        repository::topic_state(database.pool(), project, topic.id, resolved.version).await?
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
