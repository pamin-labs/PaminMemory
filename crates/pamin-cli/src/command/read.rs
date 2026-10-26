//! `pamin read` — read a topic's current or historical state.

use anyhow::{Result, bail};
use pamin_client::command::read::{Args, Read, Span};
use pamin_core::VersionOffset;
use pamin_engine::SpanView;
use pamin_index::VectorIndex;
use pamin_store::repository;

use pamin_client::command::validity;

use crate::command::resolve;
use crate::session::Session;

/// Reads the topic and returns what was found, rendering nothing.
///
/// Separated from rendering because the caller that prints it is no longer the
/// only one: a resident server computes this and hands it back to a client
/// that does the printing.
pub async fn execute(
    session: &Session,
    project_name: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Read> {
    let database = session.database();
    let project = session.project(project_name).await?;

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

    // Cut by the project's index, which is what search numbered the spans by.
    // A historical version is cut the same way: spans are computed, not kept.
    let (content, span) = match args.span {
        None => (state.content, None),
        Some(ordinal) => {
            let engine = session.engine(project_name, vector_index).await?;
            let spans = engine.cut(&topic.name, &state.content)?;
            let Some(view) = SpanView::around(&spans, ordinal, args.around) else {
                bail!(
                    "{} v{} has {} spans, numbered from 0",
                    args.topic,
                    resolved.version,
                    spans.len()
                );
            };
            (
                state.content[view.range.clone()].to_string(),
                Some(Span {
                    ordinal: view.ordinal,
                    of: view.of,
                    header: view.header,
                    start: view.range.start,
                    end: view.range.end,
                }),
            )
        }
    };

    let result = Read {
        // The name it goes by now, which differs from the one asked for when
        // that was a name it was renamed from.
        topic: topic.name,
        version: resolved.version,
        content,
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
        span,
    };

    Ok(result)
}
