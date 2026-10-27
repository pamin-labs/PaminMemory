//! `pamin search` — retrieve memories, with the reasoning attached.

use anyhow::Result;
use pamin_client::command::search::{Args, Hit, Results, Span, Trace};
use pamin_client::command::validity;
use pamin_engine::Depths;
use pamin_index::VectorIndex;

use crate::session::Session;

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Results> {
    let engine = session.engine(project, vector_index).await?;
    let depths = Depths {
        channel: args.channel_depth,
        graph: args.graph_depth,
        rerank: args.rerank_depth,
        effort: args.search_effort,
    };
    let ingested = session.ingested(project);
    let mut hits = engine
        .search_reranked(&args.query, args.limit, depths)
        .await?;
    // A hit from a file that changed since it was ingested answers with what
    // the file said then: the file is ingested now, and the search asked
    // again if that changed what pamin holds -- or if the watcher's ingest
    // did while this ranked, which leaves the file looking fresh and the hits
    // stale. A failure is the watcher's to report, not the search's.
    let topics: Vec<String> = hits.iter().map(|hit| hit.topic.clone()).collect();
    if let Err(error) = crate::watch::freshen(session, project, vector_index, &topics).await {
        tracing::warn!(error = format!("{error:#}"), "freshening hits failed");
    }
    if session.ingested(project) != ingested {
        hits = engine
            .search_reranked(&args.query, args.limit, depths)
            .await?;
    }

    let results = Results {
        query: args.query,
        hits: hits
            .into_iter()
            .map(|hit| Hit {
                topic: hit.topic,
                version: hit.state.version,
                content: match &hit.span {
                    Some(span) => hit.state.content[span.range.clone()].to_string(),
                    None => hit.state.content,
                },
                score: hit.result.score,
                why: hit.result.why.into_iter().map(Trace::from).collect(),
                recorded_at: validity::render(hit.state.recorded_at),
                valid_from: hit.state.validity.from.map(validity::render),
                valid_to: hit.state.validity.to.map(validity::render),
                span: hit.span.map(|span| Span {
                    ordinal: span.ordinal,
                    of: span.of,
                    header: span.header,
                    start: span.range.start,
                    end: span.range.end,
                }),
                parent: hit.parent,
            })
            .collect(),
    };

    Ok(results)
}
