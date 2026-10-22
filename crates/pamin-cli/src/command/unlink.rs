//! `pamin unlink` — retract a relationship without erasing that it was claimed.

use anyhow::Result;
use pamin_client::command::unlink::{Args, Unlinked};
use pamin_core::TombstoneReason;
use pamin_store::graph;

use crate::command::resolve;
use crate::session::Session;

pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Unlinked> {
    let kind = resolve::edge_kind(&args.kind)?;

    let database = session.database();
    let project = session.project(project).await?;

    let from = resolve::topic_id(database, project, &args.from).await?;
    let to = resolve::topic_id(database, project, &args.to).await?;

    // The reason is not bookkeeping. It decides whether a question about an
    // earlier instant still finds this edge: a relationship that ended did
    // hold before it ended, and one that was never true never held at all.
    // Recording both as the same retraction erases the difference and, with
    // it, the history.
    let reason = match args.reason.as_str() {
        "deleted" => TombstoneReason::Deleted,
        // The value parser admits nothing else.
        _ => TombstoneReason::Closed,
    };

    // The rows stay either way, so what was believed and when stays
    // answerable, and the truth interval is untouched.
    let closed = graph::close_edge(database.pool(), project, from, to, kind, reason).await?;

    let result = Unlinked {
        from: args.from,
        to: args.to,
        kind: kind.as_str().to_string(),
        reason: args.reason,
        closed,
    };

    Ok(result)
}
