//! `pamin unlink` — retract a relationship without erasing that it was claimed.

use anyhow::{Result, bail};
use pamin_core::{EdgeKind, TombstoneReason};
use pamin_store::{Database, Workspace, graph, repository};
use serde::Serialize;

use crate::output::Format;

#[derive(clap::Args)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// Which relationship to retract.
    #[arg(long, default_value = "related_to")]
    pub kind: String,

    /// Why it is being retracted, which decides what history keeps.
    ///
    /// `closed` means the relationship ended: it held until now, and queries
    /// about earlier instants still find it. `deleted` means the claim was
    /// wrong: it never held, and no query finds it at any instant.
    #[arg(long, default_value = "closed", value_parser = ["closed", "deleted"])]
    pub reason: String,
}

#[derive(Serialize)]
struct Unlinked {
    from: String,
    to: String,
    kind: String,
    reason: String,
    /// False when nothing was open to retract.
    closed: bool,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format, args: Args) -> Result<()> {
    let Some(kind) = EdgeKind::parse(&args.kind) else {
        bail!("unknown relationship kind {:?}", args.kind);
    };

    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;

    let Some(from) = repository::find_topic(database.pool(), project.id, &args.from).await? else {
        bail!("no topic named {}", args.from);
    };
    let Some(to) = repository::find_topic(database.pool(), project.id, &args.to).await? else {
        bail!("no topic named {}", args.to);
    };

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
    let closed =
        graph::close_edge(database.pool(), project.id, from.id, to.id, kind, reason).await?;

    let result = Unlinked {
        from: args.from,
        to: args.to,
        kind: kind.as_str().to_string(),
        reason: args.reason,
        closed,
    };

    format.emit(&result, || {
        if result.closed {
            format!(
                "Retracted {} --{}--> {} ({})",
                result.from, result.kind, result.to, result.reason
            )
        } else {
            format!(
                "Nothing to retract: {} --{}--> {} was not asserted",
                result.from, result.kind, result.to
            )
        }
    });
    Ok(())
}
