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
}

#[derive(Serialize)]
struct Unlinked {
    from: String,
    to: String,
    kind: String,
    /// False when nothing was open to retract.
    closed: bool,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format, args: Args) -> Result<()> {
    let Some(kind) = EdgeKind::parse(&args.kind) else {
        bail!("unknown relationship kind {:?}", args.kind);
    };

    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.client(), project).await?;

    let Some(from) = repository::find_topic(database.client(), project.id, &args.from).await?
    else {
        bail!("no topic named {}", args.from);
    };
    let Some(to) = repository::find_topic(database.client(), project.id, &args.to).await? else {
        bail!("no topic named {}", args.to);
    };

    // Retracts the claim. The rows stay, so what was believed and when stays
    // answerable, and the truth interval is untouched: this says we no longer
    // assert the relationship, not that it ended at this instant.
    let closed = graph::close_edge(
        database.client(),
        project.id,
        from.id,
        to.id,
        kind,
        TombstoneReason::Deleted,
    )
    .await?;

    let result = Unlinked {
        from: args.from,
        to: args.to,
        kind: kind.as_str().to_string(),
        closed,
    };

    format.emit(&result, || {
        if result.closed {
            format!(
                "Retracted {} --{}--> {}",
                result.from, result.kind, result.to
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
