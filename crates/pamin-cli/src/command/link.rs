//! `pamin link` — assert a relationship the text does not state.

use anyhow::{Result, bail};
use pamin_core::EdgeKind;
use pamin_store::graph::EdgeClaim;
use pamin_store::{Database, Workspace, graph, repository};
use serde::Serialize;

use crate::command::timestamp;
use crate::output::Format;

#[derive(clap::Args)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// What the relationship is: mentions, supports, contradicts, supersedes,
    /// related_to, part_of, derived_from, same_as, or depends_on.
    #[arg(long, default_value = "related_to")]
    pub kind: String,

    /// When the relationship starts holding, as RFC 3339. Open by default.
    #[arg(long)]
    pub valid_from: Option<String>,

    /// When it stops holding, as RFC 3339. Open by default.
    #[arg(long)]
    pub valid_to: Option<String>,
}

#[derive(Serialize)]
struct Linked {
    from: String,
    to: String,
    kind: String,
    version: u32,
    /// False when an identical claim was already live, in which case nothing
    /// was written.
    appended: bool,
    valid_from: Option<String>,
    valid_to: Option<String>,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format, args: Args) -> Result<()> {
    let Some(kind) = EdgeKind::parse(&args.kind) else {
        bail!("unknown relationship kind {:?}", args.kind);
    };

    let mut database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.client(), project).await?;

    let from = require_topic(&database, project.id, &args.from).await?;
    let to = require_topic(&database, project.id, &args.to).await?;
    if from == to {
        bail!("a topic cannot be related to itself");
    }

    let mut claim = EdgeClaim::explicit(kind);
    claim.valid_from = timestamp::parse(args.valid_from.as_deref(), "--valid-from")?;
    claim.valid_to = timestamp::parse(args.valid_to.as_deref(), "--valid-to")?;
    if let (Some(start), Some(end)) = (claim.valid_from, claim.valid_to)
        && end <= start
    {
        bail!("--valid-to must be after --valid-from");
    }

    let assertion = graph::assert_edge(database.client_mut(), project.id, from, to, &claim).await?;

    let result = Linked {
        from: args.from,
        to: args.to,
        kind: kind.as_str().to_string(),
        version: assertion.version().version,
        appended: assertion.is_new(),
        valid_from: claim.valid_from.map(timestamp::render),
        valid_to: claim.valid_to.map(timestamp::render),
    };

    format.emit(&result, || {
        if result.appended {
            format!(
                "{} --{}--> {} (v{})",
                result.from, result.kind, result.to, result.version
            )
        } else {
            format!(
                "Already linked: {} --{}--> {} (v{})",
                result.from, result.kind, result.to, result.version
            )
        }
    });
    Ok(())
}

/// Resolves a topic name, refusing to invent one.
///
/// Linking a topic that does not exist is almost always a typo, and creating it
/// silently would leave an edge pointing at an empty identity that nothing can
/// ever resolve to a state.
async fn require_topic(
    database: &Database,
    project: pamin_core::ProjectId,
    name: &str,
) -> Result<pamin_core::TopicId> {
    match repository::find_topic(database.client(), project, name).await? {
        Some(topic) => Ok(topic.id),
        None => bail!("no topic named {name}"),
    }
}
