//! `pamin link` — assert a relationship the text does not state.

use anyhow::{Result, bail};
use pamin_client::command::link::{Args, Linked};
use pamin_client::command::validity;
use pamin_store::graph;
use pamin_store::graph::EdgeClaim;

use crate::command::resolve;
use crate::session::Session;

pub async fn execute(session: &Session, project: &str, args: Args) -> Result<Linked> {
    let kind = resolve::edge_kind(&args.kind)?;

    let database = session.database();
    let project = session.project(project).await?;

    let from = resolve::topic_id(database, project, &args.from).await?;
    let to = resolve::topic_id(database, project, &args.to).await?;
    if from == to {
        bail!("a topic cannot be related to itself");
    }

    let mut claim = EdgeClaim::explicit(kind);
    claim.validity = args.validity.parse()?;

    let assertion = graph::assert_edge(database.pool(), project, from, to, &claim).await?;

    let result = Linked {
        from: args.from,
        to: args.to,
        kind: kind.as_str().to_string(),
        version: assertion.version().version,
        appended: assertion.is_new(),
        valid_from: claim.validity.from.map(validity::render),
        valid_to: claim.validity.to.map(validity::render),
    };

    Ok(result)
}
