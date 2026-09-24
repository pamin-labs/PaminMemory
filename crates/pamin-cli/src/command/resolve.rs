//! Turning what a caller typed into what the store holds.
//!
//! Six commands need one of two things -- a topic by name, or a relationship
//! kind by name -- and each had its own copy of the refusal. Four spellings of
//! "no topic named X" and three of "unknown relationship kind X", which is
//! four and three chances for one of them to start saying something else.
//!
//! **These reach `pamin-store` directly, and `docs/architecture.md` says that
//! is what only `pamin-engine` does.** The reason is cost rather than
//! oversight: opening an engine opens the projection index, measured at 489 MB
//! and about 600 ms on a 13,014-document project, and `pamin link` has no use
//! for an index. Deferring the *model* took 1,075 MB and four seconds off that
//! (see `Engine`'s `embedder` field); deferring the index as well is what
//! would let these go through the engine, and it moves a blocking open under
//! the lock whose misuse once wedged the engine for thirty-five minutes. So
//! the exception stands, in one module, named -- rather than spread across six
//! commands as if nobody had noticed.

use anyhow::{Result, bail};
use pamin_core::{EdgeKind, ProjectId, Topic, TopicId};
use pamin_store::{Database, repository};

/// The topic with this name, refusing to invent one.
///
/// Linking or reading a topic that does not exist is almost always a typo, and
/// creating one silently would leave a name pointing at an empty identity that
/// nothing can ever resolve to a state.
pub async fn topic(database: &Database, project: ProjectId, name: &str) -> Result<Topic> {
    match repository::find_topic(database.pool(), project, name).await? {
        Some(topic) => Ok(topic),
        None => bail!("no topic named {name}"),
    }
}

/// The identifier of the topic with this name.
pub async fn topic_id(database: &Database, project: ProjectId, name: &str) -> Result<TopicId> {
    Ok(topic(database, project, name).await?.id)
}

/// The relationship kind this name spells.
pub fn edge_kind(name: &str) -> Result<EdgeKind> {
    match EdgeKind::parse(name) {
        Some(kind) => Ok(kind),
        None => bail!("unknown relationship kind {name:?}"),
    }
}

/// Every one of these names as a relationship kind, or the first that is not.
pub fn edge_kinds(names: &[String]) -> Result<Vec<EdgeKind>> {
    names.iter().map(|name| edge_kind(name)).collect()
}
