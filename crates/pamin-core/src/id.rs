//! Typed identifiers.
//!
//! Every identifier is a UUID rather than a sequence. Monotonic sequences are a
//! coordination point that distributed PostgreSQL-compatible engines handle
//! poorly, and swapping them out later would mean rewriting every foreign key.
//!
//! **Version 7, time-ordered, rather than version 4.** A random key lands on a
//! random leaf of every B-tree it is indexed in, so each insert dirties a page
//! nobody else is writing and splits leave them half full; a key that leads
//! with its creation time lands on the rightmost leaf, next to the rows written
//! just before it. Measured on PostgreSQL 17, 200,000 rows in batches of 5,000
//! behind a primary key and a `(project, id)` index, three rounds each: 5.0-5.5 s
//! with version 4 against 3.5-4.3 s with version 7, and a primary key of
//! 8.5 MB against 8.1 MB. The shard key is `project_id`, not these bytes, so
//! ordering them by time skews nothing in PostgreSQL.
//!
//! It did skew the vector index, whose engine keeps a key map it never
//! compacts: keys that only ever ascend left it one file per flush. That index
//! spells a topic's key with these bytes reversed, so the random ones lead --
//! see `Keys` in `pamin-index`'s `projection.rs`.
//!
//! Identifiers already written stay version 4: a topic's id is also its key in
//! the vector index, so rewriting one means rebuilding the index, and the gain
//! is on where new rows land, which the old ones do not affect. Where ids break
//! a tie -- equal fused scores, equal path strengths -- the older of two new
//! topics now comes first, where before the order was arbitrary.

use std::fmt;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! typed_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub Uuid);

        impl $name {
            /// Generates a fresh identifier.
            pub fn new() -> Self {
                Self(Uuid::now_v7())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl From<Uuid> for $name {
            fn from(value: Uuid) -> Self {
                Self(value)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

typed_id!(
    /// A local namespace for memories and sources, and the shard key for every table.
    ProjectId
);
typed_id!(
    /// A file, folder, chat log, manual write, or API call that produced evidence.
    SourceId
);
typed_id!(
    /// An immutable snapshot of a source's content.
    SourceVersionId
);
typed_id!(
    /// A byte range into a source version.
    SourceSpanId
);
typed_id!(
    /// A stable topic identity whose states carry the content.
    TopicId
);
typed_id!(
    /// One immutable version of a topic's content.
    TopicStateId
);
typed_id!(
    /// A durable cascade work item in the outbox.
    IndexJobId
);
typed_id!(
    /// A stable identity for one edge between two topics.
    RelationshipId
);
typed_id!(
    /// One immutable fact about an edge.
    RelationshipVersionId
);
