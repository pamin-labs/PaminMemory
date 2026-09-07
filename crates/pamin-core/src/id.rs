//! Typed identifiers.
//!
//! Every identifier is a UUID rather than a sequence. Monotonic sequences are a
//! coordination point that distributed PostgreSQL-compatible engines handle
//! poorly, and swapping them out later would mean rewriting every foreign key.

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
                Self(Uuid::new_v4())
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
