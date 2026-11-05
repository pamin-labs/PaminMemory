//! Every operation's schema, as callers see it, against its snapshot.
//!
//! The contract evolves additively: a new optional field changes a snapshot
//! and is accepted; removing a field or changing its type or meaning needs a
//! new major contract version. A failure here is that review, not noise --
//! update the snapshot with `cargo insta review` only for an additive change.

use memory_contract::*;
use schemars::schema_for;

macro_rules! snapshot {
    ($($name:literal => $type:ty),* $(,)?) => {
        $(insta::assert_json_snapshot!($name, schema_for!($type));)*
    };
}

#[test]
fn every_operation_matches_its_schema_snapshot() {
    snapshot! {
        "search.args" => search::Args,
        "search.result" => search::Results,
        "read.args" => read::Args,
        "read.result" => read::Read,
        "grep.args" => grep::Args,
        "grep.result" => grep::Matches,
        "neighbors.args" => neighbors::Args,
        "neighbors.result" => neighbors::Neighborhood,
        "topics.args" => topics::Args,
        "topics.result" => topics::Topics,
        "write.args" => write::Args,
        "write.result" => write::Written,
        "link.args" => link::Args,
        "link.result" => link::Linked,
        "unlink.args" => unlink::Args,
        "unlink.result" => unlink::Unlinked,
        "import.args" => import::Args,
        "import.result" => import::Imported,
        "ingest.args" => ingest::Args,
        "ingest.result" => ingest::Ingested,
        "sync.args" => sync::Args,
        "sync.result" => sync::Synced,
        "unsync.args" => unsync::Args,
        "unsync.result" => unsync::Unsynced,
        "reindex.args" => reindex::Args,
        "reindex.result" => reindex::Reindexed,
        "purge.args" => purge::Args,
        "purge.result" => purge::Purged,
        "cascade.args" => cascade::Args,
        "cascade.drain.result" => cascade::Drained,
        "cascade.failed.result" => cascade::Failures,
        "cascade.moved.result" => cascade::Moved,
        "init.result" => init::Initialized,
        "migrate.result" => migrate::Migrated,
        "stop.result" => stop::Stopped,
    }
}
