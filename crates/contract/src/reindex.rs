//! `pamin-memory reindex` — rebuild the projection index from PostgreSQL.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Reindexed {
    /// Topics written to the projection. One document each, whatever a topic's
    /// version history holds -- searching ranks what a topic says now.
    pub indexed: usize,
    /// Of those, topics that kept a vector already embedded from their text,
    /// the replaced index's or one kept in PostgreSQL, rather than being
    /// embedded again.
    pub reused: usize,
    /// Topics whose current-state pointer disagreed with the ledger and was
    /// corrected. Zero unless something stopped maintaining it.
    pub repaired_pointers: u64,
    /// Topic names restated in the index that answers which topics a memory
    /// names.
    pub names: usize,
}

/// Renders the result for a person reading it.
pub fn render(result: &Reindexed) -> String {
    let mut rendered = format!(
        "Rebuilt the index from postgres: {} topics ({} reusing their stored vectors), {} topic names",
        result.indexed, result.reused, result.names
    );
    if result.repaired_pointers > 0 {
        rendered.push_str(&format!(
            "\nRepaired {} topics pointing at the wrong current state",
            result.repaired_pointers
        ));
    }
    rendered
}
