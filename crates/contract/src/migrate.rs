//! `pamin-memory migrate` — apply a release's migrations to an external database.
//!
//! A deploy runs it once, before the servers that need the new schema start,
//! and its log should say what it changed.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Migrated {
    /// The migration versions this run applied, oldest first; empty when the
    /// database was already current.
    pub applied: Vec<i64>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Migrated) -> String {
    match result.applied.as_slice() {
        [] => "The database was already migrated".to_string(),
        applied => format!("Applied migrations {applied:?}"),
    }
}
