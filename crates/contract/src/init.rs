//! `pamin-memory init` — provision the local database.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Initialized {
    pub project: String,
    pub home: String,
}

/// Renders the result for a person reading it.
pub fn render(result: &Initialized) -> String {
    format!("Initialized project {} in {}", result.project, result.home)
}
