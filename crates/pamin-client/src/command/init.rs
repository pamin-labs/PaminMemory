//! `pamin init` — provision the local database.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Initialized {
    pub project: String,
    pub home: String,
}

/// Renders the result for a person reading it.
pub fn render(result: &Initialized) -> String {
    format!("Initialized project {} in {}", result.project, result.home)
}
