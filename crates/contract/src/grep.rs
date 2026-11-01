//! `pamin grep` — find exact strings in the evidence, with nothing ranking.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Find an exact string in everything ever written, including writes the
/// filter held back from memory. No ranking and no model.
#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {
    /// The exact string to find. Not a pattern.
    pub literal: String,

    /// Ignore case.
    #[arg(short = 'i', long)]
    #[serde(default)]
    pub ignore_case: bool,

    /// How many matches to return.
    #[arg(long, default_value_t = default_limit())]
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    20
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Match {
    /// Where the evidence came from.
    pub source: String,
    pub source_version: String,
    pub version: u32,
    /// `promoted` or `filtered`. Filtered evidence is reachable here and
    /// nowhere else.
    pub filter_decision: String,
    /// Why the filter decided as it did.
    pub filter_reason: String,
    /// The match with surrounding text, for reading rather than for parsing.
    pub excerpt: String,
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Matches {
    pub literal: String,
    pub matches: Vec<Match>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Matches) -> String {
    if result.matches.is_empty() {
        return format!("No evidence contains {:?}", result.literal);
    }
    result
        .matches
        .iter()
        .map(|hit| {
            format!(
                "{} v{} ({})\n        {}",
                hit.source, hit.version, hit.filter_decision, hit.excerpt
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
