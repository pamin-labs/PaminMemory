//! `pamin grep` — find exact strings in the evidence, with nothing ranking.

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The exact string to find. Not a pattern.
    pub literal: String,

    /// Ignore case.
    #[arg(short = 'i', long)]
    pub ignore_case: bool,

    /// How many matches to return.
    #[arg(long, default_value_t = 20)]
    pub limit: u32,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize)]
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
