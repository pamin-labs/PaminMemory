//! `pamin link` — assert a relationship the text does not state.

use serde::{Deserialize, Serialize};

use crate::command::validity;

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// What the relationship is: mentions, supports, contradicts, supersedes,
    /// related_to, part_of, derived_from, same_as, or depends_on.
    #[arg(long, default_value = "related_to")]
    pub kind: String,

    #[command(flatten)]
    pub validity: validity::Flags,
}

#[derive(Serialize, Deserialize)]
pub struct Linked {
    pub from: String,
    pub to: String,
    pub kind: String,
    pub version: u32,
    /// False when an identical claim was already live, in which case nothing
    /// was written.
    pub appended: bool,
    pub valid_from: Option<String>,
    pub valid_to: Option<String>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Linked) -> String {
    if result.appended {
        format!(
            "{} --{}--> {} (v{})",
            result.from, result.kind, result.to, result.version
        )
    } else {
        format!(
            "Already linked: {} --{}--> {} (v{})",
            result.from, result.kind, result.to, result.version
        )
    }
}
