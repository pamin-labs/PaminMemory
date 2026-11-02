//! `pamin-memory link` — assert a relationship the text does not state.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::validity;

/// Assert a relationship between two existing topics that their text does not
/// state.
#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// What the relationship is: mentions, supports, contradicts, supersedes,
    /// related_to, part_of, derived_from, same_as, or depends_on.
    #[arg(long, default_value_t = crate::unlink::default_kind())]
    #[serde(default = "crate::unlink::default_kind")]
    pub kind: String,

    #[command(flatten)]
    #[serde(default)]
    pub validity: validity::Flags,
}

#[derive(Serialize, Deserialize, JsonSchema)]
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
