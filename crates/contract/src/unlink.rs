//! `pamin unlink` — retract a relationship without erasing that it was claimed.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Retract a relationship, keeping the record that it was claimed.
#[derive(clap::Args, Serialize, Deserialize, JsonSchema)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// Which relationship to retract.
    #[arg(long, default_value_t = default_kind())]
    #[serde(default = "default_kind")]
    pub kind: String,

    /// Why it is being retracted, which decides what history keeps.
    ///
    /// `closed` means the relationship ended: it held until now, and queries
    /// about earlier instants still find it. `deleted` means the claim was
    /// wrong: it never held, and no query finds it at any instant.
    #[arg(long, default_value_t = default_reason(), value_parser = ["closed", "deleted"])]
    #[serde(default = "default_reason")]
    pub reason: String,
}

/// The relationship `link` asserts and `unlink` retracts when none is named.
pub(crate) fn default_kind() -> String {
    "related_to".to_string()
}

fn default_reason() -> String {
    "closed".to_string()
}

#[derive(Serialize, Deserialize, JsonSchema)]
pub struct Unlinked {
    pub from: String,
    pub to: String,
    pub kind: String,
    pub reason: String,
    /// False when nothing was open to retract.
    pub closed: bool,
}

/// Renders the result for a person reading it.
pub fn render(result: &Unlinked) -> String {
    if result.closed {
        format!(
            "Retracted {} --{}--> {} ({})",
            result.from, result.kind, result.to, result.reason
        )
    } else {
        format!(
            "Nothing to retract: {} --{}--> {} was not asserted",
            result.from, result.kind, result.to
        )
    }
}
