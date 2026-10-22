//! `pamin unlink` — retract a relationship without erasing that it was claimed.

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic the relationship starts from.
    pub from: String,

    /// The topic it points at.
    pub to: String,

    /// Which relationship to retract.
    #[arg(long, default_value = "related_to")]
    pub kind: String,

    /// Why it is being retracted, which decides what history keeps.
    ///
    /// `closed` means the relationship ended: it held until now, and queries
    /// about earlier instants still find it. `deleted` means the claim was
    /// wrong: it never held, and no query finds it at any instant.
    #[arg(long, default_value = "closed", value_parser = ["closed", "deleted"])]
    pub reason: String,
}

#[derive(Serialize, Deserialize)]
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
