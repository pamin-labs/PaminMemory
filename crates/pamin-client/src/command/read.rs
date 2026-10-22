//! `pamin read` — read a topic's current or historical state.

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic to read.
    pub topic: String,

    /// How many versions back from the current one. Zero is current.
    #[arg(long, default_value_t = 0)]
    pub version_offset: u32,
}

#[derive(Serialize, Deserialize)]
pub struct Read {
    pub topic: String,
    pub version: u32,
    pub content: String,
    /// Whether this is the current state.
    pub is_current: bool,
    /// How far back the read actually reached, which differs from the request
    /// when it ran past the oldest surviving version.
    pub actual_version_offset: u32,
    pub oldest_version: u32,
    pub latest_version: u32,
    pub available_versions: u32,
    /// When this version was recorded, RFC 3339.
    ///
    /// Comparing it across versions is the only way to answer "when did this
    /// change", which the ledger has always known and no read surface said.
    pub recorded_at: String,
    /// When the source claims the fact was true, RFC 3339.
    ///
    /// The other timeline. Equal to `recorded_at` unless a writer stated
    /// otherwise, and the pair is what keeps "when we believed it" separate
    /// from "when it held" -- see Two kinds of time in docs/cli.md.
    pub observed_at: String,
    /// The asserted truth interval. Both open unless a writer bounded them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_from: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_to: Option<String>,
    /// The byte range in the source this state was derived from.
    ///
    /// Moved here from `pamin search`, which put one on every hit. Ten of them
    /// cost a caller two hundred and eighty tokens to carry an identifier no
    /// command accepts -- and `read` did not carry it at all, so the claim
    /// that every state traces back to bytes in a source had no surface on the
    /// command line. One per read is where it is affordable and where somebody
    /// auditing a single claim is already looking.
    pub source_span: String,
}

/// Renders the result for a person reading it.
pub fn render(result: &Read) -> String {
    let marker = if result.is_current {
        "current"
    } else {
        "historical"
    };
    format!(
        "{} v{} ({marker}, {} of {} versions, recorded {})\n\n{}",
        result.topic,
        result.version,
        result.actual_version_offset,
        result.available_versions,
        result.recorded_at,
        result.content
    )
}
