//! `pamin read` — read a topic's current or historical state.

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// The topic to read.
    pub topic: String,

    /// How many versions back from the current one. Zero is current.
    #[arg(long, default_value_t = 0)]
    pub version_offset: u32,

    /// Read one span of a memory the index cut into spans, from 0, rather
    /// than all of it -- what `pamin search` prints as `--span` on a hit.
    #[arg(long)]
    #[serde(default)]
    pub span: Option<u32>,

    /// With `--span`, this many spans either side as well.
    #[arg(long, default_value_t = 0, requires = "span")]
    #[serde(default)]
    pub around: u32,
}

/// Where in a memory a span, or a run of them, lies.
#[derive(Serialize, Deserialize)]
pub struct Span {
    /// The span asked about, from 0.
    pub ordinal: u32,
    /// How many spans the memory is cut into.
    pub of: u32,
    /// The memory's name, and the heading and turn the run continues -- what
    /// the run lost by being cut out of it.
    pub header: String,
    /// The run's bytes in the memory's content.
    pub start: usize,
    pub end: usize,
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
    /// With `--span`, which spans `content` is. `content` is then those spans
    /// alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
}

/// Renders the result for a person reading it.
pub fn render(result: &Read) -> String {
    let marker = if result.is_current {
        "current"
    } else {
        "historical"
    };
    let span = result.span.as_ref().map_or(String::new(), |span| {
        format!(
            "\nspan {} of {}, bytes {}..{}: {}",
            span.ordinal, span.of, span.start, span.end, span.header
        )
    });
    format!(
        "{} v{} ({marker}, {} of {} versions, recorded {}){span}\n\n{}",
        result.topic,
        result.version,
        result.actual_version_offset,
        result.available_versions,
        result.recorded_at,
        result.content
    )
}
