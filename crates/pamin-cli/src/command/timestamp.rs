//! Parsing and rendering the timestamps the graph commands accept.
//!
//! RFC 3339 on both sides, so a value this CLI prints can be handed straight
//! back to it. Kept in one place because three commands take these flags and a
//! second format would be a difference nobody meant to introduce.

use anyhow::{Context, Result};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// Parses an optional RFC 3339 timestamp, naming the flag if it is malformed.
pub fn parse(value: Option<&str>, flag: &str) -> Result<Option<OffsetDateTime>> {
    value
        .map(|raw| {
            OffsetDateTime::parse(raw, &Rfc3339)
                .with_context(|| format!("{flag} expects an RFC 3339 timestamp, got {raw:?}"))
        })
        .transpose()
}

pub fn render(at: OffsetDateTime) -> String {
    at.format(&Rfc3339).expect("rfc 3339 is always renderable")
}
