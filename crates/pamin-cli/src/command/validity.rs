//! The truth-interval flags, shared by every command that asserts one.
//!
//! RFC 3339 on both sides, so a value the CLI prints can be handed straight
//! back to it. Kept in one place because a second parser or a second validation
//! rule would be a difference nobody meant to introduce.

use anyhow::{Context, Result, bail};
use pamin_core::Validity;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

#[derive(clap::Args)]
pub struct Flags {
    /// When the claim starts holding, as RFC 3339. Open by default.
    #[arg(long)]
    pub valid_from: Option<String>,

    /// When it stops holding, as RFC 3339. Open by default.
    #[arg(long)]
    pub valid_to: Option<String>,
}

impl Flags {
    /// Parses both bounds, refusing an interval that cannot mean anything.
    pub fn parse(&self) -> Result<Validity> {
        let validity = Validity::new(
            parse(self.valid_from.as_deref(), "--valid-from")?,
            parse(self.valid_to.as_deref(), "--valid-to")?,
        );
        if validity.is_inverted() {
            bail!("--valid-to must be after --valid-from");
        }
        Ok(validity)
    }
}

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
