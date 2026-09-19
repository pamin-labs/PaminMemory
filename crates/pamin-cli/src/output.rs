//! Rendering results as text or JSON.

use serde::Serialize;

/// How to render a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Text,
    /// Machine-readable. Compact unless a person asked to read it.
    Json {
        pretty: bool,
    },
}

impl Format {
    pub fn from_flags(json: bool, pretty: bool) -> Self {
        if json {
            Self::Json { pretty }
        } else {
            Self::Text
        }
    }

    /// Prints a result, using the JSON shape or the caller's text rendering.
    ///
    /// JSON exists because the primary consumer is an agent parsing output, and
    /// the text form exists because the primary reviewer is a person reading it.
    ///
    /// Compact by default, because the agent pays for the whitespace. Indenting
    /// a ten-hit search costs about a thousand tokens of a context window and
    /// buys nothing a parser wants; `--pretty` is for the person who has piped
    /// it to a terminal, and is the rarer case.
    pub fn emit<T: Serialize>(self, value: &T, text: impl FnOnce() -> String) {
        match self {
            Self::Json { pretty: true } => println!(
                "{}",
                serde_json::to_string_pretty(value).expect("serializable result")
            ),
            Self::Json { pretty: false } => println!(
                "{}",
                serde_json::to_string(value).expect("serializable result")
            ),
            Self::Text => println!("{}", text()),
        }
    }
}
