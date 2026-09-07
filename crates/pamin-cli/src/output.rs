//! Rendering results as text or JSON.

use serde::Serialize;

/// How to render a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Text,
    Json,
}

impl Format {
    pub fn from_json_flag(json: bool) -> Self {
        if json { Self::Json } else { Self::Text }
    }

    /// Prints a result, using the JSON shape or the caller's text rendering.
    ///
    /// JSON exists because the primary consumer is an agent parsing output, and
    /// the text form exists because the primary reviewer is a person reading it.
    pub fn emit<T: Serialize>(self, value: &T, text: impl FnOnce() -> String) {
        match self {
            Self::Json => println!(
                "{}",
                serde_json::to_string_pretty(value).expect("serializable result")
            ),
            Self::Text => println!("{}", text()),
        }
    }
}
