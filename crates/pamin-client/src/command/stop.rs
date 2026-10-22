//! `pamin stop` — shut down the local database server.
//!
//! The server is deliberately left running between commands so an agent does
//! not pay cluster startup on every call. That makes an explicit way to stop it
//! part of the contract rather than an extra.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Stopped {
    pub stopped: bool,
}

/// Renders the result for a person reading it.
pub fn render(_result: &Stopped) -> String {
    "Stopped the local database server".to_string()
}
