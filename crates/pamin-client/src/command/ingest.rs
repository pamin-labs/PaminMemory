//! `pamin ingest` — record files as memories, one per section.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(clap::Args, Serialize, Deserialize)]
pub struct Args {
    /// Markdown or plain-text files, or directories to read them from.
    ///
    /// A directory is read recursively for `.md`, `.markdown` and `.txt`
    /// files, skipping hidden entries, symbolic links and what git ignores.
    /// Made absolute here, before the request leaves, so the server reads what
    /// this directory calls these paths rather than what its own does.
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
}

#[derive(Serialize, Deserialize)]
pub struct Ingested {
    /// Files read.
    pub files: usize,
    /// Sections cut out of them, each one memory.
    pub sections: usize,
    /// Of those, the ones written as a new version of their memory.
    pub written: usize,
    /// The ones that said what their memory already did, which were left as
    /// they were rather than recorded again.
    pub unchanged: usize,
    /// The ones the filter held in the evidence layer.
    pub held: usize,
    /// Memories an earlier ingest recorded from these files whose section is
    /// no longer in them, or from files under these directories that are no
    /// longer there. They stay searchable: nothing here removes a memory.
    pub missing: Vec<String>,
    /// Files skipped, with why: not UTF-8, or unreadable.
    pub skipped: Vec<String>,
    /// Whether the projection caught up before this returned.
    pub cascade: String,
    /// Set when the projection was behind at any point during the ingest.
    pub cascade_lagging: bool,
}

/// Renders the result for a person reading it.
pub fn render(result: &Ingested) -> String {
    let mut out = format!(
        "Ingested {} files as {} sections: {} written, {} unchanged, {} held in evidence only",
        result.files, result.sections, result.written, result.unchanged, result.held
    );
    if !result.missing.is_empty() {
        out.push_str(&format!(
            "\n{} sections from an earlier ingest are no longer in these files and are still searchable:",
            result.missing.len()
        ));
        for name in &result.missing {
            out.push_str(&format!("\n  {name}"));
        }
    }
    for skipped in &result.skipped {
        out.push_str(&format!("\nskipped {skipped}"));
    }
    out
}
