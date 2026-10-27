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

    /// Leave sections these files no longer have searchable, and only list
    /// them, rather than taking them off the retrieval surface.
    #[arg(long)]
    pub keep_missing: bool,

    /// Ingest these paths once and stop keeping them in sync.
    ///
    /// By default the server keeps every path it ingests in sync: it watches
    /// it and ingests what changes under it, and catches up on what changed
    /// while it was not running. With this, it ingests them and forgets any
    /// earlier ingest's asking for that.
    #[arg(long)]
    #[serde(default)]
    pub no_sync: bool,
}

#[derive(Serialize, Deserialize)]
pub struct Ingested {
    /// Files ingested: read, or left unread because their record says they
    /// have not changed since they last were.
    pub files: usize,
    /// Sections in them, each one memory: cut out of the files read, and
    /// counted from the record of those left unread.
    pub sections: usize,
    /// Of those, the ones written as a new version of their memory.
    pub written: usize,
    /// The ones that said what their memory already did, which were left as
    /// they were rather than recorded again -- every section of a file left
    /// unread among them.
    pub unchanged: usize,
    /// The ones the filter held in the evidence layer.
    pub held: usize,
    /// Memories an earlier ingest recorded from these files whose section is
    /// no longer in them, or from files under these directories that are no
    /// longer there, which this ingest took off the retrieval surface. Their
    /// history stays, and ingesting the section again brings it back.
    pub removed: Vec<String>,
    /// The same, left searchable: every one of them under `--keep-missing`,
    /// and otherwise those somebody also wrote by hand.
    pub missing: Vec<String>,
    /// Sections found under a new name -- their file moved, or a heading above
    /// them renamed -- whose memory was renamed rather than recorded afresh,
    /// keeping its history. The old name still finds it.
    pub renamed: Vec<Renamed>,
    /// Files skipped, with why: not UTF-8, or unreadable.
    pub skipped: Vec<String>,
    /// Whether the projection caught up before this returned.
    pub cascade: String,
    /// Set when the projection was behind at any point during the ingest.
    pub cascade_lagging: bool,
}

/// A memory renamed by an ingest.
#[derive(Serialize, Deserialize)]
pub struct Renamed {
    pub from: String,
    pub to: String,
}

/// Renders the result for a person reading it.
pub fn render(result: &Ingested) -> String {
    let mut out = format!(
        "Ingested {} files as {} sections: {} written, {} unchanged, {} held in evidence only",
        result.files, result.sections, result.written, result.unchanged, result.held
    );
    for (names, what) in [
        (&result.removed, "were taken off the retrieval surface"),
        (&result.missing, "are still searchable"),
    ] {
        if names.is_empty() {
            continue;
        }
        out.push_str(&format!(
            "\n{} sections from an earlier ingest are no longer in these files and {what}:",
            names.len()
        ));
        for name in names {
            out.push_str(&format!("\n  {name}"));
        }
    }
    if !result.renamed.is_empty() {
        out.push_str(&format!(
            "\n{} sections were found under a new name and renamed, keeping their history:",
            result.renamed.len()
        ));
        for renamed in &result.renamed {
            out.push_str(&format!("\n  {} -> {}", renamed.from, renamed.to));
        }
    }
    for skipped in &result.skipped {
        out.push_str(&format!("\nskipped {skipped}"));
    }
    out
}
