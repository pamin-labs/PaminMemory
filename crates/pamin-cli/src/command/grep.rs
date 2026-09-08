//! `pamin grep` — find exact strings in the evidence, with nothing ranking.
//!
//! Search asks what is relevant. This asks what is literally there, in the
//! verbatim source, including content the sensory filter held and no index
//! ever saw. It is the route back to the original when a memory lost something
//! on its way to becoming one.

use anyhow::Result;
use pamin_store::{Database, Workspace, repository};
use serde::Serialize;

use crate::output::Format;

/// Characters of surrounding text to show on each side of a match.
const CONTEXT: usize = 60;

#[derive(clap::Args)]
pub struct Args {
    /// The exact string to find. Not a pattern.
    pub literal: String,

    /// Ignore case.
    #[arg(short = 'i', long)]
    pub ignore_case: bool,

    /// How many matches to return.
    #[arg(long, default_value_t = 20)]
    pub limit: u32,
}

#[derive(Serialize)]
struct Match {
    /// Where the evidence came from.
    source: String,
    source_version: String,
    version: u32,
    /// `promoted` or `filtered`. Filtered evidence is reachable here and
    /// nowhere else.
    filter_decision: String,
    /// Why the filter decided as it did.
    filter_reason: String,
    /// The match with surrounding text, for reading rather than for parsing.
    excerpt: String,
}

#[derive(Serialize)]
struct Matches {
    literal: String,
    matches: Vec<Match>,
}

pub async fn run(workspace: &Workspace, project: &str, format: Format, args: Args) -> Result<()> {
    let database = Database::open(workspace).await?;
    let project = repository::ensure_project(database.pool(), project).await?;

    let hits = repository::grep_evidence(
        database.pool(),
        project.id,
        &args.literal,
        !args.ignore_case,
        args.limit,
    )
    .await?;

    let result = Matches {
        literal: args.literal,
        matches: hits
            .iter()
            .map(|hit| Match {
                source: hit.locator.clone(),
                source_version: hit.source_version.id.to_string(),
                version: hit.source_version.version,
                filter_decision: format!("{:?}", hit.source_version.filter_decision).to_lowercase(),
                filter_reason: hit.source_version.filter_reason.clone(),
                excerpt: excerpt(&hit.source_version.content, hit.offset),
            })
            .collect(),
    };

    format.emit(&result, || {
        if result.matches.is_empty() {
            return format!("No evidence contains {:?}", result.literal);
        }
        result
            .matches
            .iter()
            .map(|hit| {
                format!(
                    "{} v{} ({})\n        {}",
                    hit.source, hit.version, hit.filter_decision, hit.excerpt
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    });
    Ok(())
}

/// Renders the text around a match.
///
/// Boundaries are moved to character edges rather than byte ones. Slicing a
/// UTF-8 string at an arbitrary byte panics, and evidence is stored in whatever
/// language it arrived in, so most content here is multi-byte.
fn excerpt(content: &str, offset: usize) -> String {
    let start = floor_boundary(content, offset.saturating_sub(CONTEXT));
    let end = ceil_boundary(content, (offset + CONTEXT).min(content.len()));

    let mut rendered = String::new();
    if start > 0 {
        rendered.push('…');
    }
    rendered.push_str(content[start..end].trim());
    if end < content.len() {
        rendered.push('…');
    }
    rendered
}

fn floor_boundary(content: &str, mut index: usize) -> usize {
    while index > 0 && !content.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn ceil_boundary(content: &str, mut index: usize) -> usize {
    while index < content.len() && !content.is_char_boundary(index) {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::excerpt;

    #[test]
    fn a_short_match_is_shown_whole() {
        assert_eq!(excerpt("cone ten overnight", 0), "cone ten overnight");
    }

    #[test]
    fn a_long_match_is_elided_on_both_sides() {
        let content = "a".repeat(200);
        let rendered = excerpt(&content, 100);
        assert!(rendered.starts_with('…') && rendered.ends_with('…'));
    }

    #[test]
    fn boundaries_move_to_character_edges() {
        // Slicing multi-byte text at an arbitrary byte panics, and most
        // evidence here is multi-byte.
        let content = "部署流水线运行在持续集成上面".repeat(20);
        let rendered = excerpt(&content, 121);
        assert!(rendered.contains('流'));
    }
}
