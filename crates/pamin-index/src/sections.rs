//! Cutting a file into the memories `pamin ingest` records.
//!
//! A file is not one memory. Its sections are about different things, and a
//! section is what a caller asks after and what a link points at, so each
//! is a memory of its own: one topic, named by where it sits, whose history
//! is the history of that section.
//!
//! The cut follows the file's own structure. A heading -- ATX, or setext
//! under a paragraph -- opens a section, and a section is named by the
//! file's path and the headings above it, which the embedder and the
//! reranker both read ahead of the content.
//!
//! **Size is not decided here.** A section of any length is one topic. The
//! index cuts every memory into spans it can read whole, and ranks a topic by
//! its best span, so a long section is found by any part of it without being
//! split into topics that compete with each other for the reranker's slots
//! and can show nothing of their neighbours. Deciding a unit's size in two
//! places would let the two disagree.
//!
//! **A section too small to carry a claim joins a neighbour.** A heading with
//! nothing under it is a title for what follows, so it joins the next section;
//! a few words under a heading of their own belong with what came before them.
//! Under [`LEAST_TOKENS`], a section would otherwise be a passage the reranker
//! scores on its name alone.
//!
//! A pure function of the path, the text and the token counter, so ingesting
//! the same file twice yields the same sections with the same names, and an
//! edit changes only the section it falls in.

use std::collections::HashSet;
use std::ops::Range;

use crate::spans::{TokenCount, closes, fence, heading, indented};

/// The fewest tokens a section is left with on its own.
pub const LEAST_TOKENS: usize = 48;

/// What separates a file's path from its headings, and one heading from the
/// next, in a section's name.
pub const SEPARATOR: &str = " › ";

/// The most characters of a heading a section's name carries. A name is read
/// ahead of every span of the section and costs each what it takes of the
/// reranker's window, and a heading longer than this is a paragraph that lost
/// its line break.
const TITLE_CHARS: usize = 80;

/// One section of a file: a memory's topic and content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// The file's path and the headings it sits under, then `#n` for the
    /// second and later sections under the same headings.
    pub name: String,
    pub content: String,
}

/// `text`, read from the file at `path`, cut into sections.
///
/// `path` is the name the file goes by -- relative to the root it was ingested
/// from, so the same file ingested from the same root names its sections the
/// same way however the command was spelled.
pub fn sections(path: &str, text: &str, counter: &dyn TokenCount) -> Vec<Section> {
    let text = normalized(text);
    let body = without_front_matter(&text);

    let raw = headed(&text, body);
    let merged = merged(&text, raw, counter);

    // Every name this file has handed out: a heading can be spelled like the
    // number another section was given, and two sections under one name are
    // one topic, which each ingest would hand back and forth between them.
    let mut taken: HashSet<String> = HashSet::new();
    let mut sections = Vec::new();
    for section in merged {
        let content = text[section.range].trim();
        if content.is_empty() {
            continue;
        }
        let mut base = path.to_string();
        for heading in &section.path {
            base.push_str(SEPARATOR);
            base.push_str(heading);
        }
        sections.push(Section {
            name: unique(&mut taken, &base),
            content: content.to_string(),
        });
    }
    sections
}

/// `base`, or the first of `base #2`, `base #3` and on that is free, which it
/// then takes.
fn unique(taken: &mut HashSet<String>, base: &str) -> String {
    let mut name = base.to_string();
    let mut number = 1;
    while taken.contains(&name) {
        number += 1;
        name = format!("{base} #{number}");
    }
    taken.insert(name.clone());
    name
}

/// `text` with a byte-order mark removed and every line ending, Windows' and
/// old Macintosh's, made `\n`: a heading is found by the line it is on.
fn normalized(text: &str) -> String {
    text.strip_prefix('\u{feff}')
        .unwrap_or(text)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// The range of `text` after a YAML front matter block, which is metadata
/// about the file rather than anything it says.
///
/// Recognized narrowly, because a mistake here loses text rather than
/// misfiling it: a file opening with a thematic break `---` and a later one
/// would otherwise lose everything between them. So the block must open with
/// a `key:` line and hold no blank line before it closes.
fn without_front_matter(text: &str) -> Range<usize> {
    let whole = 0..text.len();
    let Some(rest) = text.strip_prefix("---\n") else {
        return whole;
    };
    let keyed = rest
        .split('\n')
        .next()
        .and_then(|line| line.split_once(':'));
    if !keyed.is_some_and(|(key, _)| {
        !key.is_empty()
            && key
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-'))
    }) {
        return whole;
    }
    let mut at = 4;
    for line in rest.split_inclusive('\n') {
        at += line.len();
        match line.trim_end() {
            "---" | "..." => return at..text.len(),
            "" => return whole,
            _ => {}
        }
    }
    // Never closed, so it was not front matter.
    whole
}

/// A run of the file under one heading, before small ones are merged.
#[derive(Debug)]
struct Headed {
    /// The headings it sits under, outermost first, its own last.
    path: Vec<String>,
    range: Range<usize>,
    /// Whether anything but its heading is in it.
    has_body: bool,
}

/// `body` of `text` cut at every heading outside a fenced block: ATX, or a
/// paragraph a setext underline makes one.
fn headed(text: &str, body: Range<usize>) -> Vec<Headed> {
    let mut sections = Vec::new();
    // The headings above the current line, with their levels.
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut start = body.start;
    let mut has_body = false;
    let mut open_fence = None;
    // The paragraph the current line continues, which a setext underline
    // after it would make a heading: where it began, its lines, and whether
    // the section had anything in it before the paragraph did.
    let mut paragraph: Option<(usize, Vec<&str>, bool)> = None;
    // Inside a list, a quotation, a table or an HTML block, until a blank
    // line: what follows on its lines is its, not a paragraph a setext
    // underline could make a heading of.
    let mut contained = false;

    // A heading over `from..`, titled `title`, at `level`.
    let mut open = |from: usize,
                    title: &str,
                    level: usize,
                    had_body: bool,
                    start: &mut usize,
                    has_body: &mut bool,
                    sections: &mut Vec<Headed>| {
        if from > *start || had_body {
            sections.push(Headed {
                path: stack.iter().map(|(_, title)| title.clone()).collect(),
                range: *start..from,
                has_body: had_body,
            });
        }
        stack.retain(|(above, _)| *above < level);
        // A bare `#` closes the headings below its level and names nothing.
        if !title.is_empty() {
            stack.push((level, title.chars().take(TITLE_CHARS).collect()));
        }
        *start = from;
        *has_body = false;
    };

    let mut at = body.start;
    for line in text[body.clone()].split_inclusive('\n') {
        let here = at;
        at += line.len();
        let line = line.strip_suffix('\n').unwrap_or(line);

        if let Some(opening) = open_fence {
            if closes(line, opening) {
                open_fence = None;
            }
            has_body = true;
            continue;
        }
        if let Some(opening) = fence(line) {
            open_fence = Some(opening);
            paragraph = None;
            contained = false;
            has_body = true;
            continue;
        }
        if let Some(level) = setext(line)
            && let Some((from, lines, had_body)) = paragraph.take()
        {
            let title = lines.join(" ");
            open(
                from,
                &title,
                level,
                had_body,
                &mut start,
                &mut has_body,
                &mut sections,
            );
            continue;
        }
        if let Some(title) = heading(line) {
            paragraph = None;
            contained = false;
            open(
                here,
                title,
                level(line),
                has_body,
                &mut start,
                &mut has_body,
                &mut sections,
            );
            continue;
        }

        let blank = line.trim().is_empty();
        let container = opens_container(line);
        contained = !blank && (contained || container);
        paragraph = match paragraph.take() {
            _ if blank || contained => None,
            Some((from, mut lines, had_body)) => {
                lines.push(line.trim());
                Some((from, lines, had_body))
            }
            None if indented(line).is_some() => Some((here, vec![line.trim()], has_body)),
            None => None,
        };
        has_body |= !blank;
    }
    sections.push(Headed {
        path: stack.into_iter().map(|(_, title)| title).collect(),
        range: start..body.end,
        has_body,
    });
    sections.retain(|section| !text[section.range.clone()].trim().is_empty());
    sections
}

/// The level of the setext heading `line` underlines, if it is an underline:
/// a run of `=` for the first level or of `-` for the second, indented at most
/// three spaces. Only after a paragraph -- elsewhere a run of `-` is a rule.
fn setext(line: &str) -> Option<usize> {
    let rest = indented(line)?.trim_end();
    let marker = rest.chars().next()?;
    let level = match marker {
        '=' => 1,
        '-' => 2,
        _ => return None,
    };
    rest.chars().all(|c| c == marker).then_some(level)
}

/// Whether `line` opens a block a setext underline cannot follow as a
/// heading's: a list item, a quotation, a table row or HTML.
fn opens_container(line: &str) -> bool {
    let Some(rest) = indented(line) else {
        return false;
    };
    let ordered = rest.split_once(['.', ')']).is_some_and(|(number, after)| {
        !number.is_empty()
            && number.chars().all(|c| c.is_ascii_digit())
            && (after.is_empty() || after.starts_with(' '))
    });
    ordered
        || rest.starts_with(['>', '|'])
        || html_block(rest)
        || ["-", "*", "+"]
            .iter()
            .any(|marker| rest == *marker || rest.starts_with(&format!("{marker} ")))
}

/// Whether `rest` opens an HTML block: a comment, or a tag of an element
/// CommonMark reads as a block. An inline tag -- `<kbd>`, `<br>` -- or an
/// autolink opens a paragraph like any other text.
fn html_block(rest: &str) -> bool {
    const BLOCKS: [&str; 30] = [
        "address",
        "article",
        "aside",
        "blockquote",
        "details",
        "dialog",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "ul",
        "script",
    ];
    if rest.starts_with("<!--") {
        return true;
    }
    let Some(tag) = rest.strip_prefix("</").or_else(|| rest.strip_prefix('<')) else {
        return false;
    };
    let name: String = tag
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    let after = tag[name.len()..].chars().next();
    BLOCKS.contains(&name.as_str()) && after.is_none_or(|c| matches!(c, ' ' | '>' | '/'))
}

/// The level of the ATX heading `line` is.
fn level(line: &str) -> usize {
    let rest = indented(line).unwrap_or(line);
    rest.len() - rest.trim_start_matches('#').len()
}

/// `sections` with each one under [`LEAST_TOKENS`] joined to a neighbour: a
/// heading with nothing under it to the section after it, anything else to the
/// section before it -- or after it, at the top of the file.
fn merged(text: &str, sections: Vec<Headed>, counter: &dyn TokenCount) -> Vec<Headed> {
    let small = |section: &Headed| counter.count(&text[section.range.clone()]) < LEAST_TOKENS;

    let mut out: Vec<Headed> = Vec::new();
    // A heading waiting to open the section after it.
    let mut waiting: Option<Headed> = None;
    for mut section in sections {
        // Named by its own headings, which under a heading that had nothing
        // else to say are that heading's and one more.
        if let Some(title) = waiting.take() {
            section.range.start = title.range.start;
            section.has_body |= title.has_body;
        }
        if small(&section) {
            match out.last_mut() {
                Some(previous) if section.has_body => {
                    previous.range.end = section.range.end;
                }
                // Nothing before it to join, or nothing in it but a title.
                _ => waiting = Some(section),
            }
            continue;
        }
        out.push(section);
    }
    // A heading at the very end, with nothing after it to open.
    if let Some(title) = waiting {
        match out.last_mut() {
            Some(previous) => previous.range.end = title.range.end,
            None => out.push(title),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spans::tests::Words;

    fn names(sections: &[Section]) -> Vec<&str> {
        sections
            .iter()
            .map(|section| section.name.as_str())
            .collect()
    }

    /// A paragraph of `n` words, each one token to [`Words`].
    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// No two sections share a name, and between them they hold every word of
    /// `expected` -- the file as the test states it should be kept, worked out
    /// by the test and not by the code under it -- in order.
    fn check(expected: &str, sections: &[Section]) {
        let names: HashSet<&str> = sections.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names.len(), sections.len(), "a name repeats: {sections:?}");
        let written: Vec<&str> = sections
            .iter()
            .flat_map(|section| section.content.split_whitespace())
            .collect();
        let expected: Vec<&str> = expected.split_whitespace().collect();
        assert_eq!(written, expected);
    }

    #[test]
    fn headings_open_sections_named_by_the_headings_above_them() {
        let text = format!(
            "{}\n\n# Deploy\n\n{}\n\n## Rollback\n\n{}\n\n# Oncall\n\n{}\n",
            words(60),
            words(60),
            words(60),
            words(60)
        );
        let sections = sections("docs/ops.md", &text, &Words);
        assert_eq!(
            names(&sections),
            [
                "docs/ops.md",
                "docs/ops.md › Deploy",
                "docs/ops.md › Deploy › Rollback",
                "docs/ops.md › Oncall",
            ]
        );
        // The heading stays in its section, so the lexical channels, which
        // index content alone, can match it.
        assert!(sections[2].content.starts_with("## Rollback"));
        check(&text, &sections);
    }

    #[test]
    fn a_hash_inside_a_fence_is_not_a_heading() {
        let text = format!(
            "# Build\n\n{}\n\n```sh\n# not a heading\nmake\n```\n\n{}\n",
            words(30),
            words(30)
        );
        let sections = sections("build.md", &text, &Words);
        assert_eq!(names(&sections), ["build.md › Build"]);
        assert!(sections[0].content.contains("# not a heading"));
        check(&text, &sections);
    }

    #[test]
    fn a_repeated_heading_is_numbered_rather_than_overwritten() {
        let text = format!(
            "# Api\n\n{}\n\n## Example\n\n{}\n\n## Example\n\n{}\n",
            words(60),
            words(60),
            words(60)
        );
        let sections = sections("api.md", &text, &Words);
        assert_eq!(
            names(&sections),
            [
                "api.md › Api",
                "api.md › Api › Example",
                "api.md › Api › Example #2"
            ]
        );
        check(&text, &sections);
    }

    #[test]
    fn a_heading_with_nothing_under_it_opens_the_next_section() {
        let text = format!("# Usage\n\n## Flags\n\n{}\n", words(60));
        let sections = sections("cli.md", &text, &Words);
        assert_eq!(names(&sections), ["cli.md › Usage › Flags"]);
        assert!(sections[0].content.starts_with("# Usage\n\n## Flags"));
        check(&text, &sections);
    }

    #[test]
    fn a_few_words_under_their_own_heading_join_the_section_before() {
        let text = format!(
            "# Setup\n\n{}\n\n## Note\n\nrestart afterwards\n",
            words(60)
        );
        let sections = sections("setup.md", &text, &Words);
        assert_eq!(names(&sections), ["setup.md › Setup"]);
        assert!(sections[0].content.ends_with("restart afterwards"));
        check(&text, &sections);
    }

    #[test]
    fn a_small_file_is_one_section() {
        let sections = sections("todo.txt", "buy milk\n", &Words);
        assert_eq!(
            sections,
            [Section {
                name: "todo.txt".into(),
                content: "buy milk".into()
            }]
        );
    }

    #[test]
    fn a_short_opening_joins_the_first_section() {
        let text = format!("Ops handbook.\n\n# Deploy\n\n{}\n", words(60));
        let sections = sections("ops.md", &text, &Words);
        assert_eq!(names(&sections), ["ops.md › Deploy"]);
        assert!(sections[0].content.starts_with("Ops handbook."));
        check(&text, &sections);
    }

    #[test]
    fn a_trailing_heading_joins_the_section_before() {
        let text = format!("# Notes\n\n{}\n\n## Later\n", words(60));
        let sections = sections("notes.md", &text, &Words);
        assert_eq!(names(&sections), ["notes.md › Notes"]);
        check(&text, &sections);
    }

    /// However long a section is, it is one topic: the index cuts it into
    /// spans, not this.
    #[test]
    fn a_long_section_is_one_section() {
        let text = format!(
            "# Guide\n\n{}\n\n{}\n\n{}\n",
            words(400),
            words(400),
            words(400)
        );
        let sections = sections("guide.md", &text, &Words);
        assert_eq!(names(&sections), ["guide.md › Guide"]);
        check(&text, &sections);
    }

    #[test]
    fn a_plain_text_file_with_no_headings_is_one_section() {
        let text = (0..8).map(|_| words(80)).collect::<Vec<_>>().join("\n\n");
        let sections = sections("notes.txt", &text, &Words);
        assert_eq!(names(&sections), ["notes.txt"]);
        check(&text, &sections);
    }

    #[test]
    fn front_matter_is_left_out() {
        let body = format!("# Ops\n\n{}\n", words(60));
        let text = format!("---\ntitle: ops\ntags: [a]\n---\n{body}");
        let sections = sections("ops.md", &text, &Words);
        assert_eq!(names(&sections), ["ops.md › Ops"]);
        check(&body, &sections);
    }

    #[test]
    fn an_unclosed_front_matter_is_kept_as_text() {
        let text = format!("---\nnot front matter\n\n{}\n", words(60));
        let sections = sections("odd.md", &text, &Words);
        assert!(sections[0].content.contains("not front matter"));
        check(&text, &sections);
    }

    /// A thematic break is not front matter, however the file goes on: losing
    /// what sits between two of them is the one mistake here that drops text.
    #[test]
    fn a_rule_at_the_top_is_not_front_matter() {
        for text in [
            format!(
                "---\n\nIntro {}\n\n---\n\n# Real\n\n{}\n",
                words(60),
                words(60)
            ),
            format!(
                "---\nnote x\nsome text {}\n...\nmore {}\n",
                words(30),
                words(30)
            ),
            format!(
                "---\nkey: value\n\nprose after a blank {}\n---\n",
                words(60)
            ),
        ] {
            check(&text, &sections("f.md", &text, &Words));
        }
    }

    /// A heading spelled like the number another section was given does not
    /// take that section's name.
    #[test]
    fn a_heading_spelled_like_a_suffix_gets_a_name_of_its_own() {
        let text = format!(
            "# Example\n\n{}\n\n# Example\n\n{}\n\n# Example #2\n\n{}\n",
            words(60),
            words(60),
            words(60)
        );
        let numbered = sections("f.md", &text, &Words);
        check(&text, &numbered);
        assert_eq!(
            names(&numbered),
            [
                "f.md › Example",
                "f.md › Example #2",
                "f.md › Example #2 #2"
            ]
        );
    }

    /// A byte-order mark does not hide the first heading, a bare `#` names
    /// nothing, and old Macintosh line endings are lines.
    #[test]
    fn odd_files_still_cut_at_their_headings() {
        let text = format!("\u{feff}# Top\n\n{}\n\n#\n\n{}\n", words(60), words(60));
        let sections = sections("f.md", &text, &Words);
        assert_eq!(names(&sections), ["f.md › Top", "f.md"]);
        check(text.trim_start_matches('\u{feff}'), &sections);

        let unix = format!("# A\n\n{}\n\n# B\n\n{}\n", words(60), words(60));
        assert_eq!(
            super::sections("f.md", &unix.replace('\n', "\r"), &Words),
            super::sections("f.md", &unix, &Words)
        );
    }

    /// A line of `=` or `-` under a paragraph makes the paragraph a heading of
    /// the first or second level, nesting as an ATX heading of that level
    /// would.
    #[test]
    fn a_setext_underline_makes_a_heading() {
        let text = format!(
            "Deploy\n======\n\n{}\n\nRollback\n--------\n\n{}\n\n### Window\n\n{}\n",
            words(60),
            words(60),
            words(60)
        );
        let sections = sections("f.md", &text, &Words);
        assert_eq!(
            names(&sections),
            [
                "f.md › Deploy",
                "f.md › Deploy › Rollback",
                "f.md › Deploy › Rollback › Window"
            ]
        );
        assert!(sections[1].content.starts_with("Rollback\n--------"));
        check(&text, &sections);
    }

    /// An inline tag or an autolink opens a paragraph like other text, and a
    /// heading or a fence after a list ends the list's hold on what follows.
    #[test]
    fn a_setext_heading_follows_inline_html_and_ended_containers() {
        for (text, title) in [
            ("<kbd>Ctrl</kbd> copies\n---\n", "<kbd>Ctrl</kbd> copies"),
            (
                "<https://example.com> explains\n---\n",
                "<https://example.com> explains",
            ),
            ("- item\n# Section\nTitle\n---\n", "Title"),
            ("- item\n```\ncode\n```\nTitle\n---\n", "Title"),
        ] {
            let text = format!("{text}\n{}\n", words(60));
            let sections = sections("f.md", &text, &Words);
            assert!(
                names(&sections)
                    .iter()
                    .any(|name| name.ends_with(&format!("› {title}"))),
                "{text:?} gave {:?}",
                names(&sections)
            );
            check(&text, &sections);
        }
    }

    /// Every line of the paragraph above an underline is the heading's title.
    #[test]
    fn a_setext_title_can_run_over_lines() {
        let text = format!("Deploy and\n  roll back\n===\n\n{}\n", words(60));
        let sections = sections("f.md", &text, &Words);
        assert_eq!(names(&sections), ["f.md › Deploy and roll back"]);
        check(&text, &sections);
    }

    /// `---` after a blank line, a list, a quotation, a table or inside a
    /// fence is a rule or text, not an underline.
    #[test]
    fn a_rule_is_not_an_underline() {
        for between in [
            "\n\n---\n",
            "\n\n- item\n---\n",
            "\n\n> quoted\n---\n",
            "\n\n| a | b |\n---\n",
            "\n\n1. first\n---\n",
            "\n\n```\ncode\n---\n```\n",
            "\n\n    indented code\n---\n",
            // What follows a list item or a quotation on its lines is its own,
            // and a list item interrupts a paragraph rather than joining it.
            "\n\n- item\n  continued\n---\n",
            "\n\n> quoted\nlazily continued\n---\n",
            "\n\nA paragraph\n- item\n---\n",
            "\n\n<div>\n---\n",
            "\n\n<!-- a comment -->\n---\n",
        ] {
            let text = format!("# Top\n\n{}{between}\n{}\n", words(60), words(60));
            let sections = sections("f.md", &text, &Words);
            assert_eq!(names(&sections), ["f.md › Top"], "{between:?}");
            check(&text, &sections);
        }
    }

    #[test]
    fn a_long_heading_is_shortened_in_the_name() {
        let title = "x".repeat(500);
        let text = format!("# {title}\n\n{}\n", words(60));
        let sections = sections("f.md", &text, &Words);
        assert_eq!(
            sections[0].name,
            format!("f.md › {}", "x".repeat(TITLE_CHARS))
        );
        check(&text, &sections);
    }

    #[test]
    fn windows_line_endings_cut_like_unix_ones() {
        let unix = format!("# A\n\n{}\n\n# B\n\n{}\n", words(60), words(60));
        let windows = unix.replace('\n', "\r\n");
        assert_eq!(
            sections("f.md", &unix, &Words),
            sections("f.md", &windows, &Words)
        );
    }

    /// An edit inside one section leaves every other section exactly as it
    /// was, which is what lets a re-ingest skip them.
    #[test]
    fn an_edit_changes_only_the_section_it_is_in() {
        let before = format!(
            "# A\n\n{}\n\n# B\n\n{}\n\n# C\n\n{}\n",
            words(60),
            words(60),
            words(60)
        );
        let after = before.replacen("# B\n\nw0 ", "# B\n\nedited w0 ", 1);
        let (before, after) = (
            sections("f.md", &before, &Words),
            sections("f.md", &after, &Words),
        );
        assert_eq!(names(&before), names(&after));
        let changed: Vec<&str> = before
            .iter()
            .zip(&after)
            .filter(|(old, new)| old.content != new.content)
            .map(|(old, _)| old.name.as_str())
            .collect();
        assert_eq!(changed, ["f.md › B"]);
    }

    #[test]
    fn nothing_but_whitespace_is_no_sections() {
        assert!(sections("empty.md", " \n\n\t\n", &Words).is_empty());
        assert!(sections("empty.md", "", &Words).is_empty());
    }
}
