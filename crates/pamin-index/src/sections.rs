//! Cutting a file into the memories `pamin ingest` records.
//!
//! A file is not one memory. The embedder reads the first 512 tokens of what it
//! is given and the reranker the first 256 of a query and a passage together,
//! so a file recorded whole is found by its opening and judged by its opening,
//! and the rest of it is evidence nothing retrieves. Cut into sections, each
//! is a memory of its own: one document in the index, one candidate in fusion
//! and one passage the reranker reads in full, which is the shape every part of
//! search already assumes.
//!
//! The cut follows the file's own structure, which is what the measurements of
//! chunking agree on: structural or fixed-size cuts of a few hundred tokens with
//! no overlap match or beat semantic chunking on realistic corpora at a
//! fraction of the cost, and what a chunk gains most from is knowing where it
//! sits. So a heading -- ATX, or setext under a paragraph -- opens a section,
//! a section is named by the file's path and the headings above it -- which
//! the embedder and the reranker both read ahead of the content, as
//! `{name}: {content}` -- and a section too long to be read whole is cut by
//! [`pack`](crate::spans::pack) at its blocks, then its sentences, then its
//! tokens.
//!
//! **Sized so the reranker reads all of it.** A section's content is at most
//! what fits the reranker's window beside its name, a query of
//! [`QUERY_TOKENS`] and the pair's special tokens. Larger sections would be
//! cut short at reranking, and this project has measured a longer reranker
//! window to buy nothing on the corpora it has.
//!
//! **A section too small to carry a claim joins a neighbour.** A heading with
//! nothing under it is a title for what follows, so it joins the next section;
//! a few words under a heading of their own belong with what came before them.
//! Under [`LEAST_TOKENS`], a section would otherwise be a passage the reranker
//! scores on its name alone.
//!
//! A pure function of the path, the text and the token counter, so ingesting
//! the same file twice yields the same sections with the same names, and an
//! edit changes only the section it falls in -- or, for a section long enough
//! to have been split, the part it falls in, the parts after it, and when it
//! leaves a part too small to stand alone, the part before.

use std::collections::HashSet;
use std::ops::Range;

use crate::spans::{TokenCount, closes, cut, fence, heading, indented, pack, trim};

/// What a query is allowed of the reranker's window when a section is sized.
///
/// The reranker truncates the longer side of a pair first, so a query longer
/// than this costs the end of a section rather than of the query; most queries
/// an agent sends are well under it.
pub const QUERY_TOKENS: usize = 48;

/// The special tokens XLM-R wraps a pair in: `<s> query </s></s> passage </s>`.
const PAIR_TOKENS: usize = 4;

/// The fewest tokens a section is left with on its own.
pub const LEAST_TOKENS: usize = 48;

/// The fewest content tokens a section is sized to, however long its name.
const LEAST_CAP: usize = 64;

/// What separates a file's path from its headings, and one heading from the
/// next, in a section's name.
pub const SEPARATOR: &str = " › ";

/// The most characters of a heading a section's name carries. A name is read
/// ahead of every section under it and costs the section what it takes of the
/// reranker's window, and a heading longer than this is a paragraph that lost
/// its line break.
const TITLE_CHARS: usize = 80;

/// The longest suffix a section's name can be given -- a number to tell it from
/// another section under the same headings, and a part -- which is what a
/// section is sized for, so that no part of it is pushed past the cap by its
/// own name.
const LONGEST_SUFFIX: &str = " #999 (9999)";

/// One section of a file: a memory's topic and content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// The file's path and the headings it sits under, then `#n` for the
    /// second and later sections under the same headings, then `(n)` for the
    /// second and later parts of a section that had to be split.
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

    // Every name this file has handed out, parts included: a heading can be
    // spelled like the number or the part another section was given, and two
    // sections under one name are one topic, which each ingest would hand
    // back and forth between them.
    let mut taken: HashSet<String> = HashSet::new();
    let mut sections = Vec::new();
    for part in merged {
        let content = text[part.range].trim();
        if content.is_empty() {
            continue;
        }
        let mut base = path.to_string();
        for heading in &part.path {
            base.push_str(SEPARATOR);
            base.push_str(heading);
        }

        let cap = cap(&format!("{base}{LONGEST_SUFFIX}"), counter);
        let pieces = if counter.count(content) <= cap {
            std::iter::once(0..content.len()).collect()
        } else {
            let packed = pack(&base, content, counter, cap)
                .into_iter()
                .map(|span| span.range)
                .collect();
            balanced(content, packed, counter, cap)
        };

        let name = unique(&mut taken, &base, pieces.len());
        for (index, range) in pieces.into_iter().enumerate() {
            sections.push(Section {
                name: part_name(&name, index),
                content: content[range].to_string(),
            });
        }
    }
    sections
}

/// The name of part `index` of a section named `name`.
fn part_name(name: &str, index: usize) -> String {
    match index {
        0 => name.to_string(),
        _ => format!("{name} ({})", index + 1),
    }
}

/// `base`, or the first of `base #2`, `base #3` and on whose `parts` parts are
/// all free, which it then takes.
fn unique(taken: &mut HashSet<String>, base: &str, parts: usize) -> String {
    let free = |name: &str, taken: &HashSet<String>| {
        (0..parts).all(|index| !taken.contains(&part_name(name, index)))
    };
    let mut name = base.to_string();
    let mut number = 1;
    while !free(&name, taken) {
        number += 1;
        name = format!("{base} #{number}");
    }
    for index in 0..parts {
        taken.insert(part_name(&name, index));
    }
    name
}

/// `pieces` of `content` with each one under [`LEAST_TOKENS`] joined to the
/// piece before it -- or after it, when it is the first -- or, when the two do
/// not fit together, the two divided again where they are most nearly equal.
///
/// [`pack`] fills a span as far as the cap allows, so a heading and the
/// opening of what follows it can leave a sliver of that block behind as a
/// part of its own: a passage of two tokens that the reranker would score on
/// its name alone.
fn balanced(
    content: &str,
    pieces: Vec<Range<usize>>,
    counter: &dyn TokenCount,
    cap: usize,
) -> Vec<Range<usize>> {
    let small = |range: &Range<usize>| counter.count(&content[range.clone()]) < LEAST_TOKENS;
    let join = |first: Range<usize>, second: Range<usize>| -> Vec<Range<usize>> {
        let joined = first.start..second.end;
        if counter.count(&content[joined.clone()]) <= cap {
            vec![joined]
        } else {
            halved(content, joined, counter, cap)
        }
    };

    let mut out: Vec<Range<usize>> = Vec::new();
    let mut carried: Option<Range<usize>> = None;
    for piece in pieces {
        if let Some(first) = carried.take() {
            let joined = join(first, piece);
            // Two slivers at the top are still one: carry it on to the next.
            match joined.as_slice() {
                [one] if small(one) => carried = Some(one.clone()),
                _ => out.extend(joined),
            }
            continue;
        }
        if !small(&piece) {
            out.push(piece);
            continue;
        }
        match out.pop() {
            Some(previous) => out.extend(join(previous, piece)),
            None => carried = Some(piece),
        }
    }
    out.extend(carried);
    out
}

/// `range` of `content`, over `cap` tokens, in two pieces that each fit,
/// divided at the sentence or line boundary nearest its middle by tokens, or
/// failing one, at the nearest whitespace -- and only when neither fits, at
/// tokens by [`cut`], which can divide a word.
///
/// The pair is a block [`pack`] already cut where the text allowed and a
/// sliver after it, so the boundary it is divided at again should be one the
/// text has rather than one the tokenizer does.
fn halved(
    content: &str,
    range: Range<usize>,
    counter: &dyn TokenCount,
    cap: usize,
) -> Vec<Range<usize>> {
    let text = &content[range.clone()];
    let offsets = counter.offsets(text);
    let middle = offsets.len() / 2;
    let tokens_before = |at: usize| offsets.partition_point(|token| token.start < at);

    let sentences: Vec<usize> = crate::spans::sentences(content, range.clone())
        .into_iter()
        .map(|sentence| sentence.start - range.start)
        .chain(text.match_indices('\n').map(|(at, _)| at + 1))
        .collect();
    let spaces: Vec<usize> = text
        .char_indices()
        .filter(|&(at, c)| c.is_whitespace() && at > 0)
        .map(|(at, _)| at)
        .collect();

    for boundaries in [sentences, spaces] {
        let mut boundaries: Vec<usize> = boundaries
            .into_iter()
            .filter(|&at| at > 0 && at < text.len())
            .collect();
        boundaries.sort_by_key(|&at| tokens_before(at).abs_diff(middle));
        for at in boundaries.into_iter().take(8) {
            let pieces = [
                trim(content, range.start..range.start + at),
                trim(content, range.start + at..range.end),
            ];
            if pieces
                .iter()
                .all(|piece| !piece.is_empty() && counter.count(&content[piece.clone()]) <= cap)
            {
                return pieces.to_vec();
            }
        }
    }
    cut(content, range, counter, cap)
        .into_iter()
        .map(|(range, _)| range)
        .collect()
}

/// `text` with a byte-order mark removed and every line ending, Windows' and
/// old Macintosh's, made `\n`: a heading is found by the line it is on.
fn normalized(text: &str) -> String {
    text.strip_prefix('\u{feff}')
        .unwrap_or(text)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

/// How many content tokens a section named `name` may hold.
fn cap(name: &str, counter: &dyn TokenCount) -> usize {
    let named = counter.count(&format!("{name}: "));
    crate::reranking::MAX_TOKENS
        .saturating_sub(PAIR_TOKENS + QUERY_TOKENS + named)
        .max(LEAST_CAP)
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
        paragraph = match paragraph.take() {
            _ if blank => None,
            Some((from, mut lines, had_body)) => {
                lines.push(line.trim());
                Some((from, lines, had_body))
            }
            None if opens_paragraph(line) => Some((here, vec![line.trim()], has_body)),
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

/// Whether `line` can open a paragraph, which a setext underline turns into a
/// heading: not a list item, a quotation or a table row, which an underline
/// after them is a rule under, and not indented code.
fn opens_paragraph(line: &str) -> bool {
    let Some(rest) = indented(line) else {
        return false;
    };
    let ordered = rest.split_once(['.', ')']).is_some_and(|(number, _)| {
        !number.is_empty() && number.chars().all(|c| c.is_ascii_digit())
    });
    !(ordered
        || rest.starts_with(['>', '|'])
        || ["- ", "* ", "+ "]
            .iter()
            .any(|marker| rest.starts_with(marker)))
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

    /// Up to three characters of a word a token, so a cut at tokens can fall
    /// inside a word -- which [`Words`] never lets happen -- the way XLM-R's
    /// pieces can.
    struct Pieces;

    impl TokenCount for Pieces {
        fn offsets(&self, text: &str) -> Vec<Range<usize>> {
            Words
                .offsets(text)
                .into_iter()
                .flat_map(|word| {
                    let starts: Vec<usize> = text[word.clone()]
                        .char_indices()
                        .map(|(at, _)| word.start + at)
                        .step_by(3)
                        .collect();
                    let ends = starts.iter().skip(1).copied().chain([word.end]);
                    starts
                        .iter()
                        .copied()
                        .zip(ends)
                        .map(|(start, end)| start..end)
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    /// A full piece and a sliver after
    /// it, which together are over the cap, are divided again at a boundary
    /// the text has -- here a sentence -- and not at a piece of a word.
    #[test]
    fn a_full_piece_and_a_sliver_are_divided_at_a_sentence() {
        let sentences: Vec<String> = (0..12)
            .map(|i| format!("Sentence{i} carries considerable wordings."))
            .collect();
        let content = sentences.join(" ");
        let sliver = content.rfind(" Sentence11").unwrap() + 1;
        let cap = Pieces.count(&content[..sliver]) + 2;
        assert!(
            Pieces.count(&content) > cap,
            "the pair has to be over the cap"
        );
        assert!(Pieces.count(&content[sliver..]) < LEAST_TOKENS);

        let pieces = balanced(
            &content,
            vec![0..sliver - 1, sliver..content.len()],
            &Pieces,
            cap,
        );
        assert_eq!(pieces.len(), 2, "{pieces:?}");
        for piece in &pieces {
            assert!(Pieces.count(&content[piece.clone()]) <= cap);
            let text = &content[piece.clone()];
            assert!(
                text.starts_with("Sentence") && text.ends_with('.'),
                "{text:?}"
            );
        }
    }

    /// Two slivers at the top of a split section are one sliver, and are
    /// carried on until they are not.
    #[test]
    fn slivers_at_the_top_are_carried_until_they_are_not() {
        let pieces = vec![0..1, 2..3, 4..200];
        let content = format!("a b {}", "c ".repeat(98));
        let balanced = balanced(&content, pieces, &Words, 500);
        assert_eq!(balanced.len(), 1);
        assert_eq!(balanced[0], 0..200);
    }

    /// A paragraph of `n` words, each one token to [`Words`].
    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Every section fits what the reranker reads beside its name, no two
    /// share a name, and between them they hold every word of `expected` -- the
    /// file as the test states it should be kept, worked out by the test and
    /// not by the code under it -- in order.
    fn check(expected: &str, sections: &[Section]) {
        for section in sections {
            let cap = cap(&section.name, &Words);
            assert!(
                Words.count(&section.content) <= cap.max(1),
                "{} holds {} tokens, over {cap}",
                section.name,
                Words.count(&section.content)
            );
        }
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

    #[test]
    fn a_long_section_is_split_into_numbered_parts_that_each_fit() {
        let text = format!(
            "# Guide\n\n{}\n\n{}\n\n{}\n",
            words(150),
            words(150),
            words(150)
        );
        let sections = sections("guide.md", &text, &Words);
        assert!(sections.len() >= 3, "{:?}", names(&sections));
        assert_eq!(sections[0].name, "guide.md › Guide");
        for (index, section) in sections.iter().enumerate().skip(1) {
            assert_eq!(section.name, format!("guide.md › Guide ({})", index + 1));
        }
        check(&text, &sections);
    }

    #[test]
    fn a_plain_text_file_with_no_headings_is_cut_by_size_alone() {
        let text = (0..8).map(|_| words(80)).collect::<Vec<_>>().join("\n\n");
        let sections = sections("notes.txt", &text, &Words);
        assert!(sections.len() > 1);
        assert_eq!(sections[0].name, "notes.txt");
        assert_eq!(sections[1].name, "notes.txt (2)");
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

    /// A heading spelled like the number or the part another section was
    /// given does not take that section's name.
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

        let text = format!("# Foo\n\n{}\n\n# Foo (2)\n\n{}\n", words(400), words(60));
        let parts = sections("f.md", &text, &Words);
        check(&text, &parts);
        assert!(
            names(&parts).contains(&"f.md › Foo (2) #2"),
            "{:?}",
            names(&parts)
        );
    }

    /// A section packed up to the cap leaves no sliver behind as a part of its
    /// own, whether the sliver is the end of a block or a heading of a few
    /// words that `merged` joined to it.
    #[test]
    fn no_part_is_a_sliver() {
        for text in [
            format!("# Foo\n\n{}\n", words(400)),
            format!(
                "# Setup\n\n{}\n\n## Note\n\nrestart afterwards\n",
                words(250)
            ),
        ] {
            let sections = sections("f.md", &text, &Words);
            check(&text, &sections);
            for section in &sections {
                assert!(
                    Words.count(&section.content) >= LEAST_TOKENS,
                    "{} is {} tokens: {:?}",
                    section.name,
                    Words.count(&section.content),
                    names(&sections)
                );
            }
        }
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

    /// The cap leaves the reranker room for the name, the query and the pair's
    /// own tokens, and never falls below the floor however long a name is.
    #[test]
    fn the_cap_is_what_the_reranker_has_left() {
        assert_eq!(
            cap("a.md", &Words),
            // `a.md:` is one token to `Words`.
            crate::reranking::MAX_TOKENS - PAIR_TOKENS - QUERY_TOKENS - 1
        );
        assert_eq!(cap(&words(400), &Words), LEAST_CAP);
    }
}
