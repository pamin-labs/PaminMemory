//! Cutting a memory into the spans the index holds a document for.
//!
//! The vector channel sees the first 512 tokens of what it embeds and the
//! reranker the first 256 of what it is shown, so a memory longer than that is
//! found by its opening and by nothing after it. Cutting it into spans that
//! each fit is what lets the rest be found.
//!
//! The cut follows the text's own structure. A memory is read as blocks --
//! fenced code, ATX headings, conversation turns and paragraphs -- and the
//! blocks are packed in order into spans of at most `cap` tokens, with no
//! overlap and a heading always opening a span. A block too large for one span
//! is cut where its kind allows: code at its lines, prose at its sentences,
//! and a line or sentence still too large at its tokens, into pieces of near
//! equal size.
//!
//! A memory that fits is one span holding the whole of it, untrimmed, under a
//! header that is its name alone -- so `{header}: {text}` is exactly the
//! passage it is embedded from today, and a corpus whose memories all fit
//! indexes as it did before spans. Every other span is trimmed of whitespace
//! at both ends, and its header carries what the span lost by being cut out:
//! the heading it sits under and the speaker of the turn it continues.
//!
//! A pure function of the content, the token counter and the cap, so spans
//! need not be stored anywhere they cannot be computed again.

use std::ops::Range;

use icu_segmenter::SentenceSegmenter;
use icu_segmenter::options::SentenceBreakInvariantOptions;

/// The most tokens a span's header takes.
const HEADER_TOKENS: usize = 32;

/// Counts tokens the way the model that will read a span counts them.
pub trait TokenCount {
    /// The byte range in `text` of each token it encodes to, in order, with
    /// no special tokens and no truncation. Ranges start and end on character
    /// boundaries.
    fn offsets(&self, text: &str) -> Vec<Range<usize>>;

    /// How many tokens `text` encodes to.
    fn count(&self, text: &str) -> usize {
        self.offsets(text).len()
    }
}

/// One span of a memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Span {
    /// Its place among the memory's spans, from 0.
    pub ordinal: u32,
    /// Where it lies in the memory's content, in bytes.
    pub range: Range<usize>,
    /// The memory's name, and the heading and turn it continues.
    pub header: String,
    /// The content in `range`.
    pub text: String,
    /// Its text's tokens, as counted when it was packed. The header's are not
    /// included.
    pub tokens: usize,
}

/// `content`, named `name`, cut into spans of at most `cap` tokens.
///
/// A span holds more than `cap` only when a single token is counted as more
/// than that on its own, which no cap the index uses comes near.
pub fn pack(name: &str, content: &str, counter: &dyn TokenCount, cap: usize) -> Vec<Span> {
    let whole = |tokens| {
        vec![Span {
            ordinal: 0,
            range: 0..content.len(),
            header: name.to_string(),
            text: content.to_string(),
            tokens,
        }]
    };
    let total = counter.count(content);
    if total <= cap {
        return whole(total);
    }
    let blocks = blocks(content);
    let units = units(content, &blocks, counter, cap);
    if units.is_empty() {
        return whole(total);
    }

    let mut packed: Vec<Packed> = Vec::new();
    let mut current: Option<Packed> = None;
    let mut pending = units.into_iter().rev().collect::<Vec<_>>();
    while let Some(unit) = pending.pop() {
        let Some(span) = current.as_mut() else {
            current = Some(Packed::from(unit));
            continue;
        };
        let fits = span.tokens + unit.tokens <= cap;
        if fits && !(unit.opens && !span.headings_only) {
            span.push(&unit);
            continue;
        }
        // A span holding only headings would end with one. The unit that does
        // not fit after them is cut so that its opening does.
        if span.headings_only && !unit.heading {
            let room = cap.saturating_sub(span.tokens);
            let head = fit(&content[unit.range.clone()], room, counter);
            if !head.is_empty() {
                let end = unit.range.start + head.len();
                span.push(&Unit {
                    range: unit.range.start..end,
                    tokens: counter.count(head),
                    ..unit.clone()
                });
                let rest = trim(content, end..unit.range.end);
                if !rest.is_empty() {
                    let tokens = counter.count(&content[rest.clone()]);
                    let pieces = if tokens <= cap {
                        vec![(rest, tokens)]
                    } else {
                        cut(content, rest, counter, cap)
                    };
                    for (range, tokens) in pieces.into_iter().rev() {
                        pending.push(Unit {
                            range,
                            tokens,
                            opens: false,
                            ..unit.clone()
                        });
                    }
                }
                packed.extend(current.take());
                continue;
            }
        }
        packed.extend(current.replace(Packed::from(unit)));
    }
    packed.extend(current);

    packed
        .into_iter()
        .enumerate()
        .map(|(ordinal, span)| {
            let range = span.start..span.end;
            let heading = span
                .heading_block
                .filter(|&block| blocks[block].range.start < range.start)
                .and_then(|block| heading(&content[blocks[block].range.clone()]))
                .filter(|heading| !heading.is_empty());
            let turn = span
                .turn
                .filter(|&block| blocks[block].range.start < range.start)
                .and_then(|block| turn(&content[blocks[block].range.clone()]));
            Span {
                ordinal: ordinal as u32,
                header: header(name, heading, turn.as_deref(), counter),
                text: content[range.clone()].to_string(),
                range,
                tokens: span.tokens,
            }
        })
        .collect()
}

/// What a block is, which decides where it may be cut and what a span cut out
/// of it is headed with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Fence,
    Heading,
    Turn,
    Paragraph,
}

#[derive(Debug)]
struct Block {
    kind: Kind,
    /// Its lines, from the first's first character that is not whitespace to
    /// the last non-blank line's last that is not.
    range: Range<usize>,
}

/// A piece of a block small enough to pack.
#[derive(Clone, Debug)]
struct Unit {
    range: Range<usize>,
    tokens: usize,
    /// The first piece of a heading, which opens a span.
    opens: bool,
    /// Any piece of a heading.
    heading: bool,
    /// The heading block this piece sits under, which may be its own.
    heading_block: Option<usize>,
    /// The turn block this piece belongs to.
    turn: Option<usize>,
}

/// A span while it is being packed.
struct Packed {
    start: usize,
    end: usize,
    tokens: usize,
    headings_only: bool,
    /// What its header is made from: its first piece's heading and turn.
    heading_block: Option<usize>,
    turn: Option<usize>,
}

impl From<Unit> for Packed {
    fn from(unit: Unit) -> Self {
        Self {
            start: unit.range.start,
            end: unit.range.end,
            tokens: unit.tokens,
            headings_only: unit.heading,
            heading_block: unit.heading_block,
            turn: unit.turn,
        }
    }
}

impl Packed {
    fn push(&mut self, unit: &Unit) {
        self.end = unit.range.end;
        self.tokens += unit.tokens;
        self.headings_only &= unit.heading;
    }
}

/// The blocks of `content`, in order. Every byte not in one is whitespace.
fn blocks(content: &str) -> Vec<Block> {
    let mut lines = Vec::new();
    let mut start = 0;
    for (at, _) in content.match_indices('\n') {
        lines.push(start..at);
        start = at + 1;
    }
    lines.push(start..content.len());
    let line = |index: usize| &content[lines[index].clone()];
    let opens =
        |text: &str| fence(text).is_some() || heading(text).is_some() || turn(text).is_some();

    let mut blocks = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let text = line(index);
        if blank(text) {
            index += 1;
            continue;
        }
        let first = index;
        index += 1;
        let kind = if let Some(opening) = fence(text) {
            while index < lines.len() {
                index += 1;
                if closes(line(index - 1), opening) {
                    break;
                }
            }
            Kind::Fence
        } else if heading(text).is_some() {
            Kind::Heading
        } else if turn(text).is_some() {
            while index < lines.len() && !opens(line(index)) {
                index += 1;
            }
            Kind::Turn
        } else {
            while index < lines.len() && !blank(line(index)) && !opens(line(index)) {
                index += 1;
            }
            Kind::Paragraph
        };
        let last = (first..index)
            .rev()
            .find(|&at| !blank(line(at)))
            .unwrap_or(first);
        blocks.push(Block {
            kind,
            range: trim(content, lines[first].start..lines[last].end),
        });
    }
    blocks
}

/// The blocks cut into pieces of at most `cap` tokens each.
fn units(content: &str, blocks: &[Block], counter: &dyn TokenCount, cap: usize) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut heading_block = None;
    for (index, block) in blocks.iter().enumerate() {
        let is_heading = block.kind == Kind::Heading;
        if is_heading {
            heading_block = Some(index);
        }
        let unit = |range: Range<usize>, tokens: usize, first: bool| Unit {
            range,
            tokens,
            opens: is_heading && first,
            heading: is_heading,
            heading_block,
            turn: (block.kind == Kind::Turn).then_some(index),
        };
        let tokens = counter.count(&content[block.range.clone()]);
        if tokens <= cap {
            units.push(unit(block.range.clone(), tokens, true));
            continue;
        }
        let pieces = match block.kind {
            Kind::Fence => lines(content, block.range.clone()),
            _ => sentences(content, block.range.clone()),
        };
        for piece in pieces {
            let tokens = counter.count(&content[piece.clone()]);
            let pieces = if tokens <= cap {
                vec![(piece, tokens)]
            } else {
                cut(content, piece, counter, cap)
            };
            for (range, tokens) in pieces {
                let first = units
                    .last()
                    .is_none_or(|last: &Unit| last.range.end <= block.range.start);
                units.push(unit(range, tokens, first));
            }
        }
    }
    units
}

/// The non-blank lines of `range`, trimmed.
fn lines(content: &str, range: Range<usize>) -> Vec<Range<usize>> {
    let mut lines = Vec::new();
    let mut start = range.start;
    for (at, _) in content[range.clone()].match_indices('\n') {
        lines.push(trim(content, start..range.start + at));
        start = range.start + at + 1;
    }
    lines.push(trim(content, start..range.end));
    lines.retain(|line| !line.is_empty());
    lines
}

/// The sentences of `range`, trimmed, by Unicode's sentence boundaries --
/// which also break after every line ending, and after `。！？` in text
/// written without spaces.
pub(crate) fn sentences(content: &str, range: Range<usize>) -> Vec<Range<usize>> {
    let text = &content[range.clone()];
    let boundaries: Vec<usize> = SentenceSegmenter::new(SentenceBreakInvariantOptions::default())
        .segment_str(text)
        .collect();
    boundaries
        .windows(2)
        .map(|pair| trim(content, range.start + pair[0]..range.start + pair[1]))
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

/// `range`, of more than `cap` tokens, cut at token boundaries into the fewest
/// pieces of near equal size that each fit -- `ceil(n / cap)` of them unless
/// counting a piece on its own finds more tokens than it held in place -- so
/// that no piece is a sliver.
pub(crate) fn cut(
    content: &str,
    range: Range<usize>,
    counter: &dyn TokenCount,
    cap: usize,
) -> Vec<(Range<usize>, usize)> {
    let text = &content[range.clone()];
    let offsets = counter.offsets(text);
    let tokens = offsets.len();
    let mut count = tokens.div_ceil(cap.max(1)).max(1);
    loop {
        let mut bounds = vec![0];
        for piece in 1..count {
            let at = offsets[piece * tokens / count]
                .start
                .max(*bounds.last().unwrap_or(&0));
            bounds.push(at);
        }
        bounds.push(text.len());
        let pieces: Vec<(Range<usize>, usize)> = bounds
            .windows(2)
            .map(|pair| trim(content, range.start + pair[0]..range.start + pair[1]))
            .filter(|piece| !piece.is_empty())
            .map(|piece| {
                let tokens = counter.count(&content[piece.clone()]);
                (piece, tokens)
            })
            .collect();
        if count >= tokens || pieces.iter().all(|(_, tokens)| *tokens <= cap) {
            return pieces;
        }
        count += 1;
    }
}

/// The longest opening of `text` ending at a token boundary that counts at
/// most `budget` tokens on its own, trimmed.
fn fit<'a>(text: &'a str, budget: usize, counter: &dyn TokenCount) -> &'a str {
    let offsets = counter.offsets(text);
    if offsets.len() <= budget {
        return text;
    }
    let mut end = budget;
    while end > 0 {
        let head = text[..offsets[end].start].trim_end();
        if counter.count(head) <= budget {
            return head;
        }
        end -= 1;
    }
    ""
}

/// A span's header: `name`, then the heading it sits under, then the date and
/// speaker of the turn it continues, in at most [`HEADER_TOKENS`] tokens. The
/// heading is shortened first; a header still too long is cut whole.
fn header(
    name: &str,
    heading: Option<&str>,
    turn: Option<&str>,
    counter: &dyn TokenCount,
) -> String {
    let compose = |heading: Option<&str>| {
        let mut header = name.to_string();
        for part in [heading, turn].into_iter().flatten() {
            header.push_str(" > ");
            header.push_str(part);
        }
        header
    };
    let mut header = compose(None);
    if let Some(heading) = heading {
        let mut room = HEADER_TOKENS.saturating_sub(counter.count(&header));
        while room > 0 {
            let candidate = compose(Some(fit(heading, room, counter)).filter(|h| !h.is_empty()));
            if counter.count(&candidate) <= HEADER_TOKENS {
                header = candidate;
                break;
            }
            room -= 1;
        }
    }
    fit(&header, HEADER_TOKENS, counter).to_string()
}

/// `range` without the whitespace at either end.
pub(crate) fn trim(content: &str, range: Range<usize>) -> Range<usize> {
    let text = &content[range.clone()];
    let start = range.start + (text.len() - text.trim_start().len());
    let end = (range.start + text.trim_end().len()).max(start);
    start..end
}

fn blank(line: &str) -> bool {
    line.trim().is_empty()
}

/// Up to three spaces of indentation, as Markdown allows before a block.
pub(crate) fn indented(line: &str) -> Option<&str> {
    let rest = line.trim_start_matches(' ');
    (line.len() - rest.len() <= 3).then_some(rest)
}

/// The character and length of the fence `line` opens, if it opens one.
pub(crate) fn fence(line: &str) -> Option<(char, usize)> {
    let rest = indented(line)?;
    let marker = rest.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let length = rest.len() - rest.trim_start_matches(marker).len();
    (length >= 3).then_some((marker, length))
}

/// Whether `line` closes a fence opened with `opening`.
pub(crate) fn closes(line: &str, (marker, length): (char, usize)) -> bool {
    indented(line).is_some_and(|rest| {
        let after = rest.trim_start_matches(marker);
        rest.len() - after.len() >= length && blank(after)
    })
}

/// The text of the ATX heading `line` is, if it is one. Setext headings are
/// not recognized.
pub(crate) fn heading(line: &str) -> Option<&str> {
    let rest = indented(line)?;
    let text = rest.trim_start_matches('#');
    let level = rest.len() - text.len();
    if !(1..=6).contains(&level) || !(text.is_empty() || text.starts_with(char::is_whitespace)) {
        return None;
    }
    // An optional closing run of `#`, when a space separates it.
    let text = text.trim();
    let open = text.trim_end_matches('#');
    Some(if open.is_empty() || open.ends_with(char::is_whitespace) {
        open.trim_end()
    } else {
        text
    })
}

/// The date and speaker of the turn `line` opens, if it opens one: `[date]
/// speaker: ` or a bare `user:`, `assistant:` or `system:`.
fn turn(line: &str) -> Option<String> {
    let line = line.lines().next().unwrap_or(line);
    if let Some(rest) = line.strip_prefix('[') {
        let (date, rest) = rest.split_once(']')?;
        let (speaker, after) = rest.split_once(':')?;
        if !(1..=64).contains(&date.chars().count())
            || speaker.is_empty()
            || speaker.trim_start().chars().count() > 64
            || !(after.is_empty() || after.starts_with(char::is_whitespace))
        {
            return None;
        }
        let speaker = speaker.trim();
        return Some(if speaker.is_empty() {
            date.to_string()
        } else {
            format!("{date}, {speaker}")
        });
    }
    ["user:", "assistant:", "system:"]
        .into_iter()
        .find_map(|role| {
            line.get(..role.len())
                .filter(|prefix| prefix.eq_ignore_ascii_case(role))
                .map(|prefix| prefix[..prefix.len() - 1].to_string())
        })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::Passage;

    /// One token per run of ASCII that is not whitespace, and one per other
    /// character that is not: words in English, characters in Chinese.
    pub(crate) struct Words;

    impl TokenCount for Words {
        fn offsets(&self, text: &str) -> Vec<Range<usize>> {
            let mut tokens = Vec::new();
            let mut word = None;
            for (at, c) in text.char_indices() {
                if c.is_ascii() && !c.is_whitespace() {
                    word.get_or_insert(at);
                    continue;
                }
                if let Some(start) = word.take() {
                    tokens.push(start..at);
                }
                if !c.is_whitespace() {
                    tokens.push(at..at + c.len_utf8());
                }
            }
            tokens.extend(word.map(|start| start..text.len()));
            tokens
        }
    }

    /// What every packing must be: spans in order, holding what their ranges
    /// say, with nothing but whitespace outside them, and none over the cap.
    fn check(content: &str, spans: &[Span], cap: usize) {
        assert!(!spans.is_empty(), "no spans for {content:?}");
        let mut end = 0;
        for (ordinal, span) in spans.iter().enumerate() {
            assert_eq!(span.ordinal as usize, ordinal);
            assert!(span.range.start >= end, "spans overlap in {content:?}");
            assert!(
                blank(&content[end..span.range.start]),
                "{:?} is in no span of {content:?}",
                &content[end..span.range.start]
            );
            assert_eq!(span.text, content[span.range.clone()]);
            assert_eq!(span.tokens, Words.count(&span.text));
            if spans.len() > 1 {
                assert!(
                    span.tokens <= cap,
                    "{} tokens in {:?}",
                    span.tokens,
                    span.text
                );
                assert_eq!(span.text.trim(), span.text, "an untrimmed span");
            }
            end = span.range.end;
        }
        assert!(
            blank(&content[end..]),
            "the end of {content:?} is in no span"
        );
    }

    fn texts(spans: &[Span]) -> Vec<&str> {
        spans.iter().map(|span| span.text.as_str()).collect()
    }

    #[test]
    fn spans_partition_the_content_in_order() {
        let content = "  Intro words before anything else.\n\n# Setup\n\nInstall the tool first. \
            Then run it once.\n\n```sh\ncargo build\ncargo test\n```\n\n\
            [2024-03-01] Ann: we shipped it.\nIt went fine.\n\nuser: and then?\n\n";
        for cap in 1..40 {
            check(content, &pack("notes", content, &Words, cap), cap);
        }
        let spans = pack("notes", content, &Words, 8);
        assert!(spans.len() > 1, "the premise: the content is over the cap");
        assert_eq!(spans[0].text, "Intro words before anything else.");
    }

    /// A memory that fits is one span of all of it, untrimmed, headed by its
    /// name alone -- so what is embedded for it is what was embedded before
    /// spans existed.
    #[test]
    fn a_memory_that_fits_is_one_span_headed_by_its_name() {
        let content = "\n  # Title\n\nShort text, ```not a fence```\n[2024] Ann: hi\n\n";
        let tokens = Words.count(content);
        for cap in [tokens, tokens + 1, 512] {
            let spans = pack("deploy notes", content, &Words, cap);
            assert_eq!(
                spans,
                vec![Span {
                    ordinal: 0,
                    range: 0..content.len(),
                    header: "deploy notes".into(),
                    text: content.into(),
                    tokens,
                }]
            );
            assert_eq!(
                format!("{}: {}", spans[0].header, spans[0].text),
                Passage::Named.render("deploy notes", content)
            );
        }
        assert!(pack("deploy notes", content, &Words, tokens - 1).len() > 1);
        assert_eq!(pack("empty", "", &Words, 8)[0].text, "");
    }

    #[test]
    fn no_span_is_over_the_cap() {
        let content = format!(
            "# One\n\n{}\n\n```\n{}\n```\n\n[d] A: {}\n\n{}",
            "Short sentence here. ".repeat(30),
            "let x = 1;\n".repeat(30),
            "Turn words go on. ".repeat(20),
            "一二三四五六七八九十".repeat(9),
        );
        for cap in [3, 5, 8, 13, 21, 34] {
            let spans = pack("n", &content, &Words, cap);
            check(&content, &spans, cap);
            assert!(spans.iter().all(|span| span.tokens <= cap));
        }
    }

    /// Code too large for a span is cut between its lines, and nowhere else.
    #[test]
    fn code_over_the_cap_is_cut_at_its_lines() {
        let body: String = (0..20)
            .map(|line| format!("    let value_{line} = compute({line}, other);\n"))
            .collect();
        let content = format!("```rust\n{body}```");
        let spans = pack("code", &content, &Words, 12);
        check(&content, &spans, 12);
        assert!(spans.len() > 1, "the premise: the code is over the cap");
        for span in &spans {
            let before = &content[..span.range.start];
            let after = &content[span.range.end..];
            assert!(
                blank(&before[before.rfind('\n').map_or(0, |at| at + 1)..]),
                "a span starts inside a line: {:?}",
                span.text
            );
            assert!(
                after.is_empty() || after.starts_with('\n'),
                "a span ends inside a line: {:?}",
                span.text
            );
        }
    }

    /// Text written without spaces is cut at its sentences first, and a
    /// sentence still too long at its tokens.
    #[test]
    fn cjk_is_cut_at_sentences_then_at_tokens() {
        let content = "今天天气很好。我们去公园散步吧！你觉得怎么样？";
        let spans = pack("日记", content, &Words, 10);
        check(content, &spans, 10);
        assert_eq!(
            texts(&spans),
            vec!["今天天气很好。", "我们去公园散步吧！", "你觉得怎么样？"]
        );

        let content = "一二三四五六七八九十".repeat(3);
        let content = &content[..content.char_indices().nth(25).unwrap().0];
        let spans = pack("日记", content, &Words, 10);
        check(content, &spans, 10);
        assert_eq!(
            spans.iter().map(|span| span.tokens).collect::<Vec<_>>(),
            vec![8, 8, 9]
        );
    }

    /// A line too long for a span is cut into the fewest pieces that fit, of
    /// near equal size, rather than into full pieces and a sliver.
    #[test]
    fn a_line_over_the_cap_is_cut_into_near_equal_pieces() {
        for words in [11, 23, 40, 41, 99] {
            let content = vec!["token"; words].join(" ");
            let spans = pack("line", &content, &Words, 10);
            check(&content, &spans, 10);
            assert_eq!(spans.len(), words.div_ceil(10), "{words} words");
            let sizes: Vec<usize> = spans.iter().map(|span| span.tokens).collect();
            let (least, most) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            assert!(most - least <= 1, "unequal pieces {sizes:?}");
        }
    }

    #[test]
    fn a_heading_never_ends_a_span() {
        let ends_with_heading =
            |span: &Span| heading(span.text.lines().last().unwrap_or("")).is_some();
        let contents = [
            // The heading would fit after the first paragraph.
            "one two three four five six seven eight\n\n## Section\n\nnine ten eleven twelve",
            // Consecutive headings stay with what follows them.
            "one two three four five six seven\n\n# Part\n## Chapter\n\nalpha beta gamma",
            // What follows a heading does not fit beside it whole.
            "lead words here\n\n## Big heading here\n\n\
             a b c d e f g h i j k l m n o p q r s t u v w x y z",
        ];
        for content in contents {
            for cap in 6..30 {
                let spans = pack("n", content, &Words, cap);
                check(content, &spans, cap);
                if let Some(span) = spans.iter().find(|span| ends_with_heading(span)) {
                    panic!("{:?} ends with a heading at cap {cap}", span.text);
                }
            }
        }
        let spans = pack("n", contents[1], &Words, 10);
        assert_eq!(spans[1].text, "# Part\n## Chapter\n\nalpha beta gamma");
    }

    /// A span cut out from under its heading, or out of the middle of a turn,
    /// says which.
    #[test]
    fn a_span_is_headed_with_the_heading_and_turn_it_continues() {
        let content = "# Trip to Paris\n\n[2023-05-08 13:56] Caroline: We left early. \
            The train was late. We ate lunch in the station.\n\n\
            user: what happened next?\n\nassistant: The museum was closed. \
            We walked by the river instead.";
        let spans = pack("travel", content, &Words, 9);
        check(content, &spans, 9);
        let headers: Vec<&str> = spans.iter().map(|span| span.header.as_str()).collect();
        assert_eq!(
            headers,
            vec![
                "travel",
                "travel > Trip to Paris > 2023-05-08 13:56, Caroline",
                "travel > Trip to Paris > 2023-05-08 13:56, Caroline",
                "travel > Trip to Paris",
                "travel > Trip to Paris > assistant",
            ],
            "{:#?}",
            texts(&spans)
        );
    }

    #[test]
    fn a_header_takes_at_most_32_tokens() {
        let title = (0..60)
            .map(|word| format!("w{word}"))
            .collect::<Vec<_>>()
            .join(" ");
        let content = format!("# {title}\n\n{}", "Body text goes here. ".repeat(40));
        let spans = pack("topic name", &content, &Words, 80);
        check(&content, &spans, 80);
        let header = &spans[1].header;
        assert!(header.starts_with("topic name > w0 w1 "), "{header}");
        assert_eq!(Words.count(header), HEADER_TOKENS);
    }

    /// The same memory always packs the same way, and every packing holds
    /// the partition and the cap, over a thousand memories assembled at
    /// random from every kind of block.
    #[test]
    fn packing_is_deterministic() {
        const PARTS: [&str; 14] = [
            "The deploy failed at noon. We rolled back within the hour.",
            "今天天气很好。我们去公园散步吧！",
            "一二三四五六七八九十一二三四五六七八九十",
            "## Heading two",
            "# Top",
            "```\nfn main() {}\n    let x = 1;\n```",
            "~~~\nunclosed fence line",
            "[2024-01-02] Ann: hello there friend. How are you?",
            "user: what now?",
            "assistant: try again later please.",
            "",
            "   ",
            "word word word word word word word word word word word word word",
            "  indented paragraph\nwith a second line",
        ];
        let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |bound: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % bound as u64) as usize
        };
        for _ in 0..1_000 {
            let mut content = String::new();
            for _ in 0..next(12) {
                content.push_str(PARTS[next(PARTS.len())]);
                content.push_str(["\n", "\n\n", " "][next(3)]);
            }
            let cap = 3 + next(40);
            let spans = pack("memory", &content, &Words, cap);
            check(&content, &spans, cap);
            assert_eq!(spans, pack("memory", &content, &Words, cap));
        }
    }
}
