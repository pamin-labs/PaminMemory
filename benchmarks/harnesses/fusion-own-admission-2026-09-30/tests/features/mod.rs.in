//! Every candidate fusion saw, one row each, for fitting a fusion offline.
//!
//! The `FEATURES` arm of each retrieval harness. It exists because the
//! question the channel diagnostic keeps arriving at -- one global weight
//! cannot tell a cross-lingual query from a same-language one -- is a question
//! about *learning* a fusion, and a learned fusion has to be fitted and scored
//! somewhere nothing can leak from the test set into the fit. That is not a
//! Rust test binary. So this writes the whole candidate matrix out, once per
//! corpus, and the fitting happens elsewhere against the file.
//!
//! One row per candidate per group: which question, whether it is relevant,
//! where every channel ranked it and what that channel scored it, where the
//! shipped fusion put it, and a handful of features anything at query time
//! could compute without a model -- how long the query and the candidate are,
//! and which script each is written in. The script is here rather than a
//! language because a detector will not commit to a language for a short query
//! (see `Engine::search_reranked`), and a script it can read off the code
//! points. The harness's own language labels are written beside it when the
//! harness has them, for analysis, and are not something the product knows.
//!
//! Set `FEATURES_OUT` to a path to run the arm.
//!
//! ## What each row asserts before it is written
//!
//! A dump that cannot rebuild what the engine ranked is a dump of something
//! else, and everything fitted to it would be fitted to that. So, per
//! question:
//!
//! - the trace was not truncated ([`channels::enough_room`]), so every
//!   channel's whole list is in it;
//! - the trace rebuilds into the engine's own order
//!   ([`channels::same_as_the_engine`]);
//! - every score written parses back to the bit-identical `f32` the channel
//!   produced, so the text in the file carries the same numbers the trace did.
//!
//! Together those make the file enough to rebuild the shipped ranking, and
//! whatever reads it should check that it does: the `fused_rank` column is the
//! answer to compare against. And at the end, [`Features::finish`] asserts that
//! every question the harness meant to ask produced rows.

// Four test binaries include this module; not all of them use every helper.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::PathBuf;

use pamin_core::{Channel, Fusion, Why};
use pamin_engine::SearchHit;

use crate::channels;

/// The channels, in the order their columns are written.
const CHANNELS: [Channel; 4] = [
    Channel::LexicalSegmented,
    Channel::LexicalNgram,
    Channel::Vector,
    Channel::Graph,
];

/// One question as a harness asks it, and how it is judged in one group.
pub struct Asked<'a> {
    /// The group this row is scored in. A corpus that scores one query in two
    /// groups -- XQuAD-R -- observes it twice.
    pub group: &'a str,
    /// Stable across runs and unique within the corpus, so a question can be
    /// paired across arms and every asking of it kept in one fold.
    pub question: &'a str,
    pub text: &'a str,
    /// The language the harness knows the query is in, in the same code the
    /// harness wrote its memories with. `None` where it does not know.
    pub language: Option<&'a str>,
    /// How many items are relevant in this group, found or not. The recall
    /// denominator and the ideal gain are built from it, so it cannot be read
    /// off the rows: a relevant item no channel returned has no row.
    pub judged: usize,
}

/// The open dump, and what has been written to it.
pub struct Features {
    corpus: &'static str,
    path: PathBuf,
    out: BufWriter<File>,
    asked: BTreeSet<(String, String)>,
    rows: usize,
}

impl Features {
    /// The dump `FEATURES_OUT` names, or `None` when the arm is not asked for.
    pub fn from_env(corpus: &'static str) -> Option<Self> {
        let path = PathBuf::from(std::env::var_os("FEATURES_OUT")?);
        let file = File::create(&path)
            .unwrap_or_else(|error| panic!("creating {}: {error}", path.display()));
        let mut out = BufWriter::new(file);

        let mut header = String::from(
            "corpus\tgroup\tquestion\tquery_chars\tquery_tokens\tquery_script\tquery_language\t\
             topic\tcandidate_chars\tcandidate_script\tcandidate_language\tsame_script\t\
             same_language\texcluded\trelevant\tjudged\tfused_rank\tfused_score",
        );
        for channel in CHANNELS {
            let name = channel.as_str();
            write!(header, "\t{name}_rank\t{name}_score\t{name}_n").expect("a string");
        }
        header.push_str("\tgraph_hops\n");
        out.write_all(header.as_bytes()).expect("write the header");

        Some(Self {
            corpus,
            path,
            out,
            asked: BTreeSet::new(),
            rows: 0,
        })
    }

    /// One question's untruncated fused trace, as rows in `asked.group`.
    ///
    /// `hits` is `search_fused` at the shipped settings with `limit` room.
    /// `judge` is a candidate's gain in this group -- zero when it is not
    /// relevant -- or `None` for a candidate the group takes out of the
    /// ranking before scoring it. That is XQuAD-R's cross-lingual group, which
    /// removes the answer in the query's own language; the row is written
    /// anyway, marked `excluded`, because it was in the channels' lists and
    /// rebuilding their fusion needs every candidate they returned.
    pub fn observe(
        &mut self,
        asked: &Asked<'_>,
        hits: &[SearchHit],
        limit: u32,
        judge: impl Fn(&str) -> Option<f64>,
    ) {
        assert!(
            !hits.is_empty(),
            "{} {:?} returned nothing, so it has no rows to write",
            self.corpus,
            asked.question
        );
        channels::enough_room(hits, limit);
        channels::same_as_the_engine(hits, &Fusion::default());

        // Per channel, how many candidates it returned. The banded combiner's
        // band is set by it, and a row only knows its own rank.
        let mut returned = [0usize; CHANNELS.len()];
        for hit in hits {
            for why in &hit.result.why {
                if let Why::Channel { channel, .. } = why {
                    returned[column(*channel)] += 1;
                }
            }
        }

        let query_script = script(asked.text);
        let mut rows = String::new();
        for (position, hit) in hits.iter().enumerate() {
            let content = &hit.state.content;
            let candidate_script = script(content);
            let candidate_language = hit.state.language.as_deref();
            let same_language = match (asked.language, candidate_language) {
                (Some(query), Some(candidate)) => u8::from(query == candidate).to_string(),
                _ => String::new(),
            };
            let gain = judge(&hit.topic);

            write!(
                rows,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                self.corpus,
                asked.group,
                asked.question,
                asked.text.chars().count(),
                asked.text.split_whitespace().count(),
                query_script,
                asked.language.unwrap_or_default(),
                clean(&hit.topic),
                content.chars().count(),
                candidate_script,
                candidate_language.unwrap_or_default(),
                u8::from(query_script == candidate_script),
                same_language,
                u8::from(gain.is_none()),
                gain.unwrap_or(0.0),
                asked.judged,
                position + 1,
                exact(hit.result.score),
            )
            .expect("a string");

            let mut placed: [Option<(u32, Option<f32>)>; CHANNELS.len()] = [None; CHANNELS.len()];
            let mut hops = None;
            for why in &hit.result.why {
                match why {
                    Why::Channel {
                        channel,
                        rank,
                        score,
                        ..
                    } => {
                        let slot = &mut placed[column(*channel)];
                        assert!(
                            slot.is_none(),
                            "{:?} lists {} twice in {channel:?}",
                            asked.question,
                            hit.topic
                        );
                        *slot = Some((*rank, *score));
                    }
                    Why::Path { hops: taken, .. } => hops = Some(*taken),
                    Why::Reranked { .. } => {
                        panic!("a fused search wrote a rerank line; this is not fusion alone")
                    }
                }
            }
            for (at, place) in placed.iter().enumerate() {
                match place {
                    Some((rank, score)) => write!(
                        rows,
                        "\t{rank}\t{}\t{}",
                        score.map(exact).unwrap_or_default(),
                        returned[at]
                    ),
                    None => write!(rows, "\t\t\t{}", returned[at]),
                }
                .expect("a string");
            }
            writeln!(
                rows,
                "\t{}",
                hops.map(|hops| hops.to_string()).unwrap_or_default()
            )
            .expect("a string");
        }

        self.out
            .write_all(rows.as_bytes())
            .unwrap_or_else(|error| panic!("writing {}: {error}", self.path.display()));
        self.rows += hits.len();
        assert!(
            self.asked
                .insert((asked.group.to_string(), asked.question.to_string())),
            "{} {:?} was observed twice in {}, so its rows would be counted twice",
            self.corpus,
            asked.question,
            asked.group
        );
    }

    /// Closes the dump, asserting it holds every question the harness asked.
    ///
    /// `expected` counts question-group pairs: XQuAD-R's 1,190 questions in
    /// two groups are 2,380.
    pub fn finish(mut self, expected: usize) {
        self.out
            .flush()
            .unwrap_or_else(|error| panic!("flushing {}: {error}", self.path.display()));
        assert_eq!(
            self.asked.len(),
            expected,
            "{} wrote rows for {} question-group pairs of {expected}",
            self.corpus,
            self.asked.len()
        );
        println!(
            "\n  {}: {} rows for {} question-group pairs in {}\n",
            self.corpus,
            self.rows,
            self.asked.len(),
            self.path.display()
        );
    }
}

fn column(channel: Channel) -> usize {
    CHANNELS
        .iter()
        .position(|it| *it == channel)
        .expect("every channel has a column")
}

/// An `f32` as text that parses back to the same bits, asserted.
///
/// Rust's `Display` for floats is the shortest representation that round-trips,
/// so this should never fire; it is asserted because the whole file is only
/// worth having if it carries the numbers the channels produced, and a rounded
/// score reorders candidates that tie to the fourth decimal.
fn exact(value: f32) -> String {
    let text = value.to_string();
    let back: f32 = text.parse().expect("a float parses");
    assert_eq!(
        back.to_bits(),
        value.to_bits(),
        "{value:?} was written as {text:?}, which reads back as {back:?}"
    );
    text
}

/// A topic name safe to put in a tab-separated field.
///
/// Asserted rather than escaped: no corpus here names a topic with a tab or a
/// newline, and one that did would shift every column after it.
fn clean(topic: &str) -> &str {
    assert!(
        !topic.contains(['\t', '\n', '\r']),
        "{topic:?} cannot be written as one field"
    );
    topic
}

/// The script most of a text's letters are written in.
///
/// Read off code-point blocks, which is all a script is. `none` when the text
/// has no letters. Ties go to the name that sorts first, so the answer is the
/// same on every run.
pub fn script(text: &str) -> &'static str {
    let mut counts: [(&'static str, usize); 11] = [
        ("arabic", 0),
        ("cyrillic", 0),
        ("devanagari", 0),
        ("greek", 0),
        ("han", 0),
        ("hangul", 0),
        ("hebrew", 0),
        ("kana", 0),
        ("latin", 0),
        ("other", 0),
        ("thai", 0),
    ];
    for letter in text.chars().filter(|c| c.is_alphabetic()) {
        let name = match u32::from(letter) {
            0x0041..=0x024F | 0x1E00..=0x1EFF => "latin",
            0x0370..=0x03FF | 0x1F00..=0x1FFF => "greek",
            0x0400..=0x052F => "cyrillic",
            0x0590..=0x05FF => "hebrew",
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => "arabic",
            0x0900..=0x097F => "devanagari",
            0x0E00..=0x0E7F => "thai",
            0x3040..=0x30FF | 0x31F0..=0x31FF => "kana",
            0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => "hangul",
            0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => "han",
            _ => "other",
        };
        if let Some(entry) = counts.iter_mut().find(|(it, _)| *it == name) {
            entry.1 += 1;
        }
    }
    counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .max_by(|left, right| left.1.cmp(&right.1).then(right.0.cmp(left.0)))
        .map_or("none", |(name, _)| name)
}

#[cfg(test)]
mod tests {
    use super::script;

    #[test]
    fn a_script_is_what_most_letters_are_written_in() {
        assert_eq!(script("how long are database backups kept"), "latin");
        assert_eq!(script("сколько хранятся резервные копии"), "cyrillic");
        assert_eq!(script("备份保留多久"), "han");
        assert_eq!(script("バックアップはいつまで"), "kana");
        assert_eq!(script("ข้อมูลสำรอง"), "thai");
        // Digits and punctuation are not letters, so they do not vote.
        assert_eq!(script("2024: 备份 v2"), "han");
        assert_eq!(script("42 -- 17"), "none");
    }
}
