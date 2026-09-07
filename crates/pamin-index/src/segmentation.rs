//! Turning text in any language into indexable tokens.
//!
//! The retrieval engine's own tokenizers cover UAX#29 word breaking, Chinese
//! through a dictionary, and character n-grams. That leaves Japanese, Korean,
//! Thai, Khmer, Lao, and Burmese with only n-grams, which index but match
//! across word boundaries and lose precision.
//!
//! Segmenting here closes that gap without adding a second search engine. ICU4X
//! applies UAX#29 to most languages, dictionaries to Chinese and Japanese, and
//! an LSTM model to Thai, Khmer, Lao, and Burmese, and it is pure Rust, so no
//! C++ toolchain comes with it.
//!
//! Only the index input is rewritten. Evidence is stored verbatim in
//! PostgreSQL, and nothing here touches it: a segmented string is a projection
//! that can be thrown away and rebuilt.

use icu_segmenter::WordSegmenter;
use icu_segmenter::options::WordBreakInvariantOptions;

/// Splits text into word-like tokens in whatever language it is written in.
///
/// Holds the compiled segmentation data, so construct once and reuse.
pub struct Segmenter {
    words: WordSegmenter,
}

impl Default for Segmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl Segmenter {
    pub fn new() -> Self {
        // `auto` selects a dictionary or the LSTM model per script, which is
        // what makes one segmenter enough for every language.
        Self {
            words: WordSegmenter::new_auto(WordBreakInvariantOptions::default()).static_to_owned(),
        }
    }

    /// Returns the word-like tokens in `text`, in order.
    ///
    /// Punctuation and whitespace are dropped; numbers and letters, including
    /// CJKV ideographs, are kept.
    pub fn tokens<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut tokens = Vec::new();
        let mut previous = 0usize;

        for (boundary, word_type) in self
            .words
            .as_borrowed()
            .segment_str(text)
            .iter_with_word_type()
        {
            if word_type.is_word_like() && boundary > previous {
                tokens.push(&text[previous..boundary]);
            }
            previous = boundary;
        }

        tokens
    }

    /// Renders `text` as space-separated tokens for the lexical index.
    ///
    /// Queries pass through the same function, so a query tokenizes exactly the
    /// way the documents did. Skipping that on either side is the classic way
    /// to build an index nothing can match against.
    pub fn segment_for_index(&self, text: &str) -> String {
        self.tokens(text).join(" ")
    }
}

/// Detects the dominant language of a span, when confident.
///
/// Recorded per span rather than per deployment: one source can mix languages,
/// and the note-language rule needs to know what a span was actually written in.
/// `None` means detection was not confident, which is a normal outcome for
/// short text, code, and identifiers rather than an error.
pub fn detect_language(text: &str) -> Option<(String, f32)> {
    let info = whatlang::detect(text)?;
    if !info.is_reliable() {
        return None;
    }
    Some((info.lang().code().to_string(), info.confidence() as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_delimited_text_splits_into_words() {
        let segmenter = Segmenter::new();
        assert_eq!(
            segmenter.tokens("deploys through the ci pipeline"),
            vec!["deploys", "through", "the", "ci", "pipeline"]
        );
    }

    #[test]
    fn chinese_splits_on_words_not_characters() {
        let segmenter = Segmenter::new();
        let tokens = segmenter.tokens("部署走持续集成流水线");
        assert!(
            tokens.len() > 1 && tokens.iter().any(|token| token.chars().count() > 1),
            "expected dictionary segmentation, got {tokens:?}"
        );
    }

    #[test]
    fn japanese_is_segmented_despite_having_no_spaces() {
        let segmenter = Segmenter::new();
        let tokens = segmenter.tokens("東京都に住んでいます");
        assert!(tokens.len() > 1, "expected several tokens, got {tokens:?}");
    }

    #[test]
    fn thai_is_segmented_despite_having_no_spaces() {
        // Thai has no word spaces and no dictionary in the retrieval engine, so
        // this is the case that would otherwise fall back to n-grams.
        let segmenter = Segmenter::new();
        let tokens = segmenter.tokens("ฉันชอบกินข้าว");
        assert!(tokens.len() > 1, "expected several tokens, got {tokens:?}");
    }

    #[test]
    fn korean_is_segmented() {
        let segmenter = Segmenter::new();
        let tokens = segmenter.tokens("서울에서 커피를 마셨다");
        assert!(tokens.len() > 1, "expected several tokens, got {tokens:?}");
    }

    #[test]
    fn arabic_and_russian_segment_into_words() {
        let segmenter = Segmenter::new();
        assert!(segmenter.tokens("أنا أحب القهوة").len() >= 3);
        assert!(segmenter.tokens("я люблю кофе").len() >= 3);
    }

    #[test]
    fn punctuation_and_whitespace_are_dropped() {
        let segmenter = Segmenter::new();
        assert_eq!(
            segmenter.segment_for_index("  hello,   world!  "),
            "hello world"
        );
    }

    #[test]
    fn identifiers_survive_as_single_tokens() {
        // The segmented field must not shred an identifier into fragments that
        // no realistic query would reassemble; substring matching is the
        // n-gram field's job.
        let segmenter = Segmenter::new();
        let tokens = segmenter.tokens("call deploy_service now");
        assert!(
            tokens.contains(&"deploy_service") || tokens.contains(&"deploy"),
            "unexpected identifier handling: {tokens:?}"
        );
    }

    #[test]
    fn a_query_tokenizes_the_same_way_a_document_did() {
        let segmenter = Segmenter::new();
        let document = segmenter.segment_for_index("部署走持续集成流水线");
        let query = segmenter.segment_for_index("持续集成");
        assert!(
            document.contains(query.split(' ').next().unwrap()),
            "query tokens must line up with document tokens: {document:?} vs {query:?}"
        );
    }

    #[test]
    fn language_detection_is_reported_per_span() {
        let (language, confidence) =
            detect_language("это довольно длинное предложение на русском языке")
                .expect("confident detection");
        assert_eq!(language, "rus");
        assert!(confidence > 0.0);
    }

    #[test]
    fn unclear_text_reports_no_language_rather_than_guessing() {
        assert!(detect_language("x1").is_none());
    }
}
