//! Span vectors kept in PostgreSQL (`span_vectors`), so a topic the index
//! lacks is written from a vector already embedded from its text rather than
//! through the model again.
//!
//! Kept vectors only save passes: a lookup or a store that fails is logged and
//! the work goes on as if nothing were kept.

use std::collections::BTreeMap;

use memory_core::TopicId;
use memory_index::{SpanDocument, SpanSource};
use memory_store::repository;

use crate::engine::Engine;

/// What a span's vector is embedded from, with the hash it is kept under,
/// taken once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Passage<'a> {
    pub(crate) text: &'a str,
    /// The SHA-256 of `text`, so the same text finds its vector whichever
    /// topic holds it.
    pub(crate) digest: [u8; 32],
}

impl<'a> Passage<'a> {
    pub(crate) fn of(text: &'a str) -> Self {
        use sha2::Digest as _;
        Self {
            text,
            digest: sha2::Sha256::digest(text.as_bytes()).into(),
        }
    }
}

/// What one pass embedded, as `span_vectors` keeps it: the model revision and
/// plan that produced it, and each passage's hash with its vector.
pub(crate) struct Embedded {
    pub(crate) revision: String,
    pub(crate) encoding: String,
    pub(crate) vectors: Vec<HashedVector>,
}

/// A passage's hash and its vector's fp16 bytes, as `span_vectors` keeps them.
pub(crate) type HashedVector = ([u8; 32], Vec<u8>);

impl Embedded {
    /// `None` when no plan can be shown to have produced the vectors
    /// ([`memory_index::Embedder::encoding`]): nothing could ever match them.
    pub(crate) fn of<'p>(
        revision: String,
        encoding: Option<&str>,
        passages: impl IntoIterator<Item = &'p Passage<'p>>,
        vectors: &[Vec<f32>],
    ) -> Option<Self> {
        Some(Self {
            revision,
            encoding: encoding?.to_string(),
            vectors: passages
                .into_iter()
                .zip(vectors)
                .map(|(passage, vector)| (passage.digest, memory_index::vector_to_bytes(vector)))
                .collect(),
        })
    }
}

/// Every span of `topics`, in order, as the index is written it, with
/// `vectors`.
pub(crate) fn span_documents<'a>(
    topics: impl IntoIterator<Item = (TopicId, &'a [SpanSource<'a>])>,
    vectors: impl IntoIterator<Item = &'a [f32]>,
) -> Vec<SpanDocument<'a>> {
    topics
        .into_iter()
        .flat_map(|(topic, spans)| {
            spans
                .iter()
                .enumerate()
                .map(move |(ordinal, span)| (topic, ordinal as u32, *span))
        })
        .zip(vectors)
        .map(|((topic, ordinal, span), vector)| span_document(topic, ordinal, span, vector))
        .collect()
}

/// Span `ordinal` of `topic` as the index is written it, with `vector`.
pub(crate) fn span_document<'a>(
    topic: TopicId,
    ordinal: u32,
    span: SpanSource<'a>,
    vector: &'a [f32],
) -> SpanDocument<'a> {
    (topic, ordinal, span.text, Some(span.header), vector)
}

impl Engine {
    /// Keeps what passes embedded, so a rebuild or another node's catch-up
    /// writes it back instead of embedding it again. One insert per model
    /// revision and plan, in its own transaction after the passes: one held
    /// across them would hold a connection for as long as the model runs.
    /// What is written to the index stands without it, so a failure is
    /// logged and costs only a later pass.
    pub(crate) async fn keep_vectors(&self, embedded: &[Embedded]) {
        if embedded.is_empty() {
            return;
        }
        let mut by_plan: BTreeMap<(&str, &str), Vec<&HashedVector>> = BTreeMap::new();
        for pass in embedded {
            by_plan
                .entry((&pass.revision, &pass.encoding))
                .or_default()
                .extend(&pass.vectors);
        }
        let kept = self
            .scoped(async |connection| {
                for ((revision, encoding), vectors) in &by_plan {
                    repository::put_span_vectors(
                        &mut *connection,
                        self.project,
                        revision,
                        encoding,
                        vectors.iter().copied(),
                    )
                    .await?;
                }
                Ok(())
            })
            .await;
        if let Err(error) = kept {
            tracing::warn!(%error, "could not keep span vectors");
        }
    }
}
