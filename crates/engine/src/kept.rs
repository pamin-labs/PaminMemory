//! Span vectors kept in PostgreSQL (`span_vectors`), so a topic the index
//! lacks is written from a vector already embedded from its text rather than
//! through the model again.
//!
//! Kept vectors only save passes: a lookup or a store that fails is logged and
//! the work goes on as if nothing were kept.

use std::collections::{BTreeMap, HashMap};

use memory_core::TopicId;
use memory_index::{SpanDocument, SpanSource};
use memory_store::repository;

use crate::engine::{Engine, off_the_runtime};

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

/// Topics written from kept vectors, each with its spans' fp16 bytes in
/// order.
pub(crate) type Supplied<'k, T> = Vec<(T, Vec<&'k [u8]>)>;

/// Vectors one model revision embedded, by plan, then by passage hash.
///
/// Held as the fp16 bytes they are kept as, half what they widen to, and
/// widened a batch at a time as they are written ([`widen`]).
#[derive(Default)]
pub(crate) struct Kept {
    pub(crate) revision: String,
    pub(crate) by_plan: BTreeMap<String, HashMap<[u8; 32], Vec<u8>>>,
}

impl Kept {
    /// Splits `topics`, each given by its spans' passages in order, into those
    /// every span of which has a vector kept under `plan` of the model at
    /// `revision`, with those vectors, and those to embed. The one test of
    /// whether a topic is written from what was kept.
    ///
    /// A topic with no spans is in neither: there is nothing to write for it,
    /// so it is neither reused nor embedded.
    pub(crate) fn split<'t, T>(
        &self,
        revision: &str,
        plan: Option<&str>,
        topics: impl IntoIterator<Item = (T, &'t [Passage<'t>])>,
    ) -> (Supplied<'_, T>, Vec<T>) {
        let kept = plan
            .filter(|_| revision == self.revision)
            .and_then(|plan| self.by_plan.get(plan));
        let (mut supplied, mut to_embed) = (Vec::new(), Vec::new());
        for (topic, passages) in topics {
            if passages.is_empty() {
                continue;
            }
            match kept.filter(|kept| covers(kept, passages)) {
                Some(kept) => supplied.push((
                    topic,
                    passages
                        .iter()
                        .map(|passage| kept[&passage.digest].as_slice())
                        .collect(),
                )),
                None => to_embed.push(topic),
            }
        }
        (supplied, to_embed)
    }

    /// The plan whose kept vectors cover the most of `topics` whole, ties to
    /// the first by name, so a choice made with no model loaded is the same
    /// every time.
    pub(crate) fn widest<'t>(
        &self,
        topics: impl Iterator<Item = &'t [Passage<'t>]> + Clone,
    ) -> Option<&str> {
        // Reversed because `max_by_key` keeps the last of equals.
        self.by_plan
            .iter()
            .rev()
            .max_by_key(|(_, kept)| topics.clone().filter(|spans| covers(kept, spans)).count())
            .map(|(plan, _)| plan.as_str())
    }
}

/// Whether every one of a topic's spans, of which it has at least one, has a
/// vector in `kept`.
fn covers(kept: &HashMap<[u8; 32], Vec<u8>>, passages: &[Passage<'_>]) -> bool {
    !passages.is_empty()
        && passages
            .iter()
            .all(|passage| kept.contains_key(&passage.digest))
}

/// The hashes of every span of `topics`, to look up what is kept for them.
pub(crate) fn digests<'t>(topics: impl IntoIterator<Item = &'t [Passage<'t>]>) -> Vec<[u8; 32]> {
    topics
        .into_iter()
        .flatten()
        .map(|passage| passage.digest)
        .collect()
}

/// Supplied vectors widened back to what the index stores, in order.
pub(crate) fn widen<T>(supplied: &[(T, Vec<&[u8]>)]) -> Vec<Vec<f32>> {
    supplied
        .iter()
        .flat_map(|(_, vectors)| vectors)
        .map(|bytes| {
            memory_index::vector_from_bytes(bytes)
                .expect("a kept vector's width is checked when it is read")
        })
        .collect()
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
        .map(|((topic, ordinal, span), vector)| span.document(topic, ordinal, vector))
        .collect()
}

impl Engine {
    /// The vectors the model at `revision` embedded for the passages hashed
    /// `hashes` ([`Passage::digest`]), under `plan`, or under every plan when
    /// it is `None`. One lookup, in its own transaction, never under the
    /// model or index lock.
    pub(crate) async fn kept_vectors(
        &self,
        revision: String,
        plan: Option<&str>,
        hashes: &[[u8; 32]],
    ) -> Kept {
        let mut kept = Kept {
            revision,
            by_plan: BTreeMap::new(),
        };
        if hashes.is_empty() {
            return kept;
        }
        let rows = self
            .scoped(async |connection| {
                Ok(
                    repository::span_vectors(
                        connection,
                        self.project,
                        &kept.revision,
                        plan,
                        hashes,
                    )
                    .await?,
                )
            })
            .await;
        let rows = match rows {
            Ok(rows) => rows,
            Err(error) => {
                tracing::warn!(%error, "could not read kept span vectors; embedding instead");
                return kept;
            }
        };
        for row in rows {
            if row.vector.len() != memory_index::VECTOR_BYTES {
                tracing::warn!(
                    encoding = row.encoding,
                    bytes = row.vector.len(),
                    "skipping a kept span vector of another width"
                );
                continue;
            }
            kept.by_plan
                .entry(row.encoding)
                .or_default()
                .insert(row.passage_hash, row.vector);
        }
        kept
    }

    /// What a rebuild that has not loaded the model can stand in for `topics`
    /// with: every plan the next load's revision kept for them, and the plan
    /// that covers the most ([`Kept::widest`]). Nothing while no revision is
    /// pinned to match.
    ///
    /// ponytail: span_vectors is per project, not per node, so the widest
    /// plan can be another node's; one image on one instance type means one
    /// plan in practice. Checking it would mean loading the model.
    ///
    /// ponytail: with no model cached there is no revision to look up by, so
    /// a rebuild embeds everything even when the database kept it all. Hosted
    /// images bake the models in; ask the loaded model's revision instead if a
    /// node ever starts without them.
    pub(crate) async fn kept_for_rebuild(
        &self,
        topics: &[&[Passage<'_>]],
    ) -> (Kept, Option<String>) {
        let Some(revision) = off_the_runtime(|| memory_index::cached_revision(&self.models.dir))
        else {
            return (Kept::default(), None);
        };
        // ponytail: holds every kept vector of these topics at once, so a
        // rebuild's peak follows the project here; page it by topic if that
        // matters.
        let kept = self
            .kept_vectors(revision, None, &digests(topics.iter().copied()))
            .await;
        let plan = kept.widest(topics.iter().copied()).map(str::to_string);
        (kept, plan)
    }

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
                        connection,
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

#[cfg(test)]
mod tests {
    use super::{Kept, Passage};

    /// A topic is supplied only when every span has a vector under the plan
    /// and revision asked for, in order; a topic with no spans is neither
    /// supplied nor embedded; and the widest plan is chosen, ties to the
    /// first by name.
    #[test]
    fn a_topic_is_kept_whole_under_one_plan_or_embedded() {
        let [x, y, z] = ["x", "y", "z"].map(Passage::of);
        let kept = Kept {
            revision: "model@1".into(),
            by_plan: [
                (
                    "a".into(),
                    [(x.digest, vec![1]), (y.digest, vec![2])].into(),
                ),
                ("b".into(), [(x.digest, vec![3])].into()),
                ("c".into(), [(y.digest, vec![4])].into()),
            ]
            .into(),
        };
        let topics: [(u8, &[Passage<'_>]); 4] = [(0, &[y, x]), (1, &[x]), (2, &[z]), (3, &[])];
        let (supplied, to_embed) = kept.split("model@1", Some("a"), topics);
        assert_eq!(
            supplied,
            [(0, vec![&[2u8][..], &[1][..]]), (1, vec![&[1][..]])]
        );
        assert_eq!(to_embed, [2], "an empty topic is in neither");
        assert_eq!(kept.split("model@1", Some("b"), topics).1, [0, 2]);
        assert_eq!(
            kept.split("model@1", None, topics),
            (Vec::new(), vec![0, 1, 2])
        );
        assert_eq!(
            kept.split("model@2", Some("a"), topics),
            (Vec::new(), vec![0, 1, 2]),
            "another revision's vectors are not this model's"
        );
        let spans = topics.map(|(_, spans)| spans);
        assert_eq!(kept.widest(spans.into_iter()), Some("a"));
        assert_eq!(
            kept.widest(spans[1..].iter().copied()),
            Some("a"),
            "a and b both cover one; a is first by name"
        );
    }
}
