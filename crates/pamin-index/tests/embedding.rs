//! Runs the real embedding model and the vector channel.
//!
//! Ignored by default: the first run downloads model weights. Run with
//! `cargo test -p pamin-index -- --ignored`.

use pamin_core::TopicId;
use pamin_index::{Access, Embedder, Profile, Projection, ProjectionIndex};

fn id(byte: u8) -> TopicId {
    TopicId(uuid::Uuid::from_bytes([byte; 16]))
}

#[test]
#[ignore = "downloads embedding model weights"]
fn the_vector_channel_recalls_across_languages_without_translating() {
    let dir = tempfile::tempdir().expect("temp dir");

    // The smallest profile, because this test is about cross-language recall
    // rather than about which profile is most accurate.
    let profile = Profile::Speed;
    let mut embedder = Embedder::load(profile, &dir.path().join("models")).expect("load model");

    let index = ProjectionIndex::open(
        &dir.path().join("index"),
        &dir.path().join("legacy"),
        profile,
        Access::ReadWrite,
        0,
    )
    .expect("open index");

    let chinese = id(1);
    let unrelated = id(2);

    // Stored verbatim in Chinese. Nothing translates it on the way in; that
    // would put a model on the write path and destroy exact-term matching.
    index
        .upsert(
            chinese,
            "部署流水线运行在持续集成上",
            &embedder
                .embed_passage("部署流水线运行在持续集成上")
                .unwrap(),
        )
        .expect("upsert chinese");
    index
        .upsert(
            unrelated,
            "the office coffee machine needs descaling",
            &embedder
                .embed_passage("the office coffee machine needs descaling")
                .unwrap(),
        )
        .expect("upsert unrelated");
    index.flush().expect("flush");

    // An English query reaches a Chinese memory through the shared multilingual
    // embedding space, with no translation anywhere in the pipeline.
    let query = embedder
        .embed_query("how does the deployment pipeline run")
        .expect("embed query");
    let hits = index.recall_vector(&query, 2).expect("vector recall");

    assert_eq!(
        hits.first().map(|hit| hit.topic),
        Some(chinese),
        "an english query should reach the chinese memory first: {hits:?}"
    );
}

#[test]
#[ignore = "downloads embedding model weights"]
fn embeddings_have_the_width_their_profile_declares() {
    let dir = tempfile::tempdir().expect("temp dir");
    let profile = Profile::Speed;
    let mut embedder = Embedder::load(profile, dir.path()).expect("load model");

    let vector = embedder.embed_passage("a durable claim").expect("embed");
    assert_eq!(
        vector.len() as u32,
        profile.dimensions(),
        "the index is built for this width, so a mismatch would be silent corruption"
    );
}

#[test]
#[ignore = "downloads embedding model weights"]
fn e5_encodes_a_query_and_a_passage_differently() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut embedder = Embedder::load(Profile::Speed, dir.path()).expect("load model");

    let text = "the deployment pipeline runs on continuous integration";
    let as_query = embedder.embed_query(text).expect("embed query");
    let as_passage = embedder.embed_passage(text).expect("embed passage");

    // E5 was trained with `query: ` and `passage: ` in front of the text.
    // Feeding it the bare string produces vectors that are merely worse, never
    // wrong, so nothing else in the pipeline would report this.
    assert_ne!(
        as_query, as_passage,
        "an asymmetric model must see its query and passage prefixes"
    );
}

#[test]
#[ignore = "downloads embedding model weights"]
fn a_symmetric_model_is_left_alone() {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut embedder = Embedder::load(Profile::Accuracy, dir.path()).expect("load model");

    let text = "the deployment pipeline runs on continuous integration";
    assert_eq!(
        embedder.embed_query(text).expect("embed query"),
        embedder.embed_passage(text).expect("embed passage"),
        "pplx takes no prefixes; adding them would be a different kind of bug"
    );
}

/// Batching has to be free of consequence, on every profile that ships.
///
/// It was asserted on `Profile::Speed` alone, which is `Model::Text`. The
/// default is `Profile::Accuracy`, which is `Model::Pooled` -- a different
/// branch, and the quantized one -- so the profile the product actually runs
/// was the one this did not cover. That matters now rather than in principle:
/// `reindex` embeds in batches of 256, the cascade in rounds of up to 64 and a
/// single write alone, so if a batch changed a vector those paths would
/// disagree about what the same text embeds to. On `Profile::Accuracy` a batch
/// also runs on different sessions from a single text -- one thread each,
/// side by side, against every thread on one -- so this is what holds that
/// the sessions and the thread count change nothing either.
#[test]
#[ignore = "downloads embedding model weights"]
fn a_batch_gives_each_text_the_vector_it_would_have_got_alone() {
    for profile in [Profile::Speed, Profile::Accuracy] {
        a_batch_changes_nothing_on(profile);
    }
}

fn a_batch_changes_nothing_on(profile: Profile) {
    let dir = tempfile::tempdir().expect("temp dir");
    let mut embedder = Embedder::load(profile, dir.path()).expect("load model");

    let texts = [
        "the deployment pipeline runs on continuous integration",
        "部署流水线运行在持续集成上",
        "the office coffee machine needs descaling",
    ];

    let alone: Vec<_> = texts
        .iter()
        .map(|text| embedder.embed_passage(text).expect("embed one"))
        .collect();
    let together = embedder.embed_passages(&texts).expect("embed a batch");

    // Position by position, so a batch that returned the right vectors in the
    // wrong order fails here. That is the failure this is for: every vector is
    // a real vector and every text has one, so an index built from a shuffled
    // batch is wrong in a way that nothing downstream can notice -- searches
    // simply return the wrong memories.
    assert_eq!(together.len(), alone.len());
    for (index, (batched, single)) in together.iter().zip(&alone).enumerate() {
        assert_eq!(
            batched, single,
            "{profile:?}: {:?} came back changed",
            texts[index]
        );
    }
}
