//! Runs the real embedding model and the vector channel.
//!
//! Ignored by default: the first run downloads model weights. Run with
//! `cargo test -p pamin-index -- --ignored`.

use pamin_core::TopicStateId;
use pamin_index::{Access, Embedder, Profile, Projection, ProjectionIndex};

fn id(byte: u8) -> TopicStateId {
    TopicStateId(uuid::Uuid::from_bytes([byte; 16]))
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
        hits.first(),
        Some(&chinese),
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
        "BGE-M3 takes no prefixes; adding them would be a different kind of bug"
    );
}
