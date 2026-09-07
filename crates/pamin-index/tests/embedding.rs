//! Runs the real embedding model and the vector channel.
//!
//! Ignored by default: the first run downloads model weights. Run with
//! `cargo test -p pamin-index -- --ignored`.

use pamin_core::TopicStateId;
use pamin_index::{Embedder, Profile, ProjectionIndex};

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

    let index = ProjectionIndex::open(&dir.path().join("index"), profile).expect("open index");

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
