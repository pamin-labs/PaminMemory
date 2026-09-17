//! What the vector channel actually returns, against what it should.
//!
//! Ignored by default: it builds an index of fifty thousand documents, which
//! takes a couple of minutes. Run with
//! `cargo test -p pamin-index --test recall -- --ignored --nocapture`.
//!
//! Every other test here asks whether a query finds the memory it was written
//! for. On a handful of documents an approximate index and an exact one answer
//! that identically, so none of them can see the graph parameters at all --
//! which is how this project ran for its whole life with a vector channel
//! returning about seven of every ten true nearest neighbours and nothing
//! reporting it. An approximate index that is tuned badly does not fail; it
//! returns plausible neighbours that are not the nearest ones.
//!
//! Fifty thousand is small against what this store is for and large enough for
//! the graph to exist. It is also large enough to be segmented the way the
//! engine segments a real project of that size, which is the shape being
//! measured: recall falls as a *graph* grows, not as a project does, so what
//! keeps this number where it is is that segments stop growing.

use pamin_core::TopicId;
use pamin_index::{Access, Profile, Projection, ProjectionIndex};

/// The default profile's width, so this measures the shape actually shipped.
const PROFILE: Profile = Profile::Accuracy;
const DOCS: usize = 50_000;
const CLUSTERS: usize = 250;
const QUERIES: usize = 100;
const TOP: u32 = 10;

/// What the shipped configuration measured when this was written.
///
/// A floor rather than a target, set below the 0.999 measured so ordinary
/// variation does not trip it. It catches the graph parameters being lowered,
/// which is the change that has no other symptom.
///
/// Raised from 0.90 when segments stopped growing without bound. One graph over
/// all fifty thousand documents returned 0.952 in 12.8 ms and took 126 s to
/// build; four graphs over the same documents return **0.999 in 9.4 ms** and
/// take 55 s. Smaller graphs are not a trade here -- they are more accurate,
/// faster to search and cheaper to build, and the floor moves with them because
/// the reason is understood rather than incidental.
const FLOOR: f64 = 0.97;

/// Deterministic pseudo-random, so two runs measure the same corpus.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
}

fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut vector {
        *x /= norm;
    }
    vector
}

/// Clustered vectors, and queries drawn from the same clusters.
///
/// Uniformly random vectors in a thousand dimensions are all nearly orthogonal
/// to each other, so every distance is about the same and "nearest" is noise.
/// Recall measured against noise measures nothing. Embeddings of real text are
/// clustered, so this is too.
fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = Rng(0x5eed);
    let dimensions = PROFILE.dimensions() as usize;

    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| unit((0..dimensions).map(|_| rng.next()).collect()))
        .collect();

    let documents = (0..DOCS)
        .map(|i| {
            let centroid = &centroids[i % CLUSTERS];
            unit(centroid.iter().map(|x| x + 0.35 * rng.next()).collect())
        })
        .collect();

    let queries = (0..QUERIES)
        .map(|i| {
            let centroid = &centroids[(i * 7) % CLUSTERS];
            unit(centroid.iter().map(|x| x + 0.35 * rng.next()).collect())
        })
        .collect();

    (documents, queries)
}

/// The true nearest documents, by the metric the index is built for.
fn nearest(documents: &[Vec<f32>], query: &[f32]) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = documents
        .iter()
        .enumerate()
        .map(|(i, d)| (d.iter().zip(query).map(|(a, b)| a * b).sum::<f32>(), i))
        .collect();
    scored.sort_by(|left, right| right.0.partial_cmp(&left.0).expect("comparable"));
    scored
        .into_iter()
        .take(TOP as usize)
        .map(|(_, i)| i)
        .collect()
}

/// Documents are keyed by topic, so the corpus index has to survive the trip.
fn topic(index: usize) -> TopicId {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&(index as u64).to_be_bytes());
    TopicId(uuid::Uuid::from_bytes(bytes))
}

fn position(topic: TopicId) -> usize {
    let bytes = topic.0.as_bytes();
    u64::from_be_bytes(bytes[..8].try_into().expect("eight bytes")) as usize
}

#[test]
#[ignore = "builds an index of fifty thousand documents"]
fn the_vector_channel_returns_the_nearest_documents_and_not_merely_near_ones() {
    let (documents, queries) = corpus();
    let dir = tempfile::tempdir().expect("temp dir");

    // Sized as the engine would size it for a project this big, so this
    // measures the shape that ships rather than one segment holding everything.
    let index = ProjectionIndex::open(
        &dir.path().join("index"),
        &dir.path().join("legacy"),
        PROFILE,
        Access::ReadWrite,
        DOCS as u64,
    )
    .expect("open index");

    let built = std::time::Instant::now();
    for (start, chunk) in documents.chunks(1024).enumerate() {
        let batch: Vec<(TopicId, &str, &[f32])> = chunk
            .iter()
            .enumerate()
            .map(|(i, vector)| (topic(start * 1024 + i), "", vector.as_slice()))
            .collect();
        index.upsert_batch(&batch).expect("write a batch");
    }
    index.flush().expect("flush");
    // Without this the documents sit in a flat buffer that vector search scans
    // exhaustively, which has perfect recall and is not what ships.
    index.optimize().expect("build the graph");
    let built = built.elapsed();

    assert!(
        index.vector_index_completeness().expect("completeness") > 0.99,
        "the graph has to be built for this to be measuring the graph"
    );

    let mut found = 0usize;
    let mut searching = std::time::Duration::ZERO;
    for query in &queries {
        // Timed around the index alone. Computing the exact answer is fifty
        // million multiply-adds in a debug build and dwarfs everything else.
        let want = nearest(&documents, query);

        let started = std::time::Instant::now();
        let got = index.recall_vector(query, TOP).expect("recall");
        searching += started.elapsed();

        found += got
            .iter()
            .filter(|topic| want.contains(&position(**topic)))
            .count();
    }
    let recall = found as f64 / (QUERIES * TOP as usize) as f64;

    println!(
        "\n  {DOCS} documents, {} dimensions\n  recall@{TOP} {recall:.4}   {:?} per query   {built:?} to build\n",
        PROFILE.dimensions(),
        searching / QUERIES as u32,
    );

    assert!(
        recall >= FLOOR,
        "recall@{TOP} fell to {recall:.4}, below the {FLOOR:.2} floor"
    );
}
