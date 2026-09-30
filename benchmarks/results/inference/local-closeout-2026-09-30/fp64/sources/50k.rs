#![allow(dead_code)]
//! What the vector channel actually returns, against what it should.
//!
//! Ignored by default: it builds an index of fifty thousand documents, which
//! takes a couple of minutes. Run with
//! `cargo test --release -p pamin-engine --test scratch_fp64_50k scratch_fp64_50k -- --exact --ignored --nocapture`.
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
use pamin_index::{Access, Profile, Projection, ProjectionIndex, VectorIndex};

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
///
/// It measures the default vector index, which is now `memory`: half-precision
/// vectors under HNSW with the f32 rescore, measured at 0.9965 over these
/// vectors (200 queries); `disk` measured 0.9985 at its shipped search width. The floor stays where
/// it was, because what it catches is a collapse -- 0.053 is what int8 with
/// rotation once returned -- and not the second decimal: this index returns
/// 0.981 at DiskANN's own default width of 300, which passes it, and the
/// width is held by the table beside `DISKANN_SEARCH_LIST` instead.
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


fn wide_dot(left:&[f32],right:&[f32])->f64 { left.iter().zip(right).map(|(a,b)|f64::from(*a)*f64::from(*b)).sum() }
fn exact(docs:&[Vec<f32>],norms:&[f64],query:&[f32])->Vec<usize> {
    let norm=wide_dot(query,query).sqrt();
    let mut rows:Vec<_>=docs.iter().enumerate().map(|(index,doc)|(wide_dot(doc,query)/(norm*norms[index]),index)).collect();
    rows.sort_by(|left,right|right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
    rows.into_iter().take(TOP as usize).map(|(_,index)|index).collect()
}
#[test]
#[ignore="paired real 50k memory-index validation of FP32, ORT FP64 and native FP64"]
fn scratch_fp64_50k() {
    use pamin_index::projection::{experimental_math,experimental_candidates};
    let (documents,queries)=corpus();
    let stored:Vec<_>=documents.iter().map(|doc|pamin_index::as_stored(doc)).collect();
    let norms:Vec<_>=documents.iter().map(|doc|wide_dot(doc,doc).sqrt()).collect();
    let stored_norms:Vec<_>=stored.iter().map(|doc|wide_dot(doc,doc).sqrt()).collect();
    let root=std::path::PathBuf::from(std::env::var("FP64_50K_HOME").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let index=ProjectionIndex::open(&root.join("index"),&root.join("legacy"),PROFILE,VectorIndex::Memory,Access::ReadWrite,DOCS as u64).unwrap();
    if !root.join("built").exists() {
        for (start,chunk) in documents.chunks(1024).enumerate() {
            let batch:Vec<_>=chunk.iter().enumerate().map(|(at,vector)|(topic(start*1024+at),"",vector.as_slice())).collect();
            index.upsert_batch(&batch).unwrap();
        }
        index.flush().unwrap(); index.optimize().unwrap();
        std::fs::write(root.join("built"),b"50k seed 0x5eed accuracy1024 memory fp16").unwrap();
    }
    assert!(index.vector_index_completeness().unwrap()>0.99);
    for mode in 0..3 {experimental_math(mode);index.recall_vector(&queries[0],TOP).unwrap();}
    let mut output=Vec::new();
    for (query_index,query) in queries.iter().enumerate() {
        let original=exact(&documents,&norms,query);
        let quantized=exact(&stored,&stored_norms,query);
        let mut expected=None;
        for round in 0..3 {
            for at in 0..3 {
                let mode=((query_index+round+at)%3)as u8;
                experimental_math(mode);
                let start=std::time::Instant::now();
                let hits=index.recall_vector(query,TOP).unwrap();
                let seconds=start.elapsed().as_secs_f64();
                let mut candidates:Vec<_>=experimental_candidates().into_iter().map(position).collect();
                candidates.sort_unstable();
                if let Some(ref before)=expected {assert_eq!(&candidates,before,"ANN candidates changed");} else {expected=Some(candidates.clone());}
                let ids:Vec<_>=hits.iter().map(|hit|position(hit.topic)).collect();
                output.push(serde_json::json!({"query":query_index,"round":round,"mode":mode,"seconds":seconds,"ids":ids,"scores":hits.iter().map(|hit|hit.score).collect::<Vec<_>>(),"gold_original":original,"gold_stored":quantized,"candidates":candidates,"recall_original":ids.iter().filter(|id|original.contains(id)).count()as f64/10.0,"recall_stored":ids.iter().filter(|id|quantized.contains(id)).count()as f64/10.0,"ordered_stored_agreement":ids.iter().zip(&quantized).filter(|(left,right)|left==right).count()}));
            }
        }
        if (query_index+1)%10==0 {println!("50k paired {}/{}",query_index+1,queries.len());}
    }
    experimental_math(0);
    std::fs::write(std::env::var("FP64_OUTPUT").unwrap(),serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
