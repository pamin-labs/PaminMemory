//! A corpus's vectors, embedded once and kept on disk between runs.
//!
//! The cache is reused only by the producer that wrote it: the embedder's
//! plans, as a record names them ([`shape::provider`]), and the build
//! ([`shape::build_digest`]), written into the file's header. It was named
//! by the model and the corpus alone, so vectors embedded on one device or
//! one build were scored as another's. A cache from any other producer is
//! embedded again and written over.

use std::io::Write as _;
use std::path::Path;

use pamin_index::Embedder;

use super::shape;

/// The first line of every cache file.
const MAGIC: &[u8] = b"pamin-vectors-1\n";

/// What [`embed`] returned: the vectors in order, unit length, and how many
/// of them came from the cache and from which producer.
pub struct Embedded {
    pub vectors: Vec<f32>,
    pub reused: usize,
    pub producer: String,
}

/// Who embeds with `embedder` now: its plans and the build, one per line.
pub fn producer(embedder: &Embedder) -> String {
    let mut lines: Vec<String> = embedder
        .plans()
        .iter()
        .map(|plan| format!("plan={}", shape::provider(plan)))
        .collect();
    lines.push(format!("build={}", shape::build_digest()));
    lines.join("\n")
}

/// `producer`'s header, as it starts a cache file.
fn header(producer: &str) -> Vec<u8> {
    let mut header = MAGIC.to_vec();
    header.extend_from_slice(producer.as_bytes());
    header.extend_from_slice(b"\n\n");
    header
}

/// The vectors in the cache at `path` that `producer` wrote, whole ones
/// only; none if another producer wrote it, or none did.
fn read(path: &Path, producer: &str, width: usize, most: usize) -> Vec<f32> {
    let Ok(raw) = std::fs::read(path) else {
        return Vec::new();
    };
    let Some(body) = raw.strip_prefix(header(producer).as_slice()) else {
        return Vec::new();
    };
    let mut done = body.len() / 4 / width;
    if done > most {
        // A truncated write can leave half a vector, and half a vector is
        // worse than none.
        done = 0;
    }
    body[..done * width * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

/// Embeds every one of `texts`, `batch` to a call, reusing and resuming the
/// cache at `path` if the current producer wrote it.
///
/// Resumable rather than merely cached: these are minutes to hours of
/// forward passes, and a machine that loses the process halfway would
/// otherwise start again from nothing. Vectors are appended a batch at a
/// time, so the length of the file says how many are done. A producer that
/// changes during the run -- a fallback -- leaves vectors from two, so the
/// cache is removed rather than kept for either.
pub fn embed(
    embedder: &mut Embedder,
    path: &Path,
    texts: &[&str],
    width: usize,
    batch: usize,
) -> Embedded {
    let producer = producer(embedder);
    let mut vectors = read(path, &producer, width, texts.len());
    let reused = vectors.len() / width;
    if reused > 0 {
        println!("  reusing {reused} embedded texts");
    } else {
        // Nothing of this producer's to keep: start the file over.
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("the cache directory");
        }
        std::fs::write(path, header(&producer)).expect("write the vector cache");
    }

    let started = std::time::Instant::now();
    let mut done = reused;
    while done < texts.len() {
        let chunk: Vec<&str> = texts[done..].iter().take(batch).copied().collect();
        let embedded = embedder.embed_passages(&chunk).expect("embed the corpus");

        let mut appended = Vec::with_capacity(chunk.len() * width * 4);
        for vector in embedded {
            let vector = unit(vector);
            for value in &vector {
                appended.extend_from_slice(&value.to_le_bytes());
            }
            vectors.extend_from_slice(&vector);
        }
        append(path, &appended);

        done += chunk.len();
        if done.is_multiple_of(batch * 50) {
            let rate = (done - reused) as f64 / started.elapsed().as_secs_f64();
            println!(
                "  embedded {done}/{} at {rate:.1}/s, {:.0} minutes left",
                texts.len(),
                (texts.len() - done) as f64 / rate / 60.0,
            );
        }
    }

    if self::producer(embedder) != producer {
        println!("  the embedder's plan changed during the run; removing its vector cache");
        let _ = std::fs::remove_file(path);
    }
    Embedded {
        vectors,
        reused,
        producer,
    }
}

fn append(path: &Path, bytes: &[u8]) {
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .unwrap_or_else(|error| panic!("opening {}: {error}", path.display()));
    file.write_all(bytes).expect("write the vector cache");
}

/// Scales a vector to unit length, so a dot product is a cosine.
pub fn unit(mut vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors another producer wrote -- another device's plan, another
    /// build -- are not read back as this one's, and a file from before the
    /// header is not read at all.
    #[test]
    fn a_cache_from_another_producer_is_not_reused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.f32");
        let ours = "plan=cpu@aaaaaaaaaaaaaaaa\nbuild=b";
        let mut file = header(ours);
        for value in [1.0f32, 0.0, 0.0, 1.0] {
            file.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, &file).unwrap();
        assert_eq!(read(&path, ours, 2, 2), [1.0, 0.0, 0.0, 1.0]);
        assert!(read(&path, "plan=cuda:0@cccccccccccccccc\nbuild=b", 2, 2).is_empty());
        assert!(read(&path, "plan=cpu@aaaaaaaaaaaaaaaa\nbuild=c", 2, 2).is_empty());
        // A half-written vector at the end is not read.
        file.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(&path, &file).unwrap();
        assert_eq!(read(&path, ours, 2, 2).len(), 4);
        // The headerless format this replaced names no producer.
        std::fs::write(&path, &file[header(ours).len()..]).unwrap();
        assert!(read(&path, ours, 2, 2).is_empty());
    }
}
