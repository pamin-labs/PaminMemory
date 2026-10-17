//! A corpus's vectors, embedded once and kept on disk between runs.
//!
//! The cache is reused only by the producer that wrote it: the embedder's
//! plans, as a record names them ([`shape::provider`]), the model revision
//! each plan ran ([`pamin_index::PlanUse::revision`]), and the build
//! ([`shape::build_digest`]), and the settings that can change what the
//! model computes ([`shape::runtime_settings`]), and this module's own
//! source ([`harness_digest`]), which normalises and encodes what it caches,
//! written into the file's header -- with the input itself
//! ([`input_digest`]): every text, in the order the cache holds their
//! vectors, so a caller that selects or orders its texts another way is
//! another cache whatever its code. It was named
//! by the model and the corpus alone, so vectors embedded on one device or
//! one build were scored as another's; and a plan's hash does not follow a
//! moved `refs/main` on a fresh download, so without the revision the old
//! weights' vectors were scored as the new ones'. A cache from any other
//! producer is embedded again and written over.

use std::io::Write as _;
use std::path::Path;

use pamin_index::{Embedder, PlanUse};
use sha2::{Digest, Sha256};

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

/// Who embeds with `embedder` now: its plans, the build and the settings
/// it runs under, one per line.
pub fn producer(embedder: &Embedder) -> String {
    let secondary = embedder.secondary_plans().unwrap_or_default();
    lines(
        &embedder.plans(),
        &secondary,
        &shape::runtime_settings(),
        &harness_digest(),
    )
}

/// The SHA-256 of this module's source: the code that normalises and
/// encodes the vectors a cache holds. Neither pamin-index's build nor a
/// model plan covers it, and the commit would discard every cache on every
/// commit, so a cache names this file alone.
pub fn harness_digest() -> String {
    format!(
        "{:x}",
        Sha256::digest(include_str!("vectors.rs").as_bytes())
    )
}

/// Who embeds with `engine`'s embedder now, as [`producer`] names an
/// embedder: what a search index the engine writes is built by. `None`
/// while its embedder has not loaded.
pub fn index_producer(engine: &pamin_engine::Engine) -> Option<String> {
    let providers = engine.providers();
    let plans = |role: &str| -> Vec<PlanUse> {
        providers
            .iter()
            .filter(|(of, _)| of == role)
            .map(|(_, plan)| plan.clone())
            .collect()
    };
    let primary = plans("embedding");
    (!primary.is_empty()).then(|| {
        lines(
            &primary,
            &plans("embedding secondary"),
            &shape::runtime_settings(),
            &harness_digest(),
        )
    })
}

/// [`producer`]'s lines for these primary and secondary plans: each plan
/// with the model revision it ran, then the build, then each setting, then
/// the harness code that transformed the vectors.
fn lines(plans: &[PlanUse], secondary: &[PlanUse], settings: &[String], harness: &str) -> String {
    let line = |kind: &str, plan: &PlanUse| {
        format!("{kind}={} model={}", shape::provider(plan), plan.revision)
    };
    let mut lines: Vec<String> = plans.iter().map(|plan| line("plan", plan)).collect();
    lines.extend(secondary.iter().map(|plan| line("secondary", plan)));
    lines.push(format!("build={}", shape::build_digest()));
    lines.extend(settings.iter().map(|setting| format!("setting={setting}")));
    lines.push(format!("harness={harness}"));
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
    embed_with(
        embedder,
        producer,
        |embedder, chunk| embedder.embed_passages(chunk).expect("embed the corpus"),
        path,
        texts,
        width,
        batch,
    )
}

/// [`embed`] over any `model`: `producer` names who embeds with it now and
/// `run` embeds one batch.
///
/// The producer is asked again after every batch, before that batch is
/// written: a model that fell back to another plan partway has just embedded
/// with a producer the file's header does not name, so the cache is removed
/// and nothing more is written to it. Checked only at the end, an
/// interruption in between left a file whose header named the first
/// producer over vectors from two, and the next run reused them.
fn embed_with<M>(
    model: &mut M,
    producer: impl Fn(&M) -> String,
    mut run: impl FnMut(&mut M, &[&str]) -> Vec<Vec<f32>>,
    path: &Path,
    texts: &[&str],
    width: usize,
    batch: usize,
) -> Embedded {
    let embedding = producer(model);
    let named = format!("{embedding}\ninput={}", input_digest(texts));
    let mut vectors = resume(path, &named, width, texts.len());
    let reused = vectors.len() / width;
    if reused > 0 {
        println!("  reusing {reused} embedded texts");
    }

    let started = std::time::Instant::now();
    let mut done = reused;
    let mut caching = true;
    while done < texts.len() {
        let chunk: Vec<&str> = texts[done..].iter().take(batch).copied().collect();
        let embedded = run(model, &chunk);
        if caching && producer(model) != embedding {
            println!("  the embedder's plan changed during the run; removing its vector cache");
            let _ = std::fs::remove_file(path);
            caching = false;
        }

        let mut appended = Vec::with_capacity(chunk.len() * width * 4);
        for vector in embedded {
            let vector = unit(vector);
            for value in &vector {
                appended.extend_from_slice(&value.to_le_bytes());
            }
            vectors.extend_from_slice(&vector);
        }
        if caching {
            append(path, &appended);
        }

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
    Embedded {
        vectors,
        reused,
        producer: named,
    }
}

/// The SHA-256 of `texts` in order, each prefixed by its length: what a
/// cache's vectors are vectors of, position by position. Named by the
/// input rather than by the code that chose it, so texts selected or
/// ordered any other way -- by another commit or an uncommitted change in
/// the caller -- never read back another order's vectors.
fn input_digest(texts: &[&str]) -> String {
    let mut digest = Sha256::new();
    for text in texts {
        digest.update((text.len() as u64).to_le_bytes());
        digest.update(text.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

/// The whole vectors `producer` already wrote to the cache at `path`, with
/// the file cut to exactly them; or a new file of this producer's when
/// there are none to keep.
///
/// Cut, because an interrupted append can leave part of a vector after the
/// last whole one: [`read`] skips it, but a resumed run appending after it
/// would shift every later vector's floats, and the next run would read
/// them back under a valid header.
fn resume(path: &Path, producer: &str, width: usize, most: usize) -> Vec<f32> {
    let vectors = read(path, producer, width, most);
    if vectors.is_empty() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("the cache directory");
        }
        std::fs::write(path, header(producer)).expect("write the vector cache");
    } else {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| file.set_len((header(producer).len() + vectors.len() * 4) as u64))
            .unwrap_or_else(|error| panic!("cutting {} to whole vectors: {error}", path.display()));
    }
    vectors
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

    /// A resumed cache whose last append was cut off partway through a
    /// vector reads back, after more are appended, exactly what an
    /// uninterrupted one does.
    #[test]
    fn a_resume_after_a_partial_vector_appends_on_a_whole_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.f32");
        let producer = "plan=cpu@aaaaaaaaaaaaaaaa\nbuild=b";
        let bytes = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        let mut file = header(producer);
        file.extend(bytes(&[1.0, 0.0]));
        file.extend(&bytes(&[0.0, 1.0])[..6]);
        std::fs::write(&path, &file).unwrap();

        assert_eq!(resume(&path, producer, 2, 3), [1.0, 0.0]);
        append(&path, &bytes(&[0.0, 1.0, 0.6, 0.8]));
        assert_eq!(
            read(&path, producer, 2, 3),
            [1.0, 0.0, 0.0, 1.0, 0.6, 0.8],
            "the resumed vectors are shifted"
        );

        // Nothing of this producer's: the file starts over with its header.
        assert!(resume(&path, "plan=cuda@cccccccccccccccc\nbuild=b", 2, 3).is_empty());
        assert_eq!(
            std::fs::read(&path).unwrap(),
            header("plan=cuda@cccccccccccccccc\nbuild=b")
        );
    }

    /// A model that falls back partway leaves no cache behind under its
    /// first producer's header, so a run interrupted after the fallback
    /// cannot hand the next one a prefix from two producers.
    #[test]
    fn a_fallback_partway_removes_the_cache_before_writing_to_it() {
        struct Model {
            batches: usize,
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.f32");
        let producer = |model: &Model| {
            if model.batches < 2 {
                "plan=cuda@a"
            } else {
                "plan=cpu@b"
            }
            .to_string()
        };
        let run = |model: &mut Model, chunk: &[&str]| {
            model.batches += 1;
            chunk.iter().map(|_| vec![1.0, 0.0]).collect()
        };
        let texts = ["a", "b", "c", "d"];
        // Interrupted in the third batch, after one batch of the new plan.
        let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            embed_with(
                &mut Model { batches: 0 },
                producer,
                |model: &mut Model, chunk: &[&str]| {
                    assert!(model.batches < 2, "interrupted");
                    run(model, chunk)
                },
                &path,
                &texts,
                2,
                1,
            )
        }));
        assert!(interrupted.is_err());
        assert!(
            read(&path, "plan=cuda@a", 2, 4).is_empty(),
            "the first producer's header kept the new plan's vectors"
        );
        assert!(!path.exists(), "the cache outlived the fallback");

        // Without a fallback, the next run resumes what was written.
        let steady = |_: &Model| "plan=cpu@b".to_string();
        let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            embed_with(
                &mut Model { batches: 2 },
                steady,
                |model: &mut Model, chunk: &[&str]| {
                    assert!(model.batches < 4, "interrupted");
                    run(model, chunk)
                },
                &path,
                &texts,
                2,
                1,
            )
        }));
        assert!(interrupted.is_err());
        let resumed = embed_with(&mut Model { batches: 2 }, steady, run, &path, &texts, 2, 1);
        assert_eq!(resumed.reused, 2);
        assert_eq!(resumed.vectors.len(), 8);
    }

    /// A cache holds the vectors of one input, in one order: the same
    /// texts in the same order read it back, and the same texts reversed --
    /// a caller that orders them otherwise -- embed again.
    #[test]
    fn a_cache_is_named_by_its_ordered_input() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vectors.f32");
        let producer = |_: &()| "plan=cpu@a".to_string();
        let run = |_: &mut (), chunk: &[&str]| -> Vec<Vec<f32>> {
            chunk
                .iter()
                .map(|text| {
                    if *text == "a" {
                        vec![1.0, 0.0]
                    } else {
                        vec![0.0, 1.0]
                    }
                })
                .collect()
        };
        let texts = ["a", "b"];
        let first = embed_with(&mut (), producer, run, &path, &texts, 2, 1);
        assert_eq!(first.reused, 0);
        let again = embed_with(&mut (), producer, run, &path, &texts, 2, 1);
        assert_eq!(again.reused, 2, "the same input in the same order");
        assert_eq!(again.vectors, first.vectors);

        let reversed = embed_with(&mut (), producer, run, &path, &["b", "a"], 2, 1);
        assert_eq!(
            reversed.reused, 0,
            "another order read the first one's vectors"
        );
        assert_eq!(reversed.vectors, [0.0, 1.0, 1.0, 0.0]);
        // Nor does a text moved across a boundary read as the same input.
        assert_ne!(input_digest(&["ab", "c"]), input_digest(&["a", "bc"]));
    }

    /// The same plans over a moved `refs/main` are another producer, so
    /// the old weights' vectors are embedded again rather than reused.
    #[test]
    fn a_moved_model_revision_is_another_producer() {
        let plan = |revision: &str| PlanUse {
            target: "cpu".into(),
            plan: "a".repeat(64),
            stable: true,
            revision: revision.into(),
        };
        let old = lines(&[plan("someone/model@c0ffee")], &[], &[], "");
        assert_ne!(old, lines(&[plan("someone/model@decaf")], &[], &[], ""));
        assert_ne!(
            lines(
                &[plan("someone/model@c0ffee")],
                &[plan("other/model@1")],
                &[],
                "",
            ),
            lines(
                &[plan("someone/model@c0ffee")],
                &[plan("other/model@2")],
                &[],
                "",
            )
        );
        assert!(old.contains("model=someone/model@c0ffee"), "{old}");
    }

    /// Another version of this module -- how it normalises or encodes
    /// what it caches -- is another producer, and the digest names it.
    #[test]
    fn another_harness_source_is_another_producer() {
        let plan = PlanUse {
            target: "cpu".into(),
            plan: "a".repeat(64),
            stable: true,
            revision: "someone/model@c0ffee".into(),
        };
        let with = |harness: &str| lines(std::slice::from_ref(&plan), &[], &[], harness);
        assert_ne!(with("0123"), with("4567"));
        let digest = harness_digest();
        assert_eq!(digest.len(), 64);
        assert!(with(&digest).ends_with(&format!("harness={digest}")));
    }

    /// Another thread count for the native kernels is another producer: a
    /// reduction split another way can round differently.
    #[test]
    fn a_native_setting_is_another_producer() {
        let plan = PlanUse {
            target: "cpu".into(),
            plan: "a".repeat(64),
            stable: true,
            revision: "someone/model@c0ffee".into(),
        };
        let with = |setting: &str| lines(std::slice::from_ref(&plan), &[], &[setting.into()], "");
        assert_ne!(with("OMP_NUM_THREADS=1"), with("OMP_NUM_THREADS=4"));
        assert_ne!(
            with("OMP_NUM_THREADS=1"),
            lines(std::slice::from_ref(&plan), &[], &[], "")
        );
        assert!(with("MKL_NUM_THREADS=2").contains("setting=MKL_NUM_THREADS=2"));
    }
}
