//! A prepared copy of a model scores exactly as the file it came from, and the
//! weights it holds are mapped rather than copied.
//!
//! Ignored by default: it needs the `fast` and `accurate` rerankers and the
//! BGE-M3 embedder, 1.2 GB between them, and writes a copy of each. Run with
//!
//! ```sh
//! cargo test -p pamin-index --test prepared -- --ignored --nocapture
//! ```
//!
//! and with `PAMIN_TEST_MODELS` pointing at a model directory that already
//! holds them to skip the download -- its `models--*` entries are linked into
//! a temporary directory, so the copies are written there and removed after.
//!
//! Each load runs in a child process of its own, through `Reranker::load` and
//! `Embedder::load` -- the functions a server calls -- and that is the point of
//! the design rather than a convenience. Anonymous memory is what the copy is
//! for, and within one process it cannot be compared fairly: an allocator that
//! kept the source's freed weights would hand them straight back to a copy that
//! had fallen back to packing on the heap, and the copy would look free. In a
//! fresh process there is nothing to hand back.

use std::path::{Path, PathBuf};
use std::process::Command;

use pamin_index::{Device, Embedder, Profile, Rerank, Reranker};

/// Which load a child process runs, and where its models are.
const ARM: &str = "PAMIN_PREPARED_ARM";
const CACHE: &str = "PAMIN_PREPARED_CACHE";

/// What a child prints its result after, so it can be found in libtest's own
/// output.
const MARK: &str = "prepared-arm-result";

/// Models loaded by the product's own entry points, by name.
const MODELS: &[&str] = &["fast", "accurate", "embedder"];

const QUERY: &str = "how does the deployment pipeline handle a failed migration";

/// Of different lengths, so a reranker batch pads and an embedding runs at
/// more than one shape.
const TEXTS: &[&str] = &[
    "When a migration fails the deployment pipeline rolls the release back and \
     leaves the previous version serving.",
    "The office coffee machine is descaled on the first Monday of each month.",
    "Deployments are frozen on Fridays after noon.",
    "Database migrations run before the new version takes traffic, and a \
     migration that cannot finish inside its window is reverted automatically \
     so that the schema never drifts from what the running code expects.",
    "Wenn eine Migration fehlschlägt, rollt die Pipeline das Release zurück.",
];

#[test]
#[ignore = "downloads the fast and accurate rerankers and the BGE-M3 embedder"]
fn a_prepared_copy_scores_the_same_and_holds_less() {
    if let Ok(model) = std::env::var(ARM) {
        return child(&model, Path::new(&std::env::var(CACHE).expect("the cache")));
    }

    let dir = tempfile::tempdir().expect("temp dir");
    let cache = dir.path().join("models");
    std::fs::create_dir_all(&cache).expect("make the cache");
    link_models(&cache);
    let copies = cache.join("prepared");

    for model in MODELS {
        let source = run(model, &cache, false);

        // The first prepared load writes the copy, and so loads the source
        // onto the heap in the same process before loading the copy: its
        // scores count, its memory does not.
        let writing = run(model, &cache, true);
        let copy = only_copy(&copies);
        let data = std::fs::metadata(copy.join("model.onnx.data"))
            .unwrap_or_else(|_| panic!("the {model} copy has no data file, so nothing is mapped"))
            .len()
            / 1024;
        let prepared = run(model, &cache, true);

        assert_eq!(
            source.bits, writing.bits,
            "the {model} copy, loaded in the process that wrote it, is not bit-identical \
             to the source"
        );
        assert_eq!(
            source.bits, prepared.bits,
            "the {model} copy is not bit-identical to the source"
        );

        println!(
            "  {model}: {} outputs bit-identical; anonymous MiB, source against copy: \
             after load {} / {}, after a pass {} / {}, live {} / {}; copy data file {} MiB",
            source.bits.len(),
            mebibytes(source.loaded),
            mebibytes(prepared.loaded),
            mebibytes(source.scored),
            mebibytes(prepared.scored),
            mebibytes(source.held),
            mebibytes(prepared.held),
            data / 1024,
        );

        // What the copy keeps off the heap, against what it maps.
        //
        // Both loads hold the same tokenizer as well, so the difference is the
        // session's. Half the data file is the line between the two things a
        // copy can be. Mapped, the `accurate` copy keeps 551 MiB of live
        // memory off the heap against a data file of 833 MiB. A copy that
        // finds none of its packed weights -- written for another CPU, or no
        // longer matched by the runtime -- still maps everything else and
        // scores identically: written without them, a bare session over it
        // held 304 MiB against the source's 667, a saving of 363, under half.
        // Nothing but this assertion tells those two apart.
        if let (Some(source), Some(prepared)) = (source.held, prepared.held) {
            let saved = source.saturating_sub(prepared);
            assert!(
                saved * 2 >= data,
                "the {model} copy saves {saved} kB of live anonymous memory against the source, \
                 under half of its {data} kB data file -- its weights are not being mapped"
            );
        }

        // The next model's copy is written on a disk that need not hold both.
        std::fs::remove_dir_all(&copies).expect("remove the copies");
    }
}

/// What one child load returned.
struct Arm {
    /// Every output as `f32` bits, so equality means bit-identical.
    bits: Vec<u32>,
    /// Anonymous memory added by the load, then after one pass, then after
    /// the allocator has returned what it holds free -- see [`trim`] -- in
    /// kB, with the model loaded throughout. `None` where the platform cannot
    /// say.
    loaded: Option<u64>,
    scored: Option<u64>,
    held: Option<u64>,
}

/// Runs one load in a fresh process, from the prepared copy or from the source.
fn run(model: &str, cache: &Path, prepared: bool) -> Arm {
    let mut command = Command::new(std::env::current_exe().expect("this test binary"));
    command
        .args([
            "--exact",
            "a_prepared_copy_scores_the_same_and_holds_less",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM, model)
        .env(CACHE, cache)
        // The copy is for the CPU; on a machine with a GPU the reranker would
        // load the fp16 export there and neither arm would test anything.
        .env("PAMIN_DEVICE", "cpu");
    if prepared {
        command.env_remove("PAMIN_PREPARED");
    } else {
        command.env("PAMIN_PREPARED", "off");
    }
    let output = command.output().expect("run the child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {model} child (prepared: {prepared}) failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let line = stdout
        .lines()
        // Mid-line: libtest prints the test's name before its output.
        .find_map(|line| line.split_once(MARK).map(|(_, result)| result))
        .unwrap_or_else(|| panic!("the {model} child printed no result:\n{stdout}"));
    let mut fields = line.split_whitespace();
    let mut memory = || fields.next().expect("a memory field").parse::<u64>().ok();
    let (loaded, scored, held) = (memory(), memory(), memory());
    let bits = fields
        .map(|bits| u32::from_str_radix(bits, 16).expect("hex bits"))
        .collect();
    Arm {
        bits,
        loaded,
        scored,
        held,
    }
}

/// The child: one load, one pass, and what it held.
fn child(model: &str, cache: &Path) {
    let before = anonymous();
    let added = |now: Option<u64>| {
        before
            .zip(now)
            .map(|(before, now)| now.saturating_sub(before))
    };

    // Kept alive until every figure below is taken: dropped any earlier, the
    // last of them would measure a process that no longer holds a model.
    let mut score: Box<dyn FnMut() -> Vec<f32>> = match model {
        "fast" | "accurate" => {
            let tier = Rerank::parse(model).expect("a tier");
            let mut reranker = Reranker::load(tier, cache).expect("load the reranker");
            assert_eq!(reranker.device(), Device::Cpu, "the copy is for the CPU");
            Box::new(move || {
                let mut scores = vec![f32::NAN; TEXTS.len()];
                for ranked in reranker.rank(QUERY, TEXTS).expect("rank") {
                    scores[ranked.position] = ranked.score;
                }
                scores
            })
        }
        "embedder" => {
            let mut embedder = Embedder::load(Profile::Accuracy, cache).expect("load the embedder");
            Box::new(move || {
                let vectors = embedder.embed_passages(TEXTS).expect("embed");
                vectors.into_iter().flatten().collect()
            })
        }
        other => panic!("no such model {other}"),
    };
    let loaded = added(anonymous());
    let outputs = score();
    let scored = added(anonymous());
    trim();
    let held = added(anonymous());

    let memory = |kb: Option<u64>| kb.map_or_else(|| "-".to_string(), |kb| kb.to_string());
    let bits: Vec<String> = outputs
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect();
    println!(
        "{MARK} {} {} {} {}",
        memory(loaded),
        memory(scored),
        memory(held),
        bits.join(" ")
    );
    drop(score);
}

/// Hands back to the system what the allocator holds free.
///
/// Loading parses a 17 MB tokenizer file and discards the parse, and the
/// allocator keeps what that freed until something makes it give memory back.
/// When that happens differs between two processes that did different
/// things, so a figure taken before it is partly the allocator's timing.
/// After a trim, what is left is what is live.
fn trim() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free memory to the system.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Anonymous memory this process holds, in kB, where the platform says.
fn anonymous() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .ok()?
        .lines()
        .find_map(|line| line.strip_prefix("Anonymous:"))?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn mebibytes(kb: Option<u64>) -> String {
    kb.map_or_else(|| "-".to_string(), |kb| (kb / 1024).to_string())
}

/// The one copy under `copies`, which the load just wrote.
fn only_copy(copies: &Path) -> PathBuf {
    let written: Vec<PathBuf> = std::fs::read_dir(copies)
        .expect("the copies directory")
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| path.is_dir())
        .collect();
    assert_eq!(
        written.len(),
        1,
        "expected exactly one complete copy, found {written:?}"
    );
    written.into_iter().next().expect("one copy")
}

/// Links the models an existing cache holds into `cache`, so nothing is
/// downloaded and nothing is written beside them.
fn link_models(cache: &Path) {
    let Ok(existing) = std::env::var("PAMIN_TEST_MODELS") else {
        return;
    };
    for entry in std::fs::read_dir(&existing).expect("PAMIN_TEST_MODELS") {
        let path = entry.expect("an entry").path();
        let name = path.file_name().expect("a name").to_owned();
        if name.to_string_lossy().starts_with("models--") {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&path, cache.join(&name)).expect("link the model");
            #[cfg(windows)]
            std::os::windows::fs::symlink_dir(&path, cache.join(&name)).expect("link the model");
        }
    }
}
