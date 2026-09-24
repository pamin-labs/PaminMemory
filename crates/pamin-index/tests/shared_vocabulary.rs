//! The two rerankers, run through this crate's own encoder over a shared
//! vocabulary, produce exactly what `fastembed` produced for them, and hold
//! one vocabulary where `fastembed` held two.
//!
//! BGE-M3 was the other model here until pplx-embed replaced it: it shared
//! the `accurate` reranker's 250,002-piece vocabulary, and pplx has one of its
//! own. The `fast` reranker's is the same one, so the pair measured is the two
//! tiers -- a server asked for both holds both.
//!
//! Ignored by default: it needs the `accurate` and `fast` rerankers, and writes
//! a prepared copy of each -- over a gigabyte of disk while it runs. Run with
//!
//! ```sh
//! cargo test --release -p pamin-index --test shared_vocabulary -- --ignored --nocapture
//! ```
//!
//! and with `PAMIN_TEST_MODELS` pointing at a model directory that already
//! holds them to skip the downloads, as `tests/prepared.rs` takes it.
//! `--release`, because the timing it prints is of tokenization as well as
//! inference.
//!
//! Each arm runs in a child process of its own, for the reason
//! `tests/prepared.rs` gives: anonymous memory is what is being compared, and
//! within one process an allocator can hand one arm what another freed. The
//! `fastembed` arm is the code the product ran before the encoder -- the same
//! `fastembed` type built the way `Reranker::load` built it, over the same
//! prepared copies, with the candidates sorted and batched the way
//! `Reranker::rank` sorts and batches them -- and the other is the product's
//! own entry point.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use fastembed::{
    OnnxSource, RerankInitOptionsUserDefined, TextRerank, TokenizerFiles, UserDefinedRerankingModel,
};
use pamin_index::{Device, Rerank, Reranker};

/// Which arm a child runs, and where its models are.
const ARM: &str = "PAMIN_SHARED_ARM";
const CACHE: &str = "PAMIN_SHARED_CACHE";
const FAST_COPY: &str = "PAMIN_SHARED_FAST_COPY";
const ACCURATE_COPY: &str = "PAMIN_SHARED_ACCURATE_COPY";

/// What a child prints before each of its results.
const MARK: &str = "shared-vocabulary-result";

/// The repositories `fastembed` read each tier from.
const FAST: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
const ACCURATE: &str = "onnx-community/bge-reranker-v2-m3-ONNX";

/// What `Reranker::rank` batched by, and truncated at, before this change.
const BATCH: usize = 8;
const RERANK_TOKENS: usize = 256;

/// Queries, with the whitespace the two tokenizers' normalizers treat
/// differently: runs of spaces, trailing and leading spaces, and none at all.
const QUERIES: &[&str] = &[
    "how does the deployment pipeline handle a failed migration",
    "Wie  geht die Pipeline mit einer   fehlgeschlagenen Migration um?  ",
    "数据库迁移失败时会发生什么",
    "   kwa nini uhamiaji ulishindwa   ",
    "",
];

/// More than one batch of candidates, in several scripts, with the same
/// whitespace cases, an empty one, and one far past both length limits.
fn passages() -> Vec<String> {
    let mut passages: Vec<String> = [
        "When a migration fails the deployment pipeline rolls the release back and leaves \
         the previous version serving.",
        "Wenn eine Migration fehlschlägt, rollt die Pipeline das Release zurück.",
        "迁移失败时，部署流水线会回滚发布，并保留之前的版本继续提供服务。",
        "เมื่อการย้ายข้อมูลล้มเหลว ระบบจะย้อนกลับการเผยแพร่",
        "Uhamiaji ukishindwa, mfumo hurudisha toleo la awali.",
        "عندما يفشل الترحيل، يعيد خط النشر الإصدار السابق.",
        "Если миграция не удалась, конвейер откатывает выпуск.",
        "移行が失敗すると、パイプラインはリリースをロールバックします。",
        "double  spaced   text  with    runs  of spaces",
        "trailing spaces after the last word     ",
        "",
        "\t tabs and\nnewlines \n between  words\t",
    ]
    .map(String::from)
    .to_vec();
    passages.push(
        "The deployment pipeline runs every migration before traffic moves, and one that \
         cannot finish inside its window is reverted so the schema never drifts. \
         Die Pipeline führt jede Migration aus. 部署流水线会先运行迁移。 "
            .repeat(40),
    );
    passages
}

/// Pairs of about two hundred tokens for the timing passes, different on every
/// pass so the reranker's score cache never answers one.
fn timed(pass: usize) -> Vec<String> {
    (0..16)
        .map(|document| {
            format!(
                "Record {pass}-{document}. When a database migration fails partway through a \
                 release, the deployment pipeline stops sending traffic to the new version, \
                 restores the schema from the snapshot it took before the migration began, \
                 and leaves the previous version serving every request until an operator has \
                 read the log and decided whether to retry. A migration that cannot finish \
                 inside its maintenance window is treated the same way, because a schema that \
                 has drifted from what the running code expects is worse than a release that \
                 has not happened yet. Operators are paged only when the rollback itself \
                 fails, which in practice means a snapshot that could not be restored, a lock \
                 that was never released, or a disk that filled while the copy was written; \
                 each of those leaves a note in the incident channel with the migration's \
                 name, its duration and the step it reached, so the next attempt starts from \
                 what is known rather than from a guess about what went wrong last time."
            )
        })
        .collect()
}

const TIMED_QUERY: &str = "what happens when a migration fails during a release";
const PASSES: usize = 20;
const ROUNDS: usize = 3;

#[test]
#[ignore = "downloads two rerankers and writes a prepared copy of each"]
fn the_encoder_matches_fastembed_and_holds_one_vocabulary() {
    if let Ok(arm) = std::env::var(ARM) {
        return child(&arm, Path::new(&std::env::var(CACHE).expect("the cache")));
    }

    let dir = tempfile::tempdir().expect("temp dir");
    let cache = dir.path().join("models");
    std::fs::create_dir_all(&cache).expect("make the cache");
    link_models(&cache);
    assert_timed_pairs_are_about_two_hundred_tokens(&cache);

    // Each copy written by a child of its own, so it can be told apart from
    // the other by being the one that appeared.
    let copies = cache.join("prepared");
    run("write-fast", &cache, &[]);
    let fast_copy = only_new(&copies, &[]);
    run("write-accurate", &cache, &[]);
    let accurate_copy = only_new(&copies, std::slice::from_ref(&fast_copy));
    let copies = [
        (FAST_COPY, fast_copy.to_str().expect("a UTF-8 path")),
        (ACCURATE_COPY, accurate_copy.to_str().expect("a UTF-8 path")),
    ];

    let mut old = Vec::new();
    let mut new = Vec::new();
    for round in 0..ROUNDS {
        // Alternating which goes first, so neither always follows the other's
        // page cache.
        if round % 2 == 0 {
            old.push(run("old", &cache, &copies));
            new.push(run("new", &cache, &copies));
        } else {
            new.push(run("new", &cache, &copies));
            old.push(run("old", &cache, &copies));
        }
    }

    for (old, new) in old.iter().zip(&new) {
        for key in ["fast", "rank"] {
            let (was, is) = (old.get(key), new.get(key));
            assert!(!was.is_empty(), "the old arm reported no {key} outputs");
            assert_eq!(
                was, is,
                "{key}: the encoder is not bit-identical to fastembed"
            );
        }
    }
    println!(
        "  bit-identical: {} `fast` and {} `accurate` scores over {} queries",
        new[0].get("fast").len(),
        new[0].get("rank").len(),
        QUERIES.len(),
    );

    let memory = |arms: &[Report], key: &str| -> Vec<u64> {
        arms.iter()
            .map(|arm| arm.get(key)[0].parse().expect("kB"))
            .collect()
    };
    for key in ["loaded", "held"] {
        println!(
            "  anonymous MiB with both models, {key}: fastembed {:?}, encoder {:?}",
            mebibytes(&memory(&old, key)),
            mebibytes(&memory(&new, key)),
        );
    }
    let (was, is) = (median(memory(&old, "held")), median(memory(&new, "held")));
    // One vocabulary measured 280 MiB. What the encoder saves has to be most
    // of one, or the two models are not sharing it.
    assert!(
        was.saturating_sub(is) >= 200 * 1024,
        "the encoder holds {is} kB against fastembed's {was} kB -- under 200 MiB less, so the \
         vocabulary is not shared"
    );

    let seconds = |arms: &[Report]| -> Vec<String> {
        arms.iter().map(|arm| arm.get("load")[0].clone()).collect()
    };
    println!(
        "  seconds to load both: fastembed {:?}, encoder {:?}",
        seconds(&old),
        seconds(&new)
    );
    let passes = |arms: &[Report]| -> Vec<Vec<f64>> {
        arms.iter()
            .map(|arm| {
                arm.get("passes")
                    .iter()
                    .map(|ms| ms.parse().expect("ms"))
                    .collect()
            })
            .collect()
    };
    for (name, rounds) in [("fastembed", passes(&old)), ("encoder", passes(&new))] {
        let medians: Vec<String> = rounds
            .iter()
            .map(|round| format!("{:.1}", median_f(round.clone())))
            .collect();
        let fastest: Vec<String> = rounds
            .iter()
            .map(|round| format!("{:.1}", round.iter().copied().fold(f64::MAX, f64::min)))
            .collect();
        println!(
            "  {name}: {PASSES} passes of 16 pairs, ms per pass by round, median {medians:?}, \
             fastest {fastest:?}"
        );
    }
}

/// The child: one arm, its outputs as bits, its memory and its timings.
fn child(arm: &str, cache: &Path) {
    for tuned in ["PAMIN_RERANK_BATCH", "PAMIN_RERANK_MAX_TOKENS"] {
        assert!(
            std::env::var_os(tuned).is_none(),
            "{tuned} is set, so the product is not batching or truncating as the old arm does"
        );
    }
    let passages = passages();
    let documents: Vec<&str> = passages.iter().map(String::as_str).collect();

    match arm {
        "write-fast" => {
            Reranker::load(Rerank::Fast, cache).expect("load the reranker");
        }
        "write-accurate" => {
            Reranker::load(Rerank::Accurate, cache).expect("load the reranker");
        }
        "old" | "new" => {
            let before = anonymous();
            let started = Instant::now();
            let (mut fast, mut rank): (Rank, Rank) = if arm == "new" {
                let product = |tier| -> Rank {
                    let mut reranker = Reranker::load(tier, cache).expect("load the reranker");
                    assert_eq!(reranker.device(), Device::Cpu, "the copy is for the CPU");
                    Box::new(move |query, documents| {
                        let mut scores = vec![f32::NAN; documents.len()];
                        for ranked in reranker.rank(query, documents).expect("rank") {
                            scores[ranked.position] = ranked.score;
                        }
                        scores
                    })
                };
                (product(Rerank::Fast), product(Rerank::Accurate))
            } else {
                let old = |copy: &str, repository: &str| -> Rank {
                    let copy = PathBuf::from(std::env::var(copy).expect("a copy"));
                    let mut reranker = TextRerank::try_new_from_user_defined(
                        UserDefinedRerankingModel::new(
                            OnnxSource::File(copy.join("model.onnx")),
                            tokenizer(cache, repository),
                        ),
                        RerankInitOptionsUserDefined::new()
                            .with_max_length(RERANK_TOKENS)
                            .with_execution_providers(vec![cpu()]),
                    )
                    .expect("load fastembed's reranker");
                    Box::new(move |query, documents| old_rank(&mut reranker, query, documents))
                };
                (old(FAST_COPY, FAST), old(ACCURATE_COPY, ACCURATE))
            };
            let load = started.elapsed().as_secs_f64();
            let loaded = anonymous();

            let scored = |rank: &mut Rank| -> Vec<f32> {
                QUERIES
                    .iter()
                    .flat_map(|query| rank(query, &documents))
                    .collect()
            };
            let fast_ranked = scored(&mut fast);
            let ranked = scored(&mut rank);

            let warm = timed(PASSES);
            rank(
                TIMED_QUERY,
                &warm.iter().map(String::as_str).collect::<Vec<_>>(),
            );
            let mut passes = Vec::with_capacity(PASSES);
            for pass in 0..PASSES {
                let pairs = timed(pass);
                let pairs: Vec<&str> = pairs.iter().map(String::as_str).collect();
                let started = Instant::now();
                let scores = rank(TIMED_QUERY, &pairs);
                passes.push(started.elapsed().as_secs_f64() * 1000.0);
                assert!(scores.iter().all(|score| score.is_finite()));
            }

            trim();
            let held = anonymous();
            let added = |now: u64| now.saturating_sub(before).to_string();
            report("loaded", &[added(loaded)]);
            report("held", &[added(held)]);
            report("load", &[format!("{load:.2}")]);
            report("fast", &bits(&fast_ranked));
            report("rank", &bits(&ranked));
            report(
                "passes",
                &passes
                    .iter()
                    .map(|ms| format!("{ms:.2}"))
                    .collect::<Vec<_>>(),
            );
            drop((fast, rank));
        }
        other => panic!("no such arm {other}"),
    }
}

/// An arm's reranker: a query and candidates in, a score per candidate out.
type Rank = Box<dyn FnMut(&str, &[&str]) -> Vec<f32>>;

/// `Reranker::rank` as it was: candidates sorted by characters, scored by
/// `fastembed` in batches of eight, and each score put back at its position.
fn old_rank(model: &mut TextRerank, query: &str, documents: &[&str]) -> Vec<f32> {
    let mut order: Vec<usize> = (0..documents.len()).collect();
    order.sort_by_key(|position| documents[*position].chars().count());
    let batch: Vec<&str> = order.iter().map(|position| documents[*position]).collect();
    let mut scores = vec![f32::NAN; documents.len()];
    for result in model
        .rerank(query, &batch, false, Some(BATCH))
        .expect("rerank")
    {
        scores[order[result.index]] = result.score;
    }
    scores
}

/// The CPU provider every model here loads with: `inference::cpu`.
fn cpu() -> fastembed::ExecutionProviderDispatch {
    ort::ep::CPU::default().with_arena_allocator(false).build()
}

fn hub(cache: &Path, repository: &str) -> hf_hub::api::sync::ApiRepo {
    hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache.to_path_buf())
        .with_progress(false)
        .build()
        .expect("the hub")
        .model(repository.to_string())
}

/// A repository's four tokenizer files, as `Repository::tokenizer` read them.
fn tokenizer(cache: &Path, repository: &str) -> TokenizerFiles {
    let hub = hub(cache, repository);
    let read = |file: &str| std::fs::read(hub.get(file).expect("a tokenizer file")).expect("read");
    TokenizerFiles {
        tokenizer_file: read("tokenizer.json"),
        config_file: read("config.json"),
        special_tokens_map_file: read("special_tokens_map.json"),
        tokenizer_config_file: read("tokenizer_config.json"),
    }
}

/// The timing passes are of the length they claim, measured with the
/// reranker's own tokenizer and without its truncation.
fn assert_timed_pairs_are_about_two_hundred_tokens(cache: &Path) {
    let file = hub(cache, ACCURATE)
        .get("tokenizer.json")
        .expect("the reranker's tokenizer");
    let tokenizer = tokenizers::Tokenizer::from_file(file).expect("read it");
    for document in timed(0) {
        let length = tokenizer
            .encode((TIMED_QUERY, document.as_str()), true)
            .expect("encode")
            .len();
        assert!(
            (180..=RERANK_TOKENS).contains(&length),
            "a timed pair is {length} tokens, not about two hundred"
        );
    }
}

fn bits(values: &[f32]) -> Vec<String> {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect()
}

fn report(key: &str, values: &[String]) {
    println!("{MARK} {key} {}", values.join(" "));
}

/// What one child reported, by key.
struct Report(Vec<(String, Vec<String>)>);

impl Report {
    fn get(&self, key: &str) -> &[String] {
        self.0
            .iter()
            .find(|(held, _)| held == key)
            .map_or(&[], |(_, values)| values.as_slice())
    }
}

/// Runs one arm in a fresh process.
fn run(arm: &str, cache: &Path, env: &[(&str, &str)]) -> Report {
    let mut command = Command::new(std::env::current_exe().expect("this test binary"));
    command
        .args([
            "--exact",
            "the_encoder_matches_fastembed_and_holds_one_vocabulary",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ARM, arm)
        .env(CACHE, cache)
        .env("PAMIN_DEVICE", "cpu")
        .env_remove("PAMIN_PREPARED");
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("run the child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {arm} child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Report(
        stdout
            .lines()
            // Mid-line: libtest prints the test's name before its output.
            .filter_map(|line| line.split_once(MARK).map(|(_, result)| result))
            .map(|result| {
                let mut fields = result.split_whitespace().map(String::from);
                let key = fields.next().expect("a key");
                (key, fields.collect())
            })
            .collect(),
    )
}

/// The one directory under `copies` not among `known`, which the last child
/// wrote.
fn only_new(copies: &Path, known: &[PathBuf]) -> PathBuf {
    let written: Vec<PathBuf> = std::fs::read_dir(copies)
        .expect("the copies directory")
        .map(|entry| entry.expect("an entry").path())
        .filter(|path| path.is_dir() && !known.contains(path))
        .collect();
    assert_eq!(written.len(), 1, "expected one new copy, found {written:?}");
    written.into_iter().next().expect("one copy")
}

/// Links the models an existing cache holds into `cache`, so nothing is
/// downloaded and nothing is written beside them.
///
/// Not the `fast` reranker's: its export is chosen for the CPU at load, and
/// one missing from a linked directory would be downloaded into the directory
/// it links to.
fn link_models(cache: &Path) {
    let Ok(existing) = std::env::var("PAMIN_TEST_MODELS") else {
        return;
    };
    let wanted = [ACCURATE].map(|repository| format!("models--{}", repository.replace('/', "--")));
    for name in wanted {
        let path = Path::new(&existing).join(&name);
        if path.exists() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&path, cache.join(&name)).expect("link the model");
            #[cfg(windows)]
            std::os::windows::fs::symlink_dir(&path, cache.join(&name)).expect("link the model");
        }
    }
}

/// Hands back to the system what the allocator holds free, so what is left
/// is what is live. See `tests/prepared.rs`.
fn trim() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: `malloc_trim` only returns free memory to the system.
    unsafe {
        libc::malloc_trim(0);
    }
}

/// Anonymous memory this process holds, in kB.
fn anonymous() -> u64 {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .expect("this test reads Linux's smaps_rollup")
        .lines()
        .find_map(|line| line.strip_prefix("Anonymous:"))
        .and_then(|kb| kb.split_whitespace().next()?.parse().ok())
        .expect("an anonymous figure")
}

fn median(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn median_f(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn mebibytes(kb: &[u64]) -> Vec<u64> {
    kb.iter().map(|kb| kb / 1024).collect()
}
