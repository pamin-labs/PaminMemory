//! Drives the actual `pamin` binary through a full lifecycle.
//!
//! Ignored by default: it provisions PostgreSQL and downloads model weights.
//! Run with `cargo test -p pamin-cli -- --ignored`.
//!
//! It invokes the binary rather than calling the library, because two defects
//! that broke every real invocation survived the library tests. The engine's
//! dynamic library is found by the test harness but not by a plainly launched
//! binary, and the engine refuses to create a collection over an existing path,
//! which only shows up on the second command against one workspace. Tests that
//! each get a fresh directory and run inside cargo see neither.

use std::path::Path;
use std::process::Command;

use serde_json::Value;

/// The smallest embedding profile. This test is about the pipeline, not about
/// which model is most accurate.
const PROFILE: &str = "speed";

struct Cli {
    home: tempfile::TempDir,
}

impl Cli {
    fn new() -> Self {
        Self {
            home: tempfile::tempdir().expect("temp home"),
        }
    }

    fn home(&self) -> &Path {
        self.home.path()
    }

    fn run(&self, args: &[&str]) -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_pamin"))
            .args(args)
            .env("PAMIN_HOME", self.home())
            .env("PAMIN_PROFILE", PROFILE)
            .output()
            .expect("running pamin");

        assert!(
            output.status.success(),
            "pamin {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("utf-8 output")
    }

    fn json(&self, args: &[&str]) -> Value {
        let mut with_json = args.to_vec();
        with_json.push("--json");
        serde_json::from_str(&self.run(&with_json)).expect("json output")
    }

    fn fails(&self, args: &[&str]) -> String {
        let output = Command::new(env!("CARGO_BIN_EXE_pamin"))
            .args(args)
            .env("PAMIN_HOME", self.home())
            .env("PAMIN_PROFILE", PROFILE)
            .output()
            .expect("running pamin");
        assert!(!output.status.success(), "expected {args:?} to fail");
        String::from_utf8_lossy(&output.stderr).into_owned()
    }
}

impl Drop for Cli {
    fn drop(&mut self) {
        // Leaves no server behind for the next test to collide with.
        let _ = Command::new(env!("CARGO_BIN_EXE_pamin"))
            .args(["stop"])
            .env("PAMIN_HOME", self.home())
            .output();
    }
}

/// One memory per language, keyed by topic.
const MEMORIES: &[(&str, &str)] = &[
    (
        "deploy_en",
        "the deployment pipeline runs on continuous integration",
    ),
    ("deploy_zh", "部署流水线运行在持续集成上面"),
    (
        "deploy_ja",
        "デプロイパイプラインは東京のサーバーで動いています",
    ),
    ("deploy_th", "ท่อการปรับใช้ทำงานอยู่บนเซิร์ฟเวอร์"),
    ("deploy_ko", "배포 파이프라인은 서울 서버에서 실행됩니다"),
    ("deploy_ar", "خط أنابيب النشر يعمل على خادم بعيد"),
    (
        "deploy_ru",
        "конвейер развёртывания работает на удалённом сервере",
    ),
    (
        "error_ref",
        "see crates/pamin-store/src/database.rs for error E1234 in deploy_service",
    ),
];

/// A query in each language, and the topic it must reach.
const QUERIES: &[(&str, &str)] = &[
    ("continuous integration", "the deployment pipeline"),
    ("流水线", "部署流水线"),
    ("東京", "東京のサーバー"),
    ("ทำงาน", "ท่อการปรับใช้"),
    ("서울", "서울 서버"),
    ("أنابيب", "أنابيب النشر"),
    ("развёртывания", "конвейер развёртывания"),
];

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn a_workspace_serves_memories_in_any_language() {
    let cli = Cli::new();

    cli.run(&["init"]);

    for (topic, content) in MEMORIES {
        cli.run(&["write", "--topic", topic, content]);
    }

    every_language_is_searchable_in_its_own_words(&cli);
    exact_strings_are_found_through_the_ngram_channel(&cli);
    results_explain_where_they_came_from(&cli);
    an_english_query_reaches_memories_written_elsewhere(&cli);
    the_ledger_keeps_history_and_the_filter_keeps_evidence(&cli);
    the_index_rebuilds_from_postgres(&cli);
    a_profile_change_is_refused_rather_than_silently_wrong(&cli);
}

fn contents(results: &Value) -> Vec<String> {
    results["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|hit| hit["content"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn every_language_is_searchable_in_its_own_words(cli: &Cli) {
    for (query, expected) in QUERIES {
        let found = contents(&cli.json(&["search", query, "--limit", "1"]));
        assert!(
            found.iter().any(|content| content.contains(expected)),
            "searching {query:?} should reach {expected:?}, got {found:?}"
        );
    }
}

fn exact_strings_are_found_through_the_ngram_channel(cli: &Cli) {
    // Word segmentation splits these apart, so only the n-gram field can match
    // them. Finding all three is what justifies carrying a second index field.
    for query in ["database.rs", "E1234", "deploy_service"] {
        let results = cli.json(&["search", query, "--limit", "1"]);
        let found = contents(&results);
        assert!(
            found.iter().any(|content| content.contains("E1234")),
            "searching {query:?} should reach the identifier memory, got {found:?}"
        );

        let channels: Vec<&str> = results["hits"][0]["why"]
            .as_array()
            .expect("why array")
            .iter()
            .filter(|why| why["kind"] == "channel")
            .filter_map(|why| why["channel"].as_str())
            .collect();
        assert!(
            channels.contains(&"lexical_ngram"),
            "{query:?} should be credited to the ngram channel, got {channels:?}"
        );
    }
}

fn results_explain_where_they_came_from(cli: &Cli) {
    let results = cli.json(&["search", "deployment pipeline", "--limit", "1"]);
    let hit = &results["hits"][0];
    let why = hit["why"].as_array().expect("why array");

    let channels: Vec<&str> = why
        .iter()
        .filter(|entry| entry["kind"] == "channel")
        .filter_map(|entry| entry["channel"].as_str())
        .collect();
    assert!(
        channels.contains(&"lexical_segmented")
            && channels.contains(&"lexical_ngram")
            && channels.contains(&"vector"),
        "an obvious match should be credited to every channel, got {channels:?}"
    );

    for entry in why.iter().filter(|entry| entry["kind"] == "channel") {
        assert!(
            entry["rank"].as_u64().unwrap_or(0) >= 1,
            "ranks start at one"
        );
    }

    // Each modifier at most once. Applying one twice inflates whatever it
    // measures with nothing in the trace revealing it happened.
    let mut modifiers: Vec<&str> = why
        .iter()
        .filter(|entry| entry["kind"] == "modifier")
        .filter_map(|entry| entry["modifier"].as_str())
        .collect();
    let applied = modifiers.len();
    modifiers.sort_unstable();
    modifiers.dedup();
    assert_eq!(applied, modifiers.len(), "a modifier was applied twice");

    assert!(
        !hit["source_span"].as_str().unwrap_or_default().is_empty(),
        "every hit must cite the span it came from"
    );
}

fn an_english_query_reaches_memories_written_elsewhere(cli: &Cli) {
    let results = cli.json(&["search", "how is the code deployed", "--limit", "8"]);
    let hits = results["hits"].as_array().expect("hits");

    // Lexical channels cannot cross scripts; the multilingual embedding can.
    // Reaching these proves cross-language recall without translating anything.
    let reached: Vec<_> = hits
        .iter()
        .filter(|hit| {
            let content = hit["content"].as_str().unwrap_or_default();
            content.contains("서울") || content.contains("خادم") || content.contains("流水线")
        })
        .collect();
    assert!(
        !reached.is_empty(),
        "an english query should reach non-latin memories: {:?}",
        contents(&results)
    );

    for hit in reached {
        let channels: Vec<&str> = hit["why"]
            .as_array()
            .expect("why")
            .iter()
            .filter(|why| why["kind"] == "channel")
            .filter_map(|why| why["channel"].as_str())
            .collect();
        assert!(
            channels.contains(&"vector"),
            "cross-language recall must come from the vector channel, got {channels:?}"
        );
    }
}

fn the_ledger_keeps_history_and_the_filter_keeps_evidence(cli: &Cli) {
    cli.run(&[
        "write",
        "--topic",
        "deploy_en",
        "the deployment pipeline now runs on argo",
    ]);

    let current = cli.json(&["read", "deploy_en"]);
    assert_eq!(current["version"], 2);
    assert_eq!(current["is_current"], true);

    let previous = cli.json(&["read", "deploy_en", "--version-offset", "1"]);
    assert_eq!(previous["version"], 1);
    assert_eq!(previous["is_current"], false);

    // Past the oldest version, resolution clamps and says how far it reached.
    let clamped = cli.json(&["read", "deploy_en", "--version-offset", "9"]);
    assert_eq!(clamped["version"], 1);
    assert_eq!(clamped["actual_version_offset"], 1);

    // Content the filter holds is still recorded as evidence, with a reason,
    // and simply does not become a topic state.
    let noise = cli.json(&["write", "--topic", "deploy_en", "ok"]);
    assert_eq!(noise["promoted"], false);
    assert!(noise["version"].is_null());
    assert!(!noise["reason"].as_str().unwrap_or_default().is_empty());
    assert!(noise["source_version"].as_u64().unwrap_or(0) > 0);

    assert_eq!(
        cli.json(&["read", "deploy_en"])["version"],
        2,
        "a held write must not advance the topic"
    );
}

fn the_index_rebuilds_from_postgres(cli: &Cli) {
    let before = contents(&cli.json(&["search", "流水线", "--limit", "1"]));

    std::fs::remove_dir_all(cli.home().join("index")).expect("delete the index");

    let rebuilt = cli.json(&["reindex"]);
    assert!(rebuilt["indexed"].as_u64().unwrap_or(0) >= MEMORIES.len() as u64);

    let after = contents(&cli.json(&["search", "流水线", "--limit", "1"]));
    assert_eq!(
        before, after,
        "the projection holds nothing postgres cannot reproduce"
    );
}

fn a_profile_change_is_refused_rather_than_silently_wrong(cli: &Cli) {
    let output = Command::new(env!("CARGO_BIN_EXE_pamin"))
        .args(["--profile", "balanced", "search", "流水线"])
        .env("PAMIN_HOME", cli.home())
        .output()
        .expect("running pamin");

    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("reindex"),
        "a profile mismatch should say how to fix it, got {error:?}"
    );
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn an_unknown_profile_is_rejected_before_anything_is_provisioned() {
    let cli = Cli::new();
    let error = cli.fails(&["--profile", "enormous", "init"]);
    assert!(error.contains("enormous"), "got {error:?}");
}
