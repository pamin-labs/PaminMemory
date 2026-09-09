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
    a_write_can_state_when_its_claim_holds(&cli);
    the_index_rebuilds_from_postgres(&cli);
    a_profile_change_is_refused_rather_than_silently_wrong(&cli);
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn the_graph_channel_reaches_what_nothing_else_can() {
    let cli = Cli::new();
    cli.run(&["init"]);

    // Deliberately disjoint: no shared vocabulary, no semantic proximity.
    // Anything that finds the second from a query about the first came
    // through the graph.
    cli.run(&[
        "write",
        "--topic",
        "release_process",
        "the release process cuts a tag and publishes artifacts",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "office_plants",
        "the ficus by the window needs watering on thursdays",
    ]);

    edges_are_derived_from_names_without_being_asked(&cli);
    a_new_topic_is_linked_to_memories_that_already_named_it(&cli);
    derivation_works_in_languages_written_without_spaces(&cli);
    an_explicit_link_makes_the_unreachable_reachable(&cli);
    the_graph_credits_nothing_the_other_channels_already_found(&cli);
    a_result_is_never_its_own_explanation(&cli);
    an_edge_can_be_bounded_in_time(&cli);
    a_query_naming_a_topic_walks_out_from_it(&cli);
    a_retraction_says_whether_the_claim_ended_or_was_wrong(&cli);
    retracting_a_link_keeps_the_record_and_drops_the_result(&cli);
    edges_survive_the_projection_being_destroyed(&cli);
}

fn neighbor_topics(results: &Value) -> Vec<String> {
    results["neighbors"]
        .as_array()
        .expect("neighbors array")
        .iter()
        .map(|neighbor| neighbor["topic"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn edges_are_derived_from_names_without_being_asked(cli: &Cli) {
    // No link command is run anywhere in this function. The edge exists
    // because the content names the topic, which is the whole claim.
    cli.run(&[
        "write",
        "--topic",
        "hotfix_process",
        "a hotfix skips the release process and ships straight from main",
    ]);

    let neighbors = cli.json(&["neighbors", "hotfix_process", "--depth", "1"]);
    let found = neighbor_topics(&neighbors);
    assert!(
        found.contains(&"release_process".to_string()),
        "naming a topic should derive an edge to it, got {found:?}"
    );

    let derived = neighbors["neighbors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|neighbor| neighbor["topic"] == "release_process")
        .expect("the derived edge");
    assert_eq!(derived["derivation"], "deterministic");
    assert_eq!(derived["edge"], "mentions");

    // Rewriting unchanged content must not stack a second edge version.
    let before = neighbor_topics(&cli.json(&["neighbors", "hotfix_process", "--depth", "1"]));
    cli.run(&[
        "write",
        "--topic",
        "hotfix_process",
        "a hotfix skips the release process and ships straight from main today",
    ]);
    let after = neighbor_topics(&cli.json(&["neighbors", "hotfix_process", "--depth", "1"]));
    assert_eq!(before, after, "re-deriving the same edge changes nothing");
}

fn a_new_topic_is_linked_to_memories_that_already_named_it(cli: &Cli) {
    // Written before any topic of that name exists. Without a backfill the
    // edge would appear only if this memory happened to be rewritten later.
    cli.run(&[
        "write",
        "--topic",
        "deploy_notes",
        "everything goes out through argo_cd now",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "argo_cd",
        "argo cd runs in the tools cluster",
    ]);

    let found = neighbor_topics(&cli.json(&["neighbors", "argo_cd", "--depth", "1"]));
    assert!(
        found.contains(&"deploy_notes".to_string()),
        "creating a topic should link memories that already named it, got {found:?}"
    );
}

fn derivation_works_in_languages_written_without_spaces(cli: &Cli) {
    // Nothing here is space-delimited, so a match is only possible because
    // both the name and the content pass through the same segmenter.
    cli.run(&["write", "--topic", "流水线", "流水线运行在持续集成上面"]);
    cli.run(&["write", "--topic", "回滚", "回滚会绕过流水线直接发布"]);

    let found = neighbor_topics(&cli.json(&["neighbors", "回滚", "--depth", "1"]));
    assert!(
        found.contains(&"流水线".to_string()),
        "name derivation must not depend on spaces, got {found:?}"
    );
}

/// The hit whose content contains `needle`, if the search returned one.
fn hit_containing<'a>(results: &'a Value, needle: &str) -> Option<&'a Value> {
    results["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .find(|hit| hit["content"].as_str().unwrap_or_default().contains(needle))
}

/// The channels credited for one hit.
fn credited_channels(hit: &Value) -> Vec<String> {
    hit["why"]
        .as_array()
        .expect("why")
        .iter()
        .filter(|entry| entry["kind"] == "channel")
        .filter_map(|entry| entry["channel"].as_str())
        .map(str::to_string)
        .collect()
}

fn an_explicit_link_makes_the_unreachable_reachable(cli: &Cli) {
    let query = "how do we ship a release";

    // A workspace this small lets the vector channel dredge up everything, so
    // the claim under test is not "it appears" but "the graph is why it
    // appears". That is also the claim that stays true at any corpus size.
    let before = cli.json(&["search", query, "--limit", "8"]);
    if let Some(hit) = hit_containing(&before, "ficus") {
        assert!(
            !credited_channels(hit).contains(&"graph".to_string()),
            "nothing connects the plant memory yet"
        );
    }

    cli.run(&[
        "link",
        "release_process",
        "office_plants",
        "--kind",
        "related_to",
    ]);

    let after = cli.json(&["search", query, "--limit", "8"]);
    let hit = hit_containing(&after, "ficus")
        .unwrap_or_else(|| panic!("the graph should reach it: {:?}", contents(&after)));
    assert!(
        credited_channels(hit).contains(&"graph".to_string()),
        "the graph channel must be credited, got {:?}",
        credited_channels(hit)
    );

    let path = hit["why"]
        .as_array()
        .expect("why")
        .iter()
        .find(|entry| entry["kind"] == "path")
        .expect("a graph hit explains the edge it came through");
    assert_eq!(path["via"], "release_process");
    assert_eq!(path["hops"], 1);
    assert_eq!(path["edge"], "related_to");
    assert_eq!(path["derivation"], "explicit");
}

fn the_graph_credits_nothing_the_other_channels_already_found(cli: &Cli) {
    let results = cli.json(&["search", "the release process cuts a tag", "--limit", "8"]);

    for hit in results["hits"].as_array().expect("hits") {
        let why = hit["why"].as_array().expect("why");

        let graph_entries = why
            .iter()
            .filter(|entry| entry["kind"] == "channel" && entry["channel"] == "graph")
            .count();
        assert!(
            graph_entries <= 1,
            "one graph rank per result at most, got {graph_entries}"
        );

        let paths = why.iter().filter(|entry| entry["kind"] == "path").count();
        assert_eq!(
            paths, graph_entries,
            "a graph rank comes with its path and nothing else does"
        );
    }
}

fn a_result_is_never_its_own_explanation(cli: &Cli) {
    // Two topics that share no vocabulary with each other, linked. Both are
    // seeds of this query in a workspace this small, so each is genuinely
    // reachable from the other and both may be credited to the graph — that
    // is agreement between candidates, not double counting.
    //
    // What must never happen is a result being reached from itself. Traversal
    // ignores direction, so without the no-backtracking rule every seed would
    // arrive back at itself two hops along the edge it just took, and the
    // explanation would name the result it is explaining.
    cli.run(&[
        "write",
        "--topic",
        "zither_tuning",
        "the zither is tuned in fourths before every recital",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "kiln_firing",
        "the kiln reaches cone ten overnight",
    ]);
    cli.run(&[
        "link",
        "zither_tuning",
        "kiln_firing",
        "--kind",
        "related_to",
    ]);

    let results = cli.json(&[
        "search",
        "tuned in fourths before a recital",
        "--limit",
        "8",
    ]);

    let reached = hit_containing(&results, "kiln").expect("the neighbour is reachable");
    assert!(
        credited_channels(reached).contains(&"graph".to_string()),
        "a topic on the far side of an edge is a genuine graph result"
    );

    for hit in results["hits"].as_array().expect("hits") {
        let topic = hit["topic"].as_str().expect("every hit names its topic");
        for path in hit["why"]
            .as_array()
            .expect("why")
            .iter()
            .filter(|entry| entry["kind"] == "path")
        {
            assert_ne!(
                path["via"], topic,
                "a result reached from itself explains nothing"
            );
            assert!(
                path["hops"].as_u64().unwrap_or(0) >= 1,
                "a path always crosses at least one edge"
            );
        }
    }
}

fn an_edge_can_be_bounded_in_time(cli: &Cli) {
    // Truth validity travels through the CLI as RFC 3339 on both sides, so a
    // timestamp the CLI prints can be handed straight back to it. The store
    // covers the filtering itself; what is checked here is the plumbing that
    // carries a bound from an argument into the query, which is where a
    // regression would be silent.
    // Neither memory names the other topic. If one did, derivation would add
    // an unbounded `mentions` edge alongside the bounded one and the temporal
    // filter would have nothing to prove.
    cli.run(&[
        "write",
        "--topic",
        "ledger_migration",
        "the rewrite ran against every account in one pass",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "old_schema",
        "balances used to live in a single wide table",
    ]);
    cli.run(&[
        "link",
        "ledger_migration",
        "old_schema",
        "--kind",
        "depends_on",
        "--valid-from",
        "2020-01-01T00:00:00Z",
        "--valid-to",
        "2021-01-01T00:00:00Z",
    ]);

    let inside = cli.json(&[
        "neighbors",
        "ledger_migration",
        "--depth",
        "1",
        "--at",
        "2020-06-01T00:00:00Z",
    ]);
    assert!(
        neighbor_topics(&inside).contains(&"old_schema".to_string()),
        "the edge holds inside its own interval: {:?}",
        neighbor_topics(&inside)
    );

    let outside = cli.json(&[
        "neighbors",
        "ledger_migration",
        "--depth",
        "1",
        "--at",
        "2026-06-01T00:00:00Z",
    ]);
    assert!(
        outside["neighbors"]
            .as_array()
            .expect("neighbors")
            .is_empty(),
        "a dependency asserted only for 2020 does not hold now: {:?}",
        neighbor_topics(&outside)
    );

    // Without --at the question is "what do we still stand behind", which the
    // edge answers regardless of when it is asserted to hold.
    assert!(
        neighbor_topics(&cli.json(&["neighbors", "ledger_migration", "--depth", "1"]))
            .contains(&"old_schema".to_string()),
        "an unretracted edge is live whatever its truth interval says"
    );

    // Malformed and impossible bounds are refused rather than silently
    // reinterpreted, which is the only way a caller learns it got them wrong.
    let error = cli.fails(&["neighbors", "ledger_migration", "--at", "yesterday"]);
    assert!(error.contains("RFC 3339"), "got {error:?}");

    let error = cli.fails(&[
        "link",
        "old_schema",
        "ledger_migration",
        "--kind",
        "depends_on",
        "--valid-from",
        "2021-01-01T00:00:00Z",
        "--valid-to",
        "2020-01-01T00:00:00Z",
    ]);
    assert!(error.contains("--valid-to must be after"), "got {error:?}");
}

fn a_query_naming_a_topic_walks_out_from_it(cli: &Cli) {
    // The seeded topic's own content shares nothing with the query, so the
    // lexical and vector channels cannot reach it. Only matching the query
    // text against known topic names can, which is the retrieval half of
    // entity linking. Without it, asking about a topic by name never walks
    // out from that topic.
    cli.run(&[
        "write",
        "--topic",
        "sourdough_starter",
        "feed it twice a day and keep it warm",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "rye_flour",
        "coarse ground, from the mill on the north road",
    ]);
    cli.run(&[
        "link",
        "sourdough_starter",
        "rye_flour",
        "--kind",
        "depends_on",
    ]);

    let results = cli.json(&["search", "what does sourdough_starter need", "--limit", "8"]);
    let reached = hit_containing(&results, "coarse ground").unwrap_or_else(|| {
        panic!(
            "naming a topic should walk out from it: {:?}",
            contents(&results)
        )
    });
    assert!(
        credited_channels(reached).contains(&"graph".to_string()),
        "and the graph is what reached it, got {:?}",
        credited_channels(reached)
    );

    let path = reached["why"]
        .as_array()
        .expect("why")
        .iter()
        .find(|entry| entry["kind"] == "path")
        .expect("a graph hit explains itself");
    assert_eq!(
        path["from"], "sourdough_starter",
        "the trace names the topic the walk began from"
    );
}

fn a_retraction_says_whether_the_claim_ended_or_was_wrong(cli: &Cli) {
    cli.run(&[
        "write",
        "--topic",
        "tram_line",
        "the tram runs every eleven minutes",
    ]);
    cli.run(&[
        "write",
        "--topic",
        "bus_line",
        "the bus runs every twenty minutes",
    ]);
    cli.run(&["link", "tram_line", "bus_line", "--kind", "related_to"]);

    let held = cli.json(&["neighbors", "tram_line", "--depth", "1"]);
    assert!(neighbor_topics(&held).contains(&"bus_line".to_string()));

    // Default is `closed`: the relationship ended. It stops being asserted now
    // and stays answerable for the time it did hold.
    let retracted = cli.json(&["unlink", "tram_line", "bus_line", "--kind", "related_to"]);
    assert_eq!(retracted["closed"], true);
    assert_eq!(retracted["reason"], "closed");

    assert!(
        neighbor_topics(&cli.json(&["neighbors", "tram_line", "--depth", "1"])).is_empty(),
        "a retracted relationship is not asserted now"
    );
    assert!(
        neighbor_topics(&cli.json(&[
            "neighbors",
            "tram_line",
            "--depth",
            "1",
            "--at",
            "2020-01-01T00:00:00Z",
        ]))
        .contains(&"bus_line".to_string()),
        "but it did hold before it ended, and that question has an answer"
    );

    // `deleted` says the claim was wrong, so no instant finds it.
    cli.run(&["link", "tram_line", "bus_line", "--kind", "same_as"]);
    cli.run(&[
        "unlink",
        "tram_line",
        "bus_line",
        "--kind",
        "same_as",
        "--reason",
        "deleted",
    ]);
    let historical = cli.json(&[
        "neighbors",
        "tram_line",
        "--depth",
        "1",
        "--kind",
        "same_as",
        "--at",
        "2020-01-01T00:00:00Z",
    ]);
    assert!(
        neighbor_topics(&historical).is_empty(),
        "a claim retracted as wrong never held: {:?}",
        neighbor_topics(&historical)
    );
}

fn retracting_a_link_keeps_the_record_and_drops_the_result(cli: &Cli) {
    let retracted = cli.json(&[
        "unlink",
        "release_process",
        "office_plants",
        "--kind",
        "related_to",
    ]);
    assert_eq!(retracted["closed"], true);

    let after = cli.json(&["search", "how do we ship a release", "--limit", "8"]);
    if let Some(hit) = hit_containing(&after, "ficus") {
        assert!(
            !credited_channels(hit).contains(&"graph".to_string()),
            "a retracted edge stops feeding recall"
        );
    }
    assert!(
        neighbor_topics(&cli.json(&["neighbors", "release_process", "--depth", "2"]))
            .iter()
            .all(|topic| topic != "office_plants"),
        "a retracted edge is not traversed"
    );

    // Retracting again reports that nothing was open, rather than pretending.
    let again = cli.json(&[
        "unlink",
        "release_process",
        "office_plants",
        "--kind",
        "related_to",
    ]);
    assert_eq!(again["closed"], false);
}

fn edges_survive_the_projection_being_destroyed(cli: &Cli) {
    cli.run(&[
        "link",
        "release_process",
        "office_plants",
        "--kind",
        "related_to",
    ]);

    std::fs::remove_dir_all(cli.home().join("index")).expect("delete the index");
    cli.run(&["reindex"]);

    let after = cli.json(&["search", "how do we ship a release", "--limit", "8"]);
    let hit = hit_containing(&after, "ficus")
        .unwrap_or_else(|| panic!("still reachable: {:?}", contents(&after)));
    assert!(
        credited_channels(hit).contains(&"graph".to_string()),
        "the graph lives in postgres, so a rebuilt projection changes nothing"
    );
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

    // Nor may it bring a topic into existence. The topic used to be created
    // before the filter ran, so a held write to a name nobody had used left an
    // empty topic behind: a name `neighbors` and the graph channel could reach
    // and that resolved to no content at all.
    let invented = cli.json(&["write", "--topic", "never_promoted", "ok"]);
    assert_eq!(invented["promoted"], false);

    let missing = cli.fails(&["read", "never_promoted"]);
    assert!(
        missing.contains("never_promoted"),
        "a topic only ever written to under the filter should not exist: {missing:?}"
    );
}

fn a_write_can_state_when_its_claim_holds(cli: &Cli) {
    // The columns existed from the first migration and nothing could write
    // them, so the bi-temporal ledger was half a ledger: relationships could
    // say when they hold and the facts they connect could not.
    let written = cli.json(&[
        "write",
        "--topic",
        "winter_timetable",
        "the winter timetable adds two evening departures",
        "--valid-from",
        "2025-11-01T00:00:00Z",
        "--valid-to",
        "2026-03-01T00:00:00Z",
    ]);
    assert_eq!(written["promoted"], true);
    assert_eq!(written["valid_from"], "2025-11-01T00:00:00Z");
    assert_eq!(written["valid_to"], "2026-03-01T00:00:00Z");

    let error = cli.fails(&[
        "write",
        "--topic",
        "winter_timetable",
        "an interval that runs backwards",
        "--valid-from",
        "2026-03-01T00:00:00Z",
        "--valid-to",
        "2025-11-01T00:00:00Z",
    ]);
    assert!(error.contains("--valid-to must be after"), "got {error:?}");
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

/// A walk deeper than the graph channel goes is refused, not quietly reduced.
///
/// `--depth` was a bare `u8`, so 255 parsed. A topic's neighbourhood grows
/// multiplicatively per hop and hub topics reach five figures of degree, so
/// what that accepted was a request no project could answer. Refusing it at the
/// boundary is what tells the operator their number was not honoured; clamping
/// silently leaves them wondering why the walk stopped at four.
///
/// Not ignored, and it needs no workspace: the argument is rejected during
/// parsing, before the command reaches anything it would have to provision.
#[test]
fn a_walk_deeper_than_the_graph_channel_goes_is_refused() {
    let home = tempfile::tempdir().expect("temp home");

    for args in [
        ["neighbors", "deploy", "--depth", "255"],
        ["search", "deploy", "--graph-depth", "255"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_pamin"))
            .args(args)
            .env("PAMIN_HOME", home.path())
            .output()
            .expect("running pamin");

        assert!(!output.status.success(), "{args:?} should be refused");

        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("255"),
            "{args:?} refused without saying what was wrong: {error:?}"
        );
    }

    assert!(
        std::fs::read_dir(home.path())
            .expect("temp home")
            .next()
            .is_none(),
        "a refused argument should not have provisioned a workspace"
    );
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn an_unknown_profile_is_rejected_before_anything_is_provisioned() {
    let cli = Cli::new();
    let error = cli.fails(&["--profile", "enormous", "init"]);
    assert!(error.contains("enormous"), "got {error:?}");
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn grep_reaches_evidence_that_search_cannot() {
    let cli = Cli::new();
    cli.run(&["init"]);

    cli.run(&[
        "write",
        "--topic",
        "incident_log",
        "the checkout service returned E5521 during the tuesday outage",
    ]);

    // Held by the filter: too short to carry a durable claim. It never became
    // a topic state and never entered the index.
    let held = cli.json(&["write", "--topic", "incident_log", "E5521"]);
    assert_eq!(held["promoted"], false);

    // Search cannot see it. That is correct behaviour, and it is also why grep
    // has to exist: a filtering mistake is only recoverable if the content is
    // reachable by some route.
    let searched = cli.json(&["search", "E5521", "--limit", "5"]);
    assert!(
        !contents(&searched)
            .iter()
            .any(|content| content.trim() == "E5521"),
        "filtered content stays off the retrieval surface: {:?}",
        contents(&searched)
    );

    let grepped = cli.json(&["grep", "E5521"]);
    let matches = grepped["matches"].as_array().expect("matches");
    assert!(
        matches
            .iter()
            .any(|hit| hit["filter_decision"] == "filtered"),
        "grep reaches what the filter held: {matches:?}"
    );
    assert!(
        matches
            .iter()
            .any(|hit| hit["filter_decision"] == "promoted"),
        "and what it promoted"
    );
    for hit in matches {
        assert!(
            !hit["filter_reason"].as_str().unwrap_or_default().is_empty(),
            "every match says why it was or was not promoted"
        );
    }

    // Any language, because nothing tokenizes here.
    cli.run(&[
        "write",
        "--topic",
        "deploy_zh",
        "部署流水线运行在持续集成上面",
    ]);
    assert!(
        !cli.json(&["grep", "持续集成"])["matches"]
            .as_array()
            .expect("matches")
            .is_empty(),
        "a literal search needs no tokenizer"
    );

    // Case sensitivity is the caller's choice.
    assert!(
        cli.json(&["grep", "e5521"])["matches"]
            .as_array()
            .expect("matches")
            .is_empty(),
        "matching is case sensitive by default"
    );
    assert!(
        !cli.json(&["grep", "e5521", "-i"])["matches"]
            .as_array()
            .expect("matches")
            .is_empty(),
        "-i folds case"
    );

    // Superseded content stays reachable, which is what makes this an audit
    // route rather than a second view of the current state.
    cli.run(&[
        "write",
        "--topic",
        "incident_log",
        "the checkout service was fixed on wednesday",
    ]);
    assert!(
        !cli.json(&["grep", "tuesday outage"])["matches"]
            .as_array()
            .expect("matches")
            .is_empty(),
        "the superseded version is still in evidence"
    );
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn one_project_cannot_crowd_another_out_of_its_own_index() {
    let cli = Cli::new();
    cli.run(&["init"]);

    // Results were always filtered to the calling project, so leakage was
    // never the symptom. Depth was: with one shared collection, every channel
    // spent its candidate budget across every project, and whatever belonged
    // to another one was fetched and then thrown away. A busy neighbour could
    // therefore consume a project's entire budget and leave it with nothing.
    //
    // Asking for a depth of one makes that measurable instead of statistical.
    for (index, note) in [
        "the release checklist covers the release checklist steps",
        "the release checklist is reviewed before every release",
        "a release checklist entry blocks the release",
        "release checklist ownership rotates with the release",
        "the release checklist lives beside the release notes",
    ]
    .iter()
    .enumerate()
    {
        cli.run(&[
            "--project",
            "crowded",
            "write",
            "--topic",
            &format!("crowded_{index}"),
            note,
        ]);
    }

    cli.run(&[
        "--project",
        "quiet",
        "write",
        "--topic",
        "quiet_topic",
        "the release checklist is short here",
    ]);

    let hits = cli.json(&[
        "--project",
        "quiet",
        "search",
        "release checklist",
        "--limit",
        "8",
        "--channel-depth",
        "1",
    ]);
    let found = contents(&hits);
    assert_eq!(
        found.len(),
        1,
        "one candidate per channel must be this project's own, got {found:?}"
    );
    assert!(found[0].contains("short here"));

    // Rebuilding one project leaves the other alone, which holds because they
    // are separate directories rather than one index filtered after the fact.
    cli.run(&["--project", "quiet", "reindex"]);
    let crowded = contents(&cli.json(&[
        "--project",
        "crowded",
        "search",
        "release checklist",
        "--limit",
        "8",
    ]));
    assert!(
        crowded.len() >= 5,
        "rebuilding one project does not disturb another: {crowded:?}"
    );
}

#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn a_pre_split_workspace_is_migrated_by_reindexing() {
    // Every workspace that existed before projects had their own directory
    // takes this path exactly once, so it is worth more than a unit test on
    // the guard: what matters is that the whole route works, from the refusal
    // through the instruction it gives to the state it leaves behind.
    let cli = Cli::new();
    cli.run(&["init"]);
    cli.run(&[
        "write",
        "--topic",
        "kept_memory",
        "a durable claim that has to survive the migration",
    ]);

    // The shape a workspace had before the split: one shared collection
    // directly under the index directory.
    std::fs::create_dir_all(cli.home().join("index").join("memories"))
        .expect("simulate the old layout");

    let error = cli.fails(&["search", "durable claim"]);
    assert!(
        error.contains("reindex"),
        "refusing is only useful if it says what to run, got {error:?}"
    );

    cli.run(&["reindex"]);
    assert!(
        !cli.home().join("index").join("memories").exists(),
        "reindexing is the migration, so it clears the old layout"
    );

    let found = contents(&cli.json(&["search", "durable claim", "--limit", "1"]));
    assert!(
        found
            .iter()
            .any(|content| content.contains("has to survive")),
        "and the memories are all still there: {found:?}"
    );
}

/// A memory outlives the process that was supposed to index it.
///
/// This is the property the outbox exists for, and it is the one that is
/// invisible from a passing test: draining after the write and never queueing
/// at all look identical once both have finished. What tells them apart is a
/// process ending between the two, and `--defer` is the only way to reach that
/// moment on purpose -- the ordinary write path always drains before it
/// returns, so the gap does not exist to observe.
///
/// Every step is a separate `pamin` process. The one that recorded the memory
/// is gone before the one that indexes it starts, which is exactly the crash
/// this is about: nothing in memory survives, and the queue is what carries
/// the work across.
#[test]
#[ignore = "provisions postgres and downloads model weights"]
fn work_a_write_left_behind_outlives_the_process_that_left_it() {
    let cli = Cli::new();
    cli.run(&["init"]);

    let written = cli.json(&[
        "write",
        "--defer",
        "--topic",
        "deferred_memory",
        "a claim recorded by a process that never indexed it",
    ]);
    assert_eq!(
        written["cascade"], "queued",
        "a deferred write should say the work is still owed: {written}"
    );
    assert_eq!(
        written["promoted"], true,
        "deferring changes when the index catches up, not what is recorded: {written}"
    );

    // The ledger has it -- `read` never goes through the index -- and the
    // retrieval surface does not yet.
    let stored = cli.json(&["read", "deferred_memory"]);
    assert!(
        stored["content"]
            .as_str()
            .unwrap_or_default()
            .contains("never indexed it"),
        "the memory is committed whatever the index knows: {stored}"
    );
    let found = contents(&cli.json(&["search", "recorded by a process", "--limit", "5"]));
    assert!(
        !found
            .iter()
            .any(|content| content.contains("never indexed")),
        "a deferred write should not be searchable before the cascade runs: {found:?}"
    );

    // Owed, not lost and not failed. Without this the test would also pass if
    // the write had silently dropped the work.
    let owed = cli.json(&["cascade", "failed"]);
    assert_eq!(
        owed["failed"].as_array().expect("failed array").len(),
        0,
        "nothing should have failed, only waited: {owed}"
    );

    let drained = cli.json(&["cascade", "drain"]);
    assert!(
        drained["completed"].as_u64().unwrap_or_default() > 0,
        "a drain should find the work the write left: {drained}"
    );
    assert_eq!(
        drained["pending"], 0,
        "and should leave nothing owed: {drained}"
    );

    let found = contents(&cli.json(&["search", "recorded by a process", "--limit", "5"]));
    assert!(
        found
            .iter()
            .any(|content| content.contains("never indexed")),
        "after the drain the memory is on the retrieval surface: {found:?}"
    );
}
