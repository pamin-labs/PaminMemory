//! What one write costs when the project already holds a lot of topics.
//!
//! `derive_mentions` used to ask the question topic by topic: it segmented
//! every name in the project on every write, so the thousandth topic made the
//! thousandth write a thousand segmentations long. It now asks the index
//! instead -- every run of tokens the *memory* contains, looked up in
//! `topic_name_tokens` -- and the cost is supposed to follow the length of
//! what was written rather than the size of what is stored.
//!
//! "Supposed to" is the whole reason this file exists. The change landed with
//! no test that would notice it being undone, and the shape it protects is
//! invisible on a project small enough for an ordinary test: at ten topics the
//! old loop and the new lookup cost the same.
//!
//! So the measurement is a ratio and not a stopwatch reading. A wall-clock
//! ceiling on a shared CI runner is a flaky gate, and a flaky gate teaches
//! people to ignore gates. What cannot be explained away is the same work
//! costing ten times more because the project around it grew ten times: that
//! is the defect, and nothing else in this path has that shape.
//!
//! Topics are inserted in bulk rather than through `ensure_topic`, because
//! building the project is setup and the API costs several round trips per
//! topic.
//!
//! The end-to-end write and its cascade are timed and printed, because that is
//! the wait a caller actually pays. The *assertion* is on `derive_mentions`
//! alone, and the reason is that the end-to-end number cannot carry it: the
//! drain is a hundred milliseconds of embedding, and a defect worth a hundred
//! more would move the total by two. Measured that way a bound loose enough to
//! survive a shared runner is also loose enough to let the defect back in --
//! which was true of the first version of this file, and is why it says this.
//! So the ratio is taken on the operation that had the problem, where the old
//! behaviour is ten times and the new one is flat -- and on the median of
//! fifty of them, because one reading of a two-millisecond operation is noise
//! wearing a number's clothes.
//!
//! Run it with a workspace that already has PostgreSQL provisioned:
//!
//! ```bash
//! PAMIN_EVAL_HOME=/somewhere \
//!   cargo test -p pamin-engine --test writecost -- --ignored --nocapture
//! ```

use std::time::Instant;

use pamin_core::{FilterDecision, Validity};
use pamin_engine::{Engine, Owed, Write};
use pamin_index::{Access, Profile};
use pamin_store::{Connections, Database, Workspace, repository};

/// Ten times apart, because the defect is a factor and not an offset.
const SCALES: [usize; 2] = [2_000, 20_000];

/// What the ratio may not exceed. Ten would be the old behaviour exactly; this
/// leaves room for the parts that genuinely do grow -- one more index page,
/// one deeper b-tree, a `MAX()` over ten times the rows -- and still fails
/// long before linear.
const AT_MOST: f64 = 3.0;

/// The smallest profile. This measures a code path, not a model.
const PROFILE: &str = "speed";

/// How many derivations each scale is timed over.
///
/// One is not a measurement. The derivation costs about a millisecond, and a
/// single reading of that on a machine doing anything else swings by more than
/// the effect: two runs of the one-shot version of this test gave 1.65x and
/// 3.17x for the same unchanged code, which is a gate that fails honest work
/// and teaches people to rerun until it is green.
const REPEATS: usize = 50;

/// Widths the second test walks, at a fixed project size.
///
/// The widest name in a project is the axis the first test holds still, and it
/// is the one an Aho-Corasick automaton would flatten: `runs_of_tokens` builds
/// every window of every width up to it, so the lookup carries `n * w` strings
/// where an automaton would carry the text once.
const WIDTHS: [usize; 3] = [3, 8, 16];

/// Long enough that the run count is realistic, and long enough to contain
/// the whole of the widest name this file builds. The other test's memory is
/// fourteen tokens, which at width three is thirty-nine runs; a real memory is
/// several hundred.
const PROSE: &str = "the incident review points at service 7 pipeline alpha \
    bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike \
    and nothing else in the fleet, though the oncall rota was paged twice \
    that evening and the rollback plan was read out in full before anybody \
    agreed to touch the staging cluster or the build cache";

/// `width` tokens each, so `widest_topic_name` is whatever the caller asked
/// for: a project whose names are all one token asks the index for one run per
/// position and never exercises the widening the real one pays for.
///
/// The first three tokens are the name the other test writes about, so width
/// three reproduces that project exactly and every wider one is a prefix of
/// `PROSE` too -- which is what keeps the derived edge count at one across the
/// whole sweep, and so keeps every row measuring the same work.
fn names(count: usize, width: usize) -> Vec<String> {
    const TAIL: [&str; 14] = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
        "juliet", "kilo", "lima", "mike", "november",
    ];
    assert!(
        (3..=TAIL.len() + 3).contains(&width),
        "width {width} is unbuildable"
    );
    (0..count)
        .map(|i| {
            let mut parts = vec!["service".to_string(), i.to_string(), "pipeline".to_string()];
            parts.extend(TAIL[..width - 3].iter().map(|t| t.to_string()));
            parts.join(" ")
        })
        .collect()
}

/// How many token runs a memory of `tokens` words implies at this width.
///
/// Printed beside the timing because it is the quantity under test: a cost
/// that grows with it is the loop an automaton replaces, and a cost that does
/// not is the batched lookup doing its job.
fn runs(tokens: usize, width: usize) -> usize {
    (1..=width.min(tokens)).map(|w| tokens - w + 1).sum()
}

/// Inserts `count` topics and the name rows the lookup reads, in two statements.
async fn fill(database: &Database, project: pamin_core::ProjectId, count: usize, width: usize) {
    let names = names(count, width);
    let ids: Vec<uuid::Uuid> = (0..count).map(|_| uuid::Uuid::new_v4()).collect();
    let now = time::OffsetDateTime::now_utc();

    sqlx::query(
        "INSERT INTO topics (id, project_id, name, created_at)
         SELECT id, $2, name, $4 FROM unnest($1::uuid[], $3::text[]) AS t (id, name)",
    )
    .bind(&ids)
    .bind(project.0)
    .bind(&names)
    .bind(now)
    .execute(database.pool())
    .await
    .expect("insert topics");

    // The key is the segmented name joined by spaces, which for these is the
    // name itself. Written directly for the same reason the topics are: this
    // is setup.
    let widths: Vec<i16> = (0..count).map(|_| width as i16).collect();
    sqlx::query(
        "INSERT INTO topic_name_tokens (project_id, topic_id, name_key, token_count)
         SELECT $1, id, key, width
           FROM unnest($2::uuid[], $3::text[], $4::int2[]) AS t (id, key, width)",
    )
    .bind(project.0)
    .bind(&ids)
    .bind(&names)
    .bind(&widths)
    .execute(database.pool())
    .await
    .expect("insert topic names");

    // What autovacuum would have done for a project that grew the ordinary
    // way. Without it the planner has no statistics for a table that went from
    // empty to twenty thousand rows in one statement, and picks a sequential
    // scan for the `MAX(token_count)` that has an index for exactly that --
    // which made the derivation look like it still scaled with the project
    // when what scaled was the test's own setup.
    sqlx::query("ANALYZE topic_name_tokens, topics")
        .execute(database.pool())
        .await
        .expect("analyze");
}

fn request<'a>(topic: &'a str, content: &'a str, hash: &'a str) -> Write<'a> {
    Write {
        topic,
        content,
        content_hash: hash,
        verdict: FilterDecision::Promoted,
        reason: "write-cost measurement",
        promoted: true,
        language: Some("eng"),
        language_confidence: Some(1.0),
        observed_at: time::OffsetDateTime::now_utc(),
        validity: Validity::ALWAYS,
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn one_write_does_not_pay_for_the_whole_project() {
    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    let profile = Profile::parse(PROFILE).expect("a known profile");
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    println!("\n   topics      write    drain    total   derive p50   fastest   edges");
    println!("  ----------------------------------------------------------------");

    let mut costs = Vec::new();
    for topics in SCALES {
        // A name nothing else uses, so a rerun measures a fresh project rather
        // than reopening one that already holds the topics from last time.
        let name = format!("writecost-{topics}-{}", uuid::Uuid::new_v4());
        let project = repository::ensure_project(database.pool(), &name)
            .await
            .expect("ensure project");
        fill(&database, project.id, topics, 3).await;

        let engine = Engine::open(&workspace, &name, profile, Access::ReadWrite)
            .await
            .expect("open the engine");

        // Once before the clock starts. The first write of a process loads the
        // model and opens the index, and neither is what this is about.
        engine
            .write(&request("warm up", "nothing in particular", "warm"))
            .await
            .expect("warm-up write");
        engine
            .drain_cascade(Owed::Everything)
            .await
            .expect("warm-up drain");

        // Names exactly one topic, at both scales, so the work the write
        // implies is identical and only the project around it differs.
        let content = "the incident review points at service 7 pipeline and \
                       nothing else in the fleet";
        let started = Instant::now();
        let recorded = engine
            .write(&request("incident review", content, "incident-1"))
            .await
            .expect("write")
            .state
            .expect("the filter promoted it");
        let wrote = started.elapsed().as_secs_f64() * 1000.0;
        let drain_started = Instant::now();
        engine.drain_cascade(Owed::Everything).await.expect("drain");
        let drained = drain_started.elapsed().as_secs_f64() * 1000.0;
        let millis = wrote + drained;

        // The same derivation again, on its own and with the edge already in
        // place. Idempotent by construction -- a name still present is found
        // unchanged and written nowhere -- so the second call does the lookup
        // and no writes, which is exactly the work whose cost is in question.
        // The first is thrown away. It pays whatever this project's first
        // derivation pays -- a connection, a plan, a page -- and leaving it in
        // made the *smaller* project look slower, which is the direction that
        // hides a regression rather than inventing one.
        engine
            .derive_mentions(&recorded)
            .await
            .expect("warm the derivation");
        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let at = Instant::now();
            engine
                .derive_mentions(&recorded)
                .await
                .expect("derive mentions");
            samples.push(at.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let derive = samples[samples.len() / 2];
        // The ratio is taken on the fastest, not the median. Noise on a shared
        // machine only ever adds, so the quickest of fifty is the closest any
        // of them gets to the work itself, and it is the statistic that moves
        // least between runs: across three runs of this file the medians gave
        // 0.58x, 1.02x and 1.57x where the minima gave 1.10x, 1.29x and 1.11x.
        // The median is printed because it is what a caller would feel.
        let fastest = samples[0];

        // The premise, asserted rather than assumed. A memory that names
        // nothing takes the short path, and two short paths would time the
        // same at any scale -- a flat line that means nothing. The edge is
        // what proves the lookup ran.
        let (edges,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM relationships
              WHERE project_id = $1 AND kind = 'mentions'",
        )
        .bind(project.id.0)
        .fetch_one(database.pool())
        .await
        .expect("count derived edges");

        println!(
            "  {topics:>7}   {wrote:>8.1} {drained:>8.1} {millis:>8.1} \
             {derive:>10.2} {fastest:>9.2} {edges:>7}"
        );
        assert_eq!(
            edges, 1,
            "the measured write should have derived exactly one mentions edge \
             at {topics} topics; {edges} means it took a path this test is not \
             measuring"
        );
        let _ = millis;
        costs.push(fastest);
    }

    let (small, large) = (costs[0], costs[1]);
    let growth = SCALES[1] as f64 / SCALES[0] as f64;
    let ratio = large / small;
    println!(
        "\n  {growth:.0}x the topics cost {ratio:.2}x the derivation \
         (at most {AT_MOST:.0}x allowed)"
    );
    assert!(
        ratio <= AT_MOST,
        "deriving one memory's edges cost {ratio:.2}x as much on a project \
         {growth:.0}x larger ({small:.1} ms at {} topics, {large:.1} ms at {}); \
         the write path is reading the project again instead of the index",
        SCALES[0],
        SCALES[1],
    );
}

/// The other axis: what the widest name in the project costs a write.
///
/// `derive_mentions` looks up every window of every width up to
/// `widest_topic_name`, so a project whose longest name is sixteen tokens asks
/// the index for roughly five times as many runs as one whose longest is
/// three. An Aho-Corasick automaton would carry the text once regardless, and
/// the deferred decision to keep the current lookup rests on that growth being
/// sub-linear -- which is what one batched `name_key = ANY($2)` should buy and
/// what nothing here had measured. The first test in this file holds this
/// width at three, so it is the one quantity it cannot see.
///
/// The assertion is that cost grows more slowly than the run count. It would
/// fail the moment the batched lookup became a lookup per run, which is the
/// shape that would make the automaton necessary.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "provisions postgres and downloads model weights"]
async fn derivation_does_not_pay_for_the_widest_name_in_the_project() {
    const TOPICS: usize = 20_000;

    let home = std::env::var("PAMIN_EVAL_HOME").ok();
    let scratch = home
        .is_none()
        .then(|| tempfile::tempdir().expect("temp workspace"));
    let workspace = match (&home, &scratch) {
        (Some(path), _) => Workspace::at(path),
        (None, Some(dir)) => Workspace::at(dir.path()),
        (None, None) => unreachable!("one of the two is always set"),
    };
    let profile = Profile::parse(PROFILE).expect("a known profile");
    let database = Database::open(&workspace, Connections::PerCommand)
        .await
        .expect("open the database");

    let tokens = PROSE.split_whitespace().count();
    println!("\n  widest   runs   derive p50   fastest   per run   edges");
    println!("  ----------------------------------------------------------");

    let mut costs = Vec::new();
    for width in WIDTHS {
        let name = format!("widest-{width}-{}", uuid::Uuid::new_v4());
        let project = repository::ensure_project(database.pool(), &name)
            .await
            .expect("ensure project");
        fill(&database, project.id, TOPICS, width).await;

        // The premise. Without it a bad `fill` would quietly re-measure width
        // three three times and print a flat line that meant nothing.
        let widest = repository::widest_topic_name(database.pool(), project.id)
            .await
            .expect("widest name");
        assert_eq!(
            widest, width,
            "the project was built with a widest name of {widest}, not {width}; \
             every row below it would be measuring the wrong axis"
        );

        let engine = Engine::open(&workspace, &name, profile, Access::ReadWrite)
            .await
            .expect("open the engine");
        let recorded = engine
            .write(&request("incident review", PROSE, "incident-wide"))
            .await
            .expect("write")
            .state
            .expect("the filter promoted it");
        engine.drain_cascade(Owed::Everything).await.expect("drain");
        engine
            .derive_mentions(&recorded)
            .await
            .expect("warm the derivation");

        let mut samples = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let at = Instant::now();
            engine
                .derive_mentions(&recorded)
                .await
                .expect("derive mentions");
            samples.push(at.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let (median, fastest) = (samples[samples.len() / 2], samples[0]);

        let (edges,): (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM relationships
              WHERE project_id = $1 AND kind = 'mentions'",
        )
        .bind(project.id.0)
        .fetch_one(database.pool())
        .await
        .expect("count derived edges");
        assert_eq!(
            edges, 1,
            "at width {width} the memory should still name exactly one topic; \
             {edges} means the lookup is finding something else"
        );

        let n = runs(tokens, width);
        println!(
            "  {width:>6} {n:>6} {median:>12.2} {fastest:>9.2} \
             {:>9.4} {edges:>7}",
            fastest / n as f64
        );
        costs.push((width, n, fastest));
    }

    let (narrow_w, narrow_runs, narrow) = costs[0];
    let (wide_w, wide_runs, wide) = costs[costs.len() - 1];
    let grew = wide / narrow;
    let implied = wide_runs as f64 / narrow_runs as f64;
    println!(
        "\n  {narrow_w} to {wide_w} tokens is {implied:.1}x the runs and {grew:.2}x \
         the cost"
    );
    assert!(
        grew < implied,
        "cost grew {grew:.2}x where the run count grew {implied:.1}x, so the lookup \
         is paying per run rather than once -- which is the shape that makes an \
         automaton necessary"
    );
}
