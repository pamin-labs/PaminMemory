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

/// Three tokens each, so `widest_topic_name` is realistic: a project whose
/// names are all one token asks the index for one run per position and never
/// exercises the widening the real one pays for.
fn names(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("service {i} pipeline"))
        .collect()
}

/// Inserts `count` topics and the name rows the lookup reads, in two statements.
async fn fill(database: &Database, project: pamin_core::ProjectId, count: usize) {
    let names = names(count);
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
    let widths: Vec<i16> = (0..count).map(|_| 3).collect();
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
        fill(&database, project.id, topics).await;

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
