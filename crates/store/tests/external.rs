//! A store on a PostgreSQL server it did not start, as hosted memory runs:
//! `StoreSettings::database_url` connects and migrates, and the workspace's
//! own cluster is neither started nor recorded.
//!
//! Run with `cargo test -p memory-store --test external -- --ignored`.

mod common;

use memory_store::{Connections, Database, DatabaseUrl, StoreSettings, Workspace};

#[tokio::test]
#[ignore = "needs a real postgres cluster"]
async fn an_external_server_is_used_and_never_started() {
    // A cluster to stand in for the hosted one: another workspace's own.
    let host = tempfile::tempdir().expect("host workspace");
    let host = Workspace::at(host.path());
    let hosted = Database::open(&host, Connections::PerCommand, &common::store())
        .await
        .expect("start the stand-in server");
    let url = host
        .read_server()
        .expect("read the host's record")
        .expect("the host recorded its server")
        .url();

    let node = tempfile::tempdir().expect("node workspace");
    let node = Workspace::at(node.path());
    let settings = StoreSettings {
        database_url: Some(DatabaseUrl(url.clone())),
        ..common::store()
    };
    let database = Database::open(&node, Connections::Resident, &settings)
        .await
        .expect("open the external server");
    assert!(
        node.read_server()
            .expect("read the node's record")
            .is_none(),
        "no cluster is started or recorded for the node"
    );

    // One database: what the node writes, the host reads.
    let name = format!("external-{}", uuid::Uuid::now_v7());
    let written = memory_store::repository::ensure_project(database.pool(), &name)
        .await
        .expect("write a project through the node");
    let read = memory_store::repository::ensure_project(hosted.pool(), &name)
        .await
        .expect("read it through the host");
    assert_eq!(written.id, read.id);

    // A second open migrates nothing new and fails nothing.
    Database::open(&node, Connections::PerCommand, &settings)
        .await
        .expect("open it again");

    // Two nodes starting at once on a database nothing has migrated, as after
    // an upgrade, both start: the loser of the migration race tries again.
    sqlx::query("CREATE DATABASE racing")
        .execute(hosted.pool())
        .await
        .expect("create an empty database");
    let racing = StoreSettings {
        database_url: Some(DatabaseUrl(format!(
            "{}/racing",
            url.rsplit_once('/').expect("a database in the URL").0
        ))),
        ..common::store()
    };
    let (first, second) = tokio::join!(
        Database::open(&node, Connections::PerCommand, &racing),
        Database::open(&host, Connections::PerCommand, &racing),
    );
    first.expect("the first node starts");
    second.expect("the second node starts");

    // The URL carries credentials and never prints.
    let printed = format!("{settings:?}");
    assert!(!printed.contains(&url), "{printed}");
}
