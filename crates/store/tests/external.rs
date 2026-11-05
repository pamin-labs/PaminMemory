//! A store on a PostgreSQL server it did not start, as hosted memory runs:
//! a deploy migrates it with `Database::migrate`, `StoreSettings::database_url`
//! connects to it, and the workspace's own cluster is neither started nor
//! recorded.
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
    let superuser = host
        .read_server()
        .expect("read the host's record")
        .expect("the host recorded its server")
        .url();
    let (server, _) = superuser.rsplit_once('/').expect("a database in the URL");
    let (_, address) = server.split_once('@').expect("credentials in the URL");
    let external = |role: &str, database: &str| StoreSettings {
        database_url: Some(DatabaseUrl(format!(
            "postgresql://{role}:{role}@{address}/{database}"
        ))),
        ..common::store()
    };

    // A hosted login: it owns its databases and is bound by row-level
    // security, with the guarded role made for it once.
    for statement in [
        "CREATE ROLE node LOGIN PASSWORD 'node' NOSUPERUSER NOBYPASSRLS",
        "GRANT pamin_memory_app TO node WITH ADMIN OPTION",
        "CREATE DATABASE nodes OWNER node",
        "CREATE ROLE bypassing LOGIN PASSWORD 'bypassing' NOSUPERUSER BYPASSRLS",
    ] {
        sqlx::query(statement)
            .execute(hosted.pool())
            .await
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }

    let node = tempfile::tempdir().expect("node workspace");
    let node = Workspace::at(node.path());
    let settings = external("node", "nodes");

    // A server never migrates the database it shares: until a deploy has, it
    // refuses to start, naming what is missing.
    let behind = Database::open(&node, Connections::Resident, &settings)
        .await
        .err()
        .expect("a database nothing has migrated is refused");
    assert!(
        matches!(&behind, memory_store::StoreError::Behind(pending) if !pending.is_empty()),
        "{behind}"
    );
    let url = settings.database_url.as_ref().expect("an external URL");
    Database::migrate(url)
        .await
        .expect("migrate as a deploy does");
    Database::migrate(url)
        .await
        .expect("a second deploy migrates nothing new and fails nothing");

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
    let as_host = sqlx::PgPool::connect(&format!("{server}/nodes"))
        .await
        .expect("the host on the node's database");
    let read = memory_store::repository::ensure_project(&as_host, &name)
        .await
        .expect("read it through the host");
    assert_eq!(written.id, read.id);

    // The node reaches the newest guarded table as the guarded role.
    let mut scoped = database.scoped(written.id).await.expect("scoped");
    memory_store::repository::put_span_vectors(
        &mut scoped,
        written.id,
        "model@1",
        "cpu:plan",
        &[([7; 32], vec![1, 0])],
    )
    .await
    .expect("keep a span vector as the node");
    let kept = memory_store::repository::span_vectors(
        &mut *scoped,
        written.id,
        "model@1",
        None,
        &[[7; 32]],
    )
    .await
    .expect("read it back as the node");
    assert_eq!(kept.len(), 1);
    scoped.commit().await.expect("commit");

    // The URL carries credentials and never prints.
    let printed = format!("{settings:?}");
    assert!(!printed.contains("node:node"), "{printed}");

    // A node's pool is capped, so nodes sharing the server fit its limit.
    let capped = StoreSettings {
        database_connections: Some(2),
        ..settings.clone()
    };
    let database = Database::open(&node, Connections::Resident, &capped)
        .await
        .expect("open with a cap");
    assert_eq!(database.pool().options().get_max_connections(), 2);

    // A login row-level security would not bind is refused, saying why, by a
    // server and by a deploy migrating: one with BYPASSRLS, as a managed
    // server's administrator often is, and a superuser.
    for (refused, how) in [
        (external("bypassing", "nodes"), "BYPASSRLS"),
        (
            StoreSettings {
                database_url: Some(DatabaseUrl(format!("{server}/nodes"))),
                ..common::store()
            },
            "a superuser",
        ),
    ] {
        let error = Database::open(&node, Connections::PerCommand, &refused)
            .await
            .err()
            .unwrap_or_else(|| panic!("a login with {how} is refused"));
        assert!(error.to_string().contains("row-level security"), "{error}");
        let url = refused.database_url.as_ref().expect("an external URL");
        let error = Database::migrate(url)
            .await
            .err()
            .unwrap_or_else(|| panic!("migrating as a login with {how} is refused"));
        assert!(error.to_string().contains("row-level security"), "{error}");
    }
}
