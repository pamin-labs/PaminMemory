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

    // A node's pool is capped, so nodes sharing the server fit its limit.
    let capped = StoreSettings {
        database_connections: Some(2),
        ..settings.clone()
    };
    let database = Database::open(&node, Connections::Resident, &capped)
        .await
        .expect("open with a cap");
    assert_eq!(database.pool().options().get_max_connections(), 2);

    // A hosted role that owns its database but may not create roles, as a
    // managed server's often is.
    for statement in [
        "CREATE ROLE hosted_node LOGIN PASSWORD 'hosted' NOCREATEROLE NOSUPERUSER",
        "CREATE DATABASE limited OWNER hosted_node",
    ] {
        sqlx::query(statement)
            .execute(hosted.pool())
            .await
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let (_, address) = url.split_once('@').expect("credentials in the URL");
    let limited = StoreSettings {
        database_url: Some(DatabaseUrl(format!(
            "postgresql://hosted_node:hosted@{}/limited",
            address.rsplit_once('/').expect("a database in the URL").0
        ))),
        ..common::store()
    };

    // Without the guarded role it is refused before migrating, saying how to
    // make it.
    sqlx::query("ALTER ROLE pamin_memory_app RENAME TO pamin_memory_app_aside")
        .execute(hosted.pool())
        .await
        .expect("set the guarded role aside");
    let refused = Database::open(&node, Connections::PerCommand, &limited).await;
    sqlx::query("ALTER ROLE pamin_memory_app_aside RENAME TO pamin_memory_app")
        .execute(hosted.pool())
        .await
        .expect("put the guarded role back");
    let refused = refused.err().expect("refused without the guarded role");
    assert!(
        refused.to_string().contains("CREATE ROLE pamin_memory_app"),
        "{refused}"
    );

    // The role made but not granted: still refused, since V20 grants it to
    // whoever migrates.
    let ungranted = Database::open(&node, Connections::PerCommand, &limited)
        .await
        .err()
        .expect("refused while the guarded role is not granted");
    assert!(
        ungranted
            .to_string()
            .contains("CREATE ROLE pamin_memory_app"),
        "{ungranted}"
    );

    // With the role made once and granted, the same role migrates and starts.
    sqlx::query("GRANT pamin_memory_app TO hosted_node WITH ADMIN OPTION")
        .execute(hosted.pool())
        .await
        .expect("grant the guarded role");
    Database::open(&node, Connections::PerCommand, &limited)
        .await
        .expect("a role without CREATEROLE starts once the guarded role exists");

    // A role with CREATEROLE on a server without the guarded role makes it
    // and grants it to itself, which PostgreSQL 16 and later allow the
    // creator. Last: the role it makes stays.
    for statement in [
        "CREATE ROLE creating_node LOGIN PASSWORD 'creating' CREATEROLE NOSUPERUSER",
        "CREATE DATABASE creating OWNER creating_node",
        "ALTER ROLE pamin_memory_app RENAME TO pamin_memory_app_before",
    ] {
        sqlx::query(statement)
            .execute(hosted.pool())
            .await
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let creating = StoreSettings {
        database_url: Some(DatabaseUrl(format!(
            "postgresql://creating_node:creating@{}/creating",
            address.rsplit_once('/').expect("a database in the URL").0
        ))),
        ..common::store()
    };
    Database::open(&node, Connections::PerCommand, &creating)
        .await
        .expect("a role with CREATEROLE makes the guarded role and starts");
}
