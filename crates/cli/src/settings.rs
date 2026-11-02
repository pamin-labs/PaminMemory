//! The engine's settings, from the server settings a command resolved.
//!
//! Resolving them -- flags over `config.toml` over the defaults -- is the
//! client's, in `memory_client::settings`, because every command does it and
//! the client links none of the libraries. What is left here is handing the
//! result to the engine, with the libraries' own defaults for everything a
//! server is not started with.

use memory_client::settings::Server;

/// The variable naming an external PostgreSQL; see
/// `memory_store::StoreSettings::database_url`.
const DATABASE_URL: &str = "PAMIN_MEMORY_DATABASE_URL";

/// What the engine runs with, for a server resolved as `server`.
pub fn engine(server: &Server) -> anyhow::Result<memory_engine::Settings> {
    let default = memory_engine::Settings::default();
    Ok(memory_engine::Settings {
        inference: memory_index::InferenceSettings {
            device: server.device,
            threads: server.threads,
            ep_libraries: server.ep_libraries.clone(),
            prepared: server.prepared,
            fused_attention: server.fused_attention,
            split_embeddings: server.split_embeddings,
            ..default.inference
        },
        store: memory_store::StoreSettings {
            jit: server.jit,
            postgres_dir: server.postgres_dir.clone(),
            // From the environment alone: it holds credentials, which belong
            // in neither a flag, `config.toml`, nor the settings a client
            // sends the server and prints.
            database_url: database_url()?,
        },
        model_idle: server.model_idle,
        models_dir: server.models_dir.clone(),
        ..default
    })
}

/// The external database the server opens, if one is named; refused when it is
/// named but unusable, since the fallback, the workspace's own cluster, would
/// keep a hosted node's data on a disk that is not where its data lives.
fn database_url() -> anyhow::Result<Option<memory_store::DatabaseUrl>> {
    match std::env::var(DATABASE_URL) {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("{DATABASE_URL} is not UTF-8"),
        Ok(url) if url.trim().is_empty() => {
            anyhow::bail!("{DATABASE_URL} is empty; unset it to use the workspace's own cluster")
        }
        Ok(url) => Ok(Some(memory_store::DatabaseUrl(url))),
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, FromArgMatches, Parser};
    use memory_client::settings::{OPEN_INDEXES, ServerArgs};

    #[derive(Parser)]
    struct Test {
        #[command(flatten)]
        server: ServerArgs,
    }

    /// A command that sets nothing runs the engine at the libraries' own
    /// defaults: the client resolves against the defaults it shares with
    /// them, so the two cannot drift apart unnoticed.
    #[test]
    fn nothing_set_is_the_library_defaults() {
        let root = tempfile::tempdir().unwrap();
        let matches = Test::command()
            .try_get_matches_from(["pamin-memory"])
            .unwrap();
        let parsed = Test::from_arg_matches(&matches).unwrap();
        let server =
            memory_client::settings::Server::resolve(root.path(), &parsed.server, &matches)
                .unwrap();
        // `PAMIN_MEMORY_DEVICE` may be set by whoever runs the tests; nothing else is
        // read from the environment.
        let settings = memory_engine::Settings {
            inference: memory_index::InferenceSettings {
                device: server.device,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(super::engine(&server).unwrap(), settings);
        assert_eq!(server.open_indexes, OPEN_INDEXES);
    }
}
