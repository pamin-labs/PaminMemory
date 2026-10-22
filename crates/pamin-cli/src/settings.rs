//! The engine's settings, from the server settings a command resolved.
//!
//! Resolving them -- flags over `config.toml` over the defaults -- is the
//! client's, in `pamin_client::settings`, because every command does it and
//! the client links none of the libraries. What is left here is handing the
//! result to the engine, with the libraries' own defaults for everything a
//! server is not started with.

use pamin_client::settings::Server;

/// What the engine runs with, for a server resolved as `server`.
pub fn engine(server: &Server) -> pamin_engine::Settings {
    let default = pamin_engine::Settings::default();
    pamin_engine::Settings {
        inference: pamin_index::InferenceSettings {
            device: server.device,
            threads: server.threads,
            ep_libraries: server.ep_libraries.clone(),
            prepared: server.prepared,
            fused_attention: server.fused_attention,
            split_embeddings: server.split_embeddings,
            ..default.inference
        },
        store: pamin_store::StoreSettings {
            jit: server.jit,
            postgres_dir: server.postgres_dir.clone(),
        },
        model_idle: server.model_idle,
        ..default
    }
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, FromArgMatches, Parser};
    use pamin_client::settings::{OPEN_INDEXES, ServerArgs};

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
        let matches = Test::command().try_get_matches_from(["pamin"]).unwrap();
        let parsed = Test::from_arg_matches(&matches).unwrap();
        let server =
            pamin_client::settings::Server::resolve(root.path(), &parsed.server, &matches).unwrap();
        // `PAMIN_DEVICE` may be set by whoever runs the tests; nothing else is
        // read from the environment.
        let settings = pamin_engine::Settings {
            inference: pamin_index::InferenceSettings {
                device: server.device,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(super::engine(&server), settings);
        assert_eq!(server.open_indexes, OPEN_INDEXES);
    }
}
