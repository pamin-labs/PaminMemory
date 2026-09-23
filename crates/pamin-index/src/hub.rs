//! Fetching a model's files from the hub.
//!
//! Both kinds of model in this crate are downloaded on first use into the
//! workspace's model directory and read from there afterwards. One place does
//! that, so the two cannot disagree about where the weights live.
//!
//! A repository can also be a directory on this machine, for an export that
//! is not published anywhere a download could reach -- the experimental
//! `pplx` profile's is one. Its files are read the same way, so what a model
//! does with them does not depend on where they came from.

use std::path::{Path, PathBuf};

use crate::error::{IndexError, Result};

/// One model repository on the hub, cached under a workspace's model directory.
pub(crate) struct Repository {
    source: Source,
    name: String,
}

/// Boxed: the hub's handle is over three hundred bytes, and a directory is a
/// path.
enum Source {
    Hub(Box<hf_hub::api::sync::ApiRepo>),
    Directory(PathBuf),
}

impl Repository {
    /// Opens `name`, cached under `cache_dir`.
    ///
    /// Or under `HF_HOME`, and fetched from `HF_ENDPOINT`, when those are set:
    /// that is what `fastembed` does when it fetches a model itself, which is
    /// how the embedding model was always fetched. The reranker used to ignore
    /// both, so under either it was fetched from somewhere the embedder was
    /// not -- a mirror set for one and not the other, or two copies of the
    /// cache. Now the two are fetched alike.
    pub(crate) fn open(cache_dir: &Path, name: &str) -> Result<Self> {
        let cache_dir = std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_dir.to_path_buf());
        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(false);
        if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
            builder = builder.with_endpoint(endpoint);
        }
        let repo = builder
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .model(name.to_string());
        Ok(Self {
            source: Source::Hub(Box::new(repo)),
            name: name.to_string(),
        })
    }

    /// The files in `dir`, which is the whole repository: nothing is fetched,
    /// and a file that is not there is an error rather than a download.
    pub(crate) fn directory(dir: &Path) -> Self {
        Self {
            source: Source::Directory(dir.to_path_buf()),
            name: dir.display().to_string(),
        }
    }

    /// The local path of one of the repository's files, downloading it first
    /// if it is not cached yet.
    pub(crate) fn get(&self, file: &str) -> Result<PathBuf> {
        match &self.source {
            Source::Hub(repo) => repo.get(file).map_err(|error| {
                IndexError::Engine(format!("fetching {file} from {}: {error}", self.name))
            }),
            Source::Directory(dir) => {
                let path = dir.join(file);
                if path.is_file() {
                    Ok(path)
                } else {
                    Err(IndexError::Engine(format!(
                        "{file} is not in {}",
                        self.name
                    )))
                }
            }
        }
    }
}
