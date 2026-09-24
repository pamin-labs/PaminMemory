//! Fetching a model's files from the hub.
//!
//! Both kinds of model in this crate are downloaded on first use into the
//! workspace's model directory and read from there afterwards. One place does
//! that, so the two cannot disagree about where the weights live.

use std::path::{Path, PathBuf};

use crate::error::{IndexError, Result};

/// One model repository on the hub, cached under a workspace's model directory.
pub(crate) struct Repository {
    repo: hf_hub::api::sync::ApiRepo,
    name: String,
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
        Self::at(cache_dir, hf_hub::Repo::model(name.to_string()))
    }

    /// Opens `name` as it was at `revision`, cached under `cache_dir` the same
    /// way.
    pub(crate) fn pinned(cache_dir: &Path, name: &str, revision: &str) -> Result<Self> {
        Self::at(
            cache_dir,
            hf_hub::Repo::with_revision(
                name.to_string(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ),
        )
    }

    fn at(cache_dir: &Path, repo: hf_hub::Repo) -> Result<Self> {
        let cache_dir = std::env::var_os("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| cache_dir.to_path_buf());
        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .with_progress(false);
        if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
            builder = builder.with_endpoint(endpoint);
        }
        let name = repo.url();
        let repo = builder
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .repo(repo);
        Ok(Self { repo, name })
    }

    /// The local path of one of the repository's files, downloading it first
    /// if it is not cached yet.
    pub(crate) fn get(&self, file: &str) -> Result<PathBuf> {
        self.repo.get(file).map_err(|error| {
            IndexError::Engine(format!("fetching {file} from {}: {error}", self.name))
        })
    }
}
