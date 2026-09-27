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
        let cache_dir = cache_root(cache_dir);
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
            repo,
            name: name.to_string(),
        })
    }

    /// The local path of one of the repository's files, downloading it first
    /// if it is not cached yet.
    pub(crate) fn get(&self, file: &str) -> Result<PathBuf> {
        self.repo.get(file).map_err(|error| {
            IndexError::Engine(format!("fetching {file} from {}: {error}", self.name))
        })
    }

    /// One of the repository's files, as `crate::prepared` asks for a model:
    /// found on disk, fetched, or removed once a mapped copy has replaced it.
    pub(crate) fn file<'a>(&'a self, cache_dir: &'a Path, file: &'a str) -> File<'a> {
        File {
            repository: self,
            cache_dir,
            file,
        }
    }
}

/// One file of a [`Repository`] -- a model's weights.
pub(crate) struct File<'a> {
    repository: &'a Repository,
    cache_dir: &'a Path,
    file: &'a str,
}

impl crate::prepared::Download for File<'_> {
    fn label(&self) -> String {
        format!("{}/{}", self.repository.name, self.file)
    }

    fn on_disk(&self) -> Option<PathBuf> {
        cached(self.cache_dir, &self.repository.name, self.file)
    }

    fn fetch(&self) -> Result<PathBuf> {
        self.repository.get(self.file)
    }

    /// Removes the file's snapshot entry and then the blob it points at, so
    /// the next [`Repository::get`] downloads it again.
    ///
    /// In that order because the hub client re-creates the entry only if it
    /// does not exist, and it asks by following it: a link left pointing at a
    /// removed blob looks absent, and creating it again fails. An interrupted
    /// removal therefore leaves at worst an unreferenced blob, which the next
    /// download of the same file overwrites.
    ///
    /// Declines -- `Ok(false)`, nothing touched -- for a file this model
    /// directory does not own: under `HF_HOME`, which is a cache other tools
    /// read too; in a model directory that is itself a link, which is how
    /// several workspaces share one cache; or a blob that resolves outside the
    /// model directory because its repository was linked in from another cache.
    fn remove(&self) -> Result<bool> {
        if std::env::var_os("HF_HOME").is_some()
            || std::fs::symlink_metadata(self.cache_dir)?.is_symlink()
        {
            return Ok(false);
        }
        let Some(entry) = self.on_disk() else {
            return Ok(false);
        };
        let blob = std::fs::canonicalize(&entry)?;
        if !blob.starts_with(std::fs::canonicalize(self.cache_dir)?) {
            return Ok(false);
        }
        std::fs::remove_file(&entry)?;
        match std::fs::remove_file(&blob) {
            // Where the platform could not link, the entry was the file.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            removed => removed?,
        }
        Ok(true)
    }
}

/// Where the hub's files are cached: `HF_HOME` when it is set, as
/// [`Repository::open`] reads it, and the workspace's model directory
/// otherwise.
fn cache_root(cache_dir: &Path) -> PathBuf {
    std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| cache_dir.to_path_buf())
}

/// Where `file` of repository `name` is on disk, without asking the hub
/// anything, or `None` if it is not.
fn cached(cache_dir: &Path, name: &str, file: &str) -> Option<PathBuf> {
    hf_hub::Cache::new(cache_root(cache_dir))
        .model(name.to_string())
        .get(file)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::prepared::Download;

    const NAME: &str = "someone/model";
    const FILE: &str = "onnx/model.onnx";

    /// A repository laid out as the hub client leaves it under `cache`: a
    /// ref naming a snapshot, whose entry links to a blob named by content.
    fn downloaded(cache: &Path) -> (PathBuf, PathBuf) {
        let repository = cache.join("models--someone--model");
        std::fs::create_dir_all(repository.join("refs")).expect("refs");
        std::fs::write(repository.join("refs/main"), "c0ffee").expect("the ref");
        std::fs::create_dir_all(repository.join("blobs")).expect("blobs");
        let blob = repository.join("blobs/0123abcd");
        std::fs::write(&blob, b"weights").expect("the blob");
        let entry = repository.join("snapshots/c0ffee").join(FILE);
        std::fs::create_dir_all(entry.parent().expect("a parent")).expect("the snapshot");
        std::os::unix::fs::symlink("../../../blobs/0123abcd", &entry).expect("the entry");
        (entry, blob)
    }

    /// Removing takes the entry and the blob both: a blob left behind is the
    /// disk the removal was for, and an entry left behind makes the next
    /// download fail to link.
    #[test]
    fn removing_a_download_takes_its_entry_and_its_blob() {
        let dir = tempfile::tempdir().expect("temp dir");
        let (entry, blob) = downloaded(dir.path());
        let repository = Repository::open(dir.path(), NAME).expect("open");
        let file = repository.file(dir.path(), FILE);
        assert_eq!(file.on_disk(), Some(entry.clone()));

        assert!(file.remove().expect("remove"));
        assert!(file.on_disk().is_none());
        assert!(
            std::fs::symlink_metadata(&entry).is_err(),
            "the entry is left"
        );
        assert!(!blob.exists(), "the blob is left");
    }

    /// A repository linked in from another cache is that cache's, and is
    /// left alone.
    #[test]
    fn a_download_linked_in_from_elsewhere_is_not_removed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let elsewhere = dir.path().join("elsewhere");
        let (_, blob) = downloaded(&elsewhere);
        let cache = dir.path().join("models");
        std::fs::create_dir_all(&cache).expect("the cache");
        std::os::unix::fs::symlink(
            elsewhere.join("models--someone--model"),
            cache.join("models--someone--model"),
        )
        .expect("link it in");
        let repository = Repository::open(&cache, NAME).expect("open");
        let file = repository.file(&cache, FILE);
        assert!(file.on_disk().is_some());

        assert!(!file.remove().expect("decline"));
        assert!(blob.exists(), "another cache's blob was removed");
        assert!(file.on_disk().is_some());
    }

    /// Nor is anything in a model directory that is itself a link: other
    /// workspaces link to the same one.
    #[test]
    fn a_download_in_a_linked_model_directory_is_not_removed() {
        let dir = tempfile::tempdir().expect("temp dir");
        let shared = dir.path().join("shared");
        let (_, blob) = downloaded(&shared);
        let cache = dir.path().join("models");
        std::os::unix::fs::symlink(&shared, &cache).expect("link the directory");
        let repository = Repository::open(&cache, NAME).expect("open");
        let file = repository.file(&cache, FILE);
        assert!(file.on_disk().is_some());

        assert!(!file.remove().expect("decline"));
        assert!(blob.exists(), "a shared directory's blob was removed");
    }
}
