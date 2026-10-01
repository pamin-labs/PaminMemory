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
    revision: String,
    cache_dir: PathBuf,
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
        Self::open_at(cache_dir, name, "main")
    }

    pub(crate) fn open_at(cache_dir: &Path, name: &str, revision: &str) -> Result<Self> {
        let cache_dir = cache_root(cache_dir);
        let mut builder = hf_hub::api::sync::ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .with_progress(false);
        if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
            builder = builder.with_endpoint(endpoint);
        }
        let repo = builder
            .build()
            .map_err(|error| IndexError::Engine(format!("reaching the model hub: {error}")))?
            .repo(hf_hub::Repo::with_revision(
                name.to_string(),
                hf_hub::RepoType::Model,
                revision.to_string(),
            ));
        Ok(Self {
            repo,
            name: name.to_string(),
            revision: revision.to_string(),
            cache_dir,
        })
    }

    /// Process-local validation keys track the cached snapshot, not just a
    /// moving model name. Before a first download the requested revision is
    /// used; a newly resolved snapshot causes one fresh calibration.
    pub(crate) fn identity(&self, cache_dir: &Path) -> String {
        let root = cache_root(cache_dir);
        let resolved = std::fs::read_to_string(
            root.join(format!("models--{}", self.name.replace('/', "--")))
                .join("refs")
                .join(&self.revision),
        )
        .unwrap_or_else(|_| self.revision.clone());
        format!(
            "{}@{}:{}",
            self.name,
            resolved.trim(),
            root.canonicalize().unwrap_or(root).display()
        )
    }

    /// The local path of one of the repository's files, downloading it first
    /// if it is not cached yet.
    pub(crate) fn get(&self, file: &str) -> Result<PathBuf> {
        let _transaction = match self.snapshot_transaction() {
            Ok(lock) => lock,
            // Read-only caches remain usable. No download is started without
            // the transaction, and cleanup also refuses if it cannot lock.
            Err(error) => {
                return cached_at(&self.cache_dir, &self.name, &self.revision, file).ok_or(error);
            }
        };
        self.repo.get(file).map_err(|error| {
            IndexError::Engine(format!("fetching {file} from {}: {error}", self.name))
        })
    }

    /// hf-hub releases its blob lock before linking the snapshot. In our
    /// owned cache, fetch and cleanup share a repository transaction through
    /// that final link/ref publication. Shared external caches are never cleaned.
    fn snapshot_transaction(&self) -> Result<std::fs::File> {
        let root = self
            .cache_dir
            .join(format!("models--{}", self.name.replace('/', "--")));
        std::fs::create_dir_all(&root)?;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(root.join(".pamin-snapshots.lock"))?;
        lock.lock()?;
        Ok(lock)
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
        if self.repository.revision == "main" {
            format!("{}/{}", self.repository.name, self.file)
        } else {
            format!(
                "{}@{}/{}",
                self.repository.name, self.repository.revision, self.file
            )
        }
    }

    fn on_disk(&self) -> Option<PathBuf> {
        cached_at(
            self.cache_dir,
            &self.repository.name,
            &self.repository.revision,
            self.file,
        )
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
    /// read too; a blob still referenced by another snapshot; in a model
    /// directory that is itself a link, which is how
    /// several workspaces share one cache; or a blob that resolves outside the
    /// model directory because its repository was linked in from another cache.
    fn remove(&self) -> Result<bool> {
        if std::env::var_os("HF_HOME").is_some()
            || std::fs::symlink_metadata(self.cache_dir)?.is_symlink()
            || std::fs::symlink_metadata(self.cache_dir.join(format!(
                "models--{}",
                self.repository.name.replace('/', "--")
            )))?
            .is_symlink()
        {
            return Ok(false);
        }
        let _transaction = self.repository.snapshot_transaction()?;
        let Some(entry) = self.on_disk() else {
            return Ok(false);
        };
        let blob = std::fs::canonicalize(&entry)?;
        if !blob.starts_with(std::fs::canonicalize(self.cache_dir)?) {
            return Ok(false);
        }
        // Pinned and moving revisions can share a content-addressed blob.
        // Removing it would strand the other snapshot's symlink, and the hub
        // client cannot repair a dangling entry by creating it again.
        // Find snapshots from the entry, not the resolved blob: platforms
        // without symlinks may cache the file directly in its snapshot.
        let snapshots = std::fs::canonicalize(
            entry
                .ancestors()
                .nth(Path::new(self.file).components().count() + 1)
                .expect("a hub entry has a snapshots ancestor"),
        )?;
        let own = std::fs::canonicalize(entry.parent().expect("snapshot entry parent"))?
            .join(entry.file_name().expect("snapshot file name"));
        let refs = snapshots
            .parent()
            .expect("snapshot repository")
            .join("refs");
        let own_ref = refs.join(&self.repository.revision);
        let snapshot = own
            .strip_prefix(&snapshots)
            .expect("entry in snapshots")
            .components()
            .next()
            .expect("snapshot commit")
            .as_os_str()
            .to_string_lossy();
        if revision_references(&refs, &own_ref, &snapshot)?
            || referenced_elsewhere(&snapshots, &own, &blob)?
        {
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

fn revision_references(directory: &Path, own: &Path, snapshot: &str) -> std::io::Result<bool> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if revision_references(&path, own, snapshot)? {
                return Ok(true);
            }
        } else if path != own && std::fs::read_to_string(&path)?.trim() == snapshot {
            return Ok(true);
        }
    }
    Ok(false)
}

fn referenced_elsewhere(directory: &Path, own: &Path, blob: &Path) -> std::io::Result<bool> {
    for item in std::fs::read_dir(directory)? {
        let item = item?;
        let path = item.path();
        if item.file_type()?.is_dir() {
            if referenced_elsewhere(&path, own, blob)? {
                return Ok(true);
            }
        } else if path != own && std::fs::canonicalize(&path).is_ok_and(|target| target == blob) {
            return Ok(true);
        }
    }
    Ok(false)
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
fn cached_at(cache_dir: &Path, name: &str, revision: &str, file: &str) -> Option<PathBuf> {
    hf_hub::Cache::new(cache_root(cache_dir))
        .repo(hf_hub::Repo::with_revision(
            name.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ))
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

    #[test]
    fn a_snapshot_cached_without_symlinks_is_removed() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, _) = downloaded(dir.path());
        std::fs::remove_file(&entry).unwrap();
        std::fs::write(&entry, b"direct weights").unwrap();
        let repository = Repository::open(dir.path(), NAME).unwrap();
        assert!(repository.file(dir.path(), FILE).remove().unwrap());
        assert!(!entry.exists());
    }

    #[test]
    fn pinned_revision_does_not_follow_main_or_reuse_its_prepared_label() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, _) = downloaded(dir.path());
        let root = dir.path().join("models--someone--model");
        std::fs::write(root.join("refs/fixed"), "c0ffee").unwrap();
        std::fs::write(root.join("refs/main"), "decaf").unwrap();
        let moving = root.join("snapshots/decaf").join(FILE);
        std::fs::create_dir_all(moving.parent().unwrap()).unwrap();
        std::fs::write(&moving, b"different weights").unwrap();
        let pinned = Repository::open_at(dir.path(), NAME, "fixed").unwrap();
        let main = Repository::open(dir.path(), NAME).unwrap();
        assert_eq!(pinned.file(dir.path(), FILE).on_disk(), Some(entry));
        assert_eq!(main.file(dir.path(), FILE).on_disk(), Some(moving));
        assert_ne!(
            pinned.file(dir.path(), FILE).label(),
            main.file(dir.path(), FILE).label()
        );
    }

    #[test]
    fn a_cached_model_is_readable_when_its_transaction_file_cannot_be_opened() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, _) = downloaded(dir.path());
        std::fs::create_dir(
            dir.path()
                .join("models--someone--model/.pamin-snapshots.lock"),
        )
        .unwrap();
        let repository = Repository::open(dir.path(), NAME).unwrap();
        assert_eq!(repository.get(FILE).unwrap(), entry);
        assert!(repository.file(dir.path(), FILE).remove().is_err());
        assert!(entry.exists());
    }

    #[test]
    fn two_revision_refs_preserve_the_same_snapshot_entry() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, blob) = downloaded(dir.path());
        std::fs::write(
            dir.path().join("models--someone--model/refs/fixed"),
            "c0ffee",
        )
        .unwrap();
        let moving = Repository::open(dir.path(), NAME).unwrap();
        let pinned = Repository::open_at(dir.path(), NAME, "fixed").unwrap();
        assert!(!moving.file(dir.path(), FILE).remove().unwrap());
        assert!(!pinned.file(dir.path(), FILE).remove().unwrap());
        assert_eq!(pinned.get(FILE).unwrap(), entry);
        assert_eq!(std::fs::read(blob).unwrap(), b"weights");
    }

    #[test]
    fn removal_waits_for_snapshot_publication_before_scanning_references() {
        use std::time::Duration;
        let dir = tempfile::tempdir().unwrap();
        let (entry, blob) = downloaded(dir.path());
        let repository = Repository::open(dir.path(), NAME).unwrap();
        let held = repository.snapshot_transaction().unwrap();
        let root = dir.path().to_path_buf();
        let (started, ready) = std::sync::mpsc::channel();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let repository = Repository::open(&root, NAME).unwrap();
            started.send(()).unwrap();
            sent.send(repository.file(&root, FILE).remove()).unwrap();
        });
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        let early = received.recv_timeout(Duration::from_millis(100));
        let other = dir
            .path()
            .join("models--someone--model/snapshots/decaf")
            .join(FILE);
        std::fs::create_dir_all(other.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink("../../../blobs/0123abcd", &other).unwrap();
        drop(held);
        worker.join().unwrap();
        assert!(
            matches!(early, Err(std::sync::mpsc::RecvTimeoutError::Timeout)),
            "cleanup ran inside snapshot publication"
        );
        assert!(
            !received
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap()
        );
        assert!(entry.exists() && blob.exists());
        assert_eq!(std::fs::read(other).unwrap(), b"weights");
    }

    #[test]
    fn removing_a_download_preserves_other_revision_references() {
        let dir = tempfile::tempdir().unwrap();
        let (entry, blob) = downloaded(dir.path());
        let other = dir
            .path()
            .join("models--someone--model/snapshots/decaf")
            .join(FILE);
        std::fs::create_dir_all(other.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink("../../../blobs/0123abcd", &other).unwrap();
        let repository = Repository::open(dir.path(), NAME).unwrap();
        assert!(!repository.file(dir.path(), FILE).remove().unwrap());
        assert_eq!(std::fs::read(&other).unwrap(), b"weights");
        assert!(entry.exists());
        assert!(blob.exists());
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
