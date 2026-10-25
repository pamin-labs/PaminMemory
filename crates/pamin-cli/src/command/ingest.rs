//! `pamin ingest` — record files as memories, one per section.
//!
//! What `pamin import` is for memories somebody has already written, this is
//! for the files they were written in. A file is cut into sections by
//! [`pamin_index::sections`], and each section is recorded the way an import
//! records a memory: against one open engine, several topics at once, with the
//! projection catching up in rounds.
//!
//! Re-ingesting is the ordinary case. A section that says what its memory
//! already does is left alone -- no evidence, no version, no work for the
//! projection -- so ingesting an unchanged tree costs a read per section and a
//! changed one costs what changed.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use pamin_client::command::ingest::{Args, Ingested};
use pamin_engine::Owed;
use pamin_index::VectorIndex;
use pamin_index::sections::{SEPARATOR, Section, sections};
use pamin_index::spans::TokenCount;

use crate::session::Session;

/// The extensions read from a directory. A file named on the command line is
/// read whatever it is called.
const EXTENSIONS: [&str; 3] = ["md", "markdown", "txt"];

/// How many sections to record between checks on the queue. See
/// `import::BETWEEN_CHECKS`, which this is for the same reason.
const BETWEEN_CHECKS: usize = 1_000;

/// The floor on how many sections are recorded at once. See
/// `import::LEAST_AT_ONCE`.
const LEAST_AT_ONCE: usize = 2;

/// A file to ingest, and the name its sections go by.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct File {
    /// Its path from the root of the repository it is in, or from what was
    /// named on the command line when it is in none, with `/` between
    /// components -- and the root's own name first, so that two repositories'
    /// `README.md` are two files and not one.
    name: String,
    path: PathBuf,
}

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Ingested> {
    let (files, mut skipped) = files(&args.paths)?;

    let mut texts = Vec::new();
    for file in files {
        match std::fs::read(&file.path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => texts.push((file.name, text)),
                Err(_) => skipped.push(format!("{}: not UTF-8", file.path.display())),
            },
            Err(error) => skipped.push(format!("{}: {error}", file.path.display())),
        }
    }

    // Counted by BGE-M3's own tokenizer, which is the reranker's vocabulary
    // too: both are XLM-R's. Loaded off the runtime, as every model load is.
    // Not loaded at all for an ingest that found nothing to read.
    let models = session.workspace().models_dir();
    let cut: Vec<(String, Vec<Section>)> = tokio::task::spawn_blocking(move || {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let counter = pamin_index::SpanTokens::load(&models)?;
        let counter: &dyn TokenCount = &counter;
        Ok::<_, pamin_index::IndexError>(
            texts
                .into_iter()
                .map(|(name, text)| {
                    let sections = sections(&name, &text, counter);
                    (name, sections)
                })
                .collect::<Vec<_>>(),
        )
    })
    .await
    .context("cutting files into sections")??;

    let engine = session.engine(project, vector_index).await?;

    let read: Vec<&str> = cut.iter().map(|(name, _)| name.as_str()).collect();
    let cut: Vec<&Section> = cut.iter().flat_map(|(_, sections)| sections).collect();
    let missing = missing(
        &read,
        &cut,
        pamin_store::repository::file_topic_names(engine.database.pool(), engine.project).await?,
    );

    // Every section is a topic of its own -- its name holds its file's path,
    // and one file's names are numbered apart -- so they all run at once, in
    // the bound `import` derives from the pool for the same reason.
    let at_once = usize::try_from(engine.database.pool().options().get_max_connections())
        .unwrap_or(LEAST_AT_ONCE)
        .max(LEAST_AT_ONCE);

    let (mut written, mut unchanged, mut held) = (0, 0, 0);
    let mut lagging = false;
    let mut since_check = 0;
    for chunk in cut.chunks(at_once) {
        let recorded = futures::future::try_join_all(chunk.iter().map(|section| {
            engine.remember_section(
                &section.name,
                &section.content,
                pamin_core::Validity::ALWAYS,
            )
        }))
        .await?;

        for outcome in recorded {
            match outcome {
                None => unchanged += 1,
                Some((verdict, _)) if verdict.is_promoted() => written += 1,
                Some(_) => held += 1,
            }
        }

        since_check += chunk.len();
        if since_check >= BETWEEN_CHECKS {
            since_check = 0;
            let behind = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;
            if !pamin_core::may_defer(behind) {
                lagging = true;
                engine.drain_cascade(Owed::WhatAMemoryNeeds).await?;
            }
        }
    }

    engine.drain_cascade(Owed::WhatAMemoryNeeds).await?;
    engine.flush_what_is_applied().await?;
    let owed = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;

    Ok(Ingested {
        files: read.len(),
        sections: cut.len(),
        written,
        unchanged,
        held,
        missing,
        skipped,
        cascade: if owed == 0 { "applied" } else { "queued" }.to_string(),
        cascade_lagging: lagging,
    })
}

/// Every file `paths` names, once each, in name order, and the ones skipped
/// with why.
///
/// A path that cannot be read at all is a refusal, because it is what the
/// command was asked to do; a file or directory found under one and unreadable
/// is skipped and reported, as one that is not UTF-8 is. Two files that would
/// go by one name are a refusal too: they would be one topic, which each
/// ingest would hand from one to the other.
fn files(paths: &[PathBuf]) -> Result<(Vec<File>, Vec<String>)> {
    let mut files = BTreeSet::new();
    let mut skipped = Vec::new();
    for path in paths {
        // Resolved here rather than by the client, which made it absolute but
        // left `..` and links in it: two spellings of one file must give its
        // sections one name.
        let path =
            std::fs::canonicalize(path).with_context(|| format!("reading {}", path.display()))?;
        let root = root(&path);
        if path.is_dir() {
            walk(&path, &root, &mut files, &mut skipped);
        } else {
            match file(&path, &root) {
                Some(file) => {
                    files.insert(file);
                }
                None => skipped.push(format!("{}: name is not UTF-8", path.display())),
            }
        }
    }

    let mut by_name: std::collections::BTreeMap<&str, Vec<&Path>> = Default::default();
    for file in &files {
        by_name.entry(&file.name).or_default().push(&file.path);
    }
    let clashes: Vec<String> = by_name
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(name, paths)| {
            let paths: Vec<String> = paths
                .iter()
                .map(|path| path.display().to_string())
                .collect();
            format!("{name}: {}", paths.join(", "))
        })
        .collect();
    if !clashes.is_empty() {
        anyhow::bail!(
            "these files would be recorded under one name; ingest them separately:\n{}",
            clashes.join("\n")
        );
    }

    Ok((files.into_iter().collect(), skipped))
}

/// What the files under `path` are named relative to: the repository it is
/// in, so that inside a repository, ingesting a tree and ingesting one file of
/// it name that file alike; or else `path` itself when it is a directory and
/// its parent when it is a file, so that outside one, a file named on its own
/// is named from its own directory rather than from a tree it was once
/// ingested with.
///
/// A submodule's `.git` is a file pointing into its superproject's
/// `.git/modules`, and is passed over, so a file in a submodule is named alike
/// whether the superproject or the submodule was ingested. A worktree's is a
/// file too, pointing anywhere else, and is a root.
fn root(path: &Path) -> PathBuf {
    let start = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(path)
    };
    let is_root = |ancestor: &Path| {
        let git = ancestor.join(".git");
        if git.is_dir() {
            return true;
        }
        std::fs::read_to_string(&git).is_ok_and(|pointer| !pointer.contains(".git/modules/"))
    };
    start
        .ancestors()
        .find(|ancestor| is_root(ancestor))
        .unwrap_or(start)
        .to_path_buf()
}

/// The file at `path`, named from `root`; `None` when a component of its name
/// is not UTF-8, which a topic's name has to be and which a lossy conversion
/// would let two files share.
fn file(path: &Path, root: &Path) -> Option<File> {
    let relative = path.strip_prefix(root).unwrap_or(path);
    let mut components = Vec::new();
    if let Some(root) = root.file_name() {
        components.push(root.to_str()?);
    }
    for component in relative.components() {
        components.push(component.as_os_str().to_str()?);
    }
    Some(File {
        name: components.join("/"),
        path: path.to_path_buf(),
    })
}

/// The files under `directory` with one of [`EXTENSIONS`], skipping hidden
/// entries -- `.git` among them -- symbolic links, which could lead out of the
/// tree or round in a circle, and whatever git ignores there. A repository's
/// `node_modules/` or `target/` is full of Markdown nobody wrote for it.
fn walk(directory: &Path, root: &Path, files: &mut BTreeSet<File>, skipped: &mut Vec<String>) {
    let walk = ignore::WalkBuilder::new(directory)
        .hidden(true)
        .follow_links(false)
        // Git's rules only: `.gitignore` in the tree and above it,
        // `.git/info/exclude` and the user's global excludes, inside a
        // repository. Not ripgrep's own `.ignore` and `.rgignore`, which a
        // user writing them meant for a search tool.
        .ignore(false)
        .parents(true)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .build();
    for entry in walk {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                skipped.push(error.to_string());
                continue;
            }
        };
        let path = entry.path();
        let is_text = entry.file_type().is_some_and(|kind| kind.is_file())
            && path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    EXTENSIONS
                        .iter()
                        .any(|known| extension.eq_ignore_ascii_case(known))
                });
        if !is_text {
            continue;
        }
        match file(path, root) {
            Some(file) => {
                files.insert(file);
            }
            None => skipped.push(format!("{}: name is not UTF-8", path.display())),
        }
    }
}

/// The names in `recorded` -- every section an ingest has put on the retrieval
/// surface -- that belong to one of `read` and are not among `cut`.
fn missing(read: &[&str], cut: &[&Section], recorded: Vec<String>) -> Vec<String> {
    let present: BTreeSet<&str> = cut.iter().map(|section| section.name.as_str()).collect();
    let mut missing: Vec<String> = recorded
        .into_iter()
        .filter(|name| !present.contains(name.as_str()))
        .filter(|name| read.iter().any(|file| belongs(name, file)))
        .collect();
    missing.sort();
    missing
}

/// Whether the section named `name` was cut out of the file named `file`: its
/// name is the file's, then any numbers ` #n` and parts ` (n)` it was given,
/// then nothing or its headings.
///
/// Read exactly, because `a.md (old).md` begins with `a.md (` and is another
/// file.
fn belongs(name: &str, file: &str) -> bool {
    let Some(mut rest) = name.strip_prefix(file) else {
        return false;
    };
    loop {
        if rest.is_empty() || rest.starts_with(SEPARATOR) {
            return true;
        }
        let after = if let Some(number) = rest.strip_prefix(" #") {
            digits(number)
        } else if let Some(part) = rest.strip_prefix(" (") {
            digits(part).and_then(|after| after.strip_prefix(')'))
        } else {
            None
        };
        match after {
            Some(after) => rest = after,
            None => return false,
        }
    }
}

/// `text` after the run of ASCII digits it opens with, if it opens with one.
fn digits(text: &str) -> Option<&str> {
    let after = text.trim_start_matches(|c: char| c.is_ascii_digit());
    (after.len() < text.len()).then_some(after)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(name: &str) -> Section {
        Section {
            name: name.to_string(),
            content: String::new(),
        }
    }

    fn names(paths: &[PathBuf]) -> Vec<String> {
        let (files, skipped) = files(paths).expect("files");
        assert!(skipped.is_empty(), "{skipped:?}");
        files.into_iter().map(|file| file.name).collect()
    }

    /// The name of the temporary directory `path` is, which every name under
    /// it starts with.
    fn base(path: &Path) -> String {
        path.file_name().unwrap().to_str().unwrap().to_string()
    }

    #[test]
    fn a_section_belongs_to_its_file_and_no_other() {
        for name in [
            "docs/a.md",
            "docs/a.md › Usage",
            "docs/a.md #2",
            "docs/a.md (3)",
            "docs/a.md #2 (3)",
            "docs/a.md #2 › Usage",
        ] {
            assert!(belongs(name, "docs/a.md"), "{name}");
        }
        for name in [
            "docs/a.md.bak › Usage",
            "docs/ab.md",
            "docs",
            "docs/a.md (old).md",
            "docs/a.md #1.md",
            "docs/a.md (2",
            "docs/a.md #",
        ] {
            assert!(!belongs(name, "docs/a.md"), "{name}");
        }
    }

    /// Only the files this ingest read are compared: a section of a file left
    /// out of the command is not missing, it is just not being looked at.
    #[test]
    fn missing_is_what_an_earlier_ingest_left_under_the_files_read() {
        let kept = section("a.md › Kept");
        let missing = missing(
            &["a.md"],
            &[&kept],
            vec![
                "a.md › Kept".to_string(),
                "a.md › Gone".to_string(),
                "b.md › Elsewhere".to_string(),
                "manual_topic".to_string(),
            ],
        );
        assert_eq!(missing, ["a.md › Gone"]);
    }

    #[test]
    fn a_directory_is_read_for_text_files_skipping_hidden_entries_and_links() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("docs/deep")).unwrap();
        std::fs::create_dir_all(root.join(".hidden")).unwrap();
        for name in [
            "README.md",
            "docs/guide.markdown",
            "docs/deep/notes.TXT",
            "docs/image.png",
            ".hidden/secret.md",
            "docs/.draft.md",
        ] {
            std::fs::write(root.join(name), "text").unwrap();
        }
        #[cfg(unix)]
        std::os::unix::fs::symlink(root.join("README.md"), root.join("docs/link.md")).unwrap();

        let base = base(root);
        assert_eq!(
            names(&[root.to_path_buf()]),
            [
                format!("{base}/README.md"),
                format!("{base}/docs/deep/notes.TXT"),
                format!("{base}/docs/guide.markdown"),
            ]
        );
    }

    /// Inside a repository, a file is named from the repository's root
    /// whether the tree or the file itself was named, so the two ingests
    /// record the same sections.
    #[test]
    fn a_file_is_named_alike_however_it_was_reached() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("docs")).unwrap();
        std::fs::write(root.join("docs/cli.md"), "text").unwrap();

        let expected = format!("{}/docs/cli.md", base(root));
        assert_eq!(names(&[root.join("docs")]), std::slice::from_ref(&expected));
        assert_eq!(names(&[root.join("docs/../docs/cli.md")]), [expected]);

        // Named twice, read once.
        assert_eq!(
            names(&[root.join("docs"), root.join("docs/cli.md")]).len(),
            1
        );
    }

    /// A submodule's `.git` file points into the superproject, so a file in
    /// it is named from the superproject's root however it is reached; a
    /// worktree's points elsewhere and is a root of its own.
    #[test]
    fn a_submodule_is_named_from_its_superproject_and_a_worktree_from_itself() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git/modules/sub")).unwrap();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/.git"), "gitdir: ../.git/modules/sub\n").unwrap();
        std::fs::write(root.join("sub/x.md"), "text").unwrap();
        let expected = format!("{}/sub/x.md", base(root));
        assert_eq!(
            names(&[root.to_path_buf()]),
            std::slice::from_ref(&expected)
        );
        assert_eq!(names(&[root.join("sub/x.md")]), [expected]);

        std::fs::create_dir_all(root.join("wt")).unwrap();
        std::fs::write(
            root.join("wt/.git"),
            "gitdir: /elsewhere/.git/worktrees/wt\n",
        )
        .unwrap();
        std::fs::write(root.join("wt/y.md"), "text").unwrap();
        assert_eq!(names(&[root.join("wt/y.md")]), ["wt/y.md"]);
    }

    /// Inside a repository, what git ignores is not read: by a `.gitignore`
    /// in the tree, in a directory under it, or above the directory named,
    /// and by `.git/info/exclude`.
    #[test]
    fn what_git_ignores_is_not_read() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join(".git/info")).unwrap();
        std::fs::write(
            root.join(".gitignore"),
            "build/\n*.draft.md\ndocs/private/\n",
        )
        .unwrap();
        std::fs::write(root.join(".git/info/exclude"), "scratch.md\n").unwrap();
        for name in [
            "keep.md",
            "a.draft.md",
            "scratch.md",
            "build/out.md",
            "docs/guide.md",
            "docs/private/secret.md",
            "docs/notes/local.md",
            "docs/notes/kept.md",
        ] {
            std::fs::create_dir_all(root.join(name).parent().unwrap()).unwrap();
            std::fs::write(root.join(name), "text").unwrap();
        }
        std::fs::write(root.join("docs/notes/.gitignore"), "local.md\n").unwrap();

        let base = base(root);
        assert_eq!(
            names(&[root.to_path_buf()]),
            [
                format!("{base}/docs/guide.md"),
                format!("{base}/docs/notes/kept.md"),
                format!("{base}/keep.md"),
            ]
        );
        // Named below the `.gitignore` that excludes part of it.
        assert_eq!(
            names(&[root.join("docs")]),
            [
                format!("{base}/docs/guide.md"),
                format!("{base}/docs/notes/kept.md"),
            ]
        );
    }

    /// Outside a repository a `.gitignore` is a file like any other, as it is
    /// to git.
    #[test]
    fn outside_a_repository_nothing_is_ignored() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path().join("notes");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join(".gitignore"), "*.md\n").unwrap();
        std::fs::write(root.join("a.md"), "text").unwrap();
        assert_eq!(names(&[root]), ["notes/a.md"]);
    }

    #[test]
    fn outside_a_repository_a_tree_is_named_from_itself() {
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("notes")).unwrap();
        std::fs::write(root.join("notes/a.md"), "text").unwrap();
        assert_eq!(names(&[root.join("notes")]), ["notes/a.md"]);
    }

    /// Two trees whose files would share names are refused rather than
    /// recorded under one topic each.
    #[test]
    fn two_files_under_one_name_are_refused() {
        let (left, right) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for tree in [&left, &right] {
            std::fs::create_dir_all(tree.path().join("notes")).unwrap();
            std::fs::write(tree.path().join("notes/a.md"), "text").unwrap();
        }
        // Different roots with different names: two names.
        assert_eq!(
            names(&[left.path().to_path_buf(), right.path().to_path_buf()]).len(),
            2
        );
        // The same root name twice: one name, refused.
        let refused = files(&[left.path().join("notes"), right.path().join("notes")])
            .expect_err("two files under one name");
        assert!(refused.to_string().contains("notes/a.md"), "{refused}");
    }

    #[test]
    #[cfg(unix)]
    fn an_unreadable_directory_is_skipped_rather_than_ending_the_ingest() {
        use std::os::unix::fs::PermissionsExt;
        let tree = tempfile::tempdir().unwrap();
        let root = tree.path();
        std::fs::create_dir_all(root.join("locked")).unwrap();
        std::fs::write(root.join("a.md"), "text").unwrap();
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o000))
            .unwrap();
        // Root reads every directory whatever its mode, so there is nothing
        // to observe there.
        let readable = std::fs::read_dir(root.join("locked")).is_ok();
        let (found, skipped) = files(&[root.to_path_buf()]).expect("files");
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert_eq!(found.len(), 1);
        if !readable {
            assert_eq!(skipped.len(), 1, "{skipped:?}");
        }
    }
}
