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
use pamin_engine::{Owed, Retired};
use pamin_index::VectorIndex;
use pamin_core::EdgeKind;
use pamin_index::sections::{
    Placed, SEPARATOR, Section, Unnamed, named, number, outline, unnamed,
};
use pamin_index::spans::TokenCount;
use pamin_store::repository::FileTopic;

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
    /// The directory it is named from, as a path.
    root: String,
}

/// A directory an ingest walked.
#[derive(Debug)]
struct Tree {
    /// Named as a file under it is named, up to it.
    name: String,
    /// The directory its files are named from, as a path.
    root: String,
    /// Whether the walk read all of it. One that met an error cannot say a
    /// file is gone from it rather than unreachable, and retires nothing.
    whole: bool,
}

pub async fn execute(
    session: &Session,
    project: &str,
    vector_index: VectorIndex,
    args: Args,
) -> Result<Ingested> {
    let Found {
        files,
        mut skipped,
        trees,
    } = files(&args.paths)?;

    let mut texts = Vec::new();
    // Found and not read: their sections are not missing, only unread.
    let mut unread = Vec::new();
    for file in files {
        match std::fs::read(&file.path) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => texts.push((file.name, file.root, text)),
                Err(_) => {
                    skipped.push(format!("{}: not UTF-8", file.path.display()));
                    unread.push((file.name, file.root));
                }
            },
            Err(error) => {
                skipped.push(format!("{}: {error}", file.path.display()));
                unread.push((file.name, file.root));
            }
        }
    }

    // Counted by BGE-M3's own tokenizer, which is the reranker's vocabulary
    // too: both are XLM-R's. Loaded off the runtime, as every model load is.
    // Not loaded at all for an ingest that found nothing to read.
    let models = session.workspace().models_dir();
    let cut: Vec<(String, String, Vec<Unnamed>)> = tokio::task::spawn_blocking(move || {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let counter = pamin_index::SpanTokens::load(&models)?;
        let counter: &dyn TokenCount = &counter;
        Ok::<_, pamin_index::IndexError>(
            texts
                .into_iter()
                .map(|(name, root, text)| {
                    let sections = unnamed(&name, &text, counter);
                    (name, root, sections)
                })
                .collect::<Vec<_>>(),
        )
    })
    .await
    .context("cutting files into sections")??;

    let engine = session.engine(project, vector_index).await?;
    let recorded =
        pamin_store::repository::file_topic_names(engine.database.pool(), engine.project).await?;
    let (cut, outline) = numbered(&engine, cut, &recorded).await?;

    let read: Vec<(String, String)> = cut
        .iter()
        .map(|(name, root, _)| (name.clone(), root.clone()))
        .collect();
    let cut: Vec<(&Section, &str)> = cut
        .iter()
        .flat_map(|(_, root, sections)| {
            sections.iter().map(move |section| (section, root.as_str()))
        })
        .collect();
    let gone = gone(
        &Looked {
            read: &read,
            unread: &unread,
            trees: &trees,
            exists: &|path: &Path| path.exists(),
        },
        &cut.iter().map(|(section, _)| *section).collect::<Vec<_>>(),
        recorded,
    );

    // Every section is a topic of its own -- its name holds its file's path,
    // and one file's names are numbered apart -- so they all run at once, in
    // the bound `import` derives from the pool for the same reason.
    let at_once = usize::try_from(engine.database.pool().options().get_max_connections())
        .unwrap_or(LEAST_AT_ONCE)
        .max(LEAST_AT_ONCE);

    let (mut written, mut unchanged, mut held) = (0, 0, 0);
    // The sections found unchanged, by the root they were read from: each
    // records that root, in one statement a root, after the writes.
    let mut kept_at: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
    let mut lagging = false;
    let mut since_check = 0;
    for chunk in cut.chunks(at_once) {
        let recorded = futures::future::try_join_all(chunk.iter().map(|(section, root)| {
            engine.remember_section(
                &section.name,
                &section.content,
                pamin_core::Validity::ALWAYS,
                root,
            )
        }))
        .await?;

        for (outcome, (section, root)) in recorded.into_iter().zip(chunk) {
            match outcome {
                None => {
                    unchanged += 1;
                    kept_at
                        .entry(*root)
                        .or_default()
                        .push(pamin_engine::file_locator(&section.name));
                }
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

    for (root, locators) in &kept_at {
        pamin_store::repository::set_roots_at(
            engine.database.pool(),
            engine.project,
            locators,
            root,
        )
        .await?;
    }

    // After the sections are recorded, so each has the state its edges cite,
    // and before retiring, whose sections the outline no longer names.
    let sections: Vec<String> = cut.iter().map(|(section, _)| section.name.clone()).collect();
    engine.restate_outline(&sections, &outline).await?;

    // After the sections are recorded, so a section that moved within a file
    // is written under its new name before its old one is retired, and before
    // the drain, which removes what retiring left for the projection.
    let mut removed = Vec::new();
    let mut kept = Vec::new();
    for gone in gone {
        if args.keep_missing || !gone.retirable {
            kept.push(gone.name);
            continue;
        }
        match engine.retire_section(&gone.name, gone.current).await? {
            Retired::Retired => removed.push(gone.name),
            Retired::Kept | Retired::Changed => kept.push(gone.name),
            Retired::Absent => {}
        }
    }
    kept.sort();

    engine.drain_cascade(Owed::WhatAMemoryNeeds).await?;
    engine.flush_what_is_applied().await?;
    let owed = pamin_store::jobs::pending(engine.database.pool(), engine.project).await?;

    Ok(Ingested {
        files: read.len(),
        sections: cut.len(),
        written,
        unchanged,
        held,
        removed,
        missing: kept,
        skipped,
        cascade: if owed == 0 { "applied" } else { "queued" }.to_string(),
        cascade_lagging: lagging,
    })
}

/// Each file's sections named, numbered against what its sections said when
/// it was last ingested wherever a number is at stake: in a file where two
/// sections share headings, or where an earlier ingest numbered one. Every
/// other file's sections are named as they are cut, with no read.
///
/// And each file's outline, by those names: the edges
/// [`Engine::restate_outline`](pamin_engine::Engine::restate_outline) states.
#[allow(clippy::type_complexity)]
async fn numbered(
    engine: &pamin_engine::Engine,
    cut: Vec<(String, String, Vec<Unnamed>)>,
    recorded: &[FileTopic],
) -> Result<(
    Vec<(String, String, Vec<Section>)>,
    Vec<(String, String, EdgeKind)>,
)> {
    // Per file, the earlier sections a number could be kept from.
    let wanted: Vec<Vec<String>> = cut
        .iter()
        .map(|(file, _, sections)| {
            let bases: BTreeSet<&str> = sections.iter().map(|s| s.base.as_str()).collect();
            let earlier: Vec<String> = recorded
                .iter()
                .filter(|topic| belongs(&topic.name, file))
                .filter(|topic| bases.iter().any(|base| number(&topic.name, base).is_some()))
                .map(|topic| topic.name.clone())
                .collect();
            let repeated = bases.len() < sections.len();
            let renumbered = earlier
                .iter()
                .any(|name| bases.iter().any(|base| number(name, base).is_some_and(|n| n > 1)));
            if repeated || renumbered {
                earlier
            } else {
                Vec::new()
            }
        })
        .collect();
    let names: Vec<String> = wanted.iter().flatten().cloned().collect();
    let contents: std::collections::HashMap<String, String> =
        pamin_store::repository::current_named(engine.database.pool(), engine.project, &names)
            .await?
            .into_iter()
            .map(|named| (named.name, named.content))
            .collect();
    let mut edges = Vec::new();
    let files = cut
        .into_iter()
        .zip(wanted)
        .map(|((file, root, sections), wanted)| {
            let previous: Vec<Section> = wanted
                .into_iter()
                .filter_map(|name| {
                    let content = contents.get(&name)?.clone();
                    Some(Section { name, content })
                })
                .collect();
            let bases: Vec<String> = sections.iter().map(|s| s.base.clone()).collect();
            let sections = named(sections, &previous);
            let bases: Vec<&str> = bases.iter().map(String::as_str).collect();
            for (from, to, placed) in outline(&bases) {
                let kind = match placed {
                    Placed::Under => EdgeKind::PartOf,
                    Placed::Before => EdgeKind::RelatedTo,
                };
                edges.push((sections[from].name.clone(), sections[to].name.clone(), kind));
            }
            (file, root, sections)
        })
        .collect();
    Ok((files, edges))
}

/// What a command's paths came to.
#[derive(Debug)]
struct Found {
    /// Every file, once each, in name order.
    files: Vec<File>,
    /// What was skipped, with why.
    skipped: Vec<String>,
    /// Every directory walked.
    trees: Vec<Tree>,
}

/// Every file `paths` names, once each, in name order, and the ones skipped
/// with why.
///
/// A path that cannot be read at all is a refusal, because it is what the
/// command was asked to do; a file or directory found under one and unreadable
/// is skipped and reported, as one that is not UTF-8 is. Two files that would
/// go by one name are a refusal too: they would be one topic, which each
/// ingest would hand from one to the other.
fn files(paths: &[PathBuf]) -> Result<Found> {
    let mut files = BTreeSet::new();
    let mut skipped = Vec::new();
    let mut trees = Vec::new();
    for path in paths {
        // Resolved here rather than by the client, which made it absolute but
        // left `..` and links in it: two spellings of one file must give its
        // sections one name.
        let path =
            std::fs::canonicalize(path).with_context(|| format!("reading {}", path.display()))?;
        let root = root(&path);
        if path.is_dir() {
            let whole = walk(&path, &root, &mut files, &mut skipped);
            match file(&path, &root) {
                Some(tree) => trees.push(Tree {
                    name: tree.name,
                    root: tree.root,
                    whole,
                }),
                None => skipped.push(format!("{}: name is not UTF-8", path.display())),
            }
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

    Ok(Found {
        files: files.into_iter().collect(),
        skipped,
        trees,
    })
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
        root: root.to_str()?.to_string(),
    })
}

/// The files under `directory` with one of [`EXTENSIONS`] -- and whether the
/// walk met no error, so that a file it did not find is one that is not
/// there -- skipping hidden
/// entries -- `.git` among them -- symbolic links, which could lead out of the
/// tree or round in a circle, and whatever git ignores there. A repository's
/// `node_modules/` or `target/` is full of Markdown nobody wrote for it.
fn walk(
    directory: &Path,
    root: &Path,
    files: &mut BTreeSet<File>,
    skipped: &mut Vec<String>,
) -> bool {
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
    let mut whole = true;
    for entry in walk {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                skipped.push(error.to_string());
                whole = false;
                continue;
            }
        };
        let path = entry.path();
        let is_text = entry.file_type().is_some_and(|kind| kind.is_file()) && walked_kind(path);
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
    whole
}

/// Where an ingest looked.
struct Looked<'a> {
    /// The files it read, by name and root.
    read: &'a [(String, String)],
    /// The files it found and could not read.
    unread: &'a [(String, String)],
    /// The directories it walked.
    trees: &'a [Tree],
    /// Whether a path is there at all: a file a walk skipped -- hidden, or
    /// ignored by git, and named on its own some earlier time -- is not gone.
    exists: &'a dyn Fn(&Path) -> bool,
}

/// A section an earlier ingest recorded that this one did not cut again.
#[derive(Debug, PartialEq, Eq)]
struct Gone {
    name: String,
    /// What its topic said when this ingest looked.
    current: pamin_core::TopicStateId,
    /// Whether this ingest can say it is gone rather than out of its reach: it
    /// was cut from a file the ingest read, or from a file of a kind a walk
    /// reads under a directory it walked whole.
    retirable: bool,
}

/// The sections in `recorded` -- every section an ingest has put on the
/// retrieval surface -- that are not among `cut` and that this ingest would
/// have cut again: from a file it read, or from a file under a directory it
/// walked that is no longer there -- deleted, moved, or ignored now.
///
/// Each from the root it was recorded from, so that another tree whose files
/// are named alike -- another clone, another `notes` -- is not taken for this
/// one. A file it found and could not read is not where a section went
/// missing, and a file of another kind named on its own an earlier time is not
/// one a walk would have found; neither is listed. Under a directory whose
/// walk met an error, a file not found may only have been out of reach, so its
/// sections are listed but not retirable.
fn gone(looked: &Looked, cut: &[&Section], recorded: Vec<FileTopic>) -> Vec<Gone> {
    let present: BTreeSet<&str> = cut.iter().map(|section| section.name.as_str()).collect();
    let mut gone = Vec::new();
    for topic in recorded {
        if present.contains(topic.name.as_str()) {
            continue;
        }
        let Some(root) = topic.root.as_deref() else {
            continue;
        };
        let name = topic.name.as_str();
        let retirable = if looked
            .read
            .iter()
            .any(|(file, at)| at == root && belongs(name, file))
        {
            Some(true)
        } else {
            let file = file_of(name);
            let unread = looked
                .unread
                .iter()
                .any(|(unread, at)| at == root && unread == file);
            if unread || !walked_kind(Path::new(file)) {
                None
            } else {
                // Under every walked tree that holds it, and retirable if any
                // of them was walked whole and the file is not on disk after
                // all, where the walk skipped it.
                looked
                    .trees
                    .iter()
                    .filter(|tree| {
                        tree.root == root
                            && file
                                .strip_prefix(tree.name.as_str())
                                .is_some_and(|rest| rest.starts_with('/'))
                    })
                    .map(|tree| tree.whole)
                    .reduce(|left, right| left || right)
                    .map(|whole| whole && !(looked.exists)(&on_disk(root, file)))
            }
        };
        if let Some(retirable) = retirable {
            gone.push(Gone {
                name: topic.name,
                current: topic.current,
                retirable,
            });
        }
    }
    gone.sort_by(|left, right| left.name.cmp(&right.name));
    gone
}

/// Where the file named `file`, named from `root`, would be: its name is the
/// root's own name and then its path from the root.
fn on_disk(root: &str, file: &str) -> PathBuf {
    let root = Path::new(root);
    let relative = match root.file_name() {
        Some(_) => file.split_once('/').map_or("", |(_, rest)| rest),
        None => file,
    };
    root.join(relative)
}

/// The name of the file the section named `name` was cut out of: its name up
/// to its headings, without the number it was given.
fn file_of(name: &str) -> &str {
    let mut file = name.split(SEPARATOR).next().unwrap_or(name);
    while let Some((before, _)) = file
        .rsplit_once(" #")
        .filter(|(_, number)| !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()))
    {
        file = before;
    }
    file
}

/// Whether a walk reads a file at `path`: one with an extension of
/// [`EXTENSIONS`].
fn walked_kind(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// Whether the section named `name` was cut out of the file named `file`: its
/// name is the file's, then any numbers ` #n` it was given, then nothing or
/// its headings.
///
/// Read exactly, because `a.md #1.md` begins with `a.md #1` and is another
/// file.
fn belongs(name: &str, file: &str) -> bool {
    let Some(mut rest) = name.strip_prefix(file) else {
        return false;
    };
    loop {
        if rest.is_empty() || rest.starts_with(SEPARATOR) {
            return true;
        }
        match rest.strip_prefix(" #").and_then(digits) {
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
        let Found { files, skipped, .. } = files(paths).expect("files");
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
            "docs/a.md #2 #3",
            "docs/a.md #2 › Usage",
        ] {
            assert!(belongs(name, "docs/a.md"), "{name}");
        }
        for name in [
            "docs/a.md.bak › Usage",
            "docs/ab.md",
            "docs",
            "docs/a.md (2)",
            "docs/a.md #1.md",
            "docs/a.md #",
        ] {
            assert!(!belongs(name, "docs/a.md"), "{name}");
        }
    }

    const ROOT: &str = "/home/a/repo";

    fn recorded(name: &str, root: &str) -> FileTopic {
        FileTopic {
            name: name.to_string(),
            current: pamin_core::TopicStateId::new(),
            root: Some(root.to_string()),
        }
    }

    fn read(name: &str) -> (String, String) {
        (name.to_string(), ROOT.to_string())
    }

    fn tree(name: &str, whole: bool) -> Tree {
        Tree {
            name: name.to_string(),
            root: ROOT.to_string(),
            whole,
        }
    }

    fn listed(gone: &[Gone]) -> Vec<(&str, bool)> {
        gone.iter()
            .map(|gone| (gone.name.as_str(), gone.retirable))
            .collect()
    }

    /// Only the files this ingest read are compared: a section of a file left
    /// out of the command is not gone, it is just not being looked at.
    #[test]
    fn gone_is_what_an_earlier_ingest_left_under_the_files_read() {
        let kept = section("repo/a.md › Kept");
        let gone = gone(
            &Looked {
                read: &[read("repo/a.md")],
                unread: &[],
                trees: &[],
                exists: &|_: &Path| false,
            },
            &[&kept],
            vec![
                recorded("repo/a.md › Kept", ROOT),
                recorded("repo/a.md › Gone", ROOT),
                recorded("repo/b.md › Elsewhere", ROOT),
            ],
        );
        assert_eq!(listed(&gone), [("repo/a.md › Gone", true)]);
    }

    /// Under a directory it walked, an ingest also finds the sections of a
    /// file that is no longer there -- but not of one it found and could not
    /// read, nor of a file of a kind a walk does not read, which was named on
    /// its own before.
    #[test]
    fn gone_includes_files_missing_from_a_walked_tree() {
        let kept = section("repo/docs/a.md › Kept");
        let gone = gone(
            &Looked {
                read: &[read("repo/docs/a.md")],
                unread: &[read("repo/docs/unreadable.md")],
                trees: &[tree("repo/docs", true)],
                exists: &|_: &Path| false,
            },
            &[&kept],
            [
                "repo/docs/a.md › Kept",
                "repo/docs/deleted.md › Usage",
                "repo/docs/deleted.md #2",
                "repo/docs/deep/moved.txt",
                "repo/docs/unreadable.md › Kept",
                "repo/docs/script.py › Comment",
                "repo/docsier/other.md",
                "repo/README.md › Top",
            ]
            .iter()
            .map(|name| recorded(name, ROOT))
            .collect(),
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/docs/deep/moved.txt", true),
                ("repo/docs/deleted.md #2", true),
                ("repo/docs/deleted.md › Usage", true)
            ]
        );
    }

    /// Another tree whose files are named alike -- another clone of `repo`
    /// somewhere else -- is not this one: its sections are not gone.
    #[test]
    fn another_tree_of_the_same_name_is_not_looked_at() {
        let gone = gone(
            &Looked {
                read: &[read("repo/a.md")],
                unread: &[],
                trees: &[tree("repo", true)],
                exists: &|_: &Path| false,
            },
            &[],
            vec![
                recorded("repo/a.md › Here", ROOT),
                recorded("repo/a.md › There", "/home/b/repo"),
                recorded("repo/b.md › There", "/home/b/repo"),
            ],
        );
        assert_eq!(listed(&gone), [("repo/a.md › Here", true)]);
    }

    /// A walk that met an error cannot tell a file that is gone from one it
    /// could not reach: its sections are listed and not retirable, unless a
    /// file the ingest read says otherwise.
    #[test]
    fn a_walk_that_met_an_error_retires_nothing_it_did_not_read() {
        let gone = gone(
            &Looked {
                read: &[read("repo/docs/a.md")],
                unread: &[],
                trees: &[tree("repo/docs", false)],
                exists: &|_: &Path| false,
            },
            &[],
            vec![
                recorded("repo/docs/a.md › Old heading", ROOT),
                recorded("repo/docs/locked/b.md › Usage", ROOT),
            ],
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/docs/a.md › Old heading", true),
                ("repo/docs/locked/b.md › Usage", false)
            ]
        );
    }

    /// A file the walk skipped but that is still on disk -- hidden, or ignored
    /// by git, and named on its own before -- is listed, not retired.
    #[test]
    fn a_file_the_walk_skipped_is_not_gone() {
        let gone = gone(
            &Looked {
                read: &[],
                unread: &[],
                trees: &[tree("repo", true)],
                exists: &|path: &Path| path == Path::new("/home/a/repo/.github/CONTRIBUTING.md"),
            },
            &[],
            vec![
                recorded("repo/.github/CONTRIBUTING.md › Style", ROOT),
                recorded("repo/deleted.md › Usage", ROOT),
            ],
        );
        assert_eq!(
            listed(&gone),
            [
                ("repo/.github/CONTRIBUTING.md › Style", false),
                ("repo/deleted.md › Usage", true)
            ]
        );
        assert_eq!(
            on_disk("/home/a/repo", "repo/docs/a.md"),
            Path::new("/home/a/repo/docs/a.md")
        );
    }

    #[test]
    fn a_sections_file_is_its_name_before_its_headings_and_numbers() {
        assert_eq!(file_of("r/a.md"), "r/a.md");
        assert_eq!(file_of("r/a.md › H › I"), "r/a.md");
        assert_eq!(file_of("r/a.md #2"), "r/a.md");
        assert_eq!(file_of("r/a.md #2 › H"), "r/a.md");
        assert_eq!(file_of("r/a (old).md"), "r/a (old).md");
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
        let Found {
            files: found,
            skipped,
            trees,
        } = files(&[root.to_path_buf()]).expect("files");
        std::fs::set_permissions(root.join("locked"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        assert_eq!(found.len(), 1);
        if !readable {
            assert_eq!(skipped.len(), 1, "{skipped:?}");
            // And the tree is not taken to be whole, so nothing under the
            // directory that could not be read is retired as gone.
            assert!(!trees[0].whole);
        }
    }
}
